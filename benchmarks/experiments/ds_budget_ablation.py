"""Predeclared same-loop DS budget ablation; actual MLX measurements only.

Use --smoke for a short correctness control (not formal performance evidence).
The complete protocol freezes native fixed-k on development, then reports every
held-out output including divergences. Prompt-cluster bootstrap is descriptive
for this fixed small corpus, not a population-wide confidence guarantee.
"""

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path

from deepinfer_mac.engine import InferenceEngine
from deepinfer_mac.policy import AdaptivePolicy, DSAdaptivePolicy, FixedPolicy
from deepinfer_mac.provenance import source_manifest
from deepinfer_mac.state import GenerationRequest

DEVELOPMENT = [
    ("dev-zh", "用两句话解释统一内存，并说明它对本地语言模型的影响。"),
    ("dev-code", "Write Python dedupe(items) preserving insertion order. Include a brief example."),
    ("dev-json", 'Return only JSON with keys "city" and "country" for the capital of France.'),
]
HELD_OUT = [
    ("test-zh", "请解释为什么前缀缓存能降低首字延迟，但不一定提高逐字生成速度。"),
    ("test-code", "Write Python is_palindrome(text) ignoring spaces and case. Show two asserts."),
    ("test-json", 'Return a JSON object with sorted numbers [9,2,5,2] and their unique count.'),
    ("test-facts", "Given apples=7, bought=5, gave_away=3, state the final count and one equation."),
    ("test-branch", "System design context: exact cache snapshots must protect branch isolation. " * 24
     + "Explain one correctness risk and one mitigation, briefly."),
    ("test-long-code", "A local Python utility must preserve user records and report bad input. " * 24
     + "Write parse_positive_int(value) returning a positive int or raising ValueError."),
]
SAME_LOOP = ["fixed0", "fixed1", "fixed2", "fixed4", "legacy", "ds"]


def cluster_interval(groups, variant, reference):
    ratios = []
    for group in groups.values():
        pairs = [(s[reference]["total_ms"] / s[variant]["total_ms"]) for s in group
                 if reference in s and variant in s and s[reference]["ids"] == s[variant]["ids"]]
        if pairs:
            ratios.append(statistics.mean(math.log(v) for v in pairs))
    if not ratios:
        return {"eligible_prompts": 0, "ratio": None, "interval": None}
    rng = random.Random(20261021)
    draws = sorted(math.exp(statistics.mean(rng.choices(ratios, k=len(ratios)))) for _ in range(2000))
    return {"eligible_prompts": len(ratios), "ratio": math.exp(statistics.mean(ratios)),
            "interval": [draws[49], draws[1949]] if len(ratios) >= 2 else None, "cluster": "prompt; paired exact-token matches only"}


def main():
    import mlx.core as mx
    from mlx_lm.generate import generate_step, speculative_generate_step
    from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache

    from deepinfer_mac.hybrid_speculative import hybrid_generate_step
    from deepinfer_mac.speculative import adaptive_generate_step

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="qwen3-4b")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/reports/phase2-ds-budget-qwen3-4b.json"))
    args = parser.parse_args()
    report = {"source": source_manifest(), "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "profile": args.profile, "protocol": {
        "development": DEVELOPMENT, "held_out": HELD_OUT, "repeats": args.repeats,
        "max_tokens": args.max_tokens, "smoke": args.smoke, "cache": "all cold KV, warm model",
        "primary_metric": "task total_ms including prefill; exact token equality required for speed ratios",
        "fixed_tuning": "minimum actual native development median total_ms among native0/1/2/4; no held-out tuning",
        "generation": "greedy, thinking=false, prefill512; all same-loop variants deferred sync",
        "quality_scope": "token equivalence control; not a model capability benchmark or two-engine claim",
        "native_prefill": "target-only for greedy; target+draft for positive fixed; same-loop controls both",
        "performance_validity": "smoke runs during possible download I/O are correctness evidence only",
    }, "samples": []}
    engine = InferenceEngine(cache_budget_bytes=0)
    engine._load(args.profile, True)
    is_hybrid = not can_trim_prompt_cache(make_prompt_cache(engine.model))
    native_variants = ["native_greedy"] if is_hybrid else ["native_greedy", "native_fixed"]
    report["native_speculation_compatibility"] = {
        "supported": not is_hybrid,
        "evidence": "MLX-LM 0.32.2 generate.py speculative_generate_step explicitly requires trimmable target cache",
        "unsupported_result": "not scored; native greedy remains a real target-only control",
    }
    report["protocol"]["fixed_tuning"] = (
        "same-loop fixed0/1/2/4, developer exact-to-greedy candidates only; own tuned control, not a competitor"
        if is_hybrid else "actual native0/1/2/4, developer exact-to-greedy candidates only"
    )
    report["protocol"]["recovery"] = "exact hybrid snapshot+consumed-prefix replay" if is_hybrid else "native trim"
    policies = {"legacy": AdaptivePolicy(), "ds": DSAdaptivePolicy()}
    tuned = 2

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    def run(prompt, variant, max_tokens):
        request = GenerationRequest(profile=args.profile, prompt=prompt, max_tokens=max_tokens, use_cache=False)
        prompt_ids = engine._tokens(request)
        target, draft = make_prompt_cache(engine.model), make_prompt_cache(engine.draft_model)
        mx.synchronize()
        mx.reset_peak_memory()
        begin = time.perf_counter()
        for offset in range(0, len(prompt_ids) - 1, 512):
            chunk = mx.array(prompt_ids[offset:min(offset + 512, len(prompt_ids) - 1)], mx.uint32)[None]
            engine.model(chunk, cache=target)
            # Same-loop controls prefill both models to isolate the scheduler.
            # Native greedy retains its real target-only prefill advantage.
            native_budget = int(variant[6:]) if variant in ("native0", "native1", "native2", "native4") else tuned
            needs_draft = variant != "native_greedy" and not (variant.startswith("native") and native_budget == 0)
            if needs_draft:
                engine.draft_model(chunk, cache=draft)
            mx.eval([c.state for c in target + (draft if needs_draft else [])])
        rounds, ids = [], []
        tail = mx.array(prompt_ids[-1:], mx.uint32)
        native_budget = int(variant[6:]) if variant in ("native0", "native1", "native2", "native4") else tuned
        if variant == "native_greedy" or (variant.startswith("native") and native_budget == 0):
            generator = generate_step(tail, engine.model, max_tokens=max_tokens, prompt_cache=target)
        elif variant.startswith("native"):
            generator = speculative_generate_step(tail, engine.model, engine.draft_model,
                                                 num_draft_tokens=native_budget, max_tokens=max_tokens,
                                                 prompt_cache=target + draft)
        else:
            budget = tuned if variant == "tuned_fixed" else int(variant[5:]) if variant.startswith("fixed") else 0
            policy = policies[variant] if variant in policies else FixedPolicy(budget)
            adapter = hybrid_generate_step if is_hybrid else adaptive_generate_step
            kwargs = {} if is_hybrid else {"defer_draft_sync": True}
            generator = adapter(prompt_ids[-1:], engine.model, engine.draft_model, target,
                                draft, policy, max_tokens, rounds, diagnostics=False, **kwargs)
        first = None
        try:
            for output in generator:
                ids.append(int(output[0]))
                if first is None:
                    first = time.perf_counter()
                if ids[-1] in engine.tokenizer.eos_token_ids:
                    break
        finally:
            generator.close()
            mx.synchronize()
        end = time.perf_counter()
        return {"total_ms": (end - begin) * 1000, "ttft_ms": (first - begin) * 1000 if first else None,
                "ids": ids, "text": engine.tokenizer.decode(ids), "rounds": rounds,
                "policy": policies[variant].summary() if variant in policies else None,
                "peak_memory_bytes": mx.get_peak_memory(), "prompt_tokens": len(prompt_ids)}

    try:
        for stage, corpus in [("warmup", DEVELOPMENT[:1]), ("development", DEVELOPMENT), ("held_out", HELD_OUT)]:
            if args.smoke:
                corpus = corpus[:1]
            repeats = 1 if stage != "held_out" or args.smoke else args.repeats
            for repeat in range(repeats):
                for pi, (name, prompt) in enumerate(corpus):
                    variants = SAME_LOOP + native_variants
                    if is_hybrid:
                        variants += ["tuned_fixed"]
                    if not is_hybrid and stage in ("warmup", "development"):
                        variants += ["native0", "native1", "native2", "native4"]
                    shift = (pi + repeat) % len(variants)
                    order = variants[shift:] + variants[:shift]
                    sample = {"stage": stage, "repeat": repeat, "prompt_id": name, "variants": {}}
                    report["samples"].append(sample)
                    for variant in order:
                        sample["variants"][variant] = run(prompt, variant, min(args.max_tokens, 24)
                                                          if args.smoke else args.max_tokens)
                        save()
                    reference = sample["variants"]["native_greedy"]["ids"]
                    sample["first_divergence"] = {
                        v: next((i for i, (a, b) in enumerate(zip(data["ids"], reference)) if a != b),
                                min(len(data["ids"]), len(reference)) if len(data["ids"]) != len(reference) else None)
                        for v, data in sample["variants"].items()
                    }
                    sample["equal_to_greedy"] = {v: data["ids"] == reference
                                                  for v, data in sample["variants"].items()}
                    save()
                    print(stage, name, json.dumps(sample["equal_to_greedy"]), flush=True)
            if stage == "development":
                dev = [s for s in report["samples"] if s["stage"] == "development"]
                family = "fixed" if is_hybrid else "native"
                equal_candidates = [k for k in [0, 1, 2, 4]
                                    if all(s["equal_to_greedy"][f"{family}{k}"] for s in dev)]
                tuned = min(equal_candidates or [0], key=lambda k: statistics.median(
                    s["variants"][f"{family}{k}"]["total_ms"] for s in dev))
                report["locked_same_loop_fixed_budget" if is_hybrid else "locked_native_fixed_budget"] = tuned
                report["fixed_tuning_exact_candidates"] = equal_candidates
                save()
        heldout = [s for s in report["samples"] if s["stage"] == "held_out"]
        groups = {}
        for sample in heldout:
            groups.setdefault(sample["prompt_id"], []).append(sample["variants"])
        report["summary"] = {"task_groups": len(heldout), "all_outputs_retained": True,
                             "equivalent_speedup_claim_supported": not args.smoke and all(
                                 all(s["equal_to_greedy"].values()) for s in heldout),
                             "competitive_advantage_claim_supported": False,
                             "competitor_scope": "one native MLX implementation only; second engine pending",
                             "native_over_ds": cluster_interval(groups, "ds", "native_greedy"),
                             "tuned_native_over_ds": None if is_hybrid else cluster_interval(groups, "ds", "native_fixed"),
                             "tuned_same_loop_over_ds": cluster_interval(groups, "ds", "tuned_fixed") if is_hybrid else None,
                             "legacy_over_ds": cluster_interval(groups, "ds", "legacy"),
                             "equal_counts": {v: sum(s["equal_to_greedy"][v] for s in heldout)
                                              for v in SAME_LOOP + native_variants + (["tuned_fixed"] if is_hybrid else [])}}
        report["source_end"] = source_manifest()
        report["source_unchanged"] = report["source"]["source_tree_sha256"] == report["source_end"]["source_tree_sha256"]
        if not report["source_unchanged"]:
            report["summary"]["equivalent_speedup_claim_supported"] = False
            report["summary"]["source_changed_warning"] = "runtime source/locks changed during run; formal validity not established"
        save()
        print(json.dumps(report["summary"], indent=2))
    finally:
        engine.close()


if __name__ == "__main__":
    main()
