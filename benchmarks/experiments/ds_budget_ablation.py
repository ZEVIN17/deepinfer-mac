"""Predeclared same-loop DS budget ablation; actual MLX measurements only.

Use --smoke for a short correctness control (not formal performance evidence).
The complete protocol freezes native fixed-k on development, then reports every
held-out output including divergences. Prompt-cluster bootstrap is descriptive
for this fixed small corpus, not a population-wide confidence guarantee.
"""

import argparse
import hashlib
import io
import json
import math
import random
import statistics
import tarfile
import time
from pathlib import Path

from deepinfer_mac.ds.recovery_cost import LaggedMeanRecoveryPolicy
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
SAME_LOOP = ["fixed0", "fixed1", "fixed2", "fixed4", "legacy", "ds", "ds_lagged_mean"]


def adaptive_policies(is_hybrid):
    policies = {"legacy": AdaptivePolicy(), "ds": DSAdaptivePolicy()}
    if is_hybrid:
        policies["ds_lagged_mean"] = LaggedMeanRecoveryPolicy()
    return policies


def same_loop_variants(is_hybrid):
    return [name for name in SAME_LOOP if is_hybrid or name != "ds_lagged_mean"]


def freeze_record(
    manifest_path, required_harness="benchmarks/experiments/ds_budget_ablation.py", *,
    executed_harness_path=None, archive_path=None,
):
    result = {"provided": manifest_path is not None, "matches": False, "formal_validity": False}
    if manifest_path is None:
        return result
    errors, mismatches = [], []
    try:
        manifest_bytes = manifest_path.read_bytes()
        data = json.loads(manifest_bytes)
        files = data.get("files")
        if not isinstance(files, dict) or not files:
            return {**result, "reason": "A nonempty per-file DS manifest is required"}
        if any(not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts for name in files):
            return {**result, "reason": "Manifest members must be repository-relative file names"}
        canonical_tree = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if data.get("tree_sha256") != canonical_tree:
            errors.append("manifest_tree_digest")
        binding = data.get("archive_binding", {})
        local_archive = archive_path
        if local_archive is None and binding.get("path"):
            local_archive = manifest_path.parent / binding["path"]
        if local_archive is None:
            return {**result, "reason": "A real source archive is required; loose file hashes cannot certify a freeze"}
        archive_bytes = Path(local_archive).read_bytes()
        archive_sha = hashlib.sha256(archive_bytes).hexdigest()
        if binding.get("sha256") is not None and binding["sha256"] != archive_sha:
            errors.append("archive_digest")
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)) or any(not member.isfile() for member in members):
                errors.append("archive_duplicate_or_nonfile_member")
            expected_names = set(files) | {"ds-source-manifest.json"}
            if set(names) != expected_names:
                errors.append("archive_member_set")
            internal_blob = archive.extractfile("ds-source-manifest.json").read()
            internal = json.loads(internal_blob)
            unbound = {key: value for key, value in data.items() if key != "archive_binding"}
            if internal != unbound:
                errors.append("internal_manifest_binding")
            internal_sha = hashlib.sha256(internal_blob).hexdigest()
            if binding.get("internal_manifest_sha256") is not None and binding["internal_manifest_sha256"] != internal_sha:
                errors.append("internal_manifest_digest")
            for name, expected in files.items():
                entry = archive.extractfile(name)
                if entry is None or hashlib.sha256(entry.read()).hexdigest() != expected:
                    errors.append(f"archive_content:{name}")
        for name, expected in files.items():
            path = Path(name)
            actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            if actual != expected:
                mismatches.append(name)
        runtime = source_manifest()["files"]
        mismatches.extend(f"imported_runtime:{name}" for name, actual in runtime.items() if files.get(name) != actual)
        missing = sorted(({*runtime, required_harness}) - files.keys())
        executing = Path(executed_harness_path) if executed_harness_path is not None else Path(__file__)
        executing_sha = hashlib.sha256(executing.read_bytes()).hexdigest()
        if files.get(required_harness) != executing_sha:
            errors.append("executing_harness_digest")
        valid = not errors and not mismatches and not missing
        return {**result, "manifest_name": manifest_path.name,
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "tree_sha256": canonical_tree,
                "archive_name": Path(local_archive).name, "archive_sha256": archive_sha,
                "internal_manifest_sha256": internal_sha, "executing_harness_sha256": executing_sha,
                "files": len(files), "errors": errors, "mismatches": mismatches,
                "missing_runtime_or_harness_files": missing, "matches": valid, "formal_validity": valid}
    except (OSError, ValueError, KeyError, AttributeError, TypeError, tarfile.TarError) as exc:
        # Local archive locations may contain a home path. Only publish error
        # categories, never OS exception strings or private binding paths.
        return {**result, "errors": [*errors, f"freeze_unreadable:{type(exc).__name__}"], "mismatches": mismatches}


def cluster_interval(groups, variant, reference):
    ratios = []
    for group in groups.values():
        pairs = [(s[reference]["total_ms"] / s[variant]["total_ms"]) for s in group
                 if reference in s and variant in s and not s[reference].get("error") and not s[variant].get("error")
                 and s[reference].get("total_ms") and s[variant].get("total_ms")
                 and s[reference]["ids"] == s[variant]["ids"]]
        if pairs:
            ratios.append(statistics.mean(math.log(v) for v in pairs))
    if not ratios:
        return {"eligible_prompts": 0, "ratio": None, "interval": None}
    rng = random.Random(20261021)
    draws = sorted(math.exp(statistics.mean(rng.choices(ratios, k=len(ratios)))) for _ in range(2000))
    return {"eligible_prompts": len(ratios), "ratio": math.exp(statistics.mean(ratios)),
            "interval": [draws[49], draws[1949]] if len(ratios) >= 2 else None, "cluster": "prompt; paired exact-token matches only"}


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="qwen3-4b")
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--max-tokens", type=positive_int, default=96)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--freeze-archive", type=Path, help="Explicit archive for legacy unbound manifests")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def frozen_execution_valid(report):
    """Require the same genuine archive, imported runtime and executing script throughout."""
    start, end = report.get("freeze", {}), report.get("freeze_end", {})
    bindings = ("manifest_sha256", "tree_sha256", "archive_sha256", "internal_manifest_sha256",
                "executing_harness_sha256")
    return bool(start.get("formal_validity") is True and end.get("formal_validity") is True
                and all(start.get(key) is not None and start[key] == end.get(key) for key in bindings)
                and report.get("source", {}).get("source_tree_sha256") is not None
                and report["source"]["source_tree_sha256"] == report.get("source_end", {}).get("source_tree_sha256")
                and report.get("harness_sha256") is not None
                and report["harness_sha256"] == report.get("harness_sha256_end")
                and report["harness_sha256"] == start.get("executing_harness_sha256"))


def heldout_coverage(samples, prompt_ids, repeats, variants):
    """Validate exact predeclared coverage and actual outputs, independent of cached equality flags."""
    expected = {(name, repeat) for name in prompt_ids for repeat in range(max(0, repeats))}
    rows = [s for s in samples if s.get("stage") == "held_out"]
    seen, errors = set(), []
    variant_set = set(variants)
    if not expected or repeats <= 0 or not variant_set or "native_greedy" not in variant_set:
        errors.append("empty_or_invalid_expected_coverage")
    for row in rows:
        key = (row.get("prompt_id"), row.get("repeat"))
        if key in seen:
            errors.append("duplicate_prompt_repeat")
        seen.add(key)
        actual = row.get("variants", {})
        if set(actual) != variant_set:
            errors.append("variant_coverage")
        reference = actual.get("native_greedy", {}).get("ids")
        for name in variant_set:
            result = actual.get(name, {})
            elapsed = result.get("total_ms")
            if (result.get("error") or not isinstance(result.get("ids"), list) or not result["ids"]
                    or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0):
                errors.append("failed_or_empty_variant")
            if result.get("ids") != reference:
                errors.append("output_mismatch")
    if seen != expected or len(rows) != len(expected):
        errors.append("prompt_repeat_coverage")
    return {"complete_and_exact": not errors, "expected_task_groups": len(expected),
            "observed_task_groups": len(rows), "expected_variants": sorted(variant_set),
            "errors": sorted(set(errors))}


def public_error(exc):
    return f"{type(exc).__name__}: {exc}".replace(str(Path.home()), "<user-home>")


def main():
    # Reject invalid/empty protocols before importing or initializing the GPU runtime.
    args = parse_args()
    import mlx.core as mx
    from mlx_lm.generate import generate_step, speculative_generate_step
    from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache

    from deepinfer_mac.hybrid_speculative import hybrid_generate_step
    from deepinfer_mac.speculative import adaptive_generate_step

    if args.output is None:
        args.output = Path(f"benchmarks/reports/phase2-ds-budget-{args.profile}.json")
    if args.output.exists():
        raise FileExistsError("Choose a fresh output path; historical benchmark evidence is immutable")
    report = {"source": source_manifest(), "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "profile": args.profile, "freeze": freeze_record(args.freeze_manifest, executed_harness_path=Path(__file__), archive_path=args.freeze_archive), "protocol": {
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
    policies = adaptive_policies(is_hybrid)
    loop_variants = same_loop_variants(is_hybrid)
    report["protocol"]["online_policy"] = {
        "hyperparameters_locked_before_held_out": True,
        "policy_state": "independent per variant; fixed algorithm may adapt from past completed runtime",
        "lagged_mean": {"min_samples": 5, "window": 9, "whole_feedback_cutoff": "N-2"} if is_hybrid else None,
    }
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
        error = None
        try:
            for output in generator:
                ids.append(int(output[0]))
                if first is None:
                    first = time.perf_counter()
                if ids[-1] in engine.tokenizer.eos_token_ids:
                    break
        except Exception as exc:  # noqa: BLE001 -- keep outputs generated before the actual failure
            error = public_error(exc)
        finally:
            try:
                generator.close()
                mx.synchronize()
            except Exception as exc:  # noqa: BLE001 -- retain close/recovery failures with partial output
                error = f"{error + '; ' if error else ''}{public_error(exc)}"
        end = time.perf_counter()
        return {"error": error, "total_ms": (end - begin) * 1000, "ttft_ms": (first - begin) * 1000 if first else None,
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
                    variants = loop_variants + native_variants
                    if is_hybrid:
                        variants += ["tuned_fixed"]
                    if not is_hybrid and stage in ("warmup", "development"):
                        variants += ["native0", "native1", "native2", "native4"]
                    shift = (pi + repeat) % len(variants)
                    order = variants[shift:] + variants[:shift]
                    sample = {"stage": stage, "repeat": repeat, "prompt_id": name, "variants": {}}
                    report["samples"].append(sample)
                    for variant in order:
                        try:
                            sample["variants"][variant] = run(prompt, variant, min(args.max_tokens, 24)
                                                              if args.smoke else args.max_tokens)
                        except Exception as exc:  # noqa: BLE001 -- retain failed variants, never a fabricated zero time
                            sample["variants"][variant] = {"error": public_error(exc),
                                                           "ids": [], "total_ms": None, "text": None}
                        save()
                    reference = sample["variants"]["native_greedy"]["ids"]
                    sample["first_divergence"] = {
                        v: next((i for i, (a, b) in enumerate(zip(data["ids"], reference)) if a != b),
                                min(len(data["ids"]), len(reference)) if len(data["ids"]) != len(reference) else None)
                        for v, data in sample["variants"].items()
                    }
                    sample["equal_to_greedy"] = {v: not data.get("error")
                                                  and not sample["variants"]["native_greedy"].get("error")
                                                  and data["ids"] == reference
                                                  for v, data in sample["variants"].items()}
                    save()
                    print(stage, name, json.dumps(sample["equal_to_greedy"]), flush=True)
            if stage == "development":
                dev = [s for s in report["samples"] if s["stage"] == "development"]
                family = "fixed" if is_hybrid else "native"
                equal_candidates = [k for k in [0, 1, 2, 4]
                                    if all(s["equal_to_greedy"][f"{family}{k}"] for s in dev)]
                tuned = min(equal_candidates, key=lambda k: statistics.median(
                    s["variants"][f"{family}{k}"]["total_ms"] for s in dev)) if equal_candidates else 0
                if not equal_candidates:
                    report["fixed_tuning_failure"] = "No exact valid development candidate; actual zero-budget fallback, not tuned optimum"
                report["locked_same_loop_fixed_budget" if is_hybrid else "locked_native_fixed_budget"] = tuned
                report["fixed_tuning_exact_candidates"] = equal_candidates
                save()
        heldout = [s for s in report["samples"] if s["stage"] == "held_out"]
        groups = {}
        for sample in heldout:
            groups.setdefault(sample["prompt_id"], []).append(sample["variants"])
        coverage = heldout_coverage(report["samples"], [name for name, _ in HELD_OUT], args.repeats,
                                    loop_variants + native_variants + (["tuned_fixed"] if is_hybrid else []))
        report["summary"] = {"heldout_coverage": coverage, "task_groups": len(heldout), "all_outputs_retained": True,
                             "failed_variants": sum(bool(v.get("error")) for s in report["samples"] for v in s["variants"].values()),
                             "equivalent_speedup_claim_supported": False,
                             "competitive_advantage_claim_supported": False,
                             "competitor_scope": "one native MLX implementation only; second engine pending",
                             "native_over_ds": cluster_interval(groups, "ds", "native_greedy"),
                             "tuned_native_over_ds": None if is_hybrid else cluster_interval(groups, "ds", "native_fixed"),
                             "tuned_same_loop_over_ds": cluster_interval(groups, "ds", "tuned_fixed") if is_hybrid else None,
                             "legacy_over_ds": cluster_interval(groups, "ds", "legacy"),
                             "native_over_ds_lagged_mean": cluster_interval(groups, "ds_lagged_mean", "native_greedy") if is_hybrid else None,
                             "original_ds_over_lagged_mean": cluster_interval(groups, "ds_lagged_mean", "ds") if is_hybrid else None,
                             "tuned_same_loop_over_lagged_mean": cluster_interval(groups, "ds_lagged_mean", "tuned_fixed") if is_hybrid else None,
                             "equal_counts": {v: sum(s["equal_to_greedy"][v] for s in heldout)
                                              for v in loop_variants + native_variants + (["tuned_fixed"] if is_hybrid else [])}}
        report["freeze_end"] = freeze_record(args.freeze_manifest, executed_harness_path=Path(__file__), archive_path=args.freeze_archive)
        report["harness_sha256_end"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report["source_end"] = source_manifest()
        report["source_unchanged"] = report["source"]["source_tree_sha256"] == report["source_end"]["source_tree_sha256"]
        report["summary"]["frozen_execution_valid"] = frozen_execution_valid(report)
        report["summary"]["equivalent_speedup_claim_supported"] = (
            not args.smoke and coverage["complete_and_exact"] and frozen_execution_valid(report))
        if not frozen_execution_valid(report):
            report["summary"]["formal_validity_warning"] = "Source, executing harness or archive binding did not all pass unchanged; no formal speed claim"
        save()
        print(json.dumps(report["summary"], indent=2))
    finally:
        engine.close()


if __name__ == "__main__":
    main()
