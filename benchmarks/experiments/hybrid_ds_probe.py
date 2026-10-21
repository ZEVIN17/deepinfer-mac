"""Actual 9B hybrid recovery/DS integration probe, not a formal speed benchmark.

Model registry must pin a downloaded, tokenizer-verified experimental draft.
Download and capability promotion belong to the model audit, not this script.
A second diagnostic pass isolates the first token divergence using identical
pre-verification cache state and history with single-token versus batch logits.
"""

import argparse
import dataclasses
import json
from pathlib import Path

from deepinfer_mac.engine import InferenceEngine
from deepinfer_mac.provenance import source_manifest
from deepinfer_mac.state import GenerationRequest


def main():
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    from deepinfer_mac.hybrid_speculative import hybrid_generate_step, restore_cache
    from deepinfer_mac.policy import FixedPolicy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="qwen3.5-9b-8bit")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--output", type=Path,
                        default=Path("benchmarks/reports/phase2-hybrid-ds-qwen35-9b8-probe.json"))
    args = parser.parse_args()
    prompt = "Write Python dedupe(items) preserving insertion order. Explain complexity and provide three asserts."
    engine = InferenceEngine(cache_budget_bytes=0, adaptive_scheduler="ds",
                             experimental_hybrid_speculation=True)
    request = GenerationRequest(profile=args.profile, prompt=prompt, use_cache=False,
                                max_tokens=args.max_tokens)
    report = {"source": source_manifest(), "scope": "experimental correctness/integration probe; no speed claim",
              "prompt": prompt, "samples": [], "divergence_diagnostic": None}

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    try:
        engine._load(args.profile, True)
        report["models"] = {"profile": args.profile, "target_path_revision": Path(engine.paths["target"]).name,
                            "draft_path_revision": Path(engine.paths["draft"]).name}
        for variant, mode, budget in [("baseline", "baseline", 1), ("fixed1", "fixed_speculative", 1),
                                      ("fixed2", "fixed_speculative", 2), ("ds", "adaptive", 2)]:
            result = engine.generate(dataclasses.replace(request, mode=mode, num_draft_tokens=budget))
            report["samples"].append({"variant": variant, "result": dataclasses.asdict(result)})
            save()
        baseline = report["samples"][0]["result"]["token_ids"]
        for sample in report["samples"]:
            ids = sample["result"]["token_ids"]
            sample["exact_to_baseline"] = ids == baseline
            sample["first_divergence"] = next((i for i, (a, b) in enumerate(zip(ids, baseline)) if a != b),
                                              min(len(ids), len(baseline)) if len(ids) != len(baseline) else None)
        # Fixed2 replay avoids timing-dependent choices and captures only the
        # first mismatch. This profiling run is separate from latency samples.
        tokens = engine._tokens(request)
        target, draft = make_prompt_cache(engine.model), make_prompt_cache(engine.draft_model)
        for position in range(0, len(tokens) - 1, 512):
            values = mx.array(tokens[position:min(position + 512, len(tokens) - 1)], mx.uint32)[None]
            engine.model(values, cache=target)
            engine.draft_model(values, cache=draft)
            mx.eval([c.state for c in target + draft])
        captured = []

        def on_verify(event):
            if captured or event["cache_snapshot"] is None:
                return
            for local in range(event["produced"]):
                absolute = event["generated_before"] + local
                if absolute < len(baseline) and event["predicted"][local] != baseline[absolute]:
                    captured.append({**event, "local_index": local, "absolute_index": absolute,
                                     "baseline_token": baseline[absolute]})
                    return

        generator = hybrid_generate_step(tokens[-1:], engine.model, engine.draft_model, target, draft,
                                         FixedPolicy(2), request.max_tokens, [], verification_hook=on_verify)
        try:
            for token, _ in generator:
                if token in engine.tokenizer.eos_token_ids:
                    break
        finally:
            generator.close()
        if captured:
            event = captured[0]
            single = make_prompt_cache(engine.model)
            restore_cache(single, event["cache_snapshot"])
            offsets_before = [getattr(c, "offset", None) for c in single]
            single_logits = None
            for token in event["inputs"][:event["local_index"] + 1]:
                single_logits = engine.model(mx.array([[token]], mx.uint32), cache=single)[0, -1]
                mx.eval(single_logits)
            batch_logits = event["logits"][event["local_index"]]
            mx.eval(batch_logits)

            def top2(vector):
                indices = mx.argsort(vector)[-2:][::-1].tolist()
                values = [float(vector[i].item()) for i in indices]
                return {"ids": indices, "logits": values, "margin": values[0] - values[1]}

            single_top, batch_top = top2(single_logits), top2(batch_logits)
            report["divergence_diagnostic"] = {
                "generated_position": event["absolute_index"], "local_verification_position": event["local_index"],
                "baseline_token": event["baseline_token"], "verification_inputs": event["inputs"],
                "same_prestate_single": single_top, "original_batch": batch_top,
                "max_abs_logit_difference": float(mx.max(mx.abs(single_logits - batch_logits)).item()),
                "cache_classes": [type(c).__name__ for c in single], "kv_offsets_before": offsets_before,
                "kv_offsets_after_single": [getattr(c, "offset", None) for c in single],
                "interpretation": "Current batch shape changes argmax on the identical prestate/history"
                if single_top["ids"][0] != batch_top["ids"][0]
                else "Current same-prestate argmax agrees; prior state trajectory or baseline path needs isolation",
                "scope": "separate fixed2 profiling pass, not timed performance evidence",
            }
        report["summary"] = {
            "exact_counts": {s["variant"]: s["exact_to_baseline"] for s in report["samples"]},
            "ds_core_decisions": sum("scheduler_replay" in r for s in report["samples"]
                                     if s["variant"] == "ds" for r in s["result"]["rounds"]),
            "equivalent_speedup_claim_supported": False, "competitive_advantage_claim_supported": False,
        }
        save()
        print(json.dumps(report["summary"], indent=2))
    finally:
        engine.close()


if __name__ == "__main__":
    main()
