#!/usr/bin/env python3
"""Conservative Qwen3.8-27B preflight; --run is reserved for the sole GPU owner.

No MLX import or network access occurs during preflight. Budget numbers are
estimates, never measurements of an allocation failure or attainable throughput.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "docs/qwen27.manifest.json"


def memory_snapshot():
    import psutil

    vm = psutil.virtual_memory()
    result = {
        "total_bytes": vm.total,
        "available_bytes": vm.available,
        "rss_bytes": psutil.Process().memory_info().rss,
    }
    try:
        swap = psutil.swap_memory()
        result.update(swap_used_bytes=swap.used, swap_available=True)
    except (OSError, PermissionError) as exc:
        result.update(swap_used_bytes=None, swap_available=False, swap_error=f"{type(exc).__name__}: {exc}")
    return result


def budget(manifest, prompt_tokens, max_tokens, cache_budget_bytes, workspace_bytes, safety_bytes):
    cfg = manifest["text_config"]
    # MLX KVCache allocates in 256-token steps; leave one extra step during growth.
    positions = math.ceil((prompt_tokens + max_tokens) / 256) * 256 + 256
    full_layers = cfg["num_hidden_layers"] // cfg["full_attention_interval"]
    linear_layers = cfg["num_hidden_layers"] - full_layers
    kv = positions * full_layers * 2 * cfg["num_key_value_heads"] * cfg["head_dim"] * 2
    recurrent = (
        linear_layers
        * cfg["linear_num_value_heads"]
        * cfg["linear_key_head_dim"]
        * cfg["linear_value_head_dim"]
        * 4
    )
    conv_dim = (
        2 * cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]
        + cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"]
    )
    convolution = linear_layers * (cfg["linear_conv_kernel_dim"] - 1) * conv_dim * 2
    text_bytes = manifest.get("text_tensor_bytes")
    weights = text_bytes if text_bytes is not None else manifest["all_tensor_bytes"]
    parts = {
        "text_weights_or_full_upper_bound_bytes": weights,
        "active_kv_upper_estimate_bytes": kv,
        "recurrent_fp32_bytes": recurrent,
        "convolution_bf16_bytes": convolution,
        "retained_cache_budget_bytes": cache_budget_bytes,
        "workspace_allowance_bytes": workspace_bytes,
        "available_memory_safety_margin_bytes": safety_bytes,
    }
    return {
        "components": parts,
        "required_available_bytes": sum(parts.values()),
        "weight_basis": "verified_safetensors_headers_text_only"
        if text_bytes is not None
        else "conservative_all_tensor_upper_bound",
        "assumptions": "batch=1; BF16 KV/conv; FP32 recurrent state; workspace is an allowance, not a measured peak; RSS and Metal bytes are not added",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--output", type=Path, default=Path("results/qwen27-preflight.json"))
    ap.add_argument("--prompt-tokens", type=int, choices=(2048, 4096), default=4096)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--prefill-step-size", type=int, default=128)
    ap.add_argument("--cache-budget-mib", type=int, default=512)
    ap.add_argument("--workspace-mib", type=int, default=2048)
    ap.add_argument("--safety-mib", type=int, default=1024)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument(
        "--run",
        action="store_true",
        help="Run real baseline/cache probe; coordinate exclusive GPU ownership first",
    )
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--workload", choices=("zh", "code"), default="zh")
    args = ap.parse_args()
    if args.run and args.preflight_only:
        ap.error("--run and --preflight-only are mutually exclusive")
    if min(args.max_tokens, args.prefill_step_size, args.repeat) < 1:
        ap.error("token counts, prefill step, and repeat must be positive")
    if min(args.cache_budget_mib, args.workspace_mib, args.safety_mib) < 0:
        ap.error("memory allowances cannot be negative")
    manifest = json.loads(args.manifest.read_text())
    # Second turn retains the first output plus role delimiters before decoding.
    probe_prompt_bound = args.prompt_tokens + args.max_tokens + 128
    estimate = budget(
        manifest,
        probe_prompt_bound,
        args.max_tokens,
        args.cache_budget_mib * 1024**2,
        args.workspace_mib * 1024**2,
        args.safety_mib * 1024**2,
    )
    before = memory_snapshot()
    enough = before["available_bytes"] >= estimate["required_available_bytes"]
    report = {
        "schema_version": 1,
        "model": manifest["repo"],
        "revision": manifest["revision"],
        "timestamp_unix": time.time(),
        "prompt_tokens_requested": args.prompt_tokens,
        "max_tokens": args.max_tokens,
        "batch_size": 1,
        "prefill_step_size": args.prefill_step_size,
        "conversation_prompt_budget_tokens": probe_prompt_bound,
        "preflight": {
            "memory": before,
            "budget": estimate,
            "memory_budget_pass": enough,
            "swap_observable": before["swap_available"],
            "status": "eligible_for_measured_probe"
            if enough and before["swap_available"]
            else "blocked_before_model_load",
            "note": "A budget gate, not an OOM experiment; passing does not establish stable performance.",
        },
        "workload": args.workload,
        "measurements": [],
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    print(json.dumps(report, indent=2), flush=True)
    if not args.run:
        return 0
    if not enough or not before["swap_available"]:
        print("Real load refused: available-memory budget or observable swap requirement failed.", flush=True)
        return 2
    # Kept after the gate: --preflight-only is usable on hosts without Metal.
    from deepinfer_mac.engine import InferenceEngine
    from deepinfer_mac.state import GenerationRequest

    engine = InferenceEngine(cache_budget_bytes=args.cache_budget_mib * 1024**2)
    cancel = threading.Event()
    stop_monitor = threading.Event()
    samples = []

    def monitor():
        while not stop_monitor.wait(0.25):
            sample = memory_snapshot()
            samples.append({"elapsed_s": time.perf_counter() - started, **sample})
            swap = [s["swap_used_bytes"] for s in samples[-4:]]
            if (
                len(swap) == 4
                and all(v is not None for v in swap)
                and all(a < b for a, b in itertools.pairwise(swap))
                and swap[-1] - before["swap_used_bytes"] > 64 * 1024**2
            ):
                report["sustained_swap_growth_cancelled"] = True
                cancel.set()
            if not sample["swap_available"]:
                report["swap_observation_lost"] = True
                cancel.set()

    started = time.perf_counter()
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        engine._load("qwen3.8-27b", False)
        # Use an exact template-counted prompt as close as possible below target.
        if args.workload == "zh":
            lead = "请阅读以下公开测试资料，然后用中文解释缓存与内存预算。\n"
            context = (
                "缓存复用相同前缀可以减少重复计算，资源预算应记录内存和交换空间。\n" * args.prompt_tokens
            )
            followup = "用三条要点总结刚才的答案，并说明一个风险。"
        else:
            lead = "Read the test notes, then implement a bounded LRU cache in Python.\n"
            context = (
                "# Public test note: track exact byte budgets; evict oldest entries; preserve shared prefixes.\n"
                * args.prompt_tokens
            )
            followup = (
                "Add a short example that branches two independent requests from the same cached prefix."
            )
        low, high = 0, len(context)
        request = GenerationRequest(
            profile="qwen3.8-27b",
            mode="baseline",
            prompt=lead,
            max_tokens=args.max_tokens,
            prefill_step_size=args.prefill_step_size,
        )
        while low < high:
            mid = (low + high + 1) // 2
            trial = dataclasses.replace(request, prompt=lead + context[:mid])
            if len(engine._tokens(trial)) <= args.prompt_tokens:
                low = mid
            else:
                high = mid - 1
        request = dataclasses.replace(request, prompt=lead + context[:low])
        # Token boundaries can merge at the truncated character boundary. Fill
        # any small gap by checking the complete formatted prompt each time.
        while len(engine._tokens(request)) < args.prompt_tokens:
            current = len(engine._tokens(request))
            for pad in (" a", " x", ".", "\n", "中"):
                trial = dataclasses.replace(request, prompt=request.prompt + pad)
                count = len(engine._tokens(trial))
                if current < count <= args.prompt_tokens:
                    request = trial
                    break
            else:
                raise ValueError("Could not construct an exact template-counted probe prompt")
        report["actual_prompt_tokens"] = len(engine._tokens(request))
        for repeat in range(args.repeat):
            reference_ids = None
            for kind, use_cache in (("cold", False), ("cache_seed", True), ("exact_prefix_hit", True)):
                if kind in ("cold", "cache_seed"):
                    engine.clear_cache()
                if cancel.is_set():
                    raise InterruptedError("Sustained swap growth or lost swap observation")
                entry = {"repeat": repeat, "kind": kind, "memory_before": memory_snapshot()}
                result = engine.generate(dataclasses.replace(request, use_cache=use_cache), cancel=cancel)
                if reference_ids is None:
                    reference_ids = result.token_ids
                entry["token_sequence_matches_cold"] = result.token_ids == reference_ids
                entry.update(result=dataclasses.asdict(result), memory_after=memory_snapshot())
                report["measurements"].append(entry)
                save()
            # A real second conversation turn reuses a checkpointed prefix. It
            # is compared with a fresh ordinary decode of the same messages.
            turn = dataclasses.replace(
                request,
                prompt="",
                messages=[
                    {"role": "user", "content": request.prompt},
                    {"role": "assistant", "content": result.text},
                    {"role": "user", "content": followup},
                ],
            )
            branch_results = []
            for kind, use_cache in (("conversation_cached", True), ("conversation_cold", False)):
                if cancel.is_set():
                    raise InterruptedError("Sustained swap growth or lost swap observation")
                generated = engine.generate(dataclasses.replace(turn, use_cache=use_cache), cancel=cancel)
                branch_results.append(generated.token_ids)
                report["measurements"].append(
                    {
                        "repeat": repeat,
                        "kind": kind,
                        "result": dataclasses.asdict(generated),
                        "memory_after": memory_snapshot(),
                    }
                )
                save()
            report["measurements"][-1]["token_sequence_matches_cached_turn"] = (
                branch_results[0] == branch_results[1]
            )
        report["status"] = "measured"
        comparisons = [
            value
            for entry in report["measurements"]
            for key, value in entry.items()
            if key in ("token_sequence_matches_cold", "token_sequence_matches_cached_turn")
        ]
        report["correctness_pass"] = bool(comparisons) and all(comparisons)
        if not report["correctness_pass"]:
            report["status"] = "measured_with_token_mismatch"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        stop_monitor.set()
        watcher.join(timeout=1)
        engine.close()
        report["memory_samples"] = samples
        report["memory_after_unload"] = memory_snapshot()
        report["swap_delta_bytes"] = (
            report["memory_after_unload"]["swap_used_bytes"] - before["swap_used_bytes"]
            if report["memory_after_unload"]["swap_available"]
            else None
        )
        save()
    return 0 if report["correctness_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
