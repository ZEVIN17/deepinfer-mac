"""Serial measurements with rotating order; never synthesize performance data."""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from importlib.resources import files

from .export import jsonable
from .metrics import aggregate, paired_comparison

MODES = ("baseline", "fixed_speculative", "adaptive")


def compare_once(engine, request, emit=None, cancel=None, order=MODES, *, retain_failures=False,
                 reset_cache=False) -> dict:
    results = []
    for variant in order:
        if cancel and cancel.is_set():
            raise InterruptedError("Comparison cancelled")
        mode, _, draft_length = variant.partition(":")
        if emit:
            emit({"type": "status", "stage": "generating", "mode": variant})

        def forward(event, current=variant):
            if emit:
                emit({**event, "mode": current})

        changes = {"mode": "baseline" if mode == "native_mlx" else mode}
        if draft_length:
            changes["num_draft_tokens"] = int(draft_length)
        if reset_cache:
            engine.clear_cache()
            changes["use_cache"] = False
        candidate = dataclasses.replace(request, **changes)
        try:
            generate = engine.native_generate if mode == "native_mlx" else engine.generate
            result = jsonable(generate(candidate, emit=forward, cancel=cancel))
            result["variant"] = variant
            results.append(result)
        except InterruptedError:
            raise
        except Exception as exc:
            if not retain_failures:
                raise
            results.append({"mode": mode, "variant": variant, "error": f"{type(exc).__name__}: {exc}",
                            "token_ids": [], "metrics": {}})
        if emit:
            emit({"type": "mode_done", "mode": variant, "result": results[-1]})
    return {"results": results, "comparison": paired_comparison(results)}


def _command(command):
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
        return completed.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def environment():
    versions = {}
    for name in ("deepinfer-mac", "mlx", "mlx-lm", "psutil"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    from ..provenance import source_manifest

    return {"source_manifest": source_manifest(), "platform": platform.platform(), "machine": platform.machine(), "versions": versions,
            "git_commit": _command(["git", "rev-parse", "HEAD"]),
            "git_status": _command(["git", "status", "--porcelain"]),
            "hardware": _command(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "power": _command(["pmset", "-g", "batt"]),
            "thermal": _command(["pmset", "-g", "therm"])}


def run_suite(engine, request, repeats=3, emit=None, cancel=None, save_prompts=False,
              suite="quick", fixed_lengths=(1, 2, 4), on_sample=None) -> dict:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if suite not in ("quick", "full"):
        raise ValueError("suite must be quick or full")
    if not fixed_lengths or any(length < 1 or length > 8 for length in fixed_lengths):
        raise ValueError("fixed_lengths must contain values from 1 to 8")
    prompts = json.loads(files("deepinfer_mac.benchmark").joinpath("quick.json").read_text())
    if suite == "quick":
        prompts = [item for item in prompts if item.get("split") == "validation"][:4]
    variants = ("native_mlx", "baseline", *(f"fixed_speculative:{n}" for n in fixed_lengths), "adaptive")
    samples = []
    before = environment()
    for prompt_index, item in enumerate(prompts):
        candidate = dataclasses.replace(request, prompt=item.get("prompt", ""), messages=item.get("messages"),
                                        use_cache=False)
        for iteration in range(-1, repeats):
            phase = "warmup" if iteration == -1 else "warm"
            shift = (prompt_index + iteration + 1) % len(variants)
            order = variants[shift:] + variants[:shift]
            compared = compare_once(engine, candidate, emit=emit, cancel=cancel, order=order,
                                    retain_failures=True, reset_cache=True)
            serialized = json.dumps({"prompt": item.get("prompt"), "messages": item.get("messages")},
                                    ensure_ascii=False, sort_keys=True)
            sample = {"sample_id": f"{item['id']}-{iteration}", "corpus_id": item["id"],
                      "split": item.get("split"), "category": item.get("category"), "iteration": iteration,
                      "phase": phase, "prompt_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
                      "order": list(order), **compared}
            if save_prompts:
                sample["input"] = json.loads(serialized)
            samples.append(sample)
            if on_sample is not None:
                on_sample(sample)
    return {"schema_version": 2, "suite": suite, "repeats": repeats,
            "environment": before, "environment_after": environment(),
            "configuration": {"max_tokens": request.max_tokens, "profile": request.profile,
                              "greedy": True, "thinking": request.enable_thinking,
                              "fixed_lengths": list(fixed_lengths), "cache": "empty KV before every run",
                              "timing": "Warm loaded model measurements; all warmups retained. No cold model speed comparison. Native MLX-LM stream_generate uses the same model/template. Upstream prompt_tps includes first-token work, not pure prefill."},
            "samples": samples, "warm_summary": aggregate(samples)}
