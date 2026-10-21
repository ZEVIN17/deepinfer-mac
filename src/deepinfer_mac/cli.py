"""Command line entry points. Importing diagnostics does not load MLX."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def parser():
    root = argparse.ArgumentParser(prog="deepinfer-mac", description="Local Apple Silicon inference powered by MLX")
    commands = root.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Check Python, Metal, disk and memory")
    doctor.add_argument("--json", action="store_true")
    models = commands.add_parser("models", help="List locked model profiles or download their weights")
    models.add_argument("--download", metavar="PROFILE")
    models.add_argument("--target-only", action="store_true", help="Download target without draft")
    models.add_argument("--json", action="store_true")
    models.add_argument("--model-config", type=Path, action="append", default=[])
    for command in ("run", "bench"):
        sub = commands.add_parser(command, help="Generate text" if command == "run" else "Run actual rotating-order measurements")
        sub.add_argument("--profile", default="qwen3-4b")
        sub.add_argument("--model", help="Local MLX directory or Hugging Face repo")
        sub.add_argument("--revision", help="Exact repo commit; omit for local directories")
        sub.add_argument("--model-id", default="custom-model")
        sub.add_argument("--model-config", type=Path, action="append", default=[])
        sub.add_argument("--fallback-to-baseline", action="store_true")
        sub.add_argument("--max-tokens", type=int, default=128 if command == "run" else 64)
        sub.add_argument("--thinking", action="store_true")
        sub.add_argument("--prefill-step-size", type=int, default=512)
        sub.add_argument("--cache-budget-mb", type=int, default=512)
        sub.add_argument("--adaptive-scheduler", choices=("legacy", "ds"), default="legacy")
        sub.add_argument("--cache-backend", choices=("native", "exact"), default="native")
        if command == "run":
            sub.add_argument("prompt", nargs="?", help="Prompt text; omit to read stdin")
            sub.add_argument("--mode", choices=("baseline", "fixed_speculative", "adaptive"), default="baseline")
            sub.add_argument("--draft-tokens", type=int, default=3)
            sub.add_argument("--no-cache", action="store_true")
            sub.add_argument("--temperature", type=float, default=0)
            sub.add_argument("--top-p", type=float, default=1)
            sub.add_argument("--top-k", type=int, default=0)
            sub.add_argument("--presence-penalty", type=float, default=0)
            sub.add_argument("--seed", type=int)
            sub.add_argument("--json", action="store_true", help="Print complete result instead of streaming text")
        else:
            sub.add_argument("--suite", choices=("quick", "full"), default="quick")
            sub.add_argument("--repeats", type=int, default=3)
            sub.add_argument("--fixed-lengths", type=int, nargs="+", default=[1, 2, 4])
            sub.add_argument("--output", type=Path, default=Path("benchmark-results"))
    serve = commands.add_parser("serve", help="Serve the bundled local chat and supported OpenAI API subset")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--queue-size", type=int, default=4)
    serve.add_argument("--model-config", type=Path, action="append", default=[])
    serve.add_argument("--token", help="Bearer token; DEEPINFER_MAC_TOKEN is preferred to avoid shell history")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        from .adapters import ModelRegistry, ModelSpec
        registry = ModelRegistry(ModelSpec.from_json(path) for path in getattr(args, "model_config", []))
        if getattr(args, "model", None):
            args.profile = registry.register(ModelSpec(id=args.model_id, source=args.model, revision=args.revision))
        if getattr(args, "revision", None) and not getattr(args, "model", None):
            raise ValueError("--revision requires --model")
        if args.command == "doctor":
            from .doctor import print_diagnostics
            return 0 if print_diagnostics(args.json) else 1
        if args.command == "models":
            from .models import download_models
            if args.download:
                result = download_models(args.download, include_draft=not args.target_only, registry=registry)
                print(json.dumps(result, indent=2, default=str))
            elif args.json:
                print(json.dumps(registry.list(), indent=2))
            else:
                for model in registry.list():
                    print(json.dumps(model, ensure_ascii=False))
            return 0
        if args.command == "serve":
            import uvicorn

            from .server.app import create_app
            if args.host not in ("127.0.0.1", "localhost", "::1") and not (args.token or os.getenv("DEEPINFER_MAC_TOKEN")):
                raise ValueError("A bearer token is required when binding outside localhost")
            uvicorn.run(create_app(queue_size=args.queue_size, token=args.token, model_registry=registry), host=args.host, port=args.port)
            return 0
        from .engine import InferenceEngine
        from .state import GenerationRequest
        prompt = (args.prompt if args.prompt is not None else sys.stdin.read()) if args.command == "run" else "Standard benchmark corpus"
        if not prompt.strip():
            raise ValueError("Prompt must not be empty")
        if args.cache_budget_mb < 0:
            raise ValueError("cache-budget-mb must be non-negative")
        request = GenerationRequest(profile=args.profile, fallback_to_baseline=args.fallback_to_baseline, prompt=prompt, max_tokens=args.max_tokens,
                                    enable_thinking=args.thinking, prefill_step_size=args.prefill_step_size,
                                    temperature=args.temperature if args.command == "run" else 0,
                                    top_p=args.top_p if args.command == "run" else 1,
                                    top_k=args.top_k if args.command == "run" else 0,
                                    presence_penalty=args.presence_penalty if args.command == "run" else 0,
                                    seed=args.seed if args.command == "run" else None,
                                    mode=args.mode if args.command == "run" else "baseline",
                                    num_draft_tokens=args.draft_tokens if args.command == "run" else 3,
                                    use_cache=not args.no_cache if args.command == "run" else False)
        engine = InferenceEngine(cache_budget_bytes=args.cache_budget_mb * 1024**2,
                                 adaptive_scheduler=args.adaptive_scheduler, cache_backend=args.cache_backend, model_registry=registry)
        try:
            if args.command == "run":
                from .benchmark.export import jsonable
                def emit(event):
                    if event.get("type") == "token" and event.get("text") and not args.json:
                        print(event["text"], end="", flush=True)
                result = jsonable(engine.generate(request, emit=emit))
                print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else "")
                if not args.json:
                    print(json.dumps(result["metrics"], ensure_ascii=False), file=sys.stderr)
            else:
                from .benchmark.export import export_report
                from .benchmark.runner import run_suite
                def progress(event):
                    if event.get("type") == "mode_done":
                        result = event["result"]
                        print(f"{event['mode']}: {result.get('error') or result.get('metrics', {}).get('total_ms')} ms", file=sys.stderr, flush=True)
                args.output.mkdir(parents=True, exist_ok=True)
                state = {"pid": os.getpid(), "started_at_unix": time.time(), "status": "running",
                         "profile": args.profile, "suite": args.suite, "repeats": args.repeats,
                         "max_tokens": args.max_tokens, "fixed_lengths": args.fixed_lengths}
                state_path = args.output / "run.json"
                state_path.write_text(json.dumps(state, indent=2) + "\n")
                try:
                    with (args.output / "samples.jsonl").open("w") as checkpoint:
                        def save_sample(sample):
                            checkpoint.write(json.dumps(sample, ensure_ascii=False) + "\n")
                            checkpoint.flush()
                        report = run_suite(engine, request, repeats=args.repeats, suite=args.suite,
                                           fixed_lengths=tuple(args.fixed_lengths), emit=progress,
                                           save_prompts=True, on_sample=save_sample)
                    destination = export_report(report, args.output)
                    state["status"] = "completed"
                    print(destination.resolve())
                finally:
                    if state["status"] == "running":
                        state["status"] = "interrupted_or_failed"
                    state["ended_at_unix"] = time.time()
                    state_path.write_text(json.dumps(state, indent=2) + "\n")
        finally:
            if hasattr(engine, "close"):
                engine.close()
        return 0
    except KeyboardInterrupt:
        print("Cancelled", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 — isolate arbitrary model/backend failures
        print(f"deepinfer-mac: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
