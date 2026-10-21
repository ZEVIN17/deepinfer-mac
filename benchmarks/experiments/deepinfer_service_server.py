"""Launch the project service with explicit benchmark cache configuration."""

import argparse
import dataclasses


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--cache-budget-bytes", type=int, default=512 * 1024**2)
    parser.add_argument("--cache-backend", choices=("native", "exact"), default="native")
    parser.add_argument("--checkpoint-interval-tokens", type=int, default=0)
    parser.add_argument("--adaptive-scheduler", choices=("ds", "legacy"), default="ds")
    parser.add_argument("--token-audit")
    parser.add_argument("--message-checkpoints", action="store_true")
    parser.add_argument("--prefill-step-size", type=int, default=512)
    parser.add_argument("--experimental-hybrid-speculation", action="store_true")
    parser.add_argument("--hybrid-cost-adapter", choices=("original", "lagged_mean"), default="original")
    parser.add_argument("--round-diagnostics", action="store_true", help="Profiling only; performance service uses lightweight counts")
    return parser


def build_app(args):
    if args.prefill_step_size < 1:
        raise ValueError("--prefill-step-size must be positive")
    from deepinfer_mac.engine import InferenceEngine
    from deepinfer_mac.server.app import create_app

    def factory():
        engine = InferenceEngine(
            cache_budget_bytes=args.cache_budget_bytes,
            cache_backend=args.cache_backend,
            checkpoint_interval_tokens=args.checkpoint_interval_tokens,
            adaptive_scheduler=args.adaptive_scheduler,
            message_checkpoints=args.message_checkpoints,
            experimental_hybrid_speculation=args.experimental_hybrid_speculation,
            hybrid_cost_adapter=args.hybrid_cost_adapter,
            round_diagnostics=args.round_diagnostics,
        )
        original_generate = engine.generate

        def generate(request, emit=None, cancel=None):
            return original_generate(dataclasses.replace(request, prefill_step_size=args.prefill_step_size),
                                     emit=emit, cancel=cancel)

        engine.generate = generate
        if args.token_audit:
            from prompt_token_audit import append

            original_tokens = engine._tokens

            def tokens(request):
                result = original_tokens(request)
                append(args.token_audit, "deepinfer", result)
                return result

            engine._tokens = tokens
        return engine

    return create_app(engine_factory=factory,
                      experimental_hybrid_speculation=args.experimental_hybrid_speculation)


def main():
    options = parser()
    args = options.parse_args()
    if args.hybrid_cost_adapter != "original" and (
        not args.experimental_hybrid_speculation or args.adaptive_scheduler != "ds"
    ):
        options.error("--hybrid-cost-adapter=lagged_mean requires --experimental-hybrid-speculation and --adaptive-scheduler=ds")
    import uvicorn

    uvicorn.run(build_app(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
