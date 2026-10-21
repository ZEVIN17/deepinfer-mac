"""Launch the project service with explicit benchmark cache configuration."""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--cache-budget-bytes", type=int, default=512 * 1024**2)
    parser.add_argument("--cache-backend", choices=("native", "exact"), default="native")
    parser.add_argument("--checkpoint-interval-tokens", type=int, default=0)
    parser.add_argument("--adaptive-scheduler", choices=("ds", "legacy"), default="ds")
    parser.add_argument("--token-audit")
    parser.add_argument("--message-checkpoints", action="store_true")
    args = parser.parse_args()
    import uvicorn

    from deepinfer_mac.engine import InferenceEngine
    from deepinfer_mac.server.app import create_app

    def factory():
        engine = InferenceEngine(
            cache_budget_bytes=args.cache_budget_bytes,
            cache_backend=args.cache_backend,
            checkpoint_interval_tokens=args.checkpoint_interval_tokens,
            adaptive_scheduler=args.adaptive_scheduler,
            message_checkpoints=args.message_checkpoints,
        )
        if args.token_audit:
            from prompt_token_audit import append

            original_tokens = engine._tokens

            def tokens(request):
                result = original_tokens(request)
                append(args.token_audit, "deepinfer", result)
                return result

            engine._tokens = tokens
        return engine

    uvicorn.run(create_app(engine_factory=factory), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
