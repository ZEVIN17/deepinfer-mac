"""Run locked upstream MLX-LM server with its declared byte budget enforced.

MLX-LM 0.32.2 parses --prompt-cache-bytes but run() passes only max_size to
LRUPromptCache. This adapter supplies max_bytes; it preserves upstream server
generation, batching, segment checkpoints, sampler, and prefix lookup.
"""

import argparse
import importlib.metadata
import json
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cache-budget-bytes", type=int, default=512 * 1024**2)
    parser.add_argument("--token-audit")
    args, rest = parser.parse_known_args()
    if args.cache_budget_bytes < 0:
        raise ValueError("Cache byte budget must be nonnegative")
    version = importlib.metadata.version("mlx-lm")
    if version != "0.32.2":
        raise RuntimeError(f"Reaudit native byte-budget adapter before changing mlx-lm {version}")
    from mlx_lm import server

    original = server.LRUPromptCache

    class ByteBudgetCache(original):
        def __init__(self, max_size=10, max_bytes=None):
            super().__init__(max_size=max_size, max_bytes=args.cache_budget_bytes)

    server.LRUPromptCache = ByteBudgetCache
    if args.token_audit:
        from prompt_token_audit import append

        original_tokenize = server.ResponseGenerator._tokenize

        def tokenize(self, *positional, **keywords):
            result = original_tokenize(self, *positional, **keywords)
            append(args.token_audit, "native_mlx", result[0])
            return result

        server.ResponseGenerator._tokenize = tokenize
    print(
        json.dumps(
            {
                "adapter": "native-byte-budget-only",
                "mlx_lm_version": version,
                "cache_max_bytes": args.cache_budget_bytes,
            }
        ),
        flush=True,
    )
    sys.argv = [sys.argv[0], *rest]
    server.main()


if __name__ == "__main__":
    main()
