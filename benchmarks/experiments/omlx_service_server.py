"""Official oMLX CLI, with optional audit-only token trace for text requests."""

import argparse
import importlib.metadata
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--token-audit")
    parser.add_argument("--runtime-audit")
    args, rest = parser.parse_known_args()
    if importlib.metadata.version("omlx") != "0.7.1":
        raise RuntimeError("Reaudit oMLX adapter when upgrading the official version")
    if args.token_audit or args.runtime_audit:
        import json
        from pathlib import Path

        from omlx.scheduler import Scheduler
        from prompt_token_audit import append

        original_add = Scheduler.add_request

        def add_request(self, request):
            result = original_add(self, request)
            if args.token_audit:
                append(args.token_audit, "omlx", request.prompt_token_ids)
            if args.runtime_audit:
                names = (
                    "hot_cache_only",
                    "hot_cache_max_size",
                    "paged_ssd_cache_max_size",
                    "prefill_step_size",
                    "max_num_seqs",
                    "block_size",
                )
                record = {name: getattr(self.config, name, None) for name in names}
                record["block_aware_cache_active"] = self.block_aware_cache is not None
                with Path(args.runtime_audit).open("a") as output:
                    output.write(json.dumps(record) + "\n")
            return result

        Scheduler.add_request = add_request
    from omlx.cli import main as official_main

    sys.argv = [sys.argv[0], *rest]
    official_main()


if __name__ == "__main__":
    main()
