"""Audit-only oMLX wrapper without replacing decoding/acceptance algorithms.

JSONL logs include prompt IDs, successful binding, actual draft calls and the
upstream verification counters. No extra tensor evaluation or synchronization is
introduced. File I/O and Python wrappers can affect adaptive budget decisions;
every run is diagnostic, never a performance benchmark. Omit this wrapper in
formal latency measurements.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import importlib.metadata
import json
import os
import struct
import sys
import threading
import time
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dflash-audit", type=Path, required=True)
    args, rest = parser.parse_known_args()
    if importlib.metadata.version("omlx") != "0.7.1":
        raise RuntimeError("Reaudit these source hooks before upgrading oMLX")
    args.dflash_audit.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def emit(event, **data):
        record = {"event": event, "monotonic_seconds": time.monotonic(), **data}
        with lock, args.dflash_audit.open("a") as output:
            output.write(json.dumps(record) + "\n")

    import mlx.core as mx
    import psutil
    from mlx.utils import tree_flatten
    from omlx.engine.batched import BatchedEngine
    from omlx.patches.mlx_lm_mtp import batch_generator as bg
    from omlx.scheduler import Scheduler
    from omlx.speculative.dflash.drafter import DFlashDrafter

    def resources():
        return {
            "metal_active_bytes": mx.get_active_memory(),
            "metal_peak_bytes": mx.get_peak_memory(),
            "metal_cache_bytes": mx.get_cache_memory(),
            "process_rss_bytes": psutil.Process().memory_info().rss,
        }

    mx.reset_peak_memory()
    emit("audit_started", pid=os.getpid(), version=importlib.metadata.version("omlx"),
         environment={p: importlib.metadata.version(p) for p in ("mlx", "mlx-lm", "mlx-vlm", "transformers")},
         hook_sources_sha256={str(Path(m.__file__).relative_to(Path(m.__file__).parents[0])):
                              hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                              for m in (bg,)}, resources=resources())

    original_add = Scheduler.add_request

    @functools.wraps(original_add)
    def add_request(self, request):
        result = original_add(self, request)
        ids = list(map(int, request.prompt_token_ids))
        emit("request_prompt", request_id=str(request.request_id), token_ids=ids,
             prompt_tokens=len(ids), token_ids_sha256_le_u32=hashlib.sha256(
                 struct.pack(f"<{len(ids)}I", *ids)).hexdigest(), resources=resources())
        return result

    Scheduler.add_request = add_request
    original_cleanup = Scheduler._cleanup_finished

    @functools.wraps(original_cleanup)
    def cleanup(self, finished_ids):
        result = original_cleanup(self, finished_ids)
        if finished_ids:
            emit("requests_cleanup_completed", request_ids=sorted(map(str, finished_ids)),
                 resources=resources())
        return result

    Scheduler._cleanup_finished = cleanup
    original_attach = BatchedEngine.set_dflash_drafter

    @functools.wraps(original_attach)
    def attach(self, drafter):
        result = original_attach(self, drafter)
        host = self._dflash_host()
        owned = [(name, array) for name, array in tree_flatten(drafter.model.parameters())
                 if not name.startswith(("embed_tokens.", "lm_head."))]
        modules = [(name, module) for name, module in drafter.model.named_modules()
                   if not name.startswith(("embed_tokens", "lm_head"))]
        emit("drafter_attached", attached_identity=getattr(host, "_omlx_drafter", None) is drafter,
             decode_enabled=getattr(host, "_omlx_mtp_decode_enabled", None),
             engine_property_identity=self.dflash_drafter is drafter,
             target_class=type(self._model).__module__ + "." + type(self._model).__name__,
             drafter_class=type(drafter.model).__module__ + "." + type(drafter.model).__name__,
             source_path=drafter.source_path, normalized_config=dataclasses.asdict(drafter.model.config),
             kind=drafter.kind, block_size=drafter.block_size, adaptive=drafter.adaptive,
             window=drafter.window, target_layer_ids=drafter.target_layer_ids,
             owned_parameter_storage_bytes=sum(a.nbytes for _, a in owned),
             owned_module_classes=dict(Counter(type(module).__name__ for _, module in modules)),
             quantized_modules=[{"name": name, "type": type(module).__name__,
                                 "bits": getattr(module, "bits", None),
                                 "group_size": getattr(module, "group_size", None)}
                                for name, module in modules if hasattr(module, "bits")],
             resources=resources())
        return result

    BatchedEngine.set_dflash_drafter = attach
    original_draft = DFlashDrafter._draft_batched

    @functools.wraps(original_draft)
    def draft(self, rows, *positional, **keywords):
        try:
            result = original_draft(self, rows, *positional, **keywords)
        except Exception as error:
            emit("draft_error", error_type=type(error).__name__, error=str(error), rows=len(rows))
            raise
        emit("draft_batched_completed", rows=len(rows), block_size=self.block_size,
             proposal_shapes=[list(tokens.shape) for tokens, _ in result])
        return result

    DFlashDrafter._draft_batched = draft
    original_stats = bg._log_mtp_stats

    @functools.wraps(original_stats)
    def stats(uid, values, finish_reason):
        emit("upstream_verify_stats", uid=str(uid), finish_reason=finish_reason,
             stats=dataclasses.asdict(values), resources=resources())
        return original_stats(uid, values, finish_reason)

    bg._log_mtp_stats = stats
    original_verify = bg._run_verify_cycle_chain

    @functools.wraps(original_verify)
    def verify(gen_batch, state, *positional, **keywords):
        before = state.stats.cycles
        try:
            result = original_verify(gen_batch, state, *positional, **keywords)
        except Exception as error:
            emit("verify_error", error_type=type(error).__name__, error=str(error))
            raise
        emit("verify_chain_completed", uid=str(state.uid),
             cycles_before=before, cycles_after=state.stats.cycles,
             accepts=state.stats.accepts, rejects=state.stats.rejects)
        return result

    bg._run_verify_cycle_chain = verify
    from omlx.cli import main as official_main

    sys.argv = [sys.argv[0], *rest]
    try:
        official_main()
    finally:
        emit("audit_stopped", resources=resources())


if __name__ == "__main__":
    main()
