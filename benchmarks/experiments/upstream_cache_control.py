"""Same engine/model/prefill/snapshot policy; compare cache lookup implementations."""

import copy
import dataclasses
import json
import statistics
from pathlib import Path

from mlx_lm.models.cache import LRUPromptCache

from deepinfer_mac.cache import PromptCacheStore
from deepinfer_mac.engine import InferenceEngine
from deepinfer_mac.provenance import source_manifest
from deepinfer_mac.state import GenerationRequest


class NativeStore:
    def __init__(self, budget):
        self.budget = budget
        self.evictions = 0
        self.clear()

    def clear(self):
        self.store = LRUPromptCache(max_size=10000, max_bytes=self.budget)

    @property
    def nbytes(self):
        return self.store.nbytes

    def get(self, identity, tokens):
        cache, rest = self.store.fetch_nearest_cache(identity, tokens)
        return cache, len(tokens) - len(rest)

    def put(self, identity, tokens, cache):
        if sum(c.nbytes for c in cache) <= self.budget:
            self.store.insert_cache(identity, list(tokens), copy.deepcopy(cache))


def main():
    engine = InferenceEngine()
    base = [
        {
            "role": "system",
            "content": "This is public benchmark context. Preserve exact input and use the requested output format. "
            * 90,
        }
    ]

    def req(i, cache=True):
        return GenerationRequest(
            messages=base + [{"role": "user", "content": f"Compute {i}+{i}. Give only the number."}],
            max_tokens=32,
            use_cache=cache,
        )

    engine.generate(req(2, False))
    out = {
        "source": source_manifest(),
        "scope": "MLX-LM LRUPromptCache vs project cache; shared engine and same model, not complete servers",
        "pairs": [],
    }
    path = Path("benchmarks/reports/phase2-native-cache-control.json")
    for i in range(3):
        ref = engine.generate(req(3, False))
        pair = {}
        for name in (
            ["current", "periodic", "native_lru"] if i % 2 == 0 else ["native_lru", "periodic", "current"]
        ):
            engine.cache = (
                NativeStore(512 * 1024**2) if name == "native_lru" else PromptCacheStore(512 * 1024**2)
            )
            engine.checkpoint_interval_tokens = 512 if name == "periodic" else 0
            seed = engine.generate(req(2))
            branch = engine.generate(req(3))
            pair[name] = {
                "seed": dataclasses.asdict(seed),
                "branch": dataclasses.asdict(branch),
                "equal": branch.token_ids == ref.token_ids,
            }
        out["pairs"].append(pair)
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    out["summary"] = {
        name: {
            "ttft_ms": statistics.median(p[name]["branch"]["metrics"]["ttft_ms"] for p in out["pairs"]),
            "hit_tokens": [p[name]["branch"]["metrics"]["cache_hit_tokens"] for p in out["pairs"]],
            "equal": all(p[name]["equal"] for p in out["pairs"]),
        }
        for name in ["current", "periodic", "native_lru"]
    }
    engine.close()
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()
