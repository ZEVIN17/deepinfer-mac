import gc
import json
from pathlib import Path

import mlx.core as mx
import psutil

from deepinfer_mac.engine import InferenceEngine

out = []


def snap(label):
    out.append(
        {
            "label": label,
            "active": mx.get_active_memory(),
            "cache": mx.get_cache_memory(),
            "rss": psutil.Process().memory_info().rss,
            "available": psutil.virtual_memory().available,
        }
    )


for i in range(4):
    e = InferenceEngine()
    e._load("qwen3-4b", True)
    snap(f"{i}-loaded")
    e.close()
    snap(f"{i}-closed")
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    snap(f"{i}-gc-cleared")
Path("benchmarks/reports/unload-resource-check.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out))
