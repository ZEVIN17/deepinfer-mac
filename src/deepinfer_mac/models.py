"""Pinned public model registry. No remote code is trusted."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

TARGET = {"repo": "mlx-community/Qwen3-4B-4bit", "revision": "4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25"}
DRAFT = {"repo": "mlx-community/Qwen3-0.6B-4bit", "revision": "73e3e38d981303bc594367cd910ea6eb48349da8"}
DEMO_27B = {"repo": "mlx-community/Qwen3.8-27B-4bit", "revision": "10c35caafbb80f7dc6a7a432cdd11af10a6d4818"}
PROFILES = {
    "qwen3-4b": {
        "target": TARGET,
        "draft": DRAFT,
        "minimum_memory_bytes": int(4.5 * 1024**3),
        "weight_bytes": 2263022529,
    },
    "qwen3-0.6b": {
        "target": DRAFT,
        "draft": DRAFT,
        "minimum_memory_bytes": int(1.5 * 1024**3),
        "weight_bytes": 335450584,
    },
    "qwen3.8-27b": {
        "target": DEMO_27B,
        "draft": None,
        "minimum_memory_bytes": 18 * 1024**3,
        "weight_bytes": 16054541349,
        "text_tensor_bytes": 15132802048,
        "context_limit": 8192,
        "supported_modes": ["baseline"],
        "status": "candidate",
        "limitations": "Text-only candidate; hybrid ArraysCache cannot trim; no verified MTP weights or compatible draft. 2K/4K stability unverified.",
    },
}


def get_profile(name):
    if name not in PROFILES:
        raise ValueError(f"Unknown profile {name!r}; choose {list(PROFILES)}")
    return PROFILES[name]


def list_models():
    return [
        {
            "id": name,
            **profile,
            "quantization": "4-bit, group size 64",
            "context_limit": profile.get("context_limit", 40960),
            "supported_modes": profile.get("supported_modes", ["baseline", "fixed_speculative", "adaptive"]),
            "capabilities": {"cache": True, "speculative": profile["draft"] is not None, "mtp": False},
            "status": profile.get("status", "experimental"),
        }
        for name, profile in PROFILES.items()
    ]


def download_models(profile="qwen3-4b", include_draft=True):
    from huggingface_hub import snapshot_download

    spec = get_profile(profile)
    out = {}
    if include_draft and spec["draft"] is None:
        raise ValueError("This profile has no verified compatible draft; use baseline")
    for kind in ("target", "draft") if include_draft else ("target",):
        model = spec[kind]
        try:
            cached = Path(snapshot_download(model["repo"], revision=model["revision"], local_files_only=True))
            index = json.loads((cached / "model.safetensors.index.json").read_text())
            required = set(index["weight_map"].values()) | {
                "tokenizer.json",
                "tokenizer_config.json",
                "config.json",
            }
            if spec["draft"] is None:
                required.add("chat_template.jinja")
            if not all((cached / name).is_file() for name in required):
                raise FileNotFoundError("Incomplete cached snapshot")
            out[kind] = str(cached)
        except (FileNotFoundError, OSError, KeyError):
            out[kind] = str(
                snapshot_download(
                    model["repo"],
                    revision=model["revision"],
                    allow_patterns=[
                        "*.json",
                        "*.jinja",
                        "*.safetensors",
                        "*.txt",
                        "*.model",
                        "LICENSE*",
                        "README*",
                    ],
                    max_workers=3,
                )
            )
    return out


def tokenizer_fingerprint(path):
    path = Path(path)
    return hashlib.sha256((path / "tokenizer.json").read_bytes()).hexdigest()


def verify_pair(target_path, draft_path):
    """Compare complete tokenizer vocabulary + special IDs, not just vocab size."""
    from transformers import AutoTokenizer

    a, b = [
        AutoTokenizer.from_pretrained(p, local_files_only=True, trust_remote_code=False)
        for p in (target_path, draft_path)
    ]
    if a.get_vocab() != b.get_vocab() or a.all_special_ids != b.all_special_ids:
        raise ValueError("Target and draft tokenizers are incompatible")
    configs = [json.loads((Path(p) / "config.json").read_text()) for p in (target_path, draft_path)]
    if configs[0]["vocab_size"] != configs[1]["vocab_size"]:
        raise ValueError("Target and draft output vocabularies differ")
    return {
        "vocab_size": configs[0]["vocab_size"],
        "tokenizer_sha256": tokenizer_fingerprint(target_path),
        "eos_token_ids": a.eos_token_id,
        "special_token_ids": a.all_special_ids,
    }


def validate_initial_budget(profile, memory_limit_bytes=None, speculative=False):
    """Reject impossible explicit budgets before loading any checkpoint arrays."""
    spec = get_profile(profile)
    required = spec["weight_bytes"] + (2 * 1024**3 if spec["draft"] is None else 512 * 1024**2)
    if speculative:
        if spec["draft"] is None:
            raise ValueError("No verified compatible draft for this profile")
        required += 335450584
    if memory_limit_bytes is not None and required > memory_limit_bytes:
        raise MemoryError(
            f"Pinned weights and runtime reserve need at least {required} bytes; explicit budget is {memory_limit_bytes}"
        )
    return required
