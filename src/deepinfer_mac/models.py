"""Pinned public model registry. No remote code is trusted."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

TARGET = {"repo": "mlx-community/Qwen3-4B-4bit", "revision": "4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25"}
DRAFT = {"repo": "mlx-community/Qwen3-0.6B-4bit", "revision": "73e3e38d981303bc594367cd910ea6eb48349da8"}
DEMO_27B = {"repo": "mlx-community/Qwen3.8-27B-4bit", "revision": "10c35caafbb80f7dc6a7a432cdd11af10a6d4818"}
HYBRID_DRAFT = {"repo": "mlx-community/Qwen3.5-0.8B-4bit", "revision": "da28692b5f139cb0ec58a356b437486b7dac7462"}
DEMO_27B_2BIT = {
    "repo": "keXjos/Qwen3.8-27B-mlx-2Bit",
    "revision": "39d5487ad4442961b64098e15aa11ce05e6f9976",
}
PROFILES = {
    "qwen3-4b": {
        "target": TARGET,
        "draft": DRAFT,
        "draft_weight_bytes": 335450584,
        "minimum_memory_bytes": int(4.5 * 1024**3),
        "weight_bytes": 2263022529,
    },
    "qwen3-0.6b": {
        "target": DRAFT,
        "draft": DRAFT,
        "draft_weight_bytes": 335450584,
        "minimum_memory_bytes": int(1.5 * 1024**3),
        "weight_bytes": 335450584,
    },
    "qwen3.5-9b-8bit": {
        "target": {"repo": "mlx-community/Qwen3.5-9B-8bit", "revision": "16daa4818c54ce5f5436f929d52542eb65bbed9d"},
        "draft": HYBRID_DRAFT, "draft_weight_bytes": 625229487,
        "vocab_size": 248320, "experimental_speculation": True, "hybrid_cache": True, "runtime_reserve_bytes": 2 * 1024**3, "minimum_memory_bytes": 0, "advisory_memory_estimates": True,
        "weight_bytes": 10426592423, "text_tensor_bytes": 9514412544, "quantization": "8-bit affine, group size 64",
        "context_limit": 8192, "supported_modes": ["baseline"], "status": "quality_review_required",
        "role": "primary_quality_candidate",
        "limitations": "Real text baseline/native/API/SSE/UI verified. Initial12-task smoke: greedy10/12, official non-thinking11/12, thinking9/12 at1024-token cap; Public27subset measured with finite-budget failures retained. Hybrid speculation has initial96token exact controls behind an experimental flag; public modes remain baseline, no formal speed claim.",
    },
    "qwen3.5-9b-4bit": {
        "target": {"repo": "mlx-community/Qwen3.5-9B-4bit", "revision": "8b2b98c00a6b4d291155e4890773ca8f769aee53"},
        "draft": HYBRID_DRAFT, "draft_weight_bytes": 625229487,
        "vocab_size": 248320, "experimental_speculation": True, "hybrid_cache": True, "runtime_reserve_bytes": 2 * 1024**3, "minimum_memory_bytes": 0, "advisory_memory_estimates": True,
        "weight_bytes": 5950221072, "quantization": "4-bit affine, group size 64",
        "context_limit": 8192, "supported_modes": ["baseline"], "status": "quality_review_required",
        "role": "same_model_efficiency_control",
        "limitations": "Text baseline/public27shared-protocol comparison measured; local Chinese/thinking smoke pending. Hybrid speculation remains experimental and unverified for this quantization.",
    },
    "qwen3-14b": {
        "target": {"repo": "mlx-community/Qwen3-14B-4bit", "revision": "a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4"},
        "draft": DRAFT, "draft_weight_bytes": 335450584,
        "minimum_memory_bytes": 0, "advisory_memory_estimates": True,
        "weight_bytes": 8307898514, "quantization": "4-bit affine, group size 64",
        "context_limit": 8192, "supported_modes": ["baseline"], "status": "quality_review_required",
        "role": "engineering_fallback",
        "limitations": "Text baseline/public27shared comparison and12task native consistency measured; separate official public control pending. Draft pairing unvalidated; baseline only.",
    },
}


HISTORICAL_PROFILES = {
    "qwen3.8-27b-2bit": {
        "target": DEMO_27B_2BIT,
        "draft": None,
        "minimum_memory_bytes": 0,
        "advisory_memory_estimates": True,
        "weight_bytes": 8409705283,
        "text_tensor_bytes": 8409463808,
        "quantization": "2-bit affine, group size 64",
        "context_limit": 8192,
        "supported_modes": ["baseline"],
        "status": "quality_failed",
        "limitations": "Real load/cache verified; six-task quality smoke failed with greedy and official sampling settings. Experimental adaptation only. Hybrid speculative/MTP unsupported.",
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
    if name in HISTORICAL_PROFILES:
        raise ValueError(f"Profile {name!r} is retired; weights were removed. See docs/PHASE2_MODEL_SELECTION.md")
    if name not in PROFILES:
        raise ValueError(f"Unknown profile {name!r}; choose {list(PROFILES)}")
    return PROFILES[name]


def list_models():
    return [
        {
            "id": name,
            **profile,
            "quantization": profile.get("quantization", "4-bit, group size 64"),
            "context_limit": profile.get("context_limit", 40960),
            "vocab_size": profile.get("vocab_size", 151936),
            "supported_modes": profile.get("supported_modes", ["baseline", "fixed_speculative", "adaptive"]),
            "capabilities": {"cache": True, "speculative": bool(set(profile.get("supported_modes", ["baseline", "fixed_speculative", "adaptive"])) & {"fixed_speculative", "adaptive"}), "mtp": False},
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
            index_path = cached / "model.safetensors.index.json"
            weights = set(json.loads(index_path.read_text())["weight_map"].values()) if index_path.exists() else {"model.safetensors"}
            required = weights | {
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
    vocab_sizes = [cfg.get("text_config", cfg)["vocab_size"] for cfg in configs]
    if vocab_sizes[0] != vocab_sizes[1]:
        raise ValueError("Target and draft output vocabularies differ")
    fixtures = [{"role": "user", "content": "Tokenizer compatibility fixture."}]
    template_modes = {
        str(thinking).lower(): a.apply_chat_template(fixtures, tokenize=True, add_generation_prompt=True, enable_thinking=thinking)
        == b.apply_chat_template(fixtures, tokenize=True, add_generation_prompt=True, enable_thinking=thinking)
        for thinking in (False, True)
    }
    return {
        "explicit_thinking_template_tokens_equal": template_modes,
        "template_text_sha256": [hashlib.sha256((t.chat_template or "").encode()).hexdigest() for t in (a, b)],
        "vocab_size": vocab_sizes[0],
        "tokenizer_sha256": tokenizer_fingerprint(target_path),
        "eos_token_ids": a.eos_token_id,
        "special_token_ids": a.all_special_ids,
    }


def validate_initial_budget(profile, memory_limit_bytes=None, speculative=False):
    """Reject impossible explicit budgets before loading any checkpoint arrays."""
    spec = get_profile(profile)
    required = spec["weight_bytes"] + spec.get("runtime_reserve_bytes", 2 * 1024**3 if spec["draft"] is None else 512 * 1024**2)
    if speculative:
        if spec["draft"] is None:
            raise ValueError("No verified compatible draft for this profile")
        required += spec["draft_weight_bytes"]
    if memory_limit_bytes is not None and required > memory_limit_bytes:
        raise MemoryError(
            f"Pinned weights and runtime reserve need at least {required} bytes; explicit budget is {memory_limit_bytes}"
        )
    return required
