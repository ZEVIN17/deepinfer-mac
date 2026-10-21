"""Portable model contracts. These are DeepInfer interfaces, not MLX APIs.

Importing/registration reads metadata only; weights are never materialized here.
Unknown architectures keep MLX baseline compatibility and no optimized claims.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class ModelSpec:
    id: str
    source: str
    revision: str | None = None
    draft: str | None = None
    draft_revision: str | None = None
    context_limit: int = 4096
    runtime_reserve_bytes: int = 2 * 1024**3
    confidence_revision: str | None = None
    calibration_revision: str | None = None

    def __post_init__(self):
        if self.draft is None and self.draft_revision is not None:
            raise ValueError("draft_revision requires a draft source")
        if not self.id or not self.source:
            raise ValueError("ModelSpec requires a nonempty id and source")
        if not 1 <= self.context_limit <= 1_048_576 or self.runtime_reserve_bytes < 512 * 1024**2:
            raise ValueError("Set a positive context_limit and at least 512 MiB runtime reserve")
        for source, revision in ((self.source, self.revision), (self.draft, self.draft_revision)):
            if source is None:
                continue
            local = Path(source).expanduser().is_dir()
            if not local and not (re.fullmatch(r"[^/\s]+/[^/\s]+", source)
                                  and re.fullmatch(r"[0-9a-f]{40}", revision or "")):
                raise ValueError("Use an existing local directory or repo with an exact 40-character revision")
            if local and revision is not None:
                raise ValueError("Local models use content identity; omit revision")

    @classmethod
    def from_json(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def metadata(self):
        def ref(source, revision):
            path = Path(source).expanduser()
            return {"path": str(path.resolve())} if path.is_dir() else {"repo": source, "revision": revision}
        return {
            "id": self.id, "target": ref(self.source, self.revision),
            "draft": ref(self.draft, self.draft_revision) if self.draft else None,
            "context_limit": self.context_limit, "minimum_memory_bytes": 0,
            "runtime_reserve_bytes": self.runtime_reserve_bytes,
            "supported_modes": ["baseline"], "status": "baseline_compatibility_unverified",
            "quantization": "read from resolved config", "custom_model": True,
            "confidence_revision": self.confidence_revision, "calibration_revision": self.calibration_revision,
            "capabilities": asdict(ModelCapabilities()),
        }


@dataclass(frozen=True)
class ModelCapabilities:
    baseline: bool = True
    sampling: bool = True
    batch: bool = False
    trim: bool = False
    snapshot_replay: bool = False
    varlen_verify: bool = False
    confidence: bool = False
    speculative: bool = False
    mtp: bool = False
    cache: bool = True


@runtime_checkable
class DraftProvider(Protocol):
    """A draft consumes target token IDs, never a separately rendered template."""
    revision: str | None

    def verify(self, target_path: str, draft_path: str) -> dict: ...


@dataclass(frozen=True)
class ExternalDraftProvider:
    revision: str | None

    def verify(self, target_path, draft_path):
        from .models import verify_pair
        return verify_pair(target_path, draft_path)


class CacheAdapter:
    """Inspect concrete cache protocols independently of architecture names.

    Batch operations do not imply per-row speculative rollback. Snapshot/replay
    is limited to audited MLX KVCache/ArraysCache state setters.
    """
    audited_states = frozenset({"mlx_lm.models.cache.KVCache", "mlx_lm.models.cache.ArraysCache"})

    def capabilities(self, caches):
        def audited(cache):
            cls = type(cache)
            return f"{cls.__module__}.{cls.__name__}" in self.audited_states
        return {
            "trim": bool(caches) and all(callable(getattr(c, "is_trimmable", None)) and c.is_trimmable() for c in caches),
            "batch": bool(caches) and all(all(callable(getattr(c, op, None))
                                               for op in ("filter", "extend", "extract", "prepare")) for c in caches),
            "snapshot_replay": bool(caches) and all(audited(c) for c in caches),
            "varlen_verify": False,
        }

    def snapshot(self, caches):
        if not self.capabilities(caches)["snapshot_replay"]:
            raise ValueError("Cache snapshot/replay is not audited; use baseline")
        from .hybrid_speculative import snapshot_cache
        return snapshot_cache(caches)

    def restore(self, caches, snapshot):
        if not self.capabilities(caches)["snapshot_replay"]:
            raise ValueError("Cache snapshot/replay is not audited; use baseline")
        from .hybrid_speculative import restore_cache
        restore_cache(caches, snapshot)


def inspect_snapshot(path):
    """Validate local files and hash full content before model-array loading."""
    root = Path(path)
    config = json.loads((root / "config.json").read_text())
    if config.get("model_file"):
        raise ValueError("Custom model_file remote code is unsupported")
    text = config.get("text_config", config)
    if not isinstance(text.get("vocab_size"), int) or text["vocab_size"] <= 0:
        raise ValueError("config.json must declare a positive text vocabulary size")
    index = root / "model.safetensors.index.json"
    # Locked MLX-LM 0.32.2 loads glob("model*.safetensors"), even with an
    # index present. A stale unindexed file would otherwise bypass both the
    # weight budget and content identity, then still reach mx.load.
    files = sorted(p.name for p in root.glob("model*.safetensors"))
    if index.exists():
        indexed = set(json.loads(index.read_text())["weight_map"].values())
        if any(Path(name).name != name or not (root / name).is_file() for name in indexed):
            raise ValueError("Missing or unsafe model weight shard")
        if indexed != set(files):
            raise ValueError("Indexed weight shards must exactly match MLX-LM model*.safetensors files")
    if not files:
        raise ValueError("No model safetensors found in local snapshot")
    paths = []
    for name in files:
        file = root / name
        if Path(name).name != name or not file.is_file():
            raise ValueError("Missing or unsafe model weight shard")
        with file.open("rb") as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError("Invalid safetensors header")
            length = struct.unpack("<Q", raw)[0]
            if length > min(100 * 1024**2, file.stat().st_size - 8):
                raise ValueError("Invalid safetensors header length")
            header = json.loads(stream.read(length))
            for tensor_name, tensor in header.items():
                if tensor_name == "__metadata__":
                    continue
                if tensor_name.endswith(("lm_head.weight", "embed_tokens.weight")) and tensor["shape"][0] != text["vocab_size"]:
                    raise ValueError("Checkpoint output/embedding vocabulary disagrees with config")
                offsets = tensor["data_offsets"]
                if not 0 <= offsets[0] <= offsets[1] <= file.stat().st_size - 8 - length:
                    raise ValueError("Invalid safetensors tensor offsets")
        paths.append(file)
    tokenizer_assets = [root / name for name in ("tokenizer.json", "tokenizer.model", "vocab.json")]
    if not any(p.is_file() for p in tokenizer_assets) or not (root / "tokenizer_config.json").is_file():
        raise ValueError("Missing tokenizer assets/config; install a complete MLX-LM snapshot")
    if (root / "tokenizer.json").is_file():
        tokenizer = json.loads((root / "tokenizer.json").read_text())
        vocab = tokenizer.get("model", {}).get("vocab", {})
        ids = (list(vocab.values()) if isinstance(vocab, dict) else list(range(len(vocab))))
        ids.extend(t["id"] for t in tokenizer.get("added_tokens", []))
        if ids and (min(ids) < 0 or max(ids) >= text["vocab_size"]):
            raise ValueError("Tokenizer IDs exceed checkpoint output vocabulary")
    # Full weights belong to local content identity. Repo snapshots also gain an
    # observed digest, guarding local edits to a pinned HF cache snapshot.
    for pattern in ("*.json", "*.jinja", "*.model", "*.txt", "*.tiktoken"):
        paths.extend(root.glob(pattern))
    hashes = {}
    for file in sorted(set(paths)):
        hasher = hashlib.sha256()
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                hasher.update(chunk)
        hashes[file.name] = hasher.hexdigest()
    return {"config": config, "weight_bytes": sum((root / name).stat().st_size for name in files),
            "content_sha256": digest(hashes), "files": hashes}


class ModelAdapter:
    def __init__(self, metadata):
        self.metadata = metadata
        self.cache = CacheAdapter()
        self.draft = ExternalDraftProvider((metadata.get("draft") or {}).get("revision"))

    def tokenize(self, tokenizer, request):
        messages = request.messages or [{"role": "user", "content": request.prompt}]
        messages = [m if isinstance(m, dict) else m.model_dump() for m in messages]
        if not getattr(tokenizer, "chat_template", None):
            if request.enable_thinking or len(messages) != 1 or messages[0]["role"] != "user":
                raise ValueError("Model has no chat template; supply one user message or a plain prompt without thinking")
            return list(tokenizer.encode(messages[0]["content"]))
        return list(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                  enable_thinking=request.enable_thinking))

    def attention_config(self, config):
        text = dict(config.get("text_config", config))
        # Architecture-specific cache geometry stays in the adapter, never DS.
        if text.get("model_type") == "qwen3_5_text" or config.get("model_type") == "qwen3_5":
            text["num_hidden_layers"] = (sum(kind == "full_attention" for kind in text["layer_types"])
                                        if "layer_types" in text else text["num_hidden_layers"] // text["full_attention_interval"])
        if "head_dim" not in text and "hidden_size" in text and text.get("num_attention_heads"):
            text["head_dim"] = text["hidden_size"] // text["num_attention_heads"]
        if "num_key_value_heads" not in text and "num_attention_heads" in text:
            text["num_key_value_heads"] = text["num_attention_heads"]
        if not all(isinstance(text.get(key), int) and text[key] > 0
                   for key in ("num_hidden_layers", "num_key_value_heads", "head_dim")):
            raise ValueError("Unsupported cache geometry; supply an audited model adapter before loading weights")
        return text

    def memory_estimate(self, config, tokens):
        cfg = self.attention_config(config)
        result = tokens * cfg["num_hidden_layers"] * cfg["num_key_value_heads"] * cfg["head_dim"] * 4
        if config.get("text_config", config).get("model_type") == "qwen3_5_text":
            result += 2 * 1024**3
        return result

    def identity(self, tokenizer, request, speculative, resolved=None):
        return digest({"target": self.metadata["target"], "draft": self.metadata["draft"] if speculative else None,
                       "resolved": resolved, "template": getattr(tokenizer, "chat_template", None),
                       "thinking": request.enable_thinking, "quantization": self.metadata.get("quantization"),
                       "position": "native-model-position-exact-kv", "kv": "exact",
                       "confidence_revision": self.metadata.get("confidence_revision"),
                       "calibration_revision": self.metadata.get("calibration_revision")})

    def select_mode(self, requested, *, fallback=False, experimental=False):
        modes = self.metadata.get("supported_modes", ["baseline", "fixed_speculative", "adaptive"])
        if requested == "native_mlx" or requested in modes:
            return requested, None
        if experimental and self.metadata.get("experimental_speculation") and requested in ("fixed_speculative", "adaptive"):
            return requested, None
        reason = f"{requested} is not verified for {self.metadata.get('id', 'model')}; available modes: {modes}"
        if fallback:
            return "baseline", reason
        raise ValueError(reason + "; use baseline or explicit fallback_to_baseline")


class ModelRegistry:
    """Per-engine/service registration, with unchanged legacy presets."""
    def __init__(self, specs=()):
        self._custom = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec):
        from .models import HISTORICAL_PROFILES, PROFILES
        if not isinstance(spec, ModelSpec):
            raise TypeError("Register a ModelSpec")
        if spec.id in PROFILES or spec.id in HISTORICAL_PROFILES or spec.id in self._custom:
            raise ValueError(f"Duplicate/reserved model id: {spec.id}")
        metadata = spec.metadata()
        for kind in ("target", "draft"):
            ref = metadata.get(kind)
            if ref and "path" in ref:
                audit = inspect_snapshot(ref["path"])
                ModelAdapter(metadata).attention_config(audit["config"])
                metadata[f"{kind}_content_sha256"] = audit["content_sha256"]
                metadata["weight_bytes" if kind == "target" else "draft_weight_bytes"] = audit["weight_bytes"]
                if kind == "target":
                    metadata["vocab_size"] = audit["config"].get("text_config", audit["config"])["vocab_size"]
        self._custom[spec.id] = metadata
        # Local source context and quantization are available without loading.
        if "path" in metadata["target"]:
            text = json.loads((Path(metadata["target"]["path"]) / "config.json").read_text())
            text = text.get("text_config", text)
            maximum = text.get("max_position_embeddings")
            if isinstance(maximum, int) and maximum > 0:
                metadata["context_limit"] = min(metadata["context_limit"], maximum)
            metadata["quantization"] = text.get("quantization", "read from resolved config")
        return spec.id

    def get(self, name):
        if name in self._custom:
            return copy.deepcopy(self._custom[name])
        from .models import get_profile
        return get_profile(name)

    @staticmethod
    def public_metadata(metadata):
        result = copy.deepcopy(metadata)
        for kind in ("target", "draft"):
            if result.get(kind) and "path" in result[kind]:
                result[kind] = {"local": True, "content_sha256": result.get(f"{kind}_content_sha256")}
        return result

    def bind_snapshots(self, name, audits):
        metadata = self._custom[name]
        for kind, audit in audits.items():
            metadata[f"{kind}_content_sha256"] = audit["content_sha256"]
            metadata["weight_bytes" if kind == "target" else "draft_weight_bytes"] = audit["weight_bytes"]
            if kind == "target":
                text = audit["config"].get("text_config", audit["config"])
                metadata["vocab_size"] = text["vocab_size"]
                maximum = text.get("max_position_embeddings")
                if isinstance(maximum, int) and maximum > 0:
                    metadata["context_limit"] = min(metadata["context_limit"], maximum)
                metadata["quantization"] = audit["config"].get("quantization", text.get("quantization", "unquantized"))

    def list(self):
        from .models import list_models
        return [*list_models(), *(self.public_metadata(item) for item in self._custom.values())]

    def adapter(self, name):
        return ModelAdapter(self.get(name))
