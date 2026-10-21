"""Single-owner real MLX inference with exact prefix reuse and greedy decoding."""

from __future__ import annotations

import gc
import hashlib
import json
import threading
import time
import uuid

from .cache import PromptCacheStore, cache_bytes
from .models import download_models, get_profile, validate_initial_budget, verify_pair
from .policy import AdaptivePolicy
from .state import GenerationResult


def finalize_segment(detokenizer):
    """Read a consuming last_segment property exactly once after finalization."""
    detokenizer.finalize()
    return detokenizer.last_segment


def read_swap_bytes():
    import psutil

    try:
        return psutil.swap_memory().used
    except (OSError, RuntimeError):
        return None


class InferenceEngine:
    def __init__(self, cache_budget_bytes=512 * 1024**2, memory_limit_bytes=None):
        self.cache = PromptCacheStore(cache_budget_bytes)
        self.memory_limit_bytes = memory_limit_bytes
        self.model = self.draft_model = self.tokenizer = None
        self.profile = None
        self.paths = {}
        self.lock = threading.Lock()
        self.policies = {}
        self.disabled_draft = set()

    def close(self):
        import mlx.core as mx

        mx.synchronize()
        self.clear_cache()
        self.model = self.draft_model = self.tokenizer = None
        self.profile = None
        gc.collect()
        mx.clear_cache()

    def clear_cache(self):
        self.cache.clear()

    def unload_draft(self):
        import mlx.core as mx

        mx.synchronize()
        self.draft_model = None
        # Combined target/draft snapshots retain draft arrays, so drop them too.
        self.clear_cache()
        gc.collect()
        mx.clear_cache()

    def _load(self, profile, speculative):
        import mlx.core as mx
        import psutil
        from mlx_lm import load

        spec = get_profile(profile)
        validate_initial_budget(profile, self.memory_limit_bytes, speculative)
        if self.memory_limit_bytes is not None:
            # MLX documents this as an allocator guideline, not a hard cap.
            mx.set_memory_limit(self.memory_limit_bytes)
        if self.profile != profile:
            self.clear_cache()
            self.model = self.draft_model = self.tokenizer = None
            self.profile = profile
            gc.collect()
            mx.clear_cache()
        if self.model is None:
            if psutil.virtual_memory().available < spec["minimum_memory_bytes"]:
                raise MemoryError(
                    "Insufficient available unified memory for this profile; close apps or use qwen3-0.6b"
                )
            self.paths = download_models(profile, include_draft=speculative)
            self.model, self.tokenizer = load(self.paths["target"], trust_remote_code=False)
            mx.eval(self.model.parameters())
        if speculative and self.draft_model is None:
            self.paths.update(download_models(profile, include_draft=True))
            verify_pair(self.paths["target"], self.paths["draft"])
            self.draft_model, _ = load(self.paths["draft"], trust_remote_code=False)
            mx.eval(self.draft_model.parameters())

    def _tokens(self, request):
        messages = request.messages or [{"role": "user", "content": request.prompt}]
        messages = [m if isinstance(m, dict) else m.model_dump() for m in messages]
        return list(
            self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=request.enable_thinking
            )
        )

    def _identity(self, request, speculative):
        spec = get_profile(request.profile)
        data = {
            "target": spec["target"],
            "draft": spec["draft"] if speculative else None,
            "template": self.tokenizer.chat_template,
            "thinking": request.enable_thinking,
            "quantization": "4bit-group64",
            "kv": "exact",
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def generate(self, request, emit=None, cancel=None):
        with self.lock:
            return self._generate(request, emit, cancel)

    def native_generate(self, request, emit=None, cancel=None):
        import dataclasses

        return self.generate(dataclasses.replace(request, mode="native_mlx", use_cache=False), emit, cancel)

    def _generate(self, request, emit, cancel):
        import mlx.core as mx
        import psutil
        from mlx_lm.generate import generate_step, speculative_generate_step, stream_generate
        from mlx_lm.models.cache import make_prompt_cache

        from .speculative import adaptive_generate_step

        start = time.perf_counter()
        spec = get_profile(request.profile)
        supported_modes = spec.get("supported_modes", ["baseline", "fixed_speculative", "adaptive"])
        if request.mode != "native_mlx" and request.mode not in supported_modes:
            raise ValueError(
                f"{request.profile} supports only {supported_modes}; compatible draft/MTP unavailable"
            )
        swap_before = read_swap_bytes()
        if cancel is not None and cancel.is_set():
            raise InterruptedError("Request cancelled")
        load_start = time.perf_counter()
        self._load(request.profile, False)
        tokens = self._tokens(request)
        policy_key = (request.profile, request.enable_thinking, len(tokens) // 256)
        policy = self.policies.setdefault(policy_key, AdaptivePolicy())
        fallback = request.mode == "adaptive" and policy_key in self.disabled_draft
        speculative = request.mode == "fixed_speculative" or (request.mode == "adaptive" and not fallback)
        if speculative:
            self._load(request.profile, True)
        load_ms = (time.perf_counter() - load_start) * 1000
        if len(tokens) + request.max_tokens > spec.get("context_limit", 40960):
            raise ValueError(
                "Prompt plus output exceeds the profile resource context limit; input is never truncated"
            )
        # Conservatively budget active exact KV as well as resident snapshots.
        config_path = self.paths["target"]
        from pathlib import Path

        cfg = json.loads((Path(config_path) / "config.json").read_text())
        text_cfg = cfg.get("text_config", cfg)
        if cfg.get("model_type") == "qwen3_5":
            # Only full-attention layers keep growing KV; recurrent layers have
            # bounded state. Reserve two GiB for recurrent and temporary state.
            cfg = dict(text_cfg)
            cfg["num_hidden_layers"] = sum(t == "full_attention" for t in text_cfg["layer_types"])
        kv_estimate = (
            (len(tokens) + request.max_tokens + 512)
            * cfg["num_hidden_layers"]
            * cfg["num_key_value_heads"]
            * cfg["head_dim"]
            * 4
        )
        if speculative:
            draft_cfg = json.loads((Path(self.paths["draft"]) / "config.json").read_text())
            kv_estimate += (
                (len(tokens) + request.max_tokens + 512)
                * draft_cfg["num_hidden_layers"]
                * draft_cfg["num_key_value_heads"]
                * draft_cfg["head_dim"]
                * 4
            )
        if text_cfg.get("model_type") == "qwen3_5_text":
            kv_estimate += 2 * 1024**3
        if (
            self.memory_limit_bytes is not None
            and mx.get_active_memory() + self.cache.nbytes + kv_estimate > self.memory_limit_bytes
        ):
            self.clear_cache()
            if mx.get_active_memory() + kv_estimate > self.memory_limit_bytes:
                raise MemoryError("Model weights plus estimated active KV exceed the explicit memory budget")
        if kv_estimate > psutil.virtual_memory().available * 0.65:
            self.clear_cache()
            mx.clear_cache()
            if kv_estimate > psutil.virtual_memory().available * 0.65:
                raise MemoryError(
                    f"Estimated exact active KV requires {kv_estimate} bytes; reduce context or max_tokens"
                )
        mx.reset_peak_memory()
        identity = self._identity(request, speculative)
        caches, hit = self.cache.get(identity, tokens) if request.use_cache else (None, 0)
        if caches is None:
            caches = make_prompt_cache(self.model)
            if speculative:
                caches += make_prompt_cache(self.draft_model)
        target_layers = len(self.model.layers) if speculative else len(caches)
        target_cache = caches[:target_layers]
        draft_cache = caches[target_layers:]
        rounds = []
        prefill_begin = time.perf_counter()
        run_id = uuid.uuid4().hex
        ids = []
        text_parts = []
        first_at = None
        finish = "length"
        accepted = 0
        generator = None
        native = request.mode == "native_mlx"
        try:
            if native:
                generator = stream_generate(
                    self.model,
                    self.tokenizer,
                    tokens,
                    max_tokens=request.max_tokens,
                    prefill_step_size=request.prefill_step_size,
                    prompt_cache=target_cache,
                )
                prefill_ms = None
            else:
                # Snapshot only prompt[: -1], before any speculative mutation.
                position = hit
                while position < len(tokens) - 1:
                    if cancel is not None and cancel.is_set():
                        raise InterruptedError("Request cancelled")
                    end = min(len(tokens) - 1, position + request.prefill_step_size)
                    chunk = mx.array(tokens[position:end], mx.uint32)[None]
                    self.model(chunk, cache=target_cache)
                    if speculative:
                        self.draft_model(chunk, cache=draft_cache)
                    mx.eval([c.state for c in caches])
                    position = end
                    # Keep one reusable chunk boundary as well as the complete
                    # prompt, so changed suffixes and multi-turn branches can
                    # reuse a long system prefix without arbitrary cache trim.
                    if request.use_cache and hit == 0 and position == request.prefill_step_size:
                        self.cache.put(identity, tokens[:position], caches)
                prefill_ms = (time.perf_counter() - prefill_begin) * 1000
                if request.use_cache:
                    self.cache.put(identity, tokens[:-1], caches)
                tail = mx.array(tokens[-1:], mx.uint32)
                if request.mode == "adaptive" and speculative:
                    generator = adaptive_generate_step(
                        tokens[-1:],
                        self.model,
                        self.draft_model,
                        target_cache,
                        draft_cache,
                        policy,
                        request.max_tokens,
                        rounds,
                        cancel,
                        memory_pressure=lambda: psutil.virtual_memory().available < 2 * 1024**3,
                    )
                elif request.mode == "fixed_speculative":
                    generator = speculative_generate_step(
                        tail,
                        self.model,
                        self.draft_model,
                        num_draft_tokens=request.num_draft_tokens,
                        max_tokens=request.max_tokens,
                        prompt_cache=caches,
                        prefill_step_size=request.prefill_step_size,
                    )
                else:
                    generator = (
                        (int(token), False)
                        for token, _ in generate_step(
                            tail,
                            self.model,
                            max_tokens=request.max_tokens,
                            prompt_cache=target_cache,
                            prefill_step_size=request.prefill_step_size,
                        )
                    )
            detokenizer = self.tokenizer.detokenizer
            detokenizer.reset()
            fixed_cycle = None
            for output in generator:
                if cancel is not None and cancel.is_set():
                    raise InterruptedError("Request cancelled")
                if native:
                    token, from_draft = int(output.token), False
                elif request.mode == "fixed_speculative":
                    token, _, from_draft = output
                    token = int(token)
                else:
                    token, from_draft = output
                now = time.perf_counter()
                if request.mode == "fixed_speculative":
                    if fixed_cycle is None:
                        fixed_cycle = {
                            "budget": min(request.num_draft_tokens, request.max_tokens - len(ids)),
                            "accepted": 0,
                            "produced": 0,
                            "fallback": False,
                        }
                        rounds.append(fixed_cycle)
                    fixed_cycle["accepted"] += int(from_draft)
                    fixed_cycle["produced"] += 1
                    if not from_draft:
                        fixed_cycle = None
                if first_at is None:
                    first_at = now
                # Upstream final EOS response can contain the detokenizer's
                # pending final segment, so emit its text before checking EOS.
                ids.append(token)
                accepted += int(from_draft)
                if native:
                    segment = output.text
                    text_parts.append(segment)
                    if emit and segment:
                        emit({"type": "token", "text": segment, "token": token, "run_id": run_id})
                    if token in self.tokenizer.eos_token_ids:
                        finish = "stop"
                        break
                    continue
                if token in self.tokenizer.eos_token_ids:
                    finish = "stop"
                    break
                else:
                    detokenizer.add_token(token)
                    segment = detokenizer.last_segment
                text_parts.append(segment)
                if emit:
                    emit({"type": "token", "text": segment, "token": token, "run_id": run_id})
            if not native:
                segment = finalize_segment(detokenizer)
                if segment:
                    text_parts.append(segment)
                    if emit:
                        emit({"type": "token", "text": segment, "run_id": run_id})
        finally:
            if generator is not None and hasattr(generator, "close"):
                generator.close()
            mx.synchronize()
        end = time.perf_counter()
        total_ms = (end - start) * 1000
        decode_ms = (end - first_at) * 1000 if first_at else 0
        if fallback:
            rounds.append(
                {
                    "budget": 0,
                    "accepted": 0,
                    "produced": len(ids),
                    "draft_ms": 0,
                    "verify_ms": decode_ms,
                    "fallback": True,
                }
            )
        drafted = sum(r["budget"] for r in rounds)
        swap_after = read_swap_bytes()
        metrics = {
            "load_ms": load_ms,
            "ttft_ms": (first_at - start) * 1000 if first_at else None,
            "total_ms": total_ms,
            "prefill_ms": prefill_ms,
            "prompt_tps": ((len(tokens) - hit - 1) / (prefill_ms / 1000)) if prefill_ms else None,
            "decode_tps": max(len(ids) - 1, 0) / (decode_ms / 1000) if decode_ms else 0,
            "prompt_tokens": len(tokens),
            "generation_tokens": len(ids),
            "peak_memory_gb": mx.get_peak_memory() / 1e9,
            "peak_memory_bytes": mx.get_peak_memory(),
            "active_memory_bytes": mx.get_active_memory(),
            "rss_bytes": psutil.Process().memory_info().rss,
            "swap_before_bytes": swap_before,
            "swap_after_bytes": swap_after,
            "swap_delta_bytes": swap_after - swap_before
            if swap_after is not None and swap_before is not None
            else None,
            "swap_measurement_available": swap_before is not None and swap_after is not None,
            "estimated_active_kv_bytes": kv_estimate,
            "cache_hit_tokens": hit,
            "cache_bytes": self.cache.nbytes,
            "active_kv_bytes": cache_bytes(caches),
            "cache_evictions": self.cache.evictions,
            "draft_acceptance": accepted / drafted if drafted else None,
            "fallback_rate": sum(r["fallback"] for r in rounds) / len(rounds) if rounds else 0,
            "cycles": len(rounds),
            "policy": policy.summary() if request.mode == "adaptive" else None,
            "draft_accepted_tokens": accepted,
            "calibrated_fallback": fallback,
        }
        if (
            request.mode == "adaptive"
            and speculative
            and all(len(policy.samples[k]) >= 2 for k in policy.candidates)
            and policy.choose(request.max_tokens) == 0
        ):
            self.disabled_draft.add(policy_key)
            self.unload_draft()
            metrics["draft_released"] = True
        return GenerationResult(
            run_id,
            request.mode,
            request.profile,
            "".join(text_parts),
            ids,
            len(tokens),
            finish,
            metrics,
            rounds,
            get_profile(request.profile),
        )
