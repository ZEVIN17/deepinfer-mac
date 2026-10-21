"""Single-owner real MLX inference with exact prefix reuse and explicit sampling."""

from __future__ import annotations

import gc
import hashlib
import json
import threading
import time
import uuid
from dataclasses import asdict

from .cache import NativePromptCacheStore, PromptCacheStore, cache_bytes
from .models import download_models, get_profile, validate_initial_budget, verify_pair
from .policy import AdaptivePolicy, DSAdaptivePolicy, FixedPolicy
from .sampling import generation_sampling, sampling_parameters
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


def attention_config(config):
    """Resolve actual attention KV geometry for flat or hybrid text metadata."""
    text = dict(config.get("text_config", config))
    if text.get("model_type") == "qwen3_5_text" or config.get("model_type") == "qwen3_5":
        if "layer_types" in text:
            text["num_hidden_layers"] = sum(kind == "full_attention" for kind in text["layer_types"])
        else:
            text["num_hidden_layers"] //= text["full_attention_interval"]
    return text


class InferenceEngine:
    def __init__(
        self, cache_budget_bytes=512 * 1024**2, memory_limit_bytes=None, checkpoint_interval_tokens=0,
        adaptive_scheduler="legacy", draft_retry_seconds=60, round_diagnostics=True,
        cache_backend="native",
        experimental_hybrid_speculation=False,
        message_checkpoints=False,
        hybrid_cost_adapter="original",
    ):
        if checkpoint_interval_tokens < 0:
            raise ValueError("checkpoint_interval_tokens must be nonnegative")
        # Experimental, opt-in: snapshot existing prefill boundaries without
        # changing computation shapes or claiming arbitrary recurrent rollback.
        self.checkpoint_interval_tokens = checkpoint_interval_tokens
        if adaptive_scheduler not in ("legacy", "ds"):
            raise ValueError("adaptive_scheduler must be legacy or ds")
        self.adaptive_scheduler = adaptive_scheduler
        self.draft_retry_seconds = draft_retry_seconds
        self.round_diagnostics = round_diagnostics
        if cache_backend not in ("native", "exact"):
            raise ValueError("cache_backend must be native or exact")
        self.cache_backend = cache_backend
        if message_checkpoints and cache_backend != "native":
            raise ValueError("message_checkpoints requires the native cache backend")
        # Explicit experiment: cold prefill shapes may change. Do not enable in
        # comparative benchmarks before real-model sampling/cache validation.
        self.message_checkpoints = message_checkpoints
        self.experimental_hybrid_speculation = experimental_hybrid_speculation
        if hybrid_cost_adapter not in ("original", "lagged_mean"):
            raise ValueError("hybrid_cost_adapter must be original or lagged_mean")
        if hybrid_cost_adapter != "original" and adaptive_scheduler != "ds":
            raise ValueError("Hybrid recovery cost adapter requires the DS scheduler")
        self.hybrid_cost_adapter = hybrid_cost_adapter
        store_type = NativePromptCacheStore if cache_backend == "native" else PromptCacheStore
        self.cache = store_type(cache_budget_bytes)
        self.memory_limit_bytes = memory_limit_bytes
        self.model = self.draft_model = self.tokenizer = None
        self.profile = None
        self.paths = {}
        self.lock = threading.Lock()
        self.policies = {}
        self.disabled_draft = {}

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
            "quantization": spec.get("quantization", "4-bit, group size 64"),
            "kv": "exact",
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def generate(self, request, emit=None, cancel=None):
        with self.lock:
            return self._generate(request, emit, cancel)

    def native_generate(self, request, emit=None, cancel=None):
        import dataclasses

        return self.generate(dataclasses.replace(request, mode="native_mlx", use_cache=False), emit, cancel)

    def _prefill_prompt(
        self, request, tokens, caches, target_cache, draft_cache, identity, hit,
        speculative, mx, cancel=None, prefill_begin=None,
    ):
        """Consume only real prompt[:-1] tokens, then clone exact snapshots.

        The disabled/uncached path retains the original chunk calculation and
        writes. Optional message splits use verified full-request token offsets;
        recurrent state is never trimmed to invent an intermediate checkpoint.
        """
        prefill_begin = prefill_begin if prefill_begin is not None else time.perf_counter()
        enabled = self.message_checkpoints and request.use_cache
        boundaries = {}
        checkpoint_metrics = None
        if enabled:
            from .session_boundaries import message_boundaries

            tokenize_begin = time.perf_counter()
            messages = request.messages or [{"role": "user", "content": request.prompt}]
            verified = message_boundaries(
                self.tokenizer, messages, tokens, enable_thinking=request.enable_thinking
            )
            # Retain verified roles for already-consumed prefixes too: a warm
            # hit at prompt[:-1] must not rewrite a user/system key as assistant.
            boundaries = {b.token_offset: b for b in verified if 0 < b.token_offset < len(tokens)}
            checkpoint_metrics = {
                "experimental": True,
                "requires_real_model_sampling_validation": True,
                "boundary_tokenize_ms": (time.perf_counter() - tokenize_begin) * 1000,
                "verified_boundaries": [asdict(b) for b in verified],
                "snapshots": [],
                "snapshot_host_ms": 0.0,
                "logical_snapshot_bytes_written": 0,
                "prefill_shapes_changed": any(
                    (offset - hit) % request.prefill_step_size != 0
                    for offset in boundaries if hit < offset < len(tokens) - 1
                ),
            }

        def snapshot(position, cache_type="assistant"):
            if checkpoint_metrics is None:
                self.cache.put(identity, tokens[:position], caches)
                return
            begin = time.perf_counter()
            logical_bytes = cache_bytes(caches)
            stored = self.cache.put(identity, tokens[:position], caches, cache_type=cache_type)
            checkpoint_metrics["snapshot_host_ms"] += (time.perf_counter() - begin) * 1000
            checkpoint_metrics["logical_snapshot_bytes_written"] += logical_bytes if stored else 0
            checkpoint_metrics["snapshots"].append(
                {"token_offset": position, "cache_type": cache_type,
                 "stored": bool(stored), "logical_bytes": logical_bytes}
            )

        position = hit
        while position < len(tokens) - 1:
            if cancel is not None and cancel.is_set():
                raise InterruptedError("Request cancelled")
            end = min(len(tokens) - 1, position + request.prefill_step_size)
            if enabled:
                # Preserve the original regular grid even when an earlier
                # message boundary splits a chunk into smaller real forwards.
                regular_end = hit + ((position - hit) // request.prefill_step_size + 1) * request.prefill_step_size
                next_boundary = min((offset for offset in boundaries if offset > position), default=end)
                end = min(end, regular_end, next_boundary)
            chunk = mx.array(tokens[position:end], mx.uint32)[None]
            self.model(chunk, cache=target_cache)
            if speculative:
                self.draft_model(chunk, cache=draft_cache)
            mx.eval([c.state for c in caches])
            position = end
            first_boundary = hit == 0 and position == request.prefill_step_size
            periodic_boundary = (
                self.checkpoint_interval_tokens > 0
                and position % self.checkpoint_interval_tokens == 0
            )
            message_boundary = boundaries.get(position)
            if request.use_cache and (first_boundary or periodic_boundary or message_boundary is not None):
                snapshot(position, message_boundary.cache_type if message_boundary is not None else "assistant")
        prefill_ms = (time.perf_counter() - prefill_begin) * 1000
        if request.use_cache:
            # Retain historical default writes. In the opt-in path a boundary
            # already at prompt[:-1] is the same consumed state, not a second
            # snapshot with a different eviction priority.
            already_written = checkpoint_metrics is not None and any(
                s["token_offset"] == len(tokens) - 1 for s in checkpoint_metrics["snapshots"]
            )
            if not already_written:
                final_boundary = boundaries.get(len(tokens) - 1)
                snapshot(len(tokens) - 1, final_boundary.cache_type if final_boundary is not None else "assistant")
        return prefill_ms, checkpoint_metrics

    def _generate(self, request, emit, cancel):
        import mlx.core as mx
        import psutil
        from mlx_lm.generate import generate_step, speculative_generate_step, stream_generate
        from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache

        from .speculative import adaptive_generate_step

        start = time.perf_counter()
        cache_timings_before = asdict(self.cache.timings) if hasattr(self.cache, "timings") else {}
        spec = get_profile(request.profile)
        supported_modes = spec.get("supported_modes", ["baseline", "fixed_speculative", "adaptive"])
        experimental_hybrid = (
            self.experimental_hybrid_speculation and spec.get("experimental_speculation", False)
            and request.mode in ("adaptive", "fixed_speculative")
        )
        if request.mode != "native_mlx" and request.mode not in supported_modes and not experimental_hybrid:
            raise ValueError(
                f"{request.profile} supports only {supported_modes}; compatible draft/MTP unavailable"
            )
        swap_before = read_swap_bytes()
        if cancel is not None and cancel.is_set():
            raise InterruptedError("Request cancelled")
        load_start = time.perf_counter()
        self._load(request.profile, False)
        tokens = self._tokens(request)
        hybrid_cost_applicable = (
            request.mode == "adaptive" and spec.get("hybrid_cache", False)
            and self.adaptive_scheduler == "ds"
        )
        effective_cost_adapter = self.hybrid_cost_adapter if hybrid_cost_applicable else "original"
        policy_key = (request.profile, request.enable_thinking, len(tokens) // 256,
                      self.adaptive_scheduler, effective_cost_adapter)
        policy_type = DSAdaptivePolicy if self.adaptive_scheduler == "ds" else AdaptivePolicy
        if effective_cost_adapter == "lagged_mean":
            from .ds.recovery_cost import LaggedMeanRecoveryPolicy

            policy_type = LaggedMeanRecoveryPolicy
        if policy_key not in self.policies:
            self.policies[policy_key] = policy_type()
        policy = self.policies[policy_key]
        disabled_until = self.disabled_draft.get(policy_key, 0)
        if disabled_until and time.monotonic() >= disabled_until:
            self.disabled_draft.pop(policy_key)
            # A fresh workload can recalibrate instead of permanently inheriting
            # a short request's target-only decision.
            policy = self.policies[policy_key] = policy_type()
        fallback = request.mode == "adaptive" and policy_key in self.disabled_draft
        speculative = request.mode == "fixed_speculative" or (request.mode == "adaptive" and not fallback)
        if speculative:
            self._load(request.profile, True)
        # Model initialization may consume random state; seed only after all loads.
        if request.seed is not None:
            mx.random.seed(request.seed)
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
        if request.top_k and request.top_k >= text_cfg["vocab_size"]:
            raise ValueError("top_k must be smaller than the model vocabulary")
        cfg = attention_config(cfg)
        kv_estimate = (
            (len(tokens) + request.max_tokens + 512)
            * cfg["num_hidden_layers"]
            * cfg["num_key_value_heads"]
            * cfg["head_dim"]
            * 4
        )
        if speculative:
            draft_cfg = attention_config(json.loads((Path(self.paths["draft"]) / "config.json").read_text()))
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
            # Retained snapshot arrays already belong to MLX active memory.
            # Cache nbytes is a logical admission limit (snapshots may share
            # immutable storage), not additional physical allocator usage.
            # KV here is a conservative future reserve, not a hard peak bound.
            and mx.get_active_memory() + kv_estimate > self.memory_limit_bytes
        ):
            self.clear_cache()
            if mx.get_active_memory() + kv_estimate > self.memory_limit_bytes:
                raise MemoryError("Model weights plus estimated active KV exceed the explicit memory budget")
        if (
            not spec.get("advisory_memory_estimates", False)
            and kv_estimate > psutil.virtual_memory().available * 0.65
        ):
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
        hybrid_speculation = speculative and not can_trim_prompt_cache(caches)
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
        message_checkpoint_metrics = None
        try:
            if native:
                generator = stream_generate(
                    self.model,
                    self.tokenizer,
                    tokens[hit:],
                    **generation_sampling(request, tokens[:hit]),
                    max_tokens=request.max_tokens,
                    prefill_step_size=request.prefill_step_size,
                    prompt_cache=target_cache,
                )
                prefill_ms = None
            else:
                prefill_ms, message_checkpoint_metrics = self._prefill_prompt(
                    request, tokens, caches, target_cache, draft_cache, identity, hit,
                    speculative, mx, cancel, prefill_begin,
                )
                tail = mx.array(tokens[-1:], mx.uint32)
                if hybrid_speculation:
                    from .hybrid_speculative import hybrid_generate_step

                    hybrid_policy = policy if request.mode == "adaptive" else FixedPolicy(request.num_draft_tokens)
                    generator = hybrid_generate_step(
                        tokens[-1:], self.model, self.draft_model, target_cache, draft_cache,
                        hybrid_policy, request.max_tokens, rounds, cancel,
                        memory_pressure=lambda: psutil.virtual_memory().available < 2 * 1024**3,
                        diagnostics=self.round_diagnostics,
                    )
                elif request.mode == "adaptive" and speculative:
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
                        defer_draft_sync=True,
                        diagnostics=self.round_diagnostics,
                    )
                elif request.mode == "fixed_speculative" and not hybrid_speculation:
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
                            **generation_sampling(request, tokens[:-1]),
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
                elif request.mode == "fixed_speculative" and not hybrid_speculation:
                    token, _, from_draft = output
                    token = int(token)
                else:
                    token, from_draft = output
                now = time.perf_counter()
                if request.mode == "fixed_speculative" and not hybrid_speculation:
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
            "sampling": sampling_parameters(request),
            "enable_thinking": request.enable_thinking,
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
            "cache_backend": self.cache_backend,
            "message_checkpoints": self.message_checkpoints,
            "message_checkpointing": message_checkpoint_metrics,
            "cache_operations": {
                key: value - cache_timings_before[key]
                for key, value in asdict(self.cache.timings).items()
            } if hasattr(self.cache, "timings") else None,
            "draft_acceptance": accepted / drafted if drafted else None,
            "fallback_rate": sum(r["fallback"] for r in rounds) / len(rounds) if rounds else 0,
            "cycles": len(rounds),
            "policy": policy.summary() if request.mode == "adaptive" else None,
            "draft_accepted_tokens": accepted,
            "calibrated_fallback": fallback,
            "adaptive_scheduler": self.adaptive_scheduler if request.mode == "adaptive" else None,
            "hybrid_cost_adapter_requested": self.hybrid_cost_adapter,
            "hybrid_cost_adapter_applicable": bool(hybrid_cost_applicable),
            "hybrid_cost_adapter_effective": effective_cost_adapter if request.mode == "adaptive" else None,
            "round_diagnostics": self.round_diagnostics,
            "speculative_recovery": "hybrid exact snapshot + consumed-prefix replay" if hybrid_speculation else None,
        }
        if (
            request.mode == "adaptive"
            and speculative
            and self.adaptive_scheduler == "legacy"
            and all(len(policy.samples[k]) >= 2 for k in policy.candidates)
            and policy.choose(request.max_tokens) == 0
        ):
            self.disabled_draft[policy_key] = time.monotonic() + self.draft_retry_seconds
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
