"""Portable request and result types; importing these never loads Metal."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

MODES = ("baseline", "fixed_speculative", "adaptive")


@dataclass
class GenerationRequest:
    prompt: str = ""
    messages: list[dict] | None = None
    profile: str = "qwen3-4b"
    mode: str = "baseline"
    max_tokens: int = 128
    num_draft_tokens: int = 3
    enable_thinking: bool = False
    use_cache: bool = True
    prefill_step_size: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    presence_penalty: float = 0.0
    seed: int | None = None
    fallback_to_baseline: bool = False

    def __post_init__(self):
        if not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be finite and between0 and2")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be finite and in(0,1]")
        if not isinstance(self.top_k, int) or isinstance(self.top_k, bool) or not 0 <= self.top_k <= 1_000_000:
            raise ValueError("top_k must be an integer between0 and1000000")
        if not math.isfinite(self.presence_penalty) or not -2 <= self.presence_penalty <= 2:
            raise ValueError("presence_penalty must be finite and between-2 and2")
        if self.seed is not None and (not isinstance(self.seed, int) or isinstance(self.seed, bool) or not 0 <= self.seed <= 2**32-1):
            raise ValueError("seed must be an unsigned32-bit integer")
        if self.mode in ("fixed_speculative", "adaptive") and (
            self.temperature != 0 or self.top_p != 1 or self.top_k != 0 or self.presence_penalty != 0
        ):
            raise ValueError("Speculative modes currently support only greedy sampling with no penalties")
        if self.mode not in (*MODES, "native_mlx"):
            raise ValueError(f"Unsupported mode: {self.mode}; choose {MODES}")
        if not 1 <= self.max_tokens <= 4096:
            raise ValueError("max_tokens must be between 1 and 4096")
        if not 1 <= self.num_draft_tokens <= 8:
            raise ValueError("num_draft_tokens must be between 1 and 8")
        if self.prefill_step_size < 1:
            raise ValueError("prefill_step_size must be positive")
        if not self.prompt and not self.messages:
            raise ValueError("Provide a prompt or messages")


@dataclass
class GenerationResult:
    run_id: str
    mode: str
    profile: str
    text: str
    token_ids: list[int]
    prompt_tokens: int
    finish_reason: str
    metrics: dict
    rounds: list[dict] = field(default_factory=list)
    model: dict = field(default_factory=dict)
