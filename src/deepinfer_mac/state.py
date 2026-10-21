"""Portable request and result types; importing these never loads Metal."""

from __future__ import annotations

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

    def __post_init__(self):
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
