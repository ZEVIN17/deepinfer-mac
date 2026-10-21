from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Mode = Literal["baseline", "fixed_speculative", "adaptive"]


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=100_000)


class ChatRequest(BaseModel):
    """Text n=1 sampling; speculative modes remain greedy; optional SSE."""
    model_config = ConfigDict(extra="forbid")
    model: str = "qwen3-4b"
    messages: list[Message] = Field(min_length=1, max_length=64)
    max_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0, ge=0, le=2, allow_inf_nan=False)
    top_p: float = Field(default=1, gt=0, le=1, allow_inf_nan=False)
    top_k: int = Field(default=0, ge=0, le=1_000_000, strict=True)
    presence_penalty: float = Field(default=0, ge=-2, le=2, allow_inf_nan=False)
    seed: int | None = Field(default=None, ge=0, le=2**32-1, strict=True)
    enable_thinking: bool = False
    use_cache: bool = True
    fallback_to_baseline: bool = False
    stream: bool = False
    mode: Mode = "baseline"
    priority: Literal["interactive", "background"] = "interactive"
    stop: str | list[str] | None = None
    n: Literal[1] = 1

    @model_validator(mode="after")
    def validate_stop(self):
        if self.mode in ("fixed_speculative", "adaptive") and (
            self.temperature != 0 or self.top_p != 1 or self.top_k != 0 or self.presence_penalty != 0
        ):
            raise ValueError("Speculative modes currently support only greedy sampling with no penalties")
        if self.stop is not None:
            raise ValueError("Custom stop strings are not supported; model EOS and max_tokens are supported.")
        return self


class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=100_000)
    max_tokens: int = Field(default=128, ge=1, le=4096)
    profile: str = "qwen3-4b"
