from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Mode = Literal["baseline", "fixed_speculative", "adaptive"]


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=100_000)


class ChatRequest(BaseModel):
    """Supported subset: text messages, greedy, n=1, EOS/length, optional SSE."""
    model_config = ConfigDict(extra="forbid")
    model: str = "qwen3-4b"
    messages: list[Message] = Field(min_length=1, max_length=64)
    max_tokens: int = Field(default=128, ge=1, le=1024)
    temperature: Literal[0] = 0
    stream: bool = False
    mode: Mode = "baseline"
    stop: str | list[str] | None = None
    n: Literal[1] = 1

    @model_validator(mode="after")
    def validate_stop(self):
        if self.stop is not None:
            raise ValueError("Custom stop strings are not supported; model EOS and max_tokens are supported.")
        return self


class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=100_000)
    max_tokens: int = Field(default=128, ge=1, le=1024)
    profile: str = "qwen3-4b"
