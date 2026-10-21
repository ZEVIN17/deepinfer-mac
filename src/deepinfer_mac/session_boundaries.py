"""Find message checkpoints only when their actual token prefix is identical.

This is a generic MLX cache integration helper, not a DeepInfer mechanism.
Callers must evaluate and snapshot the cache at the returned consumed offset;
the helper never trims recurrent state or assumes templates are compositional.
"""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class MessageBoundary:
    token_offset: int
    message_count: int
    role: str

    @property
    def cache_type(self):
        return self.role if self.role in ("system", "user", "assistant") else "assistant"


def _token_list(value):
    # Transformers 5 can return BatchEncoding by default; MLX's wrapper returns
    # a flat list. Never mistake the number of mapping keys for a token count.
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise ValueError("Message boundary tokenization must be a single sequence")
        value = value[0]
    return tuple(map(int, value))


def message_boundaries(tokenizer, messages, full_tokens, *, enable_thinking=False):
    """Return strictly verified proper prefixes, including the final user turn.

    Templates may render earlier messages differently when a later turn is
    present. A candidate is accepted only after full token-prefix comparison.
    Keep the final input token unconsumed for the generation executor.
    """
    full_tokens = _token_list(full_tokens)
    messages = [m if isinstance(m, dict) else m.model_dump() for m in messages]
    try:
        from jinja2.exceptions import TemplateError

        render_errors = (ValueError, TemplateError)
    except ImportError:
        render_errors = (ValueError,)
    found = {}
    for end, message in enumerate(messages, start=1):
        try:
            prefix = _token_list(
                tokenizer.apply_chat_template(
                    messages[:end],
                    tokenize=True,
                    add_generation_prompt=False,
                    enable_thinking=enable_thinking,
                )
            )
            offset = len(prefix)
        except render_errors:
            # Qwen3.5 requires at least one user query. Like MLX-LM server's
            # system segmentation, render an empty user probe, then retain
            # only tokens actually shared with the full real request. Dummy
            # tokens are never fed to the model or written into a cache key.
            if not all(m.get("role") == "system" for m in messages[:end]):
                continue
            try:
                probe = _token_list(
                    tokenizer.apply_chat_template(
                        messages[:end] + [{"role": "user", "content": ""}],
                        tokenize=True,
                        add_generation_prompt=False,
                        enable_thinking=enable_thinking,
                    )
                )
            except render_errors:
                continue
            offset = 0
            while offset < min(len(full_tokens), len(probe)) and full_tokens[offset] == probe[offset]:
                offset += 1
            prefix = probe[:offset]
        if 0 < offset < len(full_tokens) and full_tokens[:offset] == prefix:
            found[offset] = MessageBoundary(offset, end, message.get("role", "assistant"))
    return tuple(found[offset] for offset in sorted(found))
