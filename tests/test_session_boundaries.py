"""Template-dependent message boundaries require real full-prefix equality."""

from deepinfer_mac.session_boundaries import message_boundaries


class Template:
    def __init__(self, mapping=False, context_sensitive=False, require_user=False):
        self.mapping = mapping
        self.context_sensitive = context_sensitive
        self.require_user = require_user

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        if self.require_user and not any(m["role"] == "user" for m in messages):
            raise ValueError("No user query found")
        tokens = []
        for i, message in enumerate(messages):
            # A context-sensitive template makes the last message's closure
            # depend on whether another turn follows. Slicing such templates
            # without checking full tokens would produce an invalid snapshot.
            close = 99 if self.context_sensitive and i == len(messages) - 1 else 9
            tokens += [1, message["content"] if message["content"] != "" else 0, close]
        if add_generation_prompt:
            tokens += [7, int(enable_thinking)]
        return {"input_ids": tokens, "attention_mask": [1] * len(tokens)} if self.mapping else tokens


def test_verified_boundaries_include_system_and_final_user_without_consuming_tail():
    messages = [{"role": "system", "content": 2}, {"role": "user", "content": 3}]
    for mapping in (False, True):
        tokenizer = Template(mapping=mapping)
        full = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        result = message_boundaries(tokenizer, messages, full)
        assert [(b.token_offset, b.cache_type) for b in result] == [(3, "system"), (6, "user")]
        assert [b.message_count for b in result] == [1, 2]


def test_context_sensitive_partial_template_is_not_treated_as_a_prefix():
    tokenizer = Template(context_sensitive=True)
    messages = [{"role": "system", "content": 2}, {"role": "user", "content": 3}]
    full = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    result = message_boundaries(tokenizer, messages, full)
    assert [(b.token_offset, b.role) for b in result] == [(6, "user")]
    assert not message_boundaries(tokenizer, messages, [123, 456])


def test_no_full_sequence_or_empty_sequence_checkpoint():
    tokenizer = Template()
    messages = [{"role": "user", "content": 3}]
    full = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
    )
    assert not message_boundaries(tokenizer, messages, full)
    assert not message_boundaries(tokenizer, [], [])


def test_system_only_render_failure_uses_verified_empty_user_probe():
    tokenizer = Template(require_user=True)
    messages = [{"role": "system", "content": 2}, {"role": "user", "content": 3}]
    full = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    result = message_boundaries(tokenizer, messages, full)
    assert [(b.token_offset, b.cache_type) for b in result] == [(4, "system"), (6, "user")]
    # The 0 dummy content token is never used as a cache prefix.
    assert full[result[0].token_offset] == 3
