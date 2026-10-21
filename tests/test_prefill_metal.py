"""Tiny real upstream control; no pretrained equivalence or timing claim."""
import pytest
from test_varlen_metal import tiny

from deepinfer_mac.continuous import set_prefill_chunk

pytestmark = pytest.mark.metal


def test_change_chunk_during_prefill_cancel_and_insert():
    import mlx.core as mx
    from mlx_lm.generate import BatchGenerator

    model = tiny()
    generator = BatchGenerator(model, completion_batch_size=2, prefill_batch_size=1,
                               prefill_step_size=2, stop_tokens=[])
    try:
        first = generator.insert([[1, 2, 3, 4, 5, 6, 7, 8]], max_tokens=[3])[0]
        prompt, _ = generator.next()
        assert prompt[0].uid == first and prompt[0].progress[0] == 2
        set_prefill_chunk(generator, 1)
        prompt, _ = generator.next()
        assert prompt[0].progress[0] == 3
        generator.remove([first], return_prompt_caches=False)
        set_prefill_chunk(generator, 3)
        second = generator.insert([[2, 3, 4, 5, 6, 7, 8, 9]], max_tokens=[3])[0]
        prompt, _ = generator.next()
        assert prompt[0].uid == second and prompt[0].progress[0] == 3
        assert generator._prompt_batch.prefill_step_size == 3
        generated = []
        for _ in range(20):
            prompt, output = generator.next()
            assert all(r.uid == second for r in prompt + output)
            generated.extend(output)
            if any(r.finish_reason is not None for r in output):
                break
        assert len(generated) == 3
        assert generated[-1].finish_reason is not None
    finally:
        mx.synchronize(generator.stream)
        generator.close()
