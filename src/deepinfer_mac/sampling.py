"""MLX sampling configuration and full-history adapters after cache/prefill reuse."""
from __future__ import annotations


def sampling_parameters(request):
    return {name: getattr(request, name) for name in ('temperature', 'top_p', 'top_k', 'presence_penalty', 'seed')}


def generation_sampling(request, consumed_prefix=()):
    """Keep penalty semantics identical when generate_step receives a suffix.

    MLX accumulates only tokens passed to its generator. Our manual prefill and
    cached native path have already consumed a prefix, so penalty processors must
    see that exact prefix followed by MLX's current suffix/history. KV content is
    independent of sampler choices; cache identity remains based on exact tokens.
    """
    if request.temperature == 0 and request.presence_penalty == 0:
        return {}
    import mlx.core as mx
    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    processors = make_logits_processors(presence_penalty=request.presence_penalty, presence_context_size=0)
    if consumed_prefix and processors:
        prefix = mx.array(consumed_prefix, mx.uint32)

        def with_prefix(processor):
            def process(history, logits):
                return processor(mx.concatenate([prefix, history]), logits)
            return process

        processors = [with_prefix(processor) for processor in processors]
    return {'sampler': make_sampler(temp=request.temperature, top_p=request.top_p, top_k=request.top_k),
            'logits_processors': processors}
