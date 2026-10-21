"""Actual native transition after speculative commit, with dynamic membership."""
import copy
from types import SimpleNamespace

import pytest
from test_allocated_draft_metal import session_for
from test_varlen_metal import rows_for, tiny

from deepinfer_mac.continuous import ContinuousConfig, Prepared
from deepinfer_mac.ds.joint_budget import RequestKey
from deepinfer_mac.joint_service import MLXJointBatchSession
from deepinfer_mac.native_handoff import CommittedPrefix, transfer_to_native

pytestmark = pytest.mark.metal


def test_native_handoff_continues_exact_committed_prefix():
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    target, draft = tiny(), tiny(19)
    rows, drafts = rows_for(target), rows_for(draft)
    session = session_for(target, draft, 'serial', 'exact_replay', 'bucket', 'allocated')
    result = session.tick(rows, [r.cache for r in drafts], forced_lengths=(1, 4, 2))
    prefixes = [CommittedPrefix(i + 10, row.cache, out[-1], 5) for i, (row, out) in enumerate(zip(rows, result['outputs']))]
    expected = [[int(t) for t, _ in generate_step(mx.array([p.tail]), target,
                 prompt_cache=copy.deepcopy(p.cache), max_tokens=5)] for p in prefixes]
    generator, mapping = transfer_to_native(target, prefixes, stop_tokens=[], max_active=3, prefill_chunk=8)
    try:
        outputs = {p.uid: [] for p in prefixes}
        for _ in range(12):
            _, responses = generator.next()
            for response in responses:
                outputs[mapping[response.uid]].append(response.token)
            if all(len(x) == 5 for x in outputs.values()):
                break
        assert list(outputs.values()) == expected
    finally:
        mx.synchronize(generator.stream)
        generator.close()


def test_service_handoff_cancel_newcomer_and_close():
    import mlx.core as mx
    target, draft = tiny(), tiny(19)
    rows, drafts = rows_for(target), rows_for(draft)
    service = MLXJointBatchSession.__new__(MLXJointBatchSession)
    service.engine = SimpleNamespace(model=target, tokenizer=SimpleNamespace(eos_token_ids=[]))
    service.mx, service.config = mx, ContinuousConfig(max_active=3)
    service.closed, service.native_generator, service.native_uids = False, None, {}
    service.rows = {i: SimpleNamespace(target=r.cache, draft=d.cache, tail=r.tail,
                                       remaining=5, consumed=r.cache[0].offset, key=r.key)
                    for i, (r, d) in enumerate(zip(rows, drafts))}
    service.next_uid = 2
    service.session = session_for(target, draft, 'serial', 'exact_replay', 'bucket', 'allocated')
    service.joint_budget = service.session.budget
    service.switch_to_native()
    assert service.session is None and all(not r.draft for r in service.rows.values())
    service.switch_to_native()  # Idempotent.
    service.remove(1)
    uid = service.insert(Prepared([1, 2, 3], 0, None, None, {}, set()), SimpleNamespace(max_tokens=2),
                         key=RequestKey('new', 2))
    seen = set()
    try:
        for _ in range(20):
            prompts, outputs = service.tick()
            assert all(r.uid != 1 for r in prompts + outputs)
            seen.update(r.uid for r in outputs)
            if not service.rows:
                break
        assert not service.rows and seen == {0, 2, uid}
        assert service.last_report['strategy'] == 'native_batch'
    finally:
        service.close()
        service.close()
