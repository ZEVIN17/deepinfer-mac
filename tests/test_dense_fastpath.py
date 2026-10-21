"""Portable policy identity guards; no device or performance claims."""
import itertools

import pytest

from deepinfer_mac.ds.joint_budget import CostKey, JointBudget
from deepinfer_mac.varlen_verify import JointSpeculativeSession, dense_cache, execution_identity


def test_all_execution_policies_have_distinct_cost_identities():
    identities = [execution_identity(strategy, profile, draft_execution=draft, recovery=recovery)
                  for strategy, profile, draft, recovery in itertools.product(
                      ('bucket', 'shared_cap', 'padded_control'), (False, True),
                      ('serial', 'batched'), ('exact_replay', 'dense_trim'))]
    assert len(set(identities)) == 24
    assert execution_identity('bucket') == 'bucket'
    assert execution_identity('bucket', True) == 'bucket|draft=serial|recovery=exact_replay|profile=wall_v1'


@pytest.mark.parametrize('flags', [{'draft_execution': 'vector'}, {'recovery': 'approximate'}])
def test_unknown_policy_rejected_before_device_import(flags):
    key = CostKey('t', 'd', 'q', 1, 0, 'bucket', 'c')
    with pytest.raises(ValueError, match='policy'):
        JointSpeculativeSession(object(), object(), JointBudget(0, 't'), key,
                                draft_identity='d', calibration_identity='c', **flags)


def test_class_name_spoof_is_not_dense_cache():
    fake = type('KVCache', (), {})()
    assert not dense_cache([fake])
    assert not dense_cache([])
