"""Test-only authored logs/records exercise gates; never performance evidence."""

import ast
import copy
import functools
import hashlib
import importlib.util
import json
import struct
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).parents[1] / "benchmarks/experiments"


@pytest.fixture(autouse=True)
def imports(monkeypatch):
    monkeypatch.syspath_prepend(str(EXPERIMENTS))


def module(name):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def expected(tmp_path):
    return {"draft_snapshot": str(tmp_path / "pinned-draft"), "model_id": "fixture-target",
            "block_size": 5, "verify_mode": "adaptive"}


def authored_log(config):
    return (f"DFlash drafter quantized: 43 linear layers at 4 bits (group 64)\n"
            f"DFlash drafter loaded: path={config['draft_snapshot']} kind=dflash block=5 adaptive=True target_layers=[1, 5, 9, 13, 17, 21, 25, 29]\n"
            "DFlash drafter attached to engine: fixture-target (kind=dflash, block=5, layers=[1, 5, 9, 13, 17, 21, 25, 29])\n"
            "MTP[0] finish=stop tokens=8 cycles=2 tok/cycle=4.00 accept=4/5 (80.0%)\n")


def test_config_and_http_success_cannot_replace_actual_dflash_execution(tmp_path):
    helper = module("omlx_dflash_protocol")
    config = expected(tmp_path)
    assert not helper.execution_evidence('dflash_enabled=True HTTP/1.1 200 OK', config)["valid"]
    result = helper.execution_evidence(authored_log(config), config)
    assert result["valid"]
    assert result["completed_sequence_stats"][0]["verified_draft_positions"] == 5


@pytest.mark.parametrize("mutation", ["fallback", "wrong_pin", "wrong_mode", "wrong_block", "missing_attach", "no_verify", "wrong_quant", "impossible_accepts"])
def test_external_log_gate_fails_closed(tmp_path, mutation):
    helper = module("omlx_dflash_protocol")
    config = expected(tmp_path)
    text = authored_log(config)
    if mutation == "fallback": text += "DFlash drafter load raised ... toggle ignored\n"
    elif mutation == "wrong_pin": text = text.replace(config["draft_snapshot"], str(tmp_path / "other-draft"))
    elif mutation == "wrong_mode": text = text.replace("adaptive=True", "adaptive=False")
    elif mutation == "wrong_block": text = text.replace("block=5", "block=4")
    elif mutation == "missing_attach": text = "\n".join(line for line in text.splitlines() if "attached to engine" not in line)
    elif mutation == "no_verify": text = "\n".join(line for line in text.splitlines() if not line.startswith("MTP["))
    elif mutation == "wrong_quant": text = text.replace("4 bits", "8 bits")
    elif mutation == "impossible_accepts": text = text.replace("accept=4/5", "accept=6/5")
    assert not helper.execution_evidence(text, config)["valid"]


def test_resource_policy_is_symmetric_and_retains_system_swap_growth():
    helper = module("service_resources")
    start = {"system_swap_used_bytes": 100, "system_available_bytes": 4 * 1024**3}
    samples = {"after_warmup": start, "after_requests": [start.copy()]}
    assert helper.eligibility(samples, helper.RESOURCE_POLICY)["valid"]
    samples["after_requests"].append(start | {"system_swap_used_bytes": 101})
    failure = helper.eligibility(samples, helper.RESOURCE_POLICY)
    assert not failure["valid"] and failure["max_scored_system_swap_growth_bytes"] == 1
    samples["after_requests"] = [start | {"system_available_bytes": 1024**3 - 1}]
    assert not helper.eligibility(samples, helper.RESOURCE_POLICY)["valid"]
    assert not helper.eligibility({}, helper.RESOURCE_POLICY)["valid"]


def test_cleanup_observer_runs_after_original_preserves_return_and_exception():
    tree = ast.parse((EXPERIMENTS / "omlx_dflash_audit_server.py").read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    cleanup = next(node for node in main.body if isinstance(node, ast.FunctionDef) and node.name == "cleanup")
    calls = []

    def original(self, finished):
        calls.append(("original", finished))
        if "error" in finished:
            raise RuntimeError("upstream error")
        return "original-return"

    env = {"functools": functools, "original_cleanup": original,
           "emit": lambda *args, **kwargs: calls.append(("emit", args)),
           "resources": lambda: calls.append(("resources",)) or {}}
    exec(compile(ast.Module(body=[cleanup], type_ignores=[]), "observer-test-only", "exec"), env)  # noqa: S102 -- repository function AST, no remote code
    assert env["cleanup"](object(), {"done"}) == "original-return"
    assert [call[0] for call in calls] == ["original", "resources", "emit"]
    calls.clear()
    assert env["cleanup"](object(), set()) == "original-return"
    assert [call[0] for call in calls] == ["original"]
    calls.clear()
    with pytest.raises(RuntimeError, match="upstream error"):
        env["cleanup"](object(), {"error"})
    assert [call[0] for call in calls] == ["original"]


def test_pinned_draft_file_bytes_are_verified_without_gpu(tmp_path):
    helper = module("omlx_dflash_protocol")
    snap = tmp_path / helper.DRAFT_REVISION
    snap.mkdir()
    config = {"hidden_size": 4096, "num_target_layers": 32, "vocab_size": 248320,
              "dflash_config": {"target_layer_ids": helper.TARGET_LAYERS}}
    (snap / "config.json").write_text(json.dumps(config))
    (snap / "model.safetensors").write_bytes(b"test-only-not-model-weights")
    files = [{"path": name, "size_bytes": (snap / name).stat().st_size,
              "sha256": hashlib.sha256((snap / name).read_bytes()).hexdigest()} for name in ("config.json", "model.safetensors")]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"repo": helper.DRAFT_REPO, "revision": helper.DRAFT_REVISION, "files": files, "license": "apache-2.0"}))
    assert helper.pinned_draft(snap, manifest)["revision"] == helper.DRAFT_REVISION
    (snap / "model.safetensors").write_bytes(b"different-bytes-same-size!!")
    with pytest.raises(ValueError, match="SHA256|differently sized"):
        helper.pinned_draft(snap, manifest)


def test_4k_resource_admission_does_not_accept_2k_or_adaptive_only(tmp_path):
    helper = module("omlx_dflash_protocol")
    report = {"status": "actual_dev_probe_completed", "target_revision": "16daa4818c54ce5f5436f929d52542eb65bbed9d", "draft_revision": helper.DRAFT_REVISION,
              "results": [{"engine": {"id": "q4-b5-dflash", "memory_guard": "balanced", "settings": helper.settings({"snapshot": "test-only"}, 5, "dflash")}, "dflash_execution_proven": True,
                           "error": None, "draft_load_soft_fallback_observed": False, "official_drafter_loaded_log_observed": True,
                           "all_strict_cases_passed": True, "system_swap_delta_bytes": 0,
                           "records": [{"case": {"prompt_tokens": 4204, "max_tokens": 256}, "quality_pass": True,
                                        "system_after": {"available_bytes": 4 * 1024**3},
                                        "request": {"max_tokens": 256},
                                        "audit_events": [{"event": "request_prompt", "prompt_tokens": 4204, "token_ids": list(range(4204)),
                                                          "token_ids_sha256_le_u32": hashlib.sha256(struct.pack("<4204I", *range(4204))).hexdigest()}]}]}]}
    path = tmp_path / "resource.json"
    path.write_text(json.dumps(report))
    assert helper.resource_admission(path)["verified"]
    for mutation in ("2k", "adaptive_only", "swap", "guard", "wrong_settings", "not_completed", "fallback"):
        bad = copy.deepcopy(report)
        if mutation == "2k": bad["results"][0]["records"][0]["case"]["prompt_tokens"] = 2048
        elif mutation == "adaptive_only": bad["results"][0]["engine"]["id"] = "q4-b5-adaptive"
        elif mutation == "guard": bad["results"][0]["engine"]["memory_guard"] = "aggressive"
        elif mutation == "wrong_settings": bad["results"][0]["engine"]["settings"]["dflash_verify_mode"] = "adaptive"
        elif mutation == "not_completed": bad["status"] = "running"
        elif mutation == "fallback": bad["results"][0]["draft_load_soft_fallback_observed"] = True
        else: bad["results"][0]["system_swap_delta_bytes"] = 1
        path.write_text(json.dumps(bad))
        assert not helper.resource_admission(path)["verified"]


def test_resource_lifecycle_and_one_sample_per_scored_request_are_mandatory():
    helper = module("service_resources")
    value = {"system_swap_used_bytes": 100, "system_available_bytes": 4 * 1024**3}
    samples = {key: value.copy() for key in ("before_start", "after_ready", "after_warmup", "before_stop", "after_stop")}
    samples["after_requests"] = [value.copy(), value.copy()]
    assert helper.eligibility(samples, helper.RESOURCE_POLICY, 2)["valid"]
    for length in (1, 3):
        bad = samples | {"after_requests": [value.copy() for _ in range(length)]}
        assert not helper.eligibility(bad, helper.RESOURCE_POLICY, 2)["valid"]
    for key in ("before_start", "after_ready", "before_stop", "after_stop"):
        bad = copy.deepcopy(samples)
        bad.pop(key)
        assert not helper.eligibility(bad, helper.RESOURCE_POLICY, 2)["valid"]
