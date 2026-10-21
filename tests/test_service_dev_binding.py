"""Authored test-only records exercise selection binding, never benchmark data."""

import copy
import importlib.util
import json
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


def fixture(tmp_path):
    inv = module("service_inventory")
    payload = {"kind": "fixture", "messages": [{"role": "user", "content": "test-only"}],
               "expected_json": {}, "max_tokens": 16}
    manifest = {"sessions": [{"id": "test-only-dev", "split": "dev", "cases": [payload]}]}
    env = {"versions": {"test-only": "0"}, "files": {"fixture.py": "a" * 64}}
    env["sha256"] = inv.digest(env)
    source = {"fixture.py": "b" * 64}
    engines = [{"id": name, "model": "fixture", "tuning_group": group, "prefill_step_size": chunk,
                "command": [str(tmp_path / "python"), str(tmp_path / "dev/source/fixture.py"), "--prefill-step-size", str(chunk)],
                "cwd": str(tmp_path / "dev/source"), "weights_repo": "test-only", "weights_revision": "fixed",
                "quantization": "8bit", "cache_budget_bytes": 512}
               for name, group, chunk in (("native-chunk128", "native", 128), ("omlx", "omlx", 2048),
                                          ("omlx-dflash", "omlx", 2048))]
    cfg = {"dev_tuning_grid": True, "engines": engines, "manifest_canonical_sha256": inv.digest(manifest),
           "total_context_cap_tokens": 4096, "resource_policy": None,
           "frozen_source": {"files": source, "sha256": inv.digest(source), "archive_sha256": "c" * 64},
           "interpreter_environments": {str(tmp_path / "python"): env}}
    proof = {"verified": True, "execution_binding_verified": True, "source_sha256": inv.digest(source),
             "archive_sha256": "c" * 64, "environments": {str(tmp_path / "python"): {"sha256": env["sha256"]}}}
    raw = {"configuration": cfg, "configuration_sha256": inv.digest(cfg), "manifest": manifest,
           "status": "completed", "planned_repetitions": 1, "split": "dev", "blocks": [],
           "source_checks": {"initial": proof, "final": proof}, "order_balance": {"fully_balanced_positions": False}}
    for engine, duration in zip(engines, (10, 20, 12), strict=True):
        record = {"case": payload, "request": inv.request_body(engine["model"], payload), "text": "{}",
                  "quality_pass": True, "protocol_complete": True, "error": None,
                  "total_ms": duration, "ttft_text_ms": 2}
        raw["blocks"].append({"engine": engine["id"], "repetition": 0, "status": "completed", "error": None,
                              "process_stopped": True, "executed_command": engine["command"], "execution_cwd": engine["cwd"],
                              "records": [record | {"session_id": "test-only-dev", "split": "dev", "repetition": 0}],
                              "warmups": [record.copy()], "source_checks": {"before": proof, "after": proof}})
    return raw


def test_private_raw_public_summary_accepts_identical_settings_and_rejects_stale_choices(tmp_path):
    helper = module("service_dev_selection")
    privacy = module("public_service_report")
    raw = fixture(tmp_path)
    summary = {"dev_candidate_selection": helper.select(raw)}
    summary = privacy.public_report({"configuration": raw["configuration"], "summary": summary})["summary"]
    formal = [copy.deepcopy(raw["configuration"]["engines"][index]) for index in (0, 2)]
    formal[0]["id"] = "native-formal"
    cfg = raw["configuration"]
    evidence = helper.bind(raw, summary, formal, raw["manifest"], cfg["frozen_source"]["sha256"],
                           cfg["interpreter_environments"], formal_context=cfg)
    assert evidence["verified"]
    assert [binding["dev_engine_id"] for binding in evidence["formal_bindings"]] == ["native-chunk128", "omlx-dflash"]
    for mutation in ("summary", "manifest", "source", "environment", "chunk", "baseline", "context", "command"):
        bad_summary, bad_formal, bad_manifest = copy.deepcopy(summary), copy.deepcopy(formal), copy.deepcopy(raw["manifest"])
        source, environment, context = cfg["frozen_source"]["sha256"], copy.deepcopy(cfg["interpreter_environments"]), copy.deepcopy(cfg)
        if mutation == "summary": bad_summary["dev_candidate_selection"]["global_errors"] = ["changed"]
        elif mutation == "manifest": bad_manifest["sessions"][0]["cases"][0]["max_tokens"] = 8
        elif mutation == "source": source = "different-even-with-note"
        elif mutation == "environment": environment["different-python"] = environment.pop(next(iter(environment)))
        elif mutation == "chunk": bad_formal[0]["prefill_step_size"] = 256
        elif mutation == "baseline": bad_formal[1] = copy.deepcopy(cfg["engines"][1]); bad_formal[1]["prefill_step_size"] = 128
        elif mutation == "context": context["total_context_cap_tokens"] = 8192
        else: bad_formal[1]["command"].append("--runtime-audit")
        with pytest.raises(ValueError):
            helper.bind(raw, bad_summary, bad_formal, bad_manifest, source, environment,
                        bridge_note="test note cannot bypass drift", formal_context=context)


def test_dev_selection_keeps_failed_candidates_and_never_selects_strict_failure(tmp_path):
    helper = module("service_dev_selection")
    raw = fixture(tmp_path)
    raw["blocks"][2]["records"][0]["quality_pass"] = False
    result = helper.select(raw)
    assert "omlx" in result["selected_engine_ids"] and "omlx-dflash" not in result["selected_engine_ids"]
    assert not next(candidate for candidate in result["candidates"] if candidate["id"] == "omlx-dflash")["eligible"]


def test_public_report_removes_summary_and_raw_execution_paths(tmp_path):
    helper = module("public_service_report")
    raw_path, summary_path = str(tmp_path / "private-raw.json"), str(tmp_path / "summary.json")
    cfg = {"dev_selection_report": {"raw_path": raw_path, "path": summary_path},
           "runtime_files": [{"path": raw_path}, {"path": summary_path}]}
    public = helper.public_report({"configuration": cfg})
    text = json.dumps(public)
    assert raw_path not in text and summary_path not in text
    assert public["configuration"]["runtime_files"] == [{"path": "<DEV_RAW_REPORT>"}, {"path": "<DEV_SELECTION_REPORT>"}]


def test_new_formal_portable_audit_requires_matching_actual_dev_binding(tmp_path):
    helper = module("service_dev_selection")
    inv = module("service_inventory")
    raw = fixture(tmp_path)
    formal = [raw["configuration"]["engines"][i] for i in (0, 2)]
    cfg = raw["configuration"]
    evidence = helper.bind(raw, {"dev_candidate_selection": helper.select(raw)}, formal, raw["manifest"],
                           cfg["frozen_source"]["sha256"], cfg["interpreter_environments"], formal_context=cfg)
    evidence.update(sha256="d" * 64, raw_sha256="e" * 64)
    configured = cfg | {"engines": formal, "dev_selection_report": evidence}
    observed = {"verified": True, "binding_sha256": inv.digest(evidence)}
    assert helper.portable_formal_evidence(configured, observed)
    assert not helper.portable_formal_evidence(configured, {})
    changed = copy.deepcopy(configured)
    changed["engines"][1]["prefill_step_size"] = 128
    assert not helper.portable_formal_evidence(changed, observed)


def test_public_formal_binding_digest_changes_with_aliases_and_can_be_coherently_rebuilt(tmp_path):
    helper = module("service_dev_selection")
    inv = module("service_inventory")
    privacy = module("public_service_report")
    raw = fixture(tmp_path)
    formal = [raw["configuration"]["engines"][i] for i in (0, 2)]
    cfg = raw["configuration"]
    evidence = helper.bind(raw, {"dev_candidate_selection": helper.select(raw)}, formal, raw["manifest"],
                           cfg["frozen_source"]["sha256"], cfg["interpreter_environments"], formal_context=cfg)
    evidence.update(path=str(tmp_path / "summary.json"), raw_path=str(tmp_path / "raw.json"), sha256="d" * 64, raw_sha256="e" * 64)
    configured = cfg | {"engines": formal, "dev_selection_report": evidence}
    report = {"configuration": configured, "dev_selection_check": {"verified": True, "binding_sha256": inv.digest(evidence)}}
    public = privacy.public_report(report)
    assert not helper.portable_formal_evidence(public["configuration"], public["dev_selection_check"])
    public["dev_selection_check"]["binding_sha256"] = inv.digest(public["configuration"]["dev_selection_report"])
    assert helper.portable_formal_evidence(public["configuration"], public["dev_selection_check"])
    assert report["dev_selection_check"]["binding_sha256"] == inv.digest(evidence)
