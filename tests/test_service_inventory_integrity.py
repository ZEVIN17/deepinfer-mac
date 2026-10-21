"""Test-only fabricated records exercise integrity gates; never benchmark data."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).parents[1] / "benchmarks/experiments"


def module(name):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture(autouse=True)
def imports(monkeypatch):
    monkeypatch.syspath_prepend(str(EXPERIMENTS))


def complete_fixture():
    inventory = module("service_inventory")
    manifest = json.loads((EXPERIMENTS / "cache_sessions.v3.json").read_bytes())
    helper = module("service_resource_selection")
    engines = [{"id": name, "model": "test-fixture", "command": ["python", "fixture.py"], "cwd": "fixture"}
               for name in ("candidate", "control")]
    env = {"versions": {"test-only": "0"}, "files": {"fixture.py": "a" * 64}}
    env["sha256"] = inventory.digest(env)
    files = {"test_only_fixture.py": "b" * 64}
    configuration = {"engines": engines, "manifest_canonical_sha256": inventory.digest(manifest),
        "total_context_cap_tokens": 8192, "resource_selection": helper.resource_selection(manifest, [2048, 4096], 8192),
        "frozen_source": {"files": files, "sha256": inventory.digest(files), "archive_sha256": "c" * 64},
        "interpreter_environments": {"python": env}}
    proof = {"verified": True, "execution_binding_verified": True, "source_sha256": inventory.digest(files),
             "archive_sha256": "c" * 64, "environments": {"python": {"sha256": env["sha256"]}}}
    raw = {"status": "completed", "planned_repetitions": 2, "split": "heldout", "configuration": configuration,
           "configuration_sha256": inventory.digest(configuration), "manifest": manifest,
           "order_balance": {"fully_balanced_positions": True},
           "source_checks": {"initial": proof, "final": proof}, "blocks": []}
    for repetition in range(2):
        for engine in engines[repetition:] + engines[:repetition]:
            def record(case, engine=engine):
                return {"case": copy.deepcopy(case), "request": inventory.request_body(engine["model"], case),
                        "text": json.dumps(case["expected_json"]), "quality_pass": True, "protocol_complete": True,
                        "error": None, "total_ms": 5 if engine["id"] == "candidate" else 10, "ttft_text_ms": 2}
            records = []
            for session in helper.selected_sessions(configuration, manifest, "heldout"):
                for case in session["cases"]:
                    records.append(record(case) | {"session_id": session["id"], "split": "heldout", "repetition": repetition})
            raw["blocks"].append({"engine": engine["id"], "repetition": repetition, "status": "completed", "error": None,
                "process_stopped": True, "executed_command": engine["command"], "execution_cwd": engine["cwd"],
                "records": records, "warmups": [record(case) for case in inventory.warmup_cases(configuration, manifest)],
                "source_checks": {"before": proof, "after": proof}})
    return raw


def test_complete_test_fixture_can_pass_and_preserves_six_clusters():
    raw = complete_fixture()
    assert module("service_inventory").audit(raw)["valid"]
    result = module("summarize_service_compare").summarize(raw, "candidate", draws=20)
    total = [r for r in result["paired"] if r["metric"] == "total_ms"]
    assert total and all(r["session_clusters"] == 6 and r["engineering_target_met"] for r in total)


@pytest.mark.parametrize("mutation", ["missing_block", "duplicate_block", "extra_block", "wrong_order", "missing_kind",
    "duplicate_request", "extra_request", "request_payload", "case_body", "missing_metric", "failed_status",
    "missing_warmup", "warmup_payload", "missing_balance", "before_drift", "after_drift", "final_drift",
    "missing_final", "wrong_source_hash", "missing_environment", "execution_path", "wrong_request_order"])
def test_any_inventory_or_source_mutation_disables_target(mutation):
    raw = copy.deepcopy(complete_fixture())
    first = raw["blocks"][0]
    if mutation == "missing_block": raw["blocks"].pop()
    elif mutation == "duplicate_block": raw["blocks"].append(copy.deepcopy(first))
    elif mutation == "extra_block": raw["blocks"].append(first | {"engine": "unexpected"})
    elif mutation == "wrong_order": raw["blocks"][0], raw["blocks"][1] = raw["blocks"][1], raw["blocks"][0]
    elif mutation == "missing_kind": first["records"] = [r for r in first["records"] if r["case"]["kind"] != "changed_tail_branch"]
    elif mutation == "duplicate_request": first["records"].append(copy.deepcopy(first["records"][0]))
    elif mutation == "extra_request": first["records"].append(first["records"][0] | {"session_id": "unexpected"})
    elif mutation == "request_payload": first["records"][0]["request"]["temperature"] = 1
    elif mutation == "case_body": first["records"][0]["case"]["max_tokens"] = 8
    elif mutation == "missing_metric": first["records"][0]["total_ms"] = None
    elif mutation == "failed_status": first["status"] = "interrupted"
    elif mutation == "missing_warmup": first["warmups"].pop()
    elif mutation == "warmup_payload": first["warmups"][0]["request"]["max_tokens"] = 8
    elif mutation == "missing_balance": raw.pop("order_balance")
    elif mutation == "before_drift": first["source_checks"]["before"]["verified"] = False
    elif mutation == "after_drift": first["source_checks"]["after"]["verified"] = False
    elif mutation == "final_drift": raw["source_checks"]["final"]["verified"] = False
    elif mutation == "missing_final": raw["source_checks"].pop("final")
    elif mutation == "wrong_source_hash": first["source_checks"]["after"]["source_sha256"] = "wrong"
    elif mutation == "missing_environment": first["source_checks"]["after"]["environments"] = {}
    elif mutation == "execution_path": first["executed_command"] = ["different.py"]
    elif mutation == "wrong_request_order": first["records"][0], first["records"][1] = first["records"][1], first["records"][0]
    result = module("summarize_service_compare").summarize(raw, "candidate", draws=10)
    assert not result["protocol_inventory"]["valid"]
    assert not any(r["engineering_target_met"] for r in result["paired"])


def test_frozen_actual_helper_and_runtime_bytes_detect_midrun_drift(monkeypatch, tmp_path):
    freeze = module("service_source_freeze")
    workspace = tmp_path / "workspace"
    for name in ("src/deepinfer_mac/core.py", "benchmarks/experiments/public_service_report.py",
                 "benchmarks/experiments/prompt_token_audit.py", "benchmarks/experiments/cache_sessions.v3.json"):
        target = workspace / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}" if target.suffix == ".json" else "# fixture\n")
    frozen = freeze.freeze(workspace, tmp_path / "source", workspace / "benchmarks/experiments/cache_sessions.v3.json")
    environment = {"sha256": "test-only"}
    monkeypatch.setattr(freeze, "environment", lambda *args: environment)
    config = {"frozen_source": frozen, "interpreter_environments": {"fixture-python": environment}}
    assert freeze.check(config)["verified"]
    helper = tmp_path / "source/benchmarks/experiments/prompt_token_audit.py"
    helper.write_text("# changed after a test block\n")
    assert not freeze.check(config)["verified"]
    assert "source bytes/inventory drift" in freeze.check(config)["errors"]
