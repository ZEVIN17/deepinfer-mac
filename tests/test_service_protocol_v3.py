"""Frozen input/admission/actual CLI controls, without model execution."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).parents[1] / "benchmarks/experiments"
V3_SHA256 = "16750b5d512df3d88a0d9086f1b2070dc5ebacffb047cc2f512cda35af03deb7"


def module(name):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture(autouse=True)
def experiment_imports(monkeypatch):
    monkeypatch.syspath_prepend(str(EXPERIMENTS))


def manifest():
    raw = (EXPERIMENTS / "cache_sessions.v3.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == V3_SHA256
    return json.loads(raw)


def test_v3_preserves_every_v2_session_and_adds_independent_heldout_ledgers():
    v2 = json.loads((EXPERIMENTS / "cache_sessions.v2.json").read_bytes())
    v3 = manifest()
    assert v3["sessions"][:9] == v2["sessions"]
    assert len(v3["sessions"]) == 11
    added = v3["sessions"][9:]
    assert [s["context_seed_target_tokens"] for s in added] == [2048, 4096]
    assert all(s["split"] == "heldout" for s in added)
    # Distinct record values and prompts, not extra repetitions of old ledgers.
    for session in added:
        case = session["cases"][0]
        assert all(case["messages"] != prior["cases"][0]["messages"] for prior in v2["sessions"])
        assert all(case["expected_json"] != prior["cases"][0]["expected_json"] for prior in v2["sessions"])
        assert all(c["requested_total_context_tokens"] == c["prompt_tokens"] + 256 for c in session["cases"])
    assert len(v3["v3_token_calibration"]) == 13
    assert all(len(entry["token_ids_sha256_le_uint32"]) == 64
               for cases in v3["v3_token_calibration"].values() for entry in cases)
    assert {s["context_seed_target_tokens"] for s in v3["warmup_sessions"]} == {2048, 4096}
    assert all(warm["cases"][0]["messages"] != s["cases"][0]["messages"]
               for warm in v3["warmup_sessions"] for s in v3["sessions"])


def test_admission_has_six_clusters_and_explicit_8k_deferred_inventory():
    helper = module("service_resource_selection")
    value = manifest()
    selection = helper.resource_selection(value, [2048, 4096], 8192)
    assert selection["heldout_session_clusters"] == 6
    assert len(selection["selected_sessions"]) == 8
    assert len(selection["deferred_sessions"]) == 3
    assert all(s["seed_input_tokens"] == 8192 and s["reason"] for s in selection["deferred_sessions"])
    config = {"resource_selection": selection, "total_context_cap_tokens": 8192}
    assert len(helper.selected_sessions(config, value, "heldout")) == 6
    with pytest.raises(ValueError, match="remains deferred"):
        helper.resource_selection(value, [2048, 4096, 8192], 9216)
    with pytest.raises(ValueError, match="never truncate"):
        helper.resource_selection(value, [2048, 4096], 4096)
    selection["deferred_sessions"] = []
    with pytest.raises(ValueError, match="inventory"):
        helper.selected_sessions(config, value, "heldout")


def test_tuning_grid_cannot_launch_heldout(monkeypatch, tmp_path):
    runner = module("rotate_service_compare")
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Must reject before launch"))
    with pytest.raises(ValueError, match="dev-only"):
        runner.run({"dev_tuning_grid": True}, manifest(), tmp_path, 1, "heldout")


def test_prepared_variants_use_real_flags_and_disclose_additional_draft(monkeypatch, tmp_path):
    snapshot = tmp_path / "16daa4818c54ce5f5436f929d52542eb65bbed9d"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    root = tmp_path / "comparison"
    monkeypatch.setattr(sys, "argv", ["prepare", "--workspace", str(EXPERIMENTS.parents[1]),
        "--root", str(root), "--snapshot", str(snapshot), "--manifest", str(EXPERIMENTS / "cache_sessions.v3.json"),
        "--resource-approved-lengths", "2048", "4096", "--total-context-cap", "8192",
        "--variants", "baseline", "ds-original", "ds-lagged", "--message-checkpoints",
        "--own-prefill-step-size", "128", "--ds-lagged-prefill-step-size", "256", "--native-prefill-step-size", "1024"])
    module("prepare_service_compare").main()
    configuration = json.loads((root / "configuration.json").read_text())
    own, original, lagged, native, omlx = configuration["engines"]
    assert own["request_options"]["mode"] == "baseline" and own["additional_draft"] is None
    assert [e["prefill_step_size"] for e in (own, original, lagged, native, omlx)] == [128, 128, 256, 1024, 2048]
    server = module("deepinfer_service_server")
    for engine, adapter in ((original, "original"), (lagged, "lagged_mean")):
        options = server.parser().parse_args(engine["command"][2:])
        assert options.experimental_hybrid_speculation and options.hybrid_cost_adapter == adapter
        assert options.message_checkpoints and engine["request_options"]["mode"] == "adaptive"
        assert engine["additional_draft"]["weight_bytes"] == 625229487
        assert engine["additional_draft"]["weight_files"][0]["sha256"] == "f5a0d9dd3efa73510542a8023d610ff26be2b4b020d181cfc4bedaa1fcc5dd9e"
    assert configuration["resource_selection"]["heldout_session_clusters"] == 6
    assert all(not server.parser().parse_args(e["command"][2:]).round_diagnostics for e in (own, original, lagged))
    assert all(engine["cache_budget_bytes"] == 536870912 and engine["temperature"] == 0
               and not engine["enable_thinking"] for engine in configuration["engines"])


def test_runner_warms_every_approved_shape_per_variant_without_scoring_warmups(monkeypatch, tmp_path):
    runner = module("rotate_service_compare")
    helper = module("service_resource_selection")
    value = manifest()
    engines = [{"id": name, "weights_repo": "fixture", "weights_revision": "1", "quantization": "8bit",
                "cache_budget_bytes": 512, "url": "http://127.0.0.1:1", "model": "fixture", "command": []}
               for name in ("a", "b")]
    configuration = {"engines": engines, "total_context_cap_tokens": 8192,
                     "resource_selection": helper.resource_selection(value, [2048, 4096], 8192)}
    monkeypatch.setattr(runner, "require_free_local_port", lambda url: None)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: type("Process", (), {"pid": 1, "poll": lambda self: 0})())
    monkeypatch.setattr(runner, "await_ready", lambda *args: {})
    monkeypatch.setattr(runner, "stop", lambda process: None)
    # Test-only resource placeholder: this mocked runner never starts a process
    # and must not query host swap APIs or create real performance observations.
    monkeypatch.setattr(runner, "resource_sample", lambda *args: {})
    monkeypatch.setattr(runner, "measure", lambda url, model, case, **kwargs: {
        "case": case, "error": None, "quality_pass": False, "total_ms": None,
    })
    monkeypatch.chdir(tmp_path)
    report = runner.run(configuration, value, tmp_path / "report", 1, "dev")
    for block in report["blocks"]:
        assert len(block["warmups"]) == 8
        assert {r["case"]["prompt_tokens"] for r in block["warmups"]} == {2048, 2156, 4096, 4204}
        assert len(block["records"]) == 8
        assert all(r["session_id"] in {s["id"] for s in value["sessions"]} for r in block["records"])
        assert block["private_log"]["alias"].startswith("<PRIVATE_RESULTS>/")
    assert not list((tmp_path / "report").glob("*.log"))
    assert len(list((tmp_path / "benchmarks/results/private-service/report").glob("*.log"))) == 2


@pytest.mark.parametrize("kind", ["regular-directory", "wrong-symlink"])
def test_prepare_rejects_unrelated_model_alias_without_deleting_it(monkeypatch, tmp_path, kind):
    snapshot = tmp_path / "16daa4818c54ce5f5436f929d52542eb65bbed9d"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    root = tmp_path / "comparison"
    alias = root / "models/Qwen3.5-9B-8bit"
    alias.parent.mkdir(parents=True)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "user-file").write_text("preserve")
    if kind == "regular-directory":
        alias.mkdir()
        (alias / "user-file").write_text("preserve")
    else:
        alias.symlink_to(unrelated, target_is_directory=True)
    monkeypatch.setattr(sys, "argv", ["prepare", "--workspace", str(EXPERIMENTS.parents[1]),
        "--root", str(root), "--snapshot", str(snapshot), "--manifest", str(EXPERIMENTS / "cache_sessions.v3.json")])
    with pytest.raises(ValueError, match="exact symlink"):
        module("prepare_service_compare").main()
    assert (alias / "user-file").read_text() == "preserve"
