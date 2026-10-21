"""Portable experiment wiring and independent source-trace verification."""

import copy
import hashlib
import importlib.util
import json
import sys
import tarfile
from pathlib import Path

import pytest
from test_hybrid_speculative import Model, RecurrentCache, fake_device

from deepinfer_mac.ds.recovery_cost import LaggedMeanRecoveryPolicy
from deepinfer_mac.hybrid_speculative import hybrid_generate_step

__all__ = ["fake_device"]
ROOT = Path(__file__).parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_policy_variants_are_independent_and_no_fake_hybrid_native_fixed():
    harness = module("benchmarks/experiments/ds_budget_ablation.py", "ds_variant_harness")
    hybrid = harness.adaptive_policies(True)
    assert {"legacy", "ds", "ds_lagged_mean"} == set(hybrid)
    assert len({id(p.positions) for p in hybrid.values()}) == 3
    assert hybrid["ds"].state is not hybrid["ds_lagged_mean"].state
    hybrid["ds_lagged_mean"].positions[0][0] = 2
    assert hybrid["ds"].positions[0] == [0, 0] and hybrid["legacy"].positions[0] == [0, 0]
    assert "ds_lagged_mean" in harness.same_loop_variants(True)
    assert "ds_lagged_mean" not in harness.same_loop_variants(False)
    assert "native_fixed" not in harness.same_loop_variants(True)


def test_freeze_manifest_covers_harness_and_detects_changed_source(monkeypatch, tmp_path):
    freezer = module("scripts/freeze_ds_source.py", "ds_freeze_helper")
    harness = module("benchmarks/experiments/ds_budget_ablation.py", "ds_freeze_harness")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(harness, "source_manifest", lambda: {"files": {
        "src/deepinfer_mac/fixture.py": hashlib.sha256(Path("src/deepinfer_mac/fixture.py").read_bytes()).hexdigest(),
    }})
    source = Path("src/deepinfer_mac/fixture.py")
    runner = Path("benchmarks/experiments/ds_budget_ablation.py")
    source.parent.mkdir(parents=True)
    runner.parent.mkdir(parents=True)
    source.write_text("# unit fixture, not a benchmark\n")
    runner.write_bytes(Path(harness.__file__).read_bytes())
    output, extract = tmp_path / "source.tar.gz", tmp_path / "frozen"
    freezer.freeze([source, runner], output, extract)
    manifest = extract / "ds-source-manifest.json"
    assert harness.freeze_record(manifest)["formal_validity"] is True
    runner.write_text("# changed after freeze\n")
    record = harness.freeze_record(manifest)
    assert record["formal_validity"] is False and str(runner) in record["mismatches"]
    with pytest.raises(FileExistsError, match="fresh"):
        freezer.freeze([source], output, extract)
    summary = tmp_path / "summary.json"
    summary.write_text('{"files":47}')
    assert harness.freeze_record(summary)["formal_validity"] is False



@pytest.fixture
def frozen_source(monkeypatch, tmp_path):
    freezer = module("scripts/freeze_ds_source.py", "ds_bound_freeze_helper")
    harness = module("benchmarks/experiments/ds_budget_ablation.py", "ds_bound_harness")
    source = Path("src/deepinfer_mac/fixture.py")
    runner = Path("benchmarks/experiments/ds_budget_ablation.py")
    runner_bytes = Path(harness.__file__).read_bytes()
    monkeypatch.chdir(tmp_path)
    source.parent.mkdir(parents=True)
    runner.parent.mkdir(parents=True)
    source.write_text("# portable provenance fixture, no inference\n")
    runner.write_bytes(runner_bytes)
    monkeypatch.setattr(harness, "source_manifest", lambda: {"files": {
        str(source): hashlib.sha256(source.read_bytes()).hexdigest()}})
    output, extract = Path("source.tar.gz"), Path("frozen")
    freezer.freeze([source, runner], output, extract)
    return harness, extract / "ds-source-manifest.json", output, runner


def test_genuine_archive_binding_and_legacy_explicit_archive(frozen_source):
    harness, manifest, archive, _ = frozen_source
    record = harness.freeze_record(manifest)
    assert record["formal_validity"] and record["archive_sha256"]
    assert record["archive_name"] == "source.tar.gz"
    assert ("/" + "Users" + "/") not in json.dumps(record) and str(Path.home()) not in json.dumps(record)
    with tarfile.open(archive) as tar:
        blob = tar.extractfile("ds-source-manifest.json").read()
    assert "archive_binding" not in json.loads(blob)
    manifest.write_bytes(blob)  # A genuine legacy extracted manifest, no changes to its archive.
    assert not harness.freeze_record(manifest)["formal_validity"]
    assert harness.freeze_record(manifest, archive_path=archive)["formal_validity"]


def test_loose_wrong_tree_and_missing_archive_cannot_certify(frozen_source):
    harness, manifest, archive, _ = frozen_source
    data = json.loads(manifest.read_text())
    data.pop("archive_binding")
    data["tree_sha256"] = "incorrect-tree-digest"
    manifest.write_text(json.dumps(data))
    assert not harness.freeze_record(manifest)["formal_validity"]
    # Even with a genuine archive, the external manifest/tree must be bound to it.
    result = harness.freeze_record(manifest, archive_path=archive)
    assert not result["formal_validity"]
    assert {"manifest_tree_digest", "internal_manifest_binding"} <= set(result["errors"])
    archive.unlink()
    assert not harness.freeze_record(manifest, archive_path=archive)["formal_validity"]


def test_archive_hash_and_content_are_both_verified(frozen_source):
    import io

    harness, manifest, archive, _ = frozen_source
    with tarfile.open(archive) as tar:
        entries = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
    entries["src/deepinfer_mac/fixture.py"] += b"# tampered archived content\n"
    with tarfile.open(archive, "w:gz") as tar:
        for name, blob in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(blob)
            tar.addfile(info, io.BytesIO(blob))
    assert "archive_digest" in harness.freeze_record(manifest)["errors"]
    # Merely updating the claimed archive hash does not conceal a wrong file/member map.
    data = json.loads(manifest.read_text())
    data["archive_binding"]["sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(data))
    result = harness.freeze_record(manifest)
    assert not result["formal_validity"]
    assert "archive_content:src/deepinfer_mac/fixture.py" in result["errors"]


def test_executed_external_harness_cannot_borrow_frozen_cwd_hash(frozen_source):
    harness, manifest, _, runner = frozen_source
    external = Path("external_runner.py")
    external.write_bytes(runner.read_bytes() + b"\n# externally modified execution\n")
    # Reproduce the actual imported external __file__ case, not only an explicit override.
    external_module = module(external.resolve(), "ds_external_harness")
    external_module.source_manifest = harness.source_manifest
    assert harness.freeze_record(manifest)["formal_validity"]
    record = external_module.freeze_record(manifest)
    assert not record["formal_validity"] and "executing_harness_digest" in record["errors"]


def provenance_fixture(record):
    return {"source": {"source_tree_sha256": "runtime"}, "source_end": {"source_tree_sha256": "runtime"},
            "harness_sha256": record["executing_harness_sha256"],
            "harness_sha256_end": record["executing_harness_sha256"],
            "freeze": copy.deepcopy(record), "freeze_end": copy.deepcopy(record),
            "summary": {"correctness_and_maturity_pass": True}}


def test_probe_acceptance_requires_end_source_manifest_archive_and_executing_script(frozen_source, monkeypatch):
    harness, manifest, _, _ = frozen_source
    monkeypatch.setitem(sys.modules, "ds_budget_ablation", harness)
    oracle = module("benchmarks/experiments/verify_lagged_trace.py", "verify_lagged_trace")
    monkeypatch.setitem(sys.modules, "verify_lagged_trace", oracle)
    probe = module("benchmarks/experiments/hybrid_lagged_probe.py", "ds_probe_acceptance")
    report = provenance_fixture(harness.freeze_record(manifest))
    probe.finalize_acceptance(report)
    assert report["summary"]["accepted_frozen_probe_pass"]
    for target, key in [("source_end", "source_tree_sha256"), ("freeze_end", "manifest_sha256"),
                        ("freeze_end", "tree_sha256"), ("freeze_end", "archive_sha256"),
                        ("freeze_end", "internal_manifest_sha256"), ("freeze_end", "executing_harness_sha256")]:
        altered = copy.deepcopy(report)
        altered[target][key] = "changed-but-still-claims-formally-valid"
        probe.finalize_acceptance(altered)
        assert altered["summary"]["correctness_and_maturity_pass"]  # Raw functional outcome is retained.
        assert not altered["summary"]["accepted_frozen_probe_pass"]
    for key in ("harness_sha256_end", "fatal_error"):
        altered = copy.deepcopy(report)
        altered[key] = "changed"
        probe.finalize_acceptance(altered)
        assert not altered["summary"]["accepted_frozen_probe_pass"]


def test_protocol_rejects_nonpositive_counts_before_runtime_import():
    harness = module("benchmarks/experiments/ds_budget_ablation.py", "ds_argument_harness")
    for option in ("--repeats", "--max-tokens"):
        for value in ("0", "-1"):
            with pytest.raises(SystemExit) as rejected:
                harness.parse_args([option, value])
            assert rejected.value.code == 2
    assert harness.parse_args(["--repeats", "1", "--max-tokens", "1"]).repeats == 1


def test_heldout_requires_full_nonempty_exact_predeclared_coverage():
    harness = module("benchmarks/experiments/ds_budget_ablation.py", "ds_coverage_harness")
    variants = ["native_greedy", "ds", "ds_lagged_mean"]
    rows = [{"stage": "held_out", "prompt_id": name, "repeat": repeat,
             "variants": {v: {"ids": [3, 1], "total_ms": 1} for v in variants}}
            for name in ("p1", "p2") for repeat in range(2)]
    check = lambda samples, repeats=2: harness.heldout_coverage(samples, ["p1", "p2"], repeats, variants)
    assert check(rows)["complete_and_exact"]
    assert not check([], 0)["complete_and_exact"]
    assert not check([])["complete_and_exact"]
    assert not check(rows[:-1])["complete_and_exact"]
    assert not check(rows + [rows[0]])["complete_and_exact"]
    for mode in ("missing", "error", "empty", "mismatch", "no_time"):
        altered = copy.deepcopy(rows)
        result = altered[0]["variants"]["ds"]
        if mode == "missing":
            del altered[0]["variants"]["ds"]
        elif mode == "error":
            result["error"] = "retained failure"
        elif mode == "empty":
            for r in altered[0]["variants"].values():
                r["ids"] = []
        elif mode == "mismatch":
            result["ids"] = [3, 2]
            altered[0]["equal_to_greedy"] = {v: True for v in variants}  # Stale flags cannot certify outputs.
        else:
            result["total_ms"] = None
        assert not check(altered)["complete_and_exact"]


def test_cpu_probe_failure_product_keeps_private_archive_binding_out_of_report(frozen_source, monkeypatch):
    harness, _, _, _ = frozen_source
    monkeypatch.setitem(sys.modules, "ds_budget_ablation", harness)
    oracle = module("benchmarks/experiments/verify_lagged_trace.py", "verify_lagged_trace")
    monkeypatch.setitem(sys.modules, "verify_lagged_trace", oracle)
    probe = module("benchmarks/experiments/hybrid_lagged_probe.py", "ds_probe_failure_product")
    runner = Path("benchmarks/experiments/hybrid_lagged_probe.py")
    runner.write_bytes(Path(probe.__file__).read_bytes())
    freezer = module("scripts/freeze_ds_source.py", "ds_probe_product_freezer")
    frozen = Path("probe-frozen")
    archive = Path("probe-source.tar.gz")
    freezer.freeze([Path("src/deepinfer_mac/fixture.py"), Path("benchmarks/experiments/ds_budget_ablation.py"),
                    runner], archive, frozen)
    manifest = frozen / "ds-source-manifest.json"
    # Local absolute bindings are allowed only as private locators.
    data = json.loads(manifest.read_text())
    data["archive_binding"]["path"] = str(archive.resolve())
    manifest.write_text(json.dumps(data))
    monkeypatch.setattr(probe, "source_manifest", harness.source_manifest)

    class NoGPU:
        def __init__(self, **kwargs):
            pass

        def _load(self, *args):
            raise FileNotFoundError(str(Path.home() / "private-model-fixture"))

        def close(self):
            pass

    monkeypatch.setattr(probe, "InferenceEngine", NoGPU)
    output = Path("cpu-probe-failure.json")
    monkeypatch.setattr(sys, "argv", [str(runner), "--freeze-manifest", str(manifest), "--output", str(output)])
    with pytest.raises(FileNotFoundError):
        probe.main()
    product = json.loads(output.read_text())
    assert product["freeze"]["formal_validity"]
    assert not product["summary"]["accepted_frozen_probe_pass"]
    assert "fatal_error" in product and "<user-home>" in product["fatal_error"]
    assert "archive_binding" not in output.read_text()
    assert ("/" + "Users" + "/") not in output.read_text() and str(Path.home()) not in output.read_text()

def portable_actual_trace():
    # Stateful fake-device execution is a correctness fixture, not performance data.
    policy = LaggedMeanRecoveryPolicy(candidates=(0, 1, 2), min_samples=2, window=4)
    rounds = []
    list(hybrid_generate_step([0], Model(), Model(), [RecurrentCache()], [RecurrentCache()],
                             policy, 120, rounds))
    report = {"protocol": {"candidates": [0, 1, 2], "min_samples": 2, "window": 4},
              "samples": [{"variant": "ds_lagged_mean", "policy_stream": "fixture",
                           "result": {"prompt_tokens": 1, "rounds": rounds}}]}
    return json.loads(json.dumps(report))


def test_independent_cpp_trace_rebuilds_cost_samples_and_partial_history(fake_device):
    oracle = module("benchmarks/experiments/verify_lagged_trace.py", "lagged_actual_trace_oracle")
    report = portable_actual_trace()
    result = oracle.verify(report)
    assert result["matches"] and result["decisions"] > 20 and not result["history_errors"]
    tampered = copy.deepcopy(report)
    replay = next(r["scheduler_replay"] for r in tampered["samples"][0]["result"]["rounds"]
                  if "scheduler_replay" in r)
    replay["cost_model"]["points"][0]["measured_samples"][0]["cycle_ms"] += 1
    with pytest.raises(AssertionError, match="measured input altered"):
        oracle.verify(tampered)


def test_immature_trace_is_unknown_not_a_passing_oracle(fake_device):
    oracle = module("benchmarks/experiments/verify_lagged_trace.py", "lagged_immature_trace_oracle")
    report = portable_actual_trace()
    report["samples"][0]["result"]["rounds"] = report["samples"][0]["result"]["rounds"][:1]
    result = oracle.verify(report)
    assert result["status"] == "insufficient_calibration" and not result["matches"]


@pytest.mark.parametrize("runtime_tuples", [False, True])
def test_actual_9b_lagged_curve_json_and_memory_containers_match_exact_cpp(runtime_tuples):
    oracle = module("benchmarks/experiments/verify_lagged_trace.py", "lagged_real_trace_schema_oracle")
    report_path = ROOT / "benchmarks/reports/phase2-hybrid-lagged-dev-qwen35-9b8.json"
    original_bytes = report_path.read_bytes()
    report = json.loads(original_bytes)
    assert report["summary"]["accepted_frozen_probe_pass"] is False
    curves = [r["scheduler_replay"]["curve"] for sample in report["samples"]
              if sample["variant"] == "ds_lagged_mean" for r in sample["result"]["rounds"]
              if "scheduler_replay" in r]
    if runtime_tuples:
        for curve in curves:
            curve[:] = [tuple(point) for point in curve]
    verified = oracle.verify(report)
    assert verified["matches"] and verified["decisions"] == 28 and not verified["history_errors"]
    assert verified["actual_seqlens_q"] == verified["expected_seqlens_q"]
    # Changing even one representable Python float is still rejected. The fix
    # must normalize pair containers without rounding/tolerating altered values.
    import math

    graph, cost = curves[0][0]
    altered = (graph, math.nextafter(cost, math.inf))
    curves[0][0] = altered if runtime_tuples else list(altered)
    with pytest.raises(AssertionError, match="curve float32 projection"):
        oracle.verify(report)
    assert report_path.read_bytes() == original_bytes
