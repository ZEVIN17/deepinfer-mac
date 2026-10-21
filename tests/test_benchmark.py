import dataclasses
import json

from deepinfer_mac.benchmark.export import export_report
from deepinfer_mac.benchmark.metrics import aggregate, paired_comparison
from deepinfer_mac.benchmark.runner import run_suite
from deepinfer_mac.state import GenerationRequest


class Engine:
    def clear_cache(self):
        pass

    def generate(self, request, emit=None, cancel=None):
        assert request.use_cache is False
        if request.mode == "fixed_speculative" and request.num_draft_tokens == 4:
            raise ValueError("retained failure")
        return {"mode": request.mode, "token_ids": [1, 2], "metrics": {"total_ms": 12}, "rounds": []}

    def native_generate(self, request, emit=None, cancel=None):
        return self.generate(dataclasses.replace(request, mode="native_mlx"), emit, cancel)


def test_failures_retained_rotation_and_no_false_cold(tmp_path):
    report = run_suite(Engine(), GenerationRequest(prompt="replaced"), repeats=2)
    assert len(report["samples"]) == 12
    assert {sample["phase"] for sample in report["samples"]} == {"warm", "warmup"}
    assert len({tuple(sample["order"]) for sample in report["samples"]}) > 1
    assert report["warm_summary"]["failures"]["fixed_speculative:4"] == 8
    assert report["warm_summary"]["equal_token_samples"] == 0
    export_report(report, tmp_path)
    assert json.loads((tmp_path / "summary.json").read_text())["schema_version"] == 2
    assert "retained failure" in (tmp_path / "results.csv").read_text()
    assert (tmp_path / "report.md").is_file()


def test_mismatch_not_speedup_and_all_successes_aggregated():
    results = [{"mode": "baseline", "token_ids": [1], "metrics": {"total_ms": 10}},
               {"mode": "adaptive", "token_ids": [2], "metrics": {"total_ms": 5}}]
    comparison = paired_comparison(results)
    assert comparison["token_ids_equal"] is False
    assert comparison["speedups"] == {}
    values = aggregate([{"phase": "warm", "results": results, "comparison": comparison}])
    assert values["durations_all_successful_runs"]["adaptive"]["n"] == 1
    assert values["equal_token_samples"] == 0
    assert values["all_paired_time_ratios_work_may_differ"]["baseline"]["median"] == 2
    assert values["equivalent_speedup_claim_supported"] is False


def test_successful_pair_statistics():
    results = [{"mode": "baseline", "token_ids": [1, 2], "metrics": {"total_ms": 10}},
               {"mode": "adaptive", "token_ids": [1, 2], "metrics": {"total_ms": 5}}]
    pair = paired_comparison(results)
    assert pair["speedups"]["baseline"] == 2
    summary = aggregate([{"phase": "warm", "results": results, "comparison": pair}] * 3)
    assert summary["adaptive_speedup_on_equal_pairs_diagnostic"]["baseline"]["bootstrap_mean_95_percent_interval"] == [2, 2]
