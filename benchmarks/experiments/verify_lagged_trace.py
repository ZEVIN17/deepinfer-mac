"""Independently verify lagged-mean trace inputs and original C++ decisions.

Only Python's standard library and the unchanged C++ header are used. This is
mechanism/correctness evidence, never an inference speed or quality benchmark.
"""

import argparse
import hashlib
import json
import math
import shutil
import statistics
import struct
import subprocess
import tempfile
from pathlib import Path


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def verify(report):
    streams = {}
    for sample in report["samples"]:
        if sample["variant"] == "ds_lagged_mean" and "result" in sample:
            key = sample.get("policy_stream", str(sample["result"]["prompt_tokens"] // 256))
            streams.setdefault(key, []).extend(sample["result"]["rounds"])
    max_budget = max(report.get("protocol", {}).get("candidates", [0, 1, 2, 3, 4]))
    minimum = report.get("protocol", {}).get("min_samples", 5)
    window = report.get("protocol", {}).get("window", 9)
    commands, expected, errors = [], [], []
    for stream, rounds in streams.items():
        accepts, counts = [0] * max_budget, [0] * max_budget
        previous, current = [], []
        for index, record in enumerate(rounds):
            replay = record.get("scheduler_replay")
            if replay is not None:
                state, curve, model = replay["state"], replay["curve"], replay["cost_model"]
                nd = len(state["draft_tokens"])
                if state["accept_rates"] != current[:nd] or state["prev_accept_rates"] != previous[:nd]:
                    errors.append({"stream": stream, "cycle": index + 1, "reason": "acceptance generation"})
                if state["process_cnt"] != index or state["schedule_cnt"] != index:
                    errors.append({"stream": stream, "cycle": index + 1, "reason": "counter"})
                if replay["runtime_projection"] != {"forward_slope_ms": 0., "sampler_ms": 0.}:
                    errors.append({"stream": stream, "cycle": index + 1, "reason": "unexpected residual"})
                if model["feedback_cutoff_cycle"] != index - 1:
                    errors.append({"stream": stream, "cycle": index + 1, "reason": "N-2 cutoff"})
                rebuilt = []
                for point in model["points"]:
                    graph = point["graph_size"]
                    eligible = [(i + 1, old) for i, old in enumerate(rounds[:max(0, index - 1)])
                                if old["budget"] == graph - 1 and old["produced"] > old["accepted"]][-window:]
                    samples = point["measured_samples"]
                    if len(eligible) < minimum or [s["cycle"] for s in samples] != [i for i, _ in eligible]:
                        errors.append({"stream": stream, "cycle": index + 1, "reason": "sample eligibility/window"})
                    for sample, (cycle, actual) in zip(samples, eligible):
                        stages = {"snapshot_ms": actual["snapshot_ms"],
                                  "combined_draft_verify_ms": actual["verify_ms"],
                                  "recovery_ms": actual["recovery_ms"]}
                        if (sample["cycle"] != cycle or sample["cycle_ms"] != actual["cycle_ms"]
                                or sample["stage_ms"] != stages or sample["budget"] != actual["budget"]
                                or sample["accepted"] != actual["accepted"] or sample["produced"] != actual["produced"]):
                            errors.append({"stream": stream, "cycle": index + 1, "reason": "measured input altered"})
                    cost = statistics.mean(old["cycle_ms"] for _, old in eligible) if eligible else math.nan
                    if cost != point["estimated_cost_ms"]:
                        errors.append({"stream": stream, "cycle": index + 1, "reason": "arithmetic expected cost"})
                    rebuilt.append([graph, f32(cost)])
                # Runtime curves contain tuple pairs; JSON round-tripping uses
                # lists. Normalize containers only, retaining exact float32 values.
                if rebuilt != [list(point) for point in curve]:
                    errors.append({"stream": stream, "cycle": index + 1, "reason": "curve float32 projection"})
                commands.append("C " + str(len(curve)) + " " + " ".join(f"{g} {v}" for g, v in curve))
                values = [nd, int(state["has_draft_info"]), state["process_cnt"], state["schedule_cnt"],
                          len(state["accept_rates"]), *state["accept_rates"],
                          len(state["prev_accept_rates"]), *state["prev_accept_rates"]]
                commands.append("Q 1 " + " ".join(map(str, values)))
                expected.append(record["budget"] + 1)
            # Censored cycles advance history age but provide no new evidence.
            previous = current
            if record["produced"] > record["accepted"]:
                for position in range(record["budget"]):
                    accepts[position] += int(position < record["accepted"])
                    counts[position] += 1
                current = [(a + 1) / (n + 2) for a, n in zip(accepts, counts)]
    if errors:
        raise AssertionError(errors)
    if not expected:
        return {"matches": False, "status": "insufficient_calibration", "decisions": 0,
                "streams": len(streams), "history_errors": [], "performance_claim": False}
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        raise RuntimeError("A C++17 compiler is required for the independent oracle")
    reference = Path(__file__).resolve().parents[2] / "tests/reference"
    header_sha = hashlib.sha256((reference / "vendor/mtp_varlen.hpp").read_bytes()).hexdigest()
    if header_sha != "164f45db4cccd6d6909dd2846e7dfffb57754b108af679443bf4bd71ec33785a":
        raise ValueError("Original C++ reference bytes changed")
    with tempfile.TemporaryDirectory(prefix="deepinfer-lagged-oracle-") as directory:
        executable = Path(directory) / "oracle"
        subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(reference / "shim"),
                        str(reference / "mtp_oracle.cpp"), "-o", str(executable)], check=True, capture_output=True)
        output = subprocess.run([str(executable)], input="\n".join(commands) + "\n", text=True,
                                capture_output=True, check=True)
    actual = [int(line) for line in output.stdout.splitlines()]
    if actual != expected:
        raise AssertionError({"actual": actual, "expected": expected})
    return {"matches": True, "status": "verified", "decisions": len(expected), "streams": len(streams),
            "history_errors": [], "actual_seqlens_q": actual, "expected_seqlens_q": expected,
            "header_sha256": header_sha, "performance_claim": False,
            "method": "independent chronological full-cost/age reconstruction and unchanged C++ MTPVarlen"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(json.loads(args.input.read_text())), indent=2))


if __name__ == "__main__":
    main()
