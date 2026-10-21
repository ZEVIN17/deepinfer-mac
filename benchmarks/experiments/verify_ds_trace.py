"""Replay a complete single-policy engine trace through the original C++ oracle.

No Python scheduler is imported. This entry point requires a fixed startup curve
and a trace starting with the policy's first cycle. Runtime samples are fed chronologically, and
both empirical acceptance generations are reconstructed from completed rounds.
This checks decision semantics, not performance or production model quality.
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def verify(report_path):
    root = Path(__file__).resolve().parents[2]
    rounds = json.loads(report_path.read_text())["long_request"]["result"]["rounds"]
    max_budget = max(r["budget"] for r in rounds)
    commands, expected, history_errors = [], [], []
    accepts, counts = [0] * max_budget, [0] * max_budget
    previous, current, started, initial_curve = [], [], False, None
    for index, record in enumerate(rounds):
        replay = record.get("scheduler_replay")
        if replay:
            curve, state = replay["curve"], replay["state"]
            if not started:
                commands.append("C " + str(len(curve)) + " " + " ".join(f"{g} {v}" for g, v in curve))
                started = True
                initial_curve = curve
            elif curve != initial_curve:
                raise ValueError("Curve replacement requires a separate oracle instance; split this trace first")
            nd = len(state["draft_tokens"])
            for name, value in [("accept_rates", current[:nd]), ("prev_accept_rates", previous[:nd])]:
                if state[name] != value:
                    history_errors.append({"round": index, "field": name})
            if state["process_cnt"] != index or state["schedule_cnt"] != index:
                history_errors.append({"round": index, "field": "counter"})
            values = [nd, int(state["has_draft_info"]), state["process_cnt"], state["schedule_cnt"],
                      len(state["accept_rates"]), *state["accept_rates"],
                      len(state["prev_accept_rates"]), *state["prev_accept_rates"]]
            commands.append("Q 1 " + " ".join(map(str, values)))
            expected.append(record["budget"] + 1)
        if started and record["produced"] and not record["catchup_tokens"]:
            commands.append(f"R 1 {record['budget'] + 1} {record['cycle_ms']} 0")
        for position in range(record["budget"]):
            accepts[position] += int(position < record["accepted"])
            counts[position] += 1
        previous = current
        current = [(a + 1) / (n + 2) for a, n in zip(accepts, counts)]
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        raise RuntimeError("C++17 compiler required for independent source verification")
    reference = root / "tests/reference"
    with tempfile.TemporaryDirectory(prefix="deepinfer-trace-oracle-") as directory:
        executable = str(Path(directory) / "oracle")
        subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(reference / "shim"),
                        str(reference / "mtp_oracle.cpp"), "-o", executable], check=True)
        result = subprocess.run([executable], input="\n".join(commands) + "\n", text=True,
                                capture_output=True, check=True)
    actual = [int(line) for line in result.stdout.splitlines()]
    if not expected or actual != expected or history_errors:
        raise AssertionError({"actual": actual, "expected": expected, "history_errors": history_errors})
    return {"cycles": len(rounds), "decisions": len(expected), "matches": True,
            "history_errors": history_errors, "actual_seqlens_q": actual,
            "expected_seqlens_q": expected, "method": "unchanged C++ oracle; chronological actual runtime feedback"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("benchmarks/reports/phase2-ds-engine-smoke-qwen3-4b.json"))
    args = parser.parse_args()
    print(json.dumps(verify(args.input), indent=2))
