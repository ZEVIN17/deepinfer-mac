"""CPU-only specification replay, independent of production scheduling code.

References are frozen in phase3_vendor/README.md. The pairwise rank calculation
transcribes the TileLang mathematical rule; a separately expressed stable-sort
specification and hand-calculated JSON fixtures check it. Neither executes CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import struct
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def cumulative_from_conditional(rows, remap, nextn):
    """Match the source's float32 multiply at every prefix position."""
    for row in rows:
        if len(row) != nextn or any(not math.isfinite(x) or not 0 <= x <= 1 for x in row):
            raise ValueError("conditional rows require finite [0, 1] probabilities and fixed nextn")
    result = []
    for previous in remap:
        if previous < -1 or previous >= len(rows):
            raise ValueError("invalid remap")
        acc = 0.0 if previous == -1 else 1.0
        row = []
        for position in range(nextn):
            if previous != -1:
                acc = f32(acc * f32(rows[previous][position]))
            row.append(acc)
        result.append(row)
    return result


def validate_cumulative(rows):
    """An already cumulative provider must bypass cumprod, not call it twice."""
    for row in rows:
        previous = 1.0
        for value in row:
            if not math.isfinite(value) or not 0 <= value <= previous:
                raise ValueError("cumulative probabilities must be finite, [0, 1], and nonincreasing")
            previous = value


def rank_count_reference(cumulative, total_q):
    """Source's all-pairs comparison, including saturation outside legal budget."""
    validate_cumulative(cumulative)
    flattened = [value for row in cumulative for value in row]
    budget = total_q - len(cumulative)
    lengths = []
    offset = 0
    for row in cumulative:
        chosen = 0
        for position, value in enumerate(row):
            index = offset + position
            rank = sum(other > value or (other == value and j < index) for j, other in enumerate(flattened))
            chosen += rank < budget
        lengths.append(1 + chosen)
        offset += len(row)
    return lengths


def sorted_prefix_specification(cumulative, total_q):
    """Independent global ordering; choose positions, then assert prefix closure."""
    entries = [
        (value, row, position) for row, rates in enumerate(cumulative) for position, value in enumerate(rates)
    ]
    ordered = sorted(entries, key=lambda item: (-item[0], item[1], item[2]))
    count = min(len(entries), max(0, total_q - len(cumulative)))
    chosen = {(row, position) for _, row, position in ordered[:count]}
    lengths = []
    for row, rates in enumerate(cumulative):
        positions = sorted(position for index, position in chosen if index == row)
        assert positions == list(range(len(positions))), "selected positions must form a prefix"
        lengths.append(1 + len(positions))
        assert lengths[-1] <= 1 + len(rates)
    return lengths


def validate_budget(num_rows, nextn, total_q):
    if not num_rows <= total_q <= num_rows * (nextn + 1):
        raise ValueError("total_q includes one golden token per active row")


def identity_remap(previous, current):
    """Fixture helper: (request_id, slot_generation), not a production adapter."""
    if len(set(previous)) != len(previous) or len(set(current)) != len(current):
        raise ValueError("duplicate request identity")
    old = {identity: index for index, identity in enumerate(previous)}
    return [old.get(identity, -1) for identity in current]


def observed_labels(nextn, comparisons):
    """Only examined positions are labelled; unexamined suffixes remain censored."""
    if len(comparisons) > nextn or any(type(x) is not bool for x in comparisons):
        raise ValueError("invalid observations")
    if False in comparisons and comparisons.index(False) != len(comparisons) - 1:
        raise ValueError("prefix verification stops at the first rejection")
    return comparisons + [None] * (nextn - len(comparisons))


def source_hashes():
    expected = {
        "phase3_vendor/schedule_mtp_varlen.py.txt": "34f0d7f852dfe552965c40e817a3841be515a219",
        "phase3_vendor/dflash.py.txt": "c8f59022682c1a34956e022b26fa61edae50d934",
        "vendor/mtp_varlen.hpp": "09af680aa42b724fe86315007a160d5012a1fde9",
    }
    entries = []
    for name, expected_blob in expected.items():
        data = (HERE / name).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        assert blob == expected_blob, (name, blob, expected_blob)
        entries.append(
            {
                "path": "tests/reference/" + name,
                "sha256": hashlib.sha256(data).hexdigest(),
                "git_blob_sha": blob,
                "matches_user_source_audit": True,
            }
        )
    license_data = (HERE.parents[1] / "third_party/licenses/DeepInfer.txt").read_bytes()
    license_blob = hashlib.sha1(b"blob " + str(len(license_data)).encode() + b"\0" + license_data).hexdigest()
    assert license_blob == "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64"
    return entries


def run_cpp_oracle():
    """Execute the unchanged host header through the pre-existing test shim."""
    commands = (
        "C 2 2 1 6 1\nQ 2 2 1 0 0 2 0.9 0.09 2 0.9 0.09 "
        "2 1 0 0 2 0.8 0.64 2 0.8 0.64\n"
        "C 2 2 1 6 100\nQ 2 2 1 0 0 2 0.9 0.09 2 0.9 0.09 "
        "2 1 0 0 2 0.8 0.64 2 0.8 0.64\n"
        "C 0\nQ 2 2 1 0 0 0 0 1 1 0 0 0 0\n"
    )
    with tempfile.TemporaryDirectory(prefix="phase3-budget-reference-") as directory:
        binary = str(Path(directory) / "mtp_oracle")
        subprocess.run(
            [
                "c++",
                "-std=c++17",
                "-O2",
                "-I",
                str(HERE / "shim"),
                str(HERE / "mtp_oracle.cpp"),
                "-o",
                binary,
            ],
            check=True,
            capture_output=True,
        )
        completed = subprocess.run([binary], input=commands, text=True, capture_output=True, check=True)
    observed = [[int(x) for x in row.split()] for row in completed.stdout.splitlines()]
    expected = [[3, 3], [1, 1], [3, 2]]
    assert observed == expected, (observed, expected)
    return {
        "executed": True,
        "backend": "unchanged C++ header, existing CPU Sequence shim",
        "row_lengths": observed,
        "total_q_sums": [sum(row) for row in observed],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cpp-oracle", action="store_true")
    args = parser.parse_args()
    fixtures = json.loads((HERE / "phase3_budget_cases.json").read_text())["cases"]
    records = []
    for case in fixtures:
        cumulative = cumulative_from_conditional(case["conditional"], case["remap"], case["nextn"])
        actual = rank_count_reference(cumulative, case["total_q"])
        assert actual == case["lengths"] == sorted_prefix_specification(cumulative, case["total_q"]), case
        valid = case.get("contract_valid", True)
        try:
            validate_budget(len(case["remap"]), case["nextn"], case["total_q"])
        except ValueError:
            assert not valid
        else:
            assert valid and sum(actual) == case["total_q"]
        records.append(
            {"name": case["name"], "cumulative": cumulative, "lengths": actual, "contract_valid": valid}
        )
    exhaustive = 0
    for probabilities in itertools.product([0.0, 0.5, 1.0], repeat=4):
        rows = [list(probabilities[:2]), list(probabilities[2:])]
        for remap in ([0, 1], [1, 0], [-1, 0], [1, -1], [-1, -1]):
            cumulative = cumulative_from_conditional(rows, remap, 2)
            for budget in range(2, 7):
                actual = rank_count_reference(cumulative, budget)
                assert actual == sorted_prefix_specification(cumulative, budget)
                assert sum(actual) == budget
                exhaustive += 1
    cumulative = [[0.5, 0.3], [0.28, 0.0]]
    assert rank_count_reference(cumulative, 4) == [3, 1]
    assert rank_count_reference(cumulative_from_conditional(cumulative, [0, 1], 2), 4) == [2, 2]
    remap = identity_remap([("a", 1), ("b", 3)], [("b", 3), ("a", 2), ("c", 1)])
    assert remap == [1, -1, -1]
    censored = observed_labels(4, [True, True])
    rejected = observed_labels(4, [True, True, False])
    assert censored == [True, True, None, None] and rejected == [True, True, False, None]
    invalid_checks = 0
    for invalid in ([math.nan, 1.0], [math.inf, 0.5], [-0.1, 1.0], [1.1, 0.0]):
        try:
            cumulative_from_conditional([invalid], [0], 2)
        except ValueError:
            invalid_checks += 1
        else:
            raise AssertionError("invalid probability was accepted")
    result = {
        "schema": 1,
        "scope": "independent CPU specification, no production scheduler import",
        "source_commit": "9bd9cf243fc53ef9afbf8f04d391d989117d6ab9",
        "device_oracle_executed": False,
        "metal_executed": False,
        "fixed_cases": records,
        "exhaustive_sort_comparisons": exhaustive,
        "invalid_probability_checks": invalid_checks,
        "identity_remap": remap,
        "censoring": {"budget_or_eos_truncation": censored, "observed_rejection": rejected},
        "double_cumprod_changes_allocation": True,
        "sources": source_hashes(),
        "cpp_oracle": run_cpp_oracle() if args.cpp_oracle else {"executed": False},
        "performance_validated": False,
        "production_integration": False,
    }
    serialized = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(
        json.dumps(
            {
                "fixed_cases": len(records),
                "exhaustive_sort_comparisons": exhaustive,
                "invalid_probability_checks": invalid_checks,
                "cpp_oracle_executed": result["cpp_oracle"]["executed"],
                "device_oracle_executed": False,
                "production_integration": False,
            }
        )
    )


if __name__ == "__main__":
    main()
