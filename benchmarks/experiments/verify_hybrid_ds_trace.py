"""Verify the 9B hybrid production trace with the unchanged original C++ oracle.

The generic verifier accepts a first-cycle, single-policy trace. Hybrid records
raw pending inputs, while the policy excludes the natural one-input full-accept
lag from catch-up classification. Adapt only that logging boundary here; no
Python scheduler is called and every runtime/acceptance observation is retained.
"""

import argparse
import copy
import json
import tempfile
from pathlib import Path

from verify_ds_trace import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(
        "benchmarks/reports/phase2-hybrid-ds-qwen35-9b8-probe.json"))
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    sample = next(s for s in report["samples"] if s["variant"] == "ds")
    rounds = copy.deepcopy(sample["result"]["rounds"])
    for record in rounds:
        if record.get("recovery") != "exact snapshot restore + consumed-prefix replay":
            raise ValueError("Expected hybrid raw-pending timing records")
        record["catchup_tokens"] = max(0, record["catchup_tokens"] - 1)
    with tempfile.TemporaryDirectory(prefix="deepinfer-hybrid-trace-") as directory:
        converted = Path(directory) / "trace.json"
        converted.write_text(json.dumps({"long_request": {"result": {"rounds": rounds}}}))
        result = verify(converted)
    result["source_report"] = str(args.input)
    result["catchup_boundary"] = "raw pending length minus natural one-input draft lag"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
