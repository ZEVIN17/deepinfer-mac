"""Repeat resident ablation with an independent OS process for every mode block.

The earlier same-process run (including its memory preflight failures) remains
published separately. No failed measurement is deleted or replaced.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main():
    variants = [
        "native_mlx",
        "baseline",
        "fixed_speculative:1",
        "fixed_speculative:2",
        "fixed_speculative:4",
        "adaptive",
    ]
    folder = Path("benchmarks/reports/resident-process-blocks")
    folder.mkdir(parents=True, exist_ok=True)
    output = folder.parent / "resident-process-ablation-qwen3-4b.json"
    report = {
        "profile": "qwen3-4b",
        "design": "Every rotated mode block runs in an independent OS process; each prompt warmed before measurement, all errors retained",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "blocks": [],
    }
    for repeat in range(3):
        for variant in variants[repeat:] + variants[:repeat]:
            name = f"{repeat}-{variant.replace(':', '-')}"
            block_path = folder / (name + ".json")
            command = [
                sys.executable,
                "tests/metal_resident_ablation.py",
                "--repeat",
                str(repeat),
                "--variant",
                variant,
                "--output",
                str(block_path),
            ]
            with (folder / (name + ".log")).open("w") as log:
                completed = subprocess.run(command, stdout=log, stderr=log, check=False)
            if block_path.exists():
                block = json.loads(block_path.read_text())
                report["blocks"].append(
                    {"repeat": repeat, "variant": variant, "returncode": completed.returncode, **block}
                )
            else:
                report["blocks"].append(
                    {
                        "repeat": repeat,
                        "variant": variant,
                        "returncode": completed.returncode,
                        "error": "Child exited before report; see sibling log",
                    }
                )
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(name, completed.returncode, flush=True)
    print(output)


if __name__ == "__main__":
    main()
