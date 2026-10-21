"""Freeze public synthetic *inputs* for realistic session cache controls.

No performance or model outputs are synthesized. Canonical assistant turns
keep continuation inputs identical across engines, including after failures.
"""

import argparse
import json
from pathlib import Path


def build(version=1, rows_by_session=None, record_indices=None, context_targets=None):
    if version not in (1, 2):
        raise ValueError("Unsupported manifest version")
    sessions = []
    for position, rows in enumerate(rows_by_session or (80, 80, 80, 160, 160, 160, 320, 320, 320)):
        index = record_indices[position] if record_indices is not None else position
        language = "zh" if index % 2 else "en"
        table = "\n".join(
            f"P{i:04d}: cost={100 + (i * 37 + index) % 900}; owner=R{(i + index) % 17:02d}."
            for i in range(rows)
        )
        selected = [3, rows // 3, rows // 2, rows - 2]

        def expected(ids, index=index):
            return {
                f"P{i:04d}": {"cost": 100 + (i * 37 + index) % 900, "owner": f"R{(i + index) % 17:02d}"}
                for i in ids
            }

        instruction = (
            "根据给出的公开项目表提取记录。仅返回JSON对象，项目ID为键，值包含cost整数和owner字符串。"
            if language == "zh"
            else "Extract records from the public project table. Return only a JSON object keyed by project ID, "
            "with cost (integer) and owner (string) in each value."
        )
        if version == 2:
            instruction += (
                " 首字符必须为{，末字符必须为}。禁止Markdown、代码围栏、前言或结语；直接输出JSON。"
                if language == "zh"
                else " The first character must be { and the last character must be }. "
                "Do not use Markdown, code fences, a preamble, or an epilogue. Output JSON directly."
            )
        system = {"role": "system", "content": instruction + "\n" + table}
        question = {"role": "user", "content": "Records: " + ", ".join(f"P{i:04d}" for i in selected)}
        messages = [system, question]
        cases = []
        for kind in ("seed_cold_kv", "exact_repeat_hot_kv"):
            cases.append(
                {"kind": kind, "messages": messages, "max_tokens": 256, "expected_json": expected(selected)}
            )
        branch_ids = [4, rows // 3 + 1, rows // 2 + 1, rows - 3]
        branch = {"role": "user", "content": "Records: " + ", ".join(f"P{i:04d}" for i in branch_ids)}
        cases.append(
            {
                "kind": "changed_tail_branch",
                "messages": [system, branch],
                "max_tokens": 256,
                "expected_json": expected(branch_ids),
            }
        )
        continuation = messages + [
            {"role": "assistant", "content": json.dumps(expected(selected), separators=(",", ":"))},
            {"role": "user", "content": "Records: " + ", ".join(f"P{i:04d}" for i in branch_ids)},
        ]
        cases.append(
            {
                "kind": "multiturn_continuation",
                "messages": continuation,
                "max_tokens": 256,
                "expected_json": expected(branch_ids),
            }
        )
        sessions.append(
            {
                "id": f"public-ledger-{index:02d}-{language}",
                "split": "dev" if index in (0, 3, 6) else "heldout",
                "context_rows": rows,
                "context_target_tokens": context_targets[position] if context_targets is not None else (2000, 4000, 8000)[index // 3],
                "language": language,
                "cases": cases,
            }
        )
    return {
        "schema_version": version,
        "license": "Apache-2.0; project-authored input data",
        "context_length_note": "Targets only: actual tokenizer prompt counts must be recorded from servers.",
        "quality_scope": "Strict four-record factual JSON extraction; complements independent code/instruction suites.",
        "comparison_protocol": {
            "cache_budget_bytes": 536870912,
            "temperature": 0,
            "enable_thinking": False,
            "concurrency": 1,
            "primary_metric": "paired quality-passing total_ms and ttft_text_ms by session",
            "engineering_target_latency_reduction": 0.15,
            "statistics": "bootstrap session clusters, never individual repetitions",
            "engine_order": "rotate complete engine blocks per repetition; exclusive GPU",
            "warmup": "one excluded public dev session per engine before each block",
            "reset": "restart engines or explicitly clear all KV between blocks",
        },
        "sessions": sessions,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=(1, 2), type=int, default=1)
    args = parser.parse_args()
    target = Path(f"benchmarks/experiments/cache_sessions.v{args.version}.json")
    target.write_text(json.dumps(build(args.version), ensure_ascii=False, indent=2) + "\n")
    print(target)
