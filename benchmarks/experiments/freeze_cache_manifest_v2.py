"""CPU-only exact tokenizer calibration before any v2 GPU/heldout run."""

import argparse
import hashlib
import json
from pathlib import Path

from build_cache_manifest import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True, trust_remote_code=False)

    def count(case):
        return len(
            tokenizer.apply_chat_template(
                case["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False
            )
        )

    rows = [80, 80, 80, 160, 160, 160, 320, 320, 320]
    sessions = []
    for index in range(9):
        target = (2048, 4096, 8192)[index // 3]
        low, high = 8, target // 8
        while low < high:
            middle = (low + high + 1) // 2
            trial = rows.copy()
            trial[index] = middle
            case = build(2, rows_by_session=trial)["sessions"][index]["cases"][0]
            if count(case) <= target:
                low = middle
            else:
                high = middle - 1
        rows[index] = low
        session = build(2, rows_by_session=rows)["sessions"][index]
        system = session["cases"][0]["messages"][0]
        content = system["content"]
        for padding in range(64):
            system["content"] = content + " x" * padding
            if count(session["cases"][0]) == target:
                break
        else:
            raise RuntimeError(f"Could not calibrate exact input length for session {index}")
        session["context_seed_target_tokens"] = target
        session["context_target_tokens"] = target
        session["seed_padding_x_tokens"] = padding
        for case in session["cases"]:
            case["prompt_tokens"] = count(case)
            case["requested_total_context_tokens"] = case["prompt_tokens"] + case["max_tokens"]
            case["initial_4096_total_phase_allowed"] = case["requested_total_context_tokens"] <= 4096
            case["current_8192_profile_allowed"] = case["requested_total_context_tokens"] <= 8192
        sessions.append(session)
    result = build(2)
    result["sessions"] = sessions
    result["context_length_note"] = (
        "Seed inputs exactly calibrated to 2048/4096/8192; continuation input counts are explicit and can exceed the seed. No truncation. Output budget 256 is additional."
    )
    result["revision_reason"] = (
        "Only v1 dev showed identical Markdown fences in all three engines. v2 uniformly prohibits fences; scoring remains strict JSON. No heldout outputs inspected for this change."
    )
    result["tokenizer_calibration"] = {
        "repo": "mlx-community/Qwen3.5-9B-8bit",
        "revision": args.snapshot.name,
        "thinking": False,
        "add_generation_prompt": True,
        "tokenizer_json_sha256": hashlib.sha256((args.snapshot / "tokenizer.json").read_bytes()).hexdigest(),
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "source_level": "actual local tokenizer, CPU only, no model forward or performance data",
        "initial_total_context_cap_tokens": 4096,
        "current_profile_total_cap_tokens": 8192,
        "long_input_status": "deferred until real resource probe and explicit profile cap adjustment; never truncate",
    }
    target = Path("benchmarks/experiments/cache_sessions.v2.json")
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    summary = {
        "manifest_sha256": digest,
        "sessions": [
            {
                "id": s["id"],
                "split": s["split"],
                "seed_target": s["context_seed_target_tokens"],
                "actual_prompt_tokens": [c["prompt_tokens"] for c in s["cases"]],
                "requested_total_tokens": [c["requested_total_context_tokens"] for c in s["cases"]],
            }
            for s in sessions
        ],
    }
    Path("artifacts/phase2-competitor-audit/cache-manifest-v2-calibration.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
