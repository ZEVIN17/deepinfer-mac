"""CPU-only preregistration: preserve v2 and add two independent heldout ledgers.

No model forward, generated answers, or performance data are used. Existing
v2 sessions are retained byte-equivalent as JSON objects; only new independent
record seeds 9/10 are calibrated. The first dev inputs never change.
"""

import argparse
import hashlib
import json
import struct
from pathlib import Path

from build_cache_manifest import build

V2_SHA256 = "f891ff522edc120731781f5a0e8f3b211774b4dfea696af1690ec8f4c5d81ad5"


def token_ids(tokenizer, case):
    return tokenizer.apply_chat_template(
        case["messages"], tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_dict=False,
    )


def supplemental_session(tokenizer, record_index, target):
    def make(rows):
        return build(2, rows_by_session=[rows], record_indices=[record_index], context_targets=[target])["sessions"][0]

    low, high = 8, target // 8
    while low < high:
        middle = (low + high + 1) // 2
        if len(token_ids(tokenizer, make(middle)["cases"][0])) <= target:
            low = middle
        else:
            high = middle - 1
    session = make(low)
    system = session["cases"][0]["messages"][0]
    content = system["content"]
    for padding in range(64):
        system["content"] = content + " x" * padding
        if len(token_ids(tokenizer, session["cases"][0])) == target:
            break
    else:
        raise ValueError(f"Cannot calibrate {session['id']} to {target} input tokens")
    session.update(context_seed_target_tokens=target, seed_padding_x_tokens=padding)
    for case in session["cases"]:
        case["prompt_tokens"] = len(token_ids(tokenizer, case))
        case["requested_total_context_tokens"] = case["prompt_tokens"] + case["max_tokens"]
        case["initial_4096_total_phase_allowed"] = case["requested_total_context_tokens"] <= 4096
        case["current_8192_profile_allowed"] = case["requested_total_context_tokens"] <= 8192
    return session


def freeze(tokenizer, v2):
    # Preserve every original session, including deferred 8K and all dev cases.
    result = json.loads(json.dumps(v2))
    result["schema_version"] = 3
    result["sessions"] += [supplemental_session(tokenizer, 9, 2048), supplemental_session(tokenizer, 10, 4096)]
    result["parent_manifest"] = {"path": "benchmarks/experiments/cache_sessions.v2.json", "sha256": V2_SHA256}
    result["revision_reason"] = (
        "Before any v2/v3 heldout GPU output was observed, add one independently seeded heldout ledger "
        "to each admitted 2K/4K group. This restores six session clusters while 8K remains deferred "
        "after the 8300+256 input probe increased swap by about 2.60 GiB. Keep all v2 sessions, "
        "dev inputs, strict scoring, 15% target, six-cluster threshold, and failures unchanged."
    )
    result["resource_protocol"] = {
        "approved_seed_input_tokens": [2048, 4096],
        "deferred_seed_input_tokens": [8192],
        "deferred_reason": "8300+256 resource probe increased swap; no formal 8K performance run",
        "evidence": "benchmarks/reports/context-8300-256-qwen35-9b8.json",
        "concurrency": 1,
        "concurrency_2_4_status": "deferred, requires separate real admission and measurement",
        "independent_heldout_sessions_admitted": 6,
        "statistics_scope": "Six different ledgers across 2K+4K; each length alone has three, not six clusters.",
    }
    result["warmup_sessions"] = [supplemental_session(tokenizer, 11, 2048), supplemental_session(tokenizer, 12, 4096)]
    for session in result["warmup_sessions"]:
        session["split"] = "excluded_warmup"
        session["note"] = "Independent ledger; exercise each admitted dev length and request kind per engine/variant. Never scored; cache is not asserted empty afterward."
    result["comparison_protocol"]["warmup"] = "Separate 2K and 4K public ledgers, all four request kinds, excluded before every engine/variant block"
    result["v3_token_calibration"] = {}
    for session in result["sessions"] + result["warmup_sessions"]:
        result["v3_token_calibration"][session["id"]] = []
        for case in session["cases"]:
            ids = token_ids(tokenizer, case)
            if len(ids) != case["prompt_tokens"]:
                raise ValueError(f"Existing calibration changed: {session['id']} {case['kind']}")
            result["v3_token_calibration"][session["id"]].append({
                "kind": case["kind"], "prompt_tokens": len(ids),
                "token_ids_sha256_le_uint32": hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest(),
            })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/experiments/cache_sessions.v3.json"))
    args = parser.parse_args()
    raw = Path("benchmarks/experiments/cache_sessions.v2.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != V2_SHA256:
        raise ValueError("Frozen v2 bytes changed")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True, trust_remote_code=False)
    v2 = json.loads(raw)
    calibration = v2["tokenizer_calibration"]
    if args.snapshot.name != calibration["revision"] or (
        hashlib.sha256((args.snapshot / "tokenizer.json").read_bytes()).hexdigest() != calibration["tokenizer_json_sha256"]
        or hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != calibration["chat_template_sha256"]
    ):
        raise ValueError("Tokenizer/revision/template differ from v2")
    result = freeze(tokenizer, v2)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"manifest": str(args.output), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      "sessions": len(result["sessions"]), "approved_heldout_clusters": 6}))


if __name__ == "__main__":
    main()
