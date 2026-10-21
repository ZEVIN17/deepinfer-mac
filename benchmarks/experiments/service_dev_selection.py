"""Choose dev candidates with complete strict quality/resource/source evidence.

One complete dev repetition can choose an exploratory candidate; it cannot meet
the heldout engineering target or balanced-order requirement. Failed candidates
remain listed, and no heldout result enters selection.
"""

import statistics

from service_inventory import audit


def select(raw):
    if raw.get("split") != "dev" or not raw.get("configuration", {}).get("dev_tuning_grid"):
        return {"status": "not_a_dev_grid", "selected_engine_ids": [], "candidates": []}
    inventory = audit(raw)
    allowed = {"actual configured rotation does not cover balanced positions"}
    if raw.get("status") == "completed_with_failures":
        allowed.add("run did not complete")
    global_errors = [error for error in inventory["errors"] if not error.startswith("block ") and error not in allowed]
    candidates = []
    for engine in raw["configuration"]["engines"]:
        selected = [(index, block) for index, block in enumerate(raw["blocks"]) if block["engine"] == engine["id"]]
        reasons = list(global_errors)
        indices = {index for index, _ in selected}
        reasons += [error for error in inventory["errors"] if any(error.startswith(f"block {index}:") for index in indices)]
        if len(selected) != raw["planned_repetitions"]:
            reasons.append("Incomplete candidate repetitions")
        records = [record for _, block in selected for record in block["records"]]
        warmups = [record for _, block in selected for record in block.get("warmups", [])]
        if not records or not all(record.get("quality_pass") is True for record in [*records, *warmups]):
            reasons.append("Strict scored or warmup quality failed")
        times = [record["total_ms"] for record in records]
        ttft = [record["ttft_text_ms"] for record in records]
        candidates.append({"id": engine["id"], "tuning_group": engine.get("tuning_group", engine.get("tuning_parent", engine["id"])),
                           "eligible": not reasons, "reasons": reasons, "requests": len(records),
                           "mean_task_total_ms": statistics.mean(times) if times and not reasons else None,
                           "mean_text_ttft_ms": statistics.mean(ttft) if ttft and not reasons else None,
                           "dflash_execution_required": None if not engine.get("dflash_execution_required") else {
                               key: engine["dflash_execution_required"].get(key) for key in ("model_id", "block_size", "verify_mode")},
                           "prefill_step_size": engine.get("prefill_step_size")})
    winners = []
    for group in sorted({candidate["tuning_group"] for candidate in candidates}):
        valid = [candidate for candidate in candidates if candidate["tuning_group"] == group and candidate["eligible"]]
        if valid:
            winners.append(min(valid, key=lambda candidate: (candidate["mean_task_total_ms"], candidate["mean_text_ttft_ms"], candidate["id"])))
    return {"status": "dev_selection_only_no_competitive_claim", "global_errors": global_errors,
            "selected_engine_ids": [candidate["id"] for candidate in winners],
            "selected_candidates": winners, "candidates": candidates,
            "rule": "Complete planned dev inventory; reject strict scored/warmup quality, protocol, source, resource or execution failures; independently minimize mean dev task total, TTFT tie-break per group. One complete repetition is exploratory and never a speedup/heldout claim."}


def candidate_identity(engine):
    from pathlib import Path

    draft = engine.get("additional_draft") or {}
    execution = engine.get("dflash_execution_required") or {}
    command = engine.get("command", [])
    options, skip = [], False
    for item in command[2:]:
        if skip:
            skip = False
        elif item in ("--model", "--model-dir", "--base-path", "--port"):
            skip = True
        else:
            options.append(item)
    return {key: engine.get(key) for key in ("weights_repo", "weights_revision", "quantization", "cache_budget_bytes",
             "enable_thinking", "temperature", "concurrency", "prefill_step_size", "message_checkpoints", "request_options",
             "hybrid_cost_adapter", "round_diagnostics")} | {
        "memory_guard": engine.get("memory_guard"),
        "additional_draft": {key: draft.get(key) for key in ("repo", "revision", "weight_bytes")},
        "dflash": {key: execution.get(key) for key in ("model_id", "block_size", "verify_mode")},
        "service_script": Path(command[1]).name if len(command) > 1 else None,
        "service_options": options,
    }


def context_identity(configuration):
    return {key: configuration.get(key) for key in ("resource_selection", "total_context_cap_tokens", "resource_policy")}


def bind(raw, summary, formal_engines, manifest, source_sha256, environments=None, bridge_note=None, formal_context=None):
    """Recompute actual dev selection and bind every formal variant to its winner."""
    from service_source_freeze import digest

    actual = select(raw)
    if actual["status"] != "dev_selection_only_no_competitive_claim" or actual.get("global_errors"):
        raise ValueError("Dev source/protocol inventory is not eligible for selection")
    if summary.get("dev_candidate_selection") != actual:
        raise ValueError("Summary differs from recomputed actual dev selection")
    if digest(raw["manifest"]) != digest(manifest):
        raise ValueError("Dev and formal manifests differ")
    if formal_context is None or context_identity(raw["configuration"]) != context_identity(formal_context):
        raise ValueError("Dev/formal approved lengths, context cap or resource policy differ")
    configured = {engine["id"]: engine for engine in raw["configuration"]["engines"]}
    winners = {candidate["tuning_group"]: candidate for candidate in actual["selected_candidates"]}
    chosen = []
    for engine in formal_engines:
        group = engine.get("tuning_group", engine.get("tuning_parent", engine["id"]))
        winner = winners.get(group)
        if winner is None or candidate_identity(engine) != candidate_identity(configured[winner["id"]]):
            raise ValueError(f"Formal candidate does not match actual dev winner for {group}")
        chosen.append({"formal_engine_id": engine["id"], "dev_engine_id": winner["id"],
                       "tuning_group": group, "identity": candidate_identity(engine)})
    previous = raw["configuration"]["frozen_source"]["sha256"]
    if previous != source_sha256:
        raise ValueError("Source changed after dev; retune on the identical formal source, a note cannot validate behavioral drift")
    if environments is not None and raw["configuration"].get("interpreter_environments") != environments:
        raise ValueError("Interpreter environment bytes changed after dev; retune on the identical environment")
    return {"verified": True, "selection": actual, "formal_bindings": chosen,
            "manifest_sha256": digest(manifest), "context_identity": context_identity(formal_context),
            "interpreter_environments_sha256": digest(sorted(value["sha256"] for value in (environments or {}).values())),
            "source_bridge": {"dev_source_sha256": previous, "formal_source_sha256": source_sha256,
                              "same_source": previous == source_sha256, "review_note": bridge_note}}


def verify_formal(configuration):
    """Reload immutable evidence, rather than trusting copied selection IDs."""
    import hashlib
    import json
    from pathlib import Path

    evidence = configuration.get("dev_selection_report") or {}
    if evidence.get("verified") is not True or not evidence.get("raw_path") or not evidence.get("path"):
        raise ValueError("Actual dev winner evidence is required for every formal tuning group")
    raw_path, summary_path = Path(evidence["raw_path"]), Path(evidence["path"])
    if hashlib.sha256(raw_path.read_bytes()).hexdigest() != evidence["raw_sha256"] or hashlib.sha256(summary_path.read_bytes()).hexdigest() != evidence["sha256"]:
        raise ValueError("Immutable dev selection/raw evidence drifted")
    raw, summary = json.loads(raw_path.read_text()), json.loads(summary_path.read_text())
    if summary.get("raw_report_sha256") != evidence["raw_sha256"]:
        raise ValueError("Dev summary does not name the bound actual raw bytes")
    observed = bind(raw, summary, configuration["engines"], json.loads(Path(configuration["frozen_manifest"]).read_text()),
                    configuration["frozen_source"]["sha256"], configuration["interpreter_environments"],
                    evidence.get("source_bridge", {}).get("review_note"), configuration)
    if any(observed[key] != evidence.get(key) for key in observed):
        raise ValueError("Formal choices differ from the recomputed actual dev winners")
    return observed


def portable_formal_evidence(configuration, observed):
    """Validate path-free coherence without reopening private execution evidence."""
    from service_source_freeze import digest

    evidence = configuration.get("dev_selection_report") or {}
    if (evidence.get("verified") is not True or observed.get("verified") is not True
            or observed.get("binding_sha256") != digest(evidence)
            or not all(isinstance(evidence.get(key), str) and len(evidence[key]) == 64 for key in ("sha256", "raw_sha256"))):
        return False
    if (evidence.get("manifest_sha256") != configuration.get("manifest_canonical_sha256")
            or evidence.get("context_identity") != context_identity(configuration)
            or evidence.get("interpreter_environments_sha256") != digest(sorted(value["sha256"] for value in configuration.get("interpreter_environments", {}).values()))):
        return False
    bridge = evidence.get("source_bridge", {})
    if bridge.get("same_source") is not True or bridge.get("dev_source_sha256") != configuration["frozen_source"]["sha256"] or bridge.get("formal_source_sha256") != configuration["frozen_source"]["sha256"]:
        return False
    selection = evidence.get("selection") or {}
    if selection.get("status") != "dev_selection_only_no_competitive_claim" or selection.get("global_errors"):
        return False
    winners = {winner["tuning_group"]: winner for winner in selection.get("selected_candidates", [])}
    bindings = evidence.get("formal_bindings", [])
    if len(bindings) != len(configuration["engines"]):
        return False
    for engine, binding in zip(configuration["engines"], bindings, strict=True):
        group = engine.get("tuning_group", engine.get("tuning_parent", engine["id"]))
        winner = winners.get(group)
        if (not winner or not winner.get("eligible") or winner.get("reasons")
                or binding.get("formal_engine_id") != engine["id"] or binding.get("tuning_group") != group
                or binding.get("dev_engine_id") != winner["id"] or binding.get("identity") != candidate_identity(engine)):
            return False
    return True
