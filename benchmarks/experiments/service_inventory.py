"""Validate the complete planned experiment, rather than observed good subsets."""

import math

from http_service_compare import request_body
from service_resource_selection import selected_sessions
from service_resources import eligibility as resource_eligibility
from service_source_freeze import digest


def warmup_cases(configuration, manifest):
    ids = configuration.get("resource_selection", {}).get("warmup_session_ids")
    sessions = [s for s in manifest.get("warmup_sessions", []) if ids is None or s["id"] in ids]
    return [case for session in sessions for case in session["cases"]] or [selected_sessions(configuration, manifest, "dev")[0]["cases"][0]]


def valid_metric(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def source_evidence(configuration, observed):
    frozen = configuration.get("frozen_source", {})
    environments = configuration.get("interpreter_environments", {})
    if not frozen.get("files") or digest(frozen["files"]) != frozen.get("sha256") or not environments:
        return False
    if observed.get("verified") is not True or observed.get("execution_binding_verified") is not True:
        return False
    if observed.get("source_sha256") != frozen["sha256"] or observed.get("archive_sha256") != frozen.get("archive_sha256"):
        return False
    actual = observed.get("environments", {})
    if set(actual) != set(environments):
        return False
    for python, expected in environments.items():
        contents = {key: value for key, value in expected.items() if key != "sha256"}
        if digest(contents) != expected.get("sha256") or actual[python].get("sha256") != expected["sha256"]:
            return False
    return True


def audit(raw):
    errors = []
    try:
        configuration, manifest = raw["configuration"], raw["manifest"]
        split, repetitions = raw["split"], raw["planned_repetitions"]
        engines = configuration["engines"]
        ids = [e["id"] for e in engines]
        if len(ids) < 2 or len(set(ids)) != len(ids) or type(repetitions) is not int or repetitions < 1:
            raise ValueError("Invalid engine/repetition inventory")
        if split not in ("dev", "heldout"):
            raise ValueError("Missing/invalid selected split")
        sessions = selected_sessions(configuration, manifest, split)
        expected_cases = {(s["id"], c["kind"]): c for s in sessions for c in s["cases"]}
        if not expected_cases or len(expected_cases) != sum(len(s["cases"]) for s in sessions):
            raise ValueError("Duplicate/empty manifest session/kind inventory")
        if configuration.get("manifest_canonical_sha256") != digest(manifest):
            errors.append("manifest canonical bytes differ from preparation")
        if raw.get("configuration_sha256") != digest(configuration):
            errors.append("configuration differs from execution snapshot")
        expected_blocks = [(engine, repetition) for repetition in range(repetitions)
                           for engine in ids[repetition % len(ids):] + ids[:repetition % len(ids)]]
        actual_blocks = [(b.get("engine"), b.get("repetition")) for b in raw.get("blocks", [])]
        if actual_blocks != expected_blocks:
            errors.append("missing/extra/duplicate blocks or actual engine order mismatch")
        if repetitions % len(ids) != 0 or raw.get("order_balance", {}).get("fully_balanced_positions") is not True:
            errors.append("actual configured rotation does not cover balanced positions")
        by_id = {e["id"]: e for e in engines}
        warms = warmup_cases(configuration, manifest)
        for index, block in enumerate(raw.get("blocks", [])):
            engine = by_id.get(block.get("engine"))
            if engine is None:
                errors.append(f"block {index}: unknown engine")
                continue
            if block.get("status") != "completed" or block.get("error") is not None or block.get("process_stopped") is not True:
                errors.append(f"block {index}: failed/incomplete/process not stopped")
            if block.get("executed_command") != engine.get("command") or block.get("execution_cwd") != engine.get("cwd"):
                errors.append(f"block {index}: actual execution command/cwd mismatch")
            seen = set()
            actual_keys = []
            for record in block.get("records", []):
                key = (record.get("session_id"), record.get("case", {}).get("kind"))
                actual_keys.append(key)
                if key in seen:
                    errors.append(f"block {index}: duplicate request key {key}")
                seen.add(key)
                case = expected_cases.get(key)
                if case is None or record.get("case") != case or record.get("split") != split or record.get("repetition") != block.get("repetition"):
                    errors.append(f"block {index}: unexpected/mutated request inventory")
                elif record.get("request") != request_body(engine["model"], case, engine.get("request_options")):
                    errors.append(f"block {index}: actual request differs from planned payload")
                if record.get("error") is not None or record.get("protocol_complete") is not True:
                    errors.append(f"block {index}: incomplete SSE/request failure")
                if not all(valid_metric(record.get(metric)) for metric in ("total_ms", "ttft_text_ms")):
                    errors.append(f"block {index}: missing/nonfinite timing metrics")
            if seen != set(expected_cases):
                errors.append(f"block {index}: missing/extra session/kind requests")
            if actual_keys != list(expected_cases):
                errors.append(f"block {index}: actual request order differs from frozen sequence")
            observed_warmups = block.get("warmups", [])
            if len(observed_warmups) != len(warms):
                errors.append(f"block {index}: warmup inventory incomplete")
            for case, observed in zip(warms, observed_warmups, strict=False):
                if observed.get("case") != case or observed.get("request") != request_body(engine["model"], case, engine.get("request_options")):
                    errors.append(f"block {index}: warmup case/payload mismatch")
                if observed.get("error") is not None or observed.get("protocol_complete") is not True:
                    errors.append(f"block {index}: warmup protocol failure")
            checks = block.get("source_checks", {})
            if any(not source_evidence(configuration, checks.get(phase, {})) for phase in ("before", "after")):
                errors.append(f"block {index}: missing/failed per-block source checks")
            if configuration.get("resource_policy"):
                resource = resource_eligibility(block.get("resource_samples", {}), configuration["resource_policy"], len(block.get("records", [])))
                if resource != block.get("resource_eligibility") or resource["valid"] is not True:
                    errors.append(f"block {index}: resource eligibility failed or differs from recorded samples")
            if engine.get("dflash_execution_required") and any(
                block.get(key, {}).get("valid") is not True for key in ("dflash_warmup_execution", "dflash_execution")
            ):
                errors.append(f"block {index}: actual official DFlash execution not proven")
        if any(not source_evidence(configuration, raw.get("source_checks", {}).get(phase, {})) for phase in ("initial", "final")):
            errors.append("missing/failed initial/final source checks")
        if raw.get("status") != "completed":
            errors.append("run did not complete")
        if split == "heldout" and configuration.get("resource_policy"):
            from service_dev_selection import portable_formal_evidence

            if not portable_formal_evidence(configuration, raw.get("dev_selection_check") or {}):
                errors.append("missing/incoherent actual dev winner binding for formal report")
        return {"valid": not errors, "errors": errors, "expected_blocks": len(expected_blocks),
                "expected_requests_per_block": len(expected_cases), "expected_warmups_per_block": len(warms)}
    except (KeyError, TypeError, ValueError, IndexError) as error:
        return {"valid": False, "errors": errors + [f"Cannot establish frozen inventory: {error}"]}
