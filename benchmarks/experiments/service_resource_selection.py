"""Explicit resource admission; deferred sessions remain part of the record."""


def resource_selection(manifest, approved_lengths, total_cap):
    approved = sorted(set(approved_lengths))
    available = {s.get("context_seed_target_tokens") for s in manifest["sessions"]}
    if not approved or not set(approved) <= available:
        raise ValueError("Resource-approved lengths must name calibrated manifest seed lengths")
    protocol = manifest.get("resource_protocol", {})
    if not set(approved) <= set(protocol.get("approved_seed_input_tokens", approved)):
        raise ValueError("Requested length remains deferred in the frozen resource protocol")
    selected, deferred = [], []
    for session in manifest["sessions"]:
        entry = {"id": session["id"], "split": session["split"],
                 "seed_input_tokens": session["context_seed_target_tokens"],
                 "maximum_requested_total_tokens": max(c["requested_total_context_tokens"] for c in session["cases"])}
        if entry["seed_input_tokens"] in approved:
            if entry["maximum_requested_total_tokens"] > total_cap:
                raise ValueError(f"Admitted session {entry['id']} exceeds total cap; never truncate")
            selected.append(entry)
        else:
            deferred.append(entry | {"reason": protocol.get("deferred_reason", "Explicitly outside selected resource phase")})
    warmups = [s for s in manifest.get("warmup_sessions", []) if s["context_seed_target_tokens"] in approved]
    if any(c["requested_total_context_tokens"] > total_cap for s in warmups for c in s["cases"]):
        raise ValueError("Admitted warmup exceeds total cap; never truncate")
    return {"approved_seed_input_tokens": approved, "selected_sessions": selected, "deferred_sessions": deferred,
            "warmup_session_ids": [s["id"] for s in warmups],
            "heldout_session_clusters": sum(s["split"] == "heldout" for s in selected),
            "concurrency": 1, "concurrency_2_4": "unmeasured; project FIFO is not continuous batching"}


def selected_sessions(configuration, manifest, split):
    selection = configuration.get("resource_selection")
    if selection is None:
        return [s for s in manifest["sessions"] if s["split"] == split]
    actual = resource_selection(manifest, selection["approved_seed_input_tokens"], configuration["total_context_cap_tokens"])
    if actual != selection:
        raise ValueError("Resource selection differs from its frozen selected/deferred inventory")
    ids = {s["id"] for s in selection["selected_sessions"]}
    return [s for s in manifest["sessions"] if s["split"] == split and s["id"] in ids]
