"""Outside-timing system/process samples; RSS and Metal are never added."""

RESOURCE_POLICY = {"id": "warm-scored-no-new-swap-v1", "max_scored_swap_growth_bytes": 0,
                   "min_available_bytes": 1024**3,
                   "scope": "Load/warmup swap is reported separately. Scored-window growth or available memory below1GiB makes a block ineligible for the engineering target; retain all timings/failures. System swap changes are not attributed exclusively to the server."}


def sample(pid=None):
    import psutil

    result = {"system_available_bytes": None, "system_swap_used_bytes": None,
              "process_rss_bytes": None, "process_tree_rss_bytes": None,
              "metal_active_bytes": None, "metal_peak_bytes": None,
              "metal_note": "Unavailable from outside the official process without adding diagnostic hooks; use separately labeled correctness/resource probes. RSS is not Metal allocation."}
    try:
        result["system_available_bytes"] = psutil.virtual_memory().available
        result["system_swap_used_bytes"] = psutil.swap_memory().used
    except OSError as error:
        result["system_sample_error"] = type(error).__name__
    if pid is not None:
        try:
            process = psutil.Process(pid)
            result["process_rss_bytes"] = process.memory_info().rss
            result["process_tree_rss_bytes"] = sum(p.memory_info().rss for p in [process, *process.children(recursive=True)] if p.is_running())
        except psutil.Error as error:
            result["process_sample_error"] = type(error).__name__
    return result


def eligibility(samples, policy, expected_requests=None):
    start = samples.get("after_warmup")
    scored = samples.get("after_requests", [])
    errors = []
    lifecycle = ("before_start", "after_ready", "after_warmup", "before_stop", "after_stop")
    if expected_requests is not None and (len(scored) != expected_requests or any(not samples.get(key) for key in lifecycle)):
        return {"valid": False, "errors": ["Missing/extra per-request samples or incomplete resource lifecycle"]}
    if not start or not scored:
        return {"valid": False, "errors": ["Missing warm-scored resource samples"]}
    observed = [start, *scored]
    validated = [*observed, *(samples[key] for key in lifecycle if key in samples)]
    if any(type(s.get(key)) is not int or s[key] < 0 for s in validated
           for key in ("system_swap_used_bytes", "system_available_bytes")):
        return {"valid": False, "errors": ["Invalid system resource samples"]}
    growth = max(s["system_swap_used_bytes"] - start["system_swap_used_bytes"] for s in observed)
    available = min(s["system_available_bytes"] for s in observed)
    if growth > policy["max_scored_swap_growth_bytes"]:
        errors.append("System swap grew during scored requests")
    if available < policy["min_available_bytes"]:
        errors.append("Observed available memory fell below preregistered threshold")
    return {"valid": not errors, "errors": errors, "max_scored_system_swap_growth_bytes": growth,
            "min_observed_available_bytes": available, "policy": policy,
            "scope": "Discrete outside-timing samples, not a continuous physical or Metal peak measurement"}
