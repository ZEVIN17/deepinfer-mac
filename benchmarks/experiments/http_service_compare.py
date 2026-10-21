"""Capture one real OpenAI-compatible server block; never invent token timing.

Run engines sequentially with exclusive GPU ownership, rotate engine order in
the outer orchestration, and restart/clear caches between repetition blocks.
The manifest is fixed before measurement. Failures remain in the raw report.
"""

import argparse
import itertools
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from public_service_report import public_report


def request_body(model, case, request_options=None):
    body = {
        "model": model,
        "messages": case["messages"],
        "max_tokens": case["max_tokens"],
        "temperature": 0,
        "stream": True,
    }
    if request_options:
        if set(request_options) - {"mode", "stream_options", "chat_template_kwargs"}:
            raise ValueError("Only explicit mode/template/usage request options may vary by engine")
        body.update(request_options)
    return body


def measure(url, model, case, timeout=600, request_options=None):
    body = request_body(model, case, request_options)
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    result = {"case": case, "request": body, "events": [], "text": "", "error": None}
    first = None
    run_id = None
    try:
        # Local benchmarks must not inherit a user's external HTTP proxy.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=timeout) as response:
            result["headers"] = dict(response.headers)
            run_id = response.headers.get("X-Run-ID")
            for raw in response:
                if not raw.startswith(b"data:"):
                    continue
                value = raw[5:].strip()
                elapsed = (time.perf_counter() - start) * 1000
                if value == b"[DONE]":
                    result["done_ms"] = elapsed
                    break
                event = json.loads(value)
                result["events"].append({"elapsed_ms": elapsed, "payload": event})
                if "error" in event:
                    result["error"] = event["error"]
                for choice in event.get("choices", []):
                    fragment = choice.get("delta", {}).get("content", "") or ""
                    if fragment and first is None:
                        first = elapsed
                    result["text"] += fragment
    except (OSError, ValueError, urllib.error.HTTPError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    result["total_ms"] = (time.perf_counter() - start) * 1000
    result["ttft_text_ms"] = first
    result["protocol_complete"] = "done_ms" in result
    if not result["protocol_complete"] and result["error"] is None:
        result["error"] = "SSE ended without [DONE]"
    # Chunk arrival intervals are not token ITL: servers may coalesce tokens.
    arrivals = [
        event["elapsed_ms"]
        for event in result["events"]
        if any(c.get("delta", {}).get("content") for c in event["payload"].get("choices", []))
    ]
    result["text_chunk_intervals_ms"] = [b - a for a, b in itertools.pairwise(arrivals)]
    try:
        result["quality_pass"] = (
            result["error"] is None and json.loads(result["text"].strip()) == case["expected_json"]
        )
    except (ValueError, KeyError):
        result["quality_pass"] = False
    if "headers" in result and run_id:
        # Project's public run endpoint supplies true engine token/cache/peak
        # metrics after SSE completes. This read is outside request total_ms.
        try:
            with opener.open(url.rstrip("/") + "/api/runs/" + run_id, timeout=10) as response:
                result["server_run"] = json.load(response)
        except (OSError, ValueError) as error:
            result["server_metrics_error"] = f"{type(error).__name__}: {error}"
    run = result.get("server_run", {})
    run = run if isinstance(run, dict) else {}
    metrics = run.get("result", {}).get("metrics", {}) if isinstance(run.get("result"), dict) else {}
    observed_hit = metrics.get("cache_hit_tokens")
    source = "server_run.result.metrics.cache_hit_tokens" if observed_hit is not None else None
    if observed_hit is None:
        for event in result["events"]:
            usage = event["payload"].get("usage")
            details = usage.get("prompt_tokens_details", {}) if isinstance(usage, dict) else {}
            if isinstance(details, dict) and type(details.get("cached_tokens")) is int:
                observed_hit = details["cached_tokens"]
                source = "sse.usage.prompt_tokens_details.cached_tokens"
    result["cache_observation"] = {
        "cache_hit_tokens": observed_hit,
        "source": source,
        "proven_zero_hit": observed_hit == 0,
        "note": "Unknown is not cold KV; request-kind names do not establish cache state.",
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights-revision", required=True)
    parser.add_argument("--engine-version", required=True)
    parser.add_argument("--cache-budget-bytes", type=int, default=512 * 1024**2)
    parser.add_argument("--repetition", type=int, required=True)
    parser.add_argument("--split", choices=("dev", "heldout"), default="heldout")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    report = {
        "configuration": vars(args) | {"manifest": str(args.manifest), "output": str(args.output)},
        "scope": "real server SSE block; GPU exclusive, rotated outer engine order required",
        "records": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for session in manifest["sessions"]:
        if session["split"] != args.split:
            continue
        for case in session["cases"]:
            record = measure(args.url, args.model, case)
            record.update(session_id=session["id"], split=session["split"], repetition=args.repetition)
            report["records"].append(record)
            args.output.write_text(
                json.dumps(
                    public_report(report, snapshot=args.model if args.model.startswith("/") else None),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            )
            print(session["id"], case["kind"], record["quality_pass"], record["ttft_text_ms"], flush=True)


if __name__ == "__main__":
    main()
