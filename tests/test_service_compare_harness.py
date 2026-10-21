"""Portable controls prevent protocol failures and confounded timing claims."""

import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

EXPERIMENTS = Path(__file__).parents[1] / "benchmarks" / "experiments"


@pytest.fixture(autouse=True)
def experiment_imports(monkeypatch):
    monkeypatch.syspath_prepend(str(EXPERIMENTS))


def module(name):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / f"{name}.py")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class Response(io.BytesIO):
    @property
    def headers(self):
        return {"Content-Type": "text/event-stream"}


def test_sse_json_fragments_and_usage_are_preserved(monkeypatch):
    harness = module("http_service_compare")
    wire = b'data: {"choices":[{"delta":{"content":"{\\"x\\":"}}]}\n\n'
    wire += b'data: {"choices":[{"delta":{"content":"2}"}}]}\n\n'
    wire += b'data: {"choices":[],"usage":{"completion_tokens":5}}\n\n'
    wire += b"data: [DONE]\n\n"
    requests = []

    def open_request(request, timeout):
        requests.append(json.loads(request.data))
        return Response(wire)

    monkeypatch.setattr(harness.urllib.request, "build_opener", lambda _: SimpleNamespace(open=open_request))
    case = {"messages": [{"role": "user", "content": "fixture"}], "max_tokens": 32, "expected_json": {"x": 2}}
    result = harness.measure(
        "http://127.0.0.1:1",
        "fixture-model",
        case,
        request_options={"stream_options": {"include_usage": True}},
    )
    assert result["quality_pass"] and result["protocol_complete"]
    assert result["text"] == '{"x":2}'
    assert result["events"][-1]["payload"]["usage"]["completion_tokens"] == 5
    assert requests[0]["temperature"] == 0 and requests[0]["stream_options"]["include_usage"]
    assert len(result["text_chunk_intervals_ms"]) == 1
    assert result["cache_observation"]["cache_hit_tokens"] is None
    assert not result["cache_observation"]["proven_zero_hit"]
    with pytest.raises(ValueError, match="Only explicit"):
        harness.measure("http://127.0.0.1:1", "fixture-model", case, request_options={"temperature": 1})


@pytest.mark.parametrize("hits", [0, 17])
def test_cache_state_only_uses_explicit_usage_evidence(monkeypatch, hits):
    harness = module("http_service_compare")
    first = {"choices": [{"delta": {"content": "{}"}}], "usage": None}
    usage = {"choices": [], "usage": {"prompt_tokens_details": {"cached_tokens": hits}}}
    wire = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in (first, usage))
    wire += b"data: [DONE]\n\n"
    monkeypatch.setattr(harness.urllib.request, "build_opener", lambda _: SimpleNamespace(
        open=lambda request, timeout: Response(wire)))
    result = harness.measure("http://127.0.0.1:1", "fixture", {"messages": [], "max_tokens": 32, "expected_json": {}})
    assert result["cache_observation"]["cache_hit_tokens"] == hits
    assert result["cache_observation"]["proven_zero_hit"] == (hits == 0)
    assert result["cache_observation"]["source"] == "sse.usage.prompt_tokens_details.cached_tokens"


def test_truncated_sse_is_a_failure_even_with_valid_answer(monkeypatch):
    harness = module("http_service_compare")
    wire = b'data: {"choices":[{"delta":{"content":"{}"}}]}\n\n'
    monkeypatch.setattr(
        harness.urllib.request,
        "build_opener",
        lambda _: SimpleNamespace(open=lambda request, timeout: Response(wire)),
    )
    result = harness.measure(
        "http://127.0.0.1:1", "fixture-model", {"messages": [], "max_tokens": 16, "expected_json": {}}
    )
    assert result["text"] == "{}" and not result["quality_pass"]
    assert result["error"] == "SSE ended without [DONE]"


def test_rotating_runner_rejects_different_weights_before_launch(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(EXPERIMENTS))
    harness = module("rotate_service_compare")
    engines = [
        {
            "id": "a",
            "weights_repo": "fixture",
            "weights_revision": "1",
            "quantization": "8bit",
            "cache_budget_bytes": 512,
        },
        {
            "id": "b",
            "weights_repo": "fixture",
            "weights_revision": "2",
            "quantization": "8bit",
            "cache_budget_bytes": 512,
        },
    ]
    with pytest.raises(ValueError, match="identical pinned weights"):
        harness.run({"engines": engines}, {"sessions": []}, tmp_path, 1, "heldout")
    engines[1]["weights_revision"] = "1"
    engines[1]["cache_budget_bytes"] = 1024
    with pytest.raises(ValueError, match="Cache budgets must match"):
        harness.run({"engines": engines}, {"sessions": []}, tmp_path, 1, "heldout")


def test_repeating_one_fixture_prompt_does_not_create_independent_clusters():
    harness = module("summarize_service_compare")
    blocks = []
    for name, elapsed in (("candidate", 5), ("control", 10)):
        for repetition in range(20):
            record = {
                "session_id": "fixture-session",
                "case": {"kind": "fixture"},
                "repetition": repetition,
                "quality_pass": True,
                "error": None,
                "total_ms": elapsed,
                "ttft_text_ms": elapsed / 2,
            }
            blocks.append({"engine": name, "repetition": repetition, "records": [record], "error": None})
    result = harness.summarize({"blocks": blocks}, "candidate", draws=20)
    assert all(row["quality_passing_pairs"] == 20 for row in result["paired"])
    assert all(row["session_clusters"] == 1 and row["ci95"] is None for row in result["paired"])
    assert not any(row["engineering_target_met"] for row in result["paired"])


def test_native_adapter_enforces_byte_budget_without_replacing_server(monkeypatch):
    harness = module("native_lru_server")

    class Cache:
        def __init__(self, max_size, max_bytes):
            self.max_size, self.max_bytes = max_size, max_bytes

    captured = []
    server = SimpleNamespace(LRUPromptCache=Cache)

    def official_main():
        captured.append(server.LRUPromptCache(7))

    server.main = official_main
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(server=server))
    monkeypatch.setattr(harness.importlib.metadata, "version", lambda _: "0.32.2")
    monkeypatch.setattr(sys, "argv", ["adapter", "--cache-budget-bytes", "512", "--model", "fixture"])
    harness.main()
    assert captured[0].max_size == 7 and captured[0].max_bytes == 512
    assert sys.argv[1:] == ["--model", "fixture"]
    monkeypatch.setattr(harness.importlib.metadata, "version", lambda _: "0.33.0")
    with pytest.raises(RuntimeError, match="Reaudit"):
        harness.main()


def test_public_service_report_keeps_values_and_removes_private_execution_paths():
    helper = module("public_service_report")
    workspace = str(Path.home() / "fixture-workspace")
    snapshot = str(Path.home() / "fixture-models/pinned-revision")
    raw = {
        "command": [workspace + "/.venv/bin/python", "--model", snapshot],
        "answer": '{"cost":211}',
        "quality_pass": False,
    }
    public = helper.public_report(raw, workspace=workspace, snapshot=snapshot)
    assert public["command"] == ["<WORKSPACE>/.venv/bin/python", "--model", "<MODEL_SNAPSHOT>"]
    assert public["answer"] == raw["answer"] and not public["quality_pass"]
    assert raw["command"][0].startswith(str(Path.home()))


def test_resource_cap_rejects_8k_plus_output_before_starting_server(tmp_path):
    harness = module("rotate_service_compare")
    engines = [
        {
            "id": name,
            "weights_repo": "fixture",
            "weights_revision": "1",
            "quantization": "8bit",
            "cache_budget_bytes": 512,
        }
        for name in ("a", "b")
    ]
    manifest = {
        "sessions": [
            {
                "id": "8k-fixture",
                "split": "heldout",
                "cases": [
                    {
                        "kind": "continuation",
                        "prompt_tokens": 8300,
                        "max_tokens": 256,
                        "requested_total_context_tokens": 8556,
                    }
                ],
            }
        ]
    }
    with pytest.raises(ValueError, match="never truncate"):
        harness.run({"engines": engines, "total_context_cap_tokens": 8192}, manifest, tmp_path, 1, "heldout")
