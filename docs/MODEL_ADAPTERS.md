# Model adapters and registration

DeepInfer-Mac uses MLX-LM to execute models. `ModelSpec`, `ModelRegistry`,
`ModelAdapter`, `CacheAdapter` and `DraftProvider` are project contracts; they are
not new upstream MLX APIs. Registration is portable metadata/file inspection and
does not load arrays or download a model. The Phase2 preset registry remains
available, with its original public mode restrictions.

## Register a model

A model configuration is an explicit JSON file, loaded per process. There is no
implicit global registry or HTTP endpoint that downloads arbitrary client models.

```json
{
  "id": "my-local-model",
  "source": "/path/to/complete-mlx-model",
  "context_limit": 4096,
  "runtime_reserve_bytes": 2147483648
}
```

For a Hugging Face repository, set `source` to `organization/repository` and
`revision` to an exact 40-character commit. Branch names are rejected. Repo
registration stays offline; normal generation/download explicitly resolves its
snapshot through Hugging Face. Local sources omit `revision`, and use a full
SHA256 content identity. `draft` and `draft_revision` can describe an external
draft, but do not enable unvalidated speculative modes. Confidence/calibration
revision fields bind experiment identity and do not imply trained head support.

```sh
# Metadata only: no model arrays or download.
deepinfer-mac models --model-config model.json --json

# Register a custom model and generate through the baseline engine.
deepinfer-mac run --model-config model.json --profile my-local-model --max-tokens 32 'Hello'

# Equivalent shorthand for one local model.
deepinfer-mac run --model /path/to/complete-mlx-model --model-id my-local-model 'Hello'

# Serve this model alongside the existing presets.
deepinfer-mac serve --port 8876 --model-config model.json
```

The API request uses the same model ID:

```json
{"model":"my-local-model","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}
```

Python uses the same registration contract:

```python
from deepinfer_mac.adapters import ModelRegistry, ModelSpec
from deepinfer_mac.engine import InferenceEngine
from deepinfer_mac.state import GenerationRequest

registry = ModelRegistry([ModelSpec("my-local-model", "/path/to/complete-mlx-model")])
engine = InferenceEngine(model_registry=registry)
try:
    result = engine.generate(GenerationRequest(profile="my-local-model", prompt="Hello", max_tokens=32))
    print(result.text)
finally:
    engine.close()
```

Multiple `--model-config` flags register multiple models. Duplicate or reserved
preset IDs are rejected. CLI and API model listings preserve the same target
identity; local absolute paths are replaced by content hashes in public metadata
and result metadata. Internal paths remain necessary for loading.

## Capabilities and explicit fallback

Custom models advertise a baseline compatibility entry and baseline sampling;
`status=baseline_compatibility_unverified` is not a real-model validation claim.
`batch`, `trim`, `snapshot_replay`, `varlen_verify`, `confidence`, `speculative`
and `mtp` start false. After loading, `metrics.cache_capabilities` reports actual
cache protocol probes. Method availability is not a batch performance or
speculative correctness result; this stage does not enable concurrent batching
or per-request variable-length verification.

Unsupported requested modes are rejected before weight loading. To explicitly
allow the baseline alternative, set `fallback_to_baseline=true` in a Python/API
request or add CLI `--fallback-to-baseline`. Results include `requested_mode`,
`effective_mode` and `capability_fallback`; a fallback cannot be mistaken for an
executed speculative request. Existing profiles retain default 422 behavior for
unsupported API modes. Speculative sampling retains its strict greedy guard.

`ModelAdapter` owns chat rendering and attention/cache geometry. A model without
a chat template can receive a plain prompt or one user message, with thinking
turned off. Unknown cache geometry requires an adapter before weights load.
Head dimensions and KV head counts can be derived from ordinary dense config;
the existing Qwen3.5 hybrid geometry lives here rather than in DS scheduling.

`CacheAdapter` distinguishes trimmability, native batch protocols and audited
snapshot/replay. Only locked MLX-LM `KVCache`/`ArraysCache` state setters are
accepted for adapter snapshot/replay. The real hybrid generator receives the
adapter from the engine and uses its guarded snapshot/restore operations.
`DraftProvider.verify` is used by the engine before draft arrays load, comparing
full token vocabularies, special IDs and output vocabulary sizes. The target
renders tokens once for both models; differing default draft chat templates do
not silently re-render input.

## Identity and resource guards

Local registration validates config, tokenizer assets/ID range, safetensors
headers/shards, output/embedding vocabulary, and cache geometry before model-array
loading. This is structural compatibility, not a replacement for upstream strict
parameter loading or real model tests. SHA256 includes full weights and model
metadata, template and tokenizer files; full hashing has CPU/I/O startup cost.
Repo snapshots are content-audited after resolution as well as revision-pinned.
Loaded-file mutation refuses reuse; re-register with a fresh engine to load new
content. Cache identity additionally includes quantization, exact position/KV
semantics, thinking/template, draft, confidence and calibration revisions.

Custom model admission uses actual resolved weight sizes plus an explicit runtime
reserve (default 2 GiB, minimum 512 MiB). Insufficient explicit or available memory
rejects before model arrays load. The requested context is capped by the config's
maximum position count when present, and input is never silently truncated. The
existing future-KV admission check runs before prefill. These are conservative
estimates, not hard process/Metal caps. Hybrid target and draft each contribute
state reserve in the new adapter estimate; compared with Phase2, a hybrid draft
can therefore add another 2 GiB conservative reserve. Formal resource comparisons
must freeze this changed admission policy explicitly.

## Verification boundaries

```sh
.venv/bin/python -m pytest -q tests/test_model_adapters.py
.venv/bin/python -m pytest -q -m 'not metal'
.venv/bin/ruff check src tests scripts benchmarks/experiments benchmarks/quality
```

Portable tests use clearly fabricated tiny files and mocked engines. They verify
registration identity, content-change rejection, resource guards, tokenizer and
weight mismatch rejection, public path redaction, frontend model ID agreement,
and explicit fallback. They are not model output or performance evidence. Real
baseline integration for a different architecture and dense/hybrid optimized
execution remain separate acceptance steps with their own reports.
