# Qwen3.8-27B: pinned candidate and resource evidence

Qwen3.8-27B is an experimental text demonstration candidate. The default remains the small verified model until the 27B resource, generation, streaming API and repeated-session checks pass. No 27B speed, quality, or long-session stability result is claimed here.

The [quantized snapshot](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit/tree/10c35caafbb80f7dc6a7a432cdd11af10a6d4818) is pinned to `10c35caafbb80f7dc6a7a432cdd11af10a6d4818`. Its affine quantization is 4 bits, group size 64. [qwen27.manifest.json](qwen27.manifest.json) records all file sizes, weight SHA256 values, configuration/template digests, tokenizer identity and header audit.

The 3 weight files download **16,054,541,349 bytes**. Their tensor payload totals **16,054,262,240 bytes**. Reading all three safetensors JSON headers and cross-checking every key against the index separates **15,132,802,048 bytes of text tensors** from **921,460,192 bytes of vision tensors**. These are storage payloads; measured runtime peaks can exceed them.

MLX-LM 0.32.2 `models/qwen3_5.py` supports the config's `model_type=qwen3_5`. `Model.sanitize` removes `vision_tower.*` and forwards the native `language_model.*` names. All 48 convolution tensors already have `[10240, 4, 1]` shapes, so the text sanitizer should leave their layout unchanged. This is a source/header audit, not successful real loading. The normal loader opens the original shards before sanitizing; downloading only text files is impossible because text and vision coexist in shard 1.

The [pinned official model card](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/README.md) describes a 64-layer hybrid: 48 Gated DeltaNet layers and 16 full-attention layers. Its [license](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/LICENSE) is Apache-2.0. The quantized card also declares Apache-2.0 but does not include its own license file.

Although the config declares one MTP layer, the selected 4-bit index contains **no MTP or predictor keys**. The installed text sanitizer filters `mtp.` keys and does not execute an MTP decoder. Other repositories advertising MTP require independent runtime validation. The old Qwen3-0.6B draft is incompatible: Qwen3.8 has a padded output vocabulary of 248320, different tokenizer and special IDs. Its tokenizer JSON SHA256 is `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`.

The hybrid cache combines `ArraysCache` and `KVCache`. `ArraysCache` cannot be rolled back by ordinary trim. Baseline and exact checkpoint prefix reuse are the candidate modes; speculative decoding remains unsupported until recurrent-state checkpoint/replay correctness is demonstrated.

## Resource probe

Run without loading MLX or making network requests:

```sh
uv run python scripts/probe_qwen27.py --preflight-only --prompt-tokens 2048 --output results/qwen27-preflight-2k.json
uv run python scripts/probe_qwen27.py --preflight-only --prompt-tokens 4096 --output results/qwen27-preflight-4k.json
```

The conservative budget includes text weights, BF16 full-attention KV, FP32 recurrent state, convolution state, 512 MiB of retained snapshots, 2 GiB of temporary workspace and a 1 GiB available-memory margin. It includes room for a second conversation turn. Workspace and safety margins are assumptions, not measured peaks. RSS and Metal allocator counters must be reported separately.

| Probe | Estimated required available memory | Observed available memory | Observed existing swap | Outcome |
| --- | ---: | ---: | ---: | --- |
| 2K input + 256 output, batch 1 | 17.92 GiB | 8.06 GiB | 344.1 MiB | Refused before model load |
| 4K input + 256 output, batch 1 | 18.05 GiB | 8.26 GiB | 344.1 MiB | Refused before model load |

These observations are snapshots from this 24 GiB M4 Pro. They are **budget-gate results, not an OOM experiment**. Evidence: [2K observable preflight](evidence/qwen27-preflight-2k-observable.json), [4K observable preflight](evidence/qwen27-preflight-observable.json). [Sandbox preflight](evidence/qwen27-preflight.json) additionally demonstrates that a denied swap sysctl is represented as unavailable, never as zero.

When the preflight passes, the sole GPU owner can run the real probe sequentially:

```sh
uv run python scripts/probe_qwen27.py --run --prompt-tokens 2048 --workload zh --output results/qwen27-2k-zh.json
uv run python scripts/probe_qwen27.py --run --prompt-tokens 4096 --workload code --output results/qwen27-4k-code.json
```

The probe constructs exactly the requested **formatted input token count**, reports the actual count, uses prefill chunks of 128 and 256 output tokens, and repeats cold, cache seed, exact-prefix and second-turn comparisons. Greedy cold/cache token sequences are retained and compared. A memory monitor cancels generation if four consecutive swap samples increase and net swap growth exceeds 64 MiB, or swap observation is lost. An unavailable swap metric prevents `--run`. The gate is deliberately conservative; passing still requires reviewing the actual samples and outputs. The profile's context limit must accommodate the second turn as well as the first turn's input and output.

## Existing lower-bit alternatives

These are **community-produced candidates** derived from the original Qwen model. None has been loaded or quality-tested in this project. Exact revisions, file SHA256 values, recipes and source-audit scope are in [qwen27-alternatives.json](qwen27-alternatives.json).

| Candidate | Main weight download | Text tensor payload | Same probe's estimated required available memory |
| --- | ---: | ---: | ---: |
| [keXjos 2-bit](https://huggingface.co/keXjos/Qwen3.8-27B-mlx-2Bit/tree/39d5487ad4442961b64098e15aa11ce05e6f9976) | 8.410 GB | 8.409 GB | 11.79 GiB |
| [leonsarmiento clean 3-bit](https://huggingface.co/leonsarmiento/Qwen3.8-27B-3bit-mlx/tree/5fc234d9e6080b8388a11286380e801b7c9f535c) | 12.706 GB | 12.089 GB | 15.21 GiB |
| [rapid-mlx mixed 3.5 bpw](https://huggingface.co/rapid-mlx/Qwen3.8-27B-mixed-3.5bpw-MLX/tree/052cd9bad54ba78ef98b07ae01048bad90f26786) | 13.909 GB | 13.454 GB | 16.49 GiB |

2-bit uses uniform affine/group-64 quantization, with text-only weights and no MTP. Clean 3-bit uses affine/group-64 with selected 4-bit embedding/output and 8-bit vision layers. The mixed version uses affine/group-64 with per-module 2/3/4-bit allocation; its optional separate MTP artifact adds 849,400,403 bytes and is outside this engine's support. Both 3-bit candidates' text payloads were computed from all shard headers. The installed MLX 0.32.4 API supports affine bits 2/3/4/8 at group 64, and MLX-LM 0.32.2 supports per-module recipes and `qwen3_5`. That establishes source-format compatibility, **not an actual load or quality guarantee**.

Even the 2-bit budget exceeds the approximately 8 GiB currently available. Smaller bit width must be a separate quality/behavior comparison, never a silent replacement of 4-bit. No candidate was downloaded for this investigation. Modified or uncensored oQ2 models were not substituted for the original model. The mlx-community NVFP4 variant still downloads approximately 16.05 GB, so its name alone is not evidence of a memory-saving option.

## Paused 4-bit download and resume

The engine owner paused the 4-bit transfer after the user chose to validate the small model first. Metadata and **88/1274 chunks of shard 1, approximately 367 MiB**, were preserved under `/tmp/deepinfer-download-chunks/Qwen3.8-27B-4bit/`. No later shard was started. Treat these as resumable transfer chunks, not validated complete model weights. `/tmp` is a session checkpoint and may be cleaned by the operating system; use a persistent directory when resuming across reboots.

After confirming no other weight download is active, resume the exact same chunks and pinned source:

```sh
uv run python - <<'PY'
import json
from scripts.download_ranges import download
manifest = json.load(open('docs/qwen27.manifest.json'))
for entry in manifest['files']:
    name = entry['path']
    if name.endswith('.safetensors'):
        download(manifest['repo'], manifest['revision'], name,
                 entry['size_bytes'], entry['sha256'],
                 '/tmp/deepinfer-download-chunks/Qwen3.8-27B-4bit/' + name,
                 workers=32)
PY
```

The downloader verifies `Content-Range` and the final full-file SHA256, preserves incomplete chunks and handles shards sequentially. The model snapshot also needs the pinned JSON files and **chat_template.jinja**; the latter is separate from `tokenizer_config.json`. Weights are kept in the Hugging Face cache and excluded from release artifacts. Download completion does not waive the resource gate or qualify 27B as the default demo.
