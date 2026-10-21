# Frozen upstream source for CPU specification review

These two unmodified files are from `deepseek-ai/DeepInfer` commit
`9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`, retrieved using the GitHub connector.
They retain their original contents and are reference material only; no import
or execution of their TileLang, CUDA, distributed model, or confidence head occurs.
The `.txt` suffix marks frozen text references, keeping upstream formatting and
dependencies outside this repository's Python lint/import/runtime paths. Bytes
are unchanged; blob verification ignores the local filename.

| Local file | Upstream path | Git blob |
|---|---|---|
| `schedule_mtp_varlen.py.txt` | `deepinfer/model_executor/layers/tilelang_ops/cuda/schedule_mtp_varlen.py` | `34f0d7f852dfe552965c40e817a3841be515a219` |
| `dflash.py.txt` | `deepinfer/model_executor/models/mtp/dflash.py` | `c8f59022682c1a34956e022b26fa61edae50d934` |

Upstream: <https://github.com/deepseek-ai/DeepInfer/tree/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9>.
Copyright and attribution: DeepSeek / DeepInfer contributors; Apache-2.0.
The upstream `LICENSE` at this commit is byte-identical to
`third_party/licenses/DeepInfer.txt` (Git blob
`261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64`). See that included license.

The existing `../vendor/mtp_varlen.hpp` remains unmodified; its Git blob
`09af680aa42b724fe86315007a160d5012a1fde9` also matches this upstream commit.
The new sibling `../phase3_budget_replay.py` is an explicitly adapted CPU
specification of the source's mathematical ranking rule, with independent sort
checks and hand-authored fixtures. It is neither an upstream device execution
nor the future production allocator. See `docs/PHASE3_BUDGET_REFERENCE.md`.
