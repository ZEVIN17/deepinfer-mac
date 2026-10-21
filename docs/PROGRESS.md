# DeepInfer-Mac — resume here

Updated 2026-10-21, Asia/Shanghai. User chose small-model validation first; Qwen3.8-27B remains a separate, non-default candidate. Existing untracked work was preserved; initial pytest had no tests. First usable alpha is complete and published. Source/docs/site, local and Linux CI verification, Pages deployment and alpha artifact checks all passed. No further experiment or recurring build is running.

## Current result
- Real MLX engine, byte-budget exact prefix cache, bounded local API/UI and CLI are implemented. Default is baseline; fixed/adaptive require explicit selection. No official DSpark/full V4.1 port.
- Final checks passed:21 portable tests,14 skipped;14 explicit Metal controls; Ruff;7-page/109-link site; wheel/sdist resource+license+privacy validation. Final4B API smoke and installed-wheel4B CLI passed. No background GPU/server/download job.
- Clean-wheel real4B CLI ran outside checkout with no PYTHONPATH, default baseline,32 output tokens and zero observed swap delta. Evidence: `benchmarks/reports/clean-wheel-qwen3-4b.json` + `.meta.json`.
- Formal Qwen3-4B full corpus:216 measured/288 including warmups, zero failures, full all-variant token equality27/36, baseline/native36/36. Adaptive did not beat tuned fixed; no equivalent speculative speedup claimed.
- Exact-cache evidence: warm model,1740-token prompt,2 output tokens including EOS;1739 hot-hit tokens. Three cold/hot pairs2097–2174/36.7–38.3ms, full tokens/text equal. Eight512-prefix branches match, active+6KiB/RSS+0.26MiB, observed request swap0;1KiB admission refusal/recompute matches. This is narrow repeated-prefill evidence.
- Isolated-residency control:18 OS-process blocks,54measured+54warmups, zero failures/swap delta.6/9 all-variant groups match; native/baseline/adaptive9/9. Earlier same-process24MemoryErrors remain. Close frees Metalactive~2.6GB→8B/cache0, but native/Rust RSS retention across reloads remains unresolved; frequent switches should restart/use isolated processes.
- Static site:7 pages,109 checked links/resources, real full benchmark/cache JSON, source archive,4B recorded UI demo; desktop and actual390×844 browser viewport checked. No public inference or automatic localhost calls.

## Commands
```sh
uv sync --frozen --extra mlx --extra dev
uv run deepinfer-mac doctor
uv run deepinfer-mac run --profile qwen3-4b '用中文解释 KV 缓存。'
uv run deepinfer-mac serve
uv run pytest -q
uv run pytest -q --run-metal -m metal
uv run deepinfer-mac bench --profile qwen3-4b --suite full --repeats 3 --max-tokens 64 --fixed-lengths 1 2 4 --output results/reproduction
python3 site/build.py && python3 site/check.py
uv build && python3 scripts/verify_artifacts.py
```

## Evidence
- `docs/evidence.md` is the public index; `docs/benchmarking.md` defines measurement boundaries.
- `benchmarks/reports/m4-pro-qwen3-4b/`: full raw JSON/CSV/JSONL/report, immutable measured source snapshot+28-file SHA manifest; digest d6de01beb2dbe1fe4434036326d6c17d7872668b5090b25be3d5d00409a43c1e. Runtime had no Git commit; archive was verified byte-for-byte.
- `benchmarks/reports/cache-qwen3-4b.json`, `resident-process-ablation-qwen3-4b.json`, `unload-resource-check.json`, `metal-service-smoke*.json`: cache, residency and actual service evidence.
- `benchmarks/reports/local-ui-demo-qwen3-4b.json` and `local-ui-demo.json`: actual4B/0.6B UI exports; screenshots in `docs/evidence/`. Single interactive requests, not benchmark claims.
- `upstream.lock.json`, `models.lock.json`, `uv.lock`; exact reference-header source/license audit in `docs/architecture.md`.
- `docs/evidence/source-manifest.json`: final release source after safe defaults, explicitly distinct from frozen benchmark source. Regenerate only if runtime/locks change.
- Existing clean environment `/private/tmp/deepinfer-clean-install/bin/python` has44 dependencies; installation/doctor previously passed. Reuse it, do not download dependencies again. Verified wheel path `dist/deepinfer_mac-0.1.0a1-py3-none-any.whl`; companion sdist in the same directory. Final build is complete; preserve these bytes for upload/checksums.

## 27B checkpoint and remaining external steps
- `docs/qwen27.md`, `qwen27.manifest.json`, `qwen27-alternatives.json` and `scripts/probe_qwen27.py` contain pinned audit, preflight and resume commands.4-bit2K/4K estimates17.92/18.05GiB available vs observed8.06/8.26GiB; blocked before load, not OOM/peak. Clean3-bit/affine2-bit candidates estimate15.21/11.79GiB; neither downloaded/quality-tested. No silent replacement.
- Paused27B4-bit transfer:88/1274 first-shard chunks(~367MiB), `/tmp/deepinfer-download-chunks/Qwen3.8-27B-4bit/`. Check existing processes before resuming; `/tmp` may be cleaned across reboots. No download/GPU job currently required for small-model release.
- Published canonical release source: `9cda996b98a3967eb6395da520cf83ce5b7287a0`; tree `6c8f67ded02c06c5f495c39067bf76e519ec7d54` exactly matches142-file locally verified source. Original local7643... remains preserved. Official Git Data API was used after Git transport failure; no force update.
- Live website: https://zevin17.github.io/deepinfer-mac/ . Deploy workflow39540322347 succeeded;9 live pages/assets/JSON endpoints returned200 and matched local production bytes exactly (`docs/evidence/live-site-check.json`). Portable CI39540162708 succeeded on canonical source; initial pre-enablement Pages failure was resolved by enabling workflow Pages and dispatching deployment.
- Public alpha: https://github.com/ZEVIN17/deepinfer-mac/releases/tag/v0.1.0a1 ; prerelease=true, not latest, tag targets exact canonical source. Wheel, sdist, complete static ZIP and SHA256SUMS uploaded; GitHub returned matching digests. No credentials were printed or persisted.
- SHA256 wheel `c2fef578a8f5d654276822d59785f12509c585784d6ff9b81b747a4fdef31010`; sdist `b13994a95217fa3a862aade4b2f08950281f574845a36f564874f0d495a91bb1`; site ZIP `88f5df1fa842580966934ddec82683978bf43730f4e04b5555c4c1292cae7e39`. Published package bytes are final; do not rebuild them merely for this publication-log update.
- Remaining limits are hardware/experimental scope, not release blockers:27B not loaded, speculative numeric divergence/no adaptive gain, and reload RSS retention. Read the specific evidence before extending claims.
