# DeepInfer-Mac continuation

Updated 2026-10-21. Branch `codex/phase2-core`. First usable release delivery: **0.1.0a3**. Preserve the user's port 8765, memory guards and historical raw evidence. No repeated benchmarks without a new question.

## Delivered

- Native-first CLI service; capability-based serial fallback, bounded admission, SSE, cancellation/recovery; CLI/Python/local UI and serial prefix-cache controls.
- Allocated joint draft work, shape-specific bounded cycle observations and transactional native handoff are implemented as explicit building blocks. Default serving uses native batching.
- **364 portable + 95 Metal tests**, lint and **46/46 actual-model service checks** passed. Independent installed wheel completed actual 135M CLI/API/SSE/UI-resource smoke with no swap growth.
- Three-process native paired-task median **124.09 ms**; allocated draft **529.84 ms**, full draft **760.56 ms**. Native chosen on measured evidence. Complete scopes and historical failures remain in [runtime report](PHASE4_NATIVE_RUNTIME.md).
- [Release entry](RELEASE.md), [release evidence](evidence/release-0.1.0a3.json), [installed-wheel evidence](evidence/release-installed-wheel.json). Bilingual website has 8 pages / 144 checked local links. Packages: `dist/current/`; website: `site/dist/`.

## Reproduce

```sh
uv sync --frozen --extra mlx --extra dev
./scripts/release-check.sh dist/current
uv run pytest -q -m metal --run-metal
uv run deepinfer-mac serve --port 8876 --model-config examples/models/smollm2-135m-8bit.local.json
python3 -m http.server 8088 --directory site/dist
```

The local model example requires existing audited weights; [release guide](RELEASE.md) gives clean model-download commands. CPU tests need no GPU. Run GPU tasks serially. An installed wheel can be checked with `scripts/verify_installed.py --model <local-135M-directory> --output <report.json>` using its own environment (MLX and httpx installed).

## Publication and next work

Runtime commit `faa0a88` is pushed to both main and the work branch. Both portable CI runs and Pages passed; the CI artifact is uploaded. All 8 pages, CSS/JS and 3 new evidence files match the local build byte-for-byte (13/13). [Publication evidence](evidence/release-publication.json) records exact commits, run URLs and hashes. No PyPI upload or external community message is implied. Archive/model weights/private raw reports remain local under `artifacts/` and `results/`.

First-version functionality is complete; future work is a separate iteration: native prefix reuse, fair multi-engine arrival traces, broader model/quality coverage and cost-calibrated automatic strategy selection. See [technical scope](RELEASE.md), not a blocker to this usable release.

[Prior native checkpoint](history/PROGRESS_PHASE4_NATIVE_CHECKPOINT_20261021.md) · [Phase 3 final](history/PROGRESS_PHASE3_FINAL_20261021.md).
