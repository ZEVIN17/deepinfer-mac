# DeepInfer-Mac continuation

Updated 2026-10-21. Continue [PHASE3_EXECUTION_PLAN.md](PHASE3_EXECUTION_PLAN.md) and [CODEX_EXECUTION_PROMPT.md](CODEX_EXECUTION_PROMPT.md) on `codex/phase2-core`. Checkpoints preserve evidence, not approval gates. Root owns the GPU baton; preserve the user’s service on port 8765 and conservative memory guards.

## Current result

- Section 2 model/cache/draft interfaces connect Python, CLI and API. SmolLM2-135M-Instruct-8bit passed all 9 actual dense baseline integration checks, including native token equality, cache branches, cancellation/recovery and SSE. [Adapter evidence](evidence/phase3-adapter-verification.json).
- Section 3 opt-in continuous worker performs actual MLX-LM ticks, bounded admission, stable slots, cancellation and cleanup. Both real policy runs retain 12/13 serial-native matches, so the strict gate remains false. Fixed-history logits diagnosis found a unique serial maximum becoming a batch-path tie at Chinese output index 39; this does not isolate a kernel or justify relaxing equality. New short prefill still waits behind an existing long prefill. [Continuous guide](CONTINUOUS_EXECUTION.md), [diagnosis review](evidence/phase3-batch-logits-review.json).
- Section 4 experimental joint session now executes conditional/cumulative confidence contracts, censored observations, stable UID/generation histories, N−2 total / N−1 row allocation, real MLX ranking and real equal-length verification buckets. Target/draft cache updates are transactional with exact recovery and catch-up. Hybrid caches execute single rows. This mechanism is not connected to the public worker; trained confidence weights and automatic cost calibration are not implemented. [Joint guide](JOINT_VERIFICATION.md).
- Actual forced-length control `[1,2,4]` executed **7 versus 12 verification positions**, and **30 versus 35 total model-input positions** including draft/recovery/catch-up. These are deterministic work savings in this control, not latency or throughput advantages. The pretrained control uses the target itself as draft; English/code match serial, Chinese divergence remains and all four strict reports retain `accepted=false`.
- Fresh verification: **317 portable tests passed; 7 new actual-Metal tests passed**. The device tests include distinct tiny random networks, real buckets/rank, hybrid single-row state, rollback, EOS/output limits and dynamic batch membership. Independent [verification](evidence/phase3-joint-verification.json), [code review](evidence/phase3-joint-code-review.json) and [anti-pattern check](evidence/phase3-joint-anti-pattern-check.json) approve the scoped mechanism, not full section 4 acceptance. No stable advantage over native MLX-LM/oMLX is established.

## Reproduce and inspect

```sh
.venv/bin/python -m pytest -q -m 'not metal'
.venv/bin/ruff check src tests scripts benchmarks/experiments benchmarks/quality
.venv/bin/python -m pytest -q tests/test_joint_budget.py
# Actual Metal execution only after root assigns the GPU baton:
.venv/bin/python -m pytest -q tests/test_varlen_metal.py --run-metal
# Explicit small-model experiment; use a fresh output path and preserve failed gates:
.venv/bin/python benchmarks/experiments/phase3_verify_joint.py --execute --output results/phase3-joint/fresh.json
# Local baseline API/UI; use a free port and retain memory guards:
.venv/bin/python -m deepinfer_mac.cli serve --port 8876 --model-config examples/models/smollm2-135m-8bit.local.json
```

- Reports: `benchmarks/reports/phase3-joint-mechanism{,-diagnostic,-state-control,-final}.json`. Raw tasks, outputs and original reports stay private under excluded `artifacts/phase3-joint/`. Public reports normalize source-map path keys only; original hash values and failed measurements remain. The later harness formatter hash is separately recorded from the actual execution hash.
- Candidate wheel/sdist/site: `dist/candidates/v0.1.0a2-766d6e1/`, built from exact commit `766d6e1ac58622712a5e1394962b2bdd254f47cf`, with fresh offline CPU install/API/CLI and archive checks. **These packages contain the preceding continuous checkpoint, not this joint implementation.** [Package evidence and SHA256](evidence/phase3-checkpoint-package.json); release-ready remains false.
- Public bilingual site: https://zevin17.github.io/deepinfer-mac/ (`zh.html`). Existing website-only main `7b7beb8` and historical a1 release remain unchanged. Local product candidate has 8 pages / 138 links and verified baseline acceptance only; this checkpoint does not deploy main, create a tag or publish a release.

## Next execution

- Preserve strict serial-native failures while continuing independent work. Audit a distinct real small target/draft pair, then measure complete-cycle development costs and held-out confidence coverage before claiming cost-driven adaptation. Existing same-target/full-budget controls establish mechanics only.
- Integrate the verified joint mechanism into the public lifecycle only with resource admission, cancellation/rollback, dynamic remapping and independent output controls. Resolve hybrid cross-row and serial-equivalence boundaries before capability promotion. ZOS overlap follows an actual verified timeline.
- User cannot currently release more memory: do not ask again or bypass guards. Large-model fair MLX-LM/oMLX comparisons remain resource-limited; no simulated performance data or superiority claims.
- Use explicit public file allowlists and normal experimental-branch pushes. Exclude `artifacts/`, `results/`, weights, `dist/`, private paths and incomplete parallel audits.

## History

[Previous complete continuation entry](history/PROGRESS_PHASE3_CONTINUOUS_20261021.md) preserves the preceding bytes and detailed adapter/continuous evidence. [Phase2 history](history/PROGRESS_PHASE2_20261021.md), [alpha history](history/PROGRESS_ALPHA_20261021.md), [DS mapping](DS_DESIGN_MAPPING.md), [varlen source contract audit](evidence/phase3-varlen-contract-audit.json). Historical source hashes stay bound to their original runs; never relabel them as current execution.
