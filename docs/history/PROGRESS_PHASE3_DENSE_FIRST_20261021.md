# DeepInfer-Mac continuation

Updated 2026-10-21. Continue [PHASE3_EXECUTION_PLAN.md](PHASE3_EXECUTION_PLAN.md) and [CODEX_EXECUTION_PROMPT.md](CODEX_EXECUTION_PROMPT.md) on `codex/phase2-core`. Checkpoints preserve evidence, not approval gates. Root owns the GPU baton; preserve the user's service on port 8765 and conservative memory guards.

## Current result

- Model/cache/draft interfaces, CLI/API identity, dense baseline integration, actual continuous ticks, joint confidence/budget allocation and real variable-length verification buckets are implemented. Earlier strict batch/native equality failures remain preserved; these mechanisms do not establish general equivalence or superiority.
- A distinct real pair is audited: SmolLM2-360M-Instruct-6bit target + SmolLM2-135M-Instruct-8bit draft, pinned identities and identical vocabulary/template checks. [Pair audit](evidence/phase3-small-pair-audit.json) records the earlier CPU-only audit; subsequent runtime evidence is separate.
- Explicit experimental joint service connects actual pair execution to bounded admission, stable slots, cancellation/recovery, API and SSE. Independent actual service validation passed **35/35 checks**, including two-row execution and resource cleanup; **330 portable + 48 actual-Metal tests passed**. Swap growth was zero, service Metal peak 674,429,864 bytes, final active/cache 8/0 bytes. [Verification](evidence/phase3-joint-service-verification.json), [review](evidence/phase3-joint-service-code-review.json), [anti-pattern audit](evidence/phase3-pair-service-anti-pattern-check.json).
- Separate frozen development/held-out cost experiment: five controls × two held-out tasks all matched native tokens (**10/10**, 32 tokens/task), zero swap growth and 809,765,692-byte Metal peak. Adaptive complete-task time **576.54 ms**, fixed full depth **563.57 ms**, native serial **252.38 ms**. Adaptive was 2.3% slower than fixed and 2.28× native time; one process per stage is descriptive, not a significant performance conclusion.
- Heterogeneous buckets executed **76 vs 109 verification positions** (30.3% fewer), or **333 vs 366 total decode model-input positions** (9.0% fewer). Complete time **912.66 vs 912.24 ms** shows no measured latency benefit. These are real work savings in a bounded control, not a competitive speed advantage. [Cost experiment](PAIR_COST_EXPERIMENT.md), [summary](evidence/phase3-pair-cost-experiment.json).
- API cost calibration is **not accepted**: default experimental service uses full budget without a matching curve; no automatic calibration or trained confidence head. Synchronous prefill, verified-round cancellation and cold group/cache behavior remain explicit. No established advantage over native MLX-LM/oMLX; release-ready remains false.

- Dense fast-path opt-ins now implement actual batched draft forwards and direct dense-cache trim, with defaults retaining serial draft / exact replay. Only audited dense KVCache target/draft pairs are supported; hybrid, rotating and quantized cache classes reject. No API capability promotion. Fresh verification: **338 portable + 76 actual-Metal tests passed**.
- First depth-three pretrained development process: **20/20 request outputs match native tokens**, **40/40 next-token argmax and offset checks pass**, zero swap growth, Metal peak 817,138,492 bytes. Strict next-logit allclose(1e-5) passes **0/40**, including every original serial/replay control. Strict acceptance remains **false**; tolerance was not relaxed. This does not establish general bit identity or isolate an optimization-specific defect.
- Descriptive single-process completion: old serial/replay **316.17 ms**, fast batched/trim **250.79 ms**, native serial **180.39 ms**. These are development observations with differing native scheduling, **not a formal speed advantage**. [Protocol/results](DENSE_FASTPATH_EXPERIMENT.md), [verification](evidence/phase3-dense-fastpath-verification-first.json), [source review](evidence/phase3-dense-fastpath-code-review.json), [anti-pattern audit](evidence/phase3-dense-fastpath-anti-pattern-check.json).
- Earlier stage profiling and calibration guards remain documented in the archived continuation. Bilingual site candidate remains local; no deployment in this checkpoint.

## Reproduce and inspect

```sh
.venv/bin/python -m pytest -q -m 'not metal'
.venv/bin/ruff check src tests scripts benchmarks/experiments benchmarks/quality
# Actual Metal execution only after root assigns the GPU baton:
.venv/bin/python -m pytest -q -m metal --run-metal
.venv/bin/python benchmarks/experiments/phase3_verify_joint_service.py --execute --output artifacts/phase3-joint-service/fresh.json
# Experimental pair service, after memory admission; keep existing port 8765 untouched:
.venv/bin/python -m deepinfer_mac.cli serve --port 8876 --model-config examples/models/research-smollm2-pair.local.json --experimental-joint-verification --joint-profile research-smollm2-pair --joint-depth 3 --max-active 2 --batch-context-tokens 4096
```

- The actual diagnostic used `--execute --stage development --profile-stages --output artifacts/phase3-pair-cost/development-stage-profile-independent.json`. Its exact measured harness is preserved privately at `artifacts/phase3-pair-cost/source-profile/phase3_pair_cost.py` (SHA256 `d2367826416ae649f5c8cb1af888e85b1caae4879075ca178d760d41a8ad4585`); reproduction requires restoring that harness at the original repository path in a separate checkout plus the private workload. Current guarded harness has a different source identity; its commands are in the cost guide. Reproduce guard checks without GPU: `.venv/bin/python -m pytest -q tests/test_stage_profile.py`.
- [Service guide](JOINT_VERIFICATION.md) documents request/cost contracts and limitations. [Cost guide](PAIR_COST_EXPERIMENT.md) gives development/held-out commands; exact private workload is required, public hashes alone are not a downloadable reproducibility corpus.
- Public sanitized reports: `benchmarks/reports/phase3-joint-service.json`, `phase3-pair-cost-development.json`, `phase3-pair-cost-heldout.json`. Raw prompts, outputs and failed harness attempts remain excluded under `artifacts/`; public reports preserve failures, counts and hashes. No model weights are published.
- Candidate wheel/sdist/site remains `dist/candidates/v0.1.0a2-766d6e1/`, source `766d6e1ac58622712a5e1394962b2bdd254f47cf`: **it predates this joint implementation**. [Package evidence](evidence/phase3-checkpoint-package.json). Public bilingual site https://zevin17.github.io/deepinfer-mac/ remains website-only main `7b7beb8`; no main merge, new tag or release in this checkpoint.

## Next execution

- Next diagnostic harness v2 is unexecuted preparation and excluded from this checkpoint. Add schedule-matched replay and native same-batch controls to isolate quantized partition sensitivity; develop depths 1/2, then freeze source/configuration and run at least three independent processes on new held-out cases. Previously exposed held-out workloads are regression cases, never relabelled blind tests. Preserve failed first-run evidence and strict gate.
- Resolve wider serial-equivalence and hybrid cross-row boundaries before capability promotion. Fair 9B/MLX-LM/oMLX benchmarks remain resource-limited: user cannot free more RAM, do not ask again or bypass guards. Continue independent work without fabricated performance data.
- Use explicit public file allowlists and normal experimental-branch pushes. Exclude artifacts, results, weights, dist and private paths; historical hashes remain bound to original runs.

## History

[Immediately preceding profiling continuation](history/PROGRESS_PHASE3_PROFILE_20261021.md) preserves its exact bytes. [Previous service continuation entry](history/PROGRESS_PHASE3_SERVICE_20261021.md) preserves the earlier service entry byte-for-byte. [Previous complete joint continuation entry](history/PROGRESS_PHASE3_JOINT_20261021.md) preserves its exact bytes. Earlier [continuous](history/PROGRESS_PHASE3_CONTINUOUS_20261021.md), [Phase2](history/PROGRESS_PHASE2_20261021.md) and [alpha](history/PROGRESS_ALPHA_20261021.md) histories remain unchanged.
