# Phase 2 model quality protocol

The primary candidate is `mlx-community/Qwen3.5-9B-8bit` at
`16daa4818c54ce5f5436f929d52542eb65bbed9d`. The same-model 4bit control is
`8b2b98c00a6b4d291155e4890773ca8f769aee53`; Qwen3-14B-4bit fallback is
`a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4`.

`benchmarks/prompts/phase2-quality-smoke.json` fixes 12 tasks, six development and
six held-out, before the first new-model load. It covers Chinese explanation,
executable function repair/deduplication, JSON constraints, arithmetic/logic,
conversation recall and long-input extraction. This engineering smoke is not a
LiveCodeBench/IFEval/MMLU-Pro reproduction or evidence of statistical equivalence.
The independent public subsets and their source/license/sample IDs live under
`benchmarks/quality/`; their scores stay separate.

`benchmarks/experiments/phase2_model_quality.py` records full prompts, rendered
prompt IDs, output IDs/text, seed42, configuration, finish reason, memory and
system-swap deltas. Greedy engine and native paths use identical weights, template,
input and output budget; complete output IDs are compared. The native recommended
sampling control is retained separately. Baseline/native API and CLI now accept
temperature, top_p, top_k, full-history presence penalty, seed and thinking. The
optional Qwen3.5 non-thinking UI preset uses0.7/0.8/20/1.5 with seed42; greedy
remains the default. Fixed/adaptive modes reject non-default sampling and penalties.

The official Qwen3.5-9B card is pinned at
`c202236235762e1c871ad0ccb60c8ee5ba337b9a` in
`artifacts/phase2-model-audit/qwen35-9b-official-readme.md`. Non-thinking general
sampling uses temperature0.7/top_p0.8/top_k20/presence1.5. Thinking general uses
1.0/0.95/20/1.5; precise-code thinking uses0.6/0.95/20/0. These are distinct
protocols, not interchangeable results. The card has differing reasoning-task
recommendations in its overview and best-practice sections; this initial runner
uses the general settings consistently rather than silently tuning by answer.
MLX-LM0.32.2 presence context0 means the full token history (`tokens[-0:]`),
including the prompt. Each native control receives that full history.

Non-thinking output budgets are fixed per task. Thinking controls have a minimum
1024-token cap; unfinished reasoning is retained and fails task completion at
that budget, rather than being mistaken for a wrong final answer. The opening
`<think>` may already be in the prompt; final-answer extraction requires the
closing delimiter in thinking mode and scores only text after it.

Code smoke scoring runs restricted constructs in a separate isolated Python
process with a three-second timeout and records actual unit-test values.
Unsupported harness constructs are inconclusive/manual, not model failures.
Chinese tasks require a recorded per-response review against the fixed rubrics;
keyword presence never substitutes for coherent explanations. The current
`manual_reviewed` annotations are Codex rubric reviews, not independent human
labels or a human preference study. Independent human annotation remains
unavailable; this distinction limits subjective Chinese-quality conclusions. Dev/held-out results,
failures and truncations remain visible. No few-example pass rate establishes
full capability parity or replaces a versioned public evaluation.

Load peak is reset **before** loading and read immediately afterward. Request
peaks are reset separately; Metal active/RSS/system swap are different measures
and are not added together. These quality-run timings are diagnostic observations,
not fair competing-framework speed benchmarks. Changing model/quantization is a
model-selection improvement, not evidence that a DS-derived engine is faster.

Current sampling correctness evidence is `benchmarks/reports/phase2-sampling-qwen35-9b8.json`: seed0/42/4294967295, engine cold/hot/no-cache and native no-cache/hot all15full token sequences match. A real CLI request with the recommended non-thinking settings returns8 on the fixed ball-count task. This is control evidence, not a new aggregate quality score. The public runner's `official_engine` and `official_thinking_engine` settings exercise production sampling; `official_non_thinking`/`official_thinking` retain the standalone native controls and are identified explicitly by backend.

Qwen3-14B official control is a separate protocol: model-card revision
`40c069824f4251a91eefaf281ebe4c544efd3e18` recommends non-thinking
0.7/0.8/top_k20 and thinking0.6/0.95/top_k20; presence defaults to0, with
0–2 optional for repetition and an explicit performance tradeoff. The shared
Qwen3.5 presence1.5 setting may be evaluated on14B for a controlled common
protocol but must be labeled shared, not its default official recommendation.
The official card recommends much larger generation budgets than our resource
subsets; budget truncations here are completion failures at the stated budget,
not a claim that the model could never solve the problem. An independent native
control can use the runner's `--sampling-family qwen3` with a separately frozen
source manifest; do not overwrite the shared-protocol results.

The first frozen public27-task9B8bit run is complete (sourceV2 archive SHA
`1b928104acee51f742b8be4893b2804cfac59779aa79904395dad13f351a09d4`).
Greedy scores are IFEval6/8, MMLU-Pro11/14, LCB public-tests2/4, repair1/1;
production recommended non-thinking sampling scores7/8,9/14,3/4,1/1.
Every MMLU failure (3greedy,5sampled) reached the1024-token cap without a parsed
final answer. LCB failures (2greedy,1sampled) reached2048tokens. These are
budget-limited task completion failures; IFEval failures stopped normally and
violated the requested constraints. Sampling improves some code/instruction
examples while decreasing MMLU completion at this budget. No combined score or
universal gain is claimed. The sanitized report retains per-task decisions,
Wilson intervals, timing and finish reasons; excluded raw data retains full
responses. The runtime source hashes before/after execution are identical.

Before observing9B4bit or14B outputs, an additional **development-only budget
control** is fixed: `mmlu:1471`, `mmlu:9423`, `lcb:abc387_c`; all three profiles,
settings `greedy_engine` and `official_engine` under the common Qwen3.5 sampling
protocol,4096output tokens each, thinking disabled, seed42. These are exactly
the development tasks that hit the initial9B8bit budget in either setting.
Held-out failures do not select this control. Its source must be frozen
separately and all18responses retained, including any new failures. It is an
additional completion-budget experiment, not a replacement for the27-task
finite-budget results or a revised held-out score.
