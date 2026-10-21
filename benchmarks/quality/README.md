# Fixed local quality subsets

This is an outcome-independent engineering comparison, not a reproduction of a leaderboard. Model-reported original-precision results must never populate these local measurements. The manifest was fixed before generating any public-subset model responses. It contains 27 tasks: 8 IFEval, 14 MMLU-Pro (one per category), 4 LiveCodeBench generation, and one separate derived repair task. Dev/heldout assignments and task IDs are fixed in `manifest.json`; do not remove failures or tune prompts on heldout outcomes.

## Sources and protocol

- [Google IFEval](https://huggingface.co/datasets/google/IFEval), revision `966cd89545d6b6acfd7638bc708b98261ca58e84`, Apache-2.0. Original prompt/kwargs are preserved. Restrict to seven deterministic checker families (no-comma, placeholders, bullets, keyword inclusion, JSON, ending, title). Select the first SHA256-ranked eligible task per family and then one remaining task, seed `deepinfer-phase2-quality-v1`. This is deliberately a **supported-constraint subset**, not representative full IFEval. Strict prompt pass requires every original selected instruction to pass. `vendor/ifeval_checkers.py` copies the seven original Google checker classes unchanged from the source URL and source SHA256 in its header, with a thin base interface and constants. Explicit kwargs avoid random fallback branches. Tests check positive/negative scoring for every family. Source license/attribution are retained.
- [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro), revision `b189ec765aa7ed75c8acfea42df31fdae71f97be`, MIT. Select one SHA256-ranked test ID from each of 14 categories. Original question, option order and answer are preserved. Use the documented local **zero-shot rationale + `The answer is (X)`** prompt, deterministic final-answer extraction and no random guessing fallback. This differs from official few-shot protocols; 14 questions cannot establish benchmark ranking.
- [LiveCodeBench/code_generation_lite](https://huggingface.co/datasets/livecodebench/code_generation_lite), revision `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`, `test6.jsonl` only. The pinned official loader identifies this as the **v6 increment**, which is part of cumulative `release_v6`; it is not the complete release. Choose the first two easy and first two medium stdin tasks in original file order: `abc387_b`, `abc387_a`, `abc387_c`, `abc388_d` (2025-01-04 / 2025-01-11). **Only original public tests are run**, using whitespace-token equality. These are public-test smoke checks, never official hidden-test pass@1. No private pickle is deserialized. Functional-test tasks are not selected; unsupported scorer inputs return `pass:null`, not an automatic model failure.
- `lcb:abc387_a:repair` injects the explicit `(a+b)*2` bug into the same public task and uses its original public tests. This is a project-authored repair probe, aggregated separately from LCB generation. It follows its source task's heldout split, preventing dev/heldout leakage. Do not count the two related tasks as independent statistical units.

All selection is independent of model output. IFEval and MMLU dev/heldout alternate within the selected order; LCB alternates within its four fixed tasks. Repeats/seeds of the same task are not new independent questions. For model comparisons use matched task IDs/settings, per-benchmark pass counts and paired differences; Wilson intervals in the scorer are descriptive uncertainty for tiny task samples, **not a noninferiority or equivalence finding**. Four public programming problems cannot prove unchanged coding quality. No overall capability score is produced, and `quantization_equivalence_established` remains false.

## Data and release boundaries

The LCB dataset metadata says `license: cc` without a specific Creative Commons variant. The original contest problem rights remain with their sources. We consequently distribute **IDs, pinned download URLs, source hashes, selection code and scoring code**, not task text/public tests or raw generations. All materialized source data and task JSON live under root `results/quality-cache/`, which the repository ignores and the sdist include list does not include. Do not copy this directory, original task messages, public-test output expectations, or raw model reports into the release archive/site. Run the release archive verifier against those paths/content before publishing. `runner.py summary` emits only IDs/results/timings/configuration; `score` includes detailed test outputs and is for the private local evidence directory.

Raw reports must remain in `results/quality-cache/`, because they contain task messages. Derived aggregate/ID reports may be committed. Logs containing raw prompts or generated code also remain local. This is a distribution boundary; direct reproducible downloading from official public sources still works.

## Reproduce

Python 3.12+ runs scoring. Materialization additionally needs `pyarrow` to read the official MMLU-Pro parquet (tested with 25.0.1). Use an isolated temporary environment or an explicit temporary target, not a new runtime dependency of the inference package:

```sh
uv run --no-project --python 3.12 --with pyarrow==25.0.1 python benchmarks/quality/materialize.py
.venv/bin/python benchmarks/quality/runner.py selftest
.venv/bin/python benchmarks/quality/runner.py summary \
  --responses results/quality-cache/quality-qwen3.5-9b-8bit-public.json \
  --output benchmarks/reports/quality-qwen3.5-9b-8bit-public-summary.json
```

Materialization checks the SHA256 of every downloaded source against constants and writes a deterministic task-file hash to the committed manifest. Selection algorithms and sample IDs are reproducible; regenerating must preserve the manifest task IDs/splits/hash. Use `load_tasks(split=None)` and `score_task(task, answer_text)` from `runner.py`; neither loads MLX. Code runs in a fresh temporary working directory and separate Python `-I -S` process with wall/CPU timeout and file-size/FD limits, killing the process group afterward where host permissions allow (falling back to killing the child process in restricted environments). This is resource/process isolation, not a security sandbox for hostile programs; review new problem/code sources before execution.

## Generation report interface

Store one response per task per model/configuration run:

```json
{
  "task_id": "ifeval:3743",
  "setting": "recommended_non_thinking",
  "model": "mlx-community/Qwen3.5-9B-8bit",
  "revision": "exact-hub-revision",
  "quantization": "actual-loaded-config",
  "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5},
  "seed": 42,
  "enable_thinking": false,
  "max_tokens": 768,
  "output": "complete raw decoded response",
  "prompt_token_ids": [],
  "output_token_ids": [],
  "finish_reason": "stop",
  "elapsed_seconds": 1.23,
  "ttft_seconds": 0.12
}
```

`score --responses FILE` accepts a response list or `{ "responses": [...] }` whose `output` contains the final answer. `summary --responses FILE` accepts the quality runner's `{ "cases": [...] }` report, groups settings separately and removes a completed thinking section before scoring. Missing `</think>` in a thinking response produces an empty-answer failure; unfinished reasoning never passes output constraints accidentally. Generation limits/unfinished outputs remain visible as finish reasons. There is no model self-evaluation.

Task max-token values are predeclared in the manifest. If testing thinking, predeclare and retain the same increased reasoning budget per task across 8bit/4bit/14B before running; record each actual budget and sampling configuration. Official recommended sampling, engine greedy equivalence, thinking, non-thinking and quantization comparisons remain separate. Chinese/multiturn local smoke tests complement these public subsets; they are not relabeled as public benchmark evidence.

`compare_models.py` produces a release-safe paired report from completed model
reports. It refuses mismatched frozen source, corpus, task/settings, seed,
sampling or output caps. It lists per-task discordances, per-benchmark Wilson
intervals, normal/length completion and diagnostic resources separately. A
nonsignificant paired result does not establish quantization equivalence; these
quality-run timings are not a competing-framework speed benchmark.

```sh
.venv/bin/python benchmarks/quality/compare_models.py \
  --reports results/quality-cache/quality-public-qwen35-9b8-v2.json \
            results/quality-cache/quality-public-qwen35-9b4-v2.json \
            results/quality-cache/quality-public-qwen3-14b-v2.json \
  --output benchmarks/reports/public-quality-paired-v2.json
```

The independently preregistered development-only4096-token control is specified
in `docs/PHASE2_QUALITY_PROTOCOL.md`. It supplements the original finite-budget
scores without replacing them. Its V3 source retains exactly the V2 runtime and
locks, adding explicit `--task-ids`, `--output-cap`, and Qwen3 sampling-family
controls; archive/lineage hashes are in
`benchmarks/reports/phase2-quality-budget-source-v3-20261021/manifest.json`.
