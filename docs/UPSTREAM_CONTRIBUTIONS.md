# Upstream contribution candidates

Audited 2026-10-21. No issue, PR or maintainer message has been submitted.

## Dynamic prefill configuration

Target: [MLX-LM](https://github.com/ml-explore/mlx-lm), exact audited main base
[`72be8c955d9a0d3099bd01865af288e0e947c3fa`](https://github.com/ml-explore/mlx-lm/commit/72be8c955d9a0d3099bd01865af288e0e947c3fa).
`mlx_lm/generate.py` blob is `e2508684f5f23406ee580a968a8aa7f36267cb41`.
Its BatchGenerator class is byte-identical to the installed pinned 0.32.2 class.
No dependency upgrade was performed; main is not described as a stable release.

The integration currently changes `prefill_step_size` on the generator and its
private prompt batch. A narrow public `set_prefill_step_size()` method would keep
current and future prompt batches consistent. This is an API improvement
candidate, not a newly established numerical bug.

- [Reviewable patch](upstream/mlx-lm-prefill-setter.patch): validates a positive
  integer before updating either field; intended for the owner thread between
  `next()` calls. No scheduling algorithm, model kernels or brand dependency.
- [Standalone regression](../scripts/check_upstream_prefill_candidate.py): tiny
  random Llama, no weights or DeepInfer imports. Tests in-flight and future
  batches, cancellation/reinsertion, invalid values and empty state.
- Validation: unpatched class fails all three tests with missing-method errors;
  patched class passes all three on Metal. The runner isolates the patched class
  with installed primitives; this is not a complete latest-upstream CI run.
  No forward-path performance gain is claimed. Full upstream formatting and CI
  remain prerequisites for submission.
- Discovery: exact `set_prefill_step_size` PR query returned no matches. Broader
  searches found [#1541](https://github.com/ml-explore/mlx-lm/pull/1541)
  (length-aware admission), [#1087](https://github.com/ml-explore/mlx-lm/pull/1087)
  (adaptive long-context prefill), and
  [#943](https://github.com/ml-explore/mlx-lm/pull/943) (CLI prefill size).
  These are related areas, not proof that no duplicate proposal exists.

Apply the patch to the exact base, then run:

```sh
python scripts/check_upstream_prefill_candidate.py --source /path/to/patched/mlx_lm/generate.py
```

The [contribution rules](https://github.com/ml-explore/mlx-lm/blob/72be8c955d9a0d3099bd01865af288e0e947c3fa/CONTRIBUTING.md)
allow AI-assisted code with understanding and disclosure, but prohibit
AI-written public posts, including PR descriptions. This file is a local factual
engineering record. A human must author any eventual public description and
review/disclose AI assistance. No public message or ready-to-submit PR prose is
included in this candidate.

## Numerical reproduction

The self-authored diagnostic in `phase4_cache_diagnosis.py` separates cache
operations from Q=1/Q>1 forward partition differences. Current evidence does not
establish an upstream correctness defect. Keep it as a reproducible investigation
rather than proposing an unsupported bugfix or relaxing historical tolerance.
