# Retired 27B experiments / 27B 历史实验

The active model choice moved to Qwen3.5-9B-8bit, with same-model 4bit and a 14B fallback under evaluation. Qwen3.8-27B is **not** the priority candidate or an active API/UI profile. Its obsolete weights and partial transfer chunks have been removed with the user's authorization; do not follow old download/resume instructions as the current build path.

## What was verified

The original 4bit audit pinned `mlx-community/Qwen3.8-27B-4bit@10c35caafbb80f7dc6a7a432cdd11af10a6d4818`. [qwen27.manifest.json](qwen27.manifest.json) preserves header/identity/resource-estimate evidence. The available-memory preflight rejection was a loading estimate, not an actual OOM or measured runtime peak.

The later `keXjos/Qwen3.8-27B-mlx-2Bit@39d5487ad4442961b64098e15aa11ce05e6f9976` experiment SHA-verified two shards totaling 8,409,705,283 bytes and actually loaded through MLX-LM. Seven engine requests completed at about 24.4–25.8 decode tok/s, with cold/hot/branch cache and stream concatenation consistency. First-request system swap increased; later request deltas of zero cannot erase that observation.

Chinese/code/arithmetic quality checks failed, including repetition and incorrect/incomplete code. Native MLX-LM with the same greedy template produced identical tokens on the compared tasks. This demonstrates a shared failure at those settings; it does not isolate quantization as the sole cause. The experiment did not qualify as the showcase model and did not prove competitive superiority.

## Preserved evidence

- [2bit manifest](qwen38-2bit.manifest.json).
- Project reports `benchmarks/reports/qwen38-27b-2bit-demo.json`, `qwen38-native-quality.json`, `qwen38-quality-matrix-kexjos.json`, and `qwen38-service-probe.json` retain actual outcomes.
- `artifacts/phase2-model-audit/model-cleanup-20261021.json` records exact removed directories, process checks and byte counts. Six obsolete directories represented 19,177,137,641 logical bytes (17.86 GiB); unrelated models were retained.

The 3bit/mixed-bit alternatives in the historical audit were metadata candidates, not successful local quality results. No speed/quality is inferred for unrun quantizations. Current entry points and evidence are in [Phase 2](phase2.md) and [model selection](PHASE2_MODEL_SELECTION.md).

中文：保留失败结果和模型来源，删除不用的权重；不把旧预检当 OOM，不把真实加载当质量通过，也不把旧 27B 数字迁移到新 9B。
