# Contributing / 贡献指南

Use Python 3.12/3.13. Preserve existing work, keep changes focused, and add behavioral checks for decoding/cache/worker changes.

```sh
uv sync --frozen --extra dev
uv run pytest -q -m 'not metal'
python3 site/build.py
python3 site/check.py
uv build
```

On Apple Silicon add `--extra mlx` and run real model checks separately. Portable CI success does not mean Metal inference passed. Inspect `docs/PROGRESS.md` for current evidence, limits and reproducible commands.

For benchmark contributions, include hardware/macOS/power state, immutable model revisions, dependency lock, corpus, generation settings, warmup/cache states, repeats, full JSON and every failed or divergent run. Share only prompts you have permission to publish. Do not submit weights, secrets, downloaded model caches or personal paths. Do not relabel inspired work as an official upstream port.

中文：先运行上述无硬件测试；Apple Silicon 增加 `--extra mlx` 再独立验证真模型。性能贡献请提交机器、系统、电源状态、模型 revision、锁文件、完整语料和设置、冷热状态、重复次数及全部 JSON；失败和 token 不一致不能删去。仅公开可分享输入，不提交权重、凭据或私人文件。说明新增行为及验证证据，保持第三方许可证和来源。
