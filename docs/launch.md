# Build, share, contribute

DeepInfer-Mac brings native MLX batching, streamed chat and an inspectable execution layer to Apple Silicon. The bundled browser workspace makes the same engine approachable from the first prompt.

## Reproduce the release

```sh
./scripts/release-check.sh dist/current
uv run pytest -q -m metal --run-metal
```

The build produces a wheel, source distribution and an eight-page bilingual showcase. [Release entry](RELEASE.md) contains install commands and evidence. The GitHub Actions workflows validate pushes, upload build artifacts and deploy Pages from main. Published historical package bytes remain unchanged.

## Project introduction

**English:** Meet DeepInfer-Mac: a local inference framework for Apple Silicon, powered by MLX. Run from your terminal, connect a streamed chat API, or open the included browser workspace. Native batching, bounded memory and inspectable scheduling give builders a clear path from first prompt to reproducible measurement. Explore the real demo, read the architecture, and contribute your Mac's results.

**中文：** DeepInfer-Mac，让本地模型更懂你的 Mac。以 MLX 为执行后端，连接终端、流式 API 与浏览器工作台。原生批处理、限额内存管理和可核查的调度，让开发者从第一条提示词走到可复现的真实测量。打开演示，了解架构，带着你的 Mac 一起完善它。

These are repository launch materials. No external community messages are sent automatically. Detailed measurements retain their hardware, model, workload and correctness scope in the linked technical reports.
