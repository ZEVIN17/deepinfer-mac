# DeepInfer-Mac

**由 MLX 驱动、面向 Apple Silicon 的本地推理，并实际实现来自 DeepInfer 的可执行方法。** 独立社区项目 · Apache-2.0 · 实验性 alpha。

[English](README.md) · [第二阶段证据](docs/phase2.md) · [架构与归属](docs/architecture.md) · [复现基准](docs/benchmarking.md) · [兼容矩阵](docs/compatibility.md)

当前首选候选为 **Qwen3.5-9B-8bit**。24 GiB M4 Pro 已实测加载、普通/采样生成、API/SSE、取消恢复、缓存一致性和本地 UI。相同模型 4bit 及 14B 后备的质量/效率对照仍在进行，尚未完成最终模型验收。不再使用的 27B 权重已经清理，失败实验及来源记录保留。

DeepInfer MTPVarlen 派生调度已进入真实执行路径，使用经验接受率，并对照原始 C++ 验证。混合状态快照/恢复/重放适配在一个真实 9B 提示上取得 baseline、fixed1、fixed2、DS 各 96 token 一致。**尚未证明优于调优后的固定预算或两个常用框架，第二阶段联合目标仍未完成。**

## 安装与运行

需要 Apple Silicon、Metal、Python 3.12/3.13；模型权重单独下载，需为活跃状态、快照和系统预留内存。

```sh
git clone https://github.com/ZEVIN17/deepinfer-mac.git
cd deepinfer-mac
uv sync --frozen --extra mlx --extra dev
uv run deepinfer-mac doctor
uv run deepinfer-mac models --download qwen3.5-9b-8bit --target-only
uv run deepinfer-mac run --profile qwen3.5-9b-8bit --mode baseline '用中文解释统一内存。'
uv run deepinfer-mac serve
```

打开 `http://127.0.0.1:8765`，选择 9B 候选。默认 profile 仍为较小的 Qwen3-4B 回归模型。公开展示站是静态页面，不会连接访客 localhost。当前不假设 PyPI 已发布；[v0.1.0a1](https://github.com/ZEVIN17/deepinfer-mac/releases/tag/v0.1.0a1) 为历史版本，工作树准备 0.1.0a2 实验更新。

baseline 现支持 temperature、top-p、top-k、presence penalty 与 seed；推测模式仍严格限定 greedy。

```sh
uv run deepinfer-mac run --profile qwen3.5-9b-8bit --temperature 0.7 --top-p 0.8 --top-k 20 --presence-penalty 1.5 --seed 42 --json '本地模型如何帮助开发者？'
uv run deepinfer-mac run --profile qwen3-4b --mode adaptive --adaptive-scheduler ds --no-cache --json '写一个 Python 二分查找。'
uv run pytest -q
uv run pytest -q --run-metal -m metal
python3 site/build.py
python3 site/check.py
uv build --out-dir dist/v0.1.0a2
python3 scripts/verify_artifacts.py --dist dist/v0.1.0a2
```

公开 9B API/UI 仅开放 baseline，混合推测使用 [DS 对应文档](docs/DS_DESIGN_MAPPING.md) 中的显式实验 Python 入口。没有专用 Confidence Head、完整 DSpark、连续批处理或完整 DeepSeek V4.1。

## 实验结论与限制

实际 9B UI 的 clamp 修复请求总耗时 1834.24ms、TTFT 939.62ms、decode 27.94tok/s。这是单次交互，不能当竞争基准。固定 12 题本地质量 smoke 保留了事实/算术失败；greedy 引擎与 native 在全部 12 题 token 一致。公开子集及量化比较使用冻结源码，发布脱敏汇总，不分发受限题目正文。

历史 alpha Qwen3-4B 有 216 次正式运行，baseline/native 为 36/36 一致，所有模式为 27/36 一致；自适应未优于调优固定预算，不能宣称等价加速。这些数字不代表新 9B。

可 trim 模型复用 MLX-LM 原生 LRU，混合模型保留精确快照与后缀重放。缓存改进属于独立优化/平台适配，不算 DS 核心贡献。统一内存 RSS 与 Metal allocator 不能相加。频繁重载模型的进程 RSS 驻留仍待解决；此类基准使用独立进程。

[模型锁](models.lock.json) · [证据索引](docs/evidence.md) · [版本说明](docs/release-notes.md) · [第三方归属](NOTICE) · [贡献指南](CONTRIBUTING.md)
