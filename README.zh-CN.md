# DeepInfer-Mac

**面向 Apple Silicon、受 DeepInfer 启发、由 MLX 驱动的本地推理。** 独立社区项目，Apache-2.0，实验性 alpha。

[English](README.md) · [架构与归属](docs/architecture.md) · [基准复现](docs/benchmarking.md) · [兼容矩阵](docs/compatibility.md) · [贡献指南](CONTRIBUTING.md)

默认使用普通 baseline；固定草稿和自适应推测需显式选择。

本项目运行本地量化模型，对比普通解码、固定长度推测解码、自适应 greedy 推测解码，并提供按字节预算管理的精确前缀缓存。本地聊天和比较界面随 wheel 分发。不承诺普遍加速，不把数值差异或失败样本隐藏。首版未移植官方 DSpark 或完整 DeepSeek V4.1。


**优先主展示候选：** Qwen3.8-27B MLX 4-bit 正在核查加载、Metal 资源和稳定性，尚未成为默认。首阶段只验收纯文本 baseline 与精确缓存；混合循环状态和不同词表意味着不能沿用 Qwen3-0.6B 草稿。Qwen3-4B 保留快速回归与推测对照，小模型结果不能归因于 27B。

## 安装与运行

需要 Apple Silicon、Metal 和 Python 3.12/3.13；为模型、KV 和系统预留内存与磁盘。首台验证设备是 24 GB M4 Pro。

```sh
git clone https://github.com/ZEVIN17/deepinfer-mac.git
cd deepinfer-mac
uv sync --frozen --extra mlx --extra dev
uv run deepinfer-mac doctor
uv run deepinfer-mac models
uv run deepinfer-mac models --download qwen3-4b
uv run deepinfer-mac run --profile qwen3-4b --mode baseline '用中文解释推测解码。'
uv run deepinfer-mac serve
```

浏览器打开 `http://127.0.0.1:8765`，可聊天、比较三种模式、导出 JSON。首次使用会下载锁定 revision 的 Hugging Face 权重；发布包不包含权重。`qwen3-0.6b` 可运行较小模型基线。模型、tokenizer 和许可证见 [models.lock.json](models.lock.json)。

也可用 `python3 -m pip install 'dist/deepinfer_mac-0.1.0a1-py3-none-any.whl[mlx]'` 安装本地构建包。当前以仓库和本地产物作为安装来源，不假设 PyPI 已发布。

## 真实比较

```sh
uv run deepinfer-mac run --mode fixed_speculative --draft-tokens 2 '写一个 Python 二分查找。'
uv run deepinfer-mac run --mode adaptive --json '写一个 Python 二分查找。'
uv run deepinfer-mac bench --profile qwen3-4b --suite quick --repeats 3 --fixed-lengths 1 2 4 --output benchmark-results
uv run pytest -q -m 'not metal'
# Optional: deterministic Metal control tests on Apple Silicon
uv run pytest -q --run-metal -m metal
python3 site/build.py
python3 site/check.py
uv build
```

基准包含同版本原生 MLX-LM、完整 token 比较、轮换顺序、多种固定长度和独立缓存复用测量。冷缓存不等于冷模型；TTFT 包含边界必须按报告阅读。RSS 和 Metal allocator 不是可相加的独立内存。详见 [复现教程](docs/benchmarking.md)。

首组 M4 Pro 完整全语料没有证明等价推测加速：零失败，但仅27/36组所有模式完整token一致；自适应没有优于调优后的固定预算。普通baseline与原生MLX-LM为36/36一致。完整数据与实测源码归档见 `benchmarks/reports/m4-pro-qwen3-4b/`。

## 相比原生 MLX-LM 的增量

独立成本策略与零草稿回退；按模型身份隔离、克隆和 LRU 淘汰的精确缓存；单一推理所有者、有界队列、取消和本地 UI；模型锁、诊断及完整结果导出。

不同矩阵形状可能产生浮点 argmax 差异；每个工作负载必须验证完整 token 序列，并公开不一致样本。采样分布保证、专用 Confidence Head、V4.1 CED/CSA2/MTP、分布式服务和自定义 Metal 算子仍是后续路线。

[版本说明](docs/release-notes.md) · [路线图](docs/roadmap.md) · [第三方归属](NOTICE) · [来源锁](upstream.lock.json)

已知限制：频繁切换/重载模型时，Metal数组和cache虽释放，进程RSS仍可能因native/Rust allocator驻留而增长。此类长期会话建议独立进程或重启服务；固定模型缓存稳定性另有实测，不能混为一谈。
