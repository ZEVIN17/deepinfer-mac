# 第二阶段模型选型：24GiB，能力与实用性优先

核查日期：2026-10-21。此文件为选型依据，不是新模型本机验收报告。用户取消27B/2bit约束，并明确在资源/适配条件接近时优先模型能力和跑分。执行细节以 PHASE2_EXECUTION_PLAN.md 为准。

## 决定

首选 `mlx-community/Qwen3.5-9B-8bit`；同模型4bit做质量/效率对照。Qwen3-14B-4bit是工程后备与DS推测调度控制模型，Qwen3-8B-4bit是更轻量后备。不能仅因14B标准注意力易适配就把它作为能力最强的选择。

- [9B 8bit仓库](https://huggingface.co/mlx-community/Qwen3.5-9B-8bit)：本轮API核验revision `16daa4818c54ce5f5436f929d52542eb65bbed9d`，全部safetensors合计10,426,592,423 bytes，约9.71GiB。包含视觉相关权重，MLX-LM纯文本实际驻留需测。API快照在 `artifacts/phase2-model-audit/qwen35-9b8-selection-api.json`。
- [14B 4bit仓库](https://huggingface.co/mlx-community/Qwen3-14B-4bit)：revision `a4d9b2df59d2c150bef02fcbe0d91046b7ca33a4`，权重8,307,898,514 bytes，约7.74GiB。[配置](https://huggingface.co/mlx-community/Qwen3-14B-4bit/blob/main/config.json)为4bit/group64、40层、8KV头、head_dim128。API快照在 `artifacts/phase2-model-audit/qwen3-14b-selection-api.json`。
- 本机psutil读取物理内存25,769,803,776 bytes，即24GiB。权重大小接近不代表总内存/延迟相近：需要计入KV、循环状态、草稿、快照和临时buffer。本轮未下载或运行新模型，不宣称无swap或某个速度。

## 能力与跑分依据

[Qwen3.5-9B官方模型卡](https://huggingface.co/Qwen/Qwen3.5-9B)报告：MMLU-Pro 82.5、GPQA Diamond 81.7、IFEval 91.5、LiveCodeBench v6 65.6、BFCL-V4 66.1。这些是官方模型评测，不是本项目MLX 8bit/4bit复测，也不是本机non-thinking分数。此表不含Qwen3-14B同条件结果，不能据此计算9B相对14B的提升百分比。代码、知识和工具任务并非同一能力，不合并为未经定义的总分。

以这些公开能力证据及较小权重为依据，优先验证9B的实际任务表现是本轮工程判断，不是“全榜最强”结论。8bit用于减少量化扰动的初始选择；若本机4bit在预先固定题集上质量无实质退化且更快，Codex可自主选择4bit。反过来，不能仅因更快就接受明显质量下降。

后续为每项跑分附：模型revision/位宽、评测版本、思考模式、输出预算、采样、样本数及来源级别。官方自报、第三方独立复现、本机复测分栏；缺数据留空，不做跨协议精确排名。开发者任务优先可执行代码单测、修复成功率、指令约束、中文解释与事实提取；公开推理跑分只作为辅助，真实任务质量决定展示是否合格。

## 生态与开发者实际可用性

[Ollama Qwen3.5](https://ollama.com/library/qwen3.5)和[Qwen3](https://ollama.com/library/qwen3)均有模型系列分发及开发工具使用入口；HF存在MLX量化，降低用户复现与跨框架比较门槛。它们证明生态可获得性，不证明完整编码agent效果。页面下载数/点赞数随时间变化，系列下载不能当作某个量化的口碑得分；本次没有系统性开发者满意度调查，不编造“公认最佳”。

[Qwen2.5-Coder-14B-Instruct-4bit](https://huggingface.co/mlx-community/Qwen2.5-Coder-14B-Instruct-4bit)保留代码专项参照；[Qwen3-8B-4bit](https://huggingface.co/mlx-community/Qwen3-8B-4bit)保留轻量对照。模型选择不依赖新名字或参数量本身。

## DS方法与实际适配成本

[9B官方配置](https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/config.json)为混合线性/全注意力。当前安装的 `mlx_lm/models/qwen3_5.py` 为线性层构造ArraysCache，当前项目 `speculative.py` 要求可trim缓存，因此9B不能直接套用已有Qwen3推测回退。这是需要解决的工程成本，不是质量较好候选自动落选的理由。

14B标准Qwen3使用可trim KV缓存，可先验证DS成本/预算方法；草稿仍须完整词表/special IDs/状态与数值校验。Qwen3小草稿不能跨词表直接用于9B。9B可研究精确状态恢复、真实可用的MTP/兼容draft，或其他有具体DS来源的执行机制。生产调用、正确性及开关消融必须在最终展示模型完成，不能用14B结果包装9B。

“模型选型带来的质量提升”与“同模型同质量下DS方法带来的框架性能提升”独立展示。最终仍需相对常用框架的公平实测优势；若只有模型强、引擎没有优势，第二阶段目标仍未完成。
