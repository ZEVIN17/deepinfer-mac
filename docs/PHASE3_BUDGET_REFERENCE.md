# 跨请求验证预算：原始源码与独立 CPU 参考

此产物支持第三阶段 4.1/4.2 的算法核查。它没有接入生产路径，没有运行
MLX/Metal/CUDA，没有下载权重，也没有证明吞吐、延迟或置信头质量。

## 固定来源与可复现命令

固定 `deepseek-ai/DeepInfer@9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`。
已有 MTP header 不改动；两个必要原始 Python 文件保存在
`tests/reference/phase3_vendor/`，归属和 Apache-2.0 许可证见其 README。
Git blob 与 `docs/phase3-source-audit.json` 逐项一致。回放脚本会检查三份原文
Git blob 以及已有 DeepInfer 许可证 blob，避免把修改过的参考当原版。

在仓库根运行，无模型、MLX、Torch、TileLang 依赖：

```bash
.venv/bin/python tests/reference/phase3_budget_replay.py --cpp-oracle \
  --output docs/evidence/phase3-budget-source-reference.json
.venv/bin/ruff check tests/reference/phase3_budget_replay.py
```

`--cpp-oracle` 需要本地 C++17 编译器，运行已有 `mtp_oracle.cpp` 和 CPU
`Sequence` shim 包装的**未修改 C++ header**。不带该参数时只有 Python CPU
数学参考。这里的 C++ 执行不包括 DeepInfer 完整引擎、GPU 内核或模型。

## 概率契约

- [设备调度原文第 26–37 行](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/tilelang_ops/cuda/schedule_mtp_varlen.py#L26-L37)：
  `prev_confidence_rates` 是逐位置 conditional 值；每行以 float32 乘法计算
  `s[j] = product(p[0:j+1])`。`remap == -1` 将整行累计值设为 0。
- [CPU 状态原文第 123–126 行](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/sequence.hpp#L123-L126)：
  `accept_rates` 明确为“累乘后”，`confidence_raw` 明确为“累乘前”。
  已有 C++ MTPVarlen 的输入不能再做一次 cumprod。
- [DFlash 第 46–79 行](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/models/mtp/dflash.py#L46-L79)、
  [第 180–186 行](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/models/mtp/dflash.py#L180-L186)：
  ConfidenceHead 有模型提供的 projection 权重、hidden/Markov 输入、逐位置温度，
  委托 `confidence_head_fused` 计算。此核查没有运行融合头或验证校准，不能
  将草稿 softmax 最大值或其它架构的任意权重标为该训练头。

回放刻意设置反例：累计值 `[[0.5, 0.3], [0.28, 0]]` 在 `total_q=4`
时得到 `[3,1]`；错误地再 cumprod 后变成 `[2,2]`。两类输入入口明确分开。
概率参考校验 finite、[0,1]，累计概率还须逐行非增；这些是未来接口应保持的
显式守卫，并不声称上游 TileLang 内核自行检查错误输入。

## 跨请求排序、预算与 remap

[设备第 49–81 行](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/tilelang_ops/cuda/schedule_mtp_varlen.py#L49-L81)
对当前 batch 所有 `(row, position)` 比较累计概率。概率较大者先，平局按
`row * nextn + position` 较小者先。不是按 request-id 字符串排序。
每行累计概率非增，因此全局所选位置对每行自然形成连续前缀，行长为
`1 + selected_draft_positions`，其中 1 是必需的 golden token。

对于 B 个活跃请求、每行 D 个 draft 位置，合法预算是
`B <= total_q <= B * (D+1)`，且 `sum(seqlens_q) == total_q`。
这里 `total_q` 包含 golden tokens，不是额外 draft 个数。内核裸公式对低于
下界的预算输出全 1，对高于上界的预算输出全 `D+1`；它不会凭空满足非法
预算。参考分别核验这种饱和行为与未来入口应拒绝非法预算的契约。

fresh 行置零**不意味着禁止其分配**。当其 flat index 更小、其它候选也为零，
或预算足够大时，它仍可能获得额外位置。`[-1, 0]`、上一行概率为全零、
`total_q=3` 的手算结果是 `[2,1]`。不要把“新请求置零”改写为“新请求永不验证”。

[worker 第 195–219 行](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/worker/worker.py#L195-L219)
表明该函数用于 ZOS 交集的上一轮结果修正：CPU 已选 `total_q`，GPU 依据
当前 remap/上一轮 conditional confidence 重分行长。固定请求 identity 与
slot_generation 的 fixture 核验重新排序、取消后新增、slot 复用得到新请求；
它是未来本地接口的状态约束，不声称上游有同名 identity_remap API。

[MTPVarlen 第 163–208 行](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/mtp_varlen.hpp#L163-L208)
用 N−2 历史选择总预算，用 N−1 分配行长；未配对状态仍遵守历史隔离。
[第 231–265 行](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/mtp_varlen.hpp#L231-L265)
按期望收益/图成本选择 `best_bsz`，随后给每行先放 1，再分配
`best_bsz-B` 个位置。`graph_size` 是成本曲线的向上匹配尺寸，可能大于
实际 `best_bsz`；不是每行长度或实际有效 token 数。无曲线时直接采用各行
`1 + draft_tokens.size()`，行长可以不同，不能假定所有行都为共享 cap。

## 核验范围与结果

固定 fixture 14 项：空 batch（仅 CPU 数学定义）、零 draft（仅 CPU）、全零、
全一、同值 tie、跨请求 rank、fresh、remap 重排、golden-only、满预算、
单请求、非法上下界。另有 2,025 项穷举：两行两位置、条件概率取
`{0,0.5,1}`、5 种 remap、5 个合法总预算。逐项比较忠实的 all-pairs rank
公式和独立稳定排序表达，并断言连续前缀与预算守恒。两种表达均在此
独立参考文件中，不导入生产实现；手算 JSON fixture 提供额外锚点。

原始 C++ header 的 3 个实际 CPU 场景得到 `[3,3]`、`[1,1]`、`[3,2]`，
行长和分别 6、2、5（便宜的大图、昂贵的大图、无曲线变长行）。未运行
上游 TileLang/CUDA 设备 oracle；空/零维场景仅为 CPU 契约，不能推断
GPU 对零 launch 或 zero-sized reducer 的支持。float32 乘法逐步舍入；
此次穷举选用二进制精确值，未验证设备端 subnormal/flush-to-zero 差异。

经验接受率的数据契约另以 fixture 明示：已核验 `[true,true]` 后遇 EOS、
输出预算或取消，余下为 `[null,null]`（删失），不是拒绝；只有实际观察到
首个不一致才可标 `[true,true,false,null]`。此处没有建立训练器或新的预测头。

后续生产实现仍须在请求 tick/资源准入基础上接入，再与本参考、实际设备、
dense/hybrid 状态隔离、执行 token 工作量和完整成本作独立对照。此报告
不表示跨请求预算已运行或已取得推理加速。
