# 实验要求：单张 RTX 4090 上的 Qwen3.8-27B INT4 推理优化

状态：v1 学生实验说明。机器评分权威仍是 `contract-v1.json`；图片能力权威是
`multimodal-contract-v1.json`；一周日程、workflow 创新加分和最终报告权威是
`course-policy-v1.json`。本文负责把合同翻译成实验流程，不建立第二套分数、门槛或例外
规则。若文字说明与 JSON 合同冲突，以对应 JSON 合同或政策为准。

## 1. 实验目标

在公开的 ApxInf 接口后实现或优化固定版本的
`cyankiwi/Qwen3.8-27B-AWQ-INT4`，使它在一张 NVIDIA RTX 4090 上完成正确、稳定、可复现
的推理，并用端到端证据解释优化为什么有效。

本实验不是一个只比较单个 CUDA kernel 吞吐的比赛。你需要同时处理四类系统问题：

1. 模型语义：Qwen3.8 的 GDN/全注意力混合主干、W4A16 权重和确定性解码；
2. 服务边界：请求、KV/recurrent state、流式 token、故障恢复和并发隔离；
3. 性能边界：prefill、decode、长上下文、显存和调度；
4. 证据边界：公开/隐藏 correctness、客户端计时、MFU/BWU proxy 和可复现 artifact。

## 2. 学习目标

完成实验后，学生应能：

- 区分模型结构名称、checkpoint 格式和服务实现，理解 `qwen3_5` 是该权重的正式
  architecture identity，而不是把模型错误命名成 Qwen3.5；
- 区分 prefill、decode、TTFT、TPOT、goodput、上下文容量与服务尾延迟；
- 为量化模型建立 correctness gate，而不是用“输出看起来合理”代替验证；
- 从端到端路径定位瓶颈，再选择 kernel、布局、缓存、图执行或调度优化；
- 正确解释 MFU/BWU proxy 的分子、分母和适用边界；
- 用 PR 提交实现、测试、原始结果、负结果和设计判断。

## 3. 冻结条件

以下条件不可由学生修改：

| 项目 | 冻结值 |
|---|---|
| 模型 | `cyankiwi/Qwen3.8-27B-AWQ-INT4` |
| 模型 revision | `63768c10df38c0395e12ef49edac1bd539eaeeea` |
| architecture | `Qwen3_5ForConditionalGeneration` |
| 量化 | compressed-tensors W4A16、group size 32、asymmetric |
| 硬件 | 单张 NVIDIA RTX 4090，SM89 |
| 基础并行 | TP=PP=DP=1，基础任务一次一个请求 |
| 采样 | greedy，`temperature=0`，thinking 关闭 |
| 基础输入 | 教师预分词的精确 `input_ids` |
| 基础输出 | 完整、连续、带 request ID 的 token-ID SSE |
| 基础性能预算 | 每个 cell 生成完整 128 token |
| 正式计时 | 客户端从发送请求开始计时；server timing 只作诊断 |
| fallback | 禁止静默调用 vLLM、Transformers、CPU 或另一个模型 |

教师实现、模型文件路径、隐藏 prompt、隐藏 seed、expected output、reference token IDs 和
正式评分环境不会随 starter repository 发布。

## 4. 实验赛道

实验由一个必做主榜、两个计分 bonus 和一个非计分能力徽章组成：

| 赛道 | 是否必做 | 分值/结果 | 核心问题 |
|---|---:|---:|---|
| 单请求文字主榜 | 是 | 100 分 | 正确、稳定地改善 prefill 和 decode |
| 长上下文 | 否 | 0–10 分 | 在完整任务验证下把容量推进到 32K 以上 |
| 多请求 serving | 否 | 0–10 分 | 在 correctness 与 tail guard 下提升 correct goodput |
| 真实图片输入 | 否 | 能力徽章，v1 不加分 | 接通 processor、视觉塔、media embedding 和 mRoPE |
| Workflow 创新 | 否 | 教师评审 0–5 课程加分 | 改善实验效率、可复现性或证据质量 |

自动 leaderboard 展示上限为 120。图片徽章不进入 v1 分数；未来若要加分，必须发布新
合同并重新冻结 cohort，不能在活动轮次中追溯改分。

## 5. 阶段 A：基础环境与接口

学生首先需要完成：

1. clean checkout 能够构建；
2. 服务能够在指定模型 revision 上启动；
3. `GET /health` 返回真实身份、容量、并发、fallback 和 capability；
4. `POST /v1/evaluations/generate` 接受预分词 token IDs；
5. 非法 JSON、非法 token、非零 temperature 和超容量请求明确失败；
6. 错误请求后服务仍然健康；
7. 不支持的可选能力必须 fail closed，不能忽略输入后返回 200。

完成标准不是“进程没有退出”，而是公开协议负控制全部通过。

## 6. 阶段 B：Correctness gate

Correctness 共 30 分，同时是性能排名资格门槛。

### 6.1 公开功能题

公开功能题共 6 个：

- 1K early/middle/late 精确检索各 1 个；
- 8K 多跳关联、版本覆盖、整数聚合各 1 个。

长文本背景使用固定 SHA-256 的 Project Gutenberg《紅樓夢》。小说只是自然语言背景；
机器评分答案来自生成器插入的确定性课程档案。开放式文学解释只作演示，不进入自动分数。

### 6.2 隐藏功能题

隐藏功能题共 12 个，均不超过 16K：检索 4、干扰消歧 2、多跳 2、版本覆盖 2、聚合 2。
公开功能题必须 6/6；正式隐藏功能题至少 11/12，提交才有资格进入 TTFT/TPOT 排名。

### 6.3 Token trajectory

公开和隐藏各有两条 128-token greedy trajectory。对离散 token ID 序列计算单位代价
Levenshtein 编辑相似率；不得把 token ID 当连续数值计算 MAE、cosine 或 embedding
距离。Trajectory 影响 correctness 分并且必须提交完整证据，但不作为性能资格的一票否决。

## 7. 阶段 C：单请求性能主榜

| 部分 | 分值 | Cells |
|---|---:|---|
| TTFT | 35 | 1K、2K、4K、8K、16K prompt，均输出 128 token |
| TPOT | 25 | 1K、8K prompt，均输出 128 token |
| Reliability | 10 | 成功率、OOM、NaN、fallback、Xid、失败后恢复 |

正式 leaderboard 每个 latency cell：

- warm-up 1 次；
- measured repeats 5 次；
- 使用中位数；
- TTFT/TPOT CV 不超过 10%；
- 保存所有原始样本和环境证据。

vLLM 是同机强制 control，但不是固定满分。每个 cell 以当轮 vLLM 与所有 eligible PR 中
的最好有效中位数为满分参考：

```text
cell_points = weight × min(1, best_valid_median / observed_median)
```

因此，学生实现可以在某些 cell 超过 vLLM，并成为新的动态参考。

## 8. 阶段 D：长上下文 bonus

32K 是 0 分起点，不是满分点：

```text
32,640（诊断，不计分）
→ 32,768
→ 65,536
→ 131,072
→ 196,608
→ 262,016 prompt + 128 output
```

在申报的最大成功长度必须同时通过 early/middle/late 检索、多跳、版本覆盖和聚合六类题，
并生成完整 128 token。未达到原生上限时，必须记录第一个更大长度的失败边界，并证明失败
后的 health 与小请求能够恢复。只修改 `/health.max_model_len` 不得分。

## 9. 阶段 E：多请求 bonus

同一单请求 endpoint 上执行 closed-loop C4 和 C8，每个 cell 共 32 个 1K/128 请求。
Queueing 属于 TTFT 和 makespan。Cell 只有在以下条件全部满足时才有效：

- success=100%；
- correctness=100%；
- Jain fairness ≥ 0.95；
- 无 fallback；
- 服务结束后健康；
- p95 TTFT/TPOT 通过合同中的 candidate-relative tail guard。

每个有效 cell 先获得 1 分接口支持分，再按同轮最好 correct goodput 比例获得最多 4 分。
串行排队可以保持主榜合法，但通常无法获得有竞争力的 goodput bonus。

## 10. 阶段 F：真实图片能力徽章

图片能力不是检查 `config.json` 是否包含 `vision_config`，而是验证完整的集成路径：

```text
PNG → processor → pixel/grid → vision encoder → merger
→ image-token embedding → multimodal RoPE → language model → exact answer
```

公开 4 题覆盖七段数码 OCR、空间颜色、柱状图算术和目标计数；隐藏 8 题为同四类各 2
题。图片、JSONL 和 manifest 均使用 SHA-256 验证。

- 不支持图片：报告 `multimodal=false`，图片探针必须返回机器可读的
  `unsupported_capability`；
- 支持图片：报告 `multimodal=true`，必须执行真实 image content-part；
- 公开 4/4：获得 `multimodal-public-pass`；
- 公开 4/4 且隐藏 8/8：获得 `multimodal-ready`。

返回 200 但忽略图片、硬编码公开答案、调用隐藏 fallback，或图片请求后破坏文字服务，均
视为失败。

## 11. MFU、BWU 与 profiler 要求

`compute_efficiency.py` 的结果不直接加分，但每个 PR 必须在分析中说明至少一个代表 cell
的效率结果或无法计算的原因：

- `estimated_mfu_bf16_equivalent_pct`：冻结 dense-equivalent FLOP proxy；
- `minimum_model_bwu_pct`：冻结最低模型字节 proxy；
- `profiled_bwu_pct`：仅在 Nsight Compute 提供 phase-scoped DRAM bytes 和 kernel
  elapsed 时产生。

不得：

- 把 INT4 TOPS 直接当整个 W4A16 模型的真实 MFU 分母；
- 把 `nvidia-smi utilization.memory` 当 HBM 带宽利用率；
- 用 profiler 内计时替代正式客户端 TTFT/TPOT；
- 用单个 kernel speedup 宣称服务端到端等比例加速。

NSys/NCU 不是每次迭代的强制步骤。只有当它能回答当前因果问题，或 PR 声称 launch、
overlap、带宽、occupancy、Tensor Core/SOL 原因时，才需要保留对应 capture。

## 12. 推荐实验方法

每个优化候选建议按以下记录推进：

1. 冻结一个 workload cell 和原始输入 hash；
2. 写出可证伪假设：要删除、融合、重排或加速什么；
3. 保留上一个已接受 baseline；
4. 做最便宜的 operator/layer smoke；
5. 检查完整 token trajectory、状态副作用和失败路径；
6. 做相同机器、相同 revision、相同输入的端到端比较；
7. 需要时用 NSys/NCU 解释，而不是用 utilization 猜测；
8. 记录接受、拒绝、中性或未决结果；
9. 只有通过目标 cell 的 correctness、稳定性和端到端证据后才替换 baseline。

负结果是合格实验的一部分。一个被清楚验证并关闭的错误方向，比没有证据地保留五个
“也许有效”的分支更有价值。

## 13. 一周任务周期

课程周期为 7 天，使用 Asia/Shanghai 时区。

| 时间 | 目标 | 主要产出 |
|---|---|---|
| 第 1 天 | 环境、接口、baseline | 构建与服务 smoke、公开协议、自测 artifact |
| 第 2 天 | Correctness 与选题 | 公开功能通过、一个主要优化 cell、可证伪假设 |
| 第 3 天 | 第一版候选 | 最小端到端实现、正确性与初步性能证据 |
| 第 4 天 19:00 前 | 中期提交 | 固定 PR SHA、公开 artifact、中期简报 |
| 中期提交后 | 教师统一评测 | clean checkout、hidden、vLLM control、cohort scoring |
| 评测完成后 | 公布中期榜 | 命名 leaderboard snapshot 与简短反馈 |
| 第 5 天 | 分析中期结果 | 根据 leaderboard 和反馈收敛最终假设 |
| 第 6 天 | 最终候选与报告 | 完整验证、artifact、详细报告草稿 |
| 第 7 天 19:00 前 | 最终提交 | 最终 PR SHA、`REPORT.md`、全部证据引用 |

中期截止时，教师只接收截止前可获取的完整不可变 PR commit SHA。截止后的修改进入最终
版本，不回填中期 cohort。教师完成全部提交和 vLLM control 的统一评测后再公布榜单，不
发布只包含部分同学的滚动结果。

中期提交至少包括：

1. 完整 PR commit SHA；
2. 该 SHA 生成的公开 evaluator artifact；
3. 简短中期总结：baseline、候选、目标 cells、当前结果、失败分支和已知限制。

## 14. Workflow 创新加分：0–5 课程分

Workflow 创新分不改变 120 分 leaderboard，也不会影响任何 cell 的动态参考。它由教师在
最终报告和提交 artifact 基础上人工评审：

| 项目 | 分值 | 要求 |
|---|---:|---|
| 新颖性与系统相关性 | 2 | 解决真实实验瓶颈，而不是装饰性 wrapper |
| 可测量的证据价值 | 1 | 证明减少时间、错误或提高证据质量 |
| 可复用性、文档与测试 | 1 | 能被其他候选或同学复用，有测试和说明 |
| Correctness、安全与 SSOT | 1 | 不修改冻结 workload，不隐藏 fallback，不制造第二套事实 |

课程总评使用：

```text
final_course_grade
  = min(100,
        automated_course_points
        + pr_review_points
        + workflow_innovation_bonus)
```

因此 workflow 创新最多补足 5 分，但不能让总评超过 100，也不能让一个不满足 leaderboard
资格的实现变成 eligible。

可加分的例子：

- 一键 baseline/candidate 构建、运行、配对和 artifact 归档；
- 自动 provenance、hash、结果比较和失败诊断；
- 与端到端 gate 绑定的 shape/tactic 搜索或 profiler workflow；
- 自动上下文边界、OOM 分类与恢复验证；
- 在不泄露 hidden 的前提下改善公共/教师评测协作。

不能加分的例子：

- 没有测量价值的命令包装；
- 对公开 case、expected answer 或特定 token 的硬编码；
- 通过改变输入、输出预算或计时边界获得更好结果；
- 没有测试、复现或实际使用证据的“创新想法”。

## 15. 最终详细报告

最终 PR 必须包含 `REPORT.md`；可额外导出 PDF，但 PDF 不能替代仓库中的可审阅源文档。
建议技术内容相当于 6–10 页，重点是证据而不是凑篇幅。

报告必须包含：

1. 执行摘要与最终可验证 claim；
2. 固定模型、硬件、部署、workload、correctness 和计时合同；
3. baseline 架构、执行路径和瓶颈分析；
4. 优化假设、实现变化和影响边界；
5. correctness、reliability、TTFT、TPOT、显存、MFU/BWU 与可选 profiler 证据；
6. 接受、拒绝、负结果和未决实验；
7. 中期到最终版本的变化，以及如何解释 leaderboard 反馈；
8. 已知限制、不支持模式和回滚方案；
9. 完整 build、serve、evaluator 和复现命令；
10. 结论与一个有边界的下一步。

报告必须引用最终 PR SHA、公开 submission/raw evidence hash、效率输出，以及所有被用于
因果结论的 profiler capture ID。

## 16. 学生提交物

每名学生提交一个 PR。PR 必须包含：

1. 实现改动和受影响的 phase/cell；
2. baseline 与 candidate 的源码/build identity；
3. 可复现的 build、serve 和 evaluator 命令；
4. 公开 runner 生成的 submission、raw JSONL 与 environment evidence；
5. correctness、TTFT、TPOT、显存和稳定性结果；
6. `compute_efficiency.py` 输出或无法产生该输出的解释；
7. 至少一个针对新路径的负控制或回归测试；
8. 失败实验、已知限制与回滚方法；
9. 如声称 profiler 原因，提供对应 capture ID、命令和简短分析；
10. 不含模型权重、隐藏数据、凭据、私有 host/path 或教师结果。

正式成绩只来自教师对 PR SHA 的 clean checkout 和统一重跑；学生自报汇总 JSON 只作
开发参考，不作为正式成绩。

## 17. PR review 的 20 分

| 项目 | 分值 | 核心判断 |
|---|---:|---|
| tests 与负控制 | 8 | 是否保护了新路径的语义、边界和回归 |
| 接口与错误处理 | 4 | 是否 fail closed、能力声明真实、失败可恢复 |
| 可复现性 | 4 | 是否能从 PR SHA 重建、重跑并找到原始证据 |
| 分析与决策 | 4 | 是否解释取舍、负结果、局限和 promotion 决定 |

## 18. 最低通过清单

- [ ] clean checkout 可构建并启动；
- [ ] `/health` 身份、能力和 fallback 声明真实；
- [ ] 公开功能题 6/6；
- [ ] 公开 trajectory artifact 完整；
- [ ] 基础性能 cells 成功率和输出预算正确；
- [ ] 没有意外 OOM、NaN、Xid 或隐藏 fallback；
- [ ] 非法请求或容量失败后服务恢复；
- [ ] runner 自动生成 submission 与 raw evidence；
- [ ] PR 包含测试、复现、效率分析、已知限制和回滚说明；
- [ ] 第 4 天 19:00 前提交可获取的中期 PR SHA 与中期简报；
- [ ] 第 7 天 19:00 前提交最终 PR SHA 和详细 `REPORT.md`；
- [ ] 没有按 case ID、公开 token 序列或 expected answer 硬编码。

长上下文、多请求和图片能力均为扩展；只完成正确、稳定的单请求文字路径仍是合法主榜
提交。
