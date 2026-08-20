# Slides 大纲：ApxInf Qwen3.8-27B INT4 推理优化实验

用途：课程开场、夏令营题目发布和中期 leaderboard 说明。建议主讲 18 页，约 30–35
分钟；附录页按现场问题选用。

评分权威仍是 `contract-v1.json` 与 `multimodal-contract-v1.json`。Slides 只解释合同，
不得增加口头门槛、修改分数，或公布 hidden/teacher evidence。

## Slide 1｜封面：一张 4090，能把 27B 模型做到什么程度？

### 页面内容

- 标题：`ApxInf 推理优化实验`
- 副标题：`单张 RTX 4090 · Qwen3.8-27B INT4`
- 一句话任务：
  `在固定模型、固定 GPU 和固定评测下，同时保证正确性、延迟、容量与稳定性。`

### 建议视觉

一张 RTX 4090、27B 模型和 Token 流的极简示意图。不要在封面堆评分细节。

### 讲稿重点

强调这是端到端推理系统实验，不是单个 kernel 竞赛，也不是复刻 vLLM。

## Slide 2｜为什么这个题有挑战？

### 页面内容

用三个冲突说明问题：

1. `27B 参数` vs `24GB 显存`；
2. `INT4 能装下` vs `格式、反量化、数值正确性`；
3. `Prefill 计算密集` vs `Decode 带宽与 launch 敏感`。

补一句：`能启动 ≠ 能正确推理；Kernel 快 ≠ 服务端到端快。`

### 建议视觉

三角冲突图：模型规模、硬件限制、服务质量。

### 讲稿重点

让学生先理解优化空间为什么存在，再介绍 API 和分数。

## Slide 3｜ApxInf 是什么？

### 页面内容

- ApxInf 是本实验使用的推理引擎和优化载体；
- 它负责模型加载、CUDA 执行、KV/recurrent state、服务接口和 token 输出；
- 学生在 ApxInf 公开接口后实现或优化 Qwen3.8；
- 学生发布分支只提供接口、合同、runner 和测试，不提供教师实现。

建议使用流程：

```text
请求 → ApxInf 推理引擎 → Qwen3.8 → Token 输出
```

### 建议视觉

四节点水平流水线，ApxInf 位于中间并高亮。

### 讲稿重点

不要把 ApxInf 介绍成云服务或黑盒 API。它是学生需要理解和修改的推理 runtime。

## Slide 4｜什么固定，什么可以优化？

### 页面内容

左栏“固定”：

- Qwen3.8-27B AWQ INT4 与固定 revision；
- 单张 RTX 4090；
- greedy、temperature=0、thinking 关闭；
- 精确输入 token 与客户端计时；
- 公开/隐藏 evaluator 与评分合同。

右栏“可以优化”：

- loader、packing、W4A16 GEMM/GEMV；
- GDN、attention、KV/recurrent state；
- prefill tile、CUDA Graph、host/device 边界；
- 长上下文、调度、并发、视觉路径。

### 建议视觉

“锁定合同”与“开放优化空间”两列。

### 讲稿重点

优化自由来自冻结实验合同。不能通过换模型、换输入或缩短输出获得性能分。

## Slide 5｜Qwen3.8 的执行结构

### 页面内容

```text
Embedding
  → [GDN → GDN → GDN → Full Attention] × 16
  → Norm
  → LM Head
  → Greedy Token
```

解释：

- GDN 拥有 conv/recurrent state；
- Full Attention 拥有 KV cache；
- W4A16 权重路径不同于已有 W8A8；
- `qwen3_5` 是 checkpoint architecture identity，课程模型仍是 Qwen3.8。

### 建议视觉

64 层结构用 4 层 repeating block 表示，避免画满 64 个框。

### 讲稿重点

状态 reset、cache ownership 和 position 都是 correctness，不只是 kernel 细节。

## Slide 6｜实验地图：100 + 20 + 能力徽章

### 页面内容

```text
主榜 100 分
  Correctness 30
  TTFT        35
  TPOT        25
  Reliability 10

Bonus 20 分
  长上下文     10
  多请求       10

能力徽章
  真实图片输入
```

### 建议视觉

三张卡片：主榜、Bonus、能力徽章。图片徽章与分数视觉上分开。

### 讲稿重点

文字主榜是必做；长上下文和并发是计分扩展；图片在 v1 是非计分能力徽章。

## Slide 7｜先过 Correctness gate

### 页面内容

- 公开功能题：1K early/middle/late + 8K 多跳/版本/聚合；
- 隐藏功能题：12 题，公开类别不公开内容；
- 公开 6/6、隐藏至少 11/12 才进入性能排名；
- 公开/隐藏 token trajectory 使用离散 token Levenshtein；
- 非法请求、fallback、失败恢复也属于 correctness/reliability。

### 建议视觉

闸门图：Correctness gate 通过后才进入 TTFT/TPOT leaderboard。

### 讲稿重点

举反例：错误路由到 Llama 也可能输出流畅文字，但模型语义已经错了。

## Slide 8｜TTFT 与 TPOT：两个不同问题

### 页面内容

左侧 TTFT：

- client send → first token；
- 1K、2K、4K、8K、16K；
- 主要观察 prefill 和首 token 路径。

右侧 TPOT：

- `(last token − first token)/(N−1)`；
- 1K、8K；
- 主要观察 decode、KV、launch 与带宽。

底部：每个 cell 输出完整 128 token。

### 建议视觉

一条 token timeline：请求发送、first token、后续 token。

### 讲稿重点

一个实现可能 TTFT 差但 TPOT 好；不能用单一 tokens/s 概括所有性能。

## Slide 9｜vLLM 是 Control，不是固定满分

### 页面内容

公式：

```text
cell_points = weight × min(1, best_valid / observed)
```

说明：

- 每轮必须运行同机 vLLM control；
- 满分参考来自 vLLM 与所有 eligible PR 中的最好值；
- 学生可以在某个 cell 超过 vLLM；
- 分数随 cohort 变化，在命名 snapshot 时冻结。

### 建议视觉

多个候选柱状图，其中最好的一根成为该 cell 的 100% 参考。

### 讲稿重点

不要说“最接近 vLLM 就满分”，也不要公布当前教师实现的私有分数。

## Slide 10｜长上下文：32K 只是起点

### 页面内容

```text
32,640（诊断）
→ 32,768
→ 65,536
→ 131,072
→ 196,608
→ 262,016 + 128 output
```

必须同时通过：early/middle/late retrieval、多跳、版本覆盖、聚合；失败后还要恢复 health
和小请求。

### 建议视觉

阶梯图，32K 标为 0 分起点，262K 标为满 bonus。

### 讲稿重点

修改 health 数字或只分配 KV 不得分；容量必须由任务验证。

## Slide 11｜多请求：吞吐不能掩盖尾延迟

### 页面内容

- C4 与 C8 closed loop；
- 每个 cell 32 个 1K/128 请求；
- correct goodput 只计算正确且成功的 token；
- success=100%、correctness=100%、fairness≥0.95；
- p95 TTFT/TPOT tail guard；
- 无 fallback，结束后服务健康。

### 建议视觉

多条并发 token stream 汇入 scheduler；旁边同时显示 goodput 和 p95 latency。

### 讲稿重点

串行排队可以保持主榜合法，但不能靠 aggregate throughput 掩盖用户等待。

## Slide 12｜真实图片能力：必须接通完整路径

### 页面内容

```text
PNG → Processor → Vision Encoder → Merger
→ Image Embedding → Multimodal RoPE → Language Model
```

- 公开 4 题：OCR、空间颜色、柱状图算术、目标计数；
- 隐藏 8 题：同四类各 2；
- 公开 4/4：`multimodal-public-pass`；
- 公开 4/4 + 隐藏 8/8：`multimodal-ready`；
- v1 不加 leaderboard 分。

### 建议视觉

从图片到 token 的横向 multimodal pipeline。

### 讲稿重点

有 vision weights 不等于支持图片；返回 200 但忽略图片属于失败。

## Slide 13｜MFU、BWU 与证据

### 页面内容

- `estimated_mfu_bf16_equivalent_pct`：计算 proxy；
- `minimum_model_bwu_pct`：最低模型字节 proxy；
- `profiled_bwu_pct`：需要 NCU 的 DRAM bytes 与 kernel elapsed；
- 正式 TTFT/TPOT 始终来自无 profiler 的客户端计时；
- NSys/NCU 用于解释原因，不替代端到端结果。

红色警示：

- `GPU-Util ≠ MFU`
- `memory utilization ≠ HBM bandwidth`
- `kernel speedup ≠ serving speedup`

### 建议视觉

三层证据金字塔：End-to-End → NSys → NCU。

### 讲稿重点

指标必须说明分子、分母和时间边界；proxy 名称不能被省略。

## Slide 14｜一周任务周期

### 页面内容

用一条 7 天时间轴：

```text
Day 1  环境、接口、Baseline
Day 2  Correctness、选定 Cell
Day 3  第一版候选
Day 4  19:00 中期提交截止
       → 教师统一评测
       → 公布中期 Leaderboard
Day 5  分析榜单与反馈
Day 6  最终优化与报告
Day 7  19:00 最终 PR + REPORT.md
```

### 建议视觉

横向 7 天时间轴。Day 4 和 Day 7 使用明显的橙色 deadline 标记；中期评测和 leaderboard
作为 Day 4 后的独立教师流程。

### 讲稿重点

中期使用截止前的完整 PR SHA 冻结 cohort。截止后修改不回填中期榜，只进入最终版。

## Slide 15｜中期提交与 Leaderboard

### 页面内容

中期提交必须有：

- 完整 PR commit SHA；
- 该 SHA 生成的公开 artifact；
- 简短中期总结：baseline、候选、目标 cell、结果、限制。

教师流程：

```text
冻结 Cohort
→ Clean Checkout
→ Hidden + vLLM Control
→ Cohort Scoring
→ 发布命名 Leaderboard Snapshot
```

### 建议视觉

学生 PR 汇入一个冻结 cohort，教师统一运行，最后输出 leaderboard。

### 讲稿重点

不发布部分同学的滚动成绩；只有统一环境、统一 cohort 的 snapshot 可以横向比较。

## Slide 16｜Workflow 创新加分与最终报告

### 页面内容

左栏：`Workflow 创新 0–5 课程分`

- 新颖性与系统相关性 2；
- 测量证据价值 1；
- 可复用性、文档与测试 1；
- Correctness、安全、SSOT 1；
- 不改变 leaderboard，总评封顶 100。

右栏：`最终 REPORT.md`

- Baseline、假设与实现；
- Correctness、性能、显存、MFU/BWU；
- 中期到最终变化；
- 负结果、限制与回滚；
- 完整复现命令与 artifact hash。

### 建议视觉

左侧齿轮/自动化 workflow，右侧技术报告封面，中间用 evidence 箭头连接。

### 讲稿重点

创新不是“脚本多”，而是减少实验错误、提高复现与证据质量。报告不是流水账，要用证据
回答最终 claim 是否成立。

## Slide 17｜怎么做实验、怎么提交

### 页面内容

```text
冻结 cell
→ 建 baseline
→ 写可证伪假设
→ 最小实现
→ correctness
→ 端到端测量
→ 必要时 profile
→ 接受 / 拒绝 / 继续
```

PR 必须包含：

- 设计变化与目标 cell；
- build/serve/eval 命令；
- raw artifact 与 environment evidence；
- correctness、性能、显存、稳定性；
- MFU/BWU proxy 或缺失原因；
- 负控制、失败实验、局限和回滚方法。

### 建议视觉

实验闭环箭头图，最后连接到 GitHub PR。

### 讲稿重点

一个被证据拒绝的方向也是有效实验；没有 correctness 和端到端结果的 kernel speedup 不是
最终答案。

## Slide 18｜开始你的第一个假设

### 页面内容

给出三个开放起点：

1. `为什么当前 prefill 没有充分利用 4090？`
2. `为什么 decode 更接近带宽受限？`
3. `怎样扩大上下文或并发而不破坏 correctness 与恢复？`

结束语：

> 选择一个你真正想理解的瓶颈，先建立 baseline，再让证据决定是否保留优化。

### 建议视觉

三个问号卡片，避免给出教师实现或答案。

## 附录 A｜接口速查

- `GET /health`
- `POST /v1/evaluations/generate`
- Optional：`POST /v1/chat/completions`
- SSE token index、request ID、usage 和 `[DONE]`
- 非法输入、容量失败和 unsupported capability 的错误语义。

## 附录 B｜评分速查

- 100 分主榜与 20 分 bonus；
- course automated component 上限 80；
- PR review 20；
- workflow innovation bonus 0–5，课程总评封顶 100；
- 正式 1 warm-up + 5 measured repeats；
- median 与 CV≤10%；
- 动态 cohort snapshot。
- 第 4 天 19:00 中期截止；第 7 天 19:00 最终截止。

## 附录 C｜常见误区

- vLLM 不是固定满分；
- 32K 不是 context 满分；
- 图片 capability 不进入 v1 分数；
- Token ID 不是连续数值；
- 公开题不能被硬编码；
- server timing 不是正式计时；
- 教师实现和隐藏结果不能进入公开 slides。

## Slides 视觉制作规范

- 比例：16:9；
- 每页只保留一个中心结论；
- 正文不超过 5–6 条；
- 评分数字只来自冻结合同；
- 使用相同颜色语义：蓝色=基础路径，橙色=计分，紫色=能力扩展，红色=无效行为；
- 不使用教师代码截图、hidden case、seed、远程路径、服务器地址或教师结果；
- 图示优先，代码只保留接口或公式的最小片段；
- 所有英文缩写保持原样：ApxInf、TTFT、TPOT、MFU、BWU、vLLM、PR。
