# 教师介绍指南：如何向同学们讲清楚这个任务

本指南用于课程介绍、开题说明和 leaderboard 规则讲解。评分机器权威仍是
`contract-v1.json` 与 `multimodal-contract-v1.json`；日程、workflow 创新加分和最终报告
权威是 `course-policy-v1.json`。本指南不包含隐藏数据、教师实现细节或教师成绩。

需要直接制作课件时，使用 `SLIDE_OUTLINE.md` 的逐页标题、内容、视觉和讲稿建议。

## 1. 一句话介绍

> 在一张 24GB RTX 4090 上，让一个 27B 的 Qwen3.8 INT4 模型既答得对、跑得快、长文本
> 不崩、服务可恢复，还要用完整证据解释你的优化为什么真的有效。

这句话比“写一个更快的 CUDA kernel”更准确。课程考察的是模型、runtime、服务和评测
之间的端到端系统设计。

## 2. 三分钟版本

如果只有三分钟，按这个顺序介绍：

1. **约束**：27B 模型、单张 4090、24GB 显存、AWQ INT4；
2. **必做**：正确实现单请求文字推理；
3. **优化目标**：prefill 看 TTFT，decode 看 TPOT；
4. **评分方式**：vLLM 是同机 control，不是固定满分；同轮最好实现获得 cell 满分；
5. **Correctness gate**：公开题和隐藏题先过，性能才有资格排名；
6. **扩展方向**：32K 以上长上下文、多请求 goodput、真实图片能力；
7. **一周节奏**：第 4 天 19:00 中期 SHA，第 7 天 19:00 最终 PR 与报告；
8. **提交方式**：提交 PR，教师从 PR SHA clean checkout 统一重跑。

收尾可以说：

> 你不必在所有方向都做到最好。你需要选一个明确边界，证明语义没有坏，并让端到端指标
> 真正改善。

## 3. 十五分钟完整介绍结构

### 3.1 为什么这个题有挑战（2 分钟）

先给出三个冲突：

- 模型有 27B 参数，但 GPU 只有 24GB；
- INT4 能装下权重，却引入格式、反量化和数值正确性问题；
- prefill 与 decode 的瓶颈不同，一个实现可能 decode 很快但 prefill 很慢。

提醒同学：显存“能装下”不等于系统“能服务”，kernel 快不等于 TTFT/TPOT 快。

### 3.2 模型和 runtime 结构（3 分钟）

用一张简图讲清：

```text
input tokens
  → embedding
  → 64 layers：GDN, GDN, GDN, full attention 重复
  → MLP / residual / norm
  → LM head
  → token stream
```

补充三点：

- Hugging Face identity 是 `Qwen3_5ForConditionalGeneration`；这表示 architecture family，
  题目和 checkpoint 仍是 Qwen3.8；
- 权重是 compressed-tensors W4A16 group-32 asymmetric，不能当成已有 W8A8 路径；
- GDN 需要 recurrent/conv state，full attention 需要 KV，因此 request reset 和 cache
  ownership 都是 correctness 的一部分。

如果介绍图片扩展，再加：

```text
PNG → processor → 27-layer vision encoder → merger
→ image-token embedding → multimodal RoPE → hybrid language model
```

强调“有 vision weights”不等于“服务已经支持图片”。

### 3.3 评什么（4 分钟）

先画 100+20：

```text
100-point base
  correctness 30
  TTFT        35
  TPOT        25
  reliability 10

20-point bonus
  context      10
  multi-request 10
```

然后解释图片是独立徽章，v1 不加分。这样同学不会误以为“做图片就能绕过文字主榜”。

解释指标时使用具体语言：

- TTFT：请求发出到第一个 token；主要观察 prompt/prefill；
- TPOT：第一个到最后一个 token 的平均间隔；主要观察 decode；
- goodput：只计算正确且成功请求的 token，并包含排队时间；
- reliability：OOM 后恢复、无 NaN/Xid、无 fallback，不是“请求大多成功”；
- context：最大长度必须做任务，不能只分配 KV 或修改 health 数字。

### 3.4 为什么先 correctness 再性能（2 分钟）

给同学一个反例：如果错误地把 Qwen3.8 路由到 Llama 或普通 Qwen3-VL，服务可能返回流畅
文本，甚至 benchmark 很快，但模型语义是错的。另一个反例是只比较字符串，看不出一次
token 插入导致的轨迹整体错位。

因此课程采用：

- 公开功能题；
- 教师隐藏功能题；
- 完整 token trajectory；
- 协议负控制；
- 故障后恢复。

性能只有在这些 gate 后才进入排名。

### 3.5 学生可以从哪里优化（2 分钟）

不要给出固定答案，给出搜索边界：

- 权重 loader、packing、dequantization；
- W4A16 GEMM/GEMV；
- GDN scan、conv 和 recurrent state；
- full-attention KV 布局与 kernel；
- prefill tile、Marlin/CUTLASS/FA2；
- embedding、LM head、host/device round trip；
- CUDA Graph 和 launch overhead；
- KV 容量、长上下文 admission；
- continuous batching、队列和 tail latency；
- vision encoder、media embedding 和 mRoPE。

提醒同学：一个 PR 最好有一个清楚的主要假设，而不是同时改十个模块后无法归因。

### 3.6 提交和 leaderboard（2 分钟）

说明三个规则：

1. 提交 PR，不提交教师机器上的手填分数；
2. 教师从完整 PR SHA clean checkout 构建；
3. 中期榜是一个冻结 cohort snapshot，加入新提交后动态相对分可能变化。

再展示一周时间轴：第 4 天 19:00 冻结中期 SHA，教师统一运行 hidden 与 vLLM 后公布
leaderboard；第 7 天 19:00 提交最终 SHA 和详细 `REPORT.md`。Workflow 创新可获 0–5
课程加分，但不改变 leaderboard，课程总评仍封顶 100。

## 4. 推荐的 35 分钟课堂流程

| 时间 | 内容 | 建议材料 |
|---:|---|---|
| 0–3 分钟 | 题目动机与一句话目标 | 单张“27B 模型 vs 24GB GPU”图 |
| 3–8 分钟 | Qwen3.8 hybrid architecture 与 INT4 格式 | GDN/GDN/GDN/attention 示意图 |
| 8–13 分钟 | 接口、状态与 correctness gate | `/health` 和 SSE token 例子 |
| 13–19 分钟 | TTFT、TPOT、动态相对计分 | 100+20 评分表 |
| 19–23 分钟 | 长上下文和多请求 bonus | context staircase、C4/C8 图 |
| 23–26 分钟 | 图片能力扩展 | processor→vision→mRoPE 图 |
| 26–29 分钟 | PR artifact 与实验方法 | baseline/candidate/evidence 流程 |
| 29–32 分钟 | 一周日程、中期榜、workflow 创新与最终报告 | 7 天时间轴 |
| 32–35 分钟 | 选择第一个优化假设 | 给出三个起点，不公布答案 |

## 5. 推荐现场演示

演示应短、确定、不会泄露隐藏集。

### 演示 1：服务身份

展示 `/health`：模型 revision、max context、parallel requests、fallback 和 capabilities。
问同学：“为什么一个性能测评需要先相信这些字段？为什么还必须用请求去验证？”

### 演示 2：同一模型的 prefill/decode 取舍

运行一个公开 1K/128 或 8K/128 cell，展示客户端 TTFT、TPOT 和显存。不要只展示
`nvidia-smi GPU-Util`，也不要把 server timing 当正式分数。

### 演示 3：长文本不是把小说塞进去就结束

展示《红楼梦》正文加确定性课程档案的 prompt 结构。对比：

- 文学分析问题：适合定性演示；
- needle、多跳、版本覆盖、聚合：适合 exact 自动评分。

### 演示 4：真实图片能力

只使用公开图片。展示 health capability、发送一个 PNG data URL，并输出 exact OCR 或计数
答案。强调完整路径包括 processor、vision encoder、image embedding 和 multimodal
RoPE；不能丢掉图片后只让语言模型猜。

教师可展示黑盒能力，不应发布当前教师实现源码、隐藏题或教师端结果。

## 6. 建议幻灯片标题

1. 一张 4090，能把 27B 模型做到什么程度？
2. 题目合同：什么固定，什么可以改？
3. Qwen3.8：GDN 与 full attention 的混合主干
4. 能装下，不等于能正确推理
5. Correctness gate：公开、隐藏、trajectory
6. TTFT 与 TPOT：prefill 和 decode 是两个问题
7. vLLM 是 control，不是满分答案
8. 32K 不是终点：长上下文如何获得加分
9. 吞吐不能掩盖尾延迟：C4/C8 bonus
10. 从文字到图片：完整多模态路径
11. MFU/BWU 能说明什么，不能说明什么
12. 一个合格 PR 应留下哪些证据
13. 一周任务周期与中期 Leaderboard
14. Workflow 创新加分与最终 REPORT.md
15. 选择你的第一个可证伪假设

## 7. 推荐开场话术

> 这个任务不是让大家复刻一个现成框架，也不是只写一个 GEMM。我们固定模型、权重、
> GPU、输入 token 和计时边界，把系统里可以优化的部分留给大家。你可以优化 prefill、
> decode、长上下文、并发调度，或者挑战真实图片推理。但所有性能改进先要证明模型没有
> 被换掉、输出没有被破坏、服务失败后还能恢复。

## 8. 推荐收尾话术

> 请选择一个你真正想理解的瓶颈，先建立 baseline，再写出可证伪假设。一个被证据拒绝的
> 方向也是有效工作；一个没有 correctness 和端到端结果的 kernel speedup 不是最终答案。

## 9. 常见问题与回答

### “vLLM 是不是满分？”

不是。vLLM 是每轮必须运行的同机 control。每个 cell 的最好有效提交才是该 cell 的满分
参考，学生实现可以超过 vLLM。

### “为什么不用 BF16 作为 correctness 满分？”

这个题的目标 checkpoint 是固定 INT4 权重。功能正确性使用 exact task validators，完整
trajectory 另外按离散 token 编辑相似率计分。不能把 INT4 的每个 token 都强行要求与另一个
runtime/精度完全相同，也不能因为输出大意相似就忽略系统性错误。

### “32K 是长上下文满分吗？”

不是。32K 是 bonus 起点。只有在 32K 以上通过完整任务矩阵，才按容量进度获得 0–10 分。

### “多请求只看 tokens/s 吗？”

不是。只计算正确请求的 goodput，而且 success、correctness、fairness、p95 TTFT/TPOT、
fallback 和结束后 health 都必须通过。

### “图片输入为什么不加分？”

因为 v1 主榜已经冻结为文字 100 分加两个 10 分 bonus。图片先用独立能力徽章验证接口、
数据集和实现成熟度；若下一届要加分，应发布新合同，而不是活动轮次中追溯改规则。

### “Workflow 创新怎么加分？”

教师按 `course-policy-v1.json` 评审 0–5 分，关注系统相关性、测量证据、可复用性、测试和
correctness/SSOT 保护。它不影响 leaderboard cell，也不能让课程总评超过 100。

### “最终报告要写什么？”

最终 PR 必须包含详细 `REPORT.md`，覆盖 baseline、假设、实现、correctness、性能、效率、
负结果、中期到最终变化、限制、回滚和复现命令。建议内容相当于 6–10 页，PDF 可选。

### “可以调用 PyTorch、Transformers 或 vLLM 吗？”

不能作为候选推理 fallback。模型规定的 tokenizer/media processor 可以位于服务边界，但
实际被评分的模型计算必须来自提交实现，并且 `/health.fallback_active=false` 必须真实。

### “为什么叫 qwen35？”

权重的 Hugging Face architecture identity 是 `qwen3_5`，所以底层模块可使用该名称；课程
模型与 repo ID 是 Qwen3.8。不要为了名称整齐而把多个结构不同的模型塞进一个错误的通用
实现，也不必把每个内部符号都改成 `qwen38`。

## 10. 不要这样介绍

- 不要说“谁最接近 vLLM 谁满分”；vLLM 不是固定上限；
- 不要说“达到 32K 就拿满长上下文分”；
- 不要把 GPU utilization、memory utilization 直接称为 MFU/BWU；
- 不要把当前教师实现性能当成学生必须复现的目标；
- 不要公布隐藏 prompt、case IDs、expected answers、seed 或教师结果；
- 不要暗示图片徽章可以绕过文字 correctness gate；
- 不要把一次 microbenchmark 加速称为端到端 serving 加速；
- 不要承诺所有合法提交都具有 BF16/vLLM 的逐 token 轨迹。

## 11. 开课前教师检查清单

- [ ] 公开 starter commit 与 release manifest 已冻结；
- [ ] 模型、tokenizer、公开/隐藏数据和 reference hashes 对得上；
- [ ] vLLM control 与至少一个候选能在同一 4090 环境启动；
- [ ] 公共 CI 和 teacher-only evaluator 测试通过；
- [ ] 中期机器的 driver/CUDA/power/clock 元数据已记录；
- [ ] 所有 hidden seed 和数据权限正确；
- [ ] 公共包中没有 teacher、fixtures、results、host/path、凭据或教师实现；
- [ ] 课堂演示只使用公开数据；
- [ ] 旧课程手册中如有另一套 rubric，已经删除或明确降级为历史材料；
- [ ] 学生知道 PR、artifact、申诉和 leaderboard snapshot 的时间点。
- [ ] 第 4 天与第 7 天 19:00 截止时间已按 Asia/Shanghai 明确通知；
- [ ] Workflow 创新 rubric 与最终报告模板已向学生公开。
