# 夏令营作业：在单张 RTX 4090 上优化 Qwen3.8-27B INT4 推理

版本：v1（机器权威为 `contract-v1.json`）

学生开始实验前应先阅读 `EXPERIMENT_REQUIREMENTS.md`；授课教师可使用
`TEACHING_GUIDE.md` 介绍任务。两份文档都只解释冻结合同，不改变评分规则。

## 1. 题目

你需要在公开的 ApxInf 推理接口后实现或优化
`cyankiwi/Qwen3.8-27B-AWQ-INT4`，使它在**单张 NVIDIA RTX 4090** 上完成确定性
文本推理。学生提交的是实现 PR；模型权重、教师实现、隐藏数据和教师评分环境不随题目
发布。

核心目标不是为某个公开 prompt 写特例，而是在固定模型、固定输入 token、固定解码语义
和固定硬件条件下，同时改善：

1. 输出正确性与协议正确性；
2. 单请求 prefill/decode 性能；
3. 故障后的可恢复性；
4. 可选的 32K 以上长上下文能力；
5. 可选的多请求正确吞吐；
6. 可选的真实图片输入能力。

`contract-v1.json` 是所有 workload、门槛、权重和计分公式的唯一机器权威。本文件只解释
主榜合同，不建立第二套规则。`multimodal-contract-v1.json` 管理图片徽章；
`course-policy-v1.json` 管理一周日程、workflow 创新加分和最终报告。

## 2. 不可改变的任务合同

- 模型：固定 repo 与 revision 的 Qwen3.8-27B AWQ INT4；
- 硬件：单张 RTX 4090，不允许结构化稀疏或多卡；
- 输入：教师预先完成 tokenization，服务接收非空 `input_ids`；
- 输出：逐 token-ID SSE，索引连续、request ID 隔离；
- 采样：`temperature=0`，thinking 关闭；
- 性能用例：必须生成完整 128 token，不能以 EOS 或 fallback 提前结束；
- 计时：客户端从发送请求开始计时；server 自报时间不计分；
- API：只实现 `INTERFACE.md` 中的 `/health` 与
  `/v1/evaluations/generate` 即可参加主榜。

允许修改公开 ApxInf 仓库中的实现，但不得修改 evaluator、合同、测试数据、模型权重或
硬件配置。服务必须在 `/health` 中报告真实能力；虚报最大上下文、并发数或 fallback
状态会使相关证据失效。

## 3. Correctness：公开与隐藏

主榜 correctness 为 30 分，同时也是性能计分的资格门槛。

公开功能集共 6 题：

- 1K 长度的 early/middle/late 精确检索 3 题；
- 8K 长度的多跳关联、版本覆盖、整数聚合各 1 题。

长文背景使用固定 SHA-256 的 Project Gutenberg《紅樓夢》纯文本。数据生成时下载并
校验，不把全文提交到 starter 仓库。小说正文是自然语言背景；生成器插入明确标记的
“课程私有档案”，机器评分问题只依赖这些虚构记录，因此答案稳定且不存在对原著事实的
争议。

隐藏功能集共 12 题、均不超过 16K，公开形状为：检索 4、干扰消歧 2、多跳 2、版本
覆盖 2、聚合 2。隐藏 prompt、seed、expected output 与 manifest 留在教师侧；一轮
leaderboard 开始后不再改变。功能题由固定 tokenizer 解码，去掉 special token 和首尾
ASCII 空白后执行 case-declared exact validator，不使用 LLM judge。

此外，公开和隐藏各有两条 128-token greedy trajectory，且必须完整输出预算。评分对离散
token ID 序列计算单位代价 Levenshtein 编辑距离（插入、删除、替换各 1），再换算为通过
token 数；这避免一次插入/删除把后续全部位置误判为错。它不是字符串语义相似度、MAE、
cosine、embedding 或 LLM judge。

公开集过 6/6、隐藏集至少 11/12，并通过协议、成功率和可靠性门槛后，提交才进入
TTFT/TPOT 的动态排名。公开和隐藏 trajectory 都必须提交完整证据并按相似率获得相应
correctness 分，但不作为性能排名的一票否决条件；本 INT4 题目不声称长上下文逐 token
等价于 BF16/vLLM。

## 4. 单请求主榜：100 分

基础场景严格为一次一个文本请求：

| 部分 | 分值 | 说明 |
|---|---:|---|
| correctness | 30 | 协议、公开/隐藏功能、公开/隐藏 trajectory |
| TTFT | 35 | 1K/2K/4K/8K/16K，均输出 128 token |
| TPOT | 25 | 1K 与 8K，均输出 128 token |
| reliability | 10 | 成功率、OOM/NaN/fallback/Xid、失败后恢复 |

正式测量每个 latency cell 先 warm-up 1 次，再测 5 次，以中位数计分，TTFT 与 TPOT
的 CV 均不得超过 10%。所有原始样本必须保存在教师 artifact 中。

vLLM 是每轮强制执行的同机 control，**不是固定满分实现**。每个 cell 独立使用当轮所有
合格 PR 与 vLLM 中的最好中位数作为满分参考：

```text
cell_points = weight * min(1, best_valid_median / observed_median)
```

提交加入后相对分可能变化，所以中期榜和结课榜以冻结的 cohort snapshot 为准。

TTFT 定义为 client send 到第一个 token event；当完成 token 数大于 1 时，TPOT 定义为
`(最后 token 到达时间 - 第一个 token 到达时间) / (completion_tokens - 1)`。

## 5. 长上下文 bonus：0–10 分

32,768 prompt token 是 0 分起点，不是满分点。evaluator 先用 32,640 prompt +
128 output 诊断“总序列上限为 32,768”的实现，再进入计分 staircase：

```text
32640（不计分）→ 32768 → 65536 → 131072 → 196608 → 262016
```

262,016 为模型原生 262,144 positions 扣除 128 output token 后的计分上限。在申报的最大
成功长度，必须通过六类任务：early/middle/late 检索、多跳、版本覆盖、聚合；每条输出
必须以精确答案开头并完整产生 128 token。未到原生上限时，还必须记录第一个更大长度的
失败边界，随后 `/health` 和一个小请求都要恢复成功。

通过验证后按合同中的 log2 容量进度获得 0–10 分。只修改 `/health.max_model_len` 不会
得分，base cell 内的 OOM 仍属于 unexpected OOM。

## 6. 多请求 bonus：0–10 分

多请求不是基础要求。evaluator 在同一单请求 endpoint 上以 closed loop 执行 C4 与 C8，
每个 cell 共 32 个 1K/128 请求。排队时间包含在 TTFT 与 batch makespan 中。

只有在 success=100%、correctness=100%、Jain fairness 至少 0.95、无 fallback、tail
guard 通过且服务结束后健康时，cell 才有效。每个有效 cell 获得 1 分接口支持分，再按
当轮最好 correct goodput 比例获得最多 4 分。串行排队可以合法参加主榜，但通常无法在
goodput bonus 上竞争。

## 7. MFU、BWU 与硬件证据

`compute_efficiency.py` 输出诊断指标，不直接加分：

- BF16-equivalent MFU proxy：冻结的 dense-equivalent FLOP 估计除以 wall time 和
  RTX 4090 dense BF16 Tensor Core 峰值；
- minimum-model BWU proxy：冻结的最低模型字节数除以 wall time 和标称 HBM 带宽；
- profiled BWU：只有 phase-scoped Nsight Compute DRAM bytes 与 kernel elapsed
  同时存在时才输出。

W4A16 路径混合权重加载、反量化、BF16 Tensor Core、attention/recurrent 与 elementwise，
不能直接拿 INT4 TOPS 为整个模型定义一个“真实 MFU”。`nvidia-smi utilization.memory`
也不是带宽利用率。

## 8. 公开自测

先准备固定 corpus 与预分词数据：

```bash
python3 benchmarks/qwen38_4090/evaluation/fetch_public_corpus.py

python3 benchmarks/qwen38_4090/evaluation/generate_evaluation_cases.py \
  --model-dir /path/to/Qwen3.8-27B-AWQ-INT4 \
  --output-dir /tmp/apxinf-public \
  --suite public
```

启动服务后，canonical runner 将直接生成符合 `submission-schema-v1.json` 的 artifact；
不得手填 TTFT、TPOT、correctness 汇总或 bonus 证明。公开 profile 仅用于 bring-up，正式
中期分由教师在冻结机器上执行隐藏集、vLLM control 和全部 PR 后统一生成。

另外提供一条不计分的《红楼梦》文学分析演示：要求模型给出中心判断和两处原文证据。
它可以观察长文本表达能力，但主观解释不进入自动 correctness。

另有一个独立冻结的图片输入能力赛道，详见 `multimodal-contract-v1.json`。公开 4 题覆盖
七段数码 OCR、空间颜色、柱状图算术和目标计数；教师隐藏集为同四类各 2 题。图片由
标准库脚本确定性生成并逐文件校验 SHA-256，答案采用 exact validator，不使用 LLM
judge。公开与隐藏全部通过可获得 `multimodal-ready` 徽章；v1 暂不为该徽章增加
leaderboard 分数，避免在一轮已经冻结后追溯改变 120 分主榜。

不支持图片的实现仍是合法提交，但必须在 `/health` 声明 `multimodal=false`，并让图片
探针以机器可读的 `unsupported_capability` 失败关闭。忽略图片后返回 200、返回 500，或
静默委托 vLLM/Transformers 都不合格。

## 9. 一周日程、中期榜与 Workflow 创新

任务周期为 7 天，时区为 Asia/Shanghai：

```text
第 1 天       环境、接口、baseline
第 2 天       Correctness、选定优化 cell
第 3 天       第一版端到端候选
第 4 天 19:00 中期 PR SHA 与结果截止
随后          教师统一运行 hidden + vLLM control 并公布中期 leaderboard
第 5 天       分析中期结果，收敛最终假设
第 6 天       最终候选与报告
第 7 天 19:00 最终 PR SHA、artifact 与 REPORT.md 截止
```

中期榜使用第 4 天 19:00 前可获取的完整 PR commit SHA。教师冻结全部提交后统一 clean
checkout、运行 hidden 与 vLLM control，再公布命名 snapshot；截止后的修改只进入最终版。

另设 0–5 分 workflow 创新课程加分，不改变 120 分 leaderboard，也不影响 cell 满分参考。
加分关注实验自动化、provenance、可复用搜索/诊断、容量与恢复验证等真实系统价值；要求有
测量证据、文档、测试，并保持 correctness、计时和 SSOT。课程总评仍封顶 100。

最终 PR 必须包含详细 `REPORT.md`，说明 baseline、假设、实现、correctness、性能、效率、
负结果、中期到最终变化、限制、回滚和复现命令。建议技术内容相当于 6–10 页；可附 PDF，
但仓库内的可审阅报告不可缺失。

## 10. 提交方式与 PR 验收

每名学生提交一个 PR，并在说明中给出：

1. 设计变化与预期影响的 phase/cell；
2. 公开自测命令与结果 artifact；
3. 至少一个负控制或回归测试；
4. correctness、性能、稳定性与显存的取舍；
5. `compute_efficiency.py` 输出，或无法生成相应 MFU/BWU proxy 的原因；
6. 已知限制、回滚方法，以及未通过的实验分支。

教师只从 PR SHA clean checkout 构建和评分，不接收学生自报 JSON 作为正式成绩。公开
接口和测试必须通过；不得按 case ID、已知 token 序列或 expected answer 硬编码输出；
不得静默切换 vLLM、Transformers、CPU 或另一个模型作为 fallback。

课程成绩由自动 leaderboard 部分与 PR review 组成；精确映射和 review 分项以合同为准。
bonus 可以弥补基础性能差距，但不能绕过 correctness/reliability gate，也不能让自动部分
超过课程设定上限。

## 11. 通过标准

一个可发布、可评分的提交至少满足：

- clean checkout 可构建并启动；
- `/health` 身份与能力声明真实；
- 公开功能 correctness 通过，且公开 trajectory 证据完整生成；
- 所有基础性能 cell 不 OOM、不 fallback、输出预算完整；
- 服务在非法请求和容量失败后仍可用；
- runner 自动生成完整 submission 与 raw evidence 引用；
- 第 4 天 19:00 前提交中期 PR SHA 与简报；
- 第 7 天 19:00 前提交最终 PR SHA 和详细 `REPORT.md`；
- PR 中的测试、复现说明和结果分析足以由教师独立重放。

长上下文、多请求、图片能力徽章和文学分析演示均是对基础实现的扩展，不影响只完成
单请求 32K 以内文字实现的合法性。
