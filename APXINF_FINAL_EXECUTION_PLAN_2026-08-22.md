# ApxInf Qwen3.8-27B：单卡 4090 最高得分执行方案

> 版本：2026-08-22（基于仓库现实状态与外部审查的纠偏版）
>
> 适用截止：中期测评 2026-08-24 19:00；最终提交 2026-08-27 19:00（以课程公告时区为准）
>
> 依据：`README.md`、`benchmarks/qwen38_4090/evaluation/contract-v1.json`、`multimodal-contract-v1.json`、`submission-schema-v1.json`、`pdf/Agent4System-0820.pdf`
>
> 相关三套路线的完整技术比较见 [APXINF_QWEN38_TECHNICAL_PLANS.md](APXINF_QWEN38_TECHNICAL_PLANS.md)。本文件是实际执行时的唯一主线方案。

> **分工（据 git 记录，2026-08-26 订正）**：本项目为两人团队。**程仁龙**（`mygqdmmss`）承担约 **95%**——模型/runtime 主线、CUDA kernel、协议 surface、oracle 与 hidden 代理集、全部性能实验、显存账本、多模态与 C4 两项 bonus、REPORT 与最终集成；**王天民**（`wang1145140503`）承担约 **5%**——`scripts/campaign/` 与 `benchmarks/campaign/manifests/` 下的离线 benchmark 脚手架（shape inventory、experiment validator，8/23 交付），未进入服务/模型主线。本文原为多 lane 并行规划，实际执行为程仁龙单人串行推进（该偏差本身即提交材料要求的"设计变化"记录，正文已按实际分工订正）；服务器带锁队列（`flock`）与单卡正式重放的纪律始终严格执行。
>
> **实际执行结果（2026-08-27 订正，证据均在 REPORT.md）**：本文规划的各层目标最终完成情况——① 文本 eligibility 全门禁通过（协议 12/12、公开功能 6/6 精确、200 请求 soak 100%、7/7 性能 cell 有效）；② 性能工程 TTFT 最高约 48 倍、TPOT 约 2 倍（17 次单变量配对实验，9 接受 8 拒绝）——主线技术路线与规划的对照：GDN 按规划先 eager 取资格、后接 chunk 序列路径；prefill 解量化 + BF16 tensor-core GEMM（即规划的 M-W4-P 实验，TTFT −74%）与 decode packed GEMV（TPOT −35%）均按规划落地；规划外由测量发现的三个最大杠杆是 dequant scratch 池化（消除 cudaMalloc 风暴）、行协作 causal softmax（16K 真瓶颈，单层 −98%）与 prefill chunk 64→512；规划的 CUDA Graph decode 未实施（decode 归因显示收益不在 launch 开销），MTP 按规划的条件门槛评估后未实施；③ **多模态 bonus 已交付**（vision 塔 CUDA 移植 + `/v1/chat/completions`，自建探针 4/4、图文混合 soak 36/36、文本数值零变化，`APXINF_ENABLE_MULTIMODAL` 单开关）；④ **C4 多请求 bonus 已交付**（批式 protocol runtime，官方 evaluator 校准 run 全部效度门通过：成功率/正确率 1.0、Jain 0.9993、p95 TTFT/TPOT 门槛内、goodput 23.53 tok/s，`APXINF_Q35_MAX_CONCURRENCY` 单开关，单请求路径零回归）。未交付并如实说明：C8（显存边界已量化：8 份请求态需 4.1 GiB 超出 2.36 GiB 预算）、>32K 长上下文、MTP、prefix cache 常驻化。无平台批准的 scorer 产物，不声称任何评分结论。

## 1. 结论先行

最终采用一条“先取得资格、再按分/人时优化”的主线。当前 starter 仓库并没有 Qwen3.8、W4A16、GDN 或 HTTP/SSE 服务实现，因此不能把四周级别的完整系统路线当成两天内的默认承诺：

1. 程仁龙负责离线 oracle 生成器、协议 stub、checkpoint manifest 和 synthetic W4 fixture；王天民交付 shape inventory 与 benchmark/replay harness 的离线脚手架。需要真实 checkpoint 的 oracle 执行由程仁龙在服务器带锁队列中一次完成。当前 transformers/vLLM 仅提供 Qwen3Next 相近实现，必须做 checkpoint-specific 适配并逐层对拍，不能把相近模型直接当权威 oracle；没有完成选择性 golden 生成就不进入大规模 Rust/CUDA 调试。
2. 程仁龙负责一条可回滚的模型/runtime 主线：W4 解包/反量化、GDN 语义、full attention 特殊语义、GPU worker/状态适配和最终集成；并先行独立完成 protocol surface（stub、schema、`/health`、admission、SSE/JSON、错误映射和恢复验收），再把真实模型 runtime 接入这份已测试的接口——协议先冻结，资格闸不与核心 forward 调试相互阻塞。GDN 先保留逐 token eager 路径作为 correctness fallback；chunk-scan 是后续性能路径，不能成为取得资格的唯一前置条件。
3. 只把已证明的窄优化接入主线：prefill 先用 BF16 scratch + cuBLASLt，decode 再比较直读 packed-W4 GEMV；CUDA Graph 只覆盖已冻结 decode bucket；MTP 是 base TPOT/C4 goodput 的条件性实验，不是独立 bonus 或默认交付项，只有 target decoder 冻结后先做 K=1 feasibility probe 且端到端净收益成立才接入。
4. 本地代码准备可以并行；服务器只有一个账号，所有真实 GPU/模型任务进入一个带 `flock` 的队列，不能四卡同时常驻模型。队列优先级为：一次性 oracle（GPU1 logical lane）→ GPU0 runtime/eligibility 集成与正式重放 → GPU2 kernel/profile replay → GPU3 context/C4/C8/vision/MTP replay。GPU0 正式测量必须独占、锁频/记录温度功耗；最终成绩只在固定 GPU0 UUID 上重放。
5. 先保证 public 6/6、hidden >=11/12、protocol/reliability 和 99% success 的资格。轨迹是 5 分软目标，不是 eligibility 闸门；multimodal/context/multi 只能在文本 eligible 后作为增量。

三套路线保留为技术选项，但实际执行不是三条独立全栈项目，而是一条共同主线加两个受限优化 lane：

| 路径 | 目标 | 默认地位 | 进入主线的条件 |
|---|---|---|---|
| A. 成熟算子混合 | 最短路径取得文本资格 | 主线 | oracle、功能题、协议、恢复和基础 cell 可复现 |
| B. SM89 窄特化 | 降低 decode launch/带宽开销 | GPU2 优化路径 | 只做 decode GEMV/graph；同一单卡端到端净收益 |
| C. 状态/内存协同 | 争取 MTP、65K/131K、C4/C8 和视觉 | GPU3 优化路径 | 文本 eligible 后，逐项通过 exact verify/显存/尾延迟门禁 |

推荐组合是 `A + B 中已证明的 decode GEMV/graph + C 中已证明的 paged KV/MTP/C4 组件`。明确砍掉关键路径：全模型 mega-kernel、prefill offline autotune、prefix state cache 和 262K INT4 KV；只有在主线提前冻结且有新鲜证据时才重新评估。

## 2. 成绩目标与不可触碰的合同

### 2.1 目标

- 文字主榜：Correctness 30、TTFT 35、TPOT 25、Reliability 10，共 100 分。
- 加分：context 10、C4/C8 10、多模态 10，共 30 分，排行榜上限 130。
- 代码评审：20 分，分别是测试/负控制 8、接口/错误处理 4、可复现性 4、分析/决策 4。
- 实验工作流创新最多 5 分；不改变 130 分自动排行榜，但可直接影响课程总评。
- 自动课程部分为 `min(80, 0.8 * eligible_leaderboard_score)`，所以不能用 bonus 掩盖基础资格失败。
- `score_submission.py` 在 `eligible=false` 时把 `base_score`、`bonus_score`、`leaderboard_score` 和 automated course points 渲染为 `None`；multimodal 的“独立算法”不能作为文本不 eligible 时的对冲。PR review 20 分不依赖该自动榜资格，因此必须从第一天开始积累证据。

### 2.2 评测事实

- 固定模型：`cyankiwi/Qwen3.8-27B-AWQ-INT4`，revision `63768c10df38c0395e12ef49edac1bd539eaeeea`。
- 合同身份：leaderboard schema `apxinf.qwen38_27b.leaderboard_contract.v1`，接口 `apxinf.qwen38_27b.inference_interface.v1`；最终 artifact 由冻结 evaluator 生成。
- 固定硬件：一张 NVIDIA RTX 4090，SM89，24 GiB 级显存；正式服务 `TP=PP=DP=1`。
- 固定量化：compressed-tensors W4A16、group size 32、asymmetric；不能把全部权重先展开成 BF16。
- 请求级生成参数必须原样遵守：预分词 `input_ids`、`temperature=0`、greedy、thinking 关闭；功能题由 case 指定 `max_new_tokens=64`、`ignore_eos=false`，性能/轨迹/多请求/上下文题通常指定 `max_new_tokens=128`、`ignore_eos=true`。不能把 128 或 `ignore_eos=true` 写成所有请求的固定校验条件。
- 功能题使用严格 `normalized_exact`；服务必须从 checkpoint `generation_config.json` 核对 `eos_token_id=[248046,248044]`，在 `ignore_eos=false` 时一旦遇到任一 EOS 就立即停止且不得继续吐 token。评测器只按跳过 special token 后的解码文本做 exact 比较，不单独检查 stop reason；这不意味着可以省略 EOS stop 逻辑，也不应把“EOS 必须作为 token event 发出”写成额外资格门。usage 必须与实际 token event 数一致。`ignore_eos=true` 时即使遇到 EOS 也必须继续到请求 budget。
- 正式性能：TTFT 五个 cell（1K/2K/4K/8K/16K），TPOT 两个 cell（1K/8K）；每个 cell 1 次 warmup、5 次 measured、CV <= 10%，以客户端时间为准。
- 资格门槛：protocol 通过、public 6/6、midterm hidden 至少 11/12、请求成功率至少 99%，且五个 reliability boolean checks 全部为真；实际目标是请求成功率 100%。五项是 `no_unexpected_oom`、`no_nan`、`no_fallback`、`no_xid`、`service_healthy_after_failure`，任一项为 `false` 都会产生 `reliability_check_failed:<field>`，使 `eligible=false`，不是只扣 Reliability 分。
- 轨迹门槛为 0.0：public/hidden token trajectory 的数值只影响 correctness 分值（midterm 为 2+3 分），不是逐位相等的 eligibility 硬门；仍需记录完整 128-token budget 的 trajectory 字段，不能因阈值为 0 而省略。
- C4/C8：各 32 个请求，success/correctness 100%、Jain >= 0.95、p95 TTFT <= 自身单请求 1.5 倍、p95 TPOT <= 3 倍、无 fallback，结束后 health 正常；goodput 分母是客户端 batch makespan。
- Context：32K 是起点，不是满分；`32640` 是 non-scoring diagnostic，`32768` 按公式仍为 0 分，`65536` 才是首个正分台阶（约 3.33 分），`262016` prompt + 128 output 才是满分长度；六类任务必须 6/6，失败后要恢复 health 和小请求。
- Multimodal：按合同使用 `Qwen3VLProcessor` + `Qwen2VLImageProcessorFast`，处理 448x448 RGB PNG；`deepstack_visual_indexes=[]` 时不注入 deepstack。`/v1/chat/completions` 真实执行 image + text，`max_completion_tokens=32`、`stream=false`、`enable_thinking=false`；public 4/4、hidden 8/8、全请求成功、health 正常且 `multimodal=true` 才能拿满。未完成时 `multimodal=false`，图片请求以 `unsupported_capability` 明确拒绝。

### 2.3 绝对禁止

- 不使用其他 GPU、vLLM、Transformers、CPU 或其他模型作为提交服务 fallback。这里的禁止项针对外部 runtime/模型/GPU；同一 ApxInf 实现内部、受 feature flag 控制且可审计的 eager 回滚路径可以用于 correctness/recovery，但不得把它报告成外部 fallback，`/health.fallback_active` 仍必须为 `false`。
- 不修改 `benchmarks/qwen38_4090/evaluation/` 合同、生成器、scorer 或提交汇总结果。
- 不按 case ID、公开 prompt、答案、固定 token 序列或输出位置硬编码。
- 不把 prefix cache 用作答案缓存；不让 MTP proposal 绕过 target exact verify。
- 不利用 measured repeats 的相同 prompt 通过 prefix cache 制造近零 TTFT；prefix cache 不进入本期关键路径，除非课程规则和独立 cold/hit 对照明确证明其合规且不改变基准语义。
- 不把单 kernel 加速数字当成服务加速结论；最终结论必须有客户端 TTFT/TPOT/goodput 和正确性证据。
- 不在 `/health` 中填写尚未真实验证的 `max_model_len`、`parallel_requests` 或 `multimodal`。

## 3. 从 PDF 得到的执行原则

`Agent4System-0820.pdf` 对本题最有用的不是某个单独 kernel，而是把整个 Serving Stack 作为可测、可回滚、可持续改进的对象。执行上落实为：

```text
固定合同 -> 读取真实状态 -> 提出一个假设 -> 修改一个变量
        -> 编译/正确性 -> 客户端端到端测量 -> profile 解释
        -> 保存 raw evidence -> 接受/拒绝 -> 回滚或进入下一轮
```

必须遵守四条：

1. **短反馈优先**：先做小 shape、单层、单 token、短请求 smoke，再做完整 64 层和长上下文；不要让每个错误都等待完整榜单。
2. **证据可传递**：每轮把瓶颈、失败原因、接受条件和下一步写入 JSON/Markdown 账本，供其他 GPU lane 直接消费。
3. **Undo-and-Retry**：每个候选有完整 SHA、开关和上一通过 revision；优化失败先回到已知良好状态，再换假设。
4. **Target Specialist**：所有结论绑定 `model × hardware × workload × precision`，只迁移机制，不迁移别的模型或 GPU 上的性能结论。

## 4. 目标单卡架构

### 4.1 模块边界

严格遵守当前 ApxInf 的 model -> backend 分层：模型知道层序、状态和融合决策；CUDA backend 只暴露设备管理和单 kernel/单库调用。

| 模块 | 责任 | 第一责任人 |
|---|---|---|
| `apxinf-loader` | safetensors shard/index、revision/shape/dtype 校验、W4 metadata、lazy/mmap | 程仁龙 |
| `apxinf-model/qwen35` | nested config、64 层执行、GDN/attention、request state、graph orchestration | 程仁龙 |
| `apxinf-cuda` | W4A16、GDN、RoPE、norm、FA2、KV/page、graph primitive | 程仁龙 |
| `server`/`main.rs` | 协议 schema、`/health`、SSE/JSON、HTTP/schema admission、HTTP 错误映射、stub 和恢复负控 | 程仁龙（含 device-budget admission、runtime adapter 接入与最终集成） |
| `scripts/campaign` | 多卡编排、paired A/B、环境与 artifact hash | 离线脚手架由**王天民**交付（8/23）；实际 A/B 编排与执行由程仁龙完成 |
| `REPORT.md` | baseline、假设、结果、负结果、复现、回滚 | 程仁龙 |

不要把 Qwen3.8/Qwen3.5 的层序、GDN state 或 decode workspace 硬编码到通用 backend。checkpoint 的 `architecture` 是 `Qwen3_5ForConditionalGeneration`，目录和实现必须以真实 config/tensor 为准，而不是仅按模型商品名猜结构。当前 starter 仓库没有 qwen35、W4A16、GDN 或 HTTP/SSE；这些是新增 vertical slice，不是已有模块的简单接线。

### 4.2 权重与真实混合 dtype

启动时生成 manifest，至少记录 tensor name、shape、dtype、量化 group、packed layout、device bytes 和 dispatch。当前已知分类：

- MLP 与 full-attention projection：packed W4，group 32、asymmetric；`weight_packed` 沿 K 打包，scale 沿 K 按 group-32，zero-point 则沿 N 打包到 I32，必须先按 tensor-specific 轴解包，不能按 `int8` 元素直接读取。
- GDN `in_proj_qkv`/`in_proj_z`：W4。
- GDN `in_proj_a`、`in_proj_b`、conv/norm：BF16；除第 0 层外，GDN `out_proj` 也按真实 index 为 packed W4，不能统一当 BF16。
- embedding、lm_head、MTP、vision：BF16。
- full attention 必须实现 `attn_output_gate=true`：`q_proj` 输出 2 x 6144，按 q/gate split，attention 输出乘 `sigmoid(gate)` 后再进 `o_proj`。
- RoPE 只作用于 `partial_rotary_factor=0.25` 对应的 head_dim 前 64 维；mRoPE section/position 以 config 为准。
- GDN recurrent state 使用 `mamba_ssm_dtype=float32`；按真实 config 计入预算。每层 recurrent state 约 `48 x 128 x 128 x 4 B = 3 MiB`，48 层约 144 MiB/请求，C8 约 1.15 GiB，不能漏算。

当前 revision 的 token admission 以 checkpoint `text_config.vocab_size` 为权威（加载到
runtime model config 后为 `vocab_size`；实测 `248320`，合法范围 `[0,248320)`），不是
tokenizer `vocab_size`（实测 `248044`）。
`image_token_id=248056` 落在 model vocab 范围内，必须被 vision 路径接受；协议仍须拒绝
`4294967295`。R0 必须从实际 `config.json` 记录这些值，代码不能把 tokenizer 大小当作
embedding/lm_head 的边界。

禁止把所有 Linear 统一送进 W4 kernel，也禁止为了省事把全部权重转成 BF16。W4 kernel 的公式、pack/unpack 顺序、scale、zero-point、尾块和 group boundary 必须有 Python compressed-tensors 对照。

### 4.3 64 层状态机

- 48 层 GDN/linear attention：每请求维护 causal convolution ring buffer、FP32 recurrent/delta state、position 和 reset guard。
- 16 层 full attention：每请求维护 GQA KV pages、RoPE position、page table 和 append cursor。
- full attention 配置：24 Q heads、4 KV heads、head dim 256、query width 6144。
- 每请求有独立 `StateHandle`；请求完成、取消、非法参数、capacity rejection、CUDA error 都必须释放 GDN state、KV page、graph slot 和临时 workspace。
- prefill 与 decode 共用语义但允许不同 kernel；chunk 边界必须与逐 token reference 完全一致。

### 4.4 服务接口

单 GPU runtime owner + bounded request channel，HTTP handler 不直接共享可变模型状态。

`GET /health` 只返回真实值：

```json
{
  "status": "ok",
  "evaluation_contract": "apxinf.qwen38_27b.inference_interface.v1",
  "model_revision": "63768c10df38c0395e12ef49edac1bd539eaeeea",
  "max_model_len": 32768,
  "parallel_requests": 1,
  "fallback_active": false,
  "capabilities": {
    "pretokenized_input_ids": true,
    "token_id_output": true,
    "multimodal": false
  }
}
```

上面的 `32768` 和 `1` 仅是合法字段类型的示例；最终值必须替换为已经通过对应容量/并发门禁的真实值。revision 必须逐字使用合同给定的 `63768c10df38c0395e12ef49edac1bd539eaeeea`。

`POST /v1/evaluations/generate` 必须：

- 校验非空、逐项落在 checkpoint `text_config.vocab_size`（加载到 runtime model config 后的
  `vocab_size`）范围内的 token ID（当前实测 `[0,248320)`；不能用 tokenizer `248044`
  作为 embedding 边界；JSON 整数且非负；不能只做 `uint32` 类型检查，`4294967295` 这类
  越界值必须拒绝）、温度为 0、`max_new_tokens` 为正整数且满足
  `prompt_tokens + max_new_tokens <= max_model_len` 及当前真实 device budget、
  `ignore_eos` 为布尔值、`stream` 为布尔值；不要固定要求 `max_new_tokens=128` 或
  `ignore_eos=true`。功能题的 64-token/可 EOS 与性能题的 128-token/强制完整 budget
  必须分别支持。
- `stream=true` 时以 SSE 返回：token index 从 0 连续递增，request_id 单流唯一，结束时发送 usage 和 `[DONE]`；`stream=false` 也是合法请求，必须返回 HTTP 200 且 `type=result` 的 JSON，包含 `output_ids` 和准确 usage。`ignore_eos=false` 时检测 `eos_token_id=[248046,248044]` 并立即终止；不要额外要求 EOS 必须被作为 event 发出，功能 exact 仍由最终解码文本决定。`ignore_eos=true` 时必须输出恰好请求的 `max_new_tokens` 个 token，即使提前生成 EOS 也继续。
- capacity 在 admission 阶段拒绝，不能先触发不可恢复 CUDA OOM；
- 客户端断开时取消请求并回收状态；CUDA error 后做健康探针，context 损坏时不得继续虚报 `status=ok`。

协议资格 gate 必须对 stub 和真实服务分别执行，且逐项保留原始 HTTP/JSON 证据。六个可解析的结构化负控全部设置 `stream=false`；malformed JSON 以原始不可解析 body 发送，不能携带可解析的 `stream` 字段。当前 evaluator 对 malformed JSON 的硬条件仅为 HTTP 400；其余 6 个结构化负控必须同时返回 HTTP 400 和含 `error` 字段的 JSON。实现仍应让 malformed JSON 也返回统一 JSON error，以满足接口一致性和 PR review，但不要把这一建议误写成当前 scorer 的额外资格条件。

| ID | 请求变体 |
|---|---|
| `malformed_json` | 原始 body 为 `{not-json` |
| `empty_input_ids` | `input_ids: []` |
| `negative_token_id` | `input_ids: [-1]` |
| `out_of_vocabulary_token_id` | `input_ids: [4294967295]` |
| `unsupported_temperature` | `temperature: 0.1` |
| `over_budget` | 单 token prompt，`max_new_tokens = health.max_model_len`；由于总预算是 `prompt_tokens + max_new_tokens`，必须拒绝 |
| `unsupported_modality_field` | `images: ["x"]` |

随后执行 `valid_short_nostream_request`：发送一个 8-token、`max_new_tokens=1`、`stream=false` 的合法请求，必须返回 HTTP 200、`type: "result"`、一个 `output_ids` 和 usage（`prompt_tokens=8`、`completion_tokens=1`）。再执行 `health_after_invalid_requests`，检查 `/health.status=ok`；并执行 `health_contract_identity`，核对 `/health.evaluation_contract=apxinf.qwen38_27b.inference_interface.v1`。任何负控污染服务、恢复检查失败或合同身份不符，都使 `protocol_pass=false`。`max_model_len` 因此既是 `/health` 的真实能力声明，也是所有请求的总 token admission 上限，不能只作为展示字段。

## 5. 显存账本与容量策略

### 5.1 先承认约束

checkpoint 约 19.57 GiB，纯文本常驻权重估计约 17.93 GiB。16 个 full-attention 层的 BF16 KV 约为：

```text
64 KiB/token; 8K ~= 0.5 GiB; 16K ~= 1 GiB; 32K ~= 2 GiB; 262K ~= 16 GiB
```

所以 262K 不能靠“权重常驻 + 全量 BF16 KV + 任意 workspace”完成。纯文本权重约 17.93 GiB 后，实测可用余量还要扣除 CUDA context/library、静态 workspace、FP32 GDN state 和碎片；必须启动时测量 `cudaMemGetInfo`。embed_tokens 约 2.37 GiB 是 decode 期间只需一次 gather 的特殊候选，可在 prefill/decode 分阶段评估 host-pinned/chunk residency，但不得把 CPU 计算当成服务 fallback，且必须先做端到端 PCIe/TTFT 实测。

roofline 先给出停止条件：合同的冻结代理 54 GFLOP/token 在 16K prefill 约为 885 TFLOP；4090 BF16 峰值 165.2 TFLOPS 对应约 5.36 s 的乐观算术下限，1K 约 335 ms。合同说明该 FLOP 代理省略部分 elementwise/recurrent work，因此不能把它当成真实服务的绝对物理下限。最大 MLP 中间激活 `16384 x 17408 x 2 B` 约 570 MB（约 544 MiB），所以 prefill chunk 默认不超过 2048。最大 BF16 dequant scratch `17408 x 5120 x 2 B` 约 178 MB（约 170 MiB），双缓冲约 356 MB（约 340 MiB）。把 packed W4 解到 scratch 再调用 BF16 GEMM 的一次读/写代价相对 16K GEMM 约为 1% 数量级；是否重开 prefill 专用 kernel 要看 shape-specific MFU/roofline gap、dequant+scratch 的实测占比，以及客户端 TTFT 的 paired A/B 归因，不能只拿原始 TFLOPS 与理论峰值比较。

decode 的主导约束预计是带宽：backbone 约 13.19 GiB 加 `lm_head` 约 2.37 GiB，即约 15.56 GiB/token。将该二进制流量换算为约 16.70 GB 后，若本机有效带宽实测达到 800--850 GB/s，则乐观下限估计约 19.6--20.9 ms/token；实际还要加 scale/state/cache/launch 流量，必须由 Nsight 和客户端 TPOT 校准。`lm_head` 约占该估算流量的 15.2%，INT8 只作为实验候选，必须同时通过 top-1/top-2 margin、exact correctness、trajectory edit loss 和客户端 TPOT，不能默认接入。

### 5.2 三个 runtime profile

| Profile | 用途 | 默认能力 | 必过门禁 |
|---|---|---|---|
| `text-short` | 1K-16K 基础榜 | W4 + BF16/GDN + BF16 或 paged KV | 100 分 base cell、CV、reliability |
| `text-long` | 32K-131072（先争取） | paged + BF16/INT8 KV compression/dequant | 每一级六类任务、128 output、失败恢复；262016 仅为远期实验，不是本期关键路径 |
| `image` | 多模态 | vision + merger + mRoPE + text | public 4/4、hidden 8/8、健康 |

每个 profile 有独立预算和 feature flags；不要为了同时常驻 vision/MTP/cache 而挤掉 text-long 的安全余量。

### 5.3 KV 压缩顺序

1. 先实现 BF16 paged KV，验证 page table、append、attention 语义和 capacity admission。
2. 逐级测 32640、32768、65536；32640 只是扣除 128-token 输出后的 non-scoring diagnostic，32768 是 bonus 起点但按公式仍得 0 分，65536 才是第一个正分台阶（约 3.33 分）。若 embed residency 和 INT8 cold-page 有真实余量，再测 131072（约 6.67 分）。
3. 196608/262016 与 INT4 KV 不进入 8/24 中期或 8/27 的默认关键路径；只有 131072 已通过且有充足时间时才开独立实验。评测器单请求默认 timeout 为 1800 s，但这不是性能承诺：每次 eager fallback 必须记录实际耗时/launch 数，并评估整轮 hidden + performance campaign 是否可完成。
4. 每次压缩都做 cache-off/reference 对照和边界 position、late retrieval、multi-hop、revision、aggregate 测试；任何失败只声明上一最高完整长度。

## 6. 三条高分优化路径与接收门

### 6.1 路径 A：成熟算子混合主线

默认实现：

- prefill 大 M 第一候选是“按 chunk 将 packed W4 解到 BF16 scratch，再交给 cuBLASLt/CUTLASS BF16 GEMM”；在 16K chunk 上先测量，不预设全模型 prefill 专用 kernel 或 offline autotune 有收益。
- decode M=1/小 batch 第一候选是直读 packed W4 的 GEMV；禁止先把整块权重展开为 BF16，否则会把 decode 带宽翻倍。GDN BF16 projection/状态路径按真实 tensor manifest 分流。
- GDN prefill 同时保留两条路径：先实现逐 token eager/single-step reference-compatible 路径，作为 public/hidden correctness 和失败回滚的资格兜底；再实现一层一个 kernel 的 chunk-scan 作为 TTFT 性能路径。两者必须在 chunk 边界做 state checksum 对拍，chunk-scan 可失败并回退 eager。decode 再逐步融合 ring-buffer、gate、delta update 和 output。
- full attention prefill 使用 FA2；decode 使用 GQA/分页 KV kernel；必须实现 QK norm、partial RoPE(前 64 维)、output gate 和 KV append。
- CUDA Graph 只覆盖已经冻结的 bucket，动态长度和异常路径保持 eager。

接收条件：协议/功能题满足 eligibility；轨迹作为分值记录而非硬门；基础 cell success=100%、CV<=10%；端到端中位数改善超过噪声；显存峰值不超过 profile budget；失败可一键关闭。

### 6.2 路径 B：SM89 特化

只在 GPU2 搜索以下窄变量，不做全模型 mega-kernel：

- packed-W4 M=1 GEMV 的 load/dequant 次序、group-32 tail 和 reduction；
- 已正确的 GDN decode kernel 的 causal-conv/state update 融合；
- decode graph bucket、launch topology、register/shared-memory trade-off；
- prefill kernel 只做 profile/解释；只有 shape-specific MFU/roofline gap、dequant+scratch 的实测占比和客户端 TTFT paired A/B 都支持该归因时，才重开专用 kernel；`165.2 TFLOPS` 与约 `1%` 只是量级参考，“TTFT 是瓶颈”本身不构成条件。

每个候选必须同时提交：shape、tile、register count、occupancy、DRAM bytes、tensor pipe、Nsight artifact、客户端 TTFT/TPOT。若出现 register spill、CV 超限、功能题失败、C4/C8 p95/Jain 失效或端到端收益小于 2% 且落在噪声内，立即拒绝；轨迹下降单独计分，不再作为自动硬拒绝理由。

### 6.3 路径 C：状态/内存/调度

按“分值/人时/风险”排序。文本 `BASE_GOOD` 冻结后，多模态 vertical slice 可以与 C4 validity
在各自分支并行准备；真实 replay 仍服从服务器队列。MTP 的 K=1 probe 是低成本的
TPOT/goodput 探针，但不阻塞 eligibility、C4 或多模态：

1. paged KV 和严格 admission；
2. continuous batching、C4/C8 fairness 和 queue accounting；
3. 多模态 processor/vision vertical slice：只在独立分支和 GPU3 上推进，文本服务保持 `multimodal=false`，不让 vision 调试阻塞 eligibility；
4. MTP 作为 base TPOT/C4 goodput 的条件性实验：target decoder 冻结后先做 K=1，再 K=2/4；exact verify 防止 draft 直接决定输出，但不消除 GDN/KV/conv/position 事务和回滚实现风险；接受率不足以抵消 proposal+verify 成本就关闭。
5. 65K 后若余量充足再做 INT8 cold-page KV，目标 131K；不要把 INT4 KV 或 262K 当默认交付目标。
6. prefix state cache 本期默认不做：重复 measured prompt 可能造成 benchmark gaming，且 cold path、身份审计和状态失效风险都高于预期收益。

MTP/KV 只有在文本已经 eligible、cold path 不退化、端到端 TPOT/context 有净收益、显存和恢复门禁通过后才可打开。MTP 的 exact verify 是正确性必要条件，不等于实现天然零风险；MTP 本身不产生独立 bonus 分值，收益来自 base TPOT 或 C4/C8 goodput 的改善。

## 7. 多 GPU 逻辑 lane 与串行服务器队列

### 7.1 固定 GPU 身份

启动前保存：

```bash
nvidia-smi -L
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv
```

用 UUID 而不是随意的 ordinal 绑定实验。下面的 `GPU0..GPU3` 是逻辑 lane 标签，不表示
允许同时运行四个模型进程：

| GPU | 用途 | 允许修改的范围 | 端口/目录原则 |
|---|---|---|---|
| GPU0 | 主线集成、单卡最终候选、最终重放 | 主线集成和最终服务 | 正式 job；只跑一个候选 |
| GPU1 | 一次性 oracle/正确性 replay | 真实 checkpoint 的选择性 layer golden 和 manifest | oracle job 完成后释放；不要求完整 BF16 常驻 |
| GPU2 | SM89 kernel/Nsight replay | `apxinf-cuda` 候选和 profile 脚本 | 排队执行一个 profile；保存 `.ncu-rep/.nsys-rep` |
| GPU3 | context/C4/C8/vision/MTP replay | benchmark harness、vision 适配和 bonus 分支 | `BASE_GOOD` 后排队执行；不得修改 scorer |

服务器队列规则：所有真实模型/GPU job 先取得 `flock /tmp/apxinf-gpu-job.lock`，记录
`queue_id`、提交时间、优先级、目标 UUID、commit/model/contract SHA、预计时长和 artifact
目录；一次只允许一个 job 持有锁。其余代码和 replay 包准备可以在本地并行，但不能
把“GPU1/2/3 lane”表述成同时运行的证据。GPU0 正式 campaign 前必须确认其他 GPU 无残留
进程，记录温度/功耗/时钟；GPU1-3 结果始终是 development/replay evidence，不是正式成绩。

服务器队列优先级固定为：

1. `P0 oracle`：oracle generator 的真实 checkpoint 选择性 golden，一次性完成；
2. `P1 base`：GPU0 runtime、protocol、correctness、reliability 和 recovery；
3. `P2 kernel`：GPU2 的单变量 paired A/B/profile；
4. `P3 bonus`：GPU3 的 context/C4/C8/vision/MTP replay。

同一优先级按提交顺序执行；运行中的 job 不被并发抢占。artifact 统一写入
`/mnt/chuangxin/team2/artifacts/apxinf/<date>/<commit-sha>/<queue-id>/`。

### 7.2 实验记录格式

每个实验生成一条不可变 JSON 记录，字段至少包括：

```json
{
  "run_id": "20260823-gpu2-w4-m1-004",
  "lane": "sm89",
  "base_sha": "full-40-char-sha",
  "candidate_sha": "full-40-char-sha",
  "contract_sha256": "...",
  "model_revision": "63768c10df38c0395e12ef49edac1bd539eaeeea",
  "gpu_uuid": "GPU-...",
  "hypothesis": "...",
  "command": "...",
  "correctness": {"public": "...", "hidden": "...", "trajectory": "..."},
  "metrics": {"ttft_ms": "...", "tpot_ms": "...", "cv": "...", "peak_vram_mib": "..."},
  "profile_artifacts": ["..."],
  "verdict": "accept|reject|inconclusive",
  "rollback_sha": "full-40-char-sha"
}
```

接受、拒绝和无结论的结果都保留；不删除“失败实验”。这同时服务于 PR review 的分析分和下一轮 Agent 的上下文。

## 8. 团队协作、Git 和 Agent Prompt

### 8.1 责任边界

**实际分工（据 git 记录）**

| 人员 | 贡献占比 | 实际交付 |
|---|---:|---|
| **程仁龙** | **约 95%** | Qwen3.8-27B config/loader/weights、48 层 GDN 与 16 层全注意力 CUDA 实现、W4A16 解包与 GEMM/GEMV kernel、HTTP/SSE 协议 surface 与全部负控、admission 与断连取消、oracle 生成与 hidden 代理集、十七次配对性能实验、显存账本、多模态 vision 塔移植与图文链路、C4 批式调度与批式 kernel、REPORT.md、最终集成与提交 |
| 王天民 | 约 5% | `scripts/campaign/shape_inventory.py`、`scripts/campaign/validate_experiment.py`、`benchmarks/campaign/` 下的 manifest 与 README、`docs/collaboration/records/` 的交接记录 |

通用约束：不修改 `evaluation/` 下任何合同/scorer；不把 profile 数字直接宣称为端到端成绩；无证据时不开启 bonus、不改合同。原规划"各领一个 bounded prompt、一个分支、一个 GPU"的并行组织方式未能落地，实际为单人串行推进；服务器带锁队列与单卡正式重放的纪律始终执行。

### 8.2 分支规约

```text
contest/main                 # 只由程仁龙更新
agent/b/correctness/*        # 协议/正确性任务分支
agent/cuda/<run-id>          # kernel 单实验分支
```

规约：

- 每个任务先创建 branch，禁止直接 push `contest/main`。
- 一个 commit 只做一个可解释变化；commit message 包含实验 ID。
- PR 必须包含：改动文件、命令、测试输出、artifact 路径、基线 SHA、候选 SHA、接受/拒绝理由。
- 程仁龙 cherry-pick 前先在 GPU0 重跑最小正确性；不直接合并“看起来能编译”的分支。
- 同一时刻只有一个任务修改 qwen35 model state machine；同一 CUDA 文件不得被两个任务同时重写。

### 8.3 协议/正确性任务的 Prompt 模板

```text
你是 ApxInf 协议/正确性 Agent，职责是 correctness/service，不做性能架构决策。

仓库：/mnt/chuangxin/team2/ApxInf
分支：agent/b/<TASK_ID>
GPU：只使用已分配的 GPU UUID；不要使用其他 GPU。
固定合同：benchmarks/qwen38_4090/evaluation/contract-v1.json

任务：<只填一个具体任务，例如“实现 W4 metadata manifest 校验”或“建立 Qwen3Next 离线 oracle 适配”>

允许修改：<精确文件列表>
禁止修改：evaluation/ 下任何合同/scorer；GPU0 主线 forward；未经主线批准的 CUDA kernel。

执行顺序：
1. 先阅读 README、相关模块和现有测试；列出你理解的输入/输出契约。协议任务以 stub 为先，不等待完整模型 forward。
2. 先写最小失败测试，再实现最小改动。
3. 运行：cargo fmt --check；cargo test --workspace --locked；Python oracle/协议任务还要运行明确的最小命令和 hidden 代理集。协议 owner 必须逐项执行 malformed 原始 body（硬门为 400）和上面的 6 个 `stream=false` 结构化负控（均为 400 + JSON `error`）、`valid_short_nostream_request`、`health_after_invalid_requests`、`health_contract_identity`，并覆盖 `max_new_tokens=64, ignore_eos=false` 的功能题、`max_new_tokens=128, ignore_eos=true` 的性能题、`stream=true` SSE 和两个 EOS ID 的停止/继续语义。
4. 若失败，保留原始错误，不绕过测试，不添加 fallback。
5. 提交一个小 commit，并回复完整 SHA。

交付格式：
- 修改文件和目的
- 运行命令及关键输出
- 新增测试和负控制
- 未解决问题
- artifact 路径和 SHA256
- 是否建议 cherry-pick：YES/NO，以及理由
```

### 8.4 主线集成任务的 Prompt 模板

```text
你是 ApxInf 主线 Agent，只处理当前明确任务，不扩展范围。

先读：README.md、contract-v1.json、现有 model/backend design、相关测试。
任务：<一个 vertical slice>
完成定义：列出具体文件、接口、失败行为、测试命令和通过条件。

约束：单卡 RTX 4090；W4A16 group32 asymmetric；无 fallback；不改 evaluator；不硬编码 case/答案；保持 model -> backend 分层。

流程：先建立 reference/失败测试 -> 最小实现 -> 公开 correctness -> service/recovery -> client benchmark -> 只有证据支持才融合。
每次改动保留 feature flag 和上一通过 SHA；任何候选都必须可回滚。
```

## 9. 8 月 22 日至 24 日中期倒排

中期目标不是追求 262K 或全部 bonus，而是交出一份可被统一 cohort runner 获取和重放的真实 SHA：能启动、接口真实、文字正确性可测、至少有第一版端到端性能和报告骨架。PDF 明确中期是 Day4 19:00 freeze，之后统一 clean checkout、hidden + vLLM 评测。

| 时间 | 本地开发（实际均由程仁龙完成；王天民 8/23 交付离线脚手架） | 服务器队列 job（一次只运行一个） | 集成动作 | 共同门禁 |
|---|---|---|---|---|
| 8/22 上午 | oracle/protocol skeleton 和 synthetic W4 fixture；shape inventory 与 benchmark runner 脚手架 | `P0 oracle-prep`：核对 GPU/模型/合同 hash、建立 artifact 目录 | 固定 `feat/qwen35-runtime` 和 adapter 边界 | `test.py check`、clean build 基线 |
| 8/22 下午 | oracle generator、12 题 synthetic proxy、6 个 public golden schema；paired harness | `P0 oracle`：执行选择性 layer golden，保存 manifest/hidden state schema | 审核 oracle 输出格式，准备 loader hard gate | oracle artifact 可消费；stub 可探测 |
| 8/22 晚间 | HTTP/SSE stub、schema/admission/恢复；1K/8K timing harness | `P1 protocol-smoke`：在真实服务接线后重放七项 gate和 8-token recovery | 提供最小 runtime adapter contract | protocol gate、接口证据、首个阻塞点可复核 |
| 8/23 上午 | layer golden 对照和 loader manifest；GDN/FA2 profile 配置 | `P1 base-layer`：逐层/逐状态 correctness replay | 接入 loader、W4 dequant→BF16 GEMM 和 vertical slice | 单层 reference 对齐；不宣称完整模型 |
| 8/23 下午 | public/hidden proxy 与协议 regression；单变量 kernel candidate 准备 | `P1 base-eval`：eager GDN、full-attention gate、recovery | 接入 64 层文本执行器、state/cancel | public correctness、failure recovery 可测 |
| 8/23 晚间 | 冻结 oracle/protocol artifact 与 paired A/B replay 包 | `P2 kernel`：只跑已通过 correctness 的单变量 GEMV/Graph candidate | 审核是否进入 BASE_GOOD 候选 | base cell 初测、warmup 1 + measured 5、CV <= 10% |
| 8/24 00:00-12:00 | 完善负控/恢复限制；整理失败实验和报告 | `P1 final-base`，必要时再排 `P2/P3`，不并发启动 | 只修资格/reliability，不开新架构 | public 6/6、hidden >=11/12、protocol、success 目标 |
| 8/24 12:00-17:00 | 从冻结 SHA 生成 replay 包和报告片段 | `P1 clean-replay`：从 clean checkout 重跑最小 smoke | 冻结中期 SHA、服务命令和 artifact | 所有产物由冻结 SHA 生成 |
| 8/24 17:00-19:00 | 备份 PR/raw artifact，不再改集成文件 | 队列清空后只做 final dry-run；不得启动新长任务 | 打 tag、发布 bundle 清单 | 19:00 后不改中期 cohort |

中期明确不做：MTP、prefix cache、262K/INT4 KV、复杂 vision 集成、全模型 mega-kernel、prefill offline autotune。它们不能阻塞中期 SHA；多模态只保留协议 fail-closed 和可行性记录。

## 10. 8 月 24 日至 27 日最终倒排

### 10.1 8 月 24 日晚至 25 日：冻结 base，补高收益路径

- 以中期 SHA 为 `BASE_GOOD`，建立不可变 tag。
- 优先修复 hidden 功能题、GDN state、W4 tail、16K TTFT 和 8K TPOT；trajectory 只按实际分数/边界 margin 排优先级。
- GPU2 只比较一项 decode kernel 变量；GPU3 先完成 BF16 paged KV 和 capacity admission，再做 32K/65K。
- 任何优化必须 feature-off/on 配对；若 correctness 或 reliability 下降，立即回到 `BASE_GOOD`。

### 10.2 8 月 25 日：长上下文与 C4

- 按 32640 -> 32768 -> 65536 逐级验证；32640 只作 non-scoring diagnostic，32768 bonus 为 0 分，65536 才是首个正分台阶。只有 embed residency/INT8 cold-page 余量和功能证据都充分时再测 131072，失败后立即 health + 8-token 小请求。
- 完成 C4 32 请求闭环，客户端计时，success/correctness 100%、Jain、p95 和 health 全部满足后才记录 goodput。
- 若 paged/INT8 KV 的功能题失败或轨迹分值下降超出可接受损失，保留上一通过格式并如实声明最高长度。

### 10.3 8 月 26 日：MTP 可行性、C8 和多模态

- GPU2 logical lane 在 target decoder 已冻结后先排 MTP K=1 feasibility probe；GPU3 的 C8 validity
  只能作为后续队列 job。MTP 只有在 target exact verify、强制首/中/末 reject rollback 和
  off/on TPOT/功能题通过后才接入；接受率或端到端净收益不够就关闭。
- 只有 C8 满足 p95/Jain 才继续追 goodput；MTP probe 不得阻塞 C8 validity 或已通过的 `BASE_GOOD`。
- prefix cache 本期默认跳过；除非主线提前冻结且有明确 cold/hit 合规 A/B，否则不为 measured prompt 复用牺牲基线。
- `BASE_GOOD` 后即可在 GPU3 logical lane 的隔离分支准备 vision/merger/mRoPE；服务器
  replay 必须排队，只有文本服务和 context 配额冻结后才把它接入最终候选。public 4/4
  失败就保持 `multimodal=false`，不要冒险破坏文字服务。多模态不作为文本不 eligible 的对冲。

### 10.4 8 月 27 日：最终冻结

| 时间 | 动作 |
|---|---|
| 00:00-06:00 | 在 GPU0 逐级重跑 base、context、C4/C8、multimodal 已通过项 |
| 06:00-10:00 | clean checkout 构建，记录依赖/CUDA/GPU/model/contract hash |
| 10:00-13:00 | 统一 public run，生成 submission/artifacts；不手改结果 |
| 13:00-15:00 | 运行负控制、capacity failure、服务恢复、200-request soak |
| 15:00-16:30 | 写完 REPORT：baseline、假设、接受/拒绝、限制、回滚和完整命令 |
| 16:30-17:30 | 程仁龙逐项审计 PR review 20 + workflow 5 证据 |
| 17:30-18:30 | 最终 clean replay、SHA/tag、打包 raw artifacts |
| 18:30-19:00 | 提交；19:00 后不再修改最终 cohort |

最终回滚顺序固定为：`MTP K 降低 -> MTP off -> KV 回到上一通过格式 -> C8 回 C4/1 -> multimodal=false -> BASE_GOOD`。prefix cache 默认未启用；每一步都要重新检查 health、功能题 eligibility 和 reliability。

## 11. 验收门与停止规则

### 11.1 主线接收门

候选必须同时满足：

- public correctness 6/6；hidden 模拟目标 12/12，最低不低于合同 11/12；
- public/hidden trajectory 完整记录并计分；不得把轨迹率 0 阈值误写成 eligibility 硬门；
- base 7 cell 请求成功率 100%、输出完整 128 token、CV <= 10%；
- 五个 reliability boolean 必须全部为真：`no_unexpected_oom`、`no_nan`、`no_fallback`、`no_xid`、`service_healthy_after_failure`。评测器会把任一 `false` 写成 `reliability_check_failed:<field>` 并令 `eligible=false`；这不是单纯从 10 分 Reliability 中扣分。容量边界的预期拒绝/可恢复失败必须与计分 cell 中的 unexpected OOM 区分；
- 无 unexpected OOM、NaN、fallback、Xid；失败后服务健康；
- protocol gate 全部通过：malformed 原始 body（硬门为 400）和 6 个 `stream=false` 结构化负控（均为 400 + JSON `error`）、`valid_short_nostream_request`（8-token 非流式 `type=result`，200）、`health_after_invalid_requests`、`health_contract_identity`；`max_model_len` 按 `prompt + max_new_tokens` 执行总预算；
- 峰值显存低于当前 profile 上限并有 safety margin；
- 端到端收益超过测量噪声，或增加一个实际可得 bonus；PR review 证据从早期开始累计；
- 有 feature-off 回滚、原始日志和完整 SHA。

### 11.2 停止规则

- 连续两个实验周期没有端到端收益，换变量或回滚，不继续微调同一假设。
- 只有 kernel elapsed 变快但 client TTFT/TPOT 不变，标记为 rejected 或 inconclusive。
- 任何功能题/协议/可靠性变化优先于性能数字；轨迹变化按 5 分软目标和 top-1/top-2 margin 评估，不得掩盖功能题失败。
- 1800 s 是 evaluator 的单请求默认 timeout，不是 eager GDN 的性能目标；若 eager 请求耗时或 launch 数使整轮 hidden/性能 campaign 无法在截止时间前完成，立即转为 chunk-scan 性能实验或缩小已声明能力，不能把超时当作可接受的最终交付状态。
- bonus 方向在 8 月 26 日 12:00 仍未达到 validity，停止扩展，确保 eligible base、PR review 和报告可交；262K/prefix cache 不因截止时间强行开启。
- 多模态在最终冻结前不能通过 public + hidden + health，则安全拒绝图片，保住文字主榜。

## 12. REPORT 与 PR 证据清单

### 12.1 REPORT 必须包含

- 固定合同、model revision、GPU UUID、driver/CUDA、完整 commit SHA 和 artifact hash；
- baseline 与每个 accepted/rejected/inconclusive 实验的假设、命令、输入 hash、TTFT/TPOT median/CV、显存、正确性和结论；
- 服务协议证据必须逐项记录 malformed 原始 body（HTTP 400）和 6 个 `stream=false` 结构化负控（空 token、负 token、`4294967295`、`temperature=0.1`、`max_new_tokens=max_model_len` over-budget、`images:["x"]`，均为 HTTP 400 + JSON `error`）；实现仍可统一让 malformed 返回 JSON error。还要记录 `valid_short_nostream_request`（8-token 非流式 HTTP 200 `type=result`）、`health_after_invalid_requests`、`health_contract_identity`、functional `64 + ignore_eos=false`、performance `128 + ignore_eos=true`、`eos_token_id=[248046,248044]` 的停止/继续行为，以及 usage 与实际 token event 数一致性；
- 公开/隐藏 trajectory 方法，说明 token ID 使用离散 Levenshtein，不使用语义 judge；
- roofline 账本至少列出 54 GFLOP/token、165.2 BF16 TFLOPS、约 5.36 s@16K 算术下限、约 15.56 GiB/token decode 流量和按单位换算后的约 19.6--20.9 ms/token 乐观带宽估计；
- context 六类任务、C4/C8 validity、goodput、p95、Jain 和失败恢复；
- 若实际启用 MTP，记录 proposal/target/accepted prefix/rollback checksum；prefix cache 默认未实现，并在 REPORT 说明因 benchmark-gaming/冷路径风险而砍掉；
- 至少一个真实失败实验、一个负控制和一次回滚演练；
- clean checkout 复现命令和已知限制。

### 12.2 PR review 映射

| 评审项 | 证据 |
|---|---|
| 测试/负控制 8 | W4/GDN/KV/state/protocol/cancel/OOM/recovery/vision 测试与故障注入 |
| 接口/错误处理 4 | `/health` 真实性、严格 schema、SSE、容量拒绝、unsupported capability |
| 可复现性 4 | lockfile、CUDA arch、GPU/model/contract hash、raw artifact 和一键命令 |
| 分析/决策 4 | roofline/NSys/NCU 解释、paired A/B、负结果和回滚理由 |
| 工作流创新最多 5 | 带锁 GPU 队列、服务器一次性 oracle、hidden 代理集、typed evidence、自动化 gate、可重放 campaign |

## 13. 最终提交前一键审计

```bash
cd /mnt/chuangxin/team2/ApxInf
# 用启动前 nvidia-smi 记录的 GPU0 UUID 替换；不要使用未核实的 ordinal。
export CUDA_VISIBLE_DEVICES=GPU0_UUID_FROM_NVIDIA_SMI

python3 benchmarks/qwen38_4090/evaluation/test.py check
cargo fmt --all -- --check
cargo check --workspace --locked
cargo test --workspace --locked
git diff --check
git status --short

# 以下命令必须使用最终实现的真实服务 CLI；参数只填写已经验证的值
python3 benchmarks/qwen38_4090/evaluation/test.py prepare \
  --model-dir /mnt/chuangxin/team2/models/Qwen3.8-27B-AWQ-INT4
python3 benchmarks/qwen38_4090/evaluation/test.py run \
  --model-dir /mnt/chuangxin/team2/models/Qwen3.8-27B-AWQ-INT4 \
  --base-url http://127.0.0.1:8001
```

`APXINF_QW35_*` feature flag、`max_model_len`、`parallel_requests` 和 `multimodal` 的最终值只能来自最后一次 GPU0 真实验证，不能照抄计划中的示例。

容量验证的保守起点是 `max_model_len=32768`、`parallel_requests=1`；32768 仅为 diagnostic 且 context bonus 为 0，只有真实容量和恢复门禁通过后才能提高声明值。

## 14. 一句话决策

程仁龙维护一条可回滚的模型/runtime eligibility 主线并负责最终集成，同时交付 protocol surface、stub、负控/恢复验收和 oracle generator/hidden 证据；decode/bonus 测量只在 vertical slice 正确后进行；王天民交付离线 benchmark 脚手架。本地代码准备可并行，服务器真实 GPU/模型任务由带锁队列按 oracle → GPU0 base → GPU2 kernel → GPU3 bonus 顺序执行，GPU0 正式 campaign 独占并单卡裁决。先拿到 public 6/6、hidden >=11/12、完整 protocol gate、五项 reliability boolean 和 99% success，再按分/人时打开 decode graph、65K/131K、C4、MTP K=1 probe、C8 和多模态；prefix cache、262K/INT4 KV、mega-kernel 不进入默认交付范围。
