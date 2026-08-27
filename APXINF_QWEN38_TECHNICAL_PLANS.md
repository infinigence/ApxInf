# ApxInf Qwen3.8-27B INT4：统一 correctness 主线与三类优化 lane

> 日期：2026-08-22（基于 starter 现实状态、评分器复核与 checkpoint header 复核的纠偏版）
>
> 项目：单张 RTX 4090 上实现并优化 `cyankiwi/Qwen3.8-27B-AWQ-INT4`
>
> 文档性质：实现前的技术决策、执行计划与验收规范
>
> 目标：三套方案描述不同的高分优化路径；它们共享同一个 correctness/service 底座，实际执行以“先 eligible、再按分/人时优化”为准。当前 starter 尚未实现 Qwen3.8、W4A16、GDN 或 HTTP/SSE，因此不能把所有 bonus 视为本期默认交付，也不能把三套方案当成三个独立全栈项目。

> 执行入口：结合 `Agent4System-0820.pdf`、多卡实验环境和 2026-08-24/27 截止时间后的单主线执行方案见 [APXINF_FINAL_EXECUTION_PLAN_2026-08-22.md](APXINF_FINAL_EXECUTION_PLAN_2026-08-22.md)。本文件保留三条高分路线的技术比较；实际合入、分工和冻结以执行入口为准。

> **分工（据 git 记录，2026-08-26 订正）**：本项目为两人团队。
>
> | 成员 | 贡献占比 | 实际承担 |
> |---|---:|---|
> | **程仁龙**（`mygqdmmss`） | **约 95%** | 模型/runtime 主线、CUDA kernel、protocol surface、oracle 与 hidden 代理集、性能实验、显存账本、多模态与 C4 两项 bonus、REPORT 与最终集成 |
> | 王天民（`wang1145140503`） | 约 5% | 离线 benchmark 脚手架（shape inventory、experiment validator、campaign manifest 与交接记录，8/23 交付） |
>
> 原规划的多人并行组织实际退化为程仁龙单人串行推进（该偏差记录于执行入口文档）。代码准备可并行，服务器真实 GPU/模型任务进入带锁队列；正式成绩只以 GPU0 的单卡重放为准。
>
> **实际执行结果（2026-08-27，证据见 REPORT.md）**：文本 eligibility 全门禁通过；TTFT 最高约 48 倍、TPOT 约 2 倍（17 次配对实验，9 接受 8 拒绝）；**多模态与 C4 两项 bonus 均已交付**——多模态为 vision 塔 CUDA 移植 + `/v1/chat/completions`（自建探针 4/4、文本数值零变化），C4 为批式 protocol runtime（官方校准 run 全部效度门通过：Jain 0.9993、goodput 23.53 tok/s、单请求零回归），各有独立单开关可回滚。方案一主线的技术路线按规划落地：GDN 先逐 token eager 取资格、后接 chunk 序列路径，prefill 解量化 + BF16 tensor-core GEMM（规划的 M-W4-P，TTFT −74%），全注意力 gate/partial-RoPE 语义逐层对拍通过；规划外由测量发现的最大杠杆是 dequant scratch 池化、行协作 causal softmax（16K 真瓶颈）与 prefill chunk 64→512。方案二规划的 M=1 decode GEMV 优化以 packed GEMV 落地（TPOT −35%），其余五个 GEMV 微优化候选全部实测拒绝（含 SASS 反汇编根因），CUDA Graph 未实施（decode 归因显示收益不在 launch 开销）；方案三规划的 C4 落地、C8 因显存边界未交付（已量化）；长上下文 >32K、MTP、prefix cache 常驻化未进入交付。无平台批准的 scorer 产物，不声称任何评分结论。

---

## 1. 执行结论

本任务不是三条独立全栈路线的等量三选一。三案共享一个必须从零建立的 correctness/service 底座；真正可执行的组织方式是“一条主线 + 两个受限优化 lane”。

1. **方案一：成熟算子混合主线**。以 cuBLASLt/CUTLASS、FA2 和现有 ApxInf CUDA 能力为主，先补 W4A16、GDN 语义（逐 token eager 资格路径，再接 chunk-scan 性能路径）和 Qwen3.8 特殊 attention；协议 surface 先独立完成并验收，再做 runtime adapter/worker 集成（均由程仁龙完成）。它不是“保底版”，而是两天内最现实的资格主线。
2. **方案二：SM89 窄特化路径**。只针对 M=1 packed-W4 GEMV、已正确的 GDN decode 融合和 CUDA Graph 做 A/B；不把 prefill offline autotune、persistent mega-kernel 当作前置条件。
3. **方案三：状态/内存协同路径**。优先 paged KV、C4/C8 和条件性 MTP；MTP 属于 base TPOT/C4 goodput 优化而非独立 bonus，先在 target decoder 冻结后做 K=1 feasibility probe；prefix state cache、262K/INT4 KV 默认砍掉，只有主线提前冻结且有合规证据才重开。

130 分是自动榜理论上限，不是任一 lane 的统一硬门。评分器在 `eligible=false` 时把自动榜各项渲染为 `None`；多模态的独立算法不会挽救文本 eligibility。当前最优执行组织是方案一主线，方案二/三只吸收已通过门禁的局部成果，未通过的 bonus 明确报告为 0/unsupported，不阻塞文本提交。

### 1.1 选择摘要

| 路线 | 主要得分杠杆 | 性能上限 | 风险等级 | 最大风险 | 首选条件 |
|---|---|---:|---:|---|---|
| 方案一：成熟算子混合 | 资格、协议、W4/GDN/attention 主线 | 最高的现实期望值 | 最低 | 从零实现量大，GDN 是关键路径 | 先拿 eligible，再争基础性能 |
| 方案二：SM89 窄特化 | M=1 GEMV、decode graph、窄融合 | 局部性能上限高 | 最高 | 数值错误、寄存器压力、端到端不增益 | 有 CUDA 调优能力且主线不被阻塞 |
| 方案三：KV/C4/MTP | context、goodput、TPOT 条件性增益 | 场景相关 | 中高 | 状态一致性、显存、MTP 接受率 | base eligible 后再开 |

在实现、oracle 和第一条 vertical slice 完成前，无法诚实地给出可靠的百分比成功率；表中的风险等级只用于排序，不是成绩承诺。任何方案只有在统一评测器的正式门禁通过后才能声称获得相应能力或分数。

---

## 2. 合同、评分和硬性边界

### 2.1 得分结构

| 部分 | 分值 | 满分所需结果 |
|---|---:|---|
| Correctness | 30 | 协议正确；公开功能 6/6；隐藏功能至少 11/12，冲满分目标为 12/12；trajectory 记录并计分，但当前阈值为 0，不是 eligibility 硬门 |
| TTFT | 35 | 1K、2K、4K、8K、16K 五个 cell 均有效，warmup 1 次、测量 5 次、CV≤10%，中位数进入同轮最优参考竞争 |
| TPOT | 25 | 1K、8K 两个 cell 均有效，完整输出 128 token，CV≤10%，中位数进入同轮最优参考竞争 |
| Reliability | 10 | 资格门槛为总请求成功率≥99%，且 `no_unexpected_oom`、`no_nan`、`no_fallback`、`no_xid`、`service_healthy_after_failure` 五项全部为真；任一为假都会触发 `reliability_check_failed:<field>` 并使 `eligible=false`，不是只扣分；冲满分目标为 100% 成功率 |
| 长上下文 bonus | 10 | 最高在 262016 prompt + 128 output；该长度六类任务 6/6，随后健康检查和小请求成功 |
| C4/C8 bonus | 10 | 两个 cell 各 32 请求；先满足成功率/正确率 100%、Jain≥0.95、p95 TTFT≤自身单请求 1.5 倍、p95 TPOT≤3 倍、无 fallback、结束后健康，再以同轮最佳有效 goodput 争取每个 cell 的完整动态分值 |
| 多模态 bonus | 10 | public 4/4、hidden 8/8、请求全成功、服务健康、`multimodal=true`、`fallback_active=false` |
| PR review | 20 | 测试与负控制 8、接口与错误处理 4、可复现性 4、分析与决策 4 |

排行榜上限是 130 分。课程自动化部分为 `min(80, 0.8 × eligible_leaderboard_score)`，另有 PR review 20 分。评分器在 `eligible=false` 时把自动榜各项渲染为 `None`；因此第一目标是资格，PR review 证据必须从第一天积累，bonus 不能作为资格失败时的对冲。

### 2.2 不可违反的边界

- 固定单张 RTX 4090、compute capability 8.9；不得使用其他 GPU 或多 GPU。
- 固定模型 revision `63768c10df38c0395e12ef49edac1bd539eaeeea`，固定 compressed-tensors W4A16、group size 32、asymmetric。
- 运行时不得 fallback 到 vLLM、Transformers、CPU、其他模型或其他 GPU；同一 ApxInf 实现内部受 feature flag 控制的 eager/chunk 回滚路径属于实现内 recovery，不是外部 runtime fallback，但必须可审计且不能让 `/health.fallback_active` 变为 `true`。
- 可以用 Python/Transformers 生成离线 reference 和公开数据，但提交服务的推理路径必须由 ApxInf 执行。
- 不得修改 `benchmarks/qwen38_4090/evaluation/` 下的合同、生成器或评测逻辑来提高成绩。
- 不得按 case ID、公开 token 序列、固定 prompt、已知答案或输出位置硬编码。
- `/health` 中的 model revision、`max_model_len`、`parallel_requests`、multimodal 和 fallback 状态必须与真实能力一致。
- 长上下文必须完整生成 128 token；不能仅修改声明值或只通过短输出探针。
- 图片能力只有在真实 ApxInf vision + text 全路径通过 public 4/4 和 hidden 8/8 后才能报告 `multimodal=true`。

### 2.3 满分策略的得分优先级

三案都以 130 分为理论目标，但执行顺序必须先满足 eligibility，并受两天中期和五天最终期限约束。MTP 不在评分合同中作为独立 bonus；它只可能通过 base TPOT 或 C4/C8 goodput 间接增分，必须在 target decoder 冻结后先做 K=1 净收益探针：

```text
算子/层级数值正确
        ↓
服务协议 + 公开/隐藏 correctness + 失败恢复
        ↓
基础 7 个 TTFT/TPOT cell 稳定有效（CV≤10%）
        ↓
单请求性能优化与显存余量冻结
        ↓
长上下文 → C4 → MTP K=1 probe / 多模态 replay（按服务器队列串行） → C8 → 其余逐项验收
        ↓
clean-checkout 重放 + REPORT/PR 证据审计
```

这不是降低目标，而是防止 bonus 或激进优化破坏进入性能排名的资格。每项能力都应有独立 feature flag 和最近通过的可回滚 revision。

---

## 3. 已验证的环境、模型与项目事实

### 3.1 环境验证

- 工作目录：`/mnt/chuangxin/team2/ApxInf`。
- GPU：环境可见 NVIDIA GeForce RTX 4090，compute capability 8.9，单卡显存 24564 MiB；当前主机可见多张卡，但提交与测试必须显式固定一张卡。
- 驱动：580.82.07；CUDA 编译器：12.8。
- 本地模型目录：`/mnt/chuangxin/team2/models/Qwen3.8-27B-AWQ-INT4`。
- `python3 benchmarks/qwen38_4090/evaluation/test.py check` 已通过。
- 本地模型文件、公开 corpus hash、模型 revision 和所需 Python 依赖均已核验。
- 当前 `transformers==4.57.0` 与 `vllm==0.11.0` 都没有原生 `Qwen3_5ForConditionalGeneration`/`qwen3_5` runtime（可见的相近实现是 `Qwen3Next*`）；因此 Qwen3Next 只能作为离线语义适配起点，不能未经逐层对拍就当作权威 oracle。oracle 交付必须包含 checkpoint-specific config 修补、W4 解包、GDN state、full-attention gate/partial-RoPE 和逐层 hidden/state/logit 对照。

oracle 的执行边界：生成器、manifest/schema、synthetic W4 fixture 和选择性
layer/stage 参数先在本地准备；真实 checkpoint 由程仁龙在服务器 GPU1 logical lane
通过全局 lock 执行，一次生成可审计的 hidden/state/logit golden。本地侧不下载或
展开完整模型，只消费 golden artifact 和 SHA256。完整 BF16 展开规模、显存峰值和实际可生成的
层集合以服务器 manifest 为准，不写成远程电脑的环境前置条件。

服务器保留 raw oracle 产物和大 golden；本地重放只消费批准导出的
最小 golden bundle（不含模型权重、私有 hidden 答案或凭据）。导出方式、文件清单和
SHA256 写入 `docs/collaboration/templates/oracle-handoff.md`；没有批准的 bundle 时，
本地侧只使用 manifest/schema/hash，不自行复制服务器目录。

### 3.2 模型结构

- 架构：`Qwen3_5ForConditionalGeneration`，`model_type=qwen3_5`。
- 64 个文本层，hidden size 5120，MLP intermediate size 17408。
- 每四层中前三层为 linear attention/GDN，第四层为 full attention，共 48 个 GDN 层和 16 个 full-attention 层。
- full attention：24 query heads、4 KV heads、head dimension 256，query width 6144。
- GDN：16 key heads、48 value heads，key/value head dimension 128，causal convolution kernel 4。
- 原生最大位置为 262144。
- checkpoint 含视觉塔：27 层，vision hidden size 1152；`deepstack_visual_indexes=[]`，默认不实现 deepstack 注入。
- checkpoint 含 `mtp.*` 张量和一层 MTP 权重；当前 `config.json` 的 `mtp_num_hidden_layers` 字段并不能作为可靠开关，启用前必须以权重索引和实际 tensor shape 为准解析，不依赖版本差异的配置字段。
- 量化是逐模块混合的，不是所有 Linear 都为 W4：MLP 和 full-attention projection 使用 packed W4；GDN 的 `in_proj_qkv`/`in_proj_z` 为 W4，`in_proj_a`、`in_proj_b`、conv/norm 为 BF16，而除第 0 层外 GDN `out_proj` 也是 packed W4；embedding、lm_head、MTP 和 vision 为 BF16。`weight_packed` 沿 K 打包，scale 沿 K 按 group-32，zero-point 沿 N 打包到 I32；两个 pack 轴不能混用。kernel dispatch 和显存预算必须按真实 tensor dtype 分类。
- W4 方向性验收使用 synthetic fixture：`k_proj.weight_packed [1024,640] I32`、scale
  `[1024,160] BF16`、zero-point `[128,160] I32`；`down_proj` 对应为 packed
  `[5120,2176]`、scale `[5120,544]`、zero-point `[640,544]`。尾块、极值 nibble、
  group boundary 和 N/K 互换必须有定向断言，不能从真实 checkpoint 提交 tensor slice。
- token admission 以 checkpoint `text_config.vocab_size` 为边界（加载到 runtime model
  config 后为 `vocab_size`；实测 `248320`），不是 tokenizer `vocab_size`（实测
  `248044`）；`image_token_id=248056` 是合法 model token。
- config 的 `attn_output_gate=true` 要求 q_proj 的 12288 输出拆为 q/gate，attention 输出乘 `sigmoid(gate)`；`partial_rotary_factor=0.25` 表示 RoPE 只作用于 head_dim 前 64 维；`mamba_ssm_dtype=float32` 要求 GDN recurrent state 用 FP32。

### 3.3 权重与显存预算

本地 checkpoint 约 19.57 GiB。按 tensor 名称估算：语言主干约 13.19 GiB，embedding 约 2.37 GiB，lm_head 约 2.37 GiB，vision 约 0.86 GiB，MTP 约 0.79 GiB。纯文本常驻权重约 17.93 GiB；在 24564 MiB 显存上，CUDA context、workspace、FP32 GDN state、KV、临时 buffer 和碎片可用空间很紧。完整 BF16 展开规模只作为服务器 manifest 的预算事实，不是本地重放的前置条件。

16 个 full-attention 层的 BF16 KV 约为：

```text
每 token = 16 layers × 2(K,V) × 4 KV heads × 256 × 2 bytes = 64 KiB
8K  ≈ 0.5 GiB
16K ≈ 1.0 GiB
32K ≈ 2.0 GiB
262K ≈ 16.0 GiB
```

因此 32K 以上不能靠普通 BF16 全量 KV 常驻来完成；本期先以 65K 为现实目标，embed_tokens 约 2.37 GiB 可作为 decode 期间只需一次 gather 的分阶段 residency 候选，再争取 131K。32640 是 non-scoring diagnostic，32768 按合同公式仍为 0 分，65536 才是第一个正分台阶（约 3.33 分），131072 约 6.67 分。262K/INT4 KV 只有在前级能力已通过且有充足余量时才做实验，不进入默认交付承诺。所有 attention 计算仍必须在 CUDA 上，不能以 CPU 推理作为 fallback。

roofline 账本：合同给出的冻结代理 54 GFLOP/token 在 16K prefill 约 885 TFLOP，4090 BF16 峰值 165.2 TFLOPS 对应约 5.36 s 的乐观算术下限（1K 约 335 ms）；合同说明该代理省略部分 elementwise/recurrent work，不能当作真实服务的绝对物理下限。16K 的最大 MLP 中间激活 `16384 x 17408 x 2 B` 约 570 MB（约 544 MiB），因此 chunk 默认不超过 2048；最大 BF16 dequant scratch `17408 x 5120 x 2 B` 约 178 MB（约 170 MiB），双缓冲约 356 MB（约 340 MiB）。W4->BF16 scratch 的读/写代价相对 16K GEMM 约 1% 数量级，只有实测 MFU 显著低于 165.2 TFLOPS 且该代价明显超出此量级时才重开 prefill 专用 kernel。decode 每 token 的权重流量估算约 15.56 GiB（backbone 13.19 + lm_head 2.37）；换算为约 16.70 GB 后，若本机有效带宽实测达到 800--850 GB/s，乐观下限约 19.6--20.9 ms，实际仍需加 scale/state/cache/launch 流量并由 Nsight/客户端 TPOT 校准。`lm_head` 约占该估算流量 15.2%，INT8 只能作为带 top-1/top-2 margin、exact、trajectory 和端到端 TPOT 对照的实验。

该 roofline 数字是停止条件的量级估计，不是单一硬阈值；上一句中的 `165.2 TFLOPS` 和“约 1%”只用于量级筛查，不能单独触发或阻止重开 kernel。是否重开 prefill 专用 kernel 必须同时看具体 shape 的 MFU/roofline gap、dequant+scratch 的实测占比，以及客户端 TTFT 的 paired A/B 归因，不能只拿原始 TFLOPS 与理论峰值比较。

### 3.4 现有 ApxInf 能力与缺口

可复用能力：

- Rust workspace、model registry、`LlmTrait`、统一 `AutoModel`。
- CUDA C ABI、cuBLAS/cuBLASLt、CUTLASS adapters、FA2、custom kernels。
- CUDA Graph primitive、静态 workspace、GPU argmax、KV cache。
- Qwen3VL 文本/视觉结构、图像输入抽象、视觉 CUDA 组件和参考测试方法。
- NVTX、profiler、tuning 配置和既有 kernel 回归实践。

文本 eligibility 必须新增或大幅扩展：

- Qwen3.8 混合主干的 config、loader、48 个 GDN 层、16 个 full-attention 层和请求状态。
- compressed-tensors W4A16 asymmetric group-32 的 loader、weight view/packing 和 GEMM。
- 评测需要的 HTTP `/health`、`/v1/evaluations/generate`、严格 SSE 和错误恢复；starter 当前没有 HTTP/SSE 依赖或服务骨架，必须作为独立 vertical slice 从零加入。

文本主线冻结后再逐项打开的可选得分扩展：

- 65K/131K 长上下文的内存方案、C4/C8 调度、Qwen3.8 多模态适配。
- MTP exact-verify 状态机；prefix state cache 默认不做。

---

## 4. 共同 correctness/service 底座

任何方案都先构建下述共同底座。它是三案之间可复用、可对照和可回滚的 correctness anchor。

### 4.1 建议文件边界

| 路径 | 职责 |
|---|---|
| `crates/apxinf-loader/src/safetensors.rs` | shard index、mmap/lazy tensor、packed metadata 与 revision/shape 校验 |
| `crates/apxinf-model/src/qwen35/config.rs` | 独立解析 nested text/vision config、layer types、rope、GDN 和 MTP 元数据 |
| `crates/apxinf-model/src/qwen35/weights.rs` | BF16 与 packed W4 tensor 类型、别名/tied weight、加载清单、显存预算 |
| `crates/apxinf-model/src/qwen35/model.rs` | 混合 64 层 forward、prefill/decode、请求级 state 与 reset |
| `crates/apxinf-model/src/qwen35/gdn.rs` | GDN 层语义、conv/recurrent state 编排、reference/eager 路径 |
| `crates/apxinf-model/src/qwen35/attention.rs` | Q/K norm、RoPE、GQA、KV append、prefill/decode attention 编排 |
| `crates/apxinf-model/src/qwen35/scheduler.rs` | admission、continuous batching、C4/C8、公平性和取消/故障隔离 |
| `crates/apxinf-cuda/adapters/w4a16_adapter.cu` | W4A16 C ABI 与成熟/定制 kernel dispatch |
| `crates/apxinf-cuda/adapters/qwen35_gdn_adapter.cu` | GDN causal conv 和 recurrent/chunk kernel C ABI |
| `crates/apxinf-cuda/src/ffi/w4a16.rs`、`qwen35.rs` | 安全 Rust wrapper、shape/dtype/device 检查 |
| `src/server.rs`、`src/server/*.rs` | Axum/Tokio 协议 surface、schema、SSE/JSON、admission、错误映射、stub、health 和恢复；真实 runtime adapter/worker 接入由程仁龙完成 |
| `tests/qwen35_*`、`crates/apxinf-*/tests/qwen35_*` | loader、kernel、layer、trajectory、协议、并发和恢复回归 |
| `scripts/qwen38_campaign.py` | 不改 evaluator 的实验编排、A/B 配对、环境和 hash 记录 |
| `REPORT.md` | baseline、假设、结果、负实验、取舍、复现和回滚 |

以下是可选 lane 的文件边界，不属于取得文本 eligibility 的共同底座：`qwen35/decode_graph.rs`（验证后接入的 CUDA Graph）、`qwen35/vision.rs`（多模态）、`qwen35/mtp.rs`（MTP exact verify）和 `qwen35/cache.rs`（默认关闭的 prefix cache 审计实验）。

项目现有设计要求“模型结构在 `apxinf-model`、单 kernel API 在 backend、CUDA Graph 构造由模型拥有”。因此不能把 Qwen3.8 层序和 workspace 硬编码进通用 CUDA backend，也不能把混合主干塞进现有 `qwen3vl` 目录。目录名称实现时应以 checkpoint 的 `qwen3_5` 为准，本文统一写作 `qwen35` 以避免小数点路径歧义。

### 4.2 正确性金字塔

1. **权重级**：逐 tensor 核对名字、shape、dtype、packed `weight_shape`、scale、zero-point、group 32、asymmetric 公式；用 Python `compressed_tensors` 做 pack/unpack 对照；拒绝缺 tensor、重复 tensor、错误 revision 和隐式转 BF16 全量常驻。
2. **算子级**：用小 shape 和真实 shard 切片比较 W4A16 dequant/GEMM、RMSNorm、RoPE、QK norm、causal conv、GDN update、FA2、argmax；覆盖非对齐尾块、极值 zero-point、长 position。
3. **层级**：对 GDN 层和 full-attention 层分别保存 reference hidden/state/KV/logits；比较 prefill、单步 decode、多步 decode 和 reset 后重跑。
4. **模型级**：公开功能 6/6，hidden 代理集目标 12/12、最低 11/12；trajectory 的数值按合同计分，但不是逐位 256/256 的资格硬门，字段仍要完整记录。功能题按 case 的 64 token / `ignore_eos=false` 运行，性能题按 128 token / `ignore_eos=true` 运行。
5. **服务级**：协议负控制、连续 SSE index、单 stream request_id、usage、完整预算、并发隔离、客户端断开、容量拒绝和失败后恢复。
6. **优化级**：任何 kernel、graph、cache、MTP、KV 压缩、并发或视觉优化都做 cache-off/feature-off 对照，并重新通过上面所有门禁。

数值误差阈值只能用于定位中间层；最终 correctness 以 exact validator 和 token edit trajectory 为准。greedy argmax 的 top-1 边界必须额外记录 top-1/top-2 margin，优先修复可能改变 token 的误差。

### 4.3 服务合同实现

`GET /health` 必须返回并持续核验：

- `status=ok`；
- `evaluation_contract=apxinf.qwen38_27b.inference_interface.v1`；
- 固定 model revision；
- 真实验证的 `max_model_len`；
- 真实验证的 `parallel_requests`；
- `fallback_active=false`；
- `pretokenized_input_ids=true`、`token_id_output=true`；
- 多模态未全部验收前 `multimodal=false`。

`POST /v1/evaluations/generate` 必须严格接收预分词 `input_ids`、`temperature=0` 和请求声明的 `stream`。每个 `input_ids` 必须是 JSON 整数、非负且落在 checkpoint `text_config.vocab_size` 范围内（加载到 runtime model config 后为 `vocab_size`；当前 revision 实测 `[0,248320)`）；不能用 tokenizer `vocab_size=248044` 作为 embedding 边界，因为 `image_token_id=248056` 仍是合法 model token；也不能只检查 `uint32` 而放过 `4294967295` 这类越界值。校验 `max_new_tokens` 为正整数，并同时满足 `prompt_tokens + max_new_tokens <= max_model_len` 与当前真实 device budget；`max_model_len` 是总 token admission 上限，不是只供 `/health` 展示的字段。`ignore_eos` 为布尔值，不能固定要求 128 或 `ignore_eos=true`。功能题通常是 `max_new_tokens=64, ignore_eos=false`，性能/轨迹/上下文/多请求题是 `max_new_tokens=128, ignore_eos=true`。服务从 checkpoint `generation_config.json` 核对 `eos_token_id=[248046,248044]`：前者遇到 EOS 立即停止，后者即使遇到 EOS 也继续到完整 budget。评测器只按跳过 special token 后的解码文本做 exact 比较，不单独检查 stop reason；这不意味着可以省略 EOS stop 逻辑，也不应把“EOS 必须作为 token event 发出”写成额外资格门。`stream=true` 时 SSE index 必须连续，终止事件包含 usage 和 `[DONE]`；`stream=false` 必须返回 HTTP 200 且 `type=result` 的 JSON，包含 `output_ids` 和 usage。两种模式的 usage 都与实际返回 token 数一致。

协议资格 gate 对 stub 和真实服务分别执行：6 个可解析的结构化负控都用 `stream=false`；malformed JSON 以原始不可解析 body 发送。对 malformed JSON，当前 evaluator 的硬条件是 HTTP 400；对其余 6 个结构化负控（`input_ids=[]`、`[-1]`、`[4294967295]`、`temperature=0.1`、`max_new_tokens=health.max_model_len` 的 over-budget、`images:["x"]`），硬条件是 HTTP 400 且 JSON 含 `error`。实现可统一让 malformed JSON 也返回 JSON error，但不能把它误记成 scorer 的额外硬条件。随后必须通过 8-token 合法非流式请求（HTTP 200、`type=result`、一个 `output_ids`、usage 为 8+1），再检查 `/health.status=ok` 和合同 identity；任何负控污染、恢复失败或 identity 不符都使 `protocol_pass=false`。容量不足必须在 admission 阶段拒绝，不能先触发不可恢复 CUDA OOM。

协议清单是 eligibility gate，不是 PR review 的可选加分项。协议 owner 的原始证据必须逐项包含：`malformed_json`（原始不可解析 body，HTTP 400）、`empty_input_ids`、`negative_token_id`、`out_of_vocabulary_token_id`、`unsupported_temperature`、`over_budget`、`unsupported_modality_field`（后六项均为 `stream=false`、HTTP 400 + JSON `error`）、`valid_short_nostream_request`（8-token `stream=false` result）、`health_after_invalid_requests`，以及 `health_contract_identity`。这样 `/health.max_model_len` 同时被当作真实能力声明和 `prompt + output` 总预算上限。

服务结构采用一个 GPU runtime owner + bounded request channel；HTTP task 不直接共享可变模型。每个请求拥有独立 state handle、request id、取消标志和回收 guard。所有 CUDA error 转成请求级错误后执行 stream/context 健康探针；如果 CUDA context 已损坏，health 不得继续虚报正常。

### 4.4 共用显存与可靠性门禁

- 启动时生成逐类显存预算：resident weights、CUDA context/library、static workspace、per-request GDN state、每 KV page、vision、MTP、cache reserve 和 5%–8% safety margin。
- 模型加载和一次 warmup 后以实测 `cudaMemGetInfo` 校准预算；admission 使用校准值而不是理论值。
- 请求完成、取消、非法输入、capacity rejection 和 kernel error 都必须触发 RAII state/page 回收。
- 每个正式 campaign 前后检查 `nvidia-smi` 的 Xid、显存泄漏、温度、频率和功耗；基础 cell 内不允许 OOM/NaN/fallback/Xid。
- 连续运行至少 200 个混合长度请求，成功率目标 100%，正式最低门槛≥99%。
- TTFT/TPOT 每 cell warmup 1、measure 5，CV≤10%；若 CV 超限，该候选无论中位数多快都不接收。
- evaluator 单请求默认 timeout 是 1800 s，因此逐 token eager GDN 可以作为 eligibility fallback；但必须记录单请求耗时、launch 数和整轮 campaign 预计时长。1800 s 不是性能目标，也不能掩盖最终评测无法按截止时间完成的问题。

### 4.5 三个 bonus 的共同完成标准

**长上下文**：先验证 32640/32768；32640 是 non-scoring diagnostic，32768 按公式仍为 0 分，65536 才是首个正分台阶（约 3.33 分），131072 约 6.67 分。只有 embed 分阶段 residency、INT8 cold-page 和显存账本都有实测余量时才测更长。每个候选长度都验证 early/middle/late retrieval、multi-hop、revision、aggregate，输出完整 128 token，并在失败后立即做 health + 8-token recovery。196608/262016 与 INT4 KV 是独立研究实验，不是本期默认声明。

**多请求**：在单请求基线冻结后验证 C4 再验证 C8。每 cell 为 32 requests × 1024 prompt × 128 output；成功率和正确率 100%、Jain≥0.95、p95 TTFT≤1.5×、p95 TPOT≤3×、结束后健康。调度必须计入客户端排队时间，不能用服务端时间替代。

**多模态**：合同指定 `Qwen3VLProcessor` + `Qwen2VLImageProcessorFast`，输入为 448x448 RGB PNG；不要自行发明 processor。`deepstack_visual_indexes=[]` 时不注入 deepstack。`POST /v1/chat/completions` 接受一个 base64 PNG part 后接一个 text part，temperature 0、`max_completion_tokens=32`、`stream=false`、`enable_thinking=false`。`BASE_GOOD` 后即可在独立 GPU3 分支并行准备和对拍，服务器 replay 仍服从全局队列；只有 public 4/4、hidden 8/8、全部请求成功、health 正常、无外部 fallback 后才打开 capability；否则 `multimodal=false` 并以 HTTP 400/415/422/501 和 `error.type=unsupported_capability` fail closed。

### 4.6 共用 feature flags 与回滚

至少保留以下开关；默认配置先关闭所有未验收的优化，尤其是 prefix cache 和 MTP：

```text
APXINF_Q35_W4_KERNEL={reference,mature,sm89}
APXINF_Q35_GDN_KERNEL={eager,chunk,fused_sm89}
APXINF_Q35_CUDA_GRAPH={0,1}
APXINF_Q35_KV_MODE={bf16,paged,int8,mixed}
APXINF_Q35_PARALLEL={1,4,8}
APXINF_Q35_PREFIX_CACHE={0,1}
APXINF_Q35_MTP={0,1}
APXINF_Q35_MULTIMODAL={0,1}
```

每个优化项只有在 feature-off 与 feature-on 的 correctness、可靠性、性能、显存和 CV 对照完整后才能合入主配置。回滚不修改评测合同，也不伪造 health；关闭某项后同步降低相应 capability 声明。

---

## 5. 方案一：成熟算子混合主线

### 5.1 路线定义

该方案的核心是最大限度利用已验证的 CUTLASS、cuBLASLt、FA2、CUDA Graph 和现有 ApxInf kernel，通过离线权重 manifest/重排和少量 Qwen3.8 专用 kernel 获得高性能。新增 kernel 只解决成熟库不能直接表达或明显低效的部分：compressed-tensors W4A16 asymmetric group-32、GDN recurrent update、必要的 fused epilogue 和 paged KV。

它不是保守低分方案。它保留基础 100、context 10、C4/C8 10、multimodal 10 和 PR review 20 的上限，但每个 bonus 独立验收、独立报告，不把“全部 bonus 均通过”当作主线硬门；优势是每个组件都有独立 reference 与替代实现，出错时能局部回滚。

### 5.2 架构与关键技术

**W4A16 与 BF16 混合路径**

- loader 保留 checkpoint packed int4、BF16 scale 和按 4-bit 打包在 I32 中的 zero-point，离线一次性重排为 CUTLASS/CUDA kernel 友好布局，不产生完整 BF16 权重副本。
- loader 按实际 tensor dtype 建 dispatch manifest：W4 projection 走 weight-only kernel；GDN 的 `in_proj_a/in_proj_b`、conv/norm 与第 0 层 `out_proj` 走 BF16 路径，其余 GDN `out_proj` 是 packed W4；embedding、lm_head、MTP 和 vision 走 BF16 路径，禁止错误套用 W4 dequant。
- prefill 的大 M 第一候选是按 chunk 将 packed W4 解到 BF16 scratch，再交给 cuBLASLt/CUTLASS BF16 GEMM；chunk 默认不超过 2048，dequant 与 GEMM 可双缓冲重叠。只有 shape-specific MFU/roofline gap、dequant+scratch 的实测占比和客户端 TTFT paired A/B 都支持该归因时，才研究 fused dequant tile；`165.2 TFLOPS` 与约 `1%` 只是量级参考。
- decode 的 M=1/小 batch 使用直读 packed-W4 GEMV 作为第一候选，再与 cuBLASLt small-M 做有限 shape sweep；把通过的 tactic 写入 `configs/qwen35/sm89_mature_tactics.json`，不把离线 autotune 当成前置条件。
- QKV 与 gate/up 在权重层面拼接或建立连续 view，减少 launch 和重复读 activation；输出端用成熟 epilogue 完成 bias/scale，SiLU×up 和残差 norm 使用现有或小型 custom kernel。

**GDN 与 attention**

- GDN prefill 先实现逐 token eager/single-step reference-compatible 路径，作为 public/hidden correctness fallback；随后采用 chunk scan：projection 用成熟 GEMM，causal conv + recurrent delta update 用专用 CUDA kernel。chunk 边界 state checksum 必须与 eager 对齐，chunk-scan 失败时回退 eager。
- GDN decode 用常驻 state 的单 token kernel，融合 conv ring-buffer update、gate、delta update 和输出规整，避免 48 层中大量小 kernel。
- 16 个 full-attention 层复用 FA2 prefill；decode 复用现有 flash attention/GQA kernel，补齐 QK norm、Qwen3.8 RoPE 和 KV page append。
- decode graph 按 batch 1/4/8 和 position/KV page bucket 预捕获，图外只更新 token、position、request slot 和 page-table 指针。

**内存、bonus 与视觉**

- 先实现 BF16 paged KV 到 32K，再引入 per-head/per-page INT8 KV + BF16 scale。每个 page 量化前后做 attention output 与 token trajectory 对照；先争取 65K，再争取 131K。262016/INT4 KV 只保留为有余量时的独立实验。
- continuous batching 使用 decode-first round robin，prefill 分 1K/2K/4K chunk，避免长 prefill 阻塞 C4/C8 首 token。
- 多模态常驻 vision 权重会挤占上下文/并发空间，因此使用服务内受控的 GPU weight residency profile：图片请求到来时加载真实 vision 权重并释放可回收 text cache page；不允许 CPU 推理。若加载延迟只影响非计时图片 bonus，可以优先保证 correctness 和峰值显存。

### 5.3 分阶段执行计划

#### 阶段 M1：reference、loader 和协议

- 建立 qwen35 文件夹、config/weight manifest 和 revision hard gate。
- 写 packed W4 dequant reference、GDN 逐 token reference、full-attention reference。
- 先实现 `/health`、严格 generate SSE/JSON、请求级 `max_new_tokens`/`ignore_eos`、两个 EOS ID、错误 schema、admission 和 stub 恢复；随后提供稳定 runtime adapter/worker 接口并完成真实模型接入（均由程仁龙完成）。
- 门禁：loader tests 全部通过；7 项 protocol gate（malformed HTTP 400；其余 6 项 HTTP 400 + JSON `error`；8-token result；health recovery/identity）全部通过；公开 functional 6/6；显式覆盖 `64 + ignore_eos=false` 与 `128 + ignore_eos=true`；重复 reset token 一致。

#### 阶段 M2：成熟 W4A16 与基础模型

- 添加 cuBLASLt/BF16-scratch、成熟 W4 和窄 decode GEMV 候选；只对真实 shape 做小规模配对 benchmark。
- 接入 64 层混合 forward，GDN 先逐 token eager correctness 路径，后静态 workspace/chunk-scan。
- 门禁：公开功能 6/6；合成 hidden 代理集目标 12/12、最低 11/12；完整 trajectory 只作为 correctness 分值和诊断记录；无全量 BF16 权重；16K + 128 不 OOM。

#### 阶段 M3：TTFT/TPOT 优化

- FA2 prefill、chunked GDN、QKV/gate-up packing、GPU argmax、decode graph。
- 用 Nsight Systems/Compute 找 top 5 kernel 与 launch gap，逐项优化；只接受端到端改善。
- 门禁：7 个基础性能 cell success=100%，warmup 1 + repeat 5，TTFT/TPOT CV≤10%；服务端和客户端测量一致性只作诊断。

#### 阶段 M4：按收益/人时打开 bonus

- BF16 paged KV → mixed BF16/INT8 KV，逐级验证 65K，再视余量争取 131K；不把 262016 作为默认交付目标。
- continuous batching 先 C4 后 C8，完成公平性、尾延迟和 goodput gate。
- Qwen3.8 vision 适配，public 4/4 后才申请 hidden 验证；12/12 后开启 capability。
- 门禁：已声明最高 context 6/6 + 128 tokens + recovery；C4/C8 各 32/32；multimodal public 4/4、hidden 8/8。任何未通过项不阻塞 eligible base。

#### 阶段 M5：证据和冻结

- 在 clean checkout 重放 build、check、run、context、multi、multimodal。
- 报告 accepted/rejected 实验、性能中位数/CV、VRAM、kernel profile 和每项回滚开关。
- 门禁：文本 eligibility、已启用基础 cell、PR review 四部分均有机器证据或审查证据；每个 bonus 独立标记 pass/unsupported/0，不要求未启用 bonus 阻塞冻结；无手填 submission 汇总。

### 5.4 实验矩阵

| 实验 | 对照 | 接收条件 |
|---|---|---|
| M-W4-P | cuBLASLt/fused dequant/CUTLASS prefill | 所有真实 shape 数值通过；1K–16K TTFT 至少四个 cell 改善且无显存峰值回退 |
| M-W4-D | custom GEMV vs CUTLASS small-M vs cuBLASLt | 1K、8K TPOT 中位数改善；CV≤10%；trajectory 分值不降，若改变则记录 edit loss |
| M-GDN | 逐 token eager vs chunk scan vs fused decode | chunk 边界 state checksum 一致；功能题/eligibility 不降，trajectory 分值单独记录 |
| M-FA2 | existing attention vs FA2 bucket | 8K/16K TTFT 改善；长 position 正确；无 workspace 泄漏 |
| M-GRAPH | eager vs graph B1/B4/B8 | replay 输出逐 token 一致；TPOT/C4/C8 有端到端收益；capture 失败可回滚 |
| M-KV | BF16 vs INT8 cold page vs mixed window | 已声明长度六类 6/6；基础功能/eligibility 不降；trajectory 分值与显存变化均有记录 |
| M-MULTI | single-flight vs chunked continuous batching | C4/C8 validity 全部满足；goodput 改善；单请求配置不受影响 |
| M-VISION | text-only vs vision residency profiles | public 4/4、hidden 8/8；全请求成功；文本健康和显存恢复 |

### 5.5 风险、成功率与回滚

- **主要风险**：成熟库对 asymmetric W4A16 和 M=1 的支持不够理想，TPOT 可能落后全定制路线；INT8 KV 可能改变长上下文 token；vision residency 可能造成碎片。
- **预防**：每个 GEMM shape 保留三候选；KV 采用近端 BF16/冷端 INT8；vision 加载前先做显存 admission 和 cache eviction。
- **回滚顺序**：单一 tactic → 另一成熟 tactic → 关闭该融合 → BF16 KV/降低已声明 context → C8 回 C4/1 → multimodal=false。每次回滚后重跑 correctness、99% success、CV≤10% 和负控制。
- **判断**：该方案的优势是回滚粒度和成熟算子证据，不对尚未实现的基础/bonus 成功率给百分比承诺。当前最主要的资格风险是 GDN 语义与 W4 layout；chunk-scan 是 TTFT 性能风险，单卡显存限制 context/C4/C8，而不是继续堆叠更激进的 kernel。

### 5.6 该方案达到最优的判据

只有当成熟算子路线在基础 cell、可靠性和已声明 capability 上都有新鲜证据，且继续定制 kernel 的预估收益小于测试波动或引入更高 correctness 风险时，才认为当前主线足够好。任何 isolated kernel 加速但端到端无收益的修改都记录为 rejected experiment，不合入。

---

## 6. 方案二：SM89 窄特化 decode 路径

### 6.1 路线定义

该方案不是从零另建一套全模型 backend，而是挂在方案一的正确 vertical slice 之后的 SM89 窄特化优化路径。它只针对 M=1 packed-W4 GEMV、已通过层级对齐的 GDN decode 融合和 CUDA Graph 做 A/B；prefill 继续以 BF16 scratch + cuBLASLt 为基线。所有 specialization 只依赖合法 shape、batch、dtype、硬件和运行状态，不按 case ID 或已知 token 特判。

该路线追求在不破坏 eligibility 的前提下提高 TPOT 和 C4/C8 goodput；它不是中期或最终提交的前置条件。任何候选都必须能一键回到 mature anchor。

### 6.2 SM89 内核设计

**W4A16 Tensor Core microkernel 与 BF16 companion path**

- 离线将 int4 权重重排为 SM89 `mma.sync`/CUTLASS cute 可高效读取的 interleaved layout；scale/zero-point 按 group 32 与 tile 共载。
- 对 checkpoint 中保持 BF16 的 GDN `in_proj_a/in_proj_b` 与第 0 层 `out_proj`、embedding、lm_head、MTP 和 vision 建独立 companion kernel/tactic；其余 GDN `out_proj` 是 packed W4，不能为了统一内核私自改 dtype。
- prefill 不建立专属 kernel 族作为前置条件；只 profile 真实 cuBLASLt baseline。只有 shape-specific MFU/roofline gap、dequant+scratch 的实测占比和客户端 TTFT paired A/B 都支持该归因时，才重开专用 kernel；`165.2 TFLOPS` 与约 `1%` 只是量级参考，“TTFT 是瓶颈”本身不构成条件。
- decode 内核针对 M=1、B=4、B=8 使用 persistent CTA/stream-K；权重只读一次后为多个 request/token row 服务。融合 dequant、MMA、bias/scale 和部分 epilogue。
- lm_head + top-1 采用分块 GEMV、block top-k 和 device reduction，避免完整 logits D2H。
- `lm_head` INT8 只作为可回滚实验：先记录 top-1/top-2 margin，再同时检查 public/hidden exact、trajectory edit loss 和客户端 TPOT；任一 correctness 或端到端收益不成立就保留 BF16。

**层级融合**

- full-attention prefill：RMSNorm → packed QKV W4A16 → QK norm/RoPE → KV quantize/write → flash attention，尽量通过同一 stream 的少量 kernel 完成；不在通用 backend 中编码层结构。
- full-attention decode：persistent QKV + QK norm/RoPE/KV append；paged flash decode；O projection + residual/RMSNorm epilogue。
- GDN prefill：先以 eager 路径作为 correctness anchor，再使用 projection 后的 blockwise associative scan；chunk 间 state 只写一次并与 eager checksum 对拍。GDN decode 将 causal conv ring buffer、decay/gate、delta state update、output norm 组合为每层一个或两个 kernel。
- MLP：packed gate/up W4A16 + fused SwiGLU；down projection epilogue 直接写 residual，并生成下一层 norm 输入。

**persistent decode 与 CUDA Graph**

- 首选“层级 persistent kernel + graph”而不是单个超大 persistent model kernel，控制寄存器和调试范围。
- B1/B4/B8 各建 graph executable；page table、request slot、position、token 和 active mask 放 device control block。
- 通过 occupancy、register spill、L2 hit rate、DRAM bytes/token 和 launch gap 定量选择融合边界。若单 kernel 寄存器溢出或占用率低于阈值，拆回两个 kernel。

### 6.3 bonus 技术

**context 实验**

- 复用共同 lane 的 paged KV；先测 65K，再视显存余量和 correctness 争取 131K。INT4 KV/262016 不属于本 lane 的默认交付目标。

**C4/C8**

- B4/B8 persistent W4 内核在一次 weight sweep 中处理多个请求，理论上比方案一的常规 continuous batching 更能提高 correct goodput。
- prefill 使用 token-budget microbatch，decode 采用 age-aware round robin；超过 p95 TTFT 预算时暂停新 prefill chunk。
- 每次 kernel 接受 active mask 和 request state pointer，单请求失败只标记对应 slot，不污染其他 state。

**多模态**

- 为 448×448 评测图片建立专用 vision shape buckets，但仍正确处理合同允许的图片内容，不对类别或答案特判。
- 复用现有 vision kernels，优先定制最耗时的 patch embed、vision SDPA、merger 和 text embedding scatter；图片无性能排名，先保证 12/12 exact correctness 和可靠性，再优化显存切换。

### 6.4 分阶段执行计划

#### 阶段 S1：性能模型与 reference anchor

- 完成共同底座和 mature/eager reference，测出每层 launch、DRAM bytes、tensor pipe、L2、occupancy。
- 为每个真实 GEMM shape 建 roofline，确定 memory-bound 或 compute-bound，禁止凭直觉融合。
- 门禁：公开 6/6、hidden 代理集至少 11/12、16K 基线可运行；profile 能解释当前瓶颈；trajectory 作为软分诊断，不是硬门。

#### 阶段 S2：SM89 W4A16 内核族

- 先做 decode B1，再视端到端收益扩 B4/B8；只做有限候选配对，不把 autotuner 或 persistent mega-kernel 当作前置条件。
- 对随机输入、真实权重切片和完整层做 reference 比较；执行 deterministic 重跑。
- 门禁：所有候选 shape correctness；无非法内存、race、spill 异常；B1 TPOT 相对 mature anchor 有可重复端到端改善，否则保留 mature anchor。

#### 阶段 S3：GDN/attention/MLP 融合

- 依次优化 GDN decode、full-attention decode、MLP epilogue、GDN prefill scan、full-attention prefill。
- 每次只合并一条数据流，保留前一阶段 graph；记录融合前后 DRAM 与 launch 变化。
- 门禁：功能题和 state checksum 通过；基础 7 cell success=100%、CV≤10%；每次融合至少改善一个目标 cell且不显著损害其他高权重 cell；trajectory 变化按实际 correctness 分值计入决策，不把逐位一致当作唯一资格门。

#### 阶段 S4：persistent batching 和 KV 压缩

- B4/B8 graph + persistent GEMM，完成 C4/C8 validity。
- KV 格式沿共同 lane 从 BF16→INT8 逐级验证，目标是提高已声明的 65K/131K 能力，而非承诺 262016。
- 门禁：C4/C8 32/32、Jain≥0.95、p95 约束；已声明最高 context 六类 6/6、完整 128、失败恢复。

#### 阶段 S5：vision 与总集成

- 接入 Qwen3.8 vision，完成 public/hidden 12/12；验证启用 vision 后基础 text cell 不因显存或 fragment 失效。
- 全量 burn-in、clean checkout、正式 5-repeat campaign 和报告冻结。
- 门禁：文本 eligibility、已启用基础 cell、99% success、无 OOM/NaN/fallback/Xid、所有已测 cell CV≤10%、PR review 证据齐全；context/C4/C8/vision 各自独立标记，不把未启用 bonus 作为该 lane 的硬门。

### 6.5 实验矩阵

| 实验 | 主要变量 | 必记指标 | 接收条件 |
|---|---|---|---|
| S-TILE-D | CTA tile、warps、stages、split-K | TPOT、GB/s、TOPS、occupancy、spill | B1 TPOT 中位数优于 mature anchor；功能题/eligibility 不降，trajectory edit loss 可解释 |
| S-TILE-P | M chunk、K/N tile、pipeline stage | 1K–16K TTFT、tensor pipe、DRAM | 加权 TTFT 总收益为正；CV≤10% |
| S-FUSE-GDN | conv/update/norm 融合边界 | kernels/layer、bytes/token、registers | hidden/state checksum 通过；无 occupancy 灾难 |
| S-FUSE-ATTN | QKV/QKNorm/RoPE/KV/attention 边界 | 8K/16K TTFT、1K/8K TPOT | 所有长 position 正确；无 page race |
| S-PERSIST | B1/B4/B8 persistent vs graph-only | goodput、p95、Jain、L2 hit | C4/C8 所有 validity 满足且 goodput 更高 |
| S-KV | BF16/INT8/候选低位宽 | context pass、VRAM、attention error | 只采用最高 6/6、128-token、recovery 全过格式 |
| S-VISION | existing vs specialized vision bucket | 12 case exact、VRAM、健康 | correctness/reliability 全过；无 text regression |

### 6.6 风险、成功率与回滚

- **主要风险**：W4 asymmetric dequant 次序或 packing 错误；过度融合导致寄存器 spill/L2 下降；block scan 的 GDN state 边界错误；graph pointer 更新或 active mask 造成跨请求污染。
- **预防**：mature anchor 永久保留；每个 kernel 做 Compute Sanitizer/racecheck、小 shape exhaustive 和真实 shape differential；融合以 Nsight 数据决定；每个 request state 加 canary/checksum。
- **回滚顺序**：单 kernel tile → 上一个 kernel revision → mature W4/GDN/FA2 → graph-only → BF16/mixed KV → 降低对应 capability。回滚后必须重验 correctness、负控制、99% success、CV≤10%。
- **判断**：该 lane 的性能上限可能最高，但其收益和成功率必须由 GPU2 的 paired client measurements 决定；在 B1 vertical slice 通过前不做百分比承诺，也不让它阻塞主线或中期提交。

### 6.7 停止继续融合的判据

出现以下任一情况就停止扩大融合边界并保留较小 kernel：端到端收益低于 2% 或落在五次测量噪声内；register spill 增加且 occupancy/L2 无补偿；功能题/eligibility 失败或 trajectory edit loss 带来的 correctness 扣分超过性能收益；CV 超过 10%；C4/C8 p95 或 Jain 失效；代码复杂度无法用独立 test 和 feature flag 隔离。方案二的“全定制”不等于盲目把整个模型塞入单 kernel。

---

## 7. 方案三：状态、内存与调度路径

### 7.1 路线定义

该方案是主线正确后才打开的状态/内存优化路径，默认优先级为 paged KV、严格 admission、C4/C8 和条件性 MTP。prefix state cache 不是本期关键路径，默认关闭；只有主线提前冻结、课程规则允许、且 cold/hit A/B 证明合规和净收益时才单独实验。所有收益都必须来自真实模型状态复用或 target exact verification，并具备一键关闭和审计证据。

缓存不能保存答案或输出 token；MTP 不能跳过 target model 校验。任何 cache/MTP 失败都不能影响 `BASE_GOOD` 的文字资格。

### 7.2 可选 prefix state cache（默认跳过）

**合法 cache key**至少包含：

```text
implementation commit / binary hash
model repo + revision + complete weight manifest hash
kernel/tactic version + quantization/KV format
tokenizer/config/chat-template identity
runtime capability profile
exact input_ids bytes and length
multimodal media SHA256 + processor config + grid metadata
```

**可缓存内容**只包括由真实模型计算得到的状态：

- 16 层 full-attention 的 KV pages；
- 48 层 GDN recurrent matrix/state 和 conv ring buffer；
- prefix 末尾 hidden/position；
- 图片请求的真实 vision embeddings/merged prefix state；
- page table、dtype/scale 和 state checksum。

禁止缓存功能 case 的答案、输出 token、case ID 映射、validator 结果或按公开数据预热答案。默认提交启动时 cache 为空；若进行通用 cache warmup，必须对任意 prompt 使用同一策略并在报告中披露。评测政策若不接受跨请求 prefix state reuse，使用 `APXINF_Q35_PREFIX_CACHE=0` 完全关闭且不影响基础正确性。

**一致性和回收**：cache hit 先验证完整 identity 和 state checksum；只允许 exact prefix hit，不能近似匹配。采用 refcount + LRU + byte budget；请求 clone page table 后 copy-on-write。cache-off、cold miss、warm hit 三者必须生成完全一致的 token trajectory。

### 7.3 MTP exact-verify speculative decoding

- 从 checkpoint 的 `mtp.*` tensor 构建 proposal model；加载时检查 tensor 清单、shape 和与目标 hidden 的接口。
- MTP 一次提出 K 个候选 token；目标 64 层模型在一次 batched verification forward 中计算这些位置。
- 逐位置比较 target greedy argmax：相同则接受；首个不同时接受 target token，并撤销其后的 draft token。
- full-attention KV page、GDN state、conv ring、position 和 RNG-free greedy control 都使用 checkpoint/transaction。拒绝时回滚到最后接受位置，不能保留 speculative state 污染。
- 自适应 K 只依据最近接受率、可用显存和请求长度，不依据 case ID、token 内容白名单或已知答案。
- 若平均接受率不足以抵消 proposal + verify 成本，自动切回普通 decode；这个切换必须对输出无影响。

MTP 优化的是 TPOT 和 C4/C8 goodput，不保证必然提速。必须记录 acceptance rate、verified tokens/target pass、rollback 次数、额外 VRAM 和端到端 TPOT，而不能只报告 draft 速度。

### 7.4 内存与调度协同

**统一状态页**：为每个 request 建 `StateHandle`，引用 GDN state、KV hot/cold pages、prefix cache parent、MTP transaction 和 vision state。page allocator 按功能区保留预算，避免某一 cache 吃光 C8 或 vision 显存。

**KV 分层**：recent window 使用 BF16/INT8 hot page，冷页用经过验证的压缩格式；attention on-the-fly dequant。prefix cache 若被单独批准，可和 active request 共用物理只读页并通过 refcount 避免重复占用；默认 text profile 不依赖 cache。目标先是 65K，再视实测余量争取 131K。

**continuous batching**：调度器按最早 deadline/age 选择 decode，利用 MTP verification 的多 token batch；prefill 采用 chunk。每轮约束 cache admission、MTP K 和 active slots，使 C4/C8 p95 与 Jain 门槛优先于理论吞吐。

**vision 协同**：相同图片和 processor identity 可缓存真实 vision embedding，但不能缓存答案。不同 text prompt 可复用 image prefix embedding/state；媒体 hash 不同必定 miss。vision 权重、MTP 权重和长上下文 cache 根据能力 profile 做预算切换，任何切换都先 admission 后执行。

### 7.5 分阶段执行计划

#### 阶段 C1：无缓存、无 MTP 的强基础执行器

- 完成共同底座，采用成熟 W4A16、GDN、FA2、graph、paged KV。
- 建立 cold-path baseline，所有后续系统优化都以此为 correctness anchor。
- 门禁：公开功能 6/6、hidden 代理集至少 11/12；完整 trajectory 记录为软分诊断；基础 7 cell success=100%、CV≤10%；99% success 和负控制通过。

#### 阶段 C2：可选 cache 审计（默认跳过）

- 只有在 `BASE_GOOD` 已冻结且有独立 cold/hit 合规问题时，才实现 identity key、exact prefix lookup、state serialization/view、copy-on-write、checksum、LRU 和 byte cap。
- 增加 cold miss、warm hit、collision、不同 revision/config/media、并发同 prefix 和 eviction tests。
- 门禁：cache off/miss/hit token trajectory 完全一致；hit/miss/state checksum 可审计；capacity 和 eviction 后服务健康。

#### 阶段 C3：MTP exact verification

- 先实现 K=1 的 proposal/verify/rollback，再扩 K=2/4/8；对每个中间接受长度构造测试。
- 将 target batched verify 与 GDN/KV transaction 接入 graph；低接受率自动禁用。
- 门禁：MTP off/on trajectory 完全一致；强制首位/中位/末位 reject 的 rollback tests 通过；1K/8K TPOT 或 C4/C8 goodput有端到端净收益。

#### 阶段 C4：分级 KV、C4/C8 和条件性 MTP

- 逐级引入 BF16 paged KV，再测 INT8 cold page；建立 per-feature memory quota。cache 默认关闭，不作为长上下文前置条件。
- continuous batching 联合选择 active requests、MTP K、prefill chunk 和 cache eviction。
- 门禁：已声明最高 context 六类 6/6 + 128 + recovery；先争取 65K、再争取 131K；C4/C8 各 32/32、Jain≥0.95、p95 约束、无 fallback。

#### 阶段 C5：多模态和合规审计

- 接入真实 Qwen3.8 vision；验证 media hash、processor identity、embedding cache 和不同 prompt 复用。
- 完成 public 4/4、hidden 8/8；生成 cache-off 对照和 fail-closed 证据。
- 门禁：文本 eligibility 与已启用 bonus 各自通过；所有 cache/MTP 日志证明未缓存答案、target exact verify 无旁路；PR review 证据齐全。未启用的 bonus 记录为 0/unsupported，不阻塞文本冻结。

### 7.6 实验矩阵

| 实验 | 对照 | 必记信息 | 接收条件 |
|---|---|---|---|
| C-CACHE-COLD | cache off vs empty-cache miss | TTFT、TPOT、VRAM、tokens | miss 开销可接受；输出完全一致 |
| C-CACHE-HIT | cold miss vs exact hit | TTFT、state bytes、checksum | 通用 exact hit 有净收益；无 token 差异 |
| C-CACHE-NEG | token/config/revision/media 单字段变化 | key、hit/miss、checksum | 全部正确 miss；无碰撞复用 |
| C-MTP-K | K=1/2/4/8 vs off | acceptance、verify passes、rollback、TPOT | target exact trajectory 一致且端到端更快 |
| C-MTP-REJECT | 强制不同 reject 位置 | KV/GDN/conv/position checksum | rollback 后与非 speculative 路径一致 |
| C-KV | BF16/INT8/mixed cold pages | context pass、VRAM、tokens | 最高长度 6/6、128、recovery 全过 |
| C-SCHED | graph batch vs MTP-aware scheduler | goodput、p95、Jain、queue | C4/C8 全部 validity 满足，基础单请求不回退 |
| C-VISION-CACHE | media cache off/miss/hit | image hash、embedding checksum、answers | 12/12；只复用真实 state；不同图片必 miss |

### 7.7 合规审计要求

REPORT 和机器日志必须展示：

- cache key schema、binary/model/config/token/media identity；
- cache 初始状态、hit/miss、保存的 state 类型和 byte size；
- cache-off 与 cache-on 的完整 token 对照；
- 不存在 case ID、expected answer、公开输出 token 或 validator 字段；
- MTP 每步 proposal、target argmax、accepted prefix length 和 rollback checksum；
- 可用一个启动参数完全关闭 cache/MTP；关闭后仍满足 correctness、99% success、CV≤10% 和 bonus 中与其无关的能力。

### 7.8 风险、成功率与回滚

- **主要风险**：评测策略不接受跨请求 prefix cache；identity 不完整导致 stale state；MTP 接受率低于盈亏点；rollback 漏掉 GDN/conv/KV 任一状态；cache、C8、长上下文、vision 和 MTP 争抢显存。
- **预防**：cache 默认可关闭且不影响功能；完整 identity + checksum；MTP 强制 reject 测试；统一 state transaction；各功能硬配额和 safety margin。
- **回滚顺序**：关闭 warm cache → 关闭全部 prefix cache → 降低 MTP K → 关闭 MTP → 降 KV 压缩/已声明 context → C8 回 C4/1 → multimodal=false。每步重跑 correctness、负控制、99% success、CV≤10%。
- **判断**：MTP/paged-KV/C4 的收益必须用实际接受率、显存和客户端 goodput 证明；cache 没有合法重复 prefix 或 cold-path 对照时不贡献可确认分数。当前不对该 lane 给百分比成功率承诺。

### 7.9 该方案达到最优的判据

当 paged-KV、C4/C8 和（若启用）MTP 在配对测试中有净收益、最高已声明 context 通过六类任务与 recovery、且进一步增加状态复杂度会降低净得分或可靠性时停止优化。prefix cache 若无法证明 cold/hit 合规和净收益，永久保持关闭；任何无法清楚证明通用性和合规性的 cache 技巧一律不采用。

---

## 8. 三类 lane 横向比较与开启规则

| 维度 | 方案一：成熟算子混合主线 | 方案二：SM89 窄特化 decode lane | 方案三：状态/内存/调度 lane |
|---|---|---|---|
| 基础 100 分目标 | 主线硬目标 | 依赖方案一 anchor | 依赖方案一 anchor |
| bonus 30 分目标 | 逐项开启 | 仅做已证明局部项 | 仅做已证明局部项 |
| PR review 20 分 | 以成熟组件 A/B 证据取胜 | 以 profile、内核推导和回归取胜 | 以状态机、合规审计和系统实验取胜 |
| Correctness 风险 | 最低 | 最高 | 中高 |
| 单请求 TTFT 上限 | 高 | 最高 | 高；cache hit 时最高但 cold path取决于基础执行器 |
| 单请求 TPOT 上限 | 高 | 最高 | MTP 接受率高时很高 |
| context 阶梯潜力 | 65K/131K 优先，依赖 mixed KV | 65K/131K 优先，依赖已正确的 paged-KV kernel | 65K/131K 优先，依赖显存配额和调度 |
| C4/C8 goodput | 高 | 很高 | 很高，MTP/cache 命中时最高 |
| 多模态成功率 | 较高 | 中 | 中高，但状态更复杂 |
| 调试工具重点 | differential test、tactics、端到端 | Nsight、sanitizer、roofline、tile sweep | transaction test、checksum、fault injection、queue trace |
| 回滚粒度 | 最细 | 中 | 细，但状态组合多 |
| 维护成本 | 中 | 最高 | 高 |
| 当前可量化的综合成功率 | 尚不可量化 | 尚不可量化 | 尚不可量化 |

### 8.1 开启规则

- 方案一始终是唯一主线：程仁龙在 GPU0 维护 `BASE_GOOD`、模型/runtime eligibility 和可回滚 forward，protocol surface 与 gate evidence 独立维护，adapter 合同稳定后才集成。
- 方案二只有在方案一的 eager/chunk vertical slice 已通过层级 correctness 后，才在 GPU2 做 B1 decode GEMV/graph A/B；任何 isolated kernel 结果都不能替代客户端端到端证据。
- 方案三只有在文本 eligible、显存账本稳定后，才在 GPU3 先做 paged KV -> C4；target decoder 冻结后由 GPU2 先做 MTP K=1 probe，C8 与 vision 在隔离 lane 按各自门禁推进；prefix cache 默认关闭。
- 最终组合可以吸收方案二的已证明 decode kernel 和方案三的已证明 paged KV/C4/MTP 组件，但不切换成另一套全栈架构；未通过能力明确标记为 0/unsupported。

### 8.2 决策门

1. 共同底座无法在 16K + 128 内稳定运行：暂停所有 bonus，先修权重常驻和 workspace；不把未验证的 bonus 写入能力声明。
2. 公开 6/6 但独立隐藏模拟集不满：冻结 correctness 修复，不接受性能候选。
3. 某方案连续两个优化周期都不能改善对应端到端得分代理：回到上一通过版本并换下一杠杆。
4. 65K/131K KV 压缩破坏六类任一 correctness：只声明上一最高完整通过长度；262016/INT4 不因截止时间强行开启。
5. C8 无法满足 p95/Jain：保留 C4 的已验证 support/goodput 潜力，停止 C8 扩展，除非新证据显示调度改动不会危及 base。
6. multimodal 未到 public 4/4 + hidden 8/8：保持 `multimodal=false`，fail closed，继续独立修复，不影响文本资格。

---

## 9. 统一实施、测量与验收流程

### 9.1 建议里程碑

| 里程碑 | 产物 | 必过门禁 |
|---|---|---|
| R0 可复现基线 | 环境记录、合同/model hash、显存预算 | `test.py check`，clean status，固定单 GPU |
| R1 算子正确 | W4/GDN/attention reference 与 CUDA tests | 随机/真实切片/边界 shape 全过 |
| R2 文本正确 | 64 层模型、服务、SSE | public 6/6、hidden 至少 11/12、trajectory 记录、负控制、恢复 |
| R3 基础有效 | 7 个性能 cell | success=100%、warmup 1、repeat 5、CV≤10%、无异常 |
| R4 基础冲榜 | 方案核心优化 | paired A/B、端到端加权收益、correctness 不降 |
| R5 context | 逐级长上下文 | 最高长度 6/6、128 output、失败恢复 |
| R6 multi | C4/C8 | 32/32、correctness 100%、Jain/p95/health 全过 |
| R7 multimodal | Qwen3.8 image path | public 4/4、hidden 8/8、无 fallback、健康 |
| R8 提交冻结 | REPORT、raw artifacts、clean replay | eligibility、已启用能力和 PR review 证据分别可重放；未启用 bonus 明确标记 0/unsupported |

### 9.2 基础分锁定顺序

1. Correctness 30：先锁公开 6/6 和 hidden 至少 11/12 的 eligibility；trajectory 继续记录并按实际分值优化，不把逐位 256/256 当硬门。
2. Reliability 10：先把 success 做到 100%、所有 boolean 为真，再开展长时间性能 campaign。
3. TPOT 25：先优化 B1 decode 的权重带宽、launch 和 state 常驻，因为 128-token decode 放大每 token 成本。
4. TTFT 35：按权重优先 16K(10)、8K(8)、4K(7)、1K/2K(各5)，但必须保证五个 cell 都有效。
5. 每次性能修改都重跑公开 trajectory 和至少一组协议/恢复 smoke；正式候选重跑完整 correctness。

### 9.3 bonus 验证顺序

按“分值 / 人时 / correctness 风险”排序，而不是按理论上限排序。文本 `BASE_GOOD` 冻结后，
多模态 vertical slice 与 MTP K=1 probe 可以在各自分支并行准备；服务器 replay 仍必须按
全局 GPU 队列串行执行。MTP 不是独立计分项，若没有端到端净收益立即关闭。

| 顺序 | 能力 | 直接收益与现实门槛 | 执行规则 |
|---:|---|---|---|
| 1 | BF16 paged KV + admission | 是 65K 和 C4/C8 的共同基础；32768 得 0 分，65536 约 3.33 分 | 先 BF16、再视余量 INT8；每级六类 6/6 + 128 + recovery |
| 2 | C4 validity | 先拿 1 support 分，再争 goodput；复用 page allocator | 32/32、Jain/p95/health 全过后才记录 goodput |
| 3 | 多模态 vertical slice | 10 分纯 correctness、无 latency ranking；合同已有 processor 类 | 文本 `BASE_GOOD` 冻结后由 GPU3 logical lane 准备并排队 replay 448x448 public 4/4；不得改 GPU0 主线，hidden 前重跑文本 smoke |
| 4 | MTP K=1 feasibility probe | 不是独立 bonus，但可能改善 base TPOT 和 C4/C8 goodput；需 exact verify/rollback，接受率可能不足 | target decoder 冻结后先做 K=1；off/on 端到端净收益才接入，否则立即关闭；不阻塞 C4 validity |
| 5 | C8 | 额外 support 分和 goodput，但 p95/Jain 风险高于 C4 | 仅在 C4 稳定后开启；失败保留 C4；与 GPU2 的 MTP probe 可并行准备，但 replay 服从队列 |
| 6 | 131K / INT8 KV | 约 6.67 context 分，但显存和数值风险明显 | 65K 通过后再测；262016/INT4 不作为默认目标 |

多模态可以在顺序上与 1--2 并行开发，但只能在文本基线冻结后接入；服务器验证按队列执行，
它的独立 bonus 不会改变 text eligibility 闸门。

### 9.4 实验纪律

- 一次只改变一个主要变量，使用相同 git revision、服务命令、GPU、时钟策略、数据、warmup 和 repeats 做 paired run。
- 接收候选的硬条件：correctness 不降、success=100%、CV≤10%、无 reliability 失败；性能改善超过噪声或明确提高可得分 capability。
- 记录 accepted、rejected、negative 和 inconclusive，不删除失败结果。
- client-observed TTFT/TPOT/goodput 为准；kernel profile 只解释原因。
- 每次 campaign 保存 run id、完整 commit SHA、contract SHA256、model manifest/revision、public/hidden/context manifest hash、raw JSONL、environment JSON 和 artifact hash。
- 不手工编辑 evaluator 生成的 `submission.json` 或图片报告。

### 9.5 统一命令

```bash
cd /mnt/chuangxin/team2/ApxInf
# Replace with the UUID recorded for the single-card submission lane.
export CUDA_VISIBLE_DEVICES=GPU0_UUID_FROM_NVIDIA_SMI

# 合同和任务包检查
python3 benchmarks/qwen38_4090/evaluation/test.py check

# Rust/CUDA 单元与工作区回归；实现后执行
cargo test --workspace --locked
APXINF_CUDA_ARCH=sm_89 cargo build --release --features cuda --locked

# 准备公开数据，不提交生成物
python3 benchmarks/qwen38_4090/evaluation/test.py prepare \
  --model-dir /mnt/chuangxin/team2/models/Qwen3.8-27B-AWQ-INT4

# 服务命令以最终实现的 CLI 为准，REPORT 必须记录原样命令
./target/release/apxinf serve \
  --model /mnt/chuangxin/team2/models/Qwen3.8-27B-AWQ-INT4 \
  --host 127.0.0.1 \
  --port 8001 \
  --max-model-len 32768 \
  --parallel-requests 1

# 统一公开评测
python3 benchmarks/qwen38_4090/evaluation/test.py run \
  --model-dir /mnt/chuangxin/team2/models/Qwen3.8-27B-AWQ-INT4 \
  --base-url http://127.0.0.1:8001
```

启动示例使用最保守的 `32768/1`；32768 只是能力诊断且 context bonus 为 0 分，不代表长上下文 bonus 已获得。只有真实容量、C4/C8 和恢复门禁通过后，才分别提高 `max-model-len` 或 `parallel-requests`。最终 REPORT 还应列出 context/multi/multimodal 的平台或统一 runner 原样命令、run id 和 hash，不能以自写 benchmark 替代官方结果。

### 9.6 PR review 20 分专项设计

**测试与负控制（8）**

- packed W4 asymmetric、GDN state、KV compression、MTP rollback、cache collision/miss、vision processor 的独立回归。
- 协议 index gap、request id 混淆、非法 schema、temperature、超容量、客户端断开、并发取消、OOM admission、NaN 注入和失败恢复。
- 至少一个真实 rejected optimization 和一个故障注入负控制。

**接口与错误处理（4）**

- 严格 JSON schema、明确 HTTP status 与 `error.type`、SSE terminal/usage、RAII 资源回收、health 真实性。
- multimodal unsupported fail closed；capacity error 不破坏 GPU worker。

**可复现性（4）**

- clean checkout 一条龙命令；锁定依赖、CUDA arch、GPU identity、模型/合同 hash。
- raw artifacts、环境文件、运行日志和 SHA256；脚本自动生成而非手工抄数。

**分析与决策（4）**

- 明确 baseline、瓶颈假设、roofline/DRAM/launch 证据、paired result、取舍和回滚。
- 解释为什么接受或拒绝每项优化，区分 kernel speedup 与端到端得分。

---

## 10. 最终 REPORT 与提交检查表

### 10.1 成绩与能力

- [ ] Correctness 目标为 30/30；公开功能 6/6，隐藏目标 12/12且最低不低于 11/12，trajectory 完整输出并记录 token edit 结果。
- [ ] TTFT 五个 cell、TPOT 两个 cell 均 success=100%，warmup 1、measure 5、CV≤10%。
- [ ] Reliability 10/10：总请求成功率为 100%（≥99% 只是资格门槛），无 unexpected OOM、NaN、fallback、Xid，失败后 health 和小请求成功。
- [ ] Context：先完成已声明最高长度的六类 6/6、128 output 和失败恢复；优先 65K，再视显存和 correctness 争取 131K；若未到满分，如实报告最高完整验证长度，不虚报 262016。
- [ ] C4/C8 目标 10/10：各 32/32、correctness 100%、Jain≥0.95、按合同的 p95 TTFT≤自身单请求 1.5 倍、p95 TPOT≤3 倍、健康有效，并在同轮取得两个 cell 的最佳有效 goodput；提交前还要用未修改的冻结 scorer 复核 eligibility，并在 REPORT 记录其实际判定结果。
- [ ] Multimodal 目标 10/10：public 4/4、hidden 8/8、全请求成功、健康、`multimodal=true`、`fallback_active=false`。

### 10.2 工程和合规

- [ ] clean checkout 能构建、启动和重放公开评测。
- [ ] `/health` 所有 identity/capability 与真实验证一致。
- [ ] 无 vLLM、Transformers、CPU、其他模型/GPU runtime fallback。
- [ ] 未修改 evaluation 合同、生成器或评测器；未手填汇总产物。
- [ ] 无 case ID、公开 token、已知答案、固定 prompt 特判。
- [ ] 若启用 prefix cache，仅存真实 state、key identity 完整且有 cache-off 对照；默认关闭。MTP 必须经 target exact verify，并有 reject rollback 证据。
- [ ] 未提交权重、凭据、机器私有地址、隐藏数据或大型临时 artifact。

### 10.3 证据和审查

- [ ] REPORT 包含完整 commit SHA、合同/model/data/artifact hash、GPU/CUDA/driver、构建和服务命令。
- [ ] baseline 与每个 accepted/rejected 实验都有端到端中位数、CV、VRAM 和解释。
- [ ] 至少一个负控制、一个失败实验、一个回滚演练有原始日志。
- [ ] CUDA/Rust FFI 修改有 shape/dtype/device/lifetime tests；并发状态有 fault injection。
- [ ] PR 说明按 tests 8、interface 4、reproducibility 4、analysis 4 映射证据。

---

## 11. 三轮自查与改进记录

### 第一轮：逐合同覆盖检查

对照 `contract-v1.json`、`multimodal-contract-v1.json`、submission schema 和 README 后，确认共同底座覆盖 text eligibility、协议、正确性、CV≤10%、99% success、负控制、证据和回滚；bonus 逐项独立验收。已特别核对 functional 的 64-token/可 EOS、performance 的 128-token/强制 budget 和 multimodal processor 合同，避免把示例参数误写成全局门禁。

### 第二轮：技术可行性与合规检查

重点复查显存、状态和优化合法性：

- 没有假定 17.93 GiB 文本权重和约 16 GiB 的 262K BF16 KV 可以同时常驻；把 65K/131K 作为现实阶梯，262K/INT4 只保留为独立实验。
- 没有把 checkpoint 的 linear attention 错当普通 full attention；明确 48 GDN + 16 full-attention 的独立状态。
- 没有把 prefix cache 写成答案缓存；key、state、cache-off、checksum 和政策关闭路径均明确。
- 没有让 MTP 直接决定 token；target exact verify 和 GDN/KV/conv rollback 是硬门禁。
- 没有以 shape specialization 为名按 case/token 特判；所有优化仅依赖合法 shape、dtype、batch、硬件和运行状态。
- 多模态 capability、长上下文和并发声明都绑定真实全量测试，失败时必须降级声明。

### 第三轮：可执行性、测量与审查检查

逐案检查文件边界、阶段、实验矩阵、接收条件、风险、成功率和回滚；补充了以下改进：

- 每案都以公开/隐藏 correctness anchor 开始，并要求基础 cell 的 warmup 1、repeat 5、CV≤10%。
- 所有优化都要求 feature-off/on paired run，禁止用 isolated kernel 数字代替客户端端到端结果。
- 方案二增加 register spill/occupancy 停止条件，避免“越融合越好”的错误假设。
- 方案三增加 cache 初始状态、identity、强制 reject 和 state checksum 的审计闭环。
- 三案都加入已声明 context 失败后恢复、C4/C8 p95/Jain、vision/text 显存回归和 clean-checkout 重放。
- 补入 roofline 数字、MLP activation/dequant scratch、FP32 GDN state 和 lm_head 流量账本；把 prefill 专用 kernel 的重开条件改为 MFU/可归因证据，而不是笼统的“TTFT 是瓶颈”。
- 将 32640/32768/65536/131072 的 context 计分台阶、1800 s 单请求 timeout 与 campaign 总时长风险显式分开。
- 将多模态 processor、448x448 RGB PNG、32 token、非流式和 `deepstack_visual_indexes=[]` 写入合同验收，并把多模态移为文本 `BASE_GOOD` 冻结后的并行 vertical slice。
- 删除未经实现和正式评测支持的百分比成功率；所有上限改为待验证的假设，所有未通过能力必须 fail closed。

---

## 12. 最终建议

最终建议不是在三条全栈架构之间重新选边：程仁龙固定维护**方案一：成熟算子混合主线**，以 BF16 scratch + cuBLASLt prefill、packed-W4 decode GEMV、逐 token eager GDN 资格路径、chunk-scan 性能路径、正确的 full-attention gate/partial RoPE 和 runtime 集成为主，并交付严格 protocol surface、malformed 加六项结构化负控/合法请求/恢复证据；GPU2 只吸收方案二已证明的 decode GEMV/graph，GPU3 只吸收方案三已证明的 paged KV/C4/C8/MTP/vision 组件。这样保留三种技术路径的上限，同时把 correctness 风险集中在一个可回滚的 `BASE_GOOD` 上。

最终只提交有新鲜机器证据的能力声明：text eligibility 是硬门；context、C4/C8、multimodal、MTP 和 INT8/lm_head 等 bonus 各自 pass/0/unsupported。没有通过的 lane 不阻塞文本提交，也不能写入 `/health` 的 capability。
