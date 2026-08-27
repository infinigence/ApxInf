# ApxInf 三人协作规格

版本：`v1.1`
冻结日期：`2026-08-23`
维护人：成员1（集成 owner）

## 1. 目标与约束

目标是在固定 `cyankiwi/Qwen3.8-27B-AWQ-INT4`、单张 RTX 4090、SM89、W4A16 group-32 asymmetric 条件下，先交付一个正确、稳定、可复现的文本服务，再按证据逐项打开 context、C4/C8、MTP 和 vision bonus。

本项目不把三人做成三条独立全栈路线，而是一个可回滚的主线加两个隔离实验 lane：

```text
成熟算子混合主线
  + 已证明的 SM89 decode GEMV/Graph
  + 已通过门禁的 paged KV/C4/C8/vision/MTP
```

默认不进入关键路径：全模型 mega-kernel、prefill offline autotune、prefix cache、262K INT4 KV。任何例外都必须有 DECISIONS 记录和成员1批准。

## 2. 不可修改的合同

`benchmarks/qwen38_4090/evaluation/` 是只读合同区。禁止修改：

- `contract-v1.json`、`multimodal-contract-v1.json`、`submission-schema-v1.json`；
- `run_evaluation.py`、`score_submission.py`、`score_multimodal.py` 和公开数据生成器；
- evaluator 生成的 `runs/`、提交汇总结果和 hidden 数据。

必须保留：

- `/health` 的真实合同 identity、model revision、`max_model_len`、`parallel_requests`、`fallback_active` 和 capabilities；
- `POST /v1/evaluations/generate` 的 pre-tokenized integer `input_ids`、greedy `temperature=0`、EOS 语义、SSE/JSON 两种响应；
- `max_model_len` 是 `prompt_tokens + max_new_tokens` 的总 admission 上限，同时还要服从实测 device budget；
- token range 的权威边界是 checkpoint `config.json` 的 `text_config.vocab_size`
  （加载到 runtime model config 后为 `vocab_size`；本模型实测为 `248320`，合法范围
  `[0, 248320)`），不是 tokenizer 的 `vocab_size`（实测为
  `248044`）；`image_token_id=248056` 因此仍是合法 model token，vision 路径不能用
  tokenizer 词表值做拒绝边界。
- 六项可解析的结构化 protocol probe 明确使用 `stream=false`：空 `input_ids`、负 token、`4294967295` 越界 token、`temperature=0.1`、`max_new_tokens=health.max_model_len` over-budget、`images:["x"]`；malformed JSON 是原始不可解析 body，只检查 HTTP 400；
- malformed JSON 至少 HTTP 400；其余六项必须 HTTP 400 且 JSON 有 `error` 字段；随后 8-token 合法非流式请求必须 HTTP 200、`type:"result"`、一个 output token 和正确 usage；
- 五项 reliability boolean 都是 eligibility gate：`no_unexpected_oom`、`no_nan`、`no_fallback`、`no_xid`、`service_healthy_after_failure`。任一失败，`eligible=false`，不是简单扣分。

## 3. 单一集成架构

### 3.1 运行时边界

模型结构和 request state 放在 `apxinf-model`；单 kernel API 和 CUDA 资源放在 `apxinf-cuda`；loader 负责 checkpoint manifest、revision 和 W4 metadata；HTTP handler 只负责协议和 admission，通过 bounded channel 交给一个 GPU runtime owner。当前仓库只有 CLI `src/main.rs`，没有 server 模块；第一份协议 PR 由成员2创建 `src/server/**` 和一个可独立运行的 stub binary，成员1负责修改 `src/main.rs`/Cargo 入口并把真实 runtime 接入。

| 区域 | 默认 owner | 说明 |
| --- | --- | --- |
| `crates/apxinf-model/src/qwen35/**`、request state、GDN/attention、runtime adapter | 成员1 | 唯一模型状态机 owner；不把模型语义塞进通用 backend |
| `crates/apxinf-cuda/src/**` 的生产接口和已验收 kernel FFI | 成员1 | 成员3的候选必须经过成员1接收后进入生产路径 |
| `crates/apxinf-loader/**`、checkpoint manifest、离线 reference/oracle | 成员2 | 拥有解析/校验代码和稳定 manifest API；成员1只负责从生产 runtime 调用该 API，不重写 loader 语义 |
| `src/server/**`、`src/bin/apxinf_protocol_stub.rs`、协议测试与 probes | 成员2 | 完整 HTTP/SSE/JSON surface；stub 入口可独立运行，不改模型 forward |
| `src/main.rs`、`Cargo.toml`/`Cargo.lock` 的生产入口和 runtime 接线 | 成员1 | 将成员2 server 模块接到真实 runtime；成员2提出依赖变更，成员1合入入口文件 |
| `benchmarks/campaign/**`、`scripts/campaign/**`、Nsight/显存/paired A/B | 成员3 | 首个任务可创建这些目录并放 README；只记录证据，不修改 evaluator |
| `crates/apxinf-cuda/kernels/experimental/**` | 成员3 | 首个实验可创建；不会被自动构建，进入候选需显式 adapter/feature flag，再由成员1接入生产 FFI |
| `REPORT.md`、最终配置、集成分支 | 成员1裁决；成员3起草 | 报告必须绑定最终 SHA 和 raw artifact |

### 3.2 共享文件规则

- 同一时间只有一个 owner 修改共享文件。
- `Cargo.toml`、`Cargo.lock`、`src/main.rs`、公共 trait 和公共错误类型属于成员1的集成文件。成员2/3可在 task-spec 中提出最小 diff，但默认由成员1落地或在 review 时明确授权。
- 成员2/3不得直接修改成员1正在使用的 `qwen35` 状态机或生产 CUDA 文件；需要接口时先提交 adapter/trait 变更 PR。
- 记录文件采用“追加、短行、带 commit SHA”规则；冲突由成员1合并，不用强制覆盖别人的记录。

### 3.3 Loader 与 admission handoff

- 成员2拥有 checkpoint 文件解析、revision/shape/dtype/W4 metadata 校验和不可变 `LoaderManifest` API；成员1拥有把 manifest 消费进生产模型和显存 residency 的接线。
- W4 loader 交接必须明确两个不同的 pack 轴：以 `k_proj` 为代表，
  `weight_packed` 沿 K 打包（`[1024, 640] I32`，`640=5120/8`），
  `weight_scale` 按 K 的 group-32（`[1024, 160] BF16`，`160=5120/32`），而
  `weight_zero_point` 沿 N 打包（`[128, 160] I32`，`128=1024/8`）。不能把 zero-point
  误解为沿 K 打包；down projection 的对应 inventory 为
  `weight_packed [5120,2176]`、`weight_scale [5120,544]`、
  `weight_zero_point [640,544]`。这些 shape 是 loader/fixture 的方向性验收约束，
  不是允许提交真实权重切片的理由。
- M2-O0 的 oracle 采用“成员2写生成器，成员1服务器一次性执行”的交接：成员2不下载或
  展开完整 checkpoint，不把本地 Qwen3Next 输出当权威；成员1在 GPU1 logical lane、全局
  GPU lock 下生成选择性 layer golden 和 manifest，落到受控共享 artifact 路径。PR 只提交
生成器、synthetic W4 fixture、golden schema 和 SHA256，不提交权重、完整 BF16 副本或
  hidden 答案。
- 成员2拥有纯协议 admission：JSON/schema/type/range、token vocabulary、`temperature`、未知字段和 `prompt + max_new_tokens <= max_model_len`。
- 成员1拥有 runtime capacity admission：当前显存、per-request state/page、并发槽位和可恢复容量拒绝。
- 两类 admission 通过一个不暴露 CUDA 内部类型的 `RuntimeCapabilities`/`AdmissionDecision` 接口交接；成员2不读取 CUDA allocator，成员1不绕过协议校验。

### 3.4 Oracle artifact 与 synthetic W4 fixture

Oracle 分为两个阶段：成员2提交可在 CPU 上检查的生成器、schema 和 synthetic fixture；
成员1在服务器 GPU1 logical lane 执行一次真实 checkpoint oracle，生成选择性层的
hidden/state/logit golden。服务器使用全局 GPU lock，artifact 放在
`/mnt/chuangxin/team2/artifacts/apxinf/oracle/<revision>/<commit-sha>/`，记录完整命令、
model revision、GPU UUID、显存峰值和 SHA256。远程成员只消费 manifest、golden schema 和
获准的 golden 文件，不下载或展开完整模型。服务器原始产物默认只在受控共享存储保留；
需要远程成员本地重放时，成员1只导出不含模型权重、私有 hidden 答案或凭据的最小
golden bundle，并通过项目批准的 artifact/release 通道传递，同时在 handoff record
登记导出方式、文件清单和 SHA256。没有批准的导出包时，远程成员只消费 manifest/schema/
hash，不能自行复制服务器目录或模型。

“一次执行”按 artifact identity 定义：`generator SHA + model revision + input manifest SHA +
layer/stage selection + schema version` 不变时复用已有 artifact；任一项变化才重新排 P0
oracle job。完整 BF16 权重副本不是强制交付物，优先流式/逐层反量化并只持久化选择性
dequant block 和 golden。大型 golden 保留在服务器供 replay；远程成员默认只读取 PR 中的
manifest/hash/schema，只有小型、公开且非敏感的 synthetic fixture 才进入 Git。

提交的 synthetic W4 fixture 必须同时测试两个不同的打包方向，至少包含以下 inventory：

| Tensor | Shape / dtype | 方向与公式 |
| --- | --- | --- |
| `k_proj.weight_packed` | `[1024, 640] I32` | 沿 K 打包，`640 = 5120 / 8` |
| `k_proj.weight_scale` | `[1024, 160] BF16` | 沿 K，每 32 个 K 一组，`160 = 5120 / 32` |
| `k_proj.weight_zero_point` | `[128, 160] I32` | 沿 N 打包，`128 = 1024 / 8` |
| `down_proj.weight_packed` | `[5120, 2176] I32` | 沿 K 打包，`2176 = 17408 / 8` |
| `down_proj.weight_scale` | `[5120, 544] BF16` | 沿 K，每 32 个 K 一组，`544 = 17408 / 32` |
| `down_proj.weight_zero_point` | `[640, 544] I32` | 沿 N 打包，`640 = 5120 / 8` |

fixture 还必须覆盖非对齐尾块、极值 4-bit 值、group boundary 和 N/K 互换会失败的
定向断言。不得从真实 checkpoint 复制 tensor slice；fixture 的数值、pack/unpack 公式和
期望 BF16/F32 输出必须完全由测试生成。

真实 oracle 的 golden coverage 分三层，避免把每个 8K/16K 层 hidden 全量落盘：

1. **短序列层级层**：覆盖所有 64 层的轻量 checksum/shape/state summary，并对每种层型
   保存可数值比较的输入/输出；至少覆盖早/中/晚 GDN 和早/中/晚 full-attention。
2. **五类 kernel acceptance（F/G/H/I/J）**：每一类保存独立的 input、参数 manifest、
   output hidden/state/logit 和误差统计，能由成员1或成员2单独重放，不依赖完整模型副本。
3. **长序列层级层**：8K/16K 只对 task-spec 指定的代表层和最终 logits/state 生成 golden；
   采用流式逐层读取，服务器保留 raw artifact，Git 只记录 manifest/hash/schema。

若某个 F/G/H/I/J acceptance cell 需要增加层或序列长度，成员2在 handoff record 中修改
selection 并重新排 P0；不能让远程成员通过下载完整权重替代服务器 oracle。

## 4. 三个角色的交付边界

### 成员1：服务器与最终集成

负责 loader/runtime 接口整合后的真实模型执行、W4/GDN/full-attention 语义、GPU worker、request cancellation、device-budget admission、GPU0 正式验证、最终 branch 和 release artifact。

不负责独自重写协议；协议行为必须消费成员2已经通过 stub 的 contract tests。

### 成员2：协议与 oracle

负责完整协议 surface、schema、`/health`、SSE/JSON、错误映射、容量拒绝和失败恢复；负责 checkpoint manifest、Python/reference oracle generator、hidden 代理集、token trajectory 记录和 protocol gate 原始证据。真实 checkpoint golden 由成员1按 M2-O0 服务器队列执行。

不负责核心 forward、CUDA 性能和 scorer；不得通过修改 evaluator 让测试变绿。

### 成员3：性能、算子与实验

负责 W4/GEMV/CUDA Graph 的隔离实验、benchmark harness、GPU profile、显存账本、C4/C8/context/MTP/vision bonus 证据和报告草稿。

不把 microbenchmark 或非 RTX 4090 结果宣称为正式成绩；不在文本 `BASE_GOOD` 前强行开启 bonus。

## 5. 每个 agent 的必读资料

### 全员必读

- `README.md`
- `system_design.md`
- `APXINF_FINAL_EXECUTION_PLAN_2026-08-22.md`
- `APXINF_QWEN38_TECHNICAL_PLANS.md`
- `benchmarks/qwen38_4090/evaluation/contract-v1.json`
- `benchmarks/qwen38_4090/evaluation/submission-schema-v1.json`
- `benchmarks/qwen38_4090/evaluation/test.py`
- 本目录的 `README.md`、`SPEC.md`、Git/交接流程

### 按角色追加

- 成员1：`crates/apxinf-model/`、`crates/apxinf-cuda/`、`crates/apxinf-core/`、`score_submission.py` 的 eligibility 逻辑。
- 成员2：`run_evaluation.py` 的协议请求和 gate（重点约 `658-790` 行）、`score_submission.py` 的 reliability 检查（重点约 `220-260` 行）、`score_multimodal.py`、`multimodal_scoring.py`、`multimodal-contract-v1.json`、`src/main.rs` 当前 CLI 行为。
- 成员3：`run_evaluation.py` 的 cell 生成与 evidence、`score_submission.py` 的 latency/multi/reliability 逻辑、`contract-v1.json` 的 performance/context/multi 章节、`APXINF_QWEN38_TECHNICAL_PLANS.md` 的 profile/实验纪律章节、`doc/DEVLOG.md` 和现有 `scripts/`。

## 6. 服务器与本地拓扑

服务器上只保留成员1的工作树和验证过程；成员2/3在自己的电脑 clone 同一 `origin`，推送 feature/experiment 分支。成员1通过 fetch 和 worktree 检出 PR 分支，运行测试后合并。

服务器已核实的 GPU UUID：

| 标签 | UUID | 约定用途 |
| --- | --- | --- |
| GPU0 | `GPU-d074a13d-dbb6-fceb-4caf-a45be9be9281` | 最终集成、正式单卡成绩 |
| GPU1 | `GPU-343bc895-b011-22fa-4449-97207aa2bdec` | oracle/protocol replay |
| GPU2 | `GPU-f4efcc89-d74e-d37b-caf1-52cde9f0582e` | kernel/profile replay |
| GPU3 | `GPU-ea64faa4-13fb-ce41-1180-d6edbfb6be2f` | context/C4/C8/vision/MTP replay |

这些是逻辑分配，不代表四个任务同时运行。当前服务器只有一个账号，默认一次只运行一个 GPU job；`server-gpu-validation.md` 中的 lock 和 artifact 规则是强制的。

## 7. Git 规则

- 集成分支：`APXinf-Contest-2026`，只由成员1合并。
- 成员2：计划创建 `feat/protocol-stub`、`feat/oracle-loader`。
- 成员3：计划创建 `exp/w4-gemv`、`exp/graph-benchmark`、`exp/bonus-context`（或同格式的具体 bonus 名称）。
- 成员1：`feat/qwen35-runtime`、`integrate/<pr-number>`。
- 禁止直接 push 集成分支；禁止 force-push 他人分支；PR 必须包含完整 commit SHA、测试命令、artifact 路径和回滚点。
- 服务器验证用 `git worktree` 隔离，不在成员1主工作树中切换到远程分支。

详细命令见 [git-pr-workflow.md](workflows/git-pr-workflow.md)。

## 8. 状态、门禁和停止规则

任务状态只能使用：`planned`、`active`、`blocked`、`review`、`integrated`、`rejected`、`done`。

分层 PR 的 `integrated` 表示“已合并到集成分支并通过该 task-spec 的局部门禁”，不表示整份提交已经 eligible。局部门禁示例：协议 stub PR 只要求 fake-runtime contract tests、全部负控、恢复和接口 review；实验 harness PR 只要求脚本可复现、fixture 和静态/单元测试，不要求 GPU0 最终成绩。

最终候选的 `done/release` 至少需要：

- `python3 benchmarks/qwen38_4090/evaluation/test.py check` 通过；
- 编译/单元测试通过；
- protocol gate 全部通过；
- public 6/6、midterm hidden 至少 11/12 的正确性证据；
- 成功率至少 99%，目标 100%；
- 五项 reliability boolean 全为 true；
- 失败、容量拒绝、取消后 `/health` 和 8-token 小请求恢复；
- 没有外部 runtime/模型/GPU fallback，`fallback_active=false`；
- 性能结果有 warmup 1、measured 5、CV <= 10% 和 GPU0 环境记录。

五项 reliability 的运行映射：

| Boolean | 使其失败的事件 | 资格影响 |
| --- | --- | --- |
| `no_unexpected_oom` | 任一 scored base cell 出现 OOM/out-of-memory；已预期、被 admission 拒绝且恢复的容量边界不算 unexpected | 任一 false 使整份 `eligible=false` |
| `no_nan` | evaluator 在 scored base row 的 `output_text` 中看到 `nan/+nan/-nan`；内部 tensor NaN 需额外 instrumentation 才能发现 | 任一 false 使整份 `eligible=false` |
| `no_fallback` | health、error 或 output evidence 出现 fallback，或最终 `/health.fallback_active` 不为 `false` | 任一 false 使整份 `eligible=false` |
| `no_xid` | 运行期间发现 Xid，或 evaluator 无法取得 Xid evidence | 任一 false 使整份 `eligible=false` |
| `service_healthy_after_failure` | campaign 结束 health 不正常，或失败后恢复探针失败 | 任一 false 使整份 `eligible=false` |

这些布尔值是全局 eligibility gate，不是每个 cell 的局部扣分。每次正式 campaign 必须保存 raw rows、Xid evidence、开始/结束 health 和失败后的 8-token recovery。R0 若 `journalctl -k`/等价 Xid 证据不可读，先标记 blocked，不启动正式评分。

### 8.1 C4/C8 tail guard 的合同与实现差异

冻结 `contract-v1.json` 写的是 p95 TTFT 不超过自身单请求的 1.5 倍；当前 `score_submission.py` 实现还会乘以 concurrency（C4 为 6 倍、C8 为 12 倍）。评测器/scorer 的实际输出是官方分数来源，但本项目接收候选采用更严格的合同 guard：同时记录 `p95_ttft / single_ttft <= 1.5` 和 scorer 当前 guard，只有严格 guard 通过才进入 `BASE_GOOD`。不得修改 scorer；若官方实现改变，重新记录 DECISIONS。

任一优化使 correctness、reliability、显存或 CV 退化，立即标记 `rejected`，回到最近的 `BASE_GOOD` SHA。不要为了追分继续堆叠未解释的变量。

## 9. 数据与 artifact policy

允许提交：源代码、测试、短小公开 fixture、配置样例、Markdown 记录、artifact manifest，以及不含平台私有内容的 synthetic hidden-proxy fixture。禁止提交：平台真实 hidden case/答案、模型权重、tokenizer 私密文件、服务凭据、完整日志、`.ncu-rep`/`.nsys-rep` 大文件和 `evaluation/runs/` 生成物。大文件放服务器或共享存储，记录其路径、SHA256、生成 commit 和 GPU UUID。

## 10. 决策权和争议处理

- 合同合规和 eligibility：以 evaluator 为准，成员2负责证据，成员1裁决是否阻塞集成。
- 模型语义和生产 runtime：成员1最终裁决。
- 实验设计和数据完整性：成员3负责起草，成员1决定是否进入主线。
- 如果远程成员的本地环境与服务器不一致，代码 PR 仍可 review，但性能/显存结论必须由成员1在指定 GPU 上重放。
- 所有重大取舍写入 [DECISIONS.md](records/DECISIONS.md)，不要只留在聊天记录中。
