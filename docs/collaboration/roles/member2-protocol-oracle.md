# 成员2：协议与 oracle 手册

角色代号：`member2` / `protocol-oracle`
工作位置：成员2自己的电脑；服务器验证由成员1按提交的命令代跑。

## 必读资料

- `docs/collaboration/README.md`
- `docs/collaboration/SPEC.md`
- `docs/collaboration/workflows/git-pr-workflow.md`
- `docs/collaboration/workflows/handoff-and-review.md`
- `README.md` 的服务接口章节
- `system_design.md`
- `APXINF_FINAL_EXECUTION_PLAN_2026-08-22.md` 的协议章节
- `APXINF_QWEN38_TECHNICAL_PLANS.md` 的 correctness/service 章节
- `benchmarks/qwen38_4090/evaluation/run_evaluation.py`，重点 `protocol_checks` 约 `658-790` 行
- `benchmarks/qwen38_4090/evaluation/score_submission.py`，重点 eligibility/reliability 约 `220-260` 行
- `benchmarks/qwen38_4090/evaluation/contract-v1.json`
- `benchmarks/qwen38_4090/evaluation/submission-schema-v1.json`
- `benchmarks/qwen38_4090/evaluation/multimodal-contract-v1.json`
- `crates/apxinf-loader/`、`src/main.rs`

## 目录边界

默认允许修改：

- `src/server/**` 和 `src/bin/apxinf_protocol_stub.rs`；当前仓库没有 `src/server.rs`。
  不要直接改 `src/main.rs`。如果 stub 需要 HTTP 依赖或 bin 声明，先在 task-spec/PR
  提出最小 `Cargo.toml`/`Cargo.lock`/入口 diff；成员1确认后负责集成入口，成员2可在
  自己分支验证该依赖；
- `crates/apxinf-loader/**` 的 manifest、revision、shape、dtype、W4 metadata 校验和稳定 manifest API；成员1负责生产 runtime 消费该 API；
- `tools/protocol/**` 或 `scripts/protocol/**` 的 stub、schema probe、负控和恢复测试；这些目录不存在时由首个任务创建并附 README；
- `tools/oracle/**`、`python/oracle/**` 或同等隔离目录中的 reference/代理集生成器；
- `docs/collaboration/records/` 中 protocol/oracle 结果的追加记录。

禁止修改：

- `benchmarks/qwen38_4090/evaluation/` 任何合同、scorer、生成器和 runs；
- `crates/apxinf-model` 的核心 forward/state machine；
- `crates/apxinf-cuda` 的生产 kernel；
- 公开答案、hidden case、固定 case ID 或 token 序列硬编码。

如果协议需要一个 runtime 接口，只提交最小 trait/adapter 变更，并在 PR 中注明由成员1接线。

## 协议实现清单

### `/health`

生产服务必须持续返回：`status=ok`、固定 `evaluation_contract`、固定 model revision、实测 `max_model_len`、实测 `parallel_requests`、`fallback_active=false` 和当前 capabilities。未完成 vision 时声明 `multimodal=false`。本地 stub 只能返回带 `stub=true`/fixture 标记的 contract health；stub health 不是正式 GPU 证据，不能把 fixture 值当作生产能力声明。

### `/v1/evaluations/generate`

- 接收预分词整数 `input_ids`；范围上限取 checkpoint `text_config.vocab_size`（加载到
  runtime model config 后为 `vocab_size`；实测 `248320`，合法范围 `[0,248320)`），不能
  取 tokenizer `vocab_size`（实测 `248044`）；
  `image_token_id=248056` 必须保留为合法 model token。仍需拒绝空数组、负数和
  `4294967295`。
- 只接受 `temperature=0`；`ignore_eos` 必须是布尔值。
- 计算总预算 `len(input_ids) + max_new_tokens`，同时检查 `max_model_len` 和 device budget。
- `stream=true`：SSE index 从 0 连续递增，request_id 不串线，终止事件带 usage 和 `[DONE]`。
- `stream=false`：HTTP 200，返回 `type:"result"`、`output_ids` 和 usage。
- 请求级错误不能污染 worker；取消、capacity reject、非法输入和 CUDA error 都要释放 state/page。

### Protocol eligibility gate

六项可解析的结构化负控使用 `stream=false`；malformed JSON 以原始不可解析 body 发送。必须保存原始 status、响应 JSON/text、时间和 commit SHA：

| id | 请求 | 硬条件 |
| --- | --- | --- |
| `malformed_json` | 原始 body `{not-json`（无可解析 `stream` 字段） | HTTP 400 |
| `empty_input_ids` | `input_ids: []` | HTTP 400 + JSON `error` |
| `negative_token_id` | `input_ids: [-1]` | HTTP 400 + JSON `error` |
| `out_of_vocabulary_token_id` | `input_ids: [4294967295]` | HTTP 400 + JSON `error` |
| `unsupported_temperature` | `temperature: 0.1` | HTTP 400 + JSON `error` |
| `over_budget` | `max_new_tokens: health.max_model_len` | HTTP 400 + JSON `error` |
| `unsupported_modality_field` | `images: ["x"]` | HTTP 400 + JSON `error` |

然后执行 `valid_short_nostream_request`：8-token prompt、`max_new_tokens=1`、`stream=false`，必须 HTTP 200、`type:"result"`、一个 output token、usage 为 prompt 8/completion 1。最后执行 `health_after_invalid_requests` 和 `health_contract_identity`。任一项失败都是 eligibility failure，不是 review 加分项。

## Oracle 与 hidden 代理集

- 先锁定 checkpoint revision、tokenizer、generation config 和 EOS `[248046, 248044]`。
- M2-O0 的边界是“本地写生成器，成员1服务器执行真实 checkpoint”：你不下载约 19.57 GiB
  权重、不展开完整 BF16 模型，也不把 Qwen3Next 输出当作权威 oracle。生成器必须支持
  成员1在 GPU1 logical lane、全局 lock 下选择性生成 layer golden，并输出 manifest、
  golden schema、生成命令和 SHA256。
- oracle 输出必须包含 input manifest、reference token IDs、decoded text、trajectory 完整 budget、生成参数和 SHA256。
- 代理集覆盖 early/middle/late retrieval、distractor、multi-hop、revision-resolution、aggregate；不能复用公开答案。
- 代理集只用于本地正确性和回归，可提交 synthetic fixture/manifest；平台真实 hidden case、答案和私有 tokenizer 数据绝不可提交，也不得把代理通过率写成 hidden 正式成绩。
- 每个失败保存最小复现请求和服务恢复结果。

W4 loader 单元测试必须使用合成 fixture，不能复制真实 tensor slice。至少覆盖：

| Tensor | Shape / dtype | pack 轴 |
| --- | --- | --- |
| `k_proj.weight_packed` | `[1024,640] I32` | K (`640=5120/8`) |
| `k_proj.weight_scale` | `[1024,160] BF16` | K group-32 |
| `k_proj.weight_zero_point` | `[128,160] I32` | N (`128=1024/8`) |
| `down_proj.weight_packed` | `[5120,2176] I32` | K |
| `down_proj.weight_scale` | `[5120,544] BF16` | K group-32 |
| `down_proj.weight_zero_point` | `[640,544] I32` | N (`640=5120/8`) |

测试必须包含尾块、极值 nibble、group boundary 和 N/K 方向互换的负断言；完整规则见
`SPEC.md` 的 “Oracle artifact 与 synthetic W4 fixture”。

## 本地开发与交付

本地安装和 CPU-only/CUDA 边界见
[local-development-environment.md](../local-development-environment.md)。

本地先用 fake runtime/stub 做协议测试，不需要登录服务器。提交 PR 时必须附：

- stub 启动命令和协议测试命令；
- 七项 gate 的逐项原始结果；
- 8-token non-stream result 和 health identity 结果；
- loader/oracle manifest 样例；
- 错误映射、恢复和取消测试；
- 与成员1 runtime adapter 的接口说明。

需要 4090 证据时，在 PR 中给出：精确 commit、启动命令、base URL、超时、请求 manifest、期望输出和 artifact 保存位置。成员1会在服务器按 [server-gpu-validation.md](../workflows/server-gpu-validation.md) 串行重放。

## 给 member2 agent 的启动提示

```text
你是 ApxInf 成员2 agent，职责是 protocol/oracle/correctness evidence，不做 CUDA 性能架构。
先读 docs/collaboration/README.md、SPEC.md、git-pr-workflow.md、handoff-and-review.md、
README.md、system_design.md、APXINF_FINAL_EXECUTION_PLAN_2026-08-22.md、
APXINF_QWEN38_TECHNICAL_PLANS.md，以及 run_evaluation.py protocol_checks 和
score_submission.py reliability eligibility 代码。
你的代码必须能在没有服务器 GPU 的本地 fake runtime 上运行；不要下载 19.57 GiB checkpoint，
不要展开完整 BF16 模型，也不要修改 evaluation/、scorer、
核心 forward 或公开/hidden 答案。七项 stream=false protocol gate、8-token result、health
恢复和 contract identity 必须逐项记录。需要 GPU 的结论交给成员1在固定 UUID 上重放。
```
