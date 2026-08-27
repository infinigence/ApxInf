# 成员1：服务器与最终集成手册

角色代号：`member1`
主职责：模型/runtime eligibility 主线、服务器验证、最终合并和 release 裁决。

## 必读资料

先读：

- `docs/collaboration/README.md`
- `docs/collaboration/local-development-environment.md`
- `docs/collaboration/SPEC.md`
- `docs/collaboration/workflows/git-pr-workflow.md`
- `docs/collaboration/workflows/server-gpu-validation.md`
- `docs/collaboration/workflows/handoff-and-review.md`
- `README.md`、`system_design.md`
- `APXINF_FINAL_EXECUTION_PLAN_2026-08-22.md`
- `APXINF_QWEN38_TECHNICAL_PLANS.md`
- `benchmarks/qwen38_4090/evaluation/contract-v1.json`
- `benchmarks/qwen38_4090/evaluation/run_evaluation.py`
- `benchmarks/qwen38_4090/evaluation/score_submission.py`
- `crates/apxinf-model/`、`crates/apxinf-cuda/`、`crates/apxinf-core/`

必须理解：模型的 `architecture`、W4 group-32 asymmetric 语义、GDN state、full-attention gate、EOS/usage、bounded GPU worker、显存 admission 和五项 reliability gate。

## 目录边界

默认允许修改：

- `src/main.rs`、`Cargo.toml`/`Cargo.lock` 以及真实 runtime 与成员2 server 模块的 adapter 接线；
- `crates/apxinf-model/src/qwen35/**`、模型 state、层执行和请求生命周期；
- `crates/apxinf-cuda/src/**` 的生产接口、已验收 kernel FFI 和 CUDA 资源管理；
- `crates/apxinf-core/**` 中确有必要的公共 tensor/device 接口；
- `docs/collaboration/records/PROGRESS.md`、`DECISIONS.md` 的集成结论。

默认禁止修改：

- `benchmarks/qwen38_4090/evaluation/` 全部文件；
- 成员2的协议 stub、oracle fixture 和 loader manifest 语义；成员1只消费其稳定 API，不在集成时重写校验逻辑；
- 成员3的实验脚本和 raw artifact；
- 未经证据验证就打开 prefix cache、MTP、长上下文或 multimodal capability。

共享文件改动必须在 PR 描述中列出接口影响；同一时间只允许一个分支修改 `qwen35` state machine。

## 工作分阶段

成员1的日常代码在 `feat/qwen35-runtime` 或 `integrate/<pr-number>` 等隔离分支/
review worktree 完成；`APXinf-Contest-2026` 只接受经过 review 的合并结果。若主工作树
有未提交改动，不清理或覆盖它，先记录状态并使用 sibling worktree。

成员2提交的真实-checkpoint oracle 生成器由成员1在服务器 GPU1 logical lane 执行一次。
执行前取得 `/tmp/apxinf-gpu-job.lock`，使用固定 model revision 和完整 commit SHA；
输出的选择性 hidden/state/logit golden、manifest 和 SHA256 放入
`/mnt/chuangxin/team2/artifacts/apxinf/oracle/`。这项执行不允许被描述为成员2本地已经
完成，也不与 GPU0/GPU2/GPU3 job 并发。

### R0：可复现基线

- 固定 model revision、contract SHA256、Rust/CUDA/driver 和 GPU0 UUID。
- `test.py check`、`cargo test --workspace --locked`、clean build 通过。
- `journalctl -k --since '-10 min'` 或等价 Xid 证据命令可读；不可读时将 R0 标记 `blocked`。
- 记录当前 starter 缺口和已知不可运行项，不伪造健康字段。

### R1：runtime adapter

- 定义成员2协议层消费的 runtime trait：请求 state、cancel、token event、usage、错误和 capability。
- 让协议 stub 可以用 fake runtime 运行；真实 runtime 未接入前 `/health.multimodal=false`。
- 设计 device-budget admission，使 `prompt + max_new_tokens <= max_model_len` 且不超过实测显存预算。

### R2：模型正确性

- 先完成 loader manifest 和权重 shape/dtype/revision hard gate 的接入。
- 逐算子、逐 GDN/full-attention 层、prefill、单步 decode、多步 decode 对拍。
- 保留可审计的 eager GDN 路径；chunk/fused 路径只能作为 feature flag 候选。
- 公开 6/6 之前不做性能宣称。

### R3：文本 eligibility

- 接入成员2已验收的 HTTP/SSE/JSON surface。
- 逐项重跑七项 protocol probe、8-token non-stream result、health identity 和失败恢复。
- 连续混合请求目标 100%，最低 99%；检查 NaN/OOM/fallback/Xid/health recovery。
- 以 GPU0 运行 public 和 hidden 代理集，生成可复核 artifact。

### R4：窄优化吸收

- 只接收成员3提供且在同一模型、同一 GPU、同一 workload 上端到端净收益的 GEMV/Graph/paged KV 候选。
- 每次只打开一个 feature flag；任何 correctness、reliability、显存或 CV 退化都回滚到 `BASE_GOOD`。
- bonus 只能在文本 eligibility 冻结后进入候选。

### R5：最终冻结

- clean checkout 构建并重放 GPU0 所有通过项。
- 审核 REPORT、PR review 证据、artifact hash、依赖和 capability 声明。
- 只在所有门禁通过后合并/标记 release；最终报告不引用 GPU1-3 作为正式成绩。

## 服务器日常流程

1. `git status --short --branch`，确认主工作树无未提交改动。
2. `git fetch origin --prune`，把远程 PR 检出到独立 worktree。
3. 按 [server-gpu-validation.md](../workflows/server-gpu-validation.md) 获取锁、绑定 UUID、记录环境。
4. 先跑短 smoke 和 protocol gate，再跑完整评测；不要一上来占用长任务。
5. 把命令、SHA、GPU UUID、结果和失败原因写入 PR 和记录文件。
6. 通过分层 review 后合入 `APXinf-Contest-2026`，重新在 GPU0 跑对应局部门禁；最终候选另按 release gate 验收。

## 分层合入与最终 release 验收

`integrated` 只表示 PR 已合并且该任务的局部门禁通过，不要求尚未接入 runtime 的协议 stub 已经 public/hidden eligible。协议 stub 的局部门禁是 fake-runtime contract tests、七项负控、8-token result、health fixture/recovery 和接口 review；实验 PR 的局部门禁是可复现脚本、静态/单元测试和 artifact schema。

最终候选只有在以下条件都满足时才能标记 `done/release`：

- diff 只触及声明范围或有明确接口批准；
- `test.py check` 和相关 Rust/Python tests 通过；
- protocol gate 全部通过；
- public 6/6，hidden 代理集达到计划阈值；
- `eligible=true`，五项 reliability 全 true；
- 失败后 health 和 8-token 请求恢复；
- 性能证据有 CV、环境和 raw artifact；
- 回滚 SHA 和 feature flag 明确。

## 给 member1 agent 的启动提示

```text
你是 ApxInf 成员1 agent。只处理当前 task-spec 指定的模型/runtime 或集成范围。
先读 docs/collaboration/README.md、SPEC.md、对应 workflow、README.md、system_design.md、
APXINF_FINAL_EXECUTION_PLAN_2026-08-22.md、APXINF_QWEN38_TECHNICAL_PLANS.md 和评测合同。
先完成 P0 oracle-prep：审核成员2 generator 的 handoff，使用 GPU1 UUID
`GPU-343bc895-b011-22fa-4449-97207aa2bdec` 和全局 lock 一次执行真实 checkpoint，保存
manifest、选择性 golden、峰值显存和 SHA256；随后释放 GPU1，再按队列运行 GPU0 base。
服务器上一次只运行一个 GPU job，正式验证固定 GPU0 UUID；不要修改 evaluation/ 合同。
每次只改变一个主要变量，先做 correctness/reliability，再做性能；所有结论写入 PR 和 records。
若遇到协议行为问题，通知成员2，不要私自重写协议；若是优化候选，要求成员3提供 paired A/B 证据。
```
