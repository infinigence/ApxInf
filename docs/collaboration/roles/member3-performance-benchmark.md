# 成员3：性能、算子与实验手册

角色代号：`member3` / `performance-benchmark`
工作位置：成员3自己的电脑；服务器上的 RTX 4090 profile 和正式重放由成员1代跑。

## 必读资料

- `docs/collaboration/README.md`
- `docs/collaboration/local-development-environment.md`
- `docs/collaboration/SPEC.md`
- `docs/collaboration/workflows/git-pr-workflow.md`
- `docs/collaboration/workflows/server-gpu-validation.md`
- `docs/collaboration/workflows/handoff-and-review.md`
- `README.md`、`system_design.md`
- `APXINF_FINAL_EXECUTION_PLAN_2026-08-22.md` 的 profile、GPU 编排和接收门
- `APXINF_QWEN38_TECHNICAL_PLANS.md` 的实验纪律、显存账本和 bonus 章节
- `benchmarks/qwen38_4090/evaluation/contract-v1.json` 的 performance/context/C4/C8 条款
- `benchmarks/qwen38_4090/evaluation/run_evaluation.py` 的 cell/evidence 生成和 `score_submission.py` 的 latency/multi/reliability 逻辑
- `doc/DEVLOG.md` 和现有 `scripts/` benchmark 约定
- `crates/apxinf-cuda/` 的 backend、graph、tuning 和 tests

## 目录边界

默认允许修改：

- `benchmarks/campaign/**`、`scripts/campaign/**`；
- `crates/apxinf-cuda/kernels/experimental/**` 的候选 W4/GEMV/Graph kernel；
- `docs/collaboration/records/EXPERIMENTS.md` 和 `REPORT.md` 草稿；
- 独立 benchmark/config/manifest 文件。

这些实验目录当前可能不存在；第一个实验任务负责创建目录并加入 README，说明入口、依赖和 feature-off 默认。`crates/apxinf-cuda/build.rs` 只编译显式 adapters，experimental kernel 不会自动进入构建；进入候选必须新增显式 adapter/feature flag，并由成员1 review 后接入生产 FFI。

默认禁止修改：

- `benchmarks/qwen38_4090/evaluation/` 合同、scorer、公开数据和 runs；
- 成员1的 `qwen35` 模型 state machine、request admission 和生产 FFI；
- 成员2的协议 surface、loader hard gate 和 oracle 语义；
- 任何未通过 feature-off/on 对照的默认配置。

候选需要进入生产路径时，先提交隔离实验 PR 和 adapter 说明，由成员1在 GPU0 重放后再合并生产 FFI。

## 实验原则

每轮实验只改变一个主要变量，并固定：model revision、完整 commit SHA、contract SHA256、GPU UUID、CUDA/driver、输入 manifest、warmup/repeat、时钟策略和服务命令。结果必须包含 baseline/candidate paired A/B，不能用单 kernel 数字代替客户端端到端结论。

优先顺序：

1. packed-W4 decode GEMV 与成熟库 baseline；
2. 已冻结 bucket 的 CUDA Graph；
3. BF16/paged KV 和显存 admission；
4. C4，再 C8；
5. 文本 `BASE_GOOD` 后的 context、MTP K=1 feasibility probe 和 vision vertical slice。

默认关闭 prefix cache、mega-kernel 和 INT4 KV。任何 candidate 若出现 NaN、OOM、fallback、Xid、健康恢复失败、正确性变化或 CV > 10%，直接标记 rejected。

## 基准验收

- 正式 latency cell：warmup 1、measured 5、CV <= 10%；TTFT/TPOT 以客户端接收时间为准。
- C4/C8：32 个请求，success/correctness 100%、Jain >= 0.95、p95 TPOT <= 3x、无 fallback，结束后 health 正常。TTFT 同时记录两种 guard：合同目标 p95 TTFT <= 自身单请求 1.5x（本项目 team acceptance policy），以及当前 scorer 实现中的 concurrency * 1.5x（官方评分实现）。这是团队接收策略与官方 scorer 的差异，不是对 evaluator eligibility 的改写；不要修改 evaluator 来消除该差异。
- context：32640 仅诊断，32768 按公式仍为 0 分；65536 才是首个正分台阶。每个长度完成 6 类任务和失败后小请求恢复。
- MTP：先做 K=1 target exact verify，首/中/末 reject rollback；只有 off/on 端到端 TPOT 和功能题都通过才进入候选。
- vision：按合同 processor、448x448 RGB PNG、`stream=false`；文本服务未冻结时保持隔离。

## 交付物

每个实验 PR 必须附：

- hypothesis、baseline、candidate、唯一变量和接受/拒绝阈值；
- 可复制命令、服务启动参数、输入 manifest；
- GPU UUID、commit SHA、contract/model hash、温度/功耗/时钟；
- raw artifact 路径和 SHA256；
- correctness、reliability、显存、latency、CV、goodput 数据；
- 结论、限制、下一步和一条明确回滚命令。

完整记录使用 [experiment-record.md](../templates/experiment-record.md)，索引追加到 [EXPERIMENTS.md](../records/EXPERIMENTS.md)。

## 本地与服务器协作

本地安装和可选 CUDA 编译条件见
[local-development-environment.md](../local-development-environment.md)。M3-E0 的
harness、manifest、shape inventory 和静态检查不要求本地 NVIDIA GPU。

本地可以完成脚本、shape、CPU/reference 和静态检查；非 RTX 4090 GPU 的结果只能作为开发信号。PR 中把服务器重放写成一条命令，成员1会在 GPU2/GPU3 逻辑 lane 上一次运行一个 job。只有 GPU0 重放的候选才可作为最终报告数字。

## 给 member3 agent 的启动提示

```text
你是 ApxInf 成员3 agent，职责是 benchmark/profile/experiment evidence，不猜测模型语义。
先读 docs/collaboration/README.md、SPEC.md、server-gpu-validation.md、handoff-and-review.md、
README.md、system_design.md、两份方案文档和 contract-v1.json 的性能/bonus 条款。
本地可与成员2并行完成脚本、shape inventory、CPU/reference 和静态验证；不要把 GPU2/GPU3
logical lane 当作并发服务器。需要 RTX 4090 的结论必须给成员1精确 commit、命令、manifest、
GPU UUID 和 artifact 路径代跑，服务器一次只运行一个 job。一次只改一个变量，必须有
feature-off/on paired 对照。
不要修改 evaluation/、核心 forward 或把 microbenchmark 宣称成端到端成绩；失败实验也要记录。
```
