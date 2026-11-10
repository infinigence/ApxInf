# ApxInf 协作入口

本目录是 ApxInf Qwen3.8-27B 单卡 RTX 4090 项目的协作规范。它解决一个实际约束：成员1在服务器上负责集成和 GPU 验证，成员2、成员3在各自电脑上开发，通过 GitHub 分支和 PR 交付，避免共享账号和同时登录造成冲突。

## 先读什么

所有成员先读：

1. [SPEC.md](SPEC.md)：冻结方案、职责、目录边界、门禁和决策权。
2. [git-pr-workflow.md](workflows/git-pr-workflow.md)：本地开发、推送、PR、服务器拉取和合并。
3. [handoff-and-review.md](workflows/handoff-and-review.md)：任务交接、验收证据和 review 规则。
4. [server-gpu-validation.md](workflows/server-gpu-validation.md)：服务器 GPU、端口、锁、日志和正式重放。
5. 对应的角色手册：
   - [成员1：服务器与最终集成](roles/member1-server-integrator.md)
   - [成员2：协议与 oracle](roles/member2-protocol-oracle.md)
   - [成员3：性能、算子与实验](roles/member3-performance-benchmark.md)
6. [local-development-environment.md](local-development-environment.md)：成员2、成员3的本地开发环境、可选 CUDA 环境和验证边界。

启动 prompt 不提交到仓库，由协调者根据角色手册和本地环境文档分别发送给三名
agent。prompt 中的启动报告、首批任务和交付格式必须与本目录规范保持一致。

## 规范来源

本目录不能替代评测合同。遇到冲突时按下列优先级处理：

1. `benchmarks/qwen38_4090/evaluation/` 中冻结的 evaluator、contract 和 schema；这些文件只读。
2. 根目录的 [APXINF_FINAL_EXECUTION_PLAN_2026-08-22.md](../../APXINF_FINAL_EXECUTION_PLAN_2026-08-22.md)。
3. [APXINF_QWEN38_TECHNICAL_PLANS.md](../../APXINF_QWEN38_TECHNICAL_PLANS.md)。
4. 本目录的协作流程。
5. PR 描述和聊天消息。

任何改变上述技术边界的决定，都必须先写入 [DECISIONS.md](records/DECISIONS.md)，由成员1确认后才可实现。

## 当前执行摘要

- 主线：成熟算子混合主线，先取得文本 eligibility，再吸收已证明的窄优化。
- 成员1：模型/runtime、真实 loader 接入、GPU worker、device-budget admission、最终集成和 GPU0 正式验证。
- 成员2：完整 HTTP/SSE/JSON surface、stub、schema、`/health`、协议 admission、错误恢复、oracle generator 和 hidden 代理集；真实 checkpoint oracle 由成员1服务器执行。
- 成员3：W4/GEMV/CUDA Graph profiling、benchmark、显存账本、bonus 实验和报告证据。
- 入口 ownership：成员2负责 `src/server/**` 协议模块和独立 stub binary；成员1负责现有 `src/main.rs`、Cargo 入口和真实 runtime 接线。
- 服务器：只有成员1默认登录和运行任务。成员2/3本地开发；需要 4090 证据时提交可复现命令，由成员1在服务器上代跑。
- 集成分支：`APXinf-Contest-2026`，只由成员1合并通过 review 的 PR。
- 正式 GPU：GPU0；GPU1-3只能作为开发/重放证据，不能替代 GPU0 正式成绩。服务器真实 GPU/模型任务通过全局锁串行排队，逻辑 lane 不代表并发常驻模型。
- Oracle：成员2本地写 generator/schema/合成 fixture，成员1排 P0 在 GPU1 执行真实 checkpoint；raw artifact 留服务器，远程成员只拿批准导出的最小 golden bundle 或 manifest/schema/hash。
- 成员2、成员3的日常进度写在各自 PR 或 task-specific progress log；聚合
  records/PROGRESS.md 由成员1在合并和服务器 replay 后统一更新。

## 记录入口

- [PROGRESS.md](records/PROGRESS.md)：里程碑和当前阻塞。
- [DECISIONS.md](records/DECISIONS.md)：已确认的架构、边界和例外。
- [EXPERIMENTS.md](records/EXPERIMENTS.md)：实验索引；完整结果应附 raw artifact 路径。
- [templates/](templates/)：任务、进度、实验、PR 和事故记录模板。
  Oracle 交接使用 [oracle-handoff.md](templates/oracle-handoff.md)。

## 最小开始流程

```bash
git clone https://github.com/mygqdmmss/ApxInf.git
cd ApxInf
git switch -c feat/protocol-stub origin/APXinf-Contest-2026
python3 benchmarks/qwen38_4090/evaluation/test.py check
cargo check --workspace --locked
```

开发前先从 `task-spec.md` 建一个任务条目；完成后按 `pr-checklist.md` 生成 PR。不要把模型权重、评测生成物、日志、Nsight 文件或本地凭据提交到 Git。
