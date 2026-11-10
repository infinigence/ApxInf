# 协作进度

更新时间：2026-08-23
集成分支：APXinf-Contest-2026
集成 owner：成员1

状态定义：planned / active / blocked / review / integrated / rejected / done

| Task ID | Owner | Scope | Status | Latest SHA/PR | Evidence / next action |
| --- | --- | --- | --- | --- | --- |
| COLLAB-002 | 成员1 | 线下 agent prompt 决策、本地开发环境文档和进度写入边界 | done | `e3054497321249ea45308ff1f471e80faad0d35d` | prompt 不入库；成员2/3 CPU-only 可启动，CUDA 为成员3可选；聚合 PROGRESS 由成员1维护 |
| COLLAB-001 | 成员1 | 创建三人协作 spec、角色手册、Git/GPU 流程和模板 | done | local docs commit (HEAD) | 结构/链接/合同校验通过；`cargo check --workspace --locked` 通过；完整 cargo test 受既有 pi05_integrity_probe 默认 feature 问题阻塞，未修改源码 |
| M1-R0 | 成员1 | 固定环境、模型、合同 hash，clean build baseline | blocked | `d1d0b0aec2f9545eb0e2195e2e7ea0af1babbff1` | `test.py check`、`cargo check --workspace --locked` 通过；GPU/模型/合同已记录；`journalctl -k` 无 journal，R0 artifact `/mnt/chuangxin/team2/artifacts/apxinf/r0/d1d0b0aec2f9545eb0e2195e2e7ea0af1babbff1/`，environment SHA256 `7a7fab226351883b568af76771473095a6a2cef648490cf341fe57b747affa62`，command SHA256 `af29d9174a07a88b7f55550b99ab369f5fe0db53a5aa5811acda7846fba2534c` |
| M2-P0 | 成员2 | protocol stub、schema、/health、七项负控和恢复 gate | integrated | `d26738f` | Rust protocol 33/33、Python oracle/protocol 21/21；成员1后续只接稳定 adapter，不重写协议语义 |
| M2-P1 | 成员2 + 成员1 | production replay、runtime-neutral conformance 与 reliability harness | integrated | source `ecbe23dae3b7dd5d14393c9adcbe92d65e4585f6`; integration `ff9392aeab12265429d9035b9d93200fa32a7f23` | rollback `2056ec8e1780e25d679ac53bf530ea51cf99b8e6`; Rust protocol 38/38、Python protocol 8/8、oracle 21/21、evaluator check 与 stub/oracle manifest SHA256 通过；真实 production replay 等待成员1 runtime |
| M2-O0 | 成员2 + 成员1 | oracle generator、checkpoint manifest、选择性 layer golden、hidden 代理集 | done | `46182a1167570e7595b3e658b02fb8acadac9f7a` | GPU1 queue `P0-oracle-20260823T095000Z` 成功；峰值 20942 MiB；artifact `/mnt/chuangxin/team2/artifacts/apxinf/oracle/63768c10df38c0395e12ef49edac1bd539eaeeea/46182a1167570e7595b3e658b02fb8acadac9f7a/` |
| M2-L0 | 成员2 | synthetic W4 pack/unpack fixture 与 loader 方向性测试 | integrated | `d26738f` | loader 26/26；K-packed weight、group-32 scale、N-packed zero-point manifest validation 已集成 |
| M1-R1 | 成员1 | runtime adapter、bounded GPU worker、device-budget admission | active | `2505b09` | 增量 single-owner ProtocolRuntime transport 已通过 4 项 focused tests；真实 checkpoint CUDA step executor 尚未接入，stub=false 仍未启用 |
| M1-C0 | 成员1 | Qwen35 loader/model/state/GDN/full-attention vertical slice | active | `9920468` | `2b36aff` real packed loader、`cfa1206` CUDA device W4 projection、`3274f38` native BF16 scale preservation、`9920468` layer-3 full-attention assembly；GPU2 单投影 evidence 通过，尚未达到 REFERENCE_LAYER |
| M1-S0 | 成员1 | strict Qwen3.5 production serve validation | active | `cddde59` | `--model/--revision/--gpu-uuid/--bind/--max-model-len/--queue-capacity` 校验及 2 项 focused tests；真实 CUDA executor 未链接时明确拒绝启动 |
| M3-E0 | 成员3 + 成员1 | W4/GEMV/Graph baseline 与 paired benchmark harness | integrated | source `f62849b003d7ad395d44b5f74f6da50f577d9843`; integration `0b18aa5404c7b3acce3b1b8fee39bd4f825d7a2d` | 离线 shape inventory、实验 manifest validator、M3-E0 handoff 已集成；仅 development preparation，等待真实 BASE_GOOD/GPU0 replay |
| M3-B0 | 成员3 | 显存账本、context/C4/C8/MTP/vision bonus evidence | planned | - | 文本 BASE_GOOD 后逐项开启 |
| REL-001 | 成员1+2 | protocol/reliability eligibility campaign | planned | - | 七项 gate、五项 boolean、失败后恢复 |
| REL-002 | 成员1+3 | GPU0 clean checkout final replay | planned | - | 固定 GPU0，warmup 1 + measured 5，CV <= 10% |

## 里程碑

| Milestone | Exit condition | Owner |
| --- | --- | --- |
| R0 reproducible | clean checkout、合同/model hash、GPU identity、test.py check | 成员1 |
| P0 protocol ready | stub 与真实服务逐项通过 protocol gate | 成员2 + 成员1 |
| BASE_CORRECT | public 6/6，hidden 代理集通过，EOS/usage/recovery 正常 | 成员1 + 成员2 |
| BASE_GOOD | request success >= 99%，五项 reliability 全 true，GPU0 可重放 | 成员1 |
| OPT_CANDIDATE | 单变量 paired A/B，端到端净收益，CV 和显存通过 | 成员3 + 成员1 |
| FINAL | clean checkout、最终 report、合同/环境/artifact hash 完整 | 成员1 |

## 更新规则

成员2/3在自己的 PR 或 progress log 写日常细节；成员1合并后把源 SHA、集成 SHA、服务器 replay 和状态回填到本表。阻塞超过一个工作周期必须新增 incident 或 decision 记录。
