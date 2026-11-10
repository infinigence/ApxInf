# 协作决策记录

| ID | Date | Decision | Reason | Owner | Status |
| --- | --- | --- | --- | --- | --- |
| D-001 | 2026-08-22 | 成员1是唯一集成 owner，维护 APXinf-Contest-2026 并负责 GPU0 正式验证 | 服务器只有一个账号，避免主工作树和 GPU 任务冲突 | 成员1 | accepted |
| D-002 | 2026-08-22 | 成员2、3在自己的电脑开发，通过 GitHub feature/experiment 分支和 PR 交付 | 不要求三人同时登录服务器；本地可完成协议、oracle generator/schema、脚本和静态验证，真实 checkpoint oracle 另按 D-013 执行 | 成员1 | accepted |
| D-003 | 2026-08-22 | 成熟算子混合是唯一主线；SM89 GEMV/Graph、paged KV、C4/C8、MTP、vision 为隔离候选 | 先拿文本 eligibility，再按证据逐项吸收优化 | 成员1 | accepted |
| D-004 | 2026-08-22 | GPU0 是正式成绩来源，GPU1-3只做 replay/development evidence，服务器任务串行 | 满足固定单卡合同并降低单账号并发风险 | 成员1 | accepted |
| D-005 | 2026-08-22 | max_model_len 是 prompt + output 的总 admission 上限 | evaluator 的 over_budget probe 将 max_new_tokens 设为 health.max_model_len | 成员1 + 成员2 | accepted |
| D-006 | 2026-08-22 | 七项 protocol probe 是 eligibility gate；五项 reliability boolean 任一失败即 eligible=false | evaluator/scorer 的实际逻辑，不是 PR review 可选加分项 | 成员2 + 成员1 | accepted |
| D-007 | 2026-08-22 | evaluator、scorer、contract、公开数据生成器只读 | 保持比赛合同和提交身份完整 | 全员 | accepted |
| D-008 | 2026-08-22 | prefix cache、mega-kernel、262K INT4 KV 默认不进入交付 | 风险高且不能替代基础资格；只有新鲜证据才能重新决策 | 成员1 | accepted |
| D-009 | 2026-08-22 | `integrated` 是分层 PR 合入状态，`done/release` 才表示最终 eligibility/release gate 通过 | 协议 stub、loader 和实验 PR 必须能在真实 runtime 完成前独立合入 | 成员1 | accepted |
| D-010 | 2026-08-22 | loader 解析/API 由成员2拥有，生产接线和 device-budget admission 由成员1拥有；协议 admission 与 runtime capacity admission 通过稳定接口交接 | 避免同一 server 入口和 loader 文件发生 owner 冲突 | 成员1 + 成员2 | accepted |
| D-011 | 2026-08-22 | C4/C8 同时记录合同 1.5x TTFT guard 和当前 scorer 的 concurrency-scaled guard，候选采用更严格的 1.5x team policy | 冻结 contract 与 scorer 实现存在差异，不能修改 scorer 或混淆实验接收标准 | 成员1 + 成员3 | accepted |
| D-012 | 2026-08-22 | 三名 agent 的启动 prompt 通过聊天线下发送，不提交仓库；本地开发环境和协作边界入库；成员2/3不并发写聚合 PROGRESS | prompt 含启动上下文但不应成为生产源码/合同的一部分；共享账号下并发修改聚合记录会制造冲突 | 成员1 | accepted |
| D-013 | 2026-08-23 | M2-O0 改为“成员2本地编写 oracle generator/schema，成员1在服务器 GPU1 logical lane 一次执行真实 checkpoint”；golden/manifest 落受控共享 artifact 路径，远程成员只接收批准导出的最小 bundle 或 manifest/schema/hash | Qwen3.5 没有现成 transformers runtime；完整权重/BF16 展开和长序列逐层 oracle 不应成为成员2个人电脑的前置条件；远程电脑不默认挂载服务器 JuiceFS | 成员1 + 成员2 | accepted |
| D-014 | 2026-08-23 | W4 loader 必须用 synthetic fixture 验证 K-packed weight、K-group scale 与 N-packed zero-point；token admission 以 checkpoint `text_config.vocab_size` 为边界 | 真实 tensor slice 禁止入库；zero-point 与 weight 的 pack 轴不同；本模型 model vocab 实测 248320，而 tokenizer vocab 实测 248044，image token 248056 仍合法 | 成员1 + 成员2 | accepted |
| D-015 | 2026-08-23 | 四个 GPU 只作为逻辑 lane；服务器所有真实 GPU/模型任务通过全局锁串行队列，优先级为 oracle → GPU0 base → GPU2 kernel → GPU3 bonus | 单账号/单工作树拓扑与最终方案原“四卡并行常驻”表述冲突；本地代码可并行，服务器证据不可并行占用 | 成员1 | accepted |
| D-016 | 2026-08-24 | ProtocolRuntime adapter 分层放在顶层 server crate；model crate 只暴露 checkpoint/session 所需的中立接口，避免 apxinf-model 反向依赖 HTTP/server trait | `TokenStream`/`ProtocolRuntime` 定义在 `src/server/service.rs`；保持模型库可复用并确保生产 adapter 真正增量 | 成员1 | accepted |
| D-017 | 2026-08-24 | strict `serve` 在真实 checkpoint-backed CUDA executor 接入前必须拒绝启动，不得把 synthetic callback、旧 CPU CLI 或 protocol stub 暴露为生产 runtime | 当前只有 synthetic executor control plane 和 transport；无真实 CUDA forward 证据时 fail-closed 比伪造 RUNTIME_READY 安全 | 成员1 | accepted |
| D-018 | 2026-08-24 | 真实 W4 projection adapter 必须保留 checkpoint scale 的 BF16 表示，packed weight/zero-point 以有界 raw bytes 上传；不得把完整 checkpoint 展开为 BF16/F32 副本 | GPU2 layer-0 `in_proj_qkv` 已在 CPU K/N-packed reference 对比通过；native scale 路径由 `3274f38` 固化 | 成员1 | accepted |
| D-019 | 2026-08-24 | GPU2 单投影验证只记录为 development evidence，不得标记 BASE_CORRECT/BASE_GOOD/FORMAL_READY，也不得提前占用 GPU0 | 当前证据仅覆盖 layer-0 W4 projection，full-attention/GDN/64-layer executor/serve 尚未接通；artifact manifest 已通过 sha256sum -c | 成员1 | accepted |

## 新决策模板

新增决策时追加一行，必须包含：ID、日期、具体行为、证据/理由、owner、accepted/rejected/pending。
