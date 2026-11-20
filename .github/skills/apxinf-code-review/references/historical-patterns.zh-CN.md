# ApxInf 历史 Review 模式（简体中文）

英文版：[historical-patterns.md](historical-patterns.md)。

本文件记录可复用的失败模式，不代表对所链接 PR 的永久结论。资料统计于
2026-11-02：83 个已合并 PR、55 条独立 inline review 根评论、41 条 thread 回复、
13 个非空 Review 总评，以及分布在 27 个 merged PR 中的 52 条 issue comment。
许多问题在合并前已经修复；Review 当前代码时必须重新验证。

## 反复出现的模式与证据

1. **设备问题经常来自异步顺序和生命周期，而不是 kernel 数学。** Process-wide
   allocation cache 在工作尚未结束时发布 buffer（[#78](https://github.com/infinigence/ApxInf/pull/78#discussion_r4044354105)）；blocking stream 迁移到 non-blocking 后，default-stream copy/clear 假设失效（[#117](https://github.com/infinigence/ApxInf/pull/117#issuecomment-6468340954)）；prepared graph 可能晚于其 execution 所依赖的 runtime 析构（[#88](https://github.com/infinigence/ApxInf/pull/88#pullrequestreview-5276445893)）。必须追踪完整 handle graph 上的 stream order 和 ownership。

2. **Prepared execution identity 经常缺少关键状态。** Padded capacity 掩盖了实际
   token shape（[#99](https://github.com/infinigence/ApxInf/pull/99#discussion_r4120225493)）；共享 plan cache 与 `thread_local` native state 范围不一致，且已退役的旧 JSON 调优路径中 fused/plain tactic 可能共享 key（[#35](https://github.com/infinigence/ApxInf/pull/35#pullrequestreview-5075611780)）。旧 DB 机制已不是 cuda-new 的当前契约，但其长期经验仍适用：Recipe 和 execution key 必须覆盖所有会改变 prepare/provider 选择的事实。

3. **全局开关会破坏实例配置语义。** Model config 被写入 global Atomic 和进程环境变量（[#99](https://github.com/infinigence/ApxInf/pull/99#discussion_r4120225504)）。如果 Backend 只能支持 process-wide policy，应拒绝冲突加载，而不是静默改变已有模型。

4. **Checkpoint identity 必须跨语言一致，并覆盖所有会改变行为的资产。** Rust/Python
   对 shard 的选择和排序规则不同（[#29](https://github.com/infinigence/ApxInf/pull/29#pullrequestreview-5074847186)）；Calibration identity 漏掉 processor/statistics 文件（[#101](https://github.com/infinigence/ApxInf/pull/101#discussion_r4121159495)）。需要唯一 canonical 规则和跨语言 test vector。

5. **Fallback 与 capability 必须由结构保证。** Fused 路径按硬件 shape 准入，却没有检查
   对应 Backend 是否编译（[#101](https://github.com/infinigence/ApxInf/pull/101#discussion_r4120106199)）；没有 fallback 的 operator 迫使 model 复制架构谓词（[#88](https://github.com/infinigence/ApxInf/pull/88#pullrequestreview-5276445893)）；vision weight 变成必需资产，破坏 text-only checkpoint（[#129](https://github.com/infinigence/ApxInf/pull/129#discussion_r4268194595)）。Capability 应由实际加载资产和可用 provider 推导。

6. **Safe public path 应在昂贵工作前拒绝错误 checkpoint/request。** Weight loader 使用
   indexing、`unwrap` 和 assertion（[#99](https://github.com/infinigence/ApxInf/pull/99#discussion_r4120225512)）；image prefill 在 vision tower 后才触发 KV capacity 错误（[#129](https://github.com/infinigence/ApxInf/pull/129#discussion_r4268194586)）；一个 validation error 会永久 poison 后续 DSpark 请求（[#112](https://github.com/infinigence/ApxInf/pull/112#issuecomment-6074838568)）。Validation error 应保持可恢复且无副作用。

7. **构建与 Feature 覆盖经常没有编译真正修改的 Backend。** CUDA 全量迁移的 hosted
   check 通过，但 `--features cuda` 无法编译（[#117](https://github.com/infinigence/ApxInf/pull/117#issuecomment-6468340954)）；Feature Unification 改变了无关 crate 的 enum variant 可用性（[#101](https://github.com/infinigence/ApxInf/pull/101#discussion_r4119947568)）；只含一个模型的 AOT bundle 会使另一个 CUDA crate 构建失败（[#107](https://github.com/infinigence/ApxInf/pull/107#issuecomment-6095939102)）。必须检查真实 Feature matrix、partial bundle 和 workspace unification。

8. **内存与隐藏 Hot Path 工作经常比 kernel 数学更重要。** 关闭 diagnostics 打印后仍执行
   D2H 和同步（[#78](https://github.com/infinigence/ApxInf/pull/78#discussion_r4055999271)）；prepared session 和重复 fused weight 常驻数 GiB（[#107](https://github.com/infinigence/ApxInf/pull/107#issuecomment-6095939102)）；fused winner 仍分配/tune decomposed 路径资源（[#41](https://github.com/infinigence/ApxInf/pull/41#pullrequestreview-5089082728)）。应按 profile/layer 统计 transient 与 retained resource。

9. **端到端证据能够发现 operator test 漏掉的问题。** FP64 oracle 通过时，完整模型输出仍随
   output buffer clearing 变化（[#78](https://github.com/infinigence/ApxInf/pull/78#issuecomment-5748251587)）；non-square vision 差异需要受控逐层定位（[#129](https://github.com/infinigence/ApxInf/pull/129#discussion_r4268194606)）；一次 Backend migration 在没有 old/new model oracle 的情况下引入并修复了多个语义回归（[#88](https://github.com/infinigence/ApxInf/pull/88#pullrequestreview-5276445893)）。迁移后应使用真实 checkpoint 和精确 route 做端到端比较。

10. **跨层 option 传递与边界语义容易静默回归。** Planning-step override 被保存和展示，
    但没有传入执行（[#78](https://github.com/infinigence/ApxInf/pull/78#discussion_r4056011288)）；最后一个预算位置生成 EOS 被误报为 truncation（[#129](https://github.com/infinigence/ApxInf/pull/129#discussion_r4320978650)）；新增 HIP alias 漏过 variant admission（[#164](https://github.com/infinigence/ApxInf/pull/164#issuecomment-6410487397)）。默认值、alias 和非默认 override 都要通过公共入口验证。

11. **Scope 与层级卫生减少未经审查的生产面。** Review 从一个 PR 中移除了 probe、实验报告、
    无调用者的四层 operator stack 和不可达 perception scaffold（[#78](https://github.com/infinigence/ApxInf/pull/78#discussion_r4044289465)）；CUDA perf PR 中无关 Llama 修改被回退（[#107](https://github.com/infinigence/ApxInf/pull/107#discussion_r4236999862)）；Vendor kernel 来源和 adapter/kernel 层级归属需要补正（[#130](https://github.com/infinigence/ApxInf/pull/130#issuecomment-6263373023)）。每条新路径都必须追踪到 production caller，并遵循仓库 ownership。

12. **架构检查和测试也可能错误地固定临时例外。** 一次 migration 把 guard 从仍受支持的 legacy
    backend 上移走，并通过 source-text test 固化 registry bypass（[#88](https://github.com/infinigence/ApxInf/pull/88#pullrequestreview-5276445893)）。应对比 base 上的 policy/check script，并优先使用行为测试而不是 source grep。

## 如何使用历史信息

历史模式用于决定 Review 深度，不能因为旧 PR 出现过某问题，就直接报告当前 PR 有同样缺陷。
必须在当前固定 comparison 上重新证明不变量被破坏，给出具体触发条件和影响；如果当前设计已经
关闭旧失败模式，也应在报告中明确记录。
