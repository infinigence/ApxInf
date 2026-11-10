# 任务交接与 Review 流程

## 任务生命周期

每个任务都要有唯一 ID 和状态：

~~~text
planned -> active -> review -> integrated -> done
                    \-> rejected
active  -> blocked -> active
~~~

blocked 必须写明阻塞条件、已尝试命令、需要谁提供什么输入；不能用它隐藏“还没开始”。
rejected 必须指向回滚 SHA 或替代方案。

## 开工包

作者从 [task-spec.md](../templates/task-spec.md) 建立开工包，至少填写：

- task ID、owner、角色、分支；
- 背景、目标、非目标；
- 允许修改的目录和明确禁止的目录；
- 依赖的接口、合同条款和前置 SHA；
- 验收命令、成功条件、停止条件；
- 预期 artifact 和回滚方式。

任务没有开工包，不进入 active。

## PR 交接包

提交 PR 前，作者按 [pr-checklist.md](../templates/pr-checklist.md) 完成自检。PR 描述中必须能回答：

1. 这个改动解决哪个 task，改变了哪个接口或性能变量？
2. 哪些文件属于 owner 范围，是否触及共享文件？
3. 在本地和服务器分别运行了什么命令，结果是什么？
4. protocol、correctness、reliability、显存和 CV 证据在哪里？
5. 如果候选失败，成员1如何一条命令回滚？

## 成员1的 review 顺序

1. 先看合同和范围：确认没有修改 evaluator、没有硬编码答案、没有外部 fallback。
2. 再看短反馈：格式化、编译、单元测试、stub/oracle smoke。
3. 再看行为：protocol gate、错误映射、EOS/usage、取消、失败恢复。
4. 最后看 GPU：固定 UUID、显存账本、A/B、CV、artifact 和 clean checkout 重放。
5. 只有所有门禁通过才合并；性能优化若无端到端证据保留在 exp/*。

## 接口交接

成员2向成员1交付 runtime adapter 需求时，必须给出类型、生命周期、错误和取消语义，
不直接依赖内部 CUDA 类型。成员1接线后由成员2重新跑协议负控。成员3的 kernel 候选
同理，必须提供稳定的 FFI/API 和 feature-off 默认值。

## 进度记录

成员2、成员3每天结束时更新自己的 PR 或 task-specific progress log：完成项、下一步、
阻塞、最新 SHA、最新 artifact；不要并发修改聚合的
[PROGRESS.md](../records/PROGRESS.md)。成员1合并后更新聚合进度、集成状态和服务器
replay 结果；成员3仍把每轮实验索引追加到 [EXPERIMENTS.md](../records/EXPERIMENTS.md)，
冲突时由成员1合并记录。

## 失败实验和事故

Xid、OOM、NaN、fallback、health 未恢复、协议污染、数据/凭据误入 Git、结果不可重现，
必须写 [incident-record.md](../templates/incident-record.md)。单纯性能变慢但没有可靠性
影响，写 [experiment-record.md](../templates/experiment-record.md) 并标记 rejected。
