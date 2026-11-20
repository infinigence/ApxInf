# Review 报告格式（简体中文）

英文版：[report-format.md](report-format.md)。

报告开头记录版本元数据，以便机械判断报告是否过期：

```yaml
---
repo: infinigence/ApxInf
pr: 123
author: login
base_sha: full-sha
head_sha: full-sha
merge_base_sha: full-sha
changed_files: integer
additions: integer
deletions: integer
reviewed_at: RFC3339 timestamp
coverage: complete | qualified
coverage_notes: 空值或简洁的覆盖限制
verdict: block | needs-work | lgtm
---
```

只保留能够增加信息的章节：

1. 摘要与动机。
2. 架构或执行路径变化。
3. 改动模块，以及受影响的模型、设备、格式和 API。
4. 验证证据：测试、CI、Benchmark 和尚未完成的资格验证。
5. 按优先级排列的 Findings。
6. 总体影响和 Verdict。
7. head 变化时的增量复审结果。

每条 Finding 包含：

```text
[P1] 使用祈使句描述问题的标题
文件：repository/relative/path:changed-line
触发条件：具体输入、平台、配置或并发时序
证据：固定 comparison 中相关的控制流或数据流
影响：可观察的失败或兼容性后果
修复方向：简洁、可执行的修复或验证建议
```

删除代码应锚定在原始侧。无法检查 patch 或精确源码时必须明确说明。coverage 因
文件缺失、commit 无法访问或行数未对账而受限时，不得声称“没有 findings”。

GitHub 评论省略报告元数据，但保留标题、触发条件、证据、影响和修复方向。不要发布
只有摘要的评论，也不要在 PR stack 的每个 PR 上重复同一条评论。继承问题应评论在
最初引入问题的根 PR，并在私有报告中注明受影响的后续 PR。
