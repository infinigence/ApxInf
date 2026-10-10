# Serving 生成阶段公开截止时间修复

日期：2026-10-09。范围：串行 Rust coordinator 的生成阶段、控制写入与清理等待。

本轮修复了公开超时只发送 cancel、却继续等待 worker 清理的问题。截止时间现在可以独立结束公开请求；资源预留仍保留到 `resources_released` 校验成功或 worker 进程被回收。成功响应继续等待资源结算，不能通过提前成功来规避 deadline。

## 原问题与复现

原生成循环将 deadline 检查放在 `!control_sent` 分支内。到期只发送取消，随后仍处理输出并等待清理；最终公共状态来自 worker terminal。若先发过 `stop_generation`，之后甚至不再检查 deadline。

worker 可以合法地先报告 `completed`，执行清理后才处理晚到的取消。此时它返回 `already_terminal` 是正确行为，但 Rust 不能用这个既有 worker 终态覆盖自己的公开 deadline。另一个结果是：公开请求一直等到清理超时，再收到 worker fault。

五个原创 CPU 回归使用真实 stdin/stdout 管道和完整协议帧。它们将公开 deadline 设为 180 ms，将迟到事件或清理设为约 800 ms，并要求 450 ms 内收到公开结果。旧实现的五项均因公开结果迟到而失败；修复后全部通过。这个容差用于测试调度，不是实机 SLO 或延迟性能结论。

## 当前行为

- 公开终态与控制发送状态分别保存。已发布的公开结果不可再改写。
- `accepted` 前到期，通过 admission 结果返回错误；晚到的 accepted 不会变成成功。
- `accepted` 后到期，使用预留 final-event slot，即使内容队列已满也能保存错误。
- 已发送 stop 控制不妨碍公开到期，每个 attempt 仍最多发送一条 stop 或 cancel。
- 取消发送等待 accepted，避免控制先于 attempt 注册。已观察到 terminal 时不再发送多余控制。
- 控制写入等待期间仍检查 deadline。首次停止或公开 expiry 决定 20 秒清理宽限，后续事件不重置它。
- 公开 expiry 后继续校验迟到事件，并记录 worker timing、peak 与 cleanup；不再产生公开内容。
- 已知 parser 或 worker failure 在截止裁决时保留 failed。迟到故障会撤销 worker 可用性，但不重复记公开 outcome。
- 队列中的后续请求继续遵守原期限，只有前一个请求完成资源结算后才能执行。

独立复审还发现：当前 failed terminal 已通过校验，但解析其文本时仍可能把旧 terminal 状态传给 deadline 检查。现实现向该检查传递当前帧，保留已观察到的 worker failure。

## 契约与观测

本轮补齐了 canonical、serial、local 和 implementation plan 中的生成阶段规则。worker wire 保持 `apxinf-worker/2.0`，没有增加字段、改变 Python worker 行为或扩大每请求四条命令预算。

`apxinf_service_request_seconds` 仍在公开终态结束。`apxinf_cleanup_pending`、worker-terminal-to-settlement 和 public-terminal-to-settlement 继续反映后续清理。legacy `apxinf_request_seconds_sum` 仍以已结算 generation 为界，未静默改变口径。

HTTP 负载器等待客户端请求结束，不等于服务器资源已清理。其 metrics monitor 也会随 HTTP 请求完成停止。工具说明现明确这个边界；后续设备实验仍需另行观察 active、reserved 和 cleanup_pending。

## 验证

原始日志目录：`/private/tmp/apxinf-generation-deadline-6qzJe6`。这是本地实验记录，不是已发布数据集。

| 检查 | 结果 |
| --- | --- |
| 原实现的五项真实管道回归 | 5 failed，均复现公开结果迟到 |
| 修复后的相同五项回归 | 5 passed |
| 控制写入与公开终态 CPU 回归 | 6 passed |
| 完整 Rust serving | 101 passed，1 ignored（真实压力传感器） |
| Python 共享契约回归 | 6 passed |
| Release 构建 | `cargo build -p apxinf-serving --release --offline` 通过 |
| 格式与空白检查 | `cargo fmt --check`、`git diff --check` 通过 |
| 英文规范与工具说明 | ASD 结构检查 0 violations，未执行完整官方词典审核 |

回归另外检查了资源仍在预留、cleanup_pending 的变化、迟到 token 不进入公开输出、只计一次 expired，以及清理后可正常执行下一请求。测试失败时也会先关闭并回收自己创建的 CPU worker。

本次 release 二进制 SHA-256：`f60f8675fc8bc2df6eec2e0e34dbb74f388161bec8d52d63f6bb68039c46f944`。原始目录分别保存 red、修复后集成、控制 future、全量 Rust、Python 契约、构建与文档检查日志。

本轮没有重新运行 GPU、live HTTP 或模型质量实验，也不声称吞吐提升。真实内存校准仍缺生成阶段数据；上一轮加载后 pressure warning 的记录继续有效，但不能替代新的实机测量。
