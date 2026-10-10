# Serving 审查问题修复记录

日期：2026-10-10。范围：本轮审查确认的六项问题。

本轮手写实现、回归测试和共享样例。没有新增依赖、导入外部实现或使用代码生成器。现有固定输入快照和父子计划比对继续保留。

## 修复结果

| 问题 | 最终行为 | 回归证据 |
| --- | --- | --- |
| 接受前或终止后取消未启动回收计时 | 首次观察到取消便建立 20 秒宽限期，与控制命令发送条件分离 | 请求 deadline 为 60 秒，模拟 worker 阻塞 22 秒；两项旧实现失败的测试现在均在宽限期后完成故障恢复 |
| Rust 静默舍入超大整数 | 返回结果前检查原始整数文本，拒绝超出正负安全范围的整数 | Rust 和 Python 消费相同的 17 个原始文档样例；XML 与 JSON 工具参数均拒绝溢出 |
| 批量输出异常产生未上报 token 的尾文本 | 模型失败时先报告已成功解码的缓冲 token；解码失败或输出拒收时抑制 decoder 尾文本 | 覆盖模型错误、解码器先改变状态再抛错、慢消费者及管道断开 |
| 覆盖分析接受矛盾的压力证据 | 校验状态、原始 dispatch 值、错误及 macOS 压力策略的一致性 | 拒绝 critical 原始值配 normal 状态、读取错误配 normal 状态、布尔值和缺失字段 |
| 损坏校准报告阻断父进程结果保存 | 验证结构，保留错误报告原始字节，使用父进程完整计划恢复；最终报告写入失败仍尝试保存进程结果 | 覆盖数组、空对象、null、不完整 JSON、损坏的嵌套结构、缺少 case 以及报告写入失败 |
| 离线时间戳误用协议安全整数上限 | 生产端和分析端均采用非负 u64 纳秒整数 | 接受 2^53 及 u64 边界，拒绝负值、越界值、布尔值和浮点值 |

取消修复保留了原有的控制规则：接受前不发送取消，终止后不追加控制，每个 attempt 最多发送一次停止控制。后续事件不会重置宽限期。只有有效资源结算或确认进程回收，才能释放执行位和预留。

数值修复保留显式小数、指数、字符串及既有负零行为。Rust 额外扫描原始 JSON 文本，输入仍受现有 1 MiB 上限约束。离线校准时间戳不属于 worker 协议字段，其 u64 范围单独定义。

worker 仅将解码成功的 token 纳入待发送批次。模型失败后，成功交付的批次计入 usage。解码失败的 token 不计入 usage。已有执行失败不会被随后发生的慢消费者错误覆盖。管道失败仍交由进程故障恢复处理，不虚报资源释放。

错误报告保存在 `calibration.invalid-<UUID>.json`，前提是文件系统允许保留操作。父进程记录保留路径或保留错误。恢复 journal 不会凭空补出模型身份、输出数或成功结算。文件系统拒绝所有写入时，工具无法保证结果持久化。

## 验证

完整测试和取消回归的日志保存在 `/tmp/apxinf-serving-fixes-20261010`。

| 检查 | 结果 |
| --- | --- |
| 取消缺陷修复前回归 | 两项预期失败，原 worker 到约 22 秒才结算 |
| Rust Serving 完整测试 | 117 通过，1 忽略 |
| Python Serving 完整测试 | 232 项中 226 通过，6 跳过 |
| Rust 格式检查 | `cargo fmt -p apxinf-serving -- --check` 通过 |
| 四份修改后的英文规范文档 | STE 结构检查 0 项问题 |
| 独立复审 | 取消计时和 JSON 原始数值检查未发现新增运行时问题 |

Rust 忽略项为真实主机压力传感器检查。Python 跳过项需要已运行的真实 HTTP 服务。本轮没有加载模型或运行 GPU，不建立新的吞吐、TTFT、内存预算或模型质量结论。

首次完整 Rust 测试受沙箱限制，8 项诊断测试无法检查自有进程组。授权环境中的完整重跑通过。两次结果分别保留在 `rust-tests.log` 和 `rust-tests-authorized.log`。测试拥有的子进程保留明确回收路径。

STE 检查只覆盖结构规则。没有进行完整词典审查，也不构成认证。本文为中文讨论记录，不声明为 STE 英文。

## 对应实现和规范

- [请求生命周期](../../crates/apxinf-serving/src/supervisor.rs)和[取消回归](../../crates/apxinf-serving/src/supervisor/generation_deadline_regressions.rs)。
- [Rust JSON 契约](../../crates/apxinf-serving/src/contracts.rs)和[共享样例](../../tests/fixtures/serving/serial-v0.1.json)。
- [Python worker](../../python/apxinf/apxinf/serving/text_worker.py)和[worker 回归](../../tests/python/test_serving_text_worker.py)。
- [覆盖分析器](../../benchmarks/serving/memory_coverage.py)和[校准工具](../../benchmarks/serving/memory_calibration.py)。
- [主契约](contracts-v0.1.md)、[串行 profile](serial-profile-v0.1.md)、[校准规范](memory-calibration-v0.1.md)和[覆盖规范](memory-coverage-v0.1.md)。

本轮没有改变协议版本或 wire 字段，也没有推进 sessions、批处理或缓存阶段。真实模型回归及生成内存校准仍属于后续验收工作。
