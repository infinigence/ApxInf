# ApxInf Serving 生命周期加固记录

日期：2026-10-09。范围：串行服务的命令历史轮换、前处理结算、等待队列回收、关闭流程和终态计数。本轮保留 `apxinf-worker/2.0` 与全部既有 wire 字段，增加控制面生命周期政策。

## 修复内容

| 问题 | 本轮行为 | 验证重点 |
| --- | --- | --- |
| 命令历史达到 10000 条后 worker 故障 | 每次前处理保留四条命令余量，不足时先轮换 epoch | 旧进程退出后才加载新 worker；身份不变；不重放请求 |
| 前处理期间请求过期直接影响整个服务 | 公共请求先结束，执行占位保留至匹配响应或故障回收 | 晚到结果丢弃；后续请求不得提前执行；20 秒结算宽限 |
| stdin 阻塞使前处理无法及时处理 deadline | 写入期间继续检查请求 deadline 与取消，并设置 20 秒写入健康超时 | 未确认发送仍占位；无法安全结算时 fence worker |
| 已取消或过期的等待请求继续占队列 | 独立每 20 毫秒清理，入队前再清理 | 及时归还队列容量；剩余请求保持 FIFO；终态只计数一次 |
| 轮换后 readiness 可能沿用旧状态 | 发布新 worker 的完整 `ready` 快照 | 新 epoch 与新 memory 数据同时可见 |
| shutdown 与轮换完成并发时重新显示 ready | 在等待队列状态锁内发布可用性，关闭状态不可逆 | 替代 worker 加载期间关闭；并发发布与关闭后仍不可用 |
| Python 输入线程在 shutdown 后仍等待 stdin | 合法 shutdown 入队后结束输入读取 | 保持 stdin 开启也能正常退出；非法帧仍按原校验规则处理 |
| 服务缺少明确的宿主关闭入口 | 宿主显式调用 `Service::shutdown`，CLI 在 Ctrl-C 时调用 | 关闭准入与等待队列；活跃工作按原 deadline 结算；回收 worker |
| HTTP 结束后 runtime 可能先于进程回收退出 | 宿主通过 `Service::wait_stopped` 等待协调器完成 | 所有自有 worker 确认回收后才返回成功；协调器失败或无法回收自有 worker 时返回错误 |

公共终态指标分别统计 `completed`、`cancelled`、`failed` 和 `expired`，包括等待、前处理和 token counting 路径。成功轮换与 worker 故障另有独立计数。普通 worker 故障仍要求重启服务，不自动重放请求。

宿主不能仅依赖丢弃外部 `Arc<Service>` 来停止服务。显式 shutdown 保留活跃请求原有的 deadline 和取消规则；协调器在其结算后停止并回收 worker。宿主必须等待 `Service::wait_stopped` 再停止 async runtime；CLI 在 HTTP 正常结束或出错后都执行关闭与等待。

## 本轮验证范围

本轮以 CPU 状态测试和真实子进程管道测试检查生命周期。管道测试涉及进程、stdin/stdout、关闭和回收，但不执行 MLX 模型推理。未新增运行时依赖，未改变既有 v1 协议。

| 检查层 | 最终结果 |
| --- | --- |
| Rust 共享协议、网关与 supervisor 生命周期 | 56 项通过，0 项失败；耗时 20.19 秒 |
| Python serving 协议、worker 与测试 harness | 77 项中 71 项通过，6 项实机 HTTP 测试跳过 |
| 原有 Python v1 MLX 入口回归 | 18 项通过，0 项失败 |
| serving release 构建 | 通过，使用离线依赖 |
| 本轮 Rust 文件格式 | `rustfmt --check` 通过 |
| 英文规范结构检查 | 四份英文规范默认检查通过，0 violations，baseline 为 0 |

Rust 管道回归实际填充 9996 条命令记录，再执行请求并触发轮换。测试确认旧进程退出后才启动替代进程，且下一请求成功。其余回归覆盖替代身份不符、晚到准备成功与拒绝、管道写入阻塞、准备取消、等待队列回收，以及关闭竞态。

结算宽限测试实际等待 20 秒。它确认公共请求先返回过期，资源占位保留到 worker 被回收，且只记录一次请求终态。

关闭完成回归覆盖公共准备 deadline 后继续等待结算、完成通知发送端消失、worker 故障结果，以及关闭时替代实例仍在加载。启动后的校验失败和替代身份拒绝也显式回收进程；回收错误向上传递。最终 release 构建和格式检查均在这些修改完成后执行。

修复前的针对性检查复现了三个问题：准备超时使请求通道关闭；取消后的满队列仍返回 429；合法 shutdown 后 Python 因 stdin 缓冲锁退出，退出码为 −6。对应检查在修复后通过。

主要验证命令：

```sh
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk \
  cargo test -p apxinf-serving --offline
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk \
  cargo build -p apxinf-serving --release --offline
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s tests/python -p 'test_serving*.py' -q
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s tests/python -p 'test_apxinf_mlx_*.py' -q
rustfmt --check --edition 2021 \
  crates/apxinf-serving/src/supervisor.rs \
  crates/apxinf-serving/src/gateway.rs \
  crates/apxinf-serving/src/main.rs
python3 ~/.codex/skills/asd-ste100/scripts/ste-lint.py \
  doc/serving/contracts-v0.1.md \
  doc/serving/serial-profile-v0.1.md \
  doc/serving/local-service-v0.1.md \
  doc/serving/implementation-plan.md
```

Python serving 首次检查受沙箱的 `/bin/ps` 限制影响。放行测试对子进程的检查后，完整测试取得上表结果。Clippy 未运行：当前 Rust 工具链没有安装 `cargo-clippy`。

另一次探索检查误用了 `test_mlx*.py` 匹配范围和系统 Python，运行了无关的 mixed-quant sandbox 检查。32 项中出现 3 项失败和 7 项错误，原因是这些检查要求固定的子进程 CPython 环境。该结果不计为通过；本轮没有改动对应实现。随后使用正确的 `test_apxinf_mlx_*.py` 范围验证既有 v1 入口，18 项通过。

英文结构检查使用已安装 `asd-ste100` 技能，不放宽 baseline 或关闭规则。该检查不等于完整词典审核或认证；本文中文说明不声称符合 STE 英文标准。

## 与历史证据的关系

[既有验证报告](serving-validation-20261009.md)记录此前 Qwen3.5-2B、Claude Code 和 HTTP 并发实机结果。[Metal 参数记录](metal-results-20261009.md)记录此前参数矩阵和输出差异。本轮不覆盖或重新命名这些历史结果，也不把其测试数量累加到本轮。

本轮 CPU 与管道检查不建立吞吐、TTFT、ITL、完整内存占用或真实客户端兼容性的新增结论。新的 worker revision 与完整实机数据需要单独记录。

## 阶段边界

本轮收紧 P1a 可靠性门，不完成 P1b exact append 或完整 P1。连续批处理、自动前缀缓存、多模型驻留和系统内存压力反馈仍未由本轮交付。

批处理与缓存仍需先固定内存估算边界、延迟目标和模型质量门，再执行各自的正确性与收益验收。阶段定义见[实施计划](implementation-plan.md)，当前行为边界见[本地服务说明](local-service-v0.1.md)。
