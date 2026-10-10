# Serving 内存压力准入与预算加固

日期：2026-10-09。范围：当前串行服务的 host pressure guard、预算校验和峰值统计。

本轮增加了运行期间的系统压力准入控制。它补充固定容量预算，不代表已经完成按请求内存估算，也不构成吞吐提升结论。
worker wire 保持 `apxinf-worker/2.0`；模型、精度、MLX 参数和现有 v1 接口没有改变。

## 实现结果

默认 CLI 策略为 `--host-pressure-policy macos`。服务每秒直接读取 macOS 内存压力，不创建采样子进程。
warning、critical、未知值、读取失败，以及年龄达到三秒的样本都会阻止新准入。
首次实际采样为 normal 时允许初次加载；出现受阻状态后，需要相隔至少一秒的连续正常采样恢复。
恢复期间的过期样本会重置计时。显式 `disabled` 会公开展示，不能自动回退或伪装成 normal。

| 边界 | 行为 |
| --- | --- |
| 入队前 | 返回 `503 capacity_unavailable`，不增加请求终态计数 |
| 等待请求开始准备前 | 再次检查，受阻则产生一次 failed 并结算；已有 deadline／cancel 优先 |
| 准备结束、发送 submit 前 | 当前压力仍受阻则拒绝生成；准备操作先完成结算 |
| 已开始生成 | 保留原有 deadline、取消和回收规则，压力本身不释放资源或改变终态 |
| 初次模型加载 | 开始前与加载后检查；受阻时不发布 ready，已创建的 worker 需要确认回收 |
| 命令历史轮换 | 先 drain／reap 旧 worker，再等待压力恢复；队列继续独立处理过期和取消 |
| 轮换时关闭 | 结束等待，不误记 worker fault；真实 worker／reap 错误仍传播 |
| 优雅关闭 | 停止接收新请求，保持压力采样直到已有操作结束，避免人为制造 stale |

`/readyz` 新增 `worker_available` 和 `host_pressure`，分别表达 worker 生命周期和 host 准入。
HTTP readiness 要求两者同时允许，`/healthz` 不受压力影响。
指标采用固定 state、stage 和 reason 标签，覆盖样本年龄、读取错误和拒绝次数。
压力采样不在持锁状态下写日志，避免 stderr 阻塞或关闭导致准入路径阻塞／panic。

预算参数现在由 `Config` 统一检查，库调用与 CLI 使用相同边界。
预算和单序列预留量都必须处于 `1..=9007199254740991`，预留量不能大于预算。
resident 加预留量使用检查溢出的运算，不再让 `u64::MAX` 在 release 中回绕通过。

`apxinf_worker_peak_bytes` 现在保留 ready 报告的 peak 与 active＋cache 中较大者。
所有通过事件顺序检查的 terminal 都能更新峰值，包括阻塞 submit 后的停止路径，以及 terminal 后发生清理故障的路径。
这个指标仍是已收到的 allocator 观测最大值，不是当前 RSS、物理内存使用量或完整分配上限。

## 验证记录

原始日志目录：`/private/tmp/apxinf-serving-pressure-w24rk2kl`。

| 检查 | 本轮结果 |
| --- | --- |
| Rust 全套 | 90 passed，0 failed，1 ignored；20.15 秒 |
| 默认忽略的真实传感器检查 | 单独执行，1 passed，实际读取 normal |
| Python serving | 88 项中 82 passed，6 项 live HTTP skipped |
| v1 Python 回归 | 18 passed |
| Release 构建 | 通过，无编译 warning |
| Rust 格式 | 通过 |
| 四份英文规范／依赖记录的默认结构检查 | 0 violations，baseline 0，没有关闭规则 |

新增回归覆盖：整数边界／溢出、峰值漏报、压力恢复／过期、入队和派发拒绝、主动生成的资源所有权、轮换恢复及关闭。
一个 3.3 秒的准备操作在 shutdown 后仍完成，验证压力监控不会过早停止。
另一个回归关闭自有子进程的 stderr 管道，确认压力判断和指标读取仍成功。

真实模型启动使用已有 Metal 独占测量锁，没有与锁内的其他任务并行。
两次均保持默认压力策略、相同模型和相同执行参数。
两次启动都遇到真实 warning，返回 `Host memory pressure prevents model loading: warning`，未获得 HTTP readiness。
第一次拒绝后的三次只读采样均为 normal，随后重试仍被 warning 阻止，说明正常采样不能保证后续启动检查仍为 normal。

两次 wrapper 均以 1 退出，自有进程组 `47964`、`56486` 均确认无剩余进程。
没有关闭压力保护、提高 wired-memory 限额、改变精度或替换模型来绕过拒绝。
因此本轮不把六项 live HTTP 标为通过，也不提供新的生成性能或模型质量结论。
启动日志没有区分哪一次压力检查触发拒绝，不能据此声称完全没有发生模型分配。

本次尝试的 release SHA-256：

```text
5b2c67a085b2df8e7d7bf03a5ffe3b1b9d7eb305b2373addcb48c98634f6cb3d
```

记录环境为 Mac16,10、16 GiB、macOS 27.0.1。
启动间隙的 `vm.swapusage` 报告 used `6676.19M`；该时刻 pressure 为 normal。
这直接说明 normal 不等于无 swap，也不能证明当前模型具有足够物理余量。
桌面后台负载没有隔离，不能把系统 swap 变化全部归因于 ApxInf。

英文检查仅覆盖技能支持的结构规则，没有执行完整受控词典审核，也不表示标准认证。
Clippy 在当前工具链中仍不可用，本轮未安装。

## 尚未完成的内存门

单序列预留量仍是固定的部署估计。服务尚未按 prompt、最大输出、混合层状态和执行参数计算完整内存上界。
下一步需要在固定模型 revision 与执行 profile 下，分别校准 resident、persistent state、执行余量和 allocator cache。
取消、连续请求、容量边界和冷／热请求都需要覆盖。
系统压力与进程 footprint 应独立记录，不能用 KV tensor 大小替代完整内存预算。

P2b 批处理和 P3 缓存仍需满足实测内存估计、固定延迟目标和质量门。本轮没有开启这些能力。

## 外部参考笔记

以下资料只用于核对接口和设计选择，不作为移植实现的指令。
本轮新增源码和测试均为手写，没有新增 package 依赖。

- Apple XNU 的 sysctl 返回 dispatch 值：normal=1、warning=2、critical=4，不能使用内部压力枚举值。
  [sysctl 转换入口](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/kern_memorystatus_notify.c#L1753)。
  此键以当前系统实际支持情况为准；未来系统若移除或改变它，服务将明确拒绝，不自动降级。
- 固定 MLX-LM 0.31.3 中，当前 Qwen3.5-2B 有六个 full-attention 层、十八个 linear-attention 层。
  BF16、batch=1 下，full KV payload 约为每 token 12288 B；linear 状态另计。
  [模型结构](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/qwen3_5.py)、[缓存接口](https://github.com/ml-explore/mlx-lm/blob/v0.31.3/mlx_lm/models/cache.py)。
  这些结构量不包含运行期间新旧数组并存、临时张量、buffer 分配和宿主开销，不能直接作为完整上界。

实现语义以 [host pressure contract](contracts-v0.1.md#host-pressure-admission) 和 [本地服务 profile](local-service-v0.1.md#host-pressure-policy) 为准。
