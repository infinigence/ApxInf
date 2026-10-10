# Serving stream 同步与内存校准记录

日期：2026-10-09。范围：串行 MLX worker 和离线校准工具。

本轮修复了资源结算中的 stream 同步边界，并新增可恢复的内存采样工具。请求预算仍使用原有固定预留；本轮没有批准新的内存估算，也没有完成 batch 或缓存阶段的验收。

## 运行时修复

固定 MLX 0.32.1 的无参 `synchronize()` 只等待当前默认设备的默认 stream。固定 MLX-LM 0.31.3 使用独立生成 stream，生成器返回外部调用者时，默认 stream 已可能恢复。因此，只在清理时调用无参同步，不能证明生成 stream 的全部工作结束。

`MLXRuntime` 现在分别记录加载、请求和生成 stream。首次 `prompt_progress_callback(0, total)` 位于生成 stream 上下文内，adapter 在此记录生成 stream。最终 progress 回调位于该上下文外，不会覆盖已有记录。结算先等待这些 stream，再释放请求缓存和清理 allocator；同步失败仍向上层返回失败。

新增 `inspect_memory()` 只允许执行 owner 调用。它在同步后记录 allocator active/cache/peak、峰值 reset epoch、adapter 消费位置和逐层缓存元数据。缓存 payload 与 allocator 分开保存，不进行相加。未知 offset 保留 null，避免为 recurrent cache 编造序列位置。

生成器可能在 yield 前计算后续 token。回调位置、实际 yield 数量和物理 cache offset 因此分别记录。采样会改变同步和分配重叠，不能作为未插桩服务延迟或通用峰值上界。

实现依据是本地固定版本的公开 API 与源码行为核对。没有复制上游实现、测试或示例。worker wire 保持 `apxinf-worker/2.0`，v1 接口保持原行为。adapter 文件摘要改变，因此本轮实际加载取得新的模型身份记录。

## 校准工具

入口：`benchmarks/serving/memory_calibration.py`。格式与字段见[内存校准规范](memory-calibration-v0.1.md)，调用方式见[测量工具说明](../../benchmarks/serving/README.md)。

- 在加载模型前保存完整用例计划，逐项保留 completed、blocked、failed 或 skipped 结果。
- 父子进程共同持有继承的 Metal 锁描述符，只有本任务创建的子进程接受清理信号。
- 在加载前后、prefill 回调和每个 output yield 检查真实宿主压力。
- 独立记录 allocator、逻辑缓存 payload、进程 lifetime peak RSS 和系统 swap。
- 将样本逐条 flush 到 JSONL。中断后只恢复完整记录，不据此推断请求已成功完成。
- 先关闭生成器，再结算请求。结算失败停止后续用例。
- 分开记录请求结算和进程回收。未确认进程退出时，父进程只写自己的状态文件。
- 超时即使随后退出码为 0，仍是失败。摘要完整性检查拒绝缺失身份、未结算或未完成的成功报告。

工具只输出原始观测，不自动拟合或批准预算。默认计划覆盖 chunk、cache 容量增长和 context 边界；显式形状子集不构成完整覆盖。零输出用例沿用服务行为，不调用生成器。

## 本次实机观测

原始目录：`/private/tmp/apxinf-serving-calibration-8JtDPj`，其中 `subset/` 保存本次运行。该目录是本机实验记录，不是已发布或归档的数据集。

环境为 Mac16,10、16 GiB，Python 3.14.3、MLX 0.32.1、MLX-LM 0.31.3。模型使用已有 Qwen3.5-2B BF16 bundle，没有下载模型、切换精度或调整 wired-memory 限制。

配置为 context 16384、prefill chunk 256、output batch 1、allocator guideline 10 GiB。显式输入长度为 1、255、256、257，输出 allowance 为 1 或 8；另含零输出、prefill stop 和 output stop，共 11 个用例。

| 观测 | 加载前 | 加载后 |
| --- | --- | --- |
| 宿主 pressure | normal，dispatch 1 | warning，dispatch 2 |
| allocator active | 未加载，不适用 | 3,763,655,368 bytes |
| allocator cache | 未加载，不适用 | 4,086 bytes |
| allocator peak | 未加载，不适用 | 3,763,655,400 bytes |
| 请求 cache payload | 未加载，不适用 | 0，无请求层缓存 |
| 进程 lifetime peak RSS | 28,278,784 bytes | 988,676,096 bytes |
| 系统 swap used | 6,070,927,360 bytes | 6,599,802,880 bytes |

结果是 `blocked`，准确失败阶段为 `loaded`。这次模型已完成加载；压力门阻止了后续种子准备和生成。11 个用例全部保留为 skipped，加载前后两条样本均保留。最终 runtime settlement 成功，probe PID/PGID 76960 退出码为 1，进程组确认无成员，未发送终止信号。

RSS 是进程历史最大值，并非当前物理占用；系统 swap 包含其他应用，差值不能全部归因于模型。本次没有 TTFT、decode、长上下文峰值或真实生成 stream 的实机验证结果。

模型 revision：`8cea164a6e87242bed9ffdc71dccb09eba74b71f6817b21d93cea6ab387c1682`。校准主文件 SHA-256：`d445d52e284180f28be0fdbba8581c23b3c2361e2be9a086d0f125970a1dd317`。工件还分别保存 host/process helper 摘要、模型 manifest 和输入计划。

## 验证与后续边界

CPU 回归覆盖 stream 捕获、回调上下文切换、两次同步失败、未知缓存元数据、压力拒绝、EOS、取消、清理失败和中断证据恢复。测试使用手写 fake runtime，不等同于实机生成验证。

| 验证 | 结果 |
| --- | --- |
| Rust serving | 90 passed，1 ignored（真实压力传感器测试） |
| Python serving | 146 项中 140 passed，6 skipped（live HTTP） |
| v1 Python 回归 | 18 passed |
| 新增校准主流程测试 | 25 passed，已含在 Python serving 总数中 |
| 新增父进程恢复测试 | 10 passed，已含在 Python serving 总数中 |
| host memory helper | 13 passed，已含在 Python serving 总数中 |
| 原始实机工件核对 | journal hash/count、用例计数、tool hashes 和进程退出记录一致 |
| 英文规范、计划、依赖和工具说明 | ASD 结构检查 0 violations，未执行完整官方词典审核 |

Python 回归最初在 sandbox 下因 loopback 和进程读取权限失败，随后在允许这些测试操作的环境中通过。完整性门新增后，一项成功测试样例缺少输入摘要；补齐原始样例后，最终整组回归通过。Rust、Python 和文档检查日志位于上述原始目录。未运行新的 live HTTP 服务或生成吞吐实验。

真实生成矩阵仍需在可用宿主内存下完成。预算估算还需要独立契约、明确形状范围、安全余量和 held-out 验证。当前加载样本不足以替代这些工作，也不满足后续 batch 或 exact session 的内存验收门。
