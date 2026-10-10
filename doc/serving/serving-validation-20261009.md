# ApxInf Serving 实现与验证报告

日期：2026-10-09。验证对象是本地串行文本服务，以及初始 Claude Code 工具适配。完整架构见[系统方案](../serving-system-design-20261009.md)，运行方法见[本地服务说明](local-service-v0.1.md)。

## 当前交付

新增 `apxinf-serve`：Rust 负责 HTTP、SSE、校验、有界队列、deadline、输出解析和 worker 生命周期；独立 Python worker 通过固定版本 MLX-LM 的公开 API 执行模型。首个协议子集保持一个模型、一个活跃请求，后续请求进入有界队列。

公共接口包括 Anthropic Messages、token 计数、OpenAI Chat Completions，以及模型清单、健康检查、就绪状态和 Prometheus 文本指标。Claude Code 在自己的进程中执行工具，服务不执行工具。模型模板、分词与增量解码以 worker 为唯一执行方，Rust 和 Python 使用同一份原始 fixtures 检查协议。

所有新增产品代码、测试和 fixtures 均为独立手写。外部仓库用于研究机制与行为，未复制、翻译或移植其实现。新依赖及公开 API 边界见[依赖记录](dependencies.md)。

## 自动化验证范围

| 检查层 | 本次检查内容 | 当前结果 |
| --- | --- | --- |
| Rust 服务 | 共享协议、输入校验、SSE 终结、工具解析、背压、取消分类、worker 故障与回收 | 35 项通过 |
| Python 协议 | 原始 frame、事件序列、摘要和 Rust 互操作 | 6 项通过 |
| Python worker | prefill／decode 控制、deadline、重复命令、状态与资源结算、输出缓冲 | 19 项通过 |
| HTTP 与客户端 harness 离线检查 | 错误流不计成功、usage／goodput 计算、隔离环境、任务判定、进程目标与结算检查 | 31 项通过 |
| Metal 矩阵 harness | 参数与身份固定、输出比较、独占锁要求、仅清理自身进程 | 8 项通过 |
| 原有 Python MLX 路径 | 原 `mlx-serve` 与 `mlx-generate` 回归 | 18 项通过 |
| 原有 Rust MLX CLI | 现有 v1 CLI 回归 | 8 项通过 |

这些 CPU 检查不能代替真实模型推理。Python harness 另有六项真实 HTTP 测试，必须显式提供运行中的本地服务地址。原 Rust CLI 测试需要本机进程执行权限，受限环境中的权限失败与获取许可后的通过记录分开保留。

最终 release 服务的真实 HTTP 检查共 37 项通过，其中 31 项为上述离线检查，六项访问真实服务。两种实际流式生成、token counting、health／readiness／metrics 和错误响应均通过。日志位于 `http-live-20261009-final.txt`，不将这 37 项与离线 31 项重复相加。

新增回归特别覆盖了三类错误：客户端关闭连接却计为服务失败、内容队列满时丢失结束事件、XML 工具参数中的重复 JSON 键被静默覆盖。公共结果与 worker 结果分别记录，parser／worker 失败和 deadline 仍保留失败优先级。

已安装 ASD-STE100 skill 0.4.0。英文接口、计划、配置说明和测试指南采用统一术语与短句，并通过默认结构检查。该技能不含官方完整词典，结构检查不等于逐词审核或标准认证。中文说明不标为 STE 合规英文。

本次不完成完整 P1/P2 路线：HTTP 尚无 exact append、自动前缀复用、VLM、连续批处理、多模型驻留和自动 worker 重启。已有 v1 CLI 与会话接口保留原行为。接口明确拒绝当前不支持的采样、强制工具和严格 schema 生成选项。

## 环境与证据范围

| 项目 | 实际配置 |
| --- | --- |
| 主机 | Mac16,10，Apple M4，16 GiB 统一内存 |
| 系统 | macOS 27.0.1，build 26A434 |
| 模型 | 本地 Qwen3.5-2B BF16 bundle，MLX-LM 文本路径 |
| 模型 snapshot | `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| 运行时 | Python 3.14.3、MLX 0.32.1、MLX-LM 0.31.3 |
| 客户端 | 实际安装的 Claude Code 2.1.289 |
| 部署限制 | context 16384、output 2048、queue 16、单活跃请求 |
| 基线配置 | prefill 256，output event batch 1，greedy |
| 内存设置 | MLX guideline 10 GiB，活跃请求预留 1 GiB |

模型内容、模板、解释器、adapter 和固定运行时身份进入 manifest。Rust 独立校验本地文件内容后才发布 readiness。基线 worker revision 为 `7cca1fc79d7ba2fcda9c1ad2ff59a672b5a08498329d69f5db304f172b02206a`。基线二进制 SHA-256 为 `d1e117905cc063bd880b576f9b450afe85a7623e3e66da158b6bf13ebb37c153`；后续边界修复的二进制身份另行保存在实验记录。

实验期间持有现有 Metal 协调锁，模型按顺序加载，避免与使用同一锁的其他测试重叠。这不是整台桌面的 CPU、内存或温度隔离。测量前系统已使用约 3775 MiB swap；基线与补充客户端测试后约为 3631 MiB。同期系统 Swapouts 计数没有增加，但 Pageouts 与 Swapins 有变化，因此不能宣称完全无分页干扰。

原始 JSON、日志与随机任务目录保存在 `benchmarks/serving/results/`，默认不进入 Git。这里的报告保留可审查结论，原始数据用于复核。复现时必须保存新的硬件、模型身份和背景负载，不能直接复用本机数值。

## HTTP 并发基线

每档并发 32 个请求，输出上限 64 token，混合负载中每四个请求有一个长输入。实际输入约 32–33 token 与 1174–1175 token。每个并发客户端完成请求后才发下一条，属于 closed-loop 压测。总计 96/96 成功，usage 覆盖率 100%。

| 并发 | 成功 | 输出吞吐 token/s | TTFT p50 秒 | TTFT p95 秒 | E2E p50 秒 | TPOT p50 毫秒 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 32/32 | 20.37 | 0.222 | 1.212 | 2.860 | 41.77 |
| 2 | 32/32 | 20.29 | 3.335 | 4.199 | 6.045 | 41.98 |
| 4 | 32/32 | 20.35 | 9.856 | 10.258 | 12.491 | 41.83 |

TTFT 从客户端发起请求计至首个非空内容 delta，包含排队。E2E 计至完整消费响应。TPOT 使用 `(E2E − TTFT) / (output_tokens − 1)`，包含响应收尾。SSE 内容事件间隔单独记录，它不保证等于模型逐 token 时间。

实验预先设置的演示门槛是 TTFT ≤ 3 秒且 E2E ≤ 15 秒，并非已经承诺的产品 SLO。并发 1 时有 32/32 达标，goodput 为 0.318 请求/秒；并发 2、4 时各有 1/32 达标，约 0.010 请求/秒。32 个样本只能给出探索性尾延迟，不能据此保证 p95。

吞吐随并发基本不变，排队延迟明显增加，与单执行 owner 的设计一致。这个结果不证明 batch 性能，也不与其他引擎构成同条件排名。后续批处理验收应保留同样的交互负载，同时增加独立到达率和更长 prompt 的测试。

按 96 个 request ID 关联 worker 日志，最高 MLX 分配峰值为 4038199082 字节，约 3.761 GiB。模型加载后的 active allocation 约 3.505 GiB。这些数值不等于进程 RSS，更不等于全系统内存占用。当前内存预留是部署估计，MLX memory limit 是 guideline，尚无系统压力反馈控制器。

基线原始证据：`baseline-256-1.json`、`ready-256-1.json`、`service-release-256-1.log`、`environment-before.json`、`environment-after-baseline.json`。

## 本机复现

在仓库根目录启动已验证的本地模型：

```sh
target/release/apxinf-serve \
  --model /Users/haiyan-mini/Downloads/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc \
  --model-id apxinf-local \
  --python .apxinf/toolchains/mlx-lm-0.31.3-copies/bin/python \
  --port 8080
```

实际客户端检查：

```sh
python3 benchmarks/serving/claude_code_tasks.py \
  --base-url http://127.0.0.1:8080 --model apxinf-local \
  --tasks read edit check cancel --repeats 1 \
  --max-context 16384 --max-tokens 512 \
  --output-dir benchmarks/serving/results/claude-repeat
```

输出目录使用新的名称以保留历史结果。如其他测试使用 Metal，先按[测量指南](../../benchmarks/serving/README.md)协调模型加载和测量的独占时段。测试 harness 不修改全局客户端配置。

## Claude Code 与服务边界

初次实机测试中，读文件、修改函数、运行本地检查三个任务全部通过。工具证据分别为 Read、Read+Edit、Read+Bash；不能只输出预期答案来通过任务。耗时分别为 6.856、11.521、9.036 秒，包括客户端启动、模型调用和工具执行。

最终 release 构建再次通过三项工具任务，耗时分别为 6.142、11.674、9.038 秒。真实 Claude Code 取消测试也通过，取消计数增长，活跃请求、排队请求与预留容量全部归零。原始结果保存于 `claude-20261009-final/report.json`，没有覆盖初次发现缺陷的失败记录。

补充测试实际验证了客户端 context window 为 16384。取消测试发现并修复了一个需要回归的问题：worker 已取消并归还资源，但关闭连接被误归类为输出队列满，公开指标将取消计为失败。最终重跑结果与全部工具记录以[Claude Code 验证报告](claude-code-validation-20261009.md)为准。

独立生命周期检查也通过了断开连接取消和队列满两种场景：一个请求执行、16 个请求等待时，额外请求获得 HTTP 429 `queue_full`。关闭测试拥有的连接后，资源归零且 readiness 保持 200。证据保存于 `lifecycle-20261009-final.json`。

空闲 worker 故障检查通过：仅向经过 PID、父进程、启动身份、epoch、模型路径和监听端口验证的测试 worker 发送 SIGTERM。约 51 ms 后 `/readyz` 返回 503，`/healthz` 保持 200，网关 PID 不变，active／queued／reserved 全部为零。进程检查确认 worker 已退出并回收。证据保存于 `worker-failure-20261009-final.json`。该结果证明当前故障隔离路径，不代表自动重启或请求重放。

测试结束后已停止本次创建的网关、worker 和持锁启动器，释放设备供其他任务使用。`final-cleanup.json` 保存退出检查。当前没有保留常驻服务，按前述本机复现命令即可启动。

测试使用隔离配置目录、仅当前进程的本地地址与占位 API key，没有修改全局 Claude Code 配置。任务限制了可用工具与路径。Anthropic 不正式支持非 Claude 模型；这组小任务证明所测版本和路径的可用性，不代表完整编程能力或全部客户端特性兼容。[官方网关说明](https://code.claude.com/docs/en/llm-gateway)

## Metal 方向与下一阶段

M4 使用统一内存，模型和状态与桌面应用竞争同一物理容量。Qwen3.5-2B 本地配置包含六层 full attention 和十八层 linear attention；按 BF16 估算，前者 KV 增长为每 token 12288 字节，16384 token 时约 192 MiB。循环状态、卷积历史和 prefill 临时数组需要单独计量。[硬件依据与计算边界](metal-experiments.md)

优先验证三个方向：

1. **按延迟预算选择执行配置。** 分开约束首输出、持续输出和取消结算，比较 prefill chunk 与 host 输出聚合的代价。输出聚合只合并协议事件，不能据此声称减少了 Metal command buffer。
2. **建立混合状态的内存估计。** 区分随 context 增长的 KV、循环状态、chunk 临时数组、模型权重和 allocator reserve，并加入系统压力反馈。
3. **为 Agent 前缀提供可恢复快照。** 同时保存 attention、循环、卷积和位置状态，先证明恢复正确性与净收益，再接入缓存感知调度。不能只保存 KV 就宣称完整恢复。

完整研究矩阵、候选配置和验收门见[Metal 实验方案](metal-experiments.md)。这些方向是待验证的 ApxInf 设计候选，不声称技术首创或已获得性能提升。

## 九组 Metal 参数实验结论

完整结果见[Metal 参数实测](metal-results-20261009.md)。prefill 64／256／1024 与输出聚合 1／4／8 共九组，每组 16 个测量请求，输出上限 32 token。144/144 请求与 18/18 warmup 成功，但跨 prefill 的输出一致性检查没有通过，矩阵驱动因此返回退出码 2。这是结果一致性门的失败，不是 HTTP 请求失败。

固定 prefill 时，三种输出聚合的 16 个可见输出哈希完全一致。跨 prefill 时，部分长输入出现差异。哈希不包含完整 token ID，不能判定差异原因或任务质量影响，更不能据此宣称 token parity。

所有组的吞吐约为 17.30–17.97 token/s。输出聚合 1 时，内容事件间隔中位数约为 41–43 ms；聚合 4 时约为 165–167 ms，聚合 8 时约为 328–331 ms。当前短输出负载没有显示足以补偿交互延迟的聚合收益。

实验中系统 swap 使用量从约 3607 MiB 升至 5106 MiB，且顺序固定，因此不能把百分之几的吞吐差异当作可靠优化收益。32-token 实验与前述 64-token 基线的工作量不同，两者吞吐不作直接回归比较。

保留默认 prefill 256、输出聚合 1。后续优先级是：完成可中断的连续批处理适配，验证完整混合状态的前缀恢复，再增加系统压力感知的内存控制。任何改变 prefill 的候选配置都需要先通过 token 与任务质量检查，并在更受控、平衡顺序的实验中复测。
