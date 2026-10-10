# 2026-10-10 内存实测记录

本记录纳入 `smoke`、`stage01` 和 `stage02` 的原始记录。首次尝试因主机压力阻断；stage01 完成五个 case，其中三个达到计划边界；stage02 再次在加载后被压力阻断。尚未建立或批准 serving admission memory envelope。

原始文件保存在仓库外：`/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ`。下文的绝对链接依赖该目录仍然存在。SHA-256 用于复查文件内容，不能单独证明来源真实性。

本记录按各次原始文件核对。后续阶段结果需要另行追加，不能从计划文件推定已经执行。

## 实测身份和固定配置

| 项目 | 记录值 |
| --- | --- |
| 主机 | Mac mini，`Mac16,10`，Apple M4，10 核 GPU，16 GiB 内存 |
| 系统 | macOS `27.0.1`，build `26A434` |
| 模型 | Qwen3.5-2B，本地 snapshot `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| 模型身份摘要 | `8cea164a6e87242bed9ffdc71dccb09eba74b71f6817b21d93cea6ab387c1682` |
| Python / MLX / MLX-LM | `3.14.3` / `0.32.1` / `0.31.3` |
| Provider / precision | `mlx-lm` / `bundle` |
| Context / output limit | 16384 / 2048 tokens |
| Prefill step / output batch | 256 / 1 tokens |
| MLX memory guideline | 10737418240 bytes；这不是已验证的 admission 上限 |
| 观测策略 | `runtime_owned_streams` 同步后采样；主机压力策略 `macos` |

身份依据为各次 `calibration.json` 中的 manifest、runtime 和 requested profile。设备与 OS 另见 [hardware.json](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/hardware.json) 和 [os-version.txt](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/os-version.txt)。原始 hardware 文件含设备标识，本记录不转录这些标识。

启动命令显式设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`TOKENIZERS_PARALLELISM=false` 和 `PYTHONNOUSERSITE=1`。probe 使用仓库固定的 `.apxinf/toolchains/mlx-lm-0.31.3-copies/bin/python`。其余环境继承父进程，没有保存完整环境快照，因此不能宣称环境完全隔离或可逐变量复现。stage01 整个 probe 的超时为 180 秒，stage02 为 300 秒。

首次 smoke 与 stage01 的 model revision 相同，measurement driver 摘要不同。首次 smoke 早于显式计划和 seed 冻结扩展；不能把两次运行描述为完全相同的测量程序。stage02 使用与 stage01 相同的 driver 摘要。

## 首次 smoke：加载后压力阻断

运行 ID 为 `96d97fdf-7ed6-44d1-b657-034e3b66d5fe`。创建时间为 `2026-10-09T23:31:04.107330+00:00`，即北京时间 10 月 10 日 07:31。

该次使用原自动矩阵，计划 11 个 case，目标输出包含 0、1、8。它没有使用后来的 `01-smoke.json`，也没有 `measurement-inputs.json`。

原始 [calibration.json](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/smoke/calibration.json) 记录 `before_load=normal`、`loaded=critical`，错误为 `PressureBlocked: Host pressure blocks loaded: critical.`。两个 startup 样本之后没有 case 样本；11 个 case 均为 skipped。加载 epoch 0 的 allocator peak 为 3763655416 bytes，不属于任何 generation case。

运行 cleanup 的 `settled=true`。父进程记录子进程返回 1、`reaped=true`、进程组无剩余成员。[coverage 报告](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/smoke-coverage.json) 的内部一致性检查为 valid，但 planned/executed/generation observed/reached 分别为 11/0/0/0。

同时段的 [编译进程快照](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/compile-processes.txt) 留有一个 cargo 进程；[后续主机快照](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/post-smoke-memory.txt) 为 warning，系统 swap used 为 11765.88 MiB。这些记录说明存在背景负载，不能把全部内存压力或 swap 变化归因于模型加载。

stage01 前的 [主机快照](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/pre-stage01-memory.txt) 已恢复 normal，swap used 为 6828.50 MiB。[该次进程筛选结果](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/pre-stage01-processes.txt) 为空。这是该时刻的检查结果，不代表主机始终空闲。

## stage01：五个 case 完成，两个输出目标未达到

运行 ID 为 `4b3de419-cb04-4ebe-ba09-a42f15789864`。创建时间为 `2026-10-09T23:41:32.969843+00:00`，即北京时间 10 月 10 日 07:41。

该次使用显式 [01-smoke 计划](../../benchmarks/serving/plans/01-smoke.json) 和 [primary seed](../../benchmarks/serving/plans/seed-primary.json)。父进程将内容、profile 和来源摘要冻结到 [measurement-inputs.json](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/stage01/measurement-inputs.json)，子进程使用固定快照。prepared seed 为 66 tokens，其 token digest 为 `f1d579cae74f849a82deccf247b703ecd10fce531d5634f66d36d6e8b9dceb0b`。每个输入仍由该 seed 的 token 序列平铺和截断而成，不是完整对话质量测试。

`P` 表示实际输入 token 数；`G` 表示计划输出 allowance。实际输出数包含 EOS。下面的 peak 仅指该 case 自有 reset epoch 中观察到的 allocator peak。

| Case | P / G | 实际输出 | 完成原因 | Reached | Peak epoch / bytes | 最大观察 KV offset |
| --- | --- | --- | --- | --- | --- | --- |
| `smoke-p1-g0-r0` | 1 / 0 | 0 | zero_output | 是 | 无自有 generation epoch | 未观察 |
| `smoke-p1-g1-r0` | 1 / 1 | 1 | length | 是 | 1 / 3795527502 | 2 |
| `smoke-p32-g1-r0` | 32 / 1 | 1 | eos | 是 | 2 / 3854909954 | 33 |
| `smoke-p32-g32-r0` | 32 / 32 | 1 | eos | 否：early_eos | 3 / 3854909954 | 33 |
| `smoke-p33-g32-r0` | 33 / 32 | 4 | eos | 否：early_eos | 4 / 3856109044 | 37 |

五个 case 的状态均为 completed，`generator_closed=true`、`settled=true`，cleanup error 均为 null。零输出 case 没有创建 generation iterator；其 closed 标记不表示执行过 generation。运行级 settlement 成功，子进程 PID 94167 返回 0，父进程确认 reaped，进程组无剩余成员。详情见 [calibration.json](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/stage01/calibration.json) 与 [process.json](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/stage01/process.json)。

[journal](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/stage01/samples.jsonl) 有 32 条记录，记录中的 pressure 均为 normal。这不证明记录间持续 normal，也不证明已有足够内存余量。[coverage 报告](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/stage01-coverage.json) 为 valid，planned/executed/generation observed/reached 为 5/5/4/3。

stage01 的加载 epoch 0 peak 为 3763655408 bytes。零输出和下一请求的 before_request 观察到继承的 peak，不能为其分配新 peak ownership。`P=32,G=1` 恰好在第一个输出遇到 EOS，仍达到 allowance；另两个 `G=32` case 只证明其真实短输出路径，不能外推到 32 tokens。

四个 generation case 的 terminal 样本均记录 18 个 ArraysCache 和 6 个 KVCache，逻辑 payload 为 22683648 bytes。ArraysCache offset 保持 null。四个 generation case 的 settled 样本中 payload 和 allocator cache 均为 0；模型 active memory 仍保留到进程退出。逻辑 payload 可能已包含在 allocator active 内，二者不能相加。

## stage02：已启动，加载后 warning 阻断

stage02 最初的 [启动日志](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/stage02-launch.log) 是设备锁获取失败，不能作为模型执行证据。后续 [重试日志](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/stage02-retry-launch.log) 对应真实 probe，运行 ID 为 `ff9f2147-1ab6-4deb-9f89-bd7c0ba23398`。创建时间为 `2026-10-09T23:48:21.480290+00:00`，即北京时间 10 月 10 日 07:48。

[原始报告](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/stage02/calibration.json) 记录加载前 normal，加载后 warning，错误为 `PressureBlocked: Host pressure blocks loaded: warning.`。计划的 12 个 case 全部 skipped；两个 startup 样本之后没有请求执行。加载 epoch 0 peak 为 3763655408 bytes。

运行 settlement 成功。子进程 PID 44629 返回 1，父进程确认 reaped，进程组无剩余成员。[coverage 报告](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/stage02-coverage.json) 当前为 valid，planned/executed/generation observed/reached 为 12/0/0/0。它保留了压力阻断证据，未建立 stage02 的 KV 或 prefill 边界覆盖。

## 07:56 释放后重试：加载仍触发 critical

用户告知内存应已释放后，重新检查到压力 normal、Metal 锁空闲。使用相同模型、配置、计划和 seed 重试第二阶段。新运行 ID 为 `c0d13557-49be-4913-b223-b72b54a83449`，创建时间 `2026-10-09T23:56:05.183182+00:00`，即北京时间 07:56。

[新原始报告](/private/tmp/apxinf-memory-resumed-20261010-Om6Jze/stage02/calibration.json)及 [journal](/private/tmp/apxinf-memory-resumed-20261010-Om6Jze/stage02/samples.jsonl) 记录 `before_load=normal`、`loaded=critical`。模型加载后 allocator active 为 3763655368 bytes，加载 peak 为 3763655400 bytes。工具再次在执行请求前阻断，12 个 case 全部 skipped。这说明未加载时的 normal 状态不能证明加载后的余量。

运行 settlement 成功。[进程记录](/private/tmp/apxinf-memory-resumed-20261010-Om6Jze/stage02/process.json)确认 PID 83984 返回 1、`reaped=true`、`remaining_members=[]`，未发送清理信号。[新 coverage](/private/tmp/apxinf-memory-resumed-20261010-Om6Jze/stage02-coverage.json) 为 `valid`、冻结输入为 `verified`，计数仍是 12/0/0/0。退出后的主机快照恢复 normal；未启动后续阶段，也未修改模型精度或压力策略。

| 新原始文件 | SHA-256 |
| --- | --- |
| [calibration.json](/private/tmp/apxinf-memory-resumed-20261010-Om6Jze/stage02/calibration.json) | `46edeed7b817dee6de8a77b974852c8e6b04d03247b57ecc7bba914ff13e6dbf` |
| [process.json](/private/tmp/apxinf-memory-resumed-20261010-Om6Jze/stage02/process.json) | `8c6ad23e73245ea7704c50859a9f9ebfe4366463dc2a14b8c424f7737c37f4d7` |
| [samples.jsonl](/private/tmp/apxinf-memory-resumed-20261010-Om6Jze/stage02/samples.jsonl) | `e64492751b54d7cded87c818ddcbb2de5f35f9c294be7a0087000ac32a8aa5c5` |
| [measurement-inputs.json](/private/tmp/apxinf-memory-resumed-20261010-Om6Jze/stage02/measurement-inputs.json) | `76f11daf584a0897fce511ba7ce627f54ca164b3e56ff876121d1ef905b39b7e` |
| [stage02-coverage.json](/private/tmp/apxinf-memory-resumed-20261010-Om6Jze/stage02-coverage.json) | `821a13cf1081821964e1acd12308edf7e1d17fccf98fd8101001dd9f113e3588` |

## 工具完成与实测完成的区别

新工具支持严格的显式计划、seed messages、父进程输入冻结、子进程摘要检查，以及退出后的输入一致性核对。CPU 测试覆盖这些行为，不代表测试过对应 GPU 形状。

[最终 Python serving 日志](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/python-serving-final.log) 记录 216 项测试、210 项通过、6 项跳过，耗时 1.843 秒。跳过的是要求设置 `APXINF_SERVING_URL` 的 live serving 测试；不能把它们计为本轮成功的实机测试。此前的 196 项测试日志另行保留。本轮没有修改 Rust 实现，没有重跑 Rust 测试或重新构建服务二进制。

[实测计划](memory-evidence-plan-20261010.md) 包含 8 份 plan、40 条静态 case 记录。stage08 分别运行两个保留输入后，完整安排为 9 个 probe、44 次 case 执行。这个数量是计划执行数；本记录只有 stage01 的 5 次 case 执行。stage02 的 12 个 case 尚未执行；首次被阻断的自动 smoke 不属于这 44 次计划。

后续阶段的边界、多输出 allowance、长上下文、晚停止和重复请求均未在本记录中得到实测确认。启动日志或计划文件不能替代完成证据。

## 冻结输入复核

原 coverage 分析器只检查三个原始文件，没有检查新增的输入快照。修复后，声明快照的记录必须通过固定文件名、4 MiB 上限、严格 JSON、字节摘要及内容关联校验。关联项包括 plan、profile、seed、来源记录、实际 prepared messages 和固定 template options。旧工件继续可分析，但明确标为 `unavailable`，不能据此通过独立输入来源门。

新报告分别保存在下面三个文件中，未覆盖旧报告或实测原始文件：

| 新报告 | 整体一致性 | 冻结输入校验 | planned / executed / generation observed / reached |
| --- | --- | --- | --- |
| [stage01-coverage-frozen.json](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/stage01-coverage-frozen.json) | valid | verified | 5 / 5 / 4 / 3 |
| [stage02-coverage-frozen.json](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/stage02-coverage-frozen.json) | valid | verified | 12 / 0 / 0 / 0 |
| [smoke-coverage-legacy.json](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/smoke-coverage-legacy.json) | valid | unavailable | 11 / 0 / 0 / 0 |

`verified` 说明保留快照与记录一致，不证明来源真实性、实际 tokenizer 计算或 GPU 执行。stage02 虽然快照通过校验，仍没有请求执行。分析器不重读外部源路径，也不把规范化 JSON 的摘要当作原文件字节摘要。

新增 20 项覆盖回归通过，coverage 测试共 54 项。另一次独立只读复审用 stage01 临时副本检查 21 项变体，未修改原工件。校验拒绝文件缺失、超限、错误摘要、重复键、非 UTF-8、类型混淆、消息及模板篡改；旧格式保持明确的未验证状态。

六份相关英文文档的 ASD 结构检查为零问题，见 [document-checks-final.log](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/document-checks-final.log)。这不包含完整词典审查，也不是认证。

## 尚未完成的验收

1. 补齐后续阶段的真实执行结果，特别是 `G=2048`、`P+G=16384`、晚停止及同进程重复请求。保留 EOS 未达到的目标，不通过禁用 EOS 制造覆盖。
2. 冻结一个具有明确支持域、输入特征和 margin 的 estimator 候选后，再执行保留输入验证。若先使用这些数据拟合候选，它们不能继续作为独立验证。
3. 按[生产验证判据](production-memory-validation-v0.1.md)验证不添加采样同步的生产路径。保留生产自身的 generation 与 cleanup 同步；缺失的逐请求清理后 allocator 观测保持 unknown。
4. 补全可复现环境和观测策略身份，并定义单独、版本化的 admission envelope 验收契约。当前 raw report 的硬件字段和源码摘要不能独自覆盖所有身份维度。

当前 `admission_approved=false`、`full_memory_domain_established=false`。即使后续全部计划完成，也不能自动得出普遍内存上界、无采样同步时的峰值或生产预算批准。串行记录不覆盖 batch、持久缓存和全部主机资源状态。

## 复查摘要

以下摘要按本次读取到的原始字节计算。文件后来改变时，应保留旧版本并更新记录，不应静默替换摘要。

| 原始文件 | SHA-256 |
| --- | --- |
| [smoke/calibration.json](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/smoke/calibration.json) | `ac80d059d66677b1c3d027c55c597132daefdd7b50951cbb4351ea6de29a5f71` |
| [smoke/process.json](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/smoke/process.json) | `2800d1df2b27b3201e605e08277cb565ff0e1b74bd8f0067a8269077cf5e1bfd` |
| [smoke/samples.jsonl](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/smoke/samples.jsonl) | `8dcd662cd39fc39c8d3b257456aba214cefe79ec4c26955b51666382275c1868` |
| [smoke-coverage.json](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/smoke-coverage.json) | `0fbdf0f6591ba1439962078571c33be1fe09e998ec0afee70384a0b622134ffc` |
| [stage01/calibration.json](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/stage01/calibration.json) | `545224ce1abf8a613dc1096204fae1be5fa16b939509918db8e9664ab3b9029b` |
| [stage01/process.json](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/stage01/process.json) | `9cf0a81d8a0ee0d531773ba0a46f828d1f65a063b04927b94bca1242ff2b1fb8` |
| [stage01/samples.jsonl](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/stage01/samples.jsonl) | `6171acdb99dd1938b24dc82f11a9d7d7744d50819065c1c98b2d63ea4393f379` |
| [stage01/measurement-inputs.json](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/stage01/measurement-inputs.json) | `96e652b27197878a33a98e79de9fe65be6f22dcb26f00a6943c7f89fd340f20f` |
| [stage01-coverage.json](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/stage01-coverage.json) | `7b2dbd3de8315fa6683e8e5a1a2258d7d4616dfe06231cf8570183fe082dadf9` |
| [stage02/calibration.json](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/stage02/calibration.json) | `6b07abfc3899f8bd936cd1f1203321ef72f4f02eff69b63107da7fca069a1011` |
| [stage02/process.json](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/stage02/process.json) | `9f7191c9ed18e94bac0d7b440c939ae2023433894b95bae2a0b1c44a1a3624d2` |
| [stage02/samples.jsonl](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/stage02/samples.jsonl) | `2887f8ba825a2e38f579f836b071294ed1d3dcd4f9b6a4d4585359111a527ddc` |
| [stage02/measurement-inputs.json](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/stage02/measurement-inputs.json) | `76f11daf584a0897fce511ba7ce627f54ca164b3e56ff876121d1ef905b39b7e` |
| [stage02-coverage.json](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/stage02-coverage.json) | `5b15003e1b239cdfa3f2c4e498f011ee44826f1c24982fe57e935a51e3b19d28` |
| [python-serving.log](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/python-serving.log) | `7af6f94d6d77834eb1af2e4648f99870073b6436ead827d2f24848f055e326a7` |
| [hardware.json](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/hardware.json) | `096a9dfb9afa93b1cb61df36bbf991db309ffc5d4e8669e9090feb506dd46fde` |
| [os-version.txt](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/os-version.txt) | `4ef10a0fc370704ea84699cf3851e51488598c5f1ab35cb572eb6b96278a8e5c` |
| [stage01-coverage-frozen.json](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/stage01-coverage-frozen.json) | `206ec278011b7ffe739a5c647e77d014e62d5fe67ce3933513b550c64f8bd132` |
| [stage02-coverage-frozen.json](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/stage02-coverage-frozen.json) | `f3c47b3652cec86453fe4a532dde6393f0ef02e99bfa9a9169ea8dc5b3541df4` |
| [smoke-coverage-legacy.json](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/smoke-coverage-legacy.json) | `13c6cd299d8d36c871551402e1b5c71f537ecc16e428bf70b235f246303e37b7` |
| [python-serving-final.log](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/python-serving-final.log) | `e203e28fec741fdb15a75353593ad623b135046153f6e1169055683a25af120f` |

stage01 冻结的原 plan 文件摘要为 `3feba8f29a00d045823c74d669c90d9f5f1b9e93f7dd0effa76b1d4eca62853e`，原 seed 文件摘要为 `ac7671a287199798b00e041ef36c59279ff035db78139e89af050f46057cd10b`。这两个值记录来源文件原字节，不等同于规范化快照的摘要。

首次 smoke 的 driver 摘要为 `d445d52e284180f28be0fdbba8581c23b3c2361e2be9a086d0f125970a1dd317`；stage01 为 `6fb6d20d660cbb1f18ddec2871a1998be23179bab8b1d064998178b5ef78be06`。完整 adapter、contracts、Python、runtime pins 及模型文件摘要留在各自 manifest 中。

最终 coverage 分析器的源码 SHA-256 为 `8dd1d58ce91673730bf5355c6e81e4069c0a56dbf64839abf46ad12f36a63d2d`。该源码生成了新增的三份覆盖报告。

本记录使用中文解释，不作 STE 英文字典合规或认证声明。
