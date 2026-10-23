# Clef 文本决策 serving 实现与验收

日期：2026-10-23。实现比较基线为 `7d72515685dddfd505b18fb7c8787d876138268c`。
本报告继续[此前的缺口审查](clef-validation-20261023.md)，区分功能验证与默认内存准入。
新增接口与限制以 [Clef 文本契约](clef-text-v0.1.md)为准。

## 实现范围

新增显式 `python-clef` provider 和 `POST /v1/systemone`。
它执行 Clef joint schema head，返回 `noul`、`choice`、`score` 概率及实际输入用量。
它调用本地外部依赖的公开加载、编码和 logits 接口，没有移植模型或 kernel 实现。
外部 loader、模型文件、运行时和第一方 adapter 均参与身份检查。
Rust 与 Python 读取同一份固定模型文件摘要表和原创协议语料。

决策请求复用现有协调器、队列、截止时间、压力检查、诊断和资源所有权。
请求与结果采用独立类型，不伪造 token 生成事件。
首版每次决策使用一个 worker 进程，结束后回收并加载相同身份的替代实例。
成功结果在 `drained`、干净 EOF、退出码零和成功回收后公开。
超时或断流可以先结束公共请求，资源必须等到进程确认回收后释放。
替代实例加载期间不接受新请求，不自动重放已执行请求。

输入先准备完整编码，再以不可变 preparation 绑定请求、模型、能力和进程实例。
执行前重放编码，执行后比较全部 token、问题跨度、选项跨度和顺序。
完整输入上限为 4096 token，不截断，不预留生成输出，不保留跨请求 KV cache。

## CPU 验证

| 检查 | 结果 | 范围 |
| --- | --- | --- |
| Rust 默认 serving 套件 | 200 通过、1 忽略 | 含真实子进程管道、HTTP 路由和信号测试，不执行真实模型 |
| Rust `native-minicpm5` 套件 | 223 通过、1 忽略 | 包含默认套件，验证既有原生路径，不与默认数量相加 |
| Python serving 全量 | 355 运行，348 通过、7 跳过 | 使用共享 Python 环境，跳过项不算通过 |
| 指定 Clef Python 环境 | 30 通过，无跳过 | 协议、运行时替身、身份与 worker 测试，未加载权重 |
| 两端共享原创语料 | 67 个场景 | 20 请求、10 编码、10 概率、4 摘要、9 frame、14 stream |
| Rust 决策生命周期 | 7 个测试通过 | 私有结果、非零退出、多余输出、上下文失败、替换、取消、超时、队列、关闭和断流 |

上述套件存在重叠，不能相加。
CPU worker 使用测试专属的身份校验后启动入口，不能证明真实模型身份或设备执行。
独立审查另行运行了决策生命周期、共享契约和 Python 身份测试。
审查发现并修复了 loader 字节码缓存绕过源文件摘要的问题。
实现现在直接编译和执行已校验的同一组源文件字节。
实机启动还发现 macOS 共享缓存镜像没有独立文件路径，现仅严格解析选中的 MLX 动态库。
实际 wheel 构建发现固定摘要表未随 Python 代码打包，现已显式加入 package data。
修复后的 wheel 内该文件与仓库逐字节相同，解包后按 worker 独立模块路径以 `-S` 导入可读取全部 11 条记录。

## 真实模型与 HTTP

本机为 Apple M4、16 GiB 内存。
模型为 `mlx-community/clef-flash-4bit`，revision 为 `b5adbe2c3a95200d4bf04a0727c8ec457157d5e1`。
运行时选择 Python 3.12.10、MLX 0.32.3、MLX-LM 0.32.0 和明确的语言模型后端。
head 使用 BF16，allocator guideline 为 8 GiB，空闲 cache 上限为零。
这些配置不是进程 RSS 或系统物理内存的硬上限。

`http-default-001` 在就绪前暴露共享缓存路径解析错误，进程组已回收。
修复后的 `http-default-002` 被 macOS `warning` 压力状态阻止加载，没有业务请求成功。
这两次失败记录保留，不以之后的诊断结果覆盖。

`http-diagnostic-003` 显式使用 `--host-pressure-policy disabled`，运行独立的严重压力及传感器故障终止监测。
采样包含 `normal` 和 `warning`，没有 `critical`。
因此下表证明 HTTP 功能，不证明默认压力门控可以放行。

| 实际 HTTP 场景 | 结果 |
| --- | --- |
| 故障文本，同时询问三种问题类型 | 200，328 输入 token，零输出 token |
| 健康文本，同时询问三种问题类型 | 200，331 输入 token，路由选择无需修复 |
| 重复故障文本 | 200，完整响应与首次相同 |
| 长文本 | 200，2828 输入 token，与独立编码计数相同 |
| 超过上下文 | 400，`context_limit`，回收 worker 并加载替代实例 |
| 上下文错误后的普通请求 | 200，328 输入 token |
| 非法 streaming、图片字段 | 400，`invalid_request`，不消耗 worker |
| 错误模型名 | 404，`model_not_found`，不消耗 worker |

五次成功响应均在旧 worker 不再存活后到达。
每次消耗 worker 的场景均更换 epoch，并保留相同模型和能力 revision。
服务退出码为零，所属进程组为空，没有清理错误。
服务端容量所有权和成功回收的严格顺序另由受控生命周期测试验证。

真实截止时间测试 `http-lifecycle-diagnostic-005` 对长输入返回 504 `deadline_exceeded`。
错误响应到达时旧 worker 仍存活，之后回收并以新 epoch 恢复就绪，符合提前公开错误、继续结算的约定。
该次脚本随后过早读取断流后的就绪状态，误将异步取消尚未完成判为失败，完整失败记录保留。
此前 `http-lifecycle-diagnostic-004` 也因错误地要求 504 前完成回收而失败，属于同类观察断言错误。
修正脚本后，`http-disconnect-diagnostic-006` 在请求活跃时关闭真实 HTTP 连接。
最终取消计数增加一次，完成计数为零，活跃请求和预留容量均为零，worker epoch 已更换。
该次服务退出码为零，进程组为空，没有清理错误。
这两类实机生命周期检查也使用显式关闭压力门控的诊断配置。

独立公开依赖对照 `text-canonical-reference-004` 使用相同规范排序后的原始问题。
直接运行时与 HTTP 的输入计数均为 328。
公开响应中的七个概率数值最大绝对差为 `1.1102230246251565e-16`，小于比较阈值 `1e-12`。
此比较不是逐位一致声明，也没有使用依赖 HTTP server 的四位小数输出作为原始概率。
该直接对照同样允许诊断期 `warning`，不能完成默认压力准入验收。

原始记录位于忽略目录 `devlocal/clef-serving/`，不提交模型、虚拟环境或测量日志。

| 记录 | SHA-256 |
| --- | --- |
| `http-diagnostic-003/receipt.json` | `75fcd8d50e56aecf823255e5abc1b022677d25ab6dbd1f15b7c634ba08428de1` |
| 该次 release binary | `309f9187f0b794ef48291cae468c9b8272cc0d77d8a6b3586808a0488e04ce29` |
| `text-canonical-reference-004/receipt.json` | `cf4fc3df9fe0503b8b08377ad2c30db4a253d2661aca6695d368a81bb14eb2e1` |
| `http-lifecycle-diagnostic-005/receipt.json` | `ebe56724ca1e72a80b747249df895cd818227194e2fe3f14f31d63c6b797c979` |
| `http-disconnect-diagnostic-006/receipt.json` | `5f5723c370f3dfb61fa894e6d6d7141407a2d696c548a96dd4f2ac7235634526` |

## 保留限制

这不是聊天生成模型接口，不能直接充当 Claude Code 的 Anthropic Messages provider。
本轮不支持图片、视频、流式输出、生成采样或 thinking。
尚未验证边界 4096 token 的真实完整 forward，2828 token 是本次成功 HTTP 长输入范围。
每次请求重新加载和校验模型，重复请求的加载开销明显，未提供吞吐或延迟承诺。
默认 macOS 压力策略仍可能阻止就绪，不能把诊断配置推荐为已验证的正常部署策略。
常驻 worker 复用需要先定义并验证完整状态重置接口。
跨请求 KV cache 复用、批处理和多模型调度不在首版范围内。

英文契约和运行说明使用 STE 结构检查，不把该检查等同于完整词典审核或认证。
