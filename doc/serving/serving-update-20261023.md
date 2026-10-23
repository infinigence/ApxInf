# Serving PR #116 增量与验证范围

日期：2026-10-23。比较基线为 PR #116 原 head `4b44566614048d9151f79adafae1d607299fd806`。
本轮以 serving 能力为验收重点。鹈鹕骑车的绘图质量不作为本次 PR 的完成条件。
原始运行记录保存在本机 `devlocal/serving-minicpm/`，不随 PR 提交。

## 相对原 PR 的变化

| 能力 | 原 PR | 本轮增量 |
| --- | --- | --- |
| 模型执行 | Python MLX-LM 串行文本 worker | 增加显式选择的原生 MiniCPM5-2B worker；保留原配置行为 |
| 工具调用 | Qwen XML／JSON 工具解析 | 增加 MiniCPM function／param／CDATA 增量解析；处理引号内分隔符和布尔参数 |
| 生成控制 | 贪心生成 | 增加显式采样配置、温度、top-p 和种子；固定配置身份；修复极小温度的数值溢出 |
| 初始思考 | 未提供专用配置 | 增加隐藏初始思考及独立的 512-token 有界配置；完整计入生成用量 |
| 上下文 | 声明配置上限，实测范围有限 | 分块 prefill、完整输入计数、输出空间预留与越界拒绝；增加上下文检索验证 |
| 图片 | 不接受图片 | 增加独立 MLX-VLM MiniCPM-V-4.7-1B 配置，支持 Anthropic 用户／工具结果内的 base64 PNG |
| 输入一致性 | 文本准备 | Rust／Python 共同校验 PNG；保留图片顺序、重复项和工具 ID；重放核对 processor 输入与 token 数 |
| 资源结算 | 串行请求与取消结算 | 增加原生生成进度／取消接口及视觉请求状态清理；不确定清理永久隔离 worker |
| 客户端验收 | Qwen 文本 Read／Edit／Check／Cancel | 增加真实 MiniCPM 工具链、视觉 HTTP 和客户端图片传输检查；记录失败任务 |
| CPU CI | Rust／Python 契约测试 | 扩展共享 fixtures、图片与视觉 worker 测试；显式安装 NumPy／Pillow |

MiniCPM-V 接入调用 MLX-VLM 的公开接口。它不是新增的 Rust 原生视觉推理实现。
初始思考配置不是 Anthropic extended-thinking API。后者仍不支持。
原生模型、采样、tokenizer 和 MLX 层的配套接口是 serving 的必要依赖。
这些接口不构成其他后端或模型的优化交付。

## 已验证的范围

| 验证 | 结果 | 边界 |
| --- | --- | --- |
| Rust 默认测试 | 186 通过，1 忽略 | 包含真实 HTTP transport 和退出信号测试；不执行真实模型 |
| Rust native-minicpm5 测试 | 209 通过，1 忽略 | 包含默认测试；不能与默认数量相加 |
| 共享 Python 环境 | 325 运行，7 跳过 | 318 通过；包含两项 Clef 错误接入回归；跳过项不算通过 |
| 独立 CPU 环境 | 325 运行，20 跳过；生成脚本另有 10 通过 | 仅安装 NumPy／Pillow，无 MLX；额外跳过 13 项设备状态检查 |
| 固定视觉环境 | 35 通过，无跳过 | 与 Python 全量有重叠；不能相加 |
| Rust／Python worker 2.5 协议对照 | 85 frame、15 stream、4 digest 用例；PNG 48 用例 | 两端使用同一套原创输入；计入上述套件 |
| 实际视觉 worker | 5 个串行请求完成、退出并回收 | 文本、两种图片、取消和取消后请求；不证明 prefill 内任意位置可取消 |
| 直接运行时对照 | 3 个不同请求的完整 token 序列一致，另有 2 次重复比较 | 公共依赖 API 对照；不是 5 个不同请求 |
| 长输入检索 | 4094、8180、14999 个准备后输入 token 均返回精确 JSON 并到达 EOS | 直接运行时诊断，各预留 256 输出；不是 15K 的 HTTP／Claude 端到端验收 |
| 真实视觉 HTTP | 图片计数与生成一致；非法 URL、温度和上下文越界拒绝 | `service-actual-004`；图片请求为 89 输入、132 输出 token |
| HTTP 流断开 | 取消后恢复空闲容量，取消计数增加 | 服务退出 0，worker 与进程组回收；不以客户端干净 EOF 作为证据 |
| 原生 MiniCPM Claude Code | 实际 Read／Edit／Bash 工具循环完成 | 鹈鹕绘图质量未通过独立图像检查 |
| Claude Code PNG 传输 | 真实客户端将 Read 图片放入下一次请求，内容摘要一致 | 使用模拟服务响应；不证明模型看图修订能力 |

详细证据见[视觉 worker 报告](minicpm-vision-worker-validation-20261023.md)、
[有界思考报告](native-bounded-thinking-validation-20261023.md)及
[默认上下文报告](minicpm-default-context-validation-20261023.md)。
不同模型、配置和执行路径的结果不能互相替代。

## 明确保留的限制

- 真实 MiniCPM-V Claude Code 图片修订未通过。模型路径大小写错误、读取二进制和未知工具调用阻止任务完成。
- 加载后内存压力仍可能阻止服务就绪。加载前正常不保证加载后正常。
- 10 秒启动等待试验未恢复就绪。本次 PR 不包含该试验实现。
- 当前模型检查不构成通用内存上限证明。allocator 统计不是进程 RSS。
- 视觉输入仅限有界 PNG。URL、其他图片格式、OpenAI 图片接口和视频不支持。
- 不提供跨请求 KV cache 复用、连续批处理、多模型调度或持久会话。
- Clef 决策接口尚未接入。文本 provider 明确拒绝决策头模型包；[参考启动验证](clef-validation-20261023.md)仍受内存压力阻断。
- 初始思考预算计入总输出额度。提高额度不能保证模型会正确使用工具或完成绘图。

本次 PR 保留 serving 及必要原生接口。不提交本地实验、权重、虚拟环境或其他子系统修改。
上游合并带入的 CUDA、HIP、Walloss 等历史提交不属于相对最新上游的 PR 文件差异。
