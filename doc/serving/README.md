# ApxInf serving

本目录描述串行服务的接口、所有权、部署配置和支持边界。
完整系统方案仍是分阶段设计提案。
已实现的串行子集不代表会话、连续合批或其他规划能力已经完成。

| 文档 | 用途 |
| --- | --- |
| [系统方案](../serving-system-design-20261009.md) | 架构提案、历史基线和设计依据。 |
| [接口规范](contracts-v0.1.md) | 术语、所有权、状态、事件、错误和兼容规则。 |
| [实施计划](implementation-plan.md) | 模块边界、阶段依赖和验收条件。 |
| [串行协议](serial-profile-v0.1.md) | Rust 和 Python 共同校验的字段、限额和事件顺序。 |
| [本地服务](local-service-v0.1.md) | 启动命令、HTTP 接口和当前配置。 |
| [支持范围与验证边界](support-status.md) | 各配置的实现范围、证据来源和未验证能力。 |
| [依赖边界](dependencies.md) | 原生运行时与外部 provider 的依赖及执行边界。 |
| [原生 MiniCPM5](native-minicpm-v0.1.md) | 显式原生文本配置和运行时身份。 |
| [原生采样](native-sampling-v0.1.md) | 采样接口、选项和模型边界。 |
| [初始思考](native-thinking-v0.1.md) | 隐藏初始思考和公开输出的处理。 |
| [有界初始思考](native-bounded-thinking-v0.1.md) | 显式思考预算与完整用量计数。 |
| [准备代码复用](native-prepared-v0.1.md) | 单个编译代码项、请求状态清理和分配器结算。 |
| [上下文配置](minicpm-context-v0.1.md) | 受控预填充、部署上下文和取消边界。 |
| [输出配置](minicpm-output-v0.1.md) | 模型输出结构和客户端投影。 |
| [PNG 输入](owned-media-v0.1.md) | 有界图片数据、完整性和跨语言用例。 |
| [MiniCPM-V](minicpm-vision-v0.1.md) | 独立 Python provider、图片准备和请求状态。 |
| [Clef 文本](clef-text-v0.1.md) | 独立决策类型、概率结果和 worker 结算。 |
| [内存校准](memory-calibration-v0.1.md) | 同步采样、缓存观测和进程记录格式。 |
| [内存覆盖](memory-coverage-v0.1.md) | 计划、实际执行、达到的位置和缺失范围。 |
| [生产内存验证](production-memory-validation-v0.1.md) | 保持生产同步行为的独立验证条件。 |
| [基准工具](../../benchmarks/serving/README.md) | 可复用的 HTTP、生命周期、客户端和内存检查。 |

新 serving 接口以接口规范为语义依据，具体 worker 配置定义其实现子集。
系统方案中的初始审计针对历史 fork，不把其中的旧实现当作当前支持能力。
原生模型和调用外部引擎的 Python provider 分别声明执行身份。
原始测量记录、环境快照和开发历史保存在本地开发目录，不属于服务接口。
