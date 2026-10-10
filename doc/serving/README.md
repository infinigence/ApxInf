# ApxInf Serving 设计入口

当前状态：系统框架和统一契约已完成首轮设计。P1a 串行服务与初始 P2a 工具适配已完成实机验证。真实模型、Claude Code、生命周期与性能结论以测试报告的具体范围为准，其余路线继续分阶段实现。

本目录将设计依据、接口规范和实施步骤分开。后续实现使用同一份接口规范，避免 Rust、Python 和公共 API 各自定义语义。

| 文档 | 用途 | 优先级 |
| --- | --- | --- |
| [系统方案](../serving-system-design-20261009.md) | 中文架构说明、现状审计和调研依据 | 解释设计选择 |
| [接口规范](contracts-v0.1.md) | 英文术语、所有权、字段、事件、状态、错误和兼容规则 | 新 serving 接口的语义依据 |
| [实施计划](implementation-plan.md) | 模块边界、阶段依赖、交付内容和验收门 | 规定实施顺序 |
| [串行协议](serial-profile-v0.1.md) | 首个实现子集的完整字段、限额和事件顺序 | Rust／Python 共同校验 |
| [本地服务](local-service-v0.1.md) | 启动方式、HTTP 能力与当前限制 | 当前实现说明 |
| [依赖边界](dependencies.md) | 依赖版本、公开 API 和替换边界 | 独立实现审计 |
| [Metal 实验](metal-experiments.md) | M4 上的实验矩阵与创新候选 | 测量方案 |
| [Metal 参数实测](metal-results-20261009.md) | 九组参数、输出差异、内存和交互延迟 | 配置选择依据 |
| [实现与验证](serving-validation-20261009.md) | 实际实现范围、HTTP 指标、资源边界和下一阶段 | 交付证据 |
| [生命周期加固](lifecycle-hardening-20261009.md) | 命令轮换、准备结算、队列回收、关闭竞态和回归结果 | 本轮可靠性修复证据 |
| [阶段观测与到达率负载](observability-load-20261009.md) | 阶段直方图、submit deadline、绝对 HTTP 超时及实机过载结果 | 后续内存准入和调度的测量依据 |
| [内存压力准入与预算加固](host-pressure-memory-20261009.md) | 运行期压力控制、预算溢出修复、峰值补报及验证边界 | 内存准入的当前实现与证据 |
| [内存校准规范](memory-calibration-v0.1.md) | 专用 stream 同步、缓存观测、离线样本与进程回收 | 内存估算前的原始证据格式 |
| [stream 同步与校准实测](memory-calibration-20261009.md) | 同步修复、失败恢复和本次加载后压力阻塞记录 | 本轮实现与验证边界 |
| [生成公开截止时间修复](generation-deadline-20261009.md) | 及时公开 expiry、继续资源结算、迟到事件与单次计数 | 生成生命周期回归证据 |
| [内存覆盖规范](memory-coverage-v0.1.md) | 计划、执行、实际达到、峰值归属与缺失范围 | 校准证据的离线检查规则 |
| [诊断输出隔离与覆盖检查](diagnostics-coverage-20261010.md) | stderr 故障修复、CPU 回归和旧实测覆盖回放 | 2026-10-10 实现与验证结果 |
| [分级内存实测计划](memory-evidence-plan-20261010.md) | 长输出、上下文边界、较晚取消、重复与独立输入 | 待执行的原创测量矩阵 |
| [内存实测记录](memory-evidence-20261010.md) | 短请求生成、EOS 缺口、压力阻断和输入冻结 | 2026-10-10 实测与验证范围 |
| [生产内存验证判据](production-memory-validation-v0.1.md) | 正常生产同步、指标口径、结算和独立输入验收 | 待执行的 P1 验证协议，无批准数值 |
| [Qwen3.5 缓存几何](qwen35-memory-geometry-20261010.md) | 固定状态、KV 扩容、预计算和采样边界 | 固定版本源码推断，不替代实测 |
| [Claude Code 验证](claude-code-validation-20261009.md) | 真实客户端任务、工具记录和取消检查 | Agent 路径实测 |
| [仓库规则](../../AGENTS.md) | 独立编写、接口一致性和文档要求 | 约束后续仓库工作 |

接口规范中的 `OR`、`ID`、`EV`、`SS`、`ST`、`RE`、`IC` 等规则编号用于追踪实现与测试。完整规范仍是设计草案。首个实现子集已经通过串行协议补齐字段、限额和原始样例；其余能力仍需通过各自阶段验收。

## 独立编写的边界

允许阅读其他仓库，理解机制、公开接口和行为。新增 ApxInf 实现、测试和 fixtures 必须独立编写。不复制、不翻译、不移植外部实现，也不通过修改名称来重用其代码。不导入外部代码模板或生成产品骨架。

继续调用现有依赖的公开 API。例如，MLX adapter 可以调用 MLX-LM，但不能把其 scheduler 源码复制进 ApxInf。新增依赖需要在设计中记录用途和替换边界。已有 vendored 代码保留来源，不因本次设计删除或改写来源记录。

此前调研已阅读上游实现，因此本方案使用“独立编写”这一表述，不声称执行了未接触源码的 clean-room 流程。外部源码只作为研究材料，新的实现从 ApxInf 需求、接口规范和独立测试出发。

## ASD STE100 的应用

采用 ASD-STE100 Issue 9 的技术英语写作原则。当前版本由官方页面确认为 2025-01-15 发布。[官方介绍](https://asd-ste100.org/about_STE.html)

英文接口规范与实施计划使用短句、主动语态和统一术语。每条规则尽量只描述一个义务，并指明执行主体。步骤句上限为 20 词，描述句上限为 25 词。中文系统方案用于解释，不标为 STE 合规英文。

ASD-STE100 不规定软件模块、API schema 或并发语义。接口一致性由规范、版本约束、共享原始 fixtures 和双向契约测试保证。写作检查与工程验收分别记录。

已安装的技能：

- 来源：[danyuchn/asd-ste100-skill](https://github.com/danyuchn/asd-ste100-skill/tree/32511c6992ecb5f1971e46a2943f2e6adceedafe)。
- 固定 revision：`32511c6992ecb5f1971e46a2943f2e6adceedafe`。
- 技能版本：`0.4.0`。
- 安装位置：`~/.codex/skills/asd-ste100/`。
- 范围：写作规则和结构 linter，不含官方完整受控词典。
- 归属：个人开发工具，不作为 ApxInf 产品代码或运行时依赖。

该社区技能不是 ASD 官方工具。结构 linter 不检查所有术语的词性、词典资格、技术含义或全部语法规则。通过检查不等于完整符合 ASD-STE100，也不构成标准认证。[官方工具说明](https://asd-ste100.org/STEsoftware.html)

## 文档检查

2026-10-09 初始设计文档检查记录（不代表后续运行时验收）：

- `AGENTS.md`、英文接口规范和实施计划通过技能默认结构检查，0 violations，未提高 baseline，未禁用规则。
- 补充检查未发现超出 20／25 词限制的步骤句或描述句。
- 本地文档链接均可解析，规则编号没有重复。
- 人工接口复核修正了重复提交、拒绝后的租约结算，以及 stop／cancel／failure 的优先关系。
- 未运行模型或 serving 测试。上述结果仅证明文档检查完成，不能替代未来接口与实现验收。

```sh
python3 ~/.codex/skills/asd-ste100/scripts/ste-lint.py \
  AGENTS.md \
  doc/serving/contracts-v0.1.md \
  doc/serving/implementation-plan.md
```

不把中文文档交给英文词数检查器并据此宣称通过。后续文档更新需要重新检查。官方完整词典的逐词审核仍未完成。
