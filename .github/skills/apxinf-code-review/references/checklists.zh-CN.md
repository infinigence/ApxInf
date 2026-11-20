# ApxInf Review 检查表（简体中文）

英文版：[checklists.md](checklists.md)。
扩展检查项的历史依据见：
[historical-patterns.zh-CN.md](historical-patterns.zh-CN.md)。

只使用与当前改动路径相关的章节。高风险章节要求完整检查 changed production
code、集成调用方和相关测试。

## 风险路由

高风险：

- Metal、CUDA、HIP、CoreML、MLX kernel，以及 `unsafe` 或 FFI 边界。
- 量化、权重解码、Tensor layout、KV cache、Attention、Recurrence、Sampling、
  模型加载、Tokenizer 到 Embedding 的边界、多模态输入。
- Runtime Trait、模型分发、缓存发布、并发和生命周期。
- Execution Session、CUDA Graph Capture、Prepared Profile、cuda-new Autotune 与
  exact-key Recipe 持久化、native instantiation/build fingerprint、持久化缓存和
  设备内存 Arena。
- Backend 迁移、Feature Graph 变化、公共 ABI 变化，以及 Provider/Fallback 选择逻辑。

中风险：

- 配置解析、Python/CLI 接口、Benchmark harness、构建 Feature、平台选择、
  序列化和转换工具。
- 会弱化架构检查、验证门槛、设备支持声明或兼容性保证的文档与策略修改。

低风险：

- 文档、示例、格式化和机械重命名；但如果它们会改变命令、兼容性声明、
  unsafe contract 或公共行为，则不能按低风险处理。

## Kernel、Tensor、FFI 与设备

- 使用 host 端 checked arithmetic 重新计算所有 grid、group、index、byte offset、
  stride、alignment、padding、broadcast 和 output extent。
- 检查 0、1、tile 边界、非整数 tile tail、最大可表示范围、混合 batch/head 数，
  以及 partial rotary dimension。
- 确认 shape/dtype guard 支配所有 cast 和 dispatch；原生调用前检查有符号窄化和
  `usize` 乘法。
- 验证 buffer 生命周期、stream/device 所有权、同步、alias、临时分配和异步发布。
- Stream 在 blocking/default 与 non-blocking 之间变化时，重新推导执行顺序。普通
  `cudaMemcpy`、`cudaMemset`、host read、cache clear、buffer reuse 和 pageable upload
  不会自动继承另一条 stream 的顺序。
- 当 kernel 声称完整覆盖输出时，用 zero、dirty、poison 三种初始化验证。模型输出随
  初始化变化时，应当先找到第一个 partial writer 或 ordering gap，不能解释成普通数值误差。
- 追踪 native runtime 对每个 execution、session、graph 和 eager handle 的所有权。
  不依赖 struct 字段声明顺序完成析构；成功和错误路径上 owner 都必须晚于依赖对象销毁。
- 确认 capture begin/end 在所有提前返回上都正确清理。Capture 内不能意外进行资源创建、
  tuning、allocation 或 host callback。
- 将数值运算顺序、累加 dtype、舍入位置、epsilon、NaN 和 Infinity 行为同声明的
  reference 对比。
- 检查不支持的架构、关闭 Feature 的构建、CPU fallback 和错误传播。编译成功不等于
  通过设备资格验证。

## 量化与权重

- 验证 packing 顺序、group size、scale/bias layout、transpose 约定、row slicing、
  padding、endianness 和 dtype 转换。
- 将参数名、shape 和 optional tensor 同真实 checkpoint metadata 对照，并考虑
  sharding、tied weight、缺失 key 和额外 key。
- 确保临时解码权重不会造成意外常驻或 hot path 上的重复转换；分别验证每一种
  format 和 arithmetic variant。
- Calibration 和 execution-selection artifact 的身份必须覆盖所有会影响推理的资产，
  而不只是 safetensors。对 cuda-new，这意味着 Recipe identity；只有路径仍实际消费旧格式时
  才应用旧 Tactic DB 规则。Processor config、statistics、tokenizer、sharding index 和辅助
  bundle 都可能改变模型行为。
- Rust/Python 身份计算必须字节级一致：相同 shard 选择、相对路径规范化、排序、编码和
  schema version。
- 明确定义旧 artifact 的迁移与拒绝策略。无效/不支持的 artifact 应 fail closed 或明确
  fallback，不能静默切换到数值行为不同的路径。
- 检查 fused/decomposed 路径是否重复 observation、重复 quantization 或保留重复权重；
  如果必须保留，应公开并测量内存成本。

## 模型、Tokenizer 与多模态输入

- 在有符号 cast 和 embedding lookup 前，对 tokenizer ID 和生成 ID 做范围检查；
  验证 special-token ID 与 checkpoint vocabulary 一致。
- 追踪 prompt template、schema/decision 序列化、停止条件、最大生成限制、输出解析，
  以及错误模型配置的行为。
- 对 image、audio、video，验证所声明的能力确实连接到 preprocessing、encoder、
  projection、batching 和 runtime dispatch；不能仅凭 upstream 模型支持就推断本实现支持。
- 检查 cache 在不同请求、batch size 和 conversation 之间的 reset 与复用。
- 在昂贵的 vision/audio/device 工作和状态修改之前，检查 capacity、divisor、dimension
  和请求限制，包括 0、overflow 和精确边界。
- 可选模态权重必须条件加载；Capability 从实际加载的资产推导，并保留 text-only 或
  精简 checkpoint 的兼容性。
- 验证公共 Trait 的完整 shape/语义契约，而不只满足共享 generation caller 的需要。
- 端到端追踪用户 override 和默认值：Python/CLI -> LoadOptions -> admission -> runtime；
  Metadata 不能报告一个执行路径实际忽略的选项。

## Runtime 与 API 设计

- 在概念上分离模型类别、加载协议和执行协议。共享 Trait 应当表达真正可替换的契约，
  不能强迫无关模型实现没有意义的方法。
- 保持当前契约边界：`LlmTrait` 负责自回归 LLM/VLM generation，`VlaRuntime` 负责
  observation-to-action inference；Decision 模型协议必须由它自身可替换的语义证明。
  仅仅都接收语言或图像输入，不足以共用一个执行 Trait。
- 检查 object safety、所有权、线程安全、错误类型、能力发现、downcast、公共 enum 的
  穷尽性，以及与现有 loader 的兼容性。
- 当调用者必须了解特殊 layout、dtype 或固定 geometry 时，公共名称应明确表达这些
  限制；不要给固定倍数或固定 shape 使用看似通用的名字。
- 模块应聚合单一职责；不要用 `runtime`、`utils`、`gemm` 等宽泛名称掩盖多阶段
  调用协议。
- 模型数学由 model 层拥有，执行/Capture 选择由 runner 拥有。Backend 暴露模型无关的
  operator 和 support query；model 不能复制 provider 的架构准入谓词。
- 保持 model family 与 backend-native port 隔离。新的 native port 应位于符合当前架构的
  独立 family module，不能导入另一 family 的私有 config、weights、model、runner 或 graph，
  同时应维持共享的 registry/load 用户契约。运行并检查仓库 boundary guard；不能为了放行
  跨 family import 而增加 allowlist。
- Execution/Profile/Cache identity 必须包含所有影响选择的状态：精确动态 shape、dtype、
  layout、必要时的 pointer identity、precision、epilogue/fusion、device capability、
  policy、variant 和 generation。
- Instance config 不能通过全局 Atomic 或环境变量改变已经加载的模型。配置应绑定实例，
  或在 process-wide 限制下明确拒绝冲突配置。
- Registry、alias 和 admission 必须同步；每个接受的模型名和设备路由都执行相同的
  `model_variant`、capability 和 option 校验。
- 显式提供但不支持的 tuning path、variant、precision、sampling 或 backend 选择必须报错，
  不能静默忽略。
- Readiness report 只是观测数据，不是执行证明。成功的 MLX prepare 必须通过
  `TextPreparationStatus` 报告实际 compiled scope 和允许 shape；VLA `PreparationStatus`
  是另一套契约，`RuntimeManaged` 不代表 graph 或 compiled readiness。

## 执行计划、Graph Capture、Autotune 与 Recipe 持久化

- 区分逻辑 capacity 与精确执行 shape。只有所有 native key 和 operator sequence 仍有效时
  才能复用 prepared session；同一 padded capacity 内改变 `M`、token count 或 grid 也可能
  需要重新 prepare。
- Cache 共享范围必须与 native state 范围一致。Process/context 共享的 Rust cache 不能跳过
  存在于 `thread_local`、stream-local 或 device-local 的 native prepare。
- 防止 fused/decomposed、不同 epilogue/provider、dynamic/static decode、shape-pinned/bucketed
  implementation 使用同一个 key。cuda-new Recipe key 必须覆盖所有会影响选择的事实；
  持久化 winner 在 prepare execution 前必须重新解析并验证。
- Graph capture 内必须禁止 prepare/tuning；capture 中 cache miss 应明确失败，不能懒创建
  descriptor 或 benchmark candidate。
- 限制 execution/profile cache 和 device allocation 的增长；检查 eviction、invalidation、
  pointer stability、shape 变化、cold cache 和内存压力。
- 当前 cuda-new 的持久化单元是一个 exact-key `.recipe` 文件，不是旧的合并式
  `tactics.json` 数据库。除非改动路径确实读写旧格式，否则不要套用 JSON record merge、
  version filter 或 stale lock 检查。对 Recipe 应检查：即使文件名仅为 hash，也会保存并校验
  完整 key；损坏或过期 winner 会安全 miss；同进程和跨进程 writer 不会暴露半写文件。
- Fast path 准入必须包含实际编译/Backend availability，并提供永远正确的 fallback；
  不支持组合应在 prepare 阶段拒绝，而不是第一次 launch 才失败。

## MLX 与 Metal Prepared Execution

- 遵守线程限制：MLX `Array`、`Stream` 和 `Compiled` 所有权不能跨线程，也不能虚构
  `Send`/`Sync` 保证；thread-local compiler state 与 process-wide allocator state 要分别分析。
- MLX 操作是 lazy 的。发布 request position 或 cache metadata 前，必须 evaluate 每个输出和
  新 KV/state array；只有 synchronize 不能证明 lazy expression 已执行。
- 验证真实 compiled scope 与 shape profile。Variant 名称或非空 compiled handle 不能证明
  whole-model readiness；prepare 必须覆盖允许的 callable、变化输入、保留状态和失败恢复。
- `MLX_DISABLE_COMPILE` 采用“变量存在即禁用”的规则，包括值为 `0`；不能把 eager callable
  报告为 compiled path。
- 声明内存有界时，应在独立进程中同步后测量 warm resource；区分 lazy temporary array、
  retained model state、compiler cache 与 allocator cache。

## 内存与 Hot Path

- 按 layer/profile 统计 allocation、常驻 arena、重复权重、scratch 和 execution object；
  文档中的内存需求必须覆盖 prepared 与 fallback 路径。
- 查找禁用打印后仍执行的 D2H readback、digest/statistics、synchronization、host interpolation、
  重复转换和 plan 创建。
- 纯 metadata reshape/reinterpret 不应启动 kernel；元素顺序不变时复用 view。
- Fallback-only buffer/plan 应懒创建；fused winner 不应分配或 tune 仅 decomposed 路径使用的资源。
- Benchmark 要覆盖 cold/warm、交替 profile、cache miss、受限内存和重复请求；单一 warm shape
  不能证明生产行为有界。

## 并发、缓存与文件

- 分别分析同一进程内线程、多进程、进程崩溃和重试。
- 临时目录和 staging 路径必须对每个 writer 唯一。发布过程应当原子、可验证，并能
  正确处理另一个 writer 先完成的情况。
- 检查 lock 范围、stale lock、partial artifact、清理所有权、权限、cache key 完整性，
  以及 eviction 后继续使用的问题。
- 修改共享 metadata 前先完成 validation。被拒绝的调用不能更新 mapped decode state、
  cache generation、进程配置或已发布 artifact。
- 区分可恢复的 request validation error 与 runtime poison。只有设备/模型状态已经发生无法
  安全回滚的改变时，一个坏请求才可以阻止后续请求。

## 构建、Feature、ABI、Native Instantiation 与 Vendor Code

- 覆盖真实支持的 Feature matrix：default/CPU、每种 device feature、workspace build 和相关组合。
  Hosted CI 如果从未编译改动的 CUDA/HIP/Metal 路径，就不能提供该路径的构建证据。
- 分析 Cargo Feature Unification；enum variant、match arm 和 dependency 应由真正定义它的
  Feature 控制，不能依赖碰巧相关的 model/backend Feature。
- Backend 迁移需要对比旧路径的 dtype 支持、fast path、配置传递、环境变量、profiling 和
  架构检查。总体变快不能掩盖默认用户路径损坏。
- 对 cuda-new，验证 architecture selection、provider source inclusion、生成/模板实例化覆盖，
  以及目标 SM 的 compile/link。所有会使 Recipe 失效的 candidate/kernel/ABI 输入都必须进入
  build fingerprint，但不能让无关文档或 model-only 修改造成无意义失效。
- 对齐 FFI 两端的 status type、整数宽度、signedness、struct layout/version、optional pointer
  和 ownership；同一 operation family 尽量使用统一错误约定。
- 第三方 kernel 必须记录 upstream repo/revision、原路径、license、必要时的内容 hash 和本地修改。
- 遵循仓库层级职责：device kernel 位于 kernel 层，adapter 负责 validation/resource/dispatch，
  model-specific policy 不进入 shared backend API。

## Python、CLI 与跨语言契约

- Python 公共参数新增时保持 positional compatibility，优先使用末尾 keyword-only 参数。
- CLI/Python 默认值、alias、显式 override 与 Rust admission 保持一致，同时测试默认与非默认值。
- 沿当前语义追踪兼容性选项。尤其是 Python `tactics` 或 Rust `tuning_path` 进入 cuda-new 后，
  可能选择 Recipe directory 或同名 `.recipes` 目录；不能仅凭参数名推断它会读取
  `tactics.json`。
- 显式路径不存在时应报错；synthetic load 不得从 CWD 偷读无关文件。用户指定 artifact 不能静默
  变成 `None` 或 unrelated fallback。
- EOS/budget 边界判断必须使用 generation 实际的 token accounting 顺序；最后一个预算位置生成
  stop token 属于正常完成，不是 truncation。
- 提交文档中不放机器私有路径和 PR checkout 指令；用户说明应从普通仓库 checkout 可执行。

## 测试与 Benchmark

- 数值 kernel 需要 reference comparison；模型支持需要端到端模型测试。包括被拒绝的
  shape、dtype 和错误配置。
- 确认 ignored/hardware test 确实在明确命名的设备和精确 revision 上运行。历史证据
  必须保留原本的适用范围。
- Benchmark 应记录 revision、硬件、OS/SDK、模型和量化格式、数据、命令、warmup、
  重复次数、统计量、baseline、原始输出和正确性检查。
- 优先使用独立 reference。两个一起修改成相同 rounding 的 kernel 不能互为 oracle；同一 binary
  重复得到相同输出也不是 expected-token regression gate。
- 验证实际路由：记录 provider、stream、binary/library hash、model variant、environment gate 和
  实际加载 artifact；不能通过环境变量是否存在来猜测 route。
- Backend/operator stack 迁移后必须比较端到端输出，而不只是单 kernel；覆盖真实 checkpoint、
  non-square/multi-shape、交替请求、stop boundary 和 state reset。
- 用行为测试错误路径。Source-text grep 会被注释欺骗，也会固化临时架构例外；设备测试在不支持的
  host 上应正确 skip/ignore，不能直接 `unwrap()` 创建设备。
- 运行精确 commit 上与改动相关的 architecture/family-boundary guard，并检查 guard 的覆盖范围。
  如果它没有枚举新 family、operator layer、Feature 或 native source，即使通过也不代表覆盖。
- 最终 repair head 需要重新运行受影响的 hardware evidence。源码连续性只能保留有限证据，不能把
  旧运行变成当前 head 的设备资格。

## Scope、文档与 Dead Path

- Probe、一次性实验报告和中间产物不进入 production diff；除非是长期用户文档，否则放在
  `devlocal/`。
- 每个新 operator 都沿 Rust、FFI、adapter、kernel 追踪到 production caller。只有测试或不可达的
  完整调用栈应删除，或拆到真正启用它的 PR。
- Performance/Migration PR 中拒绝无关 family 修改和顺手公共重命名，除非它们是目标路径所必需。
- 对齐 PR description、README、架构文档、环境变量表、测试名和实际代码。放宽 policy rule 是
  实质修改，不能伪装成 cleanup。
- 大规模 delete-and-restore history 会显著降低可审查性；优先拆为 runtime migration、逐模型迁移、
  legacy deletion。

## AI 生成代码诊断

- 验证引用的 API 和符号在固定依赖版本中真实存在。
- 比较近似重复的代码路径，查找遗漏分支或不一致的校验。
- 将注释和 safety proof 同实现核对，而不是只看设计意图。
- 警惕复制实现逻辑的测试、只断言 shape 的测试，或绕开失败集成路径的测试。
- 检查注释是否夸大 core guarantee，而真正 validation 只存在于 backend；检查 thread/device isolation
  声明是否与实际 ownership type 相符。
- 搜索“接收/保存/展示但没有进入 production call”的 option，以及可选权重或 provider 不可用时
  仍无条件声明的 capability。
