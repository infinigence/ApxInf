# 通用 Backend 与平台融合边界

本文定义 PI0.5、Qwen3VL 的通用组网契约和后续迁移边界。阶段一新增接口与验证器，不切换现有 CUDA 执行路径；新增方法默认返回 `UnsupportedOp`。本文中的通用组合在阶段二实现并逐段验证，不能把接口存在视为后端已经支持该算子。

## 两层职责

- `apxinf-core::Backend`：稳定的数学语义、逻辑形状、数值精度和设备操作。接口保持 object-safe；模型通用路径只依赖 `Tensor`、`Backend`、公共配置与缓存接口。
- `CudaBackend` / 后续 `HipBackend`：专用融合、设备布局、库句柄、workspace、launch、graph、tactic 和权重 packing。相同数学算子可调用不同平台库或自定义 kernel；公共接口不携带 CUTLASS、hipBLAS、warp/wave 大小或算法编号。
- 模型：预处理语义、位置 ID、层顺序、条件分支、flow 时间步、生成策略。平台优化以 block 为替换单位，在 prepare 时选择并校验能力；通用路径是替换失败时的明确实现，不能把“不支持”静默转换成 CPU 往返。

现有 `as_any`、CUDA decode graph、PI0.5 专用 executor 在本阶段保留。阶段二把平台选择集中到 prepare/执行计划中，避免在每个公共 OP 内识别模型或在完整模型组网中遍布平台分支。通用路径不要求平台必须支持 graph capture。

## 两模型 OP 盘点

以下是完整前向路径的算子族映射；预处理、采样和缓存管理单独列出，避免只覆盖 Transformer 主干。

| 模型 / 路径 | 通用表达 | 平台优化边界 |
|---|---|---|
| PI0.5 图像预处理、patch embedding | RGB 归一化/patch 排列由共享预处理确定；`cast`、`reshape`、`permute`、`matmul`、bias/position `add` | RGB→patch 融合、投影 epilogue |
| PI0.5 vision block | LayerNorm、QKV matmul+bias、`slice_axis`、独立视图的 full attention、输出投影、residual、LayerNorm、GELU MLP | split-QKV、MHA、bias-residual-LayerNorm、GEMM 激活融合 |
| PI0.5 vision→语言、prefix | 投影、embedding/scale、沿 token 轴 concat、RMSNorm、QKV+RoPE、full MQA、residual、GeGLU | embedding/concat、split-QKV-RoPE、MQA、dual-GEMM-GeGLU、residual-RMS |
| PI0.5 action 条件 | 共享 sinusoidal time embedding、两层 matmul+bias+SiLU、各层 style 投影和切片 | 固定时间步 style 预计算与缓存 |
| PI0.5 action block | AdaRMS 组合、QKV+RoPE、prefix/suffix KV concat、full MQA、gate-residual、GeGLU、最终 AdaRMS | Q-RoPE+KV 写入、gate-residual-AdaRMS、workspace 复用 |
| PI0.5 flow | action 输入/输出投影、Euler `state + dt * velocity`、固定步数循环 | Euler kernel、整段 graph；不得更改时间步或 horizon |
| Qwen3VL vision 输入 | patch matmul+bias、共享位置插值/位置 ID、position add | 固定 grid 的位置数据预计算；不在 hot path 上传重复数据 |
| Qwen3VL vision block | LayerNorm、QKV+bias、切片、2D RoPE、full attention、投影/residual、GELU MLP | split-QKV-RoPE、vision attention、MLP epilogue |
| Qwen3VL merger / deepstack | LayerNorm、reshape、matmul/GELU；保存选定层输出 | merger block、连续 patch 重排；norm 在 reshape 前后的顺序不可交换 |
| Qwen3VL 图文融合 | embedding，按已知图像 token 区间 slice/concat 替换行；deepstack 在这些区间 add 后重新 concat | scatter/copy-add 融合；不能对全部文本行添加视觉特征 |
| Qwen3VL text prefill/decode | RMSNorm、Q/K/V 投影、Q/K head RMSNorm、mRoPE、KV append、causal GQA、输出/residual、SwiGLU | packed QKV、QK norm/RoPE/cache 融合、decode attention、GEMM/MLP、graph |
| Qwen3VL 输出 | 最终 RMSNorm、lm_head、`SamplingBackend`；teacher-forced logits 与 greedy 分别验证 | logits/sampling 融合；RNG 和 token 选择契约不变 |

来源：`crates/apxinf-model/src/pi05/{bf16_executor,bf16_runtime,vla_runtime}.rs`，`crates/apxinf-model/src/qwen3vl/{general,vision,decode_graph}.rs`。当前 Qwen3VL 的 slice、merge、scatter 包含 CPU 往返；该事实不构成新通用实现允许隐式往返的契约。图像 token 区间由模型元数据生成，零长度区间跳过，不调用空 slice。

## 公共张量与数值语义

公共张量为 dense、contiguous、row-major，轴顺序由算子定义，不使用隐藏 stride。`reshape` 保持线性元素顺序，可共享存储；需要换轴时显式 `permute` 并物化。新算子输出不与输入共享可写存储；不得写入输入，包括其 clone/reshape 的别名。缓存、graph 输入更新是单独的显式可变操作。

阶段二通用浮点实现以 F32/F16/BF16 为支持集合，每个后端明确实际支持的子集。FP8/INT8 的格式、scale、量化和累加策略不能套用此浮点契约；现有优化路径保留，其通用化另行设计。不同 dtype 的二元算子先显式 cast，不自动提升；不支持的 dtype/device 返回错误。跨设备传输只能显式调用 transfer API。

| 算子族 | 形状和数值要求 |
|---|---|
| `cast` | 保持形状/设备；F32/F16/BF16 转换采用 nearest-even；同 dtype 也返回独立存储 |
| slice / concat / permute / broadcast | 位保持；`AxisSlice` 为非空 `[start,end)`；concat 除指定轴外各维相同；permute 必须为轴的排列；broadcast 右对齐且只扩展维度 1；输出连续 |
| add / mul / scale / bias | 通用组网显式 broadcast 后执行等形状运算，bias 仅末轴；每个 OP 在 F32 计算后舍入到输出 dtype。需要保留多个步骤之间的 F32 临时值时，先 cast 到 F32，组合后再 cast 回去 |
| matmul | 公共二维 `[M,K] @ [K,N] → [M,N]`，F32 累加；输出为输入 dtype；批次通过显式模型循环或后续独立接口表达，不隐藏 packed/transpose 标记 |
| RMSNorm / LayerNorm | 最末轴归一化，weight/bias `[D]`；F32 统计和 affine，正且有限 eps，最终舍入一次。RMSNorm 为 `x / sqrt(mean(x²)+eps) * weight`；LayerNorm 使用总体方差 |
| SiLU / GELU | F32 中计算 `x*sigmoid(x)` 或 tanh 近似 GELU，最后舍入；GeGLU=`GELU(gate)*up`，SwiGLU=`SiLU(gate)*up`，不得互换 |
| embedding / RoPE | embedding 保持 table dtype，检查 token 范围；RoPE 保持形状和 dtype，采用已有 half-split、mRoPE 和 vision 2D 三种显式位置语义，不靠平台默认推断 |

上表是通用路径的目标规范，不追溯宣称所有既有 CUDA/CPU kernel 已符合。例如既有融合可减少中间 BF16 舍入，F32 GEMM 也须核实是否使用 TF32。阶段二在每个实现接入时校验，平台替换须满足固定误差阈值，而不是要求浮点 reduction 逐 bit 一致。不能为使融合结果过关而改基线阈值。

### Attention

新增 `attention` 接收 Q `[B,Q,Hq,D]`、K/V `[B,K,Hkv,D]`；Q/K/V dtype/device 相同，`Hq % Hkv == 0`，query head `h` 对应 KV head `floor(h/(Hq/Hkv))`。结果形状同 Q。无隐藏 KV 修改或位置推进。

- `Full`：所有 key 可见；PI0.5 prefix、action 使用 full attention，但 action 的 K/V 为 prefix 加本轮 suffix。多视图放入 batch 维，不能跨视图 attention。
- `Causal { q_start, k_start }`：允许 `k_start+j <= q_start+i`，适用于 Q/K 长度不同的增量查询。
- `Additive`：同设备 F32 `[B|1,Hq|1,Q|1,K|1]`，有限 bias 或负无穷；NaN/正无穷非法。全部被 mask 的行输出零。`AttentionOptions::validate()` 只检查结构（shape、dtype、device、storage extent）和标量配置，不扫描张量内容、不传输数据，成本随 rank 而非元素数量增长。
- QK 的 F32 累加结果依次应用 scale、bias/mask，再舍入为 `scores`；softmax 采用 F32 reduction，结果舍入为 `probabilities`；PV 用 F32 累加，再舍入为输入 dtype。中间类型只能是 F32 或输入 dtype，配置中明确记录。不允许悄悄改为 TF32。

`contracts` 提供形状、设备、dtype、容量和溢出验证。验证通过只证明参数结构合法，不证明后端具备实现或数值结果正确。正维度限制适用于新算子，scalar 可用于布局操作；现有 Tensor 对空张量的行为未在本阶段改写。

### 验证入口与实现钩子

`Backend` 只声明 `*_impl` 钩子（`cast_impl`、`slice_axis_impl`、`concat_axis_impl`、`permute_impl`、`broadcast_to_impl`、`attention_impl`），默认返回 `UnsupportedOp`。调用方一律走 `PortableOps` trait：它对每个 `Backend`（含 `dyn Backend`）blanket 实现，先用 `contracts` 校验参数、再分发到 `*_impl`、最后核对返回形状是否与契约一致。后端无法覆盖或跳过校验，只实现 `*_impl` 并假定参数结构已合法。校验先于分发，因此契约违规（如非排列 axes、越界 slice、dtype 不符）报告为对应的结构化错误而非 `UnsupportedOp`，不会被“未实现”掩盖。返回形状核对成本随 rank 增长，release 下保留，使后端 kernel 的形状 bug 直接暴露为错误。

布局算子保持位不变，接受任意 dtype（走 `tensor_storage` 只校验设备与 storage extent）；`cast` 等算术仅接受 F32/F16/BF16（走 `float_tensor`），FP8/INT8 返回 `UnsupportedDType`。`contracts` 的校验错误使用结构化变体：形状不符用 `ShapeMismatch`，dtype 不符用 `UnsupportedDType`/`DTypeMismatch`，无形状可展示的不变量违规用 `Contract(&'static str)`（不分配）。所有 `Backend` 默认方法的“未实现”统一用 `UnsupportedOp(name)`，可经 `Error::unsupported_op()` 识别缺失算子名，无需匹配消息文本。

### Mask 校验时机

`validate_mask_values(&Tensor)` 显式扫描 CPU F32 additive mask 的逻辑元素，复杂度 O(mask.numel())，不展开广播；允许有限 bias 和负无穷，拒绝 NaN/正无穷。它不替代 attention 的 rank 和广播检查。非 CPU mask 明确返回 `UnsupportedDevice`，不隐式下载，也不静默跳过。

阶段二应在 mask 构建或内容更新时校验一次，CPU mask 在上传前校验，跨层复用不重复扫描。设备生成的 mask 由保证该数值契约的受控生成逻辑或显式设备值校验路径负责。校验不是永久有效的标记：任何内容更新（包括共享存储写入）都必须重新保证契约。debug/release 使用相同的显式校验语义，不用 debug-only 扫描掩盖缺失的检查。

### 校验时机:计划期与热路径

校验分成两层,避免把测试已能覆盖的确定性检查重复放进热路径。

- **配置级**(计划期一次):dtype 集合、rank、`Hq % Hkv`、scale、mask 广播兼容、元素数与字节数溢出。这些只由模型 config 和组网结构决定,同一计划内每次调用结果相同,因此由 `AttentionPlan::new()` 在构建执行计划时验证一次。
- **实例级**(每次调用):device、精确 shape、dtype、storage extent。这些必须对实际传入的 tensor 成立,由 `AttentionPlan::check_operands()` 用固定次数的比较完成,无 dim 循环、无分配。

热路径经 `PortableOps::attention_planned(&plan, ...)` 分派;`attention()` 仍保留为便利入口,内部自行建计划后分派,适合非热路径与测试。计划不是可信标记:形状、dtype、options 或 mask 与计划不符一律报错,不会被静默复用。 `attention_planned` 在进入后端实现前检查 backend device 与 plan device 完全一致（含设备编号）。计划固定 mask 类型及 causal 的 `q_start`/`k_start`；位置偏移变化须重建计划并重新验证位置溢出。Additive mask 可替换为同设备、同 shape/dtype 且容量足够的张量，内容仍按前述构建/更新规则独立校验。

其余算子的输出校验改为惰性比较(`check_output` 接受迭代器),只在出错时才为错误消息分配。`validate_permutation`、`validate_concat`、`AxisSlice::validate`、`checked_bytes_iter` 是对应的无分配校验核心;返回 `Vec` 的 `permuted_shape`/`concatenated_shape`/`output_shape` 保留为计划期与测试用的便利封装。

实测(PI0.5 decode 形状 B=1 Q=10 K=522 Hq=8 Hkv=1 D=256,release,两次运行取区间):每次全量校验约 55–57 ns → 计划期分派约 11–12 ns,快 4.5–5.4x;计划构建约 81–83 ns,摊到 36 层约 2.3 ns/op。作为对照,单次 CUDA kernel launch 约 3–10 μs,所以校验在改动前也只占约 1–2%;此处收益在于把重复的确定性检查移出热路径,而非解决瓶颈。`tests/portable_hot_path.rs` 固定这两条性质:分派路径零分配,且计划不可被静默误用。

### 融合分解与舍入位置

PI0.5 style `[3D]` 顺序为 scale、shift、gate。AdaRMS 的数学表达为 `rms(x) * (1+scale) + shift`，gate-residual 为 `residual + gate*projection`。使用 `slice_axis` 取 style，broadcast 到 token 维后组合。为了匹配一次舍入的融合语义，可把输入、style 和 residual 转为 F32，在归一化/乘加完成后显式转回 BF16；不能简单串联多个 BF16 OP 后声称数值相同。

同理，bias-residual-norm 的 residual 物化位置、GeGLU 的激活/乘积精度、Euler 的乘加精度必须成为各 block 的固定执行配方。通用和融合路径按相同输入验证 block 输出及端到端输出；不能仅比较其中一层。实际配方由阶段二对原 CUDA kernel 的测量确定并记录，接口允许显式表达这些转换。

## KV、权重与执行资源归属

缓存的物理布局属于后端。逻辑 K/V 为每层的 `[tokens,Hkv,D]`；append 写入当前 `seq_len` 起的区间，不推进位置。所有层完成一次 prefill/decode 后，由模型统一 `advance(n)` 一次。attention 只读取本次已写入的有效区间；clear 使旧 token 不再可见。若已有 graph 引用缓存地址，clear 必须保留这些地址，或由执行计划显式销毁并重新捕获 graph；不能使缓存 graph 留下悬空指针。既有 API 的容量/错误传播完善不在阶段一改动范围，阶段二必须在通用路径进入前检查容量和层号。跨平台不能复用同一个具体缓存实例。

公共权重保留 canonical 的数学意义：线性层按 `[in,out]`、embedding 按 `[vocab,hidden]`、norm 按 `[hidden]`，载入时完成 checkpoint 方向映射及模型特有 scale 处理。普通 QKV/Gate-Up concat 仍是可解释的逻辑布局；NVIDIA interleave、AMD tile/swizzle、量化 packed buffer 均不是公共权重格式。

平台 prepared weights 持有 canonical 权重的派生数据，不修改公共权重；缓存 key 至少区分权重版本、device/架构、dtype/量化配置、packing 版本和 shape/tactic。workspace、stream、graph、autotune 结果由平台执行计划管理，其生命周期覆盖异步执行和 graph replay。CUDA 与 HIP 不共享二进制策略缓存，也不在公共结构中放置另一平台的占位字段。

阶段一不迁移 `static_bf16_weights` 或 Qwen3VL 现有 packed 字段，避免在接口定稿时同时改变执行与加载。阶段二再把这些归属规则落实到模型/平台边界。

## 验收及实现顺序

1. 阶段一：补全通用接口与静态验证，保留原执行路径；contract 单测验证模型所需 shape、mask、dtype 和错误语义，CPU-only 下可编译且保持 object-safe。
2. 阶段二：实现 CUDA 通用 OP 与两模型通用组网，建立 block 配方/替换点；所有预处理和配置仍沿用原模型语义。
3. 阶段三：HIP 实现同一公共契约，接入同一份模型；阶段四再做平台融合和性能优化。

阶段一最终验收还必须用阶段零冻结的命令对两个 CUDA 模型执行正确性和性能回归。本地契约单测不能替代该验收；结果、环境、输入和命令清单保存在相应 `devlocal/backend-contracts/` 记录中。
