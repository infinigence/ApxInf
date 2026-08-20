# π0.5 CUDA Benchmark 规范

> 状态：本文档定义统一的 benchmark workload、调优规则和结果格式。
> 尚未运行的组合保持为 `待测`，不用推算值填充。
> 2026-08-13 已按本文口径完成 Thor SM110 和 Orin SM87 的 2/3 views、T=10/21 baseline。

## 1. 目标

这个 benchmark 回答两个问题：

1. ApxInf 的 π0.5 在 Thor SM110 和 Orin SM87 上的实际推理延迟是多少。
2. 真实 LIBERO 语言长度和预留长度下的算子 shape 分别应该选择什么 tactic。

主成绩使用官方 LIBERO 中的 10-token 指令，以便和 FlashRT 常用的短 prompt workload 对齐。21 tokens 使用官方 LIBERO 最长指令进行真实扩展测试。50/200 tokens 只用于算子 autotune，不运行端到端 benchmark。

## 2. 固定 workload

| 参数 | 固定值 |
|---|---:|
| Batch size | 1 |
| 相机数 | 2 / 3 views |
| 图像 | 224×224 RGB，NHWC `uint8` |
| Action horizon `H` | 10 |
| Action dimension | 32 |
| Flow-matching steps | 10 |
| Token 执行模式 | exact length，不 padding 到 200 |
| 真实 benchmark token 数 `T` | 10 / 21 |
| Autotune-only token shape | 50 / 200 |
| Warm-up | 10 次 |
| 正式采样 | 30 次 |
| 计时统计 | P50 / P95 / min / max / mean / standard deviation |

`H` 只表示 action horizon，在所有测试中都是 10。`T` 表示语言 token 数。

## 3. Token 数据集

| `T` | 数据来源 | 定位 |
|---:|---|---|
| 10 | 官方 LIBERO，共有 10 条指令的 PaliGemma token 长度正好为 10 | **Primary LIBERO** |
| 21 | 官方 LIBERO 的最长指令，共 2 条 | LIBERO worst-case language |
| 50 | 没有对应的 LIBERO 仿真数据 | 只生成算子 shape 并 autotune |
| 200 | 没有对应的 LIBERO 仿真数据 | 只生成算子 shape 并 autotune |

10-token 主成绩可以使用这条官方 LIBERO 指令：

```text
put the bowl on top of the cabinet
```

21-token 扩展测试可以使用：

```text
pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate
```

50 和 200 tokens 没有对应的 LIBERO 仿真任务，因此不构造伪造的端到端数据，不报告延迟、成功率或正确性 PASS。它们只通过 `Pi05ExecutionSchedule` 生成对应的物理 GEMM shape，运行算子 autotune 并保存 tactic。

对于 10 和 21 tokens，正式运行前要将文本、token IDs、tokenizer 哈希和仿真 fixture 哈希一起固定下来。

### 3.1 2026-08-13 Thor baseline 固定 fixture

本次 Thor baseline 使用仓库已有的真实 LIBERO first-replan fixture。实际使用的 prompt 和 token IDs 如下；它们替代上面的可选示例 prompt，不能在复现实验时互换。

| `T` | Fixture | Prompt | PaliGemma token IDs |
|---:|---|---|---|
| 10 | `task_08_first_replan.npz` | `put both moka pots on the stove` | `2,1065,2145,705,1161,37801,611,573,37932,108` |
| 21 | `task_04_first_replan.npz` | `put the white mug on the left plate and put the yellow and white mug on the right plate` | `2,1065,573,2674,24464,611,573,2731,8811,578,2507,573,8123,578,2674,24464,611,573,1833,8811,108` |

固定 artifact SHA256：

| Artifact | SHA256 |
|---|---|
| PaliGemma `tokenizer.model` | `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6` |
| π0.5 checkpoint `model.safetensors` | `21b8711787c4a75861b02cff6aa81675a3a943d32b435a68262ac4461e476ba4` |
| T=10 原始 NPZ fixture | `97f9d8b112605a67277cca65e4cadc06f7fd4ccd5e21f339a215670ea9e56473` |
| T=21 原始 NPZ fixture | `2663c33a3b801a7bf67bdefdea1526fdd9acad8564a0ede5ec98ee10f03381d6` |

图像由上述 fixture 的归一化 patches 确定性还原为 224×224 NHWC `uint8`，再走 ApxInf CUDA preprocessing。3-view 的第三路输入固定复用 wrist 图像，所有 3-view 结果均标记为 **duplicated wrist fixture**，不是 LIBERO 真实第三相机。

## 4. Views 的含义

| Views | 含义 |
|---:|---|
| 2 | LIBERO 真实 workload：base camera + wrist camera |
| 3 | 三相机生产 workload，不是 LIBERO 官方相机配置 |

3-view fixture 必须保存稳定的第三路图像。如果暂时复用 wrist 图像，结果必须标记 `duplicated wrist fixture`，不得把它描述为真实 LIBERO 第三视角。

## 5. 执行路径

| 设备 | 精度路径 | 用途 |
|---|---|---|
| Thor SM110 | BF16 | Thor 高精度基线 |
| Thor SM110 | FP8 native | Thor 原生 FP8 量化路径 |
| Orin SM87 | BF16 | Orin 高精度基线 |
| Orin SM87 | INT8 (W8A8) | Orin 原生 INT8 量化路径 |

真实 benchmark 矩阵共有：

```text
4 条设备/精度路径 × 2 种 views × 2 种真实 token 长度 = 16 组
```

另外有 16 组 `T=50/200` 的 view/device/precision autotune-only profile。它们不计入端到端 benchmark 成绩。

NVFP4 不在当前 ApxInf benchmark 中。ApxInf 目前没有 π0.5 NVFP4 executor、calibration、tactic 或已验证结果。

## 6. 当前已有结果

### 6.1 2026-08-13 Thor SM110 baseline

下表是最新主干代码的 Thor BF16/FP8 baseline。每个延迟单元格为 **P50 / P95**，单位为 ms；每组均为 10 次 warm-up + 30 次正式采样。PASS 同时要求 eager/graph 一致并通过独立 BF16 reference，而不是 graph-only 检查。

| 路径 | Views | `T` | Graph replay P50 / P95 | Input update + graph P50 / P95 | 状态 |
|---|---:|---:|---:|---:|---|
| Thor SM110 BF16 | 2 | 10 | **91.048 / 92.384** | 90.881 / 91.830 | PASS |
| Thor SM110 BF16 | 2 | 21 | **95.040 / 95.726** | 95.030 / 95.970 | PASS |
| Thor SM110 BF16 | 3 | 10 | **96.438 / 97.481** | 96.230 / 97.729 | PASS |
| Thor SM110 BF16 | 3 | 21 | **99.919 / 101.038** | 99.530 / 100.896 | PASS |
| Thor SM110 FP8 native | 2 | 10 | **50.021 / 51.110** | 50.179 / 50.837 | PASS |
| Thor SM110 FP8 native | 2 | 21 | **53.684 / 55.129** | 53.518 / 54.232 | PASS |
| Thor SM110 FP8 native | 3 | 10 | **65.088 / 65.816** | 64.418 / 65.122 | PASS |
| Thor SM110 FP8 native | 3 | 21 | **69.030 / 69.871** | 68.864 / 69.428 | PASS |

独立 BF16 reference 的完整 `[10,32]` normalized-action 误差如下：

| 路径 | Views | `T` | Cosine | Relative L2 | Max abs |
|---|---:|---:|---:|---:|---:|
| BF16 | 2 | 10 | 0.999993 | 0.003823 | 0.003765 |
| BF16 | 2 | 21 | 0.999993 | 0.003826 | 0.004934 |
| BF16 | 3 | 10 | 0.999992 | 0.004177 | 0.004821 |
| BF16 | 3 | 21 | 0.999993 | 0.004402 | 0.004164 |
| FP8 native | 2 | 10 | 0.998866 | 0.052198 | 0.068614 |
| FP8 native | 2 | 21 | 0.999233 | 0.047680 | 0.060927 |
| FP8 native | 3 | 10 | 0.997843 | 0.069410 | 0.064527 |
| FP8 native | 3 | 21 | 0.997998 | 0.064807 | 0.085384 |

八组 eager/graph 输出均逐值一致，`max_abs=0`。FP8 相对同 shape BF16 的 graph replay P50 加速为 1.82×、1.77×、1.48×、1.45×（依次为 2v/T10、2v/T21、3v/T10、3v/T21）。

此前记录的 Thor 和 Orin 数字来自不同代码或 fixture 口径，保留为历史数据，不再填入当前结果矩阵。下面的 Orin 数据是在独立 SM87 设备上按相同固定输入重新测得。

### 6.2 2026-08-13 Orin SM87 baseline

Orin 使用与 Thor 完全相同的固定 LIBERO fixture、token IDs、NHWC `uint8` 图像、BF16 noise 和独立 BF16 reference。每个延迟单元格仍为 **P50 / P95**，单位为 ms；每组均为 10 次 warm-up + 30 次正式采样。

| 路径 | Views | `T` | Graph replay P50 / P95 | Input update + graph P50 / P95 | 状态 |
|---|---:|---:|---:|---:|---|
| Orin SM87 BF16 | 2 | 10 | **213.354 / 215.585** | 213.780 / 216.308 | PASS |
| Orin SM87 BF16 | 2 | 21 | **187.576 / 187.719** | 187.647 / 187.797 | PASS |
| Orin SM87 BF16 | 3 | 10 | **233.883 / 235.411** | 233.638 / 235.434 | PASS |
| Orin SM87 BF16 | 3 | 21 | **232.322 / 232.778** | 232.462 / 233.030 | PASS |
| Orin SM87 INT8 W8A8 | 2 | 10 | **125.280 / 125.341** | 125.359 / 125.416 | FAIL |
| Orin SM87 INT8 W8A8 | 2 | 21 | **125.481 / 125.553** | 125.565 / 125.643 | FAIL |
| Orin SM87 INT8 W8A8 | 3 | 10 | **167.036 / 167.082** | 167.127 / 167.219 | FAIL |
| Orin SM87 INT8 W8A8 | 3 | 21 | **167.823 / 167.930** | 167.939 / 168.009 | FAIL |

独立 BF16 reference 的完整 `[10,32]` normalized-action 误差如下：

| 路径 | Views | `T` | Cosine | Relative L2 | Max abs |
|---|---:|---:|---:|---:|---:|
| BF16 | 2 | 10 | 0.999992 | 0.003998 | 0.003765 |
| BF16 | 2 | 21 | 0.999992 | 0.004028 | 0.005850 |
| BF16 | 3 | 10 | 0.999992 | 0.004455 | 0.005170 |
| BF16 | 3 | 21 | 0.999992 | 0.004534 | 0.004164 |
| INT8 W8A8 | 2 | 10 | 0.972811 | 0.239067 | 0.328767 |
| INT8 W8A8 | 2 | 21 | 0.995328 | 0.098435 | 0.171377 |
| INT8 W8A8 | 3 | 10 | 0.930928 | 0.373479 | 0.447523 |
| INT8 W8A8 | 3 | 21 | 0.993758 | 0.112780 | 0.151288 |

八组 eager/graph 输出均逐值一致，`max_abs=0`。四组 BF16 全部通过正式门限；四组 INT8 都未通过第 11.3 节门限，因此 INT8 延迟只能作为性能诊断，不能作为可发布量化成绩。最接近门限的是 2-view/T=21：cosine 和 relative L2 通过，但 max-abs 0.171377 高于 0.125。

#### Orin INT8 精度结论与 TODO

当前结果不能解释为 Orin 不支持 INT8，也不能归因于 SM87 INT8 kernel 计算错误。诊断中，CUDA Graph 与 eager 输出逐值一致；将 SM87 CUTLASS W8A8 GEMM 强制替换为 cuBLAS 后，八种混合精度组合的最终输出也逐值一致（`max_abs=0`）。问题来自当前朴素 PTQ W8A8 量化算法：权重采用逐输出通道 absmax scale，激活采用逐 token row 动态 absmax scale，且没有 calibration、SmoothQuant、异常值处理或 QAT。

分阶段测试显示，2-view/T=10 下仅量化 vision 和仅量化 action 的最终 relative L2 分别为 0.004364 和 0.012807，而仅量化 language 时达到 0.307038。逐层诊断进一步发现，language 第 0 层量化后的 K/V relative L2 已达到 0.135745/0.163758，第 1 层局部 V relative L2 达到 0.436036；这些误差会写入 prefix KV cache，并被后续层和 10 个去噪步骤重复使用。因此，**当前 full INT8 W8A8 路径精度不足，不能交付；表中延迟仅用于性能与后续量化研究，不得标记为有效发布成绩。** 该结论针对当前量化算法，不代表经过校准和量化优化后的 Orin full INT8 不可行。

TODO：

- 优先验证 language QKV/KV 保留 BF16、vision/action 及 language 其余线性层使用 INT8 的混合精度方案。
- 使用代表性 π0.5/LIBERO workload 建立 activation calibration，并评估 SmoothQuant 或其他 outlier-aware scaling。
- 如仍要求所有 GEMM 使用 INT8，进一步评估更细粒度 activation scale、异常通道处理和 QAT。
- 扩充任务和轨迹覆盖；只有所有正式 workload 均通过第 11.3 节门限，才允许将 Orin INT8 标记为可交付。

BF16 的 T=10 比 T=21 慢约 25.8 ms（2 views）和 1.6 ms（3 views）。针对 2-view 异常另起独立进程复测，T=10/T=21 graph P50 分别为 214.225/187.762 ms，与首轮 213.354/187.576 ms 一致，排除单次采样噪声。该差异可能来自尚未定位的 shape-dependent cuBLAS 算法选择；在完成 exact-shape tactic 固化前，不将其解释为语言长度越短越慢的模型规律。

## 7. 结果矩阵

### 7.1 2 views

| 设备/路径 | T=10 Primary | T=21 |
|---|---:|---:|
| Thor SM110 FP8 native | **50.021 / 50.179 ms, PASS** | **53.684 / 53.518 ms, PASS** |
| Thor SM110 BF16 | **91.048 / 90.881 ms, PASS** | **95.040 / 95.030 ms, PASS** |
| Orin SM87 BF16 | **213.354 / 213.780 ms, PASS** | **187.576 / 187.647 ms, PASS** |
| Orin SM87 INT8 (W8A8) | **125.280 / 125.359 ms, FAIL** | **125.481 / 125.565 ms, FAIL** |

每个单元格中的两个延迟依次为 `graph replay P50 / input update + graph P50`。

### 7.2 3 views

| 设备/路径 | T=10 Primary | T=21 |
|---|---:|---:|
| Thor SM110 FP8 native | **65.088 / 64.418 ms, PASS** | **69.030 / 68.864 ms, PASS** |
| Thor SM110 BF16 | **96.438 / 96.230 ms, PASS** | **99.919 / 99.530 ms, PASS** |
| Orin SM87 BF16 | **233.883 / 233.638 ms, PASS** | **232.322 / 232.462 ms, PASS** |
| Orin SM87 INT8 (W8A8) | **167.036 / 167.127 ms, FAIL** | **167.823 / 167.939 ms, FAIL** |

### 7.3 Autotune-only profiles

| Views | T=50 | T=200 |
|---:|---|---|
| 2 | 只生成 exact shapes 和 tactics | 只生成 exact shapes 和 tactics |
| 3 | 只生成 exact shapes 和 tactics | 只生成 exact shapes 和 tactics |

这些 profile 对四条设备/精度路径分别生成。不填写 graph latency、input-update latency、LIBERO 成功率或端到端正确性状态。

## 8. Autotune 规则

Autotune 不按数据集名称复用，而是按完整的物理算子 key 选择 tactic。至少要包含设备指纹、精度、`M/N/K`、layout、scale mode、epilogue 和 workspace 上限。

两种 view 和四种 token 长度产生的关键 `M` 如下：

| Views | `T` | Vision `M` | Language prefix `M` |
|---:|---:|---:|---:|
| 2 | 10 | 512 | 522 |
| 2 | 21 | 512 | 533 |
| 2 | 50 | 512 | 562 |
| 2 | 200 | 512 | 712 |
| 3 | 10 | 768 | 778 |
| 3 | 21 | 768 | 789 |
| 3 | 50 | 768 | 818 |
| 3 | 200 | 768 | 968 |

Action expert 的关键 `M=10` 在两种 view 中相同。

要求：

1. 每个表中的 exact shape 都要完成 autotune。
2. 正式 benchmark 不允许静默使用相同 `M bucket` 的其他 shape。缺失 exact tactic 时应立即报错。
3. 所有候选 tactic 先和高精度参考验证，只有正确候选才能参与计时。
4. Autotune 和正式 benchmark 必须使用不同进程。前者生成 JSON 后退出，后者在新进程中只读加载。
5. 每个候选使用 10 次 warm-up 和 30 次正式采样，保存全部候选的时间和最终 winner。
6. `kernel_build_id`、GPU、SM、CUDA 或 cuBLAS 版本不匹配时，拒绝加载旧 tactic。

目标文件为：

```text
thor-sm110-bf16-v2-v3-h10.tactics.json
thor-sm110-fp8-native-v2-v3-h10.tactics.json
orin-sm87-bf16-v2-v3-h10.tactics.json
orin-sm87-int8-w8a8-v2-v3-h10.tactics.json
```

现有 `pi05_cutlass_tune` 可以覆盖 Thor FP8 native 的部分 GEMM。Thor BF16、Orin BF16 和 Orin INT8 仍需按各自实际使用的 vendor/CUTLASS 路径生成 exact-shape tactic；没有可选 tactic 的固定 vendor 路径也要记录算法、workspace 和版本信息。在四条路径的 JSON 均生成并校验前，不能宣称正式矩阵已经完成 autotune。

## 9. FP8 calibration

Calibration 和 tactic 不是同一件事：

```text
tactic      = 选择哪个实现最快
calibration = 确定 FP8 数值缩放是否正确
```

Thor FP8 native 在 2/3 views 下的 10/21-token 真实 workload 都要验证 calibration 的数值门限。如果一份 calibration 无法同时覆盖这些 workload，则按 view/token profile 分开保存，不得用 autotune 结果代替 calibration。Orin INT8 不使用 FP8 calibration，但仍必须验证 activation/weight scale 和端到端精度。

T=50/200 不做端到端 calibration 验证，因为没有对应的仿真数据。但 autotune 仍然要使用数值可控的随机/合成矩阵，将每个候选算子的输出与同 shape 高精度 GEMM 参考比较，排除错误 tactic。

## 10. 计时边界

每组结果必须同时报告两种边界：

```text
Graph replay
  = 稳态 CUDA Graph launch + synchronize

Input update + graph
  = 已 resize 的 uint8 图像、token 和 noise 更新
    + CUDA preprocessing
    + graph replay
    + synchronize
```

图像解码、相机旋转和 CPU resize 默认不在这两个边界中。如果另外测量 Python/client 端到端延迟，必须放在独立表格中。

Nsight Systems 和 Nsight Compute 只用于定位瓶颈。Profiler 会改变时序，其时间不能代替未插桩的正式 benchmark。

## 11. 精度 baseline 与正确性

### 11.1 独立数学 baseline

所有精度路径统一对比独立的 OpenPI-compatible BF16 数学参考：

```text
相同 checkpoint
  + 相同预处理图像
  + 相同 token IDs
  + 相同初始 noise
  + H=10
  + 10 flow steps
          ↓
独立 OpenPI-compatible BF16 reference
          ↓
normalized actions [10, 32]
```

| 被测路径 | 主精度 baseline |
|---|---|
| Thor SM110 BF16 | 独立 OpenPI-compatible BF16 reference |
| Thor SM110 FP8 native | 独立 OpenPI-compatible BF16 reference |
| Orin SM87 BF16 | 独立 OpenPI-compatible BF16 reference |
| Orin SM87 INT8 (W8A8) | 独立 OpenPI-compatible BF16 reference |

对比发生在 action 反归一化之前，并且使用完整的 `[10,32]` 输出，不先截取 LIBERO 的前 7 个 action 维度。

设备上的 ApxInf BF16 输出可以作为 INT8/FP8 问题定位的辅助对照，但不取代独立 baseline。否则本地 BF16 实现中的公共错误可能被量化路径一起复制，导致错误的 PASS。

### 11.2 CUDA Graph 执行一致性

每条 ApxInf 路径还要独立比较：

```text
ApxInf eager 执行 ↔ ApxInf CUDA Graph replay
```

这只证明 graph capture/replay 没有改变该 ApxInf 路径的输出，不证明模型数学实现正确。正式 PASS 必须同时通过 eager/graph 和独立 BF16 reference 两道门限。

### 11.3 精度门限

| 路径 | Minimum cosine | Maximum relative L2 | Maximum absolute error |
|---|---:|---:|---:|
| BF16 | 0.999 | 0.05 | 记录，暂不单独设门限 |
| FP8 native | 0.997 | 0.10 | 记录，暂不单独设门限 |
| INT8 (W8A8) | 0.995 | 0.10 | 0.125 |
| eager vs graph | 0.999999 | — | 0.01 |

每个 reference artifact 必须记录 checkpoint、fixture、token IDs、noise、views、`T`、`H`、flow steps、reference 实现版本和 SHA256。

当前的 benchmark 程序允许不传 reference 文件，此时只要 eager/graph 一致就可能输出 `PASS`。这种结果只能标记为 `GRAPH_ONLY_PASS`，不得记为正式精度 PASS。本规范的正式 benchmark 必须提供匹配 shape 的独立 BF16 reference；后续应将其改为程序必需参数。

### 11.4 结果状态

| 状态 | 含义 |
|---|---|
| PASS | eager/graph 一致，且通过指定的 BF16/高精度参考门限 |
| FAIL | 任意必需正确性门限未通过；延迟仅供诊断 |
| GRAPH_ONLY_PASS | 只通过 eager/graph，没有独立 BF16 reference；不是正式精度结果 |

每种 2/3-view、T=10/21 真实 workload 分别执行端到端正确性验证。一个 shape 的 PASS 不能代表其他 shape 也 PASS。T=50/200 只执行算子级 tactic 正确性校验，不产生端到端 PASS/FAIL。

## 12. 设备和可复现性

正式运行前必须：

1. 设置固定功耗模式并锁定 CPU/GPU/EMC 频率。
2. 记录锁频前后状态、温度、功耗模式和是否发生 thermal throttling。
3. 关闭其他 GPU workload，每条精度路径单独进程运行。
4. Autotune 和正式结果在同一频率设置下生成。

结果 JSON 至少保存：

- Git commit 和 dirty-worktree 状态；
- `kernel_build_id`；
- checkpoint、calibration、tactics 和 fixture 的 SHA256；
- tokenizer 文件的 SHA256；
- GPU 名称、SM 版本、CUDA driver/runtime 和 cuBLAS 版本；
- views、`H`、flow steps、`T`、图像布局和计时边界；
- warm-up/采样数、全部原始样本和汇总统计；
- 数值误差和最终 PASS/FAIL/GRAPH_ONLY_PASS 状态。

### 12.1 2026-08-13 Thor baseline 环境

| 项目 | 记录值 |
|---|---|
| GitHub `master` 测试提交 | `f52884b5f9e86a8d5f621f022d138659b6e53d4d` |
| 实际 checkout | `acf689c2ffff66e5a5df39b4951406cea5f4d23d` |
| 两者相同 Git tree | `523312d55349d82df73973fd8128cfd70b023a97` |
| GPU / SM | NVIDIA Thor / compute capability 11.0 |
| Driver / CUDA / cuBLAS | 580.00 / 13.0 / 13.0.0 |
| `kernel_build_id` | `kb1-b23b75f2ffa478deff009c4736779a6f` |
| 编译架构 | `APXINF_CUDA_ARCH=sm_110`, `APXINF_CUDA_ARCH_CUTLASS=sm_110a` |
| 功耗模式 | MAXN |
| CPU / GPU GPC / GPU NVD / EMC | 2.601 / 1.575 / 1.692 / 4.266 GHz，全部锁频 |
| 温度 | 测试前 41°C，测试后 44°C |
| GPU 隔离 | 每条路径独立进程；独占锁与 compute-process 监测均通过 |
| 独立 reference | OpenPI-compatible BF16 mathematical path |
| FP8 calibration SHA256 | `f819475714f6e6fa915ec00f7c34f98b0f1dbf15008e399effed6a1d023b4606` |
| FP8 tactic DB SHA256 | `33f84be77558d919d44b6630377c28f3cd77ae1e9d42374f5f6f77571806a265` |

测试时 GitHub `master` 指向 `f52884b`；之后的 `a223cdb` 只更新 benchmark/design 文档，没有修改运行时代码。benchmark example 为读取固定 fixture、输出全部 raw samples 和正确区分 PASS/GRAPH_ONLY_PASS 做了未提交的测试 harness 改动；模型权重处理、runtime 和 CUDA kernel 源码未修改。

当前结果还有以下限制：

1. 主干提交的 Thor FP8 tactic DB 名义上是 2-view/H=10 profile，没有覆盖 T=21 与 3-view 的全部 exact shape。因此这些是有效的主干 baseline，但不能宣称完整 exact-shape autotune 已完成。
2. T=50/200 autotune-only profile 尚未运行，保持 `待测`。
3. 3-view 是 duplicated wrist fixture，不代表真实三相机 LIBERO workload。
4. NHWC `uint8` 图像由仓库保存的 LIBERO normalized-patch fixture 确定性还原；该转换及其输出在本次运行中固定并记录 SHA256。

### 12.2 2026-08-13 Orin baseline 环境

| 项目 | 记录值 |
|---|---|
| Git checkout | `a223cdb7b385d3b327f6cd827922112534bf53b4` |
| Git tree | `c6bd48eb1dc859cdac0fab71c47b661fd9f78c03` |
| GPU / SM | Jetson AGX Orin / compute capability 8.7，16 SMs |
| Driver / CUDA / cuBLAS | 540.5.0 / 12.6 / 12.6.1 |
| `kernel_build_id` | `kb1-b4c7eaec8af637c5c45a2926fbc3dc28` |
| 编译架构 | `APXINF_CUDA_ARCH=sm_87` |
| 功耗模式 | MAXN |
| CPU / GPU / EMC | 2.2016 / 1.3005 / 3.199 GHz，全部锁频 |
| 温度 | 测试前最高热区 49.9°C，测试后 50.9°C |
| GPU 隔离 | 每条路径独立进程；独占锁与纯数字 compute-PID 监测均通过 |
| Checkpoint / fixture / reference | 与 2026-08-13 Thor baseline 完全相同 |

最新主干在 SM87 构建时会无条件导出只存在于 `apxinf_cutlass_gemm` cfg 下的 `autotune_cutlass_gemm_f16`，导致编译失败。本次测试只应用了最小构建修复：为该 re-export 增加相同的 `#[cfg(apxinf_cutlass_gemm)]`；没有修改模型数学、BF16/INT8 runtime 或 CUDA kernel。benchmark example 另有与 Thor 相同的未提交测试 harness 改动，用于加载固定输入、输出全部 raw samples 和执行独立 reference 门限。

Orin BF16/INT8 的 exact-shape tactic JSON 和 T=50/200 autotune-only profile 尚未实现，不能将本次结果描述为完整 autotune 后的最终性能。

## 13. PyTorch baseline

这一节记录迁移前 PyTorch 实现的性能和数值行为。它不是 ApxInf 的成绩，也不覆盖第 7 节的 ApxInf 结果；它的用途是给后续 Rust 实现提供可重复的迁移基线。


### 13.2 设备专用执行路径

| 设备 | PyTorch 路径 | 说明 |
|---|---|---|
| Thor SM110 | FP8 native | Static FP8 V/L/A，使用预生成的 Thor tactic cache 和 calibration |
| Thor SM110 | BF16 | `compile-bf16-no-quant`，统一 CUDA Graph |
| Orin SM87 | BF16 | 通用 H=10 attention，统一 CUDA Graph |
| Orin SM87 | INT8 W8A8 | V/L/A 动态 INT8，BF16 output，统一 CUDA Graph |


### 13.3 延迟结果

下表每个单元格为 `P50 / P95`，单位为毫秒；状态表示相对同 fixture、同 noise 的 BF16 数学参考是否通过第 11 节门限。

#### 2 views

| 设备/路径 | T=10 Primary | T=21 |
|---|---:|---:|
| Thor SM110 FP8 native | **44.591 / 44.740**, FAIL | **45.151 / 45.328**, FAIL |
| Thor SM110 BF16 | **92.531 / 92.907**, BASELINE | **93.449 / 93.695**, BASELINE |
| Orin SM87 BF16 | **227.937 / 228.075**, BASELINE | **230.375 / 230.685**, BASELINE |
| Orin SM87 INT8 W8A8 | **235.290 / 235.426**, FAIL | **236.591 / 236.699**, PASS |

#### 3 views

| 设备/路径 | T=10 Primary | T=21 |
|---|---:|---:|
| Thor SM110 FP8 native | **57.949 / 58.213**, FAIL | **58.446 / 58.768**, FAIL |
| Thor SM110 BF16 | **99.013 / 100.388**, BASELINE | **99.497 / 100.373**, BASELINE |
| Orin SM87 BF16 | **297.687 / 298.002**, BASELINE | **301.555 / 301.787**, BASELINE |
| Orin SM87 INT8 W8A8 | **322.358 / 322.475**, PASS | **324.348 / 324.578**, PASS |


## 14. 与 FlashRT 比较

和 FlashRT 主要数字最接近的 ApxInf workload 单元格是：

```text
Thor SM110 + 2 views + T=10 + H=10 + 10 flow steps
```

ApxInf 现在包含 Thor FP8 native，因此它是与 FlashRT FP8 最接近的同精度对照；Thor BF16 只和 BF16 对照。即使精度相同，也只有在 prompt、views、H、flow steps、warm-up/采样数、calibration、时钟和计时边界全部一致时，才能做严格横向比较。FlashRT NVFP4 不得与 ApxInf FP8/BF16 混在同一精度排名中。
