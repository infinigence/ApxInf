# Qwen3.5-2B 请求状态的静态内存推导

日期：2026-10-10。

本文根据固定 bundle 的配置、权重文件头和已安装依赖源码，推导请求缓存的形状与字节数。它不是生成实测，也不提供完整内存上界。结果只用于选择校准边界，不批准 admission envelope、sequence reservation 或模型准入。

本次分析仅使用文件读取和 Python 标准库整数计算。没有导入 MLX、加载 tensor、执行 GPU 工作、修改量化或改变系统内存设置。权重检查只读取 safetensors 的 76,648 字节 JSON header，没有读取 tensor payload。以下源码是依赖行为的参考资料，不是移植实现的指令。

## 固定对象与假设

| 对象 | 本次依据 |
| --- | --- |
| 模型目录 | `~/Downloads/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc` |
| 模型配置 | [本地 config.json][config]；text dtype 为 BF16，24 层，每四层一个 full attention |
| 既有服务模型 revision | `8cea164a6e87242bed9ffdc71dccb09eba74b71f6817b21d93cea6ab387c1682`，见[校准记录](memory-calibration-20261009.md#本次实机观测) |
| Python / MLX / MLX-LM | 3.14.3 / 0.32.1 / 0.31.3；服务[固定版本检查](../../python/apxinf/apxinf/serving/text_worker.py#L105)，依赖 [MLX metadata][mlx-metadata]、[MLX-LM metadata][lm-metadata] |
| 依赖来源目录 | `.apxinf/toolchains/mlx-lm-0.31.3-copies/lib/python3.14/site-packages/` |
| 执行路径 | 当前串行文本路径，batch=1，新建请求缓存，没有 prefix restore、KV 量化、投机解码或多序列合批 |
| 主公式条件 | prefill chunk=256；完整执行至少一次模型 forward；使用实际消费位置，而非仅使用 callback 的 prompt position |

权重文件头的核对项为：embedding `[248320, 2048]`、linear QKV `[6144, 2048]`、conv `[6144, 1, 4]`、full-attention K/V `[512, 2048]`，均为 BF16。`A_log [16]` 为 F32。文件是模型目录中的 `model.safetensors-00001-of-00001.safetensors`；二进制文件头没有源码行号。

当前 adapter 通过公开的 `make_prompt_cache(model)` 建立缓存，并调用 `generate_step`，没有传入 `kv_bits` 或 `max_kv_size`。[adapter 调用](../../python/apxinf/apxinf/serving/text_worker.py#L184)；[依赖默认参数][generate-options]。模型的缓存构造返回十八个 `ArraysCache(size=2)` 和六个 `KVCache`。[模型层选择][layer-choice]；[缓存构造][make-cache]。

配置的 `max_position_embeddings=262144` 不是当前服务的准入长度。adapter 使用部署 context 与模型限制的较小值，并校验输入加输出 allowance。[context 限制](../../python/apxinf/apxinf/serving/text_worker.py#L158)；[请求校验](../../python/apxinf/apxinf/serving/text_worker.py#L594)。本文的 16384 行是既有部署边界的计算示例。

## 状态字节公式

定义 `P` 为准备后的 prompt token 数，`G` 为已经 yield 的输出 token 数，`N` 为缓存实际消费位置，`A` 为 full KV 的分配容量。以下 KiB 和 MiB 分别为 1024 和 1048576 字节。

在完整执行模型 forward 后，三个缓存部分如下。

| 状态 | 单层形状与 dtype | 层数 | 全部层的逻辑字节 |
| --- | --- | --- | --- |
| Full-attention K 和 V | 各 `[1, 2, A, 256]`，BF16 | 6 | `12288 × A` |
| GDN recurrent matrix | `[1, 16, 128, 128]`，FP32 | 18 | `18874368`，即 18 MiB |
| GDN convolution history | `[1, 3, 6144]`，BF16 | 18 | `663552`，即 648 KiB |

Full KV 由两个 KV head、256 head dimension、K/V 两份和两字节元素组成。查询 head 数是 8，不能把它代入 KV head 数。[配置维度][attention-config]；[K/V 形状与缓存更新][attention-cache]。

GDN 的 key/value head dimension 都为 128，head 数都为 16。QKV 投影宽度为 `2×16×128 + 16×128 = 6144`。卷积核宽度为 4，因此保留三个位置。[配置维度][linear-config]；[卷积状态][conv-state]。循环矩阵显式以 FP32 初始化，kernel 输出保持 state dtype；不能因为模型权重是 BF16，就按两字节计算循环矩阵。[循环矩阵初始化][recurrent-init]；[kernel 输出类型][recurrent-output]。

完成初始化后的固定状态项因此为：

```text
fixed_state = 18 MiB + 648 KiB = 19,537,920 bytes
cache_payload(N) = 19,537,920 + 12,288 × A(N)
```

这个固定项按每条已执行的序列计。创建空缓存时，`KVCache` 尚无数组，`ArraysCache` 两项均为 `None`，因此此时 payload 为零。中途在某一层失败时，也不能假定全部十八层均已初始化。[空 KV][kv-growth]；[空 ArraysCache][arrays-empty]。

这些数值描述 `nbytes` 的逻辑数组容量，不证明实际 buffer 的唯一占有量。`KVCache.nbytes` 包含未填满的容量，`ArraysCache.nbytes` 是现有数组字节之和。[KV 字节接口][kv-nbytes]；[ArraysCache 字节接口][arrays-nbytes]。当前 observer 将这些值单独记录，不与 allocator active 相加；循环缓存没有受支持的 offset 时保留 null。[observer](../../python/apxinf/apxinf/serving/text_worker.py#L253)。

## 容量增长与 P/G 边界

`KVCache.step=256`。在当前 chunk=256、新建且没有裁剪的路径中：

```text
A(N) = 256 × ceil(N / 256), N > 0
每增加一个容量块：6 层 KV 合计增加 3,145,728 bytes，即 3 MiB
```

| 实际位置 N | KV 容量 A | 全部 full KV | 加上固定状态后的 payload |
| --- | --- | --- | --- |
| 1～256 | 256 | 3 MiB | 21.6328125 MiB |
| 257～512 | 512 | 6 MiB | 24.6328125 MiB |
| 513～768 | 768 | 9 MiB | 27.6328125 MiB |
| 2048 | 2048 | 24 MiB | 42.6328125 MiB |
| 8192 | 8192 | 96 MiB | 114.6328125 MiB |
| 16384 | 16384 | 192 MiB | 210.6328125 MiB |

不能把上述整块公式直接推广到任意 prefill chunk。依赖在容量不足时，保留此前已消费的 `prev` 个位置，再增加 `256×ceil(本次输入数/256)` 的容量。例如 chunk=255 时，第二块结束的位置为 510，而容量为 511。没有裁剪或 restore 时，每次扩容后的尾部余量最多为 255 个位置，但容量起点未必是 256 的倍数。[扩容路径][kv-growth]。

固定 `generate_step` 在首次 yield 前已执行下一个 token 的 forward。因此正常生成 `G≥1` 个输出后，同步后的缓存位置是 `N=P+G`，而不是 `P+G-1`。最终 prompt callback 报告 `P` 时，也可能已经消费到 `P+1`；此时输出 yield 数仍为零。[预计算与 yield 次序][lookahead]；[adapter 位置记录](../../python/apxinf/apxinf/serving/text_worker.py#L215)。

EOS 或取消后，应用已看到的输出数量不能替代物理 offset。尤其是在最终 prompt callback 内停止时，`G=0` 仍可能对应 `N=P+1`。零输出 allowance 则是另一种情况：当前服务及校准工具不调用生成器，因此它不是 prefill-only 测量。[校准语义](memory-calibration-v0.1.md#explicit-plans-and-seed-messages)；[零输出分支](../../benchmarks/serving/memory_calibration.py#L335)。

## 实测优先检查的边界

以下是待验证点，不是已成功完成的实验。继续实测仍需通过当前 host pressure 门。

| 待比较的形状或阶段 | 应验证的区别 |
| --- | --- |
| `P=1,G=1`，与 loaded / settled 对比 | 首次完整 forward 初始化固定 18.6328125 MiB 状态及首个 3 MiB KV 块；同时观察初次执行 workspace |
| `P=254/255/256,G=1` | 最终位置 255/256/257，跨 KV 容量边界；最后一项应增加一个 3 MiB 块 |
| `P=255,G=1/2` | 同一 prompt 的 decode 从位置 256 到 257；将 KV 扩容与 prompt 长度变化分开 |
| `P=256/257/258,G=1` | prefill 循环只处理 `P-1` 个位置；chunk=256 时，P=257 刚好一个完整块，P=258 才需要第二个 prefill 调用 |
| `P=511/512/513,G=1` | 检查后续容量块及 callback / 首输出边界，避免只验证首次分配 |
| `P=2048、8192` 与接近 `P+G=16384` | 检查随上下文增长的 KV 和 attention workspace，不能只外推短请求 |
| 相同 `P+G`，不同 P/G 比例 | 状态容量可能相同，prefill 与 decode 的临时数组及峰值仍可不同 |
| prefill stop、first-output stop、正常终止、settled | 区分中间位置、未 yield 的预计算、缓存释放和 allocator reserve |
| 首请求与相同形状的后续请求 | 检查编译、allocator 复用、清理后的残留增长；不可把首请求峰值直接代表全部后续请求 |

当前默认计划包含 P 的 255/256/257 与 chunk 附近点，但 P=258 不一定自动出现。应检查具体 plan；不能仅凭“覆盖 chunk±1”判断覆盖了上述第二次 prefill 调用。[计划生成](../../benchmarks/serving/memory_calibration.py#L49)。当前输出样本为首个输出、每十六个输出及 terminal，若要检查指定 decode 跨块，可通过该点终止的独立 case 获取 terminal 样本。[采样点](../../benchmarks/serving/memory_calibration.py#L345)。

## 静态缓存公式不能给出的峰值

1. **KV 扩容的新旧数组并存。** 扩容使用新数组和拼接；旧数组、追加块与结果何时释放取决于求值和 buffer 复用。最终 `nbytes` 只包含最终缓存，不能代表扩容峰值。[扩容路径][kv-growth]。
2. **GDN 更新的临时数组。** QKV、z、卷积输入、归一化、gate 和旧/新循环矩阵均有执行期生命周期。循环矩阵大小固定，不代表每步只需要一份矩阵。[投影到更新][conv-state]；[kernel 输入与输出][recurrent-output]。
3. **Prefill 的 attention 和 MLP workspace。** Q、K、V、gate 及两个 MLP 投影随当次 chunk 长度增长；attention 还受历史 KV 长度影响。公开 SDPA 接口支持 GQA，并在 FP32 做 softmax，但没有承诺固定 workspace 上界。不能把完整 attention score 矩阵一定物化或一定不物化写入预算。[attention 路径][attention-cache]；[MLP](../../.apxinf/toolchains/mlx-lm-0.31.3-copies/lib/python3.14/site-packages/mlx_lm/models/qwen3_next.py#L161)；[公开 SDPA 接口][sdpa]。
4. **Logits 与 lazy graph。** 文本模型定义了 vocabulary 投影，vocab 为 248320。prefill 调用的返回值未直接使用，只求值 cache state；不能把 `chunk×vocab` 张量视为必然全部物化，也不能由源码形状证明整个图的峰值。[模型输出](../../.apxinf/toolchains/mlx-lm-0.31.3-copies/lib/python3.14/site-packages/mlx_lm/models/qwen3_5.py#L287)；[prefill 求值][prefill]。
5. **异步预计算与观测同步。** 生成器在 yield 前启动后续计算。observer 同步所拥有的 streams，可能改变任务重叠、旧 buffer 寿命和 allocator 复用。插桩后的峰值与未插桩服务峰值不能无条件等同。[预计算][lookahead]；[同步观测](../../python/apxinf/apxinf/serving/text_worker.py#L246)。
6. **Allocator、Metal 与宿主开销。** `active`、`cache`、历史 `peak`、进程 RSS/footprint 和系统 swap 是不同口径。碎片、allocator 保留、runtime/编译开销、tokenizer 和 CPU 对象不在缓存公式内。公开 `set_memory_limit` 是 graph evaluation guideline，不是硬上界。[allocator 接口][allocator]；[memory limit](../../.apxinf/toolchains/mlx-lm-0.31.3-copies/lib/python3.14/site-packages/mlx/core/__init__.pyi#L828)。

既有记录只证明该次模型加载后约 3.764 GB allocator active，并在 `loaded` 阶段因 host pressure 停止；请求缓存 payload 为零。它没有验证本文的任意生成状态字节、长上下文峰值或 workspace。[既有实机观测](memory-calibration-20261009.md#本次实机观测)。任何在 loaded 阶段停止的后续运行，同样不能作为上述生成公式的动态验证。

本文的输出是固定版本下的可核查预测。后续应先比较逐层 type / offset / nbytes，再分别比较 allocator 与宿主观测；不能将它们重复相加，也不能从有限形状的成功运行批准完整内存 envelope。

[config]: ../../../../Downloads/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc/config.json#L7
[attention-config]: ../../../../Downloads/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc/config.json#L55
[linear-config]: ../../../../Downloads/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc/config.json#L45
[mlx-metadata]: ../../.apxinf/toolchains/mlx-lm-0.31.3-copies/lib/python3.14/site-packages/mlx-0.32.1.dist-info/METADATA#L3
[lm-metadata]: ../../.apxinf/toolchains/mlx-lm-0.31.3-copies/lib/python3.14/site-packages/mlx_lm-0.31.3.dist-info/METADATA#L3
[generate-options]: ../../.apxinf/toolchains/mlx-lm-0.31.3-copies/lib/python3.14/site-packages/mlx_lm/generate.py#L307
[layer-choice]: ../../.apxinf/toolchains/mlx-lm-0.31.3-copies/lib/python3.14/site-packages/mlx_lm/models/qwen3_5.py#L209
[make-cache]: ../../.apxinf/toolchains/mlx-lm-0.31.3-copies/lib/python3.14/site-packages/mlx_lm/models/qwen3_5.py#L304
[attention-cache]: ../../.apxinf/toolchains/mlx-lm-0.31.3-copies/lib/python3.14/site-packages/mlx_lm/models/qwen3_next.py#L121
[conv-state]: ../../.apxinf/toolchains/mlx-lm-0.31.3-copies/lib/python3.14/site-packages/mlx_lm/models/qwen3_5.py#L143
[recurrent-init]: ../../.apxinf/toolchains/mlx-lm-0.31.3-copies/lib/python3.14/site-packages/mlx_lm/models/gated_delta.py#L262
[recurrent-output]: ../../.apxinf/toolchains/mlx-lm-0.31.3-copies/lib/python3.14/site-packages/mlx_lm/models/gated_delta.py#L171
[kv-growth]: ../../.apxinf/toolchains/mlx-lm-0.31.3-copies/lib/python3.14/site-packages/mlx_lm/models/cache.py#L325
[kv-nbytes]: ../../.apxinf/toolchains/mlx-lm-0.31.3-copies/lib/python3.14/site-packages/mlx_lm/models/cache.py#L403
[arrays-empty]: ../../.apxinf/toolchains/mlx-lm-0.31.3-copies/lib/python3.14/site-packages/mlx_lm/models/cache.py#L594
[arrays-nbytes]: ../../.apxinf/toolchains/mlx-lm-0.31.3-copies/lib/python3.14/site-packages/mlx_lm/models/cache.py#L726
[prefill]: ../../.apxinf/toolchains/mlx-lm-0.31.3-copies/lib/python3.14/site-packages/mlx_lm/generate.py#L424
[lookahead]: ../../.apxinf/toolchains/mlx-lm-0.31.3-copies/lib/python3.14/site-packages/mlx_lm/generate.py#L453
[sdpa]: ../../.apxinf/toolchains/mlx-lm-0.31.3-copies/lib/python3.14/site-packages/mlx/core/fast.pyi#L79
[allocator]: ../../.apxinf/toolchains/mlx-lm-0.31.3-copies/lib/python3.14/site-packages/mlx/core/__init__.pyi#L801

## 2026-10-10 stage01 动态核对

本节单独记录实测工件与上文预测的比较，不改变静态推导的适用边界。核对过程只读取 JSON 并用标准库计算，没有再次运行模型或 GPU。原始工件位于本机临时目录 `/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/stage01`，不是已归档数据集。

依据为 [calibration.json](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/stage01/calibration.json)、[samples.jsonl](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/stage01/samples.jsonl)、[process.json](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/stage01/process.json) 和相邻的 [coverage](/private/tmp/apxinf-memory-evidence-20261010-Ds4wOQ/stage01-coverage.json)。三个原始文件的 SHA-256 均与 coverage 记录一致。samples 的 SHA-256 为 `6171acdb99dd1938b24dc82f11a9d7d7744d50819065c1c98b2d63ea4393f379`。

该运行的 model revision、MLX/MLX-LM 版本、BF16 bundle、chunk=256、batch=1 与上文对象一致。五个用例均为 completed，其中四个调用生成器，三个达到计划 allowance，两个提前 EOS。`G` 包含实际 yield 的 EOS；`P=32,Gmax=1` 的 EOS 正好达到 allowance，因此不属于提前 EOS。

| P | 输出 allowance | 实际 G / 终止原因 | 终态 N | 终态缓存 payload，bytes | 本次生成 epoch 的 allocator peak，bytes |
| --- | --- | --- | --- | --- | --- |
| 1 | 0 | 0 / zero_output | 0，未执行 | 0 | 无新 epoch，继承加载峰值 |
| 1 | 1 | 1 / length | 2 | 22,683,648 | 3,795,527,502 |
| 32 | 1 | 1 / eos | 33 | 22,683,648 | 3,854,909,954 |
| 32 | 32 | 1 / 提前 eos | 33 | 22,683,648 | 3,854,909,954 |
| 33 | 32 | 4 / 提前 eos | 37 | 22,683,648 | 3,856,109,044 |

原始 terminal 样本分别是 samples 第 4、10、17、24、31 行。四次生成共检查 96 个层记录：每次十八个 `ArraysCache` 的 `nbytes=1,085,440`、`offset=null`，六个 `KVCache` 的 `nbytes=524,288`、`offset=P+G`。层位置也与每四层一个 full attention 一致。

因此，四次生成的固定状态合计均为 `18×1,085,440=19,537,920 bytes`，KV 合计均为 `6×524,288=3,145,728 bytes`，总和与公式完全相等。逐层观测只导出 ArraysCache 的合计，没有分别导出两个内部数组的 shape/dtype；它验证固定项的合计，不能单独证明两个组成项的 dtype。

四个最终 prompt callback 样本在 `output_tokens=0` 时，KV offset 已分别为 2、33、33、34，比报告的 prompt position 大一。这验证了本次运行中的首输出预计算现象；不能用 callback 的 P 直接代替物理 offset。

加载样本的 allocator active 为 `3,763,655,368 bytes`、cache 为 `4,086 bytes`。五个 settled 样本的请求 `layers=[]`、`cache_payload_bytes=0`。零输出用例保留原加载值；四次生成清理后的 allocator cache 均为零，active 均为 `3,763,655,370 bytes`，较加载样本高 2 bytes。该差值在这四次清理后没有继续增长，但记录没有标识这 2 bytes 的所有者，也不足以证明长期无泄漏。settled 原始样本是第 5、11、18、25、32 行。

全部 32 条样本的 host pressure 均为 normal。process 记录 PID/PGID 94167、returncode 0、`reaped=true`、`remaining_members=[]`，未发送清理信号。

这批数据验证了短请求的固定状态、首个 KV 容量块、`P+实际G` 和清理后请求缓存归零。实际最大位置仅为 37；它没有验证 256 边界扩容、完整 32-token 输出、第二个 prefill chunk 或长上下文峰值。相同缓存 payload 对应不同 allocator peak，也表明缓存公式不能代替执行峰值。本节不批准 admission envelope。
