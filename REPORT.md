# REPORT.md — ApxInf-Qwen38-4090 推理内核优化报告

- 项目：项目6 - Agent for Kernel
- 作品：ApxInf-Qwen38-4090（Qwen3.8-27B-AWQ-INT4 单卡 RTX 4090 推理内核）
- 代码实现 revision：`8203c709938177440b61f4c1cf40f29504ddc206`（分支 APXinf-Contest-2026）
- 本报告随文档 commit 一并入库；验收以提交材料中的完整 commit SHA 为准。

## 0. 环境与模型

| 项 | 值 |
| --- | --- |
| GPU | 1 x NVIDIA RTX 4090 (24 GB) |
| 模型 | Qwen3.8-27B-AWQ-INT4（路径以 `<MODEL_DIR>` 表示，不随包提交权重） |
| 结构 | 64 层 = 16 full-attention + 48 GatedDeltaNet（线性注意力），hidden 5120，24 头 / 4 KV 头，head_dim 256，vocab 248320 |
| 引擎上限 | MAX_LEN = 16384 tokens（prefill 上限，超出返回 400） |
| 构建 | Rust + CUDA（`cargo build --release`），CUDA graph decode，W4A16 packed 权重驻留 |

## 1. Baseline

两个参照点（同一机器、同一官方评测 harness、public_calibration profile）：

### 1.1 初始可运行实现（rev `6dee706`，run 20260825-123707）

| 指标 | 1K | 2K | 4K | 8K | 16K |
| --- | --- | --- | --- | --- | --- |
| TTFT (s) | 1.525 | 2.647 | 5.648 | 14.536 | 44.394 |
| TPOT (ms) | 698.5 | 706.5 | 723.4 | 757.5 | 828.2 |

峰值显存 23252 MiB；public 6/6，轨迹 256/256。

### 1.2 本次最终改动前的直接基线（rev `4b71212`，run 20260827-042039）

| 指标 | 1K | 2K | 4K | 8K | 16K |
| --- | --- | --- | --- | --- | --- |
| TTFT (s) | 0.513 | 0.914 | 1.760 | 3.479 | 7.032 |
| TPOT (ms) | 53.3 | 61.8 | 78.7 | 112.4 | 180.1 |

峰值显存 23246–23248 MiB；public 6/6，轨迹 256/256，协议检查通过，成功率 1.0。

## 2. 假设（Hypothesis）

1. **TTFT（prefill）瓶颈假设**：16K prefill 从 44.4 s 起步，分桶剖析显示朴素注意力、逐元素/反量化、embedding gather、GDN recurrence 各占大头；假设依次替换为 FlashAttention-2 causal fp16（L>128）、融合反量化、f16 host-embed gather、recurrence v3 后，TTFT 可压到 ~7 s；剩余部分由 W4A16 量化 GEMM（16K 时约 4.75 s）主导。
2. **TPOT（decode）瓶颈假设**：M=1 解码的 GEMV 若走慢路径会把 TPOT 推到 ~700 ms；恢复融合 W4A16 m=1 GEMV 后，短上下文 TPOT 应降到 ~50 ms。此后 TPOT 随上下文长度线性上升（1K→16K：53→180 ms），说明长上下文下的瓶颈转移到 16 个 full-attention 层对 16K KV cache 的单 query 注意力（访存受限且单 CTA 串行）；假设采用 split-KV FlashDecoding 风格并行（按 KV 块切分到多个 CTA 做局部 softmax 再归约）可把 16K TPOT 压到接近短上下文水平。

## 3. 实现（设计变化及影响的执行阶段）

按提交顺序的关键改动（完整列表见 `git log`）：

| commit | 内容 | 影响的执行阶段 | 实测效果 |
| --- | --- | --- | --- |
| `21a90cf` | 修复 GatedDeltaNet conv 方向与 gated-norm gate；fp16 kernel 存储 | 正确性基础（prefill+decode 的 48 个 GDN 层） | 轨迹对齐参考实现 |
| `97f6d1b` | 真 per-token SSE 流式；16K prefill；workspace 显存复用 | 服务层 + prefill | 支持满 16K，显存不随请求膨胀 |
| `344d7d2` | 确定性 GEMM 累加（短上下文位级一致） | prefill/decode GEMM | 轨迹可复现 |
| `8fa29ce` | decode 的 linear a/b 投影改回 cuBLAS | decode GEMM | -27 ms/token（融合版反而回退，见失败实验） |
| `6dee706` | 修复 16K elementwise 截断 | prefill/decode 全长 | 16K 正确性闭环 |
| `0007373` | 恢复融合 W4A16 m=1 GEMV（gemm_q4） | decode 全部线性层 | TPOT 698 ms → 53 ms |
| `9a067d1` | prefill 接入 FlashAttention-2 causal fp16（L>128） | prefill 16 个 full-attn 层 | 16K TTFT 44.4 s → 13.6 s |
| `4b71212` | f16 host-embed gather + dequant v2 + recurrence v3 | prefill 嵌入/反量化/GDN | 16K TTFT 13.6 s → 7.2 s；1K 1.40 s → 0.59 s |
| `8203c70`（本次最终改动） | decode split-KV FlashDecoding 风格注意力（full-attn 层） | decode 阶段 16 个 full-attn 层，全长度生效 | 16K TPOT 180.1 → 46.5 ms，各长度 TPOT 拉平 ~45 ms |

split-KV 设计要点：把当前层的 KV 序列按块切分给多个 CTA 并行，各块计算局部 (max, sum, partial-out)，再用一个小归约核合成最终输出；归约缓冲为 O(splits × heads × head_dim) 的临时张量，不随批处理常驻。仅改 decode 路径，prefill 仍走 FA2，不改变数值合同（轨迹 256/256 保持）。

## 4. 测量（Measurement）

- 评测工具：官方 `evaluation/run_evaluation.py`，profile=`public_calibration`，数据集 `.cache/public`。
- 正确性：6 个公开用例（含 1K NIAH 三针位与 16K 解码用例）+ 256 条 token 轨迹对参考实现逐位比对 + 协议检查。
- 延迟：5 个长度档（1K/2K/4K/8K/16K）的 TTFT 与 TPOT；显存取进程峰值。
- 计分：官方 `evaluation/score_submission.py`，profile=`public_calibration`。

## 5. 结果（Results）

### 5.1 最终成绩（rev `8203c70`，run 20260827-092538）

| 指标 | 1K | 2K | 4K | 8K | 16K | 对比直接基线 |
| --- | --- | --- | --- | --- | --- | --- |
| TTFT (s) | 0.509 | 0.910 | 1.745 | 3.452 | 7.003 | 基本持平（-0.4%~-1.4%） |
| TPOT (ms) | 44.6 | 44.7 | 45.0 | 45.5 | 46.5 | 1K -16%，16K -74%（3.9x），且不再随长度上升 |

峰值显存 23248–23250 MiB（与基线持平）；public 6/6；轨迹 256/256；协议通过；请求成功率 1.0。

### 5.2 计分诊断（public_calibration，单提交、无 vLLM control，provisional）

| 板块 | 得分 | 说明 |
| --- | --- | --- |
| correctness | 30 / 30 | public 20 + public 轨迹 5 + 协议 5（hidden 部分该 profile 不计分） |
| TTFT | 35 / 35 | 1K/2K/4K/8K/16K 全部达标（5+5+7+8+10） |
| TPOT | 25 / 25 | 1K 与 8K 达标（10+15） |
| reliability | 10 / 10 | success rate 1.0；no_fallback / no_nan / no_unexpected_oom / no_xid / service_healthy_after_failure 全部为真 |
| bonus | 0 | 32K+ 长上下文、多请求、多模态均未提交 |
| **诊断总分** | **100 / 100** | 折算课程点 **80 / 80** |

说明：该分为 public_calibration 的单提交诊断分（provisional），正式榜还需 hidden correctness 与平台 control 校准；此前轨迹参考不匹配的一次诊断约为 95 分（轨迹 0/256），本次已用匹配的轨迹参考拿满。

### 5.3 稳定性 / 显存权衡

- 连续评测全程无 Xid、无 NaN、无意外 OOM；失败请求后服务即恢复（见 §6 负控制）。
- 显存：权重以 W4A16 packed 驻留 + fp16 激活/KV，峰值 ~23.2 GB，已接近 24 GB 卡的安全上限；因此未再启用更长的上下文或更大的 KV 预算（见已知限制）。
- 曾做文本显存压缩（量化 KV / 权重进一步压缩）的初步评估：可腾出的显存有限且对延迟/正确性有代价，在截止时间前不足以支撑 32K 或视觉分支，故未合入（B 线，不 commit）。

## 6. 负控制与回归测试

脚本：`negctl.py`（对运行中的服务发起滥用流量并验证恢复），命令与结果：

```
$ python3 negctl.py     # 目标 127.0.0.1:8030
PASS over_capacity_rejected :: status=400 body={"error":{"message":"input length 20000 exceeds supported prefill length 16384",...}}
PASS malformed_json_rejected :: status=400 body={"error":{"message":"malformed JSON body",...}}
PASS unknown_route_rejected :: status=501 body={"error":{"message":"endpoint not supported",...}}
PASS bad_field_type_rejected :: status=400 body={"error":{"message":"input_ids must be an array",...}}
PASS zero_max_new_tokens_handled :: status=400 body={"error":{"message":"max_new_tokens out of supported range",...}}
PASS recovery_normal_request :: status=200 in 0.900s body={"output_ids":[...],"type":"result",...}
negctl summary: 6/6 passed
```

要点：超容量（20000 > 16384）、畸形 JSON、未知路由、错误字段类型、零长度生成全部被结构化错误干净拒绝，服务不崩溃；随后正常请求立即恢复（200，0.9 s）。

## 7. test.py check 与 test.py run

### 7.1 test.py check（通过）

```
$ cd benchmarks/qwen38_4090 && python3 test.py check
... (cargo check --profile dev)
assignment checks passed
```

### 7.2 test.py run（按设计需要轨迹参考）

```
$ python3 test.py run
ValueError: provide --trajectory-reference, or capture one from the vLLM control
```

这不是缺陷：契约要求以 vLLM control 采集的轨迹作为参考。实际评测改用等价的直接命令：

```
$ cd benchmarks/qwen38_4090/evaluation
$ python3 run_evaluation.py \
    --dataset .cache/public \
    --model-dir <MODEL_DIR> \
    --base-url http://127.0.0.1:8030 \
    --implementation-name apxinf-student \
    --implementation-revision 8203c709938177440b61f4c1cf40f29504ddc206 \
    --backend apxinf \
    --profile public_calibration \
    --trajectory-reference <traj_ref.json> \
    --output-dir runs
# => public_correctness 6/6, public_trajectory 256/256,
#    request_success_rate 1.0, EVAL_EXIT=0
```

随后计分：

```
$ python3 score_submission.py \
    --submission runs/20260827-092538-apxinf/submission.json \
    --profile public_calibration
# => diagnostic_score 100.0, automated_course_points 80.0（见 §5.2）
```

## 8. 已知限制、失败实验与回滚

### 已知限制
- 上下文上限 16384：受 24 GB 显存与 KV 预算约束；32K+ 加分项经可行性评估（需重排显存/压缩权重）在剩余时间内无法可靠完成，未做。
- 单流评测：多请求并发（c4/c8 goodput）加分项未实现（计分输出中为 missing_cell）。
- 多模态（图片）加分项：显存预算不足，已放弃。
- TTFT 仍由 W4A16 量化 GEMM 主导（16K 约 4.75 s / 7.0 s）；进一步优化需要真正的 SM89 packed W4A16 Tensor Core GEMM，当前仓库没有可直接复用的该后端。

### 失败实验（未合入或已回退）
1. decode linear a/b 投影的融合 kernel：比 cuBLAS 慢 27 ms/token，回退（`8fa29ce`）。
2. BF16 dense staging + cuBLAS gemmEx 的 Tensor Core 实验路径：数值正确但 decode 明显慢于 packed fused 基线，仅用于方向评估，不作为权重驻留方案。
3. group-32 元数据复用首版用错 warp mask 导致尾部 lane 挂起，改用 `__activemask()` 修复（过程记录在开发日志）。
4. B 线显存压缩（量化 KV 等）：初步评估收益有限且影响延迟/正确性裕度，未 commit。

### 回滚方法
- 回到最终改动前：`git revert 8203c70` 或 `git checkout 4b71212`，重新 `cargo build --release` 后按 §9 重启服务即可，无数据迁移。
- 单个历史改动均可按同样方式 revert；评测数据全部由官方 harness 重新生成，无状态残留。

## 9. 复现步骤

1. 准备：Rust 工具链 + CUDA 12.x，单卡 RTX 4090（24 GB），模型目录 `<MODEL_DIR>`。
2. 构建：`cargo build --release`（约 24 分钟）。
3. 启动服务：
   `./target/release/apxinf serve --addr 127.0.0.1:8030 --model <MODEL_DIR> --device cuda`
   （约 81 s 就绪，GPU 占用 ~23.2 GB。）
4. 检查：`python3 test.py check` → `assignment checks passed`。
5. 轨迹参考：由 vLLM control 按契约采集一次（本提交使用与基线一致的参考文件）。
6. 评测与计分：按 §7.2 的两条命令执行。
7. 可选：`python3 negctl.py` 复现负控制（脚本已随提交包提供于 `evidence/negctl.py`）。

## 10. 后续可优化方向（超出本次提交范围）

- SM89 packed W4A16 Tensor Core dequant-in-GEMM（TTFT 的最大剩余项）。
- split-KV 的 split 数按长度自适应 + 归约核融合（短长度再降 1–2 ms）。
- 多请求调度（连续批处理）以争取 goodput 加分。
- 若换更大显存平台：32K 上下文与视觉分支。
