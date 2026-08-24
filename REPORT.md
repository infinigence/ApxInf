# ApxInf 实现报告（REPORT.md）

本文按 `README.md` §7 要求记录：baseline、假设、实现、测量、结果和复现步骤，以及设计变化、负控制、正确性/性能/稳定性/显存取舍、已知限制、失败实验与回滚方法。

---

## 1. Baseline

- **任务合同**：`benchmarks/qwen38_4090/evaluation/contract-v1.json` 是 workload、门槛与分数的唯一机器合同；服务接口遵循 `README.md` §3。
- **当前实现 baseline**：提交 `28fd35e`（已包含队友后续的 qwen35 性能与数值对齐修复；更早的官方公开运行产物 `runs/20260822-012115-apxinf/submission.json` 记录的 revision 为 `07313d67`）。
- **性能参照**：契约规定每个性能 cell 的满分参照为“同轮平台 vLLM 对照的最佳有效中位数”，该值只在平台侧存在，本地不可获取；因此 TTFT/TPOT 只能给出实测值，不能本地计算绝对分数。
- **接口协议对照**：`protocol_checks()`（9 项）作为协议级负控制 + 回归测试，全部通过。
- 更早的 rule-baseline 仅用于接口交叉验证，不参与最终成绩。

## 2. 假设

- 硬件：单张 NVIDIA RTX 4090（compute capability 8.9，sm_89 族，128 SMs，24 GB）。
- 模型：revision `63768c10df38c0395e12ef49edac1bd539eaeeea`；权重 W4A16、group size 32、asymmetric。
- 输入为预分词 `input_ids`；解码 greedy（`temperature=0`，thinking 关闭）。
- 基础场景一次一个请求（`parallel_requests=1`），性能用例输出 128 token、`ignore_eos=true`（公开功能用例输出 64 token，按契约用例参数）。
- public 功能用例 6 个本地可测；hidden 用例与 hidden trajectory 不可本地获取，只能由平台判定。

## 3. 设计变化及影响的执行阶段

| 变更 | 位置 | 影响阶段 |
|---|---|---|
| HTTP 服务与协议实现（health/generate/SSE/400/501） | `src/serve.rs`、`src/main.rs` | 服务/协议阶段 |
| health 与错误路由解耦（accept 线程 + 模型 worker 线程） | `src/serve.rs` | 服务稳定性阶段 |
| CUDA 静态库链接补充（`build.rs` 追加 `-L …/apxinf-cuda-*/out` 与 `-lapxinf_kernels`） | `build.rs`（仓库根） | 构建/链接阶段 |
| 全 GPU 前向：W4A16 融合 dequant-GEMM + FA2 + linear-attention 递推 | `crates/apxinf-model/src/qwen35/cuda.rs`、`crates/apxinf-cuda/**` | 生成/性能阶段 |

> 未修改 `benchmarks/qwen38_4090/evaluation/` 中的合同、数据生成器或评测程序。

## 4. 实现

### 4.1 服务与协议层（`src/serve.rs`）

- **路由**：`GET /health`、`GET /` 返回契约 health JSON；`POST /v1/evaluations/generate` 支持 `stream=true/false`；`POST /v1/chat/completions` 明确拒绝（图片未实现，返回 501 `unsupported_capability`）。
- **校验**：`input_ids` 非空且逐项在词表内；`max_new_tokens` 为正且 `prompt + max_new_tokens ≤ max_model_len`；`temperature` 仅接受 0.0；`images` 字段被拒（`unsupported_capability`）。
- **SSE**：token 事件 `index` 从 0 连续递增；结束发 `done`（含 `usage`）与 `[DONE]`；`request_id` 来自服务端单调计数（当前单并发，天然不会混淆）。
- **错误面**：非法请求回 HTTP 400 + `error.type`；未支持能力回 501；内部不可用回 503 `capacity_unavailable`。
- **health 非阻塞**：accept 循环在本线程直接应答 `/health`、`/` 与错误路由；生成请求放入 `mpsc` 队列，由**单个模型 worker 线程**顺序执行（模型 `Engine` 含非 `Send` trait object，不能跨线程移动，故用 worker 线程持有模型而非线程池）。效果：长生成期间 `GET /health` 仍毫秒级返回。

### 4.2 构建/链接修复（`build.rs`）

- 服务器首次构建失败于 `undefined reference to apxinf_static_matmul(...)`（CUDA kernels 静态库未被链接）。根因：`apxinf-cuda` 的 build script 把 `libapxinf_kernels.a` 输出在 `target/release/build/apxinf-cuda-<hash>/out/`，但未把该目录和库名注入最终链接。
- 修复：在仓库根 `build.rs` 中定位 `apxinf-cuda-*/out` 并追加 `cargo:rustc-link-search` 与 `cargo:rustc-link-lib=static=apxinf_kernels`；构建命令设置 `APXINF_CUDA_ARCH=sm_89`。

### 4.3 前向层（`crates/apxinf-model/src/qwen35/cuda.rs`）

- 激活、KV cache、linear-attention 递推状态全部常驻 GPU，层间不再 CPU↔GPU 往返。
- GEMM：W4A16 权重。解码单行用免分配的融合 dequant-GEMM kernel；预填充按 `CHUNK=2048` 分块，反量化到持久 scratch 后走 cuBLAS，摊销 dequant 开销。
- 注意力：full-attention 层把 `q_norm`/`k_norm` 与 partial-RoPE 直接写入 per-layer KV cache，再用 cutlass FA2 flash attention（预填充 split-warp，解码单行）；linear-attention 层的 causal conv + SiLU + gated 状态也在 GPU。
- 容量：`MAX_SEQ_LEN=16640`（= 16384 prompt + 128 output + margin），省约 1 GB 显存；超出按容量错误拒绝。

## 5. 测量

### 5.1 `test.py check`

命令（在仓库根）：

```bash
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
python3 benchmarks/qwen38_4090/evaluation/test.py check
```

结果：

- **当前（提交 `28fd35e`）**：`cargo check --workspace --locked` 通过，`assignment checks passed`（全绿）。
- 历史记录（已在队友提交中消除）：开发中途曾因未提交的 `crates/apxinf-cuda/kernels/custom/qwen35.cuh` 中 `extern __shared__ sh[]` 的 `float`/`uint8_t` 重复声明冲突导致 cargo 步骤失败。

### 5.2 协议负控制（`protocol_checks()`，9/9 通过）

`PROTOCOL_PASS: True`；逐项：

| 用例 | 期望 | 实际 |
|---|---|---|
| malformed JSON | 400 invalid_request | ✅ |
| empty input_ids | 400 invalid_request | ✅ |
| negative token id | 400 invalid_request | ✅ |
| out-of-vocabulary token id | 400 invalid_request | ✅ |
| unsupported temperature | 400 invalid_request | ✅ |
| over budget | 400 invalid_request | ✅ |
| unsupported modality field | 400 unsupported_capability | ✅ |
| valid short no-stream | 200 + result/usage | ✅ |
| health after invalid requests | 服务仍健康 | ✅ |

### 5.3 流式与图片接口实测

- `stream=true`（3 prompt tokens + 5 output）：`index 0..4` 连续 5 个 token 事件 → `done`（`usage:{prompt_tokens:3,completion_tokens:5,total_tokens:8}`）→ `[DONE]`，`request_id` 一致。✅
- `POST /v1/chat/completions`：HTTP 501，`{"type":"unsupported_capability","message":"image input is not supported"}`。✅

### 5.4 `test.py run`（官方公开运行）

命令形态：

```bash
python3 benchmarks/qwen38_4090/evaluation/test.py run \
  --model-dir <model_dir> --base-url http://127.0.0.1:8002 \
  --data-dir benchmarks/qwen38_4090/evaluation/.cache/public \
  --output-dir <output_dir>
```

官方公开运行产物：`runs/20260822-012115-apxinf/submission.json`（实测 raw rows 见 `evidence.raw_jsonl`）。

## 6. 结果（按评分标准）

### 6.1 门槛判定

| 门槛 | 要求 | 实际 | 判定 |
|---|---|---|---|
| 公开功能正确率（6 用例） | 100% | `6/6` | ✅ |
| 未公开正确率（12 用例） | ≥11/12 | 本地不可测（null） | ❓平台判定 |
| 请求成功率 | ≥99% | `1.0` | ✅ |
| 协议 | 通过 | `protocol_pass=true` | ✅ |
| 可靠性 | 无 fallback/OOM/NaN/Xid、失败后恢复 | 全 true | ✅ |

### 6.2 Correctness（30 分，public_calibration 可见口径）

| 子项 | 满分 | 实际 | 得 |
|---|---|---:|---:|
| protocol | 5 | 通过 | 5.00 |
| 公开用例 6/6 | 20 | 6/6 | 20.00 |
| 公开 token trajectory 186/256 | 5 | 72.66% | 3.63 |
| **小计** | 30 | — | **28.63** |

> 注意：公开 trajectory 不参与资格门槛（`public_token_trajectory_rate_min=0.0`），只扣分。hidden 15 分与 hidden trajectory 3 分（midterm 口径）本地不可测。

### 6.3 TTFT（35 分）与 TPOT（25 分）

契约公式：`cell 得分 = weight × min(1, 同轮 vLLM 对照最佳中位数 / 我方中位数)`。对照值平台侧未知，**本地只能给实测中位数**（公开运行 profile 为 1 次测量，非正式 5 次中位数）：

TTFT（权重合计 35）：

| cell | 权重 | 实测 TTFT |
|---|--:|---:|
| 1K | 5 | 2.64 s（另一次运行曾测到 1.02 s，单次测量噪声大） |
| 2K | 5 | 2.30 s |
| 4K | 7 | 5.72 s |
| 8K | 8 | 16.02 s |
| 16K | 10 | 50.73 s |

TPOT（权重合计 25）：

| cell | 权重 | 实测 TPOT |
|---|--:|---:|
| 1K | 10 | 0.206 s |
| 8K | 15 | 0.222 s |

TPOT 稳定（≈0.21–0.22 s/token，CV≈0）；TTFT 在 16K 明显上翘。

### 6.4 Reliability（10 分）

成功率 `1.0` → 5/5；`no_unexpected_oom/no_nan/no_fallback/no_xid/service_healthy_after_failure` 全 true → 5/5。**合计 10/10** ✅

### 6.5 加分（0 / 30）

- 长上下文：`max_verified_prompt_tokens=0`（未跑 32K+）→ 0/10。
- 多请求 C4/C8：`multi_request.cells` 空（`parallel_requests=1`）→ 0/10。
- 图片：未实现（`capabilities.multimodal=false`）→ 0/10（不影响纯文本提交合格性）。

### 6.6 汇总

```
本地可确定：Correctness 28.63 + Reliability 10.00 = 38.63
性能上限：  TTFT ≤35 + TPOT ≤25（按 min(1, 对照/实测) 折算，对照未公开）
基础分上限： 38.63 + 35 + 25 = 98.63
加分上限：   0
课程自动分： min(80, 0.8 × eligible_leaderboard_score)，另加 PR review 20
```

## 7. 负控制与回归测试

1. `protocol_checks()` 9 项（§5.2）：7 个非法/边界请求的 400 + 1 个合法请求 200 + 失败后 health 恢复。
2. 图片接口 501 `unsupported_capability`（§5.3）。
3. 容量失败后可恢复：protocol 套件中先发非法/失败请求后 `GET /health` 仍 200；`service_healthy_after_failure=true`。
4. 服务层回归：`test.py check` 中 `cargo check --workspace --locked` 覆盖服务层编译（当前因未提交 CUDA 改动暂失败，见 §5.1）。

## 8. 正确性 / 性能 / 稳定性 / 显存之间的取舍

- **W4A16 INT4**：把 27B 权重压到 ~14 GB，使单卡可跑 + 留出 KV/激活空间；代价是公开 trajectory 只有 186/256（72.66%），可能来自量化/前向数值差异。
- **KV/激活常驻 GPU + CHUNK=2048**：换取端到端吞吐（短请求 0.136 s），代价是预填充 workspace 常驻；峰值显存 22.86 GB，贴近 24 GB 上限。
- **`MAX_SEQ_LEN=16640`**：为省 ~1 GB 显存把 KV 预算压到 16384+128+margin；副作用是超长请求被拒为容量错误。服务层已用 `min(cli, qwen35::cuda::MAX_SEQ_LEN)` 将 `/health.max_model_len` 对齐为真实值 16640。
- **`parallel_requests=1`**：忠实于契约基本场景、避免并发正确性与锁复杂度；代价是拿不到多请求加分。
- **health 解耦的代价**：模型仍在单 worker 线程串行生成，`/health` 秒回不代表模型空闲；只改善可观测性与服务存活，不提高并发吞吐。

## 9. 已知限制、失败实验与回滚

### 9.1 已知限制

- hidden 正确性不可本地验证（名额门槛 11/12 未知）。
- 公开 token trajectory 72.66%，丢失约 1.4–1.5 的正确性分。
- 16K TTFT 50.7 s 偏慢；TPOT 约 0.21 s/token 仍需与官方 vLLM 对照比较。
- 长上下文（32K+）、多请求（C4/C8）、图片能力均为 0 分。
- `/health` 的 `max_model_len` 已与后端 `MAX_SEQ_LEN=16640` 对齐（真实声明）。要支持 32K+ 长上下文，需先把 KV 分配动态化并提升后端上限（见长上下文路线）。
- 显存峰值 22.86 GB：再叠加长上下文或多请求会 OOM，目前无余量。
- `serve8002.log` 为未跟踪文件，提交前需移除或加入 `.gitignore`。

### 9.2 失败实验（保留的负结果）

1. **CPU 参考前向**：`general.rs` 每层 CPU↔GPU 往返 + CPU 注意力，≈8 s/step，1K+ 前向不可完成 → 弃用，改为 `cuda.rs`。
2. **首次服务器构建链接失败**：`undefined reference to apxinf_static_matmul` → 根 `build.rs` 链接修复。
3. **并行/错峰编译抢锁**：`cuda` 与 `cuda-no-nvtx` 两种 feature 集互相作废 `apxinf-cuda` 产物，反复触发全量 nvcc 重编，链接进程一度卡死 → 结论：全队统一 `--features cuda`，共享同一构建产物。
4. **热替换 OOM**：旧 serve 进程未释放显存时立即起新 serve → 新进程 CUDA OOM；清空显存后重启恢复 → 结论：替换服务前须确认显存归零。
5. **（已解决）**开发中途 `qwen35.cuh` 的 `sh[]` 声明冲突曾让 `cargo check --locked` 失败；随后队友的修复（含 attention 数值对齐）已提交，check 恢复通过。

### 9.3 回滚方法

- 服务层独立回滚：`git checkout src/serve.rs src/main.rs build.rs` 后按 §10 重编。
- 前向回退：`crates/apxinf-model/src/qwen35/cuda.rs` 可回退到 `general.rs` CPU 路径（协议正确、性能慢），只需让 `from_weights_with_backend` 不再构造 CUDA 后端。
- 二进制回退：保存/重编任一历史 commit 的 `target/release/apxinf`，杀掉当前 `apxinf serve` 进程后重启 8002 即可。
- 数据/评测目录 `benchmarks/.../evaluation/` 未改动，始终可直接重放 `test.py check/run`。

## 10. 复现步骤（clean checkout）

```bash
# 1. 环境
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
export CUDA_PATH=/usr/local/cuda
export APXINF_CUDA_ARCH=sm_89

# 2. 构建（先检查，再 release）
cargo check --workspace --locked
cargo build --release --features cuda -p apxinf

# 3. 启动服务（单卡 0）
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  ./target/release/apxinf serve --model <model_dir> --host 0.0.0.0 --port 8002

# 4. 协议与公开测试
python3 benchmarks/qwen38_4090/evaluation/test.py check
# 当前 test.py run 未转发 run_evaluation.py 强制要求的 trajectory reference。
# 在修复冻结评测包前，使用等价的底层命令并传入平台提供的公开 reference：
python3 benchmarks/qwen38_4090/evaluation/run_evaluation.py \
  --dataset benchmarks/qwen38_4090/evaluation/.cache/public \
  --model-dir <model_dir> --base-url http://127.0.0.1:8002 \
  --implementation-name apxinf --implementation-revision <commit_sha> \
  --backend apxinf --profile public_calibration \
  --trajectory-reference <trajectory_reference> --output-dir <output_dir>

# 5. 产物
# <output_dir>/submission.json（correctness/cells/reliability/context/multi_request）
# <output_dir>/environment.json、<output_dir>/raw.jsonl
```

> 注：正式测量的 `implementation-revision` 必须使用完整 commit SHA；本轮未提交 worktree 的回归产物明确标记为 `worktree-paired`，不能作为正式提交证据。

## 11. 2026-08-24 本轮更新（覆盖前文旧性能数据）

### 11.1 问题与决策

- 解码剖析显示每 token 有约 400 次 W4A16 GEMM；单层约 0.97 ms，GEMM 几乎占满层耗时。互不依赖的投影仍逐个启动，kernel tail/launch 开销显著。
- 新增成对 tensor-core W4A16 kernel：同一 grid 处理两组权重和两个输出。解码时合并 linear-attention 的 `qkv/z`、full-attention 的 `k/v`，以及全部 64 层 MLP 的 `gate/up`；prefill、dense 权重和不兼容形状继续走原路径。计算顺序和每个输出的 f32 累加顺序不变。
- 删除每个 full-attention 层重复的同步 `upload_pos`，并删除 decode 快路径前不会使用的临时 Tensor 包装。该同步删除是负控制：单独测量只把等待从 layer 段移到最终同步，端到端收益很小，因此主要收益归因于 paired GEMM。
- 服务现在只在 CUDA 模型加载成功后绑定端口；模型加载失败直接退出。`serve --device cpu` 和未知 device 退出，不再与 `fallback_active=false` 冲突。显式非布尔 `stream`/`ignore_eos` 返回 400。生成中途失败不再伪造正常 `done`/`[DONE]`。

### 11.2 本轮公开回归结果

比较 `runs/iterate2` 与 `runs/paired-regression`；两者均为 `public_calibration`（0 warm-up、1 repeat），只用于本地回归，不是正式 1+5 中位数。

| Cell | iterate2 TTFT | paired TTFT | iterate2 TPOT | paired TPOT | TPOT 改善 |
|---|---:|---:|---:|---:|---:|
| 1K | 0.765 s | 0.768 s | 67.19 ms | **55.28 ms** | **17.7%** |
| 2K | 1.554 s | 1.557 s | 69.50 ms | **57.64 ms** | **17.1%** |
| 4K | 3.189 s | 3.201 s | 74.15 ms | **62.42 ms** | **15.8%** |
| 8K | 6.609 s | 6.647 s | 83.42 ms | **71.86 ms** | **13.9%** |
| 16K | 14.122 s | 14.178 s | 102.08 ms | **90.65 ms** | **11.2%** |

峰值显存保持 23166 MiB；全部性能 cell 输出完整 128 token。1K decode 从 14.9 提升到约 18.1 token/s，prefill 基本不变。

### 11.3 正确性、协议与负控制

- `python3 benchmarks/qwen38_4090/evaluation/test.py check`：`assignment checks passed`。
- 公开功能用例：6/6；协议与 reliability 全通过；请求成功率 1.0。
- paired 回归中 1K/8K 的 128-token 输出哈希逐字节等于 `iterate2`。回归运行使用 `iterate2` 输出作为**本地不回退基线**，所以产物中的 `256/256` 不是官方 trajectory 分数；同样输出对冻结官方 reference 的既有成绩仍为 186/256。
- 非布尔 `stream` 返回 HTTP 400 `invalid_request`；图片 probe 返回 501 `unsupported_capability`；容量失败后 `/health` 仍返回 200；CPU service 启动退出码为 2；无效模型目录启动退出码为 1 且不会开放健康端口。
- 定向单元测试 `serve::tests::generate_rejects_non_boolean_flags` 通过。

### 11.4 已知限制与回滚

- 冻结的 `test.py run` 没有把必需的 `--trajectory-reference` 传给 `run_evaluation.py`，按当前源码会在工作负载执行后报错。本轮未修改冻结评测目录；回归直接调用 `run_evaluation.py` 并显式传入仅用于等价输出检查的本地 reference。正式复现必须使用平台提供的冻结 reference。
- hidden correctness、正式 1 warm-up + 5 repeats/CV、32K+、C4/C8 和图像能力仍未本地验证。
- 回滚 paired GEMM：还原 `qwen35.cuh`、`custom_kernels.cu`、`ffi/custom.rs`、`kernels/quantization.rs` 与 `qwen35/cuda.rs` 中本节对应改动，然后按 §10 重新构建。服务协议改动可独立还原 `src/serve.rs` 与 `src/main.rs`。

---

*本报告不含模型权重、凭据、机器地址或未公开评测数据。*