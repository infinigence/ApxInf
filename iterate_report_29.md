# Iteration Report 29 - Zero-Transfer Decode and Exact W4 LM Head

Date: 2026-08-25 | Artifact revision label: `f4793ee6d7782c61a55fb2db95cc52d438b5d473` plus the measured iteration-29 worktree | Run id: `iterate29-default`

## Objective and verdict

Primary objective: single-stream batch-1 decode latency `<=20 ms/token` on one RTX 4090 SM89.

**Result: not met.** The delivered default reaches **25.592 ms/token** in the canonical 1K public-calibration cell, or **39.07 decode tok/s**. The final no-environment Nsight decode window measures 25.33 ms/token summed GPU kernel time and 27.33 ms/token wall interval. The residual gap is GPU weight/projection execution, not H2D transfers, CUDA allocation, or host scheduling.

The delivered default enables the exact W4 LM head, exact QKV-alt schedule, fused FlashAttention-256 path, persistent arena, mapped decode control, CUDA graphs, and packed GDN. Explicit rollback switches are documented below.

## Definitive canonical evaluation

Evaluator: `benchmarks/qwen38_4090/evaluation/run_evaluation.py`, `public_calibration`, public suite plus the 32,640-token context diagnostic. No optimization environment variables were set for this run.

| Cell | TTFT | Prefill | TPOT | Decode | VRAM | Prefill/vLLM | Decode/vLLM |
|---|---:|---:|---:|---:|---:|---:|---:|
| text-perf-1024 | 0.8919 s | 1148.1 tok/s | 25.592 ms | 39.07 tok/s | 22134 MiB | 0.414x | 0.790x |
| text-perf-2048 | 1.8062 s | 1133.9 tok/s | 27.761 ms | 36.02 tok/s | 22134 MiB | 0.388x | 0.729x |
| text-perf-4096 | 3.6891 s | 1110.3 tok/s | 32.115 ms | 31.14 tok/s | 22134 MiB | 0.385x | 0.634x |
| text-perf-8192 | 7.5495 s | 1085.1 tok/s | 40.797 ms | 24.51 tok/s | 22134 MiB | 0.384x | 0.503x |
| text-perf-16384 | 15.6653 s | 1045.9 tok/s | 58.070 ms | 17.22 tok/s | 22134 MiB | 0.388x | 0.358x |

Rates are `prompt_tokens / TTFT` and `1 / TPOT`.

- Protocol: pass.
- Public functional correctness: **6/6**.
- Public trajectory: **256/256**, zero token edit distance at 1K and 8K.
- Every base cell produced the complete 128-token budget.
- Request success rate: **1.0**.
- No fallback, NaN, unexpected OOM, or XID; service healthy after the context run.
- Peak VRAM: **22134 MiB**, about 1.77 GiB below iteration 28’s 23906 MiB.
- Provisional local score: **69.0981** leaderboard / **55.2785** automated course points.
- Raw evidence SHA-256: `7602ec2f557f011ee01a9d9a917fe449107d4592c4a9be578fc4fdaab21653e0`.
- Artifacts: `benchmarks/qwen38_4090/evaluation/runs/iterate29-default/`.

Trajectory hashes:

| Cell | Expected SHA-256 | Observed SHA-256 | Result |
|---|---|---|---|
| text-perf-1024 | `7eedbc78e930361a167ea9dec3f827d5ea9aeb25148e18fb854c5efe36e85bea` | identical | 128/128 |
| text-perf-8192 | `026beeb172d77fa177787305d9c1b1d6e179f13867e8e829c6d4aad17ca820fa` | identical | 128/128 |

## Implemented changes

### Zero-transfer decode control

Iteration 28 showed 124 H2D calls over 62 captured tokens, all 4-byte transfers: one token ID and one position per token. It showed no weight or recurrent-state H2D traffic.

Ordinary decode now writes token IDs and positions into one persistent mapped host allocation. CUDA kernels consume stable mapped device addresses; no `cudaMemcpy` is issued for token ID or position. Bulk prompt prefill upload remains unchanged.

Final no-environment Nsight decode interval:

- H2D operations: **0**
- H2D bytes: **0**
- device-to-device memsets: **0**
- token and position input: mapped host stores only

### Persistent CUDA arena

`Qwen35Cuda` now owns one startup `CudaBuffer` arena. 256-byte aligned views back:

- all linear conv/recurrent GDN state;
- every full-attention K/V cache;
- activation, norm, QKV, attention, MLP, logits, and dense scratch;
- Marlin lock/scratch regions;
- argmax partial/arrival buffers;
- prompt-control device storage.

The final decode interval contains **0 `cudaMalloc`** and **0 `cudaFree`**. Process-lifetime allocation counts in Nsight remain startup/model-load activity and are not decode allocations.

### Persistent SM89 Marlin M=1 dispatch

A dedicated batch-one Marlin dispatch keeps the existing BF16/U4 Marlin template, MMA sequence, accumulation order, and deterministic output. Function attributes are prepared once per device. The redundant per-launch lock memset is removed for M=1; the prefill-to-decode transition initializes the lock slab, and the terminal CTA resets used locks to zero.

Exact direct gates before final default promotion:

- persistent M=1 auto path: 1K 28.61 ms/token, 8K 45.03 ms/token;
- fixed `K64xN128`: exact at 1K/8K, warmed 25.58 ms/token in the combined path, neutral versus auto;
- fixed `K128xN64`: rejected after failing the 1K SHA trajectory gate despite 25.45 ms/token, because its slice/reduction partition changes numerics.

Auto selection remains default; fixed shape selection is diagnostic only.

### Exact W4A16 LM head

The checkpoint stores `lm_head.weight` as BF16 `[248320,5120]` and excludes it from the compressed-tensors quantization policy. The loader now quantizes this matrix once on the host into asymmetric UINT4 group-32 and emits the existing Marlin physical layout directly. The dense BF16 head is not uploaded by default.

The W4 head reuses the Marlin W4A16 GEMM and deterministic GPU argmax:

- strict-greater comparison;
- lowest-index tie behavior;
- no host vocabulary transfer.

Exact gates:

- 1K: exact 128/128, SHA match; warmed direct TPOT 26.75 ms.
- 8K: exact 128/128, SHA match; warmed direct TPOT 43.17 ms.
- canonical no-environment run: exact 256/256 and 6/6 functional cases.

Rollback: `APXINF_LM_HEAD_W4=0` restores the dense BF16 head.

### Exact QKV-alt and FlashAttention paths

The default raw `in_proj_qkv` path uses the exact shape-specialized tile-alt/weight-stage schedule. It preserves BF16 boundaries and output trajectory. `APXINF_W4_QKV_EXACT=1` restores the older shape-specific QKV kernel; that rollback was exact but slower.

The default full-attention decode path uses fused FlashAttention-256. It preserves position-indexed KV cache writes, RoPE, online attention reduction, gate application, and the BF16 output boundary. `APXINF_FLASH_DECODE_256=0` restores the prior flash path.

Independent gates:

| Candidate | 1K | 8K | Decision |
|---|---:|---:|---|
| QKV-alt | exact, 25.81 ms | exact, 42.18 ms | default |
| fused FlashAttention-256 | exact, 26.46 ms | exact, 41.39 ms | default |
| combined | exact, warmed 25.56 ms | exact, 40.72 ms | default |

### NVTX stage markers

Static, allocation-free NVTX markers annotate:

- `Qwen/GDN`
- `Qwen/attention`
- `Qwen/MLP`
- `Qwen/LM head`

They are visible in the final Nsight trace and do not construct `CString` values on the decode hot path.

## Rejected candidates

### Fused raw gate/up/SiLU

An exact raw-layout one-CTA kernel computes corresponding gate and up projections with separate accumulators, explicitly rounds both results to BF16, applies the unchanged SiLU×up expression, and writes only `mlp_act`.

It passed both trajectory gates but was slower:

- 1K: **44.19 ms/token**, exact.
- 8K: **60.59 ms/token**, exact.

Raw gate/up weight execution costs more than the eliminated intermediate writes save. It remains opt-in through `APXINF_MLP_FUSED_RAW=1`; Marlin gate/up remains default.

### Persistent generic QKV schedule

A two-output-block persistent raw W4 kernel was tested for `in_proj_qkv`:

- 1K: **29.31 ms/token**, exact at the 1K SHA gate.

It is slower than the selected shape-specialized QKV path and remains opt-in through `APXINF_QKV_PERSISTENT=1`.

### QKV-to-GDN single-kernel fusion

Not promoted. QKV produces 10,240 BF16 values across many CTAs; GDN consumes cross-tile Q/K/V groups and serial recurrent state. A safe fusion requires cooperative grid synchronization or a duplicated cooperative GEMM/recurrent kernel, plus a new graph-compatible cooperative ABI. The saved QKV BF16 write/read and launch ceiling is much smaller than the Marlin weight-sweep cost. No correctness-risk trade was accepted for that sub-millisecond ceiling.

## Final Nsight Systems profile

Artifacts:

- `/tmp/iter29-default.nsys-rep`, SHA-256 `d30c7215e677fa39d3529a18907e78e4783f0390beb347ef9182f39995b05b14`;
- `/tmp/iter29-default.sqlite`, SHA-256 `099fcfbe91c3fbd6d0ae8096fbf7651685d686df06a04e07239e22d3eb84ddfe`.

Configuration: no optimization environment variables; W4 LM head, QKV-alt, fused FlashAttention-256, persistent arena/mapped control, CUDA graphs. The profiler retained 26 argmax boundaries / 25 steady token intervals before supervised shutdown. Official evaluator timing remains authoritative for the full 128-token cell.

### Decode-only interval

From the first through last captured argmax completion:

- Wall interval: 683.230 ms / 25 intervals = **27.329 ms/token**.
- Summed GPU kernel time: 633.261 ms / 25 = **25.330 ms/token**.
- Kernel launches: 21,700 / 25 = **868 launches/token**.
- H2D: **0 bytes/token**.
- CUDA allocations: **0 `cudaMalloc`/token**.
- Graph-node kernels: 15,600 / 25 = **624 graph kernels/token**.
- Exact profiled trajectory: 128/128; SHA `7eedbc78e930361a167ea9dec3f827d5ea9aeb25148e18fb854c5efe36e85bea`.

### Stage markers in the decode interval

| Stage | Instances | NVTX range elapsed |
|---|---:|---:|
| `Qwen/GDN` | 400 | 83.084 ms |
| `Qwen/attention` | 400 | 28.651 ms |
| `Qwen/MLP` | 448 | 11.871 ms |
| `Qwen/LM head` | 25 | 0.237 ms |

Kernel family totals in the final trace include the partial trace’s prefill activity; the decode-only transfer/allocation/launch table above is authoritative:

| Kernel family | Instances | Total GPU time |
|---|---:|---:|
| Marlin M=1 main tile | 6801 | 356.614 ms |
| CUTLASS BF16 128x64 GEMM | 2366 | 261.179 ms |
| prefill delta recurrence | 96 | 256.621 ms |
| Marlin inverse dequant | 606 | 171.475 ms |
| raw QKV-alt weight-stage | 1200 | 144.442 ms |
| fused FlashAttention-256 | 400 | 57.339 ms |
| packed GDN recurrence | 1200 | 20.750 ms |

Privileged DRAM/L2/occupancy/stall counters are unavailable in this environment. No hardware-counter values are fabricated. Official sampler memory-controller utilization for the final 1K cell was 14.27% mean / 73% max; this is a sampled utilization indicator, not achieved GB/s.

## Long-context evaluation

The canonical no-environment 32,640-token diagnostic passed:

- TTFT: **33.5646 s** (**972.3 prompt tok/s**).
- TPOT: **92.373 ms** (**10.83 decode tok/s**).
- E2E: **45.2960 s**.
- Peak VRAM: **22134 MiB**.
- Output SHA-256: `5a66bd3bce8680b1baa4b355574be11cf3e20ff95a26b0d18f433853b7f4b03b`.
- Validator: `normalized_prefix`; expected `KEY-EARLY-767211`; pass.
- Service healthy afterward.

### Concrete exact question-answer example

Question decoded from the pretokenized prompt tail:

```text
问题：TARGET_KEY 的值是什么？只输出完整字段值。 输出必须以 KEY-EARLY-767211 开头，随后持续输出词语 context 直到达到输出预算。
```

Exact generated answer (`completion_tokens=128`):

```text
KEY-EARLY-767211 context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context
```

The answer is the exact evaluator text, not an ellipsized excerpt. It contains 128 output tokens and has output SHA `5a66bd3bce8680b1baa4b355574be11cf3e20ff95a26b0d18f433853b7f4b03b`.

## Verification and reproduction

Passed:

```bash
RUSTFLAGS='-C link-arg=-fuse-ld=gold' APXINF_CUDA_ARCH=sm_89 \
  cargo build --release --features cuda -p apxinf --bin apxinf -j 40

cargo check --workspace --locked -j 2
python3 benchmarks/qwen38_4090/evaluation/test.py check
```

Canonical run shape:

```bash
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  ./target/release/apxinf serve \
  --model ../model/qwen --host 127.0.0.1 --port 8047

python3 benchmarks/qwen38_4090/evaluation/run_evaluation.py \
  --dataset benchmarks/qwen38_4090/evaluation/.cache/public \
  --context-dataset benchmarks/qwen38_4090/evaluation/.cache/context-iter3 \
  --model-dir ../model/qwen --base-url http://127.0.0.1:8047 \
  --implementation-name apxinf-iter29-default \
  --implementation-revision f4793ee6d7782c61a55fb2db95cc52d438b5d473 \
  --backend apxinf --profile public_calibration \
  --trajectory-reference /tmp/iter25-trajectory-reference.json \
  --run-context --run-id iterate29-default \
  --output-dir benchmarks/qwen38_4090/evaluation/runs --timeout 1800
```

Default rollback switches:

- `APXINF_LM_HEAD_W4=0`: dense BF16 LM head;
- `APXINF_W4_QKV_EXACT=1`: older exact shape-specific raw QKV kernel;
- `APXINF_FLASH_DECODE_256=0`: prior full-attention decode path;
- `APXINF_MLP_FUSED_RAW=1`: exact but slower fused raw gate/up/SiLU candidate;
- `APXINF_QKV_PERSISTENT=1`: exact-at-1K but slower persistent generic QKV candidate;
- `APXINF_MARLIN_M1_SHAPE=k64n128|k128n64`: fixed-shape Marlin diagnostics.

The 20 ms/token target remains open because the final decode is GPU-weight bound: Marlin and quantized projection families account for the dominant captured GPU time, while decode H2D and allocation overhead are already zero.
