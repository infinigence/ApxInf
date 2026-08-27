# Iteration Report 35 - Two-Warp Exact GDN Prefill Recurrence

Date: 2026-08-26  
Base repository revision: `f4793ee6d7782c61a55fb2db95cc52d438b5d473` plus the measured iteration-35 worktree  
Canonical run id: `iterate35-default`

## Verdict

Iteration 35 promotes the repaired two-warp GDN prefill recurrence. The official evaluator shows a **1.69-2.36% prefill-throughput improvement** over iteration 34 across every performance cell, with exact correctness and stable decode.

**Correctness passes. The requested 1.2x vLLM throughput target remains unmet.**

- Public functional correctness: **6/6**.
- Public frozen trajectory: **256/256 tokens**.
- Protocol and reliability gates: pass.
- Long context: **32,640 prompt tokens + 128 output tokens**, validator pass.
- Peak VRAM: **22,138 MiB**.
- Best canonical ApxInf/vLLM ratio: **0.590x prefill** and **0.835x decode** at 1K; required ratio: **1.2x**.

## Problem and design

Iteration 34's matched Nsight capture attributed 30.7% of kernel time to the exact prefill delta recurrence. The production kernel launched four independent one-warp CTAs per value head:

- grid: `(4, 48, 1)` = 192 CTAs;
- block: `(32, 1, 1)`;
- one 32-column recurrent-state tile per CTA.

Q/K rows, decay, and beta were redundantly loaded or computed across four CTAs. The iteration-33 two-warp candidate instead launches two CTAs per value head:

- grid: `(2, 48, 1)` = 96 CTAs;
- block: `(64, 1, 1)`;
- one disjoint 64-column recurrent-state tile per CTA;
- Q/K and scalar work shared across the two warps in each CTA;
- token order and each value column's K-ordered FP32 recurrence unchanged.

The candidate initially failed because only half of the Q/K row was loaded and a repair temporarily removed the `for (s...)` token loop. Iteration 33 restored both the strided Q/K load and complete token loop. Iteration 35 is the first canonical remeasurement of that repaired kernel.

## Controlled geometry matrix

Three identical iteration-34 binaries ran concurrently on three RTX 4090 GPUs. Only `APXINF_PREFILL_GDN_WARPS` differed. Each TTFT round sent the same frozen prompt concurrently and generated one token, isolating prefill.

### Median TTFT

| Case | One-warp control | Two-warp repaired | Two-warp gain | Four-warp | Four-warp vs control |
|---|---:|---:|---:|---:|---:|
| 1K | 0.6366 s | **0.6196 s** | **2.66%** | 0.6385 s | 0.30% slower |
| 8K | 5.4991 s | **5.3712 s** | **2.33%** | 5.5185 s | 0.35% slower |
| 16K | 11.5990 s | **11.3428 s** | **2.21%** | 11.6444 s | 0.39% slower |

All three shapes returned the same one-token output SHA in every run. Four-warp is slower because its 48 CTAs leave 80 of 128 SMs without a recurrence CTA. Two-warp gives 96 CTAs and shares work within each 64-column pair.

### Full frozen trajectory gate

Two 128-token rounds per frozen case:

| Case | Control SHA | Two-warp SHA | Result |
|---|---|---|---|
| text-perf-1024 | `7eedbc78...5bea` in 2/2 | `7eedbc78...5bea` in 2/2 | exact |
| text-perf-8192 | `026beeb1...20fa` in 2/2 | `026beeb1...20fa` in 2/2 | exact |

## Source change

File: `crates/apxinf-model/src/qwen35/cuda.rs`.

- Added `PREFILL_GDN_WARPS: LazyLock<usize>` so the environment is parsed once.
- Missing or invalid `APXINF_PREFILL_GDN_WARPS` now selects the repaired two-warp kernel.
- `APXINF_PREFILL_GDN_WARPS=0` rolls back to the proven one-warp control.
- `APXINF_PREFILL_GDN_WARPS=4` selects the exact but slower four-warp diagnostic.
- No arithmetic, tensor layout, BF16 boundary, or decode path changed.

The implementation uses standard-library `LazyLock`; no `once_cell` dependency or repeated `getenv` was added.

## Canonical official evaluation

Evaluator: `benchmarks/qwen38_4090/evaluation/run_evaluation.py`  
Profile: `public_calibration`  
Dataset: `benchmarks/qwen38_4090/evaluation/.cache/public`  
Context dataset: `benchmarks/qwen38_4090/evaluation/.cache/context-iter3`  
Warmups / measured repeats: 0 / 1  
Optimization overrides: none  
Model: `../model/qwen`, revision `63768c10df38c0395e12ef49edac1bd539eaeeea`

| Cell | TTFT | Prefill | TPOT | Decode | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| text-perf-1024 | 0.6254 s | **1,637.5 tok/s** | 24.203 ms | **41.32 tok/s** | 22,138 MiB |
| text-perf-2048 | 1.2728 s | **1,609.1 tok/s** | 25.750 ms | **38.84 tok/s** | 22,138 MiB |
| text-perf-4096 | 2.6283 s | **1,558.4 tok/s** | 28.837 ms | **34.68 tok/s** | 22,138 MiB |
| text-perf-8192 | 5.4278 s | **1,509.3 tok/s** | 35.002 ms | **28.57 tok/s** | 22,138 MiB |
| text-perf-16384 | 11.4096 s | **1,436.0 tok/s** | 47.331 ms | **21.13 tok/s** | 22,138 MiB |

Correctness and reliability:

- protocol pass: true;
- public functional cases: **6/6**;
- public frozen trajectory: **256/256**, zero edit distance at 1K and 8K;
- request success rate: **1.0**;
- no fallback, NaN, unexpected OOM, or XID;
- service healthy after failures and context recovery;
- all performance cells completed exactly 128 output tokens.

## Iteration 34 comparison

| Cell | Prefill throughput ratio | Decode throughput ratio |
|---|---:|---:|
| 1K | **1.0236x** | 1.0030x |
| 2K | **1.0196x** | 1.0027x |
| 4K | **1.0192x** | 1.0025x |
| 8K | **1.0176x** | 1.0022x |
| 16K | **1.0169x** | 1.0012x |

The canonical gain agrees with the controlled matrix. Decode changes only through reduced prefill heat/state before the measured output stream; the decode algorithm itself is unchanged.

## vLLM comparison and target gate

Control: `benchmarks/qwen38_4090/evaluation/runs/vllm-one-gpu-control/submission.json`.

| Cell | ApxInf/vLLM prefill | ApxInf/vLLM decode | Required ratio | Result |
|---|---:|---:|---:|---|
| 1K | 0.590x | 0.835x | 1.200x | fail |
| 2K | 0.551x | 0.786x | 1.200x | fail |
| 4K | 0.540x | 0.706x | 1.200x | fail |
| 8K | 0.533x | 0.586x | 1.200x | fail |
| 16K | 0.533x | 0.439x | 1.200x | fail |

The requested 1.2x threshold remains unmet in every cell.

## Long-context evaluation and exact example

| Metric | Result |
|---|---:|
| Prompt tokens | 32,640 |
| Output tokens | 128 |
| TTFT | 25.1466 s |
| Prefill throughput | 1,298.0 tok/s |
| TPOT | 71.774 ms/token |
| Decode throughput | 13.93 tok/s |
| End to end | 34.2621 s |
| Validator | pass |
| Output SHA-256 | `5a66bd3bce8680b1baa4b355574be11cf3e20ff95a26b0d18f433853b7f4b03b` |

Iteration 34 long-context E2E was 34.6510 s; iteration 35 reduces it by 0.3890 s.

Concrete case: `context-32640-retrieval-early`.

Exact question at the end of the 32,640-token input:

```text
问题：TARGET_KEY 的值是什么？只输出完整字段值。 输出必须以 KEY-EARLY-767211 开头，随后持续输出词语 context 直到达到输出预算。
```

Exact decoded answer:

```text
KEY-EARLY-767211 context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context
```

The answer contains exactly 128 tokens and passes the normalized-prefix validator.

## Nsight Systems profile

Profiler: NVIDIA Nsight Systems 2025.1.1, CUDA graph child-node tracing enabled. The matched request is the frozen 8K prompt plus 128 output tokens.

### Recurrence geometry and time

| Iteration | Kernel | Grid | Block | Instances | Total GPU time | Average |
|---|---|---|---|---:|---:|---:|
| 34 | `qwen35_prefill_delta_step_kernel` | `(4,48,1)` | `(32,1,1)` | 768 | 2,029.5 ms | 2.6426 ms |
| 35 | `qwen35_prefill_delta_step_2w_kernel` | `(2,48,1)` | `(64,1,1)` | 768 | **1,932.0 ms** | **2.5157 ms** |

The recurrence kernel is 4.8% faster in the matched capture. It falls from 30.7% to 29.6% of captured kernel time.

### Iteration-35 kernel breakdown

| Kernel family | Instances | GPU time | Share |
|---|---:|---:|---:|
| Two-warp prefill recurrence | 768 | 1,932.02 ms | 29.6% |
| M512 MLP Marlin | 3,072 | 1,725.79 ms | 26.5% |
| CUTLASS `Kernel2` | 6,144 | 471.85 ms | 7.2% |
| GQA6 flash partial | 527 | 410.60 ms | 6.3% |
| GDN Marlin | 6,291 | 315.12 ms | 4.8% |
| Other Marlin | 4,441 | 292.75 ms | 4.5% |
| Attention SGEMM 128x128 | 3,456 | 278.12 ms | 4.3% |
| Conv/SiLU | 768 | 177.90 ms | 2.7% |
| Attention Marlin | 1,024 | 174.84 ms | 2.7% |
| Pair-prefetch raw W4 | 1,581 | 168.16 ms | 2.6% |

Prefill recurrence and quantized projections remain co-dominant. The next occupancy candidate is three value-state tiles per head, which would expose 144 CTAs and cover all 128 SMs while retaining shared work within each tile; it must first prove exactness because 128 columns do not divide evenly by three.

### Memory-controller utilization

Evaluator hardware samples, 200 ms interval:

| Request | Mean | Max | GPU-util mean | GPU-util max |
|---|---:|---:|---:|---:|
| 1K / 128 | 13.05% | 81% | 25.00% | 100% |
| 2K / 128 | 15.70% | 81% | 25.00% | 100% |
| 4K / 128 | 11.16% | 76% | 25.00% | 100% |
| 8K / 128 | 8.96% | 69% | 25.00% | 100% |
| 16K / 128 | 5.84% | 45% | 24.94% | 100% |
| 32,640 / 128 | 5.53% | 44% | 25.00% | 100% |

These are coarse whole-request samples, not per-kernel sustained bandwidth. Nsight Compute counters remain unavailable because the host driver sets `RmProfilingAdminOnly=1`.

## Artifacts and hashes

Official evaluation:

- `benchmarks/qwen38_4090/evaluation/runs/iterate35-default/submission.json` — `86933d996961d5dd4adcce10fffc2a136a067b637c67d0bf4108b2f887d8350f`
- `benchmarks/qwen38_4090/evaluation/runs/iterate35-default/raw.jsonl` — `df2aca7dcfb3754928a341e89c21d09c86b78f378bf75817102f78c42d5a6b4b`
- `benchmarks/qwen38_4090/evaluation/runs/iterate35-default/environment.json` — `f0015dad7f30ed55125eb9e69ef0fe82343859e2856ed940d071e5e180a51c62`
- trajectory reference — `904fa6f54dec58d9638a2b067c78b145a48c63a4b838fba26e473d866576b793`

Profiler/build:

- `/tmp/iterate35-decode-graph.nsys-rep` — `f574f44cbbfaa99c089f7a2d5aafcfbee1be22d4d3e25757ba60a95ed913a69d`
- `/tmp/iterate35-decode-graph.sqlite` — `bb85db61390ab13ec4eeaee0debe458056b57d75dbb7f16c461ec2d9ecfaa182`
- `/tmp/iterate35-kernels_cuda_gpu_kern_sum_nvtx-name_base.csv` — `d90886d21c2791c7ecba3c0837f5a214ce55f0af81329c9c17597524d75d6bcf`
- `target/release/apxinf` — `24c5d0267ded02647902fc8119446a29f8344e3b4b5c02b5c0783f366028a059`

`/tmp` artifacts are workstation-local and require durable copying before workstation shutdown.

## Reproduction

```bash
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export APXINF_CUDA_ARCH=sm_89
export RUSTFLAGS='-C link-arg=-fuse-ld=gold'

cargo build --release --features cuda -p apxinf --bin apxinf -j 40
CUDA_VISIBLE_DEVICES=0 ./target/release/apxinf serve \
  --model ../model/qwen --host 127.0.0.1 --port 8323

python3 benchmarks/qwen38_4090/evaluation/run_evaluation.py \
  --dataset benchmarks/qwen38_4090/evaluation/.cache/public \
  --context-dataset benchmarks/qwen38_4090/evaluation/.cache/context-iter3 \
  --model-dir ../model/qwen --base-url http://127.0.0.1:8323 \
  --implementation-name apxinf-iteration35 \
  --implementation-revision iterate35-working --backend apxinf \
  --profile public_calibration \
  --trajectory-reference /tmp/iter25-trajectory-reference.json \
  --run-context --timeout 2400 --run-id iterate35-default \
  --output-dir benchmarks/qwen38_4090/evaluation/runs
```

The release CUDA build, canonical evaluator, and graph-node profile completed. No service or profiler process remained live after capture.