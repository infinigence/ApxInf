# Iteration Report 34 - Marlin M1 Shared-Memory Shape

Date: 2026-08-26  
Base repository revision: `f4793ee6d7782c61a55fb2db95cc52d438b5d473` plus the measured iteration-34 worktree  
Canonical run id: `iterate34-default`

## Verdict

Iteration 34 promotes the exact K64xN128, 128-thread Marlin M1 launch as the RTX 4090 decode default. Repeated concurrent A/B measurements show a small, stable TPOT reduction, but the canonical one-repeat evaluator is statistically neutral relative to iteration 33.

**Correctness passes. The requested 1.2x vLLM throughput target remains unmet.**

- Public functional correctness: **6/6**.
- Public frozen trajectory: **256/256 tokens**.
- Protocol and reliability gates: pass.
- Long context: **32,640 prompt tokens + 128 output tokens**, validator pass.
- Peak VRAM: **22,138 MiB**.
- Best canonical ApxInf/vLLM ratio: **0.577x prefill** and **0.833x decode**, both at 1K; required ratio: **1.2x**.

## Problem and decision

Iteration 33's graph-node profile showed that quantized projection kernels remain the decode bottleneck. Marlin M1 uses a persistent 128-CTA grid on this 128-SM RTX 4090. The prior automatic launch was K128xN128 with 256 threads and 54,272 bytes of dynamic shared memory per CTA.

The adapter already contained two diagnostic 128-thread shapes:

| Shape | Threads | Dynamic shared memory | Frozen result | Repeated TPOT result |
|---|---:|---:|---|---|
| K128xN128 control | 256 | 54,272 B | Exact | Control |
| K64xN128 | 128 | 27,136 B | Exact at 1K and 8K | Faster in every A/B round |
| K128xN64 | 128 | 35,328 B | Diverged at 1K | Slower |

Decision: make K64xN128 the automatic M1 shape. Keep explicit rollback/diagnostic controls:

- `APXINF_MARLIN_M1_SHAPE=auto256`: previous K128xN128, 256-thread launch;
- `APXINF_MARLIN_M1_SHAPE=k64n128`: explicit promoted shape;
- `APXINF_MARLIN_M1_SHAPE=k128n64`: rejected diagnostic shape.

The environment variable is parsed once when the thread-local device launch state is initialized. No `getenv` remains on the GEMM hot path.

## Source change

File: `crates/apxinf-cuda/adapters/marlin_adapter.cu`.

- `DeviceLaunchState::m1_shape` now defaults to K64xN128.
- `get_launch_state()` maps `auto256` to the former 256-thread shape, `k128n64` to the alternate 128-thread shape, and every other value to K64xN128.
- `launch_gemm_m1()` retains the existing exact Marlin arithmetic and changes only the selected compile-time tile/CTA geometry.
- Multi-row Marlin dispatch is unchanged; M512 prefill still uses the 256-thread `m_block_size_8=false` instantiation.

Build integrity evidence:

- edited source timestamp: 2026-08-26 21:57:38 +0800;
- rebuilt `marlin_adapter.o`: 2026-08-26 21:58:16 +0800;
- rebuilt CUDA archive: 2026-08-26 21:58:17 +0800;
- linked `target/release/apxinf`: 2026-08-26 21:58:38 +0800;
- binary contains the new `auto256` override string;
- Nsight shows block-128 M1 Marlin launches and block-256 M512 launches in the same executable.

## Controlled A/B matrix

Three identical iteration-33 binaries were loaded concurrently on three RTX 4090 GPUs. Only `APXINF_MARLIN_M1_SHAPE` differed. Each round sent the same frozen case to all three services concurrently, avoiding serial thermal drift. Four rounds were measured per case.

### 1K case

| Shape | Output SHA behavior | TPOT runs | Median TPOT | Versus control |
|---|---|---|---:|---:|
| K128xN128 control | Exact `7eedbc...5bea` in 4/4 | 24.202, 24.141, 24.144, 24.154 ms | 24.149 ms | control |
| K64xN128 | Exact `7eedbc...5bea` in 4/4 | 24.095, 24.014, 24.035, 24.047 ms | **24.041 ms** | **0.445% lower TPOT** |
| K128xN64 | Wrong `e277e4...86bc` in 4/4 | 24.301, 24.237, 24.237, 24.251 ms | 24.244 ms | 0.394% slower; reject |

### 8K case

| Shape | Output SHA behavior | TPOT runs | Median TPOT | Versus control |
|---|---|---|---:|---:|
| K128xN128 control | Exact `026beeb...820fa` in 4/4 | 34.943, 34.952, 34.951, 34.973 ms | 34.952 ms | control |
| K64xN128 | Exact `026beeb...820fa` in 4/4 | 34.785, 34.776, 34.776, 34.784 ms | **34.780 ms** | **0.491% lower TPOT** |
| K128xN64 | Exact at 8K | 35.168, 35.171, 35.174, 35.258 ms | 35.172 ms | 0.631% slower; reject |

The K128xN64 first-token trajectory divergence makes it ineligible regardless of speed. K64xN128 is the only exact candidate and its A/B direction is stable.

## Canonical official evaluation

Evaluator: `benchmarks/qwen38_4090/evaluation/run_evaluation.py`  
Profile: `public_calibration`  
Dataset: `benchmarks/qwen38_4090/evaluation/.cache/public`  
Context dataset: `benchmarks/qwen38_4090/evaluation/.cache/context-iter3`  
Warmups / measured repeats: 0 / 1  
Service optimization overrides: none  
Model: `../model/qwen`, revision `63768c10df38c0395e12ef49edac1bd539eaeeea`

| Cell | TTFT | Prefill | TPOT | Decode | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| text-perf-1024 | 0.6401 s | **1,599.7 tok/s** | 24.275 ms | **41.19 tok/s** | 22,138 MiB |
| text-perf-2048 | 1.2978 s | **1,578.1 tok/s** | 25.820 ms | **38.73 tok/s** | 22,138 MiB |
| text-perf-4096 | 2.6788 s | **1,529.0 tok/s** | 28.908 ms | **34.59 tok/s** | 22,138 MiB |
| text-perf-8192 | 5.5233 s | **1,483.2 tok/s** | 35.079 ms | **28.51 tok/s** | 22,138 MiB |
| text-perf-16384 | 11.6025 s | **1,412.1 tok/s** | 47.390 ms | **21.10 tok/s** | 22,138 MiB |

Correctness and reliability:

- protocol pass: true;
- public functional cases: **6/6**;
- public frozen trajectory: **256/256**, zero edit distance at 1K and 8K;
- request success rate: **1.0**;
- no fallback, NaN, unexpected OOM, or XID;
- service healthy after failures and context recovery;
- every performance cell completed 128 output tokens.

## Iteration 33 comparison

| Cell | Prefill ratio vs iteration 33 | Decode ratio vs iteration 33 |
|---|---:|---:|
| 1K | 1.0000x | 0.9998x |
| 2K | 1.0033x | 1.0002x |
| 4K | 1.0000x | 1.0006x |
| 8K | 1.0004x | 0.9993x |
| 16K | 0.9996x | 0.9998x |

The official single-repeat result is neutral: all decode deltas are within plus or minus 0.07%. The report therefore does not claim an official throughput improvement. The default is retained because the controlled four-round A/B matrix is exact and consistently favors K64xN128, while also halving dynamic shared memory.

## vLLM comparison and target gate

Control: `benchmarks/qwen38_4090/evaluation/runs/vllm-one-gpu-control/submission.json`.

| Cell | ApxInf/vLLM prefill | ApxInf/vLLM decode | Required ratio | Required prefill | Required decode | Result |
|---|---:|---:|---:|---:|---:|---|
| 1K | 0.577x | 0.833x | 1.200x | 3,329.1 tok/s | 59.35 tok/s | fail |
| 2K | 0.540x | 0.784x | 1.200x | 3,503.9 tok/s | 59.28 tok/s | fail |
| 4K | 0.530x | 0.704x | 1.200x | 3,460.8 tok/s | 58.94 tok/s | fail |
| 8K | 0.524x | 0.585x | 1.200x | 3,395.2 tok/s | 58.50 tok/s | fail |
| 16K | 0.524x | 0.438x | 1.200x | 3,233.5 tok/s | 57.77 tok/s | fail |

Iteration 34 does not meet the requested 1.2x threshold in any performance cell.

## Long-context evaluation and exact example

| Metric | Result |
|---|---:|
| Prompt tokens | 32,640 |
| Output tokens | 128 |
| TTFT | 25.5269 s |
| Prefill throughput | 1,278.6 tok/s |
| TPOT | 71.843 ms/token |
| Decode throughput | 13.92 tok/s |
| End to end | 34.6510 s |
| Validator | pass |
| Output SHA-256 | `5a66bd3bce8680b1baa4b355574be11cf3e20ff95a26b0d18f433853b7f4b03b` |

Concrete case: `context-32640-retrieval-early`.

Exact question at the end of the 32,640-token input:

```text
问题：TARGET_KEY 的值是什么？只输出完整字段值。 输出必须以 KEY-EARLY-767211 开头，随后持续输出词语 context 直到达到输出预算。
```

Exact decoded answer:

```text
KEY-EARLY-767211 context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context
```

The answer contains the exact 128 output tokens and passes the normalized-prefix validator.

## Profiling

### Tools and limitation

- Nsight Systems 2025.1.1 captured the production graph with `--cuda-graph-trace=node:host-only`.
- `nvprof` 11.5 refuses profiling on compute capability 8.9.
- Nsight Compute 2025.1.1 is installed, but the host driver sets `RmProfilingAdminOnly=1`; metric collection fails with `ERR_NVGPUCTRPERM`, even as root inside this environment.
- Consequently, no per-kernel DRAM-throughput, L2-hit-rate, or achieved-occupancy counters are fabricated. Static resource use, launch geometry, Nsight kernel time, and evaluator memory-controller samples are reported.

### Static resource evidence

The compiled M1 Marlin family uses 117-120 registers per thread and dynamic shared memory determined by the selected tile:

- K128xN128: 54,272 bytes, 256 threads;
- K64xN128: 27,136 bytes, 128 threads;
- K128xN64: 35,328 bytes, 128 threads.

K64xN128 halves shared memory relative to the former automatic shape. The 128-CTA persistent grid remains one CTA per RTX 4090 SM.

### Nsight launch verification

The iteration-34 graph trace contains both phases and proves the intended dispatch separation:

- decode-owned `Qwen/GDN/Marlin`: `grid=(128,1,1)`, `block=(128,1,1)`, 6,255 instances, 312.98 ms total, 50.04 us average;
- decode LM-head Marlin: `grid=(128,1,1)`, `block=(128,1,1)`, 33 instances, 25.61 ms total, 776.16 us average;
- attention Marlin includes block-128 decode launches and block-256 prefill launches;
- M512 `Qwen/MLP/Marlin`: `grid=(128,1,1)`, `block=(256,1,1)`, 3,072 instances, 1,729.69 ms total, 563.05 us average.

Iteration 33 used block 256 for GDN and LM-head Marlin. Iteration 34 switches those M1 sites to block 128 while leaving M512 prefill unchanged.

### Capture-wide kernel breakdown

The profile is one frozen 8K prompt plus 128 generated tokens; it contains prefill and decode.

| Kernel family | Instances | GPU time | Share |
|---|---:|---:|---:|
| Prefill delta recurrence | 768 | 2,029.52 ms | 30.7% |
| M512 MLP Marlin | 3,072 | 1,729.69 ms | 26.2% |
| CUTLASS `Kernel2` | 6,144 | 472.93 ms | 7.2% |
| GQA6 flash partial | 524 | 406.86 ms | 6.2% |
| GDN Marlin | 6,255 | 312.98 ms | 4.7% |
| Other Marlin | 4,417 | 291.98 ms | 4.4% |
| Attention SGEMM 128x128 | 3,456 | 278.24 ms | 4.2% |
| Conv/SiLU | 768 | 178.08 ms | 2.7% |
| Attention Marlin | 1,024 | 175.24 ms | 2.7% |
| Pair-prefetch raw W4 | 1,572 | 167.44 ms | 2.5% |

The bottleneck conclusion is unchanged: prefill recurrence and quantized projections dominate. M1 geometry tuning is too small to close the target gap.

### Request-window memory-controller utilization

| Request | Mean | Max | GPU-util mean | GPU-util max |
|---|---:|---:|---:|---:|
| 1K / 128 | 13.87% | 81% | 25.00% | 100% |
| 2K / 128 | 15.65% | 80% | 25.00% | 100% |
| 4K / 128 | 12.61% | 69% | 24.92% | 100% |
| 8K / 128 | 8.95% | 69% | 25.00% | 100% |
| 16K / 128 | 6.11% | 58% | 25.00% | 100% |
| 32,640 / 128 | 5.61% | 43% | 24.91% | 100% |

These are coarse 200 ms whole-request samples. They show burst peaks, not sustained per-kernel bandwidth. Nsight Compute access is required for definitive Marlin DRAM and cache metrics.

## Artifacts and hashes

Official evaluation:

- `benchmarks/qwen38_4090/evaluation/runs/iterate34-default/submission.json` — `de858752d85e43966800b4d061c43be81ded5e20855b54f3ecb197372c171c26`
- `benchmarks/qwen38_4090/evaluation/runs/iterate34-default/raw.jsonl` — `8bf740badec881ce3f63484fa4e2a3221292ac9f1d08e68454e4c47f5ec9d111`
- `benchmarks/qwen38_4090/evaluation/runs/iterate34-default/environment.json` — `1d6983cbb79d85632d45f394723ebdff07ba895eecf1a1e435b95a478726916a`
- trajectory reference — `904fa6f54dec58d9638a2b067c78b145a48c63a4b838fba26e473d866576b793`

Profiler and build artifacts:

- `/tmp/iterate34-decode-graph.nsys-rep` — `d27254861161820ec5f7a593318a994606f6e92d4d08561cdb7d730ba7c782d6`
- `/tmp/iterate34-decode-graph.sqlite` — `85697c18ba09582802afe33b9df17cbd0b4c08bbcc302c931745224be00977fc`
- `/tmp/iterate34-kernels_cuda_gpu_kern_sum_nvtx-name_base.csv` — `566d292fa7561d85570e46861c8ed0ead60e7697aebb49313e6bdc37f94f101e`
- `target/release/apxinf` — `69ac61e82de77dac2fb916d4cc271b7e903c9c0adf16ecc47653142b3d6251a5`
- rebuilt `marlin_adapter.o` — `20fe14a77aad426c59650647bfa91056532e7b44da164d1497e4822980b36d14`

`/tmp` artifacts are workstation-local and require copying to durable storage before workstation shutdown.

## Reproduction

```bash
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export APXINF_CUDA_ARCH=sm_89
export RUSTFLAGS='-C link-arg=-fuse-ld=gold'

cargo build --release --features cuda -p apxinf --bin apxinf -j 40
CUDA_VISIBLE_DEVICES=0 ./target/release/apxinf serve \
  --model ../model/qwen --host 127.0.0.1 --port 8313

python3 benchmarks/qwen38_4090/evaluation/run_evaluation.py \
  --dataset benchmarks/qwen38_4090/evaluation/.cache/public \
  --context-dataset benchmarks/qwen38_4090/evaluation/.cache/context-iter3 \
  --model-dir ../model/qwen --base-url http://127.0.0.1:8313 \
  --implementation-name apxinf-iteration34 \
  --implementation-revision iterate34-working --backend apxinf \
  --profile public_calibration \
  --trajectory-reference /tmp/iter25-trajectory-reference.json \
  --run-context --timeout 2400 --run-id iterate34-default \
  --output-dir benchmarks/qwen38_4090/evaluation/runs
```

The release CUDA build, canonical evaluator, and graph-node profile completed. No service or profiler process remained live after capture.