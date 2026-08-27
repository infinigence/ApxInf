# Iteration Report 21 - Projection Tile Optimization

Date: 2026-08-25  
Accepted run: `iterate21-default`  
Baseline: `iterate20-default`  
Build: LLD via `cc`, 40 Cargo jobs

## Scope

Eight tracks ran concurrently: paired W4 metadata staging, vectorized W4 loads, alternate W4 tile geometry, weight swizzle, packed prefill metadata, fused-GDN tail cleanup, argmax event audit, and full-token CUDA graph capture. Runnable paths were isolated behind gates and compared with the exact iteration-20 default.

## Accepted default

`APXINF_W4_TILE_ALT` now defaults on unless explicitly set to `0` for eligible raw-layout, single-row, group-32 Qwen projections. It uses 32 output rows per CTA, four warps, and 128 threads instead of the prior 64-row/eight-warp geometry. It preserves row mapping, packed-weight and metadata indices, BF16 dequantization, ordered MMA calls, accumulator layout, and final reduction.

Two independent stable probes returned the frozen 128/128 trajectory and measured 49.79-49.91 ms TPOT versus 51.30 ms for iteration 20.

## Candidate matrix

Artifact: `target/iterate21-candidate-matrix.json`.

| Candidate | Stable 1K TPOT | Exactness | Decision |
|---|---:|---:|---|
| Iteration-20 control | 51.30 ms | 128/128 | Baseline |
| Paired metadata staging | 54.17 ms | 128/128 | Rejected as slower |
| Vectorized W4 loads | 68.26 ms | 128/128 | Rejected as slower |
| Alternate 32-row tile | 49.79 ms | 128/128 | Accepted |
| Full-stream event audit | 51.43 ms | 128/128 | Rejected; event wait remains faster |
| Packed prefill metadata | 51.41 ms decode | 128/128 | Opt-in; no decode gain |
| Alternate tile + pair metadata | 52.67 ms | 128/128 | Rejected as slower |

Blocked paths:

- Full-token CUDA graph capture is unsafe because RoPE/KV launches capture `start_pos` by value, token IDs and positions are uploaded outside the captured loop, recurrent/conv/KV state mutates per token, and activation/logit buffers overlap.
- The accepted fused GDN already has no redundant tail copy or launch; further removal would erase required BF16 boundaries.
- Non-identity row swizzling cannot improve the current contiguous raw-TC row access without inverse gathers or worse coalescing.

## Official evaluation

Artifacts: `benchmarks/qwen38_4090/evaluation/runs/iterate21-default/`

| Prompt | TTFT | Prefill | TPOT | Decode | Prefill/vLLM | Decode/vLLM |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.7831 s | 1307.6 tok/s | 49.85 ms | 20.06 tok/s | 0.471x | 0.406x |
| 2,048 | 1.5863 s | 1291.1 tok/s | 52.20 ms | 19.16 tok/s | 0.442x | 0.388x |
| 4,096 | 3.2465 s | 1261.7 tok/s | 56.92 ms | 17.57 tok/s | 0.437x | 0.358x |
| 8,192 | 6.6632 s | 1229.4 tok/s | 66.36 ms | 15.07 tok/s | 0.435x | 0.309x |
| 16,384 | 13.8818 s | 1180.2 tok/s | 85.23 ms | 11.73 tok/s | 0.438x | 0.244x |

Correctness and reliability:

- Public functional cases: **6/6**
- Public trajectory: **256/256**
- Protocol: **pass**
- Request success rate: **1.0**
- No fallback, NaN, unexpected OOM, or XID
- 32,640-token context: **pass**, 128 output tokens
- Peak VRAM: **23,852 MiB**
- Raw SHA-256: `66c9e70accdb63887a98be472e710c032e5e6b67fc860b191c7ecceae9634e68`

At 1K, ApxInf reaches 1307.6 prefill tok/s and 20.06 decode tok/s. The vLLM control reaches 2774.2 prefill tok/s and 49.45 decode tok/s. The required 1.2x threshold remains unmet; required minimums are 3329.1 prefill tok/s and 59.35 decode tok/s.

## Nsight Systems profile

Artifacts:

- `target/iterate21-default-profile.nsys-rep`
- `target/iterate21-default-profile.sqlite`

Captured workloads: one exact 1K/128-token request and one exact 32,640/128-token request.

| Kernel family | GPU share | Total | Instances | Median |
|---|---:|---:|---:|---:|
| Paired raw W4 TC | 37.4% | 1.4032 s | 7,746 | 192.4 us |
| Alternate-tile raw W4 TC | 29.1% | 1.0936 s | 8,653 | 77.5 us |
| Fused norm/delta/gated GDN | 8.6% | 0.3217 s | 3,001 | 16.8 us |
| BF16 CUTLASS 128x64 prefill GEMM | 6.9% | 0.2576 s | 2,366 | 120.1 us |
| Dense LM-head GEMV | 4.3% | 0.1619 s | 61 | 2.653 ms |
| Qwen flash prefill | 4.0% | 0.1494 s | 968 | 154.2 us |
| BF16 CUTLASS 128x128 prefill GEMM | 2.7% | 0.1024 s | 640 | 158.0 us |

W4 projections still account for 66.5% of GPU kernel time. The alternate tile reduces the single-projection median from iteration 20's 83.2 us to 77.5 us. Paired W4 is now the dominant target, but the tested pair-metadata path was slower.

CUDA API summary:

- 60,301 `cudaLaunchKernel` calls: 0.563 s host API time.
- 61 `cudaEventSynchronize` calls: 2.737 s; these enforce the autoregressive dependency.
- 1,894 `cudaMemcpy` calls: 2.856 s, dominated by model upload.
- 1,924 `cudaMalloc` calls: 0.651 s during process lifetime.
- GPU memory-operation time was 99.8% host-to-device, dominated by startup/model upload.

Privileged SM/DRAM counters remain unavailable due to host `ERR_NVGPUCTRPERM`. Official 32K sampling measured memory-controller utilization at **31% peak / 4.87% mean**, GPU utilization at 100% peak / 25% mean, and 401.53 W peak power.

## Exact long-context example

Question:

```text
问题：TARGET_KEY 的值是什么？只输出完整字段值。 输出必须以 KEY-EARLY-767211 开头，随后持续输出词语 context 直到达到输出预算。
```

Exact answer:

```text
KEY-EARLY-767211 context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context
```

Exact row evidence: `functional_pass=true`, `prompt_tokens=32640`, `completion_tokens=128`, `ttft_s=29.999435625970364`, `tpot_s=0.12261037815978208`, `e2e_s=45.571051344275475`, output SHA-256 `5a66bd3bce8680b1baa4b355574be11cf3e20ff95a26b0d18f433853b7f4b03b`.

## Reproduction

```bash
cargo build --release --features cuda -p apxinf --bin apxinf -j 40
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  ./target/release/apxinf serve --model ../model/qwen --host 127.0.0.1 --port 8002
python3 benchmarks/qwen38_4090/evaluation/run_evaluation.py \
  --dataset benchmarks/qwen38_4090/evaluation/.cache/public \
  --context-dataset benchmarks/qwen38_4090/evaluation/.cache/context-iter3 \
  --model-dir ../model/qwen --base-url http://127.0.0.1:8002 \
  --implementation-name apxinf-iter21 --implementation-revision default-tile-alt \
  --backend apxinf --profile public_calibration \
  --trajectory-reference target/iterate14-trajectory-reference.json \
  --run-context --warmups 0 --repeats 1 --timeout 1800 \
  --run-id iterate21-default --output-dir benchmarks/qwen38_4090/evaluation/runs
```
