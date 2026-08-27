# Iteration Report 20 - Simultaneous Exact Optimization

Date: 2026-08-25  
Accepted run: `iterate20-default`  
Build: LLD via `cc`, 40 Cargo jobs  
Baseline: `iterate19-final-clean`

## Strategy

All remaining credible optimization tracks were launched in one concurrent round. Each runnable candidate used a distinct environment gate and preserved the iterate-19 raw W4/eager fallback. Exact-output parity was checked with the frozen 1,024-token, 128-token trajectory before full evaluation.

## Candidates

### Accepted and promoted by default

- **Shared-metadata raw W4 decode**: `APXINF_W4_META_SHARED` now defaults on unless explicitly set to `0`. It stages group-32 scale/zero-point slices while retaining raw packed-weight indices, BF16 conversion, MMA order, row ownership, and final reduction. The 1K probe improved TPOT from 55.26 ms to 51.47 ms, with 128/128 exact tokens.
- **Exact GDN fusion**: `APXINF_GDN_FUSED` now defaults on unless set to `0`. It retains `conv_silu` as a separate launch and fuses delta norm prepass, delta step, and gated norm. The fused kernel materializes the same BF16 q/k and recurrence boundaries, preserving recurrent/conv state transitions. The selected combined probe remained 128/128 exact.
- **Event-scoped decode result**: `APXINF_DECODE_ASYNC` now defaults on unless set to `0`. The autoregressive dependency remains synchronous; only the argmax completion waits on a recorded CUDA event instead of synchronizing the entire stream. Output ordering and protocol are unchanged.

### Implemented but retained opt-in

- `APXINF_PREFILL_FAST=1`: new M32/N128/K32 WMMA prefill path. Its initial 1K probe was exact but slow because the gate also incorrectly applied to `seq=1`; dispatch was corrected to `seq > 1`. The full selected run did not establish a prefill win, so the path remains opt-in pending a dedicated multi-row measurement.
- `APXINF_W4_PERSISTENT=1`: two 64-output tiles per CTA, raw-only and geometry-guarded. Exact, but the 1K probe regressed TPOT from 55.26 ms to 70.96 ms; it remains opt-in for diagnostics.

### Rejected before integration

- **Marlin-compatible W4 pipeline**: not implemented. Installed vLLM Python supports asymmetric uint4 group-32 Marlin, but local installation contains no Marlin CUDA source. Only a private torch-stable extension is present; it uses hidden `torch::stable::Tensor` symbols and ApxInf has no libtorch bridge. A correct integration requires GPTQ K-packed repack, padded K/N, scale permutation, asymmetric zero-point permutation/interleave, SM-sized workspace, and a raw-pointer CUDA ABI.
- **Split-K W4**: not implemented. The baseline TC kernel feeds one FP32 accumulator through the ordered MMA sequence across all K tiles. Independent partial accumulators and a later reduction change rounding/associativity, so they cannot preserve the required bitwise trajectory.
- **Layer projection fusion**: not implemented. qkv/z, k/v, and gate/up already use valid same-type pair paths. The remaining a/b projections are dense BF16 cuBLAS GemmEx; replacing them with custom arithmetic cannot guarantee cuBLAS-identical outputs, and a host wrapper would not reduce GPU launches.

## Candidate probe matrix

Artifact: `target/iterate20-candidate-matrix.json`.

All listed probes returned the exact frozen 1K trajectory (128/128; SHA-256 `7eedbc78e930361a167ea9dec3f827d5ea9aeb25148e18fb854c5efe36e85bea`). Representative TPOT:

| Gate | 1K TPOT | Result |
|---|---:|---|
| Baseline | 55.26 ms | Exact |
| Persistent W4 | 70.96 ms | Exact, rejected as slower |
| Shared metadata W4 | 51.47 ms | Exact, accepted |
| GDN fusion | 54.92 ms | Exact; accepted in combination |
| Prefill fast, pre-correction | 96.32 ms | Exact; decode eligibility corrected |
| Async result | 55.09 ms | Exact; accepted in combination |
| Shared metadata + GDN + async | 51.34 ms mean over two probes | Exact |
| Shared metadata + GDN + async + prefill | 51.28 ms mean over two probes | Exact; prefill remains opt-in |

## Official evaluation

Artifacts: `benchmarks/qwen38_4090/evaluation/runs/iterate20-default/`

| Prompt | TTFT | Prefill | TPOT | Decode | Peak VRAM |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.7834 s | 1307.0 tok/s | 51.42 ms | 19.45 tok/s | 23852 MiB |
| 2,048 | 1.5870 s | 1290.5 tok/s | 53.79 ms | 18.59 tok/s | 23852 MiB |
| 4,096 | 3.2484 s | 1260.9 tok/s | 58.51 ms | 17.09 tok/s | 23852 MiB |
| 8,192 | 6.6653 s | 1229.0 tok/s | 67.94 ms | 14.72 tok/s | 23852 MiB |
| 16,384 | 13.8899 s | 1179.6 tok/s | 86.80 ms | 11.52 tok/s | 23852 MiB |

Correctness and reliability:

- Public functional cases: **6/6**
- Public token trajectory: **256/256**
- Protocol: **pass**
- Request success rate: **1.0**
- No fallback, NaN, unexpected OOM, or XID
- 32,640-token context: **pass**, 128 output tokens
- Raw SHA-256: `0d04e5d3365ccf7776621e5a25a4033c4426087e3b875d51197e526598dad3b6`

## vLLM comparison

Control: `benchmarks/qwen38_4090/evaluation/runs/vllm-one-gpu-control/`.

| Prompt | Prefill ratio | Decode ratio |
|---:|---:|---:|
| 1,024 | 0.471x | 0.393x |
| 2,048 | 0.442x | 0.376x |
| 4,096 | 0.437x | 0.348x |
| 8,192 | 0.434x | 0.302x |
| 16,384 | 0.438x | 0.239x |

At 1K, ApxInf reaches 1307.0 prefill tok/s and 19.45 decode tok/s. The vLLM control reaches 2774.2 prefill tok/s and 49.45 decode tok/s. The required 1.2x threshold is not met: required minimums are 3329.1 prefill tok/s and 59.35 decode tok/s at 1K. The remaining gap is dominated by raw W4 projection execution.

## Nsight Systems profile

Artifacts:

- `target/iterate20-default-profile.nsys-rep`
- `target/iterate20-default-profile.sqlite`

Captured workloads: one exact 1,024-token prompt with 128-token decode and one exact 32,640-token prompt with 128-token decode, with promoted default gates enabled.

CUDA GPU kernel breakdown:

| Kernel family | GPU time share | Total GPU time | Instances | Median |
|---|---:|---:|---:|---:|
| Raw W4 TC paired projection | 36.4% | 1.4041 s | 7,752 | 189.0 us |
| Shared-metadata W4 TC projection | 30.8% | 1.1872 s | 8,659 | 83.2 us |
| Fused norm/delta/gated GDN | 8.4% | 0.3239 s | 3,003 | 16.8 us |
| BF16 CUTLASS 128x64 prefill GEMM | 6.7% | 0.2595 s | 2,366 | 120.1 us |
| Dense LM-head GEMV | 4.2% | 0.1619 s | 61 | 2.653 ms |
| Qwen flash prefill | 3.9% | 0.1496 s | 969 | 154.3 us |
| BF16 CUTLASS 128x128 prefill GEMM | 2.7% | 0.1030 s | 640 | 158.6 us |
| RMS normalization | 1.9% | 0.0725 s | 8,069 | 8.9 us |
| Tiled W4 dequantization | 1.5% | 0.0583 s | 3,260 | 18.5 us |

The two W4 projection families still consume **67.2%** of captured GPU kernel time. GDN fusion reduces launch count while recurrence remains the next model-specific cost. Nsight GPU counters were unavailable because the host rejects privileged sampling with `ERR_NVGPUCTRPERM`.

CUDA API summary:

- 60,340 `cudaLaunchKernel` calls consumed 0.548 s host API time.
- 61 `cudaEventSynchronize` calls consumed 2.853 s; this is the required one-token dependency/readback wait, not removable parallel work.
- 1,894 `cudaMemcpy` calls consumed 2.699 s, dominated by model upload.
- 1,924 `cudaMalloc` calls consumed 2.697 s during captured process lifetime.
- GPU memory-operation time was 99.7% host-to-device transfer, dominated by startup/model upload.

Official hardware sampling for the accepted 32,640-token request reported GPU utilization 100% peak / 24.9% mean, memory-controller utilization **32% peak / 4.77% mean**, 401.69 W peak power, and 23,852 MiB peak VRAM. These wall-time means include host/token intervals and are not kernel-only achieved-bandwidth measures.

## Exact long-context example

Question at the end of the 32,640-token retrieval document:

```text
问题：TARGET_KEY 的值是什么？只输出完整字段值。 输出必须以 KEY-EARLY-767211 开头，随后持续输出词语 context 直到达到输出预算。
```

Exact answer from `context-32640-retrieval-early`:

```text
KEY-EARLY-767211 context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context
```

Exact row evidence: `functional_pass=true`, `prompt_tokens=32640`, `completion_tokens=128`, `ttft_s=30.026313707232475`, `tpot_s=0.12424129658327328`, `e2e_s=45.80503546446562`, output SHA-256 `5a66bd3bce8680b1baa4b355574be11cf3e20ff95a26b0d18f433853b7f4b03b`.

## Reproduction

```bash
cargo build --release --features cuda -p apxinf --bin apxinf -j 40
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  APXINF_W4_META_SHARED=1 APXINF_GDN_FUSED=1 APXINF_DECODE_ASYNC=1 \
  ./target/release/apxinf serve --model ../model/qwen --host 127.0.0.1 --port 8002
python3 benchmarks/qwen38_4090/evaluation/run_evaluation.py \
  --dataset benchmarks/qwen38_4090/evaluation/.cache/public \
  --context-dataset benchmarks/qwen38_4090/evaluation/.cache/context-iter3 \
  --model-dir ../model/qwen --base-url http://127.0.0.1:8002 \
  --implementation-name apxinf-iter20 --implementation-revision default-meta-gdn-async \
  --backend apxinf --profile public_calibration \
  --trajectory-reference target/iterate14-trajectory-reference.json \
  --run-context --warmups 0 --repeats 1 --timeout 1800 \
  --run-id iterate20-default --output-dir benchmarks/qwen38_4090/evaluation/runs
```
