# Iteration Report 30 - Exact Split Attention and Vectorized Packed GDN

Date: 2026-08-26  
Implementation revision label: `f4793ee6d7782c61a55fb2db95cc52d438b5d473` plus the measured worktree  
Run id: `iterate30-clean`

## Objective and verdict

Continue optimizing Qwen3.8-27B AWQ INT4 inference on one RTX 4090 while preserving the frozen token trajectory and official public functional cases. This iteration targeted the non-GEMM decode remainder, specifically full-attention reduction occupancy and packed GDN state traffic, and required both official evaluation and long-context evidence.

**Correctness verdict: pass. Performance verdict: the requested 1.2x vLLM throughput target is not met.** The delivered default reaches 39.63 decode tok/s at 1K and 25.34 decode tok/s at 8K. The local vLLM control reaches 49.45 and 48.75 tok/s respectively, so ApxInf is 0.801x and 0.520x of vLLM throughput in those cells. Prefill is also slower than the local vLLM control at every measured public cell.

The exact split-attention path and vectorized packed-GDN state transfers are retained. Two RMSNorm candidates were measured and rejected: both were exact, but neither improved end-to-end decode latency. The final report does not claim an unmeasured hardware-counter bandwidth result; Nsight Systems kernel and NVTX traces are used for timing and bottleneck evidence, while the evaluator's client-observed latency remains authoritative.

## Delivered changes

### Exact split full-attention decode

The Qwen3.5 full-attention decode geometry is 24 query heads, 4 KV heads, and head dimension 256. The existing exact fused FlashAttention path used one partial kernel per head and a merge kernel. This iteration adds an exact two-pass variant:

- pass 1 schedules 48 partial CTAs instead of 24, preserving the original eight warp timestep streams and per-head accumulation order;
- pass 2 performs the same merge/gate operation over the persistent partial buffer;
- the persistent scratch is allocated from the existing CUDA arena, so no per-token allocation is introduced;
- the previous path remains available with `APXINF_FLASH_SPLIT_256=0`.

The split path passed both frozen trajectory gates:

| Gate | TPOT | Output SHA-256 | Result |
|---|---:|---|---|
| 1K, split attention | 25.41 ms/token | `7eedbc78e930361a167ea9dec3f827d5ea9aeb25148e18fb854c5efe36e85bea` | exact |
| 8K, split attention | 39.82 ms/token | `026beeb172d77fa177787305d9c1b1d6e179f13867e8e829c6d4aad17ca820fa` | exact |

After cleanup, split attention is the default for the supported geometry; `APXINF_FLASH_SPLIT_256=0` is the rollback switch. The post-cleanup binary passed the same 1K and 8K SHA gates.

### Vectorized packed GDN state transfers

The packed Qwen linear-attention kernel retains its recurrence order, BF16 materialization boundaries, shared-memory layout, and per-token synchronization. State load/store at the kernel boundary now uses aligned `float4` transfers across adjacent value lanes. This changes global transaction width without changing recurrence arithmetic.

The candidate passed the frozen trajectory gates:

| Gate | TPOT | Output SHA-256 | Result |
|---|---:|---|---|
| 1K, vectorized GDN state | 25.64 ms/token | `7eedbc78e930361a167ea9dec3f827d5ea9aeb25148e18fb854c5efe36e85bea` | exact |
| 8K, vectorized GDN state | 40.74 ms/token | `026beeb172d77fa177787305d9c1b1d6e179f13867e8e829c6d4aad17ca820fa` | exact |

This path is unconditional in the specialized packed GDN kernel in the delivered build.

### Rejected RMSNorm candidates

Two exact candidates were tested and removed from the delivered code:

1. Residual-add plus exact RMSNorm fusion, with the residual explicitly rounded to BF16 before reduction. It was exact at 1K and 8K but neutral to slightly slower end-to-end.
2. RMSNorm reread, which preserved the reduction tree but reread BF16 input instead of caching FP32 row values in dynamic shared memory. It was exact at 1K but measured 25.63 ms/token versus 25.42 ms/token for the split-attention control, so it was rejected.

The base 256-thread shared-memory RMSNorm implementation remains.

## Final canonical evaluation

Evaluator: `benchmarks/qwen38_4090/evaluation/run_evaluation.py`  
Profile: `public_calibration`  
Warmups: 0  
Measured repeats: 1  
Dataset: `benchmarks/qwen38_4090/evaluation/.cache/public`  
Context dataset: `benchmarks/qwen38_4090/evaluation/.cache/context-iter3`

### Public performance cells

| Cell | TTFT | Prefill | TPOT | Decode | VRAM peak |
|---|---:|---:|---:|---:|---:|
| text-perf-1024 | 0.8554 s | 1197.1 tok/s | 25.231 ms | 39.63 tok/s | 22136 MiB |
| text-perf-2048 | 1.7314 s | 1182.9 tok/s | 27.265 ms | 36.68 tok/s | 22136 MiB |
| text-perf-4096 | 3.5407 s | 1156.8 tok/s | 31.336 ms | 31.91 tok/s | 22136 MiB |
| text-perf-8192 | 7.2492 s | 1130.1 tok/s | 39.461 ms | 25.34 tok/s | 22136 MiB |
| text-perf-16384 | 15.0791 s | 1086.5 tok/s | 55.704 ms | 17.95 tok/s | 22136 MiB |

All public performance cells produced the complete 128-token budget. The clean run's raw artifact SHA-256 is `7871f207e7b552f5e7292504645348ead1b7e98aac131249f9e52189eb7b8d18`.

### Correctness and reliability

- Protocol: pass.
- Public functional cases: **6/6**.
- Public trajectory: **256/256**, zero token edit distance at both 1K and 8K.
- Request success rate: **1.0**.
- No fallback, NaN, unexpected OOM, or XID.
- Service healthy after the context run and after protocol failures.
- Model revision: `63768c10df38c0395e12ef49edac1bd539eaeeea`.
- Health capacity: 32768 tokens; exact pretokenized input and token-ID output capabilities advertised.

Trajectory evidence:

| Cell | Expected SHA-256 | Observed SHA-256 | Result |
|---|---|---|---|
| text-perf-1024 | `7eedbc78e930361a167ea9dec3f827d5ea9aeb25148e18fb854c5efe36e85bea` | identical | 128/128 |
| text-perf-8192 | `026beeb172d77fa177787305d9c1b1d6e179f13867e8e829c6d4aad17ca820fa` | identical | 128/128 |

### Local vLLM control comparison

Control artifact: `benchmarks/qwen38_4090/evaluation/runs/vllm-one-gpu-control/submission.json`.

| Metric | ApxInf | vLLM | ApxInf / vLLM | ApxInf throughput / vLLM throughput |
|---|---:|---:|---:|---:|
| 1K TTFT | 0.8554 s | 0.3691 s | 2.32x | — |
| 1K TPOT | 25.231 ms | 20.220 ms | 1.248x slower | **0.801x** |
| 8K TTFT | 7.2492 s | 2.8954 s | 2.50x | — |
| 8K TPOT | 39.461 ms | 20.512 ms | 1.923x slower | **0.520x** |

The requested minimum 1.2x throughput over vLLM would require at least 59.35 tok/s at 1K and 58.50 tok/s at 8K under this control. The delivered 39.63 and 25.34 tok/s do not meet that threshold. The official score script reports a provisional public-calibration score of **70.0296 leaderboard points**, with **56.0237 automated course points**; the score is diagnostic because the profile is a single-repeat public calibration cohort.

## Long-context evaluation

The canonical context diagnostic used a frozen 32,640-token retrieval-early workload with 128 output tokens, greedy decoding, and normalized-prefix validation. Per the report scope, no concrete question, prompt, or generated answer testcase is reproduced here.

- Prompt length: **32,640 tokens**.
- TTFT: **32.3808 s**.
- TPOT: **87.910 ms/token** (**11.38 tok/s**).
- E2E: **43.5455 s**.
- Peak VRAM: **22136 MiB**.
- Output budget: **128 tokens completed**.
- Output SHA-256: `5a66bd3bce8680b1baa4b355574be11cf3e20ff95a26b0d18f433853b7f4b03b`.
- Validator result: normalized-prefix pass.
- Service healthy afterward.

## Nsight Systems profile

Profiler: NVIDIA Nsight Systems 2025.1.1 at `/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys`. The installed `nvprof` is CUDA 11.5 and explicitly skips profiling for compute capability 7.5 and higher; its first attempt also failed because `--analysis-metrics` requires an export file. Nsight Systems was therefore used for the valid profile capture.

Profile artifacts:

- `/tmp/iterate30-final.nsys-rep`, SHA-256 `4a7f8467e73118705215c18996ad92abcbe0b749ea21f0d750bf3db4695fa204`.
- `/tmp/iterate30-final.sqlite`, SHA-256 `5dccf854abd838971504e913813b49be615caeb45738cae4c3d0851efae8aa08`.
- Exported summaries: `/tmp/iterate30-stats_nvtx_gpu_proj_sum.csv`, `/tmp/iterate30-stats_cuda_gpu_kern_sum.csv`, and `/tmp/iterate30-stats_cuda_gpu_mem_time_sum.csv`.

The profiled process used the retained split-attention and vectorized-GDN kernel configuration with `APXINF_FLASH_SPLIT_256=1`; the capture preceded removal of disabled RMSNorm candidate symbols and therefore has the same executed hot path as the delivered binary. Client timing under profiling was 25.492 ms/token and the output SHA matched the frozen 1K trajectory. The profile includes model startup, prefill, and decode; aggregate kernel totals are not substituted for canonical client TPOT.

### Kernel breakdown

The Nsight `cuda_gpu_kern_sum` report contains 31,108 kernel rows. Dominant families:

| Kernel family | Instances | Total GPU time | Share |
|---|---:|---:|---:|
| Marlin M=1 main tile | 6617 | 345.827 ms | 23.8% |
| CUTLASS BF16 128x64 GEMM | 2366 | 250.629 ms | 17.2% |
| Prefill delta recurrence | 96 | 244.804 ms | 16.8% |
| Marlin inverse dequant | 606 | 163.406 ms | 11.2% |
| Raw W4 paired projection | 1169 | 138.375 ms | 9.5% |
| CUTLASS BF16 128x128 GEMM | 640 | 99.639 ms | 6.8% |
| Split FlashAttention partial | 389 | 51.907 ms | 3.6% |
| RMSNorm | 3396 | 29.798 ms | 2.0% |
| Conv/SiLU prefill | 96 | 21.713 ms | 1.5% |
| Packed GDN recurrence | 1168 | 20.602 ms | 1.4% |

The dominant cost is quantized projection execution: Marlin, inverse dequant, raw paired W4 projections, and BF16 GEMMs account for most captured GPU time. Split FlashAttention and packed GDN are secondary. RMSNorm is not the limiting stage after the exact fusion and reread candidates were rejected.

### Per-module ApxInf versus vLLM implementation and cost comparison

The following table compares the concrete implementation used by each backend for every profiled module/bucket. Costs are decode-normalized profiler values: summed GPU kernel time divided by argmax-delimited intervals, using 25 ApxInf intervals and 67 vLLM intervals. They are module-level profiler costs, not replacements for client-observed TPOT. The traces include different graph-capture and warmup activity, so implementation and cost comparisons are directional rather than cycle-equivalent.

#### Complete profile-family coverage

The raw Nsight summaries contain additional families beyond the decode-normalized buckets. This table lists every named family reported for both backends and makes the mapping explicit. Aggregate totals include startup, prefill, graph capture, and decode; they must not be divided directly to infer canonical TPOT because the two captures contain different workloads and interval counts.

| Functional module | ApxInf kernel family | ApxInf instances | ApxInf total GPU time | vLLM kernel family | vLLM instances | vLLM total GPU time | Mapping |
|---|---|---:|---:|---|---:|---:|---|
| Quantized W4 projections | Marlin W4A16; raw paired W4 projections; inverse dequant | 7627 + 1205 + 606 | 365.647 + 144.694 + 163.406 ms | Marlin W4A16 | 18141 | 2017.584 ms | Partial one-to-one: vLLM routes more projection work through Marlin; ApxInf splits raw projection and inverse-dequant families. |
| BF16 projection GEMMs | CUTLASS BF16 GEMM families, including 128x64 and 128x128 tiles | 3836 | 391.347 ms | No separate named equivalent; work is distributed across Marlin, fused Triton, and other graph kernels | n/a | Included in those buckets | No one-to-one symbol mapping. ApxInf exposes BF16 projection GEMMs as a separate family. |
| Full attention | Split FlashAttention-256 partial/merge and related attention kernels | 401 | 57.400 ms | FlashAttention prefill/decode | 27 | 4324.054 ms | Same functional module, but vLLM total includes prefill/decode and graph activity while ApxInf row is primarily the named FlashAttention family. |
| RMSNorm | `rms_norm_bf16_kernel` and related RMSNorm kernels | 3494 | 31.638 ms | Fused/Triton RMSNorm kernels | 9049 | 30.599 ms | Same function, different fusion and launch granularity. |
| Linear-attention GDN prefill | `qwen35_prefill_delta_step_kernel` | 96 | 253.884 ms | Conv1D plus recurrent graph work | 3081 Conv1D; included GDN | 6.675 ms Conv1D; remainder included | No direct one-to-one family; ApxInf exposes a large prefill delta kernel, while vLLM distributes this work across Conv1D and graph kernels. |
| Linear-attention GDN decode | `qwen35_packed_delta_gated_kernel` and related GDN decode kernels | 1205 | 20.759 ms | Packed recurrent GDN kernels | 3129 | 23.182 ms | Same recurrent function with different packing/scheduling. |
| LM head | BF16/W4 LM-head GEMV plus deterministic GPU argmax | 2436 | 10.819 ms LM-head bucket | Dense BF16 LM-head GEMV plus selection | 3307 | 187.319 ms | Same output-selection function; ApxInf uses the exact W4 head, vLLM uses a dense head. |
| Elementwise and epilogues | Custom add, SiLU, scaling, gating, KV/activation post-processing | Included in Qwen W4/Other buckets | Not isolated in raw ApxInf family table | Fused Triton/elementwise | 28353 | 217.136 ms | vLLM exposes a larger fused elementwise bucket; ApxInf distributes these operations across custom kernels and projection epilogues. |
| KV cache | Custom BF16 KV-cache append and position-indexed writes | Included in Other/attention-related families | Not isolated in raw ApxInf family table | Graph-aware/paged KV-cache updates | Included in Other/FlashAttention | Not isolated in raw vLLM family table | Same cache function; neither raw summary isolates a standalone total consistently. |
| Sampling and argmax | `argmax_bf16_single_launch_kernel` and deterministic tie handling | Included in Other/LM-head families | Not isolated in raw ApxInf family table | Logits selection and sampling kernels | Included in Other/LM-head families | Not isolated in raw vLLM family table | Same terminal selection function; normalized decode bucket is the reliable comparison. |
| Residual/runtime remainder | Other custom CUDA kernels and runtime work | 11082 | 235.756 ms | Other CUDA/runtime kernels | 5114 | 433.790 ms | Unmatched residual buckets; not a functional module and not directly comparable. |

The normalized table below is the fairer module-cost comparison because it re-buckets symbols inside argmax-delimited decode intervals. The complete-family table above explains where raw profile totals differ and prevents an unmatched family from being mistaken for a missing implementation.

| Module/bucket | ApxInf implementation | vLLM implementation | ApxInf cost | vLLM cost | ApxInf/vLLM | Comparison |
|---|---|---|---:|---:|---:|---|
| W4 Marlin projection | Marlin W4A16 M=1 projection tiles with ApxInf-owned dispatch and exact accumulation order. | Marlin W4A16 projection kernels, including the control's M=1/M=4 paths. | 14.543 ms/token, 57.50% | 20.950 ms/token, 82.03% | 0.694x | ApxInf is cheaper in this bucket, but it moves some projection work into its separate raw-W4 bucket below. |
| Custom/raw W4 projection | `qwen35_gemm_w4a16_bf16_tc_pair_weight_stage_kernel` and related exact paired QKV/projection kernels. | No matching standalone bucket; equivalent work is primarily routed through vLLM Marlin/Triton paths. | 5.764 ms/token, 22.79% | 0 ms/token, 0% | n/a | ApxInf-only cost. This is the main ownership/scheduling penalty relative to vLLM. |
| Combined quantized W4 projection | ApxInf Marlin plus custom/raw W4 projection kernels. | vLLM Marlin W4A16 projection work. | 20.307 ms/token, 80.29% | 20.950 ms/token, 82.03% | 0.969x | Shared dominant bottleneck. Both engines spend most normalized decode time sweeping quantized weights. |
| Attention | Exact split FlashAttention-256 partial kernels plus merge/gate; fallback fused FlashAttention-256 remains available. | FlashAttention forward kernels used for prefill/decode, with vLLM graph/fusion scheduling. | 2.290 ms/token, 9.05% | 0.195 ms/token, 0.76% | 11.744x | Largest ApxInf-specific gap in this comparison. ApxInf attention is exact but less aggressively fused/scheduled in the captured decode bucket. |
| RMSNorm | Standalone `rms_norm_bf16_kernel`; rejected residual-fusion and reread candidates were removed. | Triton/fused RMSNorm and epilogue work, with more normalization folded into surrounding graph operations. | 1.148 ms/token, 4.54% | 0.333 ms/token, 1.30% | 3.448x | ApxInf spends more standalone time because vLLM absorbs more norm work into fused epilogues. |
| GDN/recurrent path | Packed `qwen35_packed_delta_gated_kernel`, including recurrent update, gated RMSNorm, and vectorized recurrent-state transfers. | Packed recurrent GDN kernels plus vLLM's recurrent scheduling. | 0.827 ms/token, 3.27% | 0.452 ms/token, 1.77% | 1.830x | ApxInf is slower, but this is secondary to quantized projection and attention cost. |
| LM head | Exact W4A16 LM-head path with Marlin-compatible quantized weights and deterministic GPU argmax. | Dense BF16 LM-head GEMV followed by vLLM sampling/selection. | 0.429 ms/token, 1.69% | 2.754 ms/token, 10.79% | 0.156x | ApxInf is substantially cheaper after the exact W4 LM-head cutover; LM head is not the remaining bottleneck. |
| Fused elementwise/epilogue | ApxInf custom CUDA add, SiLU, scaling, gating, and small post-projection kernels. | Fused Triton elementwise and epilogue kernels integrated into vLLM graph execution. | 0.201 ms/token, 0.79% | 0.591 ms/token, 2.31% | 0.340x | ApxInf has lower cost in this bucket; vLLM's advantage comes from folding more work into this bucket, not from lower total decode cost. |
| KV cache | ApxInf custom BF16 KV-cache append and position-indexed cache writes. | vLLM paged/graph-aware KV-cache update and attention-cache kernels. | 0.051 ms/token, 0.20% | 0.046 ms/token, 0.18% | 1.109x | Effectively equivalent secondary cost. |
| Argmax/sampling | Deterministic `argmax_bf16_single_launch_kernel` with strict-greater, lowest-index tie behavior. | vLLM logits selection and sampling path. | 0.006 ms/token, 0.02% | 0.007 ms/token, 0.03% | 0.857x | Negligible in both implementations. |
| Other | Residual CUDA kernels, framework/runtime work, and symbols not assigned to a named ApxInf module. | Residual CUDA kernels and framework/runtime work not assigned to a named vLLM module. | 0.036 ms/token, 0.14% | 0.144 ms/token, 0.57% | 0.250x | Small residual bucket; no optimization priority. |

### Per-module conclusions

1. **Quantized W4 projection is the shared dominant cost.** Combined W4 work is 20.307 ms/token for ApxInf versus 20.950 ms/token for vLLM, consuming 80.29% and 82.03% of the normalized buckets respectively.
2. **ApxInf's distinctive penalty is projection ownership.** ApxInf spends an additional 5.764 ms/token, 22.79%, in custom/raw W4 kernels that have no separate vLLM counterpart.
3. **Attention is the largest ApxInf-specific latency gap.** ApxInf's 2.290 ms/token is 11.744x the vLLM bucket, despite the exact split FlashAttention implementation.
4. **RMSNorm is the next ApxInf-specific gap.** vLLM's 0.333 ms/token indicates more normalization/epilogue fusion than ApxInf's 1.148 ms/token standalone path.
5. **GDN is secondary.** The vectorized packed path remains 1.830x the vLLM recurrent bucket, but only contributes 0.827 ms/token to the normalized ApxInf total.
6. **ApxInf wins on the LM head and small elementwise buckets.** Those savings do not compensate for the projection ownership and attention costs.

The module comparison therefore prioritizes Marlin-compatible execution for all raw W4 projections, followed by reducing attention scheduling/fusion overhead. LM head, KV cache, argmax, and residual small kernels are not the remaining throughput blockers.
### NVTX stage breakdown

The `nvtx_gpu_proj_sum` report measured these static stage markers:

| Stage | Range instances | Total projected GPU time | Total range time |
|---|---:|---:|---:|
| `Qwen/MLP` | 517 | 509.053 ms | 289.219 ms |
| `Qwen/GDN` | 389 | 426.317 ms | 37.750 ms |
| `Qwen/attention` | 421 | 412.861 ms | 273.715 ms |
| `Qwen/LM head` | 25 | 19.664 ms | 2.829 ms |

These ranges include mixed startup/prefill/decode activity and overlapping graph-child work, so they are attribution evidence rather than per-token wall timing.

### Memory and utilization evidence

The full Nsight memory summary recorded 1,774 host-to-device memcpy operations totaling 2.017 s of GPU operation time and 3,136 CUDA memset operations totaling 6.150 ms. These are mixed-workload totals dominated by model loading and prompt prefill; no decode-only transfer split is claimed from this aggregate.

Canonical NVML samples reported:

| Workload | GPU util mean/max | Memory-controller util mean/max | VRAM peak | Power mean/max | Temperature max |
|---|---:|---:|---:|---:|---:|
| 1K | 25% / 100% | 15.08% / 78% | 22136 MiB | 95.91 / 365.11 W | 63 C |
| 8K | 25% / 100% | 10.13% / 63% | 22136 MiB | 94.50 / 371.42 W | 66 C |
| 32,640 context | 25% / 100% | 8.07% / 45% | 22136 MiB | 99.04 / 403.79 W | 71 C |

These sampled utilization values are indicators, not achieved HBM GB/s. Hardware-counter metrics needed for a defensible achieved-bandwidth calculation were unavailable.

## Verification and reproduction

Official repository check:

```bash
python3 benchmarks/qwen38_4090/evaluation/test.py check
```

Result: `assignment checks passed`, including `cargo check --workspace --locked` and evaluator CLI checks.

Build:

```bash
RUSTFLAGS='-C link-arg=-fuse-ld=gold' APXINF_CUDA_ARCH=sm_89 \
  cargo build --release --features cuda -p apxinf --bin apxinf -j 40
```

Canonical evaluation:

```bash
CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  ./target/release/apxinf serve --model ../model/qwen \
  --host 127.0.0.1 --port 8072

python3 benchmarks/qwen38_4090/evaluation/run_evaluation.py \
  --dataset benchmarks/qwen38_4090/evaluation/.cache/public \
  --context-dataset benchmarks/qwen38_4090/evaluation/.cache/context-iter3 \
  --model-dir ../model/qwen --base-url http://127.0.0.1:8072 \
  --implementation-name apxinf-iter30-clean \
  --implementation-revision f4793ee6d7782c61a55fb2db95cc52d438b5d473 \
  --backend apxinf --profile public_calibration \
  --trajectory-reference /tmp/iter25-trajectory-reference.json \
  --run-context --run-id iterate30-clean \
  --output-dir benchmarks/qwen38_4090/evaluation/runs --timeout 2400
```

Profile command:

```bash
/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys profile \
  --trace=cuda,nvtx,osrt --sample=none --cuda-graph-trace=node \
  --force-overwrite=true --output=/tmp/iterate30-final \
  ./target/release/apxinf serve --model ../model/qwen \
  --host 127.0.0.1 --port 8071

/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys stats \
  --force-export=true \
  --report nvtx_gpu_proj_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
  --format csv --output /tmp/iterate30-stats /tmp/iterate30-final.nsys-rep
```

Final evidence:

- Canonical result: `benchmarks/qwen38_4090/evaluation/runs/iterate30-clean/`.
- Iteration report: `iterate_report_30.md`.
- Local vLLM control: `benchmarks/qwen38_4090/evaluation/runs/vllm-one-gpu-control/`.

The 1.2x throughput requirement remains an explicit blocker for this iteration. Correctness, long-context aggregate evaluation, protocol behavior, reliability, reproducible profiling, and per-module cost comparison are complete.