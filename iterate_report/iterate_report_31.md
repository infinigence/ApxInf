# Iteration Report 31 - Coordinated Prefill, Decode, Attention, and Launch Optimization

Status: **final rebuilt verification completed**.

Date: 2026-08-26  
Implementation revision label: `f4793ee6d7782c61a55fb2db95cc52d438b5d473` plus the measured worktree  
Run id: `iterate31-final`

## Objective and verdict

This iteration ran the remaining optimization tracks together: W4 prefill execution, raw W4 decode scheduling, long-context attention occupancy, and launch/epilogue overhead. Every candidate was tested against the frozen 1K and 8K token trajectories before promotion.

**Correctness verdict: pass. Performance verdict: the 1.2x vLLM target remains unmet.** The final combined default reaches 40.73 decode tok/s at 1K and 27.81 decode tok/s at 8K. The local vLLM control reaches 49.45 and 48.75 tok/s, so ApxInf reaches 0.824x and 0.570x of vLLM throughput. The 1.2x target would require 59.35 and 58.50 tok/s respectively.

The promoted changes are:

- W4 pair-prefetch decode scheduling, default-on with rollback `APXINF_W4_PAIR_PREFETCH=0`.
- One-warp-CTA split FlashAttention partial scheduling, default-on with rollback `APXINF_FLASH_SPLIT_256_1W=0`.

The prefill candidates were rejected after exactness testing. A full-attention CUDA graph experiment was also rejected because position-dependent RoPE/KV-cache launches capture `start_pos` by value and cannot be safely replayed without a graph-update or device-parameter ABI.

## Coordinated candidate matrix

### W4 prefill candidates

| Candidate | Exact 1K trajectory | Result | Reason |
|---|---|---|---|
| Native raw W4 prefill, original row cap | No | Rejected | Changed the frozen output trajectory. |
| Packed W4 prefill, 256-row cap | Yes in isolated short probe | Retained opt-in | Did not cover the official 512-row chunk; no production promotion. |
| Fast W4 prefill, 256-row cap | Yes in isolated short probe | Retained opt-in | Did not cover the official 512-row chunk; no production promotion. |
| Packed W4 prefill extended to 512 rows | No | Rejected | Failed the exact 1K trajectory after exercising the real chunk size. |
| Fast W4 prefill extended to 512 rows | No | Rejected | Failed the exact 1K trajectory after exercising the real chunk size. |
| Packed/fast prefill combined with pair-prefetch | No | Rejected | Same exactness failure; no combined promotion. |

The 512-row experiment was reverted. The established exact prefill path remains unchanged. This is the central unresolved prefill gap: ApxInf measures about 1148 prompt tok/s at 1K and 1085 prompt tok/s at 8K, while 1.2x vLLM would require about 3329 and 3395 prompt tok/s.

### Raw W4 decode scheduling candidates

All candidates below preserved the exact 1K trajectory. The first probe used the promoted pair-prefetch control; timings are client-observed TPOT and subject to normal single-request noise.

| Candidate | 1K TPOT | 8K TPOT | Decision |
|---|---:|---:|---|
| Pair occupancy | 25.930 ms | not promoted | Exact, no improvement. |
| Pair register-bound | 26.052 ms | not promoted | Exact, slower. |
| Pair prefetch | 25.067 ms | 39.571 ms | Promoted after exact 1K/8K gates. |
| Vector MMA | 28.454 ms | not promoted | Exact, slower. |
| Pair-prefetch plus all decode schedules | 24.711 ms | 39.702 ms | Exact, but not promoted as a bundle. |

The final default uses pair-prefetch only. It stages the next activation and metadata tile with `cp.async` while preserving the existing packed weights, group-32 metadata, BF16 conversion boundaries, MMA sequence, accumulator layout, and output order.

### Long-context attention candidates

The existing split attention path schedules two four-warp CTAs per head. Two new exact variants changed only CTA decomposition and retained the same eight partial streams, partial-buffer layout, online-softmax recurrence, merge order, and BF16 gate boundary.

| Candidate | 1K probe 0 | 1K probe 1 | 8K probe | Decision |
|---|---:|---:|---:|---|
| Existing split attention | 24.944 ms | 24.889 ms | 39.528 ms | Control |
| Two-warp CTAs, four CTAs/head | 24.891 ms | 24.833 ms | 39.299 ms | Exact, small gain, not final |
| One-warp CTA, eight CTAs/head | 24.534 ms | 24.479 ms | 35.945 ms | Promoted |

The one-warp path maps one CTA to each of the eight established attention timestep streams. It increases CTA count without changing the numerical reduction tree. The final combined binary passed the frozen trajectory SHA at both 1K and 8K.

### Launch and graph optimization

A full-attention per-layer CUDA graph was attempted after the validated linear-attention segment graphs. It was not promoted. Full-attention RoPE/KV-cache launches consume position-dependent values and cache state; capturing `start_pos` by value would make replay incorrect for later tokens. The incomplete experiment was removed.

Existing safe optimizations remain active:

- contiguous linear-attention segment CUDA graphs;
- persistent arena allocation;
| Cell | TTFT | Prefill | TPOT | Decode | VRAM peak |
|---|---:|---:|---:|---:|---:|
| text-perf-1024 | 0.8920 s | 1147.5 tok/s | 24.450 ms | **40.90 tok/s** | 22134 MiB |
| text-perf-2048 | 1.8065 s | 1133.1 tok/s | 26.085 ms | 38.34 tok/s | 22134 MiB |
| text-perf-4096 | 3.6894 s | 1110.5 tok/s | 29.356 ms | 34.06 tok/s | 22134 MiB |
| text-perf-8192 | 7.5500 s | 1085.0 tok/s | 35.880 ms | **27.87 tok/s** | 22134 MiB |
| text-perf-16384 | 15.6658 s | 1045.9 tok/s | 48.912 ms | 20.45 tok/s | 22134 MiB |

Definitive rebuilt artifact: `benchmarks/qwen38_4090/evaluation/runs/iterate32-final/`. Its raw evaluator SHA-256 is `40cdc61885f56961a96cc93f3786e886266d7b779fd019c957016419b83f5144`; submission SHA-256 is `4d16b054763d9903173ca6c813d5d0c2bf4495c1bfe81211926c71c3548b2f81`. It passed 6/6 public functional cases, 256/256 public trajectory tokens, request success rate 1.0, no fallback/NaN/OOM/XID, and the 32,640-token context diagnostic.
Evaluator: `benchmarks/qwen38_4090/evaluation/run_evaluation.py`  
Profile: `public_calibration`  
Warmups: 0  
Measured repeats: 1  
Dataset: `benchmarks/qwen38_4090/evaluation/.cache/public`  
Context dataset: `benchmarks/qwen38_4090/evaluation/.cache/context-iter3`

| Cell | TTFT | Prefill | TPOT | Decode | VRAM peak |
|---|---:|---:|---:|---:|---:|
| text-perf-1024 | 0.8917 s | 1148.4 tok/s | 24.550 ms | **40.73 tok/s** | 22134 MiB |
| text-perf-2048 | 1.8061 s | 1132.1 tok/s | 26.180 ms | 38.20 tok/s | 22134 MiB |
| text-perf-4096 | 3.6883 s | 1110.9 tok/s | 29.439 ms | 33.97 tok/s | 22134 MiB |
| text-perf-8192 | 7.5497 s | 1085.1 tok/s | 35.958 ms | **27.81 tok/s** | 22134 MiB |
| text-perf-16384 | 15.6699 s | 1045.4 tok/s | 49.007 ms | 20.40 tok/s | 22134 MiB |

Correctness and reliability:

- Protocol: pass.
- Public functional cases: **6/6**.
- Public trajectory: **256/256**, zero token edit distance at 1K and 8K.
- Every public performance cell completed the requested 128-token budget.
- Request success rate: **1.0**.
- No fallback, NaN, unexpected OOM, or XID.
- Service healthy after context and failure checks.
- Peak VRAM: **22134 MiB**.
- Model revision: `63768c10df38c0395e12ef49edac1bd539eaeeea`.

Final raw evaluator SHA-256: `8d1f9319e86a116c8c0caba8001b6ac75d5580e94f936143e40848a0d4834712`.

### Long-context aggregate

The 32,640-token context diagnostic passed. Per report scope, no concrete prompt, question, or generated-answer testcase is reproduced.

- Prompt length: **32,640 tokens**.
- TTFT: **33.5797 s**.
- TPOT: **74.909 ms/token** (**13.35 tok/s**).
- E2E: **43.0933 s**.
- Peak VRAM: **22134 MiB**.
- Output budget: **128 tokens completed**.
- Output SHA-256: `5a66bd3bce8680b1baa4b355574be11cf3e20ff95a26b0d18f433853b7f4b03b`.
- Validator: normalized-prefix pass.
- Service healthy afterward.

## vLLM target comparison

Control artifact: `benchmarks/qwen38_4090/evaluation/runs/vllm-one-gpu-control/submission.json`.

| Metric | ApxInf | vLLM | ApxInf/vLLM throughput | Required for 1.2x |
|---|---:|---:|---:|---:|
| 1K prefill | 1148.4 tok/s | 2774.2 tok/s | 0.414x | 3329.1 tok/s |
| 8K prefill | 1085.1 tok/s | 2829.3 tok/s | 0.384x | 3395.2 tok/s |
| 1K decode | 40.73 tok/s | 49.45 tok/s | 0.824x | 59.35 tok/s |
| 8K decode | 27.81 tok/s | 48.75 tok/s | 0.570x | 58.50 tok/s |

The final path improves decode versus iteration 30, especially at 8K, but remains below the target. Prefill remains the largest absolute gap. The official public-calibration score is provisional because it uses one measured repeat; hidden correctness and private five-repeat leaderboard measurements are unavailable locally.

## Final Nsight Systems profile

Profiler: NVIDIA Nsight Systems 2025.1.1 at `/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys`.

Artifacts:

- `/tmp/iterate31-combined.nsys-rep`, SHA-256 `986a3aa87c518c1d269607bf94f63ea5e0d1ed758f1847bc0856451bc65f9cbf`.
- `/tmp/iterate31-combined.sqlite`, SHA-256 `040e6418b2d585ff1576ba14c71a399457f72c9ab04c9d61c8e75edcf56ca37d`.
- Exported summaries: `/tmp/iterate31-stats_cuda_gpu_kern_sum.csv`, `/tmp/iterate31-stats_nvtx_gpu_proj_sum.csv`, and `/tmp/iterate31-stats_cuda_gpu_mem_time_sum.csv`.

The profiled combined path used pair-prefetch and one-warp attention. The exact 1K profiled request measured 24.751 ms/token and matched the frozen SHA.

### Kernel breakdown

| Kernel family | Instances | Total GPU time | Share |
|---|---:|---:|---:|
| Marlin M=1 main tile | 6639 | 348.027 ms | 23.6% |
| CUTLASS BF16 128x64 GEMM | 2366 | 260.962 ms | 17.7% |
| Prefill delta recurrence | 96 | 256.519 ms | 17.4% |
| Marlin inverse dequant | 606 | 171.141 ms | 11.6% |
| Pair-prefetch raw W4 projection | 1173 | 125.802 ms | 8.5% |
| CUTLASS BF16 128x128 GEMM | 640 | 103.723 ms | 7.0% |
| One-warp split FlashAttention partial | 390 | 42.432 ms | 2.9% |
| RMSNorm | 3406 | 30.906 ms | 2.1% |
| Conv/SiLU | 96 | 22.437 ms | 1.5% |
| Packed GDN recurrence | 1173 | 21.446 ms | 1.5% |

The remaining bottleneck is still quantized projection execution. Marlin, inverse dequant, pair-prefetch raw W4, and BF16 GEMM families dominate the trace. Attention improved materially through one-warp CTA decomposition, but it is no longer the main blocker. Prefill delta recurrence is also a large cost and prevents prefill from approaching the 1.2x target.

### NVTX stages

| Stage | Range instances | Total projected GPU time | Total range time |
|---|---:|---:|---:|
| `Qwen/MLP` | 518 | 529.148 ms | 300.738 ms |
| `Qwen/GDN` | 390 | 417.194 ms | 49.493 ms |
| `Qwen/attention` | 422 | 415.037 ms | 287.622 ms |
| `Qwen/LM head` | 25 | 19.678 ms | 2.978 ms |

These totals include mixed startup, prefill, graph, and decode activity. They are attribution evidence, not substitutes for canonical client TPOT.

### Memory and utilization

The full Nsight memory summary recorded 1,774 host-to-device memcpy operations totaling 5.135 s of GPU operation time and 3,136 CUDA memset operations totaling 6.692 ms. These are mixed-workload totals dominated by initialization and prompt upload; no decode-only transfer claim is made from this aggregate.

Canonical NVML sampling for the final run reported peak VRAM of 22134 MiB. Hardware-counter achieved-bandwidth metrics were unavailable in this environment, so no DRAM bandwidth percentage is fabricated.

## Reproduction

```bash
RUSTFLAGS='-C link-arg=-fuse-ld=gold' APXINF_CUDA_ARCH=sm_89 \
  cargo build --release --features cuda -p apxinf --bin apxinf -j 40

CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  ./target/release/apxinf serve --model ../model/qwen \
  --host 127.0.0.1 --port 8180

python3 benchmarks/qwen38_4090/evaluation/run_evaluation.py \
  --dataset benchmarks/qwen38_4090/evaluation/.cache/public \
  --context-dataset benchmarks/qwen38_4090/evaluation/.cache/context-iter3 \
  --model-dir ../model/qwen --base-url http://127.0.0.1:8180 \
  --implementation-name apxinf-iter31-final \
  --implementation-revision f4793ee6d7782c61a55fb2db95cc52d438b5d473 \
  --backend apxinf --profile public_calibration \
  --trajectory-reference /tmp/iter25-trajectory-reference.json \
  --run-context --run-id iterate31-final \
  --output-dir benchmarks/qwen38_4090/evaluation/runs --timeout 2400
```

Default rollback switches for this iteration:

- `APXINF_W4_PAIR_PREFETCH=0`
- `APXINF_FLASH_SPLIT_256_1W=0`
- `APXINF_FLASH_SPLIT_256_2W=1` selects the exact two-warp experimental attention decomposition.
- `APXINF_FLASH_SPLIT_256=0` disables split attention and restores the prior fused path.

## Conclusion

The coordinated optimization pass is complete and evidence-backed. It improved decode from iteration 30 to approximately 40.7 tok/s at 1K and 27.8 tok/s at 8K while preserving all public correctness gates. The 1.2x vLLM target remains unmet because W4 projection execution and prefill delta recurrence dominate the remaining cost. The next required step is a new exact native W4 execution core that removes the custom/raw projection bucket and a fused or substantially more parallel prefill recurrence path; further small launch or attention changes will not close the remaining gap.

## Rebuilt verification

After the unsafe full-attention graph experiment was removed, the release binary was rebuilt and rechecked. `python3 benchmarks/qwen38_4090/evaluation/test.py check` returned `assignment checks passed`. Exact 1K and 8K smoke gates matched the frozen output SHA values. The final canonical run was `iterate32-final` on the rebuilt binary, with no candidate environment variables set; the earlier iteration-31 measurements remain included above as the coordinated candidate matrix, while the rebuilt artifact is the definitive delivery evidence.
