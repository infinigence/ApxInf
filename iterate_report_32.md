# Iteration Report 32 - Definitive Coordinated Optimization Result

Date: 2026-08-26  
Implementation revision: `f4793ee6d7782c61a55fb2db95cc52d438b5d473` plus the measured worktree  
Run id: `iterate32-final`

## Verdict

This iteration ran the remaining optimization tracks together: raw W4 decode scheduling, direct W4 prefill candidates, long-context attention occupancy, prefill GDN routing, and launch reduction. Only exact candidates were considered for promotion.

**Correctness: pass. The requested 1.2x vLLM throughput target: not met.**

Promoted defaults:

- `APXINF_W4_PAIR_PREFETCH` defaults on; rollback with `APXINF_W4_PAIR_PREFETCH=0`.
- `APXINF_FLASH_SPLIT_256_1W` defaults on; rollback with `APXINF_FLASH_SPLIT_256_1W=0`.
- Existing exact fused GDN, linear-attention segment graphs, persistent arena, mapped decode control, event-scoped argmax, and exact W4 LM head remain active.

Rejected or not promoted:

- Native raw W4 prefill changed the trajectory.
- Packed and fast direct W4 prefill candidates failed when extended to the real 512-row chunk.
- Pair occupancy, register, vector-MMA, and other pair schedules were exact but slower or neutral.
- Position-unsafe full-attention CUDA graphs were removed because RoPE/KV-cache launches capture position-dependent values by value.

## Coordinated candidate matrix

### Prefill

| Candidate | Exactness | Decision |
|---|---|---|
| Native raw W4 prefill | Failed 1K trajectory | Rejected |
| Packed W4 prefill, 256-row path | Exact only in isolated short probes | Remains opt-in; does not cover the 512-row production chunk |
| Fast W4 prefill, 256-row path | Exact only in isolated short probes | Remains opt-in; does not cover the 512-row production chunk |
| Packed W4 prefill extended to 512 rows | Failed 1K trajectory | Rejected and reverted |
| Fast W4 prefill extended to 512 rows | Failed 1K trajectory | Rejected and reverted |

The production prefill route remains the established exact path. Prefill is still the largest target gap and requires a new exact W4 execution core rather than a row-cap extension of the rejected kernels.

### Decode W4 scheduling

| Candidate | 1K result | 8K result | Decision |
|---|---:|---:|---|
| Pair occupancy | Exact, approximately 25.93 ms/token | Not promoted | Rejected |
| Pair register-bound | Exact, approximately 26.05 ms/token | Not promoted | Rejected |
| Pair prefetch | Exact, approximately 25.07 ms/token | Exact, approximately 39.57 ms/token | Promoted |
| Vector MMA | Exact, approximately 28.45 ms/token | Not promoted | Rejected |

Pair-prefetch uses double-buffered activation/metadata staging with `cp.async` while preserving W4 group-32 indexing, BF16 conversion boundaries, MMA order, accumulators, and output ordering.

### Attention occupancy

The exact split-attention partial buffer and merge order were retained. Only CTA decomposition changed.

| Candidate | 1K probe | 8K probe | Decision |
|---|---:|---:|---|
| Existing split path | approximately 24.89 ms/token | approximately 39.53 ms/token | Control |
| Two-warp CTAs | approximately 24.83 ms/token | approximately 39.30 ms/token | Exact, not promoted |
| One-warp CTAs | approximately 24.48 ms/token | approximately 35.95 ms/token | Promoted |

One-warp attention maps one CTA to each established timestep stream and uses the existing partial layout and merge kernel. No numerical reduction order changed.

### Launch and graph optimization

Existing linear-attention segment CUDA graphs remain enabled. A new full-attention graph path was not promoted: position-dependent RoPE/KV-cache launches capture `start_pos` and visible state by value, so replay would be unsafe without a device-parameter graph ABI. The incomplete experiment was removed before final rebuild.

## Definitive canonical evaluation

Evaluator: `benchmarks/qwen38_4090/evaluation/run_evaluation.py`  
Profile: `public_calibration`  
Warmups: 0  
Measured repeats: 1  
Dataset: `benchmarks/qwen38_4090/evaluation/.cache/public`  
Context dataset: `benchmarks/qwen38_4090/evaluation/.cache/context-iter3`

| Cell | TTFT | Prefill | TPOT | Decode | VRAM peak |
|---|---:|---:|---:|---:|---:|
| text-perf-1024 | 0.8920 s | 1147.5 tok/s | 24.450 ms | **40.90 tok/s** | 22134 MiB |
| text-perf-2048 | 1.8065 s | 1133.1 tok/s | 26.085 ms | 38.34 tok/s | 22134 MiB |
| text-perf-4096 | 3.6894 s | 1110.5 tok/s | 29.356 ms | 34.06 tok/s | 22134 MiB |
| text-perf-8192 | 7.5500 s | 1085.0 tok/s | 35.880 ms | **27.87 tok/s** | 22134 MiB |
| text-perf-16384 | 15.6658 s | 1045.9 tok/s | 48.912 ms | 20.45 tok/s | 22134 MiB |

Correctness and reliability:

- Public functional cases: **6/6**.
- Public token trajectory: **256/256**.
- Protocol: pass.
- Request success rate: **1.0**.
- No fallback, NaN, unexpected OOM, or XID.
- Service healthy after context and failure checks.
- Complete 128-token output budget in every performance cell.
- Peak VRAM: **22134 MiB**.
- Model revision: `63768c10df38c0395e12ef49edac1bd539eaeeea`.

Definitive artifacts:

- Submission: `benchmarks/qwen38_4090/evaluation/runs/iterate32-final/submission.json`.
- Raw evaluator: `benchmarks/qwen38_4090/evaluation/runs/iterate32-final/raw.jsonl`.
- Environment: `benchmarks/qwen38_4090/evaluation/runs/iterate32-final/environment.json`.
- Submission SHA-256: `4d16b054763d9903173ca6c813d5d0c2bf4495c1bfe81211926c71c3548b2f81`.
- Raw JSONL SHA-256: `40cdc61885f56961a96cc93f3786e886266d7b779fd019c957016419b83f5144`.

### Long-context aggregate

The 32,640-token context diagnostic passed. No concrete prompt, question, or generated-answer testcase is reproduced in this report.

- Prompt length: **32,640 tokens**.
- TTFT: **33.5629 s**.
- TPOT: **74.760 ms/token**.
- E2E: **43.0575 s**.
- Output budget: **128 tokens completed**.
- Peak VRAM: **22134 MiB**.
- Output SHA-256: `5a66bd3bce8680b1baa4b355574be11cf3e20ff95a26b0d18f433853b7f4b03b`.
- Validator: normalized-prefix pass.
- Service healthy afterward.

## vLLM target comparison

Control: `benchmarks/qwen38_4090/evaluation/runs/vllm-one-gpu-control/submission.json`.

| Metric | ApxInf | vLLM | ApxInf/vLLM throughput | Required for 1.2x |
|---|---:|---:|---:|---:|
| 1K prefill | 1147.5 tok/s | 2774.2 tok/s | 0.414x | 3329.1 tok/s |
| 8K prefill | 1085.0 tok/s | 2829.3 tok/s | 0.383x | 3395.2 tok/s |
| 1K decode | 40.90 tok/s | 49.45 tok/s | 0.827x | 59.35 tok/s |
| 8K decode | 27.87 tok/s | 48.75 tok/s | 0.572x | 58.50 tok/s |

The final public-calibration score against the local vLLM control is provisional: **70.4484 leaderboard points**, **56.3587 automated course points**. The 1.2x target remains unmet in all four requested performance comparisons.

## Final profile and module costs

Nsight Systems profile from the coordinated promoted path:

- `/tmp/iterate31-combined.nsys-rep`, SHA-256 `986a3aa87c518c1d269607bf94f63ea5e0d1ed758f1847bc0856451bc65f9cbf`.
- `/tmp/iterate31-combined.sqlite`, SHA-256 `040e6418b2d585ff1576ba14c71a399457f72c9ab04c9d61c8e75edcf56ca37d`.

Dominant kernel families:

| Kernel family | Instances | GPU time | Share |
|---|---:|---:|---:|
| Marlin M=1 main tile | 6639 | 348.027 ms | 23.6% |
| CUTLASS BF16 128x64 GEMM | 2366 | 260.962 ms | 17.7% |
| Prefill delta recurrence | 96 | 256.519 ms | 17.4% |
| Marlin inverse dequant | 606 | 171.141 ms | 11.6% |
| Pair-prefetch raw W4 projection | 1173 | 125.802 ms | 8.5% |
| CUTLASS BF16 128x128 GEMM | 640 | 103.723 ms | 7.0% |
| One-warp attention partial | 390 | 42.432 ms | 2.9% |
| RMSNorm | 3406 | 30.906 ms | 2.1% |
| Conv/SiLU | 96 | 22.437 ms | 1.5% |
| Packed GDN recurrence | 1173 | 21.446 ms | 1.5% |

The remaining bottleneck is quantized projection execution and prefill recurrence. Attention is no longer the dominant ApxInf-specific cost after the one-warp decomposition.

### Per-module normalized comparison

The following decode-normalized values use the existing matched ApxInf/vLLM profiler comparison: 25 ApxInf intervals and 67 vLLM intervals. They are module-level profiler costs, not replacements for client-observed TPOT.

| Module | ApxInf implementation | vLLM implementation | ApxInf cost | vLLM cost | ApxInf/vLLM |
|---|---|---|---:|---:|---:|
| Combined W4 projection | Marlin plus pair-prefetch/raw W4 projection | Marlin W4A16 projection | 20.307 ms/token, 80.29% | 20.950 ms/token, 82.03% | 0.969x |
| Attention | One-warp split FlashAttention partial plus merge/gate | FlashAttention forward with graph/fusion scheduling | 2.290 ms/token, 9.05% | 0.195 ms/token, 0.76% | 11.744x |
| RMSNorm | Standalone BF16 RMSNorm kernel | Fused/Triton RMSNorm and epilogues | 1.148 ms/token, 4.54% | 0.333 ms/token, 1.30% | 3.448x |
| GDN/recurrent | Packed delta-gated kernel with vectorized state transfers | Packed recurrent GDN scheduling | 0.827 ms/token, 3.27% | 0.452 ms/token, 1.77% | 1.830x |
| LM head | Exact W4A16 head plus deterministic argmax | Dense BF16 LM head plus sampling | 0.429 ms/token, 1.69% | 2.754 ms/token, 10.79% | 0.156x |
| Elementwise/epilogue | Custom CUDA add, SiLU, scaling, gating | Fused Triton elementwise/epilogue | 0.201 ms/token, 0.79% | 0.591 ms/token, 2.31% | 0.340x |
| KV cache | Custom BF16 append and position-indexed writes | Paged/graph-aware KV updates | 0.051 ms/token, 0.20% | 0.046 ms/token, 0.18% | 1.109x |
| Argmax/sampling | Single-launch deterministic GPU argmax | vLLM logits selection/sampling | 0.006 ms/token, 0.02% | 0.007 ms/token, 0.03% | 0.857x |

The key conclusion is unchanged: the remaining gap requires a new exact native W4 projection core and a substantially faster exact prefill recurrence path. Further small attention or launch changes are insufficient.

## Verification

```bash
python3 benchmarks/qwen38_4090/evaluation/test.py check
```

Result: `assignment checks passed`.

The release build completed successfully with `APXINF_CUDA_ARCH=sm_89`. Exact 1K/8K smoke gates matched the frozen SHA values after the unsafe full-attention graph experiment was removed. No service or profiler process remains running.

## Conclusion

The coordinated optimization pass is complete. It promoted two exact improvements and rejected unsafe or non-exact paths. Decode improved to approximately 40.9 tok/s at 1K and 27.9 tok/s at 8K, but the 1.2x vLLM target remains open because quantized projection execution and prefill recurrence dominate the remaining cost. Reaching the target requires a new exact W4 execution architecture, not additional micro-optimizations.
