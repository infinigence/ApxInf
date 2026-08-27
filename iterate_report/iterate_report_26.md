# Iteration Report 26 - All-at-Once Decode Topology Pass

Date: 2026-08-25 | Implementation revision: `f4793ee6d7782c61a55fb2db95cc52d438b5d473` | Run id: `iterate26-all-final`

## Result

This pass integrated projection topology, recurrent fusion, broader decode graphs, final selection, and full-row prefill experiments at once. Correctness and reliability remain complete. Decode improves 1.6-2.8% over iteration 25; prefill is unchanged after rejecting the slower full-row WMMA candidate.

**The required 1.2x vLLM throughput gate is still not met.** ApxInf is 0.43-0.47x vLLM prefill and 0.25-0.43x vLLM decode, not 1.2x.

## Official evaluation

`run_evaluation.py`, `public_calibration`, public dataset plus 32,640-token context data, no-profiler timing.

| Cell | TTFT | Prefill | TPOT | Decode | TPOT gain vs iter25 | VRAM | Prefill/vLLM | Decode/vLLM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| text-perf-1024 | 0.784 s | 1306.5 tok/s | 46.91 ms | 21.32 tok/s | 2.77% | 23858 MiB | 0.471x | 0.431x |
| text-perf-2048 | 1.587 s | 1290.3 tok/s | 49.27 ms | 20.30 tok/s | 2.65% | 23858 MiB | 0.442x | 0.411x |
| text-perf-4096 | 3.248 s | 1261.0 tok/s | 53.97 ms | 18.53 tok/s | 2.45% | 23858 MiB | 0.437x | 0.377x |
| text-perf-8192 | 6.665 s | 1229.2 tok/s | 63.41 ms | 15.77 tok/s | 2.08% | 23858 MiB | 0.434x | 0.323x |
| text-perf-16384 | 13.886 s | 1179.9 tok/s | 82.28 ms | 12.15 tok/s | 1.60% | 23858 MiB | 0.438x | 0.252x |

The 1K requirement is at least 3329.1 tok/s prefill and 59.35 tok/s decode. Measured values are 1306.5 and 21.32 tok/s.

- Protocol: pass.
- Public functional: **6/6**.
- Public trajectory: **256/256**, zero edit distance on both scored cases.
- Request success rate: **1.0**.
- No fallback, NaN, unexpected OOM, or XID; service healthy after context run.
- Raw evidence SHA-256: `3f0bd29d2c9f8de9d9ccdea5f38c77d2f636bd8521e3230d7b8c7b459ebe7571`.
- Artifacts: `benchmarks/qwen38_4090/evaluation/runs/iterate26-all-final/`.

## Integrated changes

1. **Staged multi-projection W4.** Linear-attention QKV/Z/A/B now share one staged decode launch instead of four independent projections. Full-attention Q/K/V now share one launch instead of Q plus a K/V pair. Every projection preserves raw compressed-tensors indices, group-32 asymmetric dequantization, BF16 conversion, K-ordered MMA accumulation, and independent BF16 output. The 48-row A/B tails use predicated loads/stores. `APXINF_W4_MULTI=0` restores fallback routing.
2. **Packed decode GDN.** Causal convolution, Q/K normalization, delta recurrence, and gated RMSNorm are one guarded kernel for the model's K=V=128, conv-width-4 geometry. The convolution accumulation order was aligned to the established zero/history/live order. Relative to the previous default, two launches become one; relative to the fully eager sequence, four become one. `APXINF_GDN_FUSED=0` restores fallback.
3. **Linear-segment CUDA graphs.** Each maximal contiguous run of linear-attention layers is captured as one graph. Embedding, full-attention layers, final head, and host wait remain eager. Pair/GDN subgraphs are disabled during segment capture to prevent nesting. Full-attention cannot join the graph because RoPE/cache/flash kernels capture `start_pos` and visible length by value. `APXINF_DECODE_LAYER_GRAPH=0` disables segment graphs.
4. **Single-launch exact argmax.** One grid-wide last-block protocol replaces partial plus finalize launches. `__threadfence` and an arrival counter establish visibility; strict greater-than and lowest-index ties are unchanged. The counter resets inside the stream-ordered kernel. `APXINF_ARGMAX_SINGLE_LAUNCH=0` restores the two-launch path.
5. **LM head unchanged.** cuBLAS GEMM_EX plus BF16 materialized logits remains required for proven arbitrary-activation equivalence. The head is still about 2.65 ms/token.

## Rejected full-row prefill candidate

`APXINF_W4_PREFILL_NATIVE=1` was extended from 256 to 2048 rows, allowing official 512-row chunks into the direct raw-W4 WMMA path. The all-enabled exact probe stayed 128/128 but regressed warmed 1K TTFT from approximately 0.784 s to **1.586 s**. The cause is repeated packed-weight/metadata streaming for every 32-row activation tile. The gate remains opt-in and is not enabled in the accepted run.

## Long-context evaluation

The 32,640-token diagnostic passed with all 128 requested output tokens, followed by successful health and recovery requests.

- Prompt: **32,640 tokens**.
- TTFT: **30.020 s**.
- TPOT: **119.73 ms**.
- End-to-end: **45.225 s**.
- Validator: `normalized_prefix`, pass.

### Exact question-answer example

Question decoded from the pretokenized prompt tail:

```text
问题：TARGET_KEY 的值是什么？只输出完整字段值。 输出必须以 KEY-EARLY-767211 开头，随后持续输出词语 context 直到达到输出预算。
```

Exact 128-token output:

```text
KEY-EARLY-767211 context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context
```

## Nsight Systems profile

Two exact 1K/128-token requests were captured with CUDA graph node tracing. Warmed profiler TPOT was 47.01 ms/token.

### Kernel breakdown

| Kernel/group | GPU share | Instances | Mean |
|---|---:|---:|---:|
| staged paired W4 | 35.9% | 9,210 | 179.90 us |
| remaining single W4 | 30.0% | 10,442 | 132.46 us |
| fused GDN fallback observed during prefill/capture | 5.9% | 96 | 2.852 ms |
| CUTLASS attention/prefill GEMM | 5.6% | 2,366 | 109.22 us |
| dense LM-head GEMV | 4.8% | 83 | 2.653 ms |
| flash attention | 4.4% | 1,315 | 156.00 us |
| staged multi W4 | 3.3% | 1,315 | 115.90 us |
| packed decode GDN | 1.5% | 3,948 | 17.32 us |
| single-launch argmax | <0.01% | 83 | 4.01 us |

Paired and single W4 still account for **65.9%** of GPU kernel time. Staged multi adds 3.3%; projection execution remains the dominant limit.

### Host launch reduction

Normalized across the two captures:

- `cudaLaunchKernel`: approximately **838 -> 294 calls/token**, a **64.9% reduction**.
- `cudaGraphLaunch`: approximately **126 -> 32 calls/token**, a **74.9% reduction**.
- Final TPOT improves only 2.77% at 1K because kernel execution, especially W4, dominates after launch compression.

Performance counters remain unavailable: both Nsight Compute and Nsight Systems GPU metrics return `ERR_NVGPUCTRPERM`. No DRAM/L2/occupancy values are fabricated.

## Native Marlin audit

The installed vLLM scheme is mathematically compatible with unsigned asymmetric W4, BF16 activation/output, group size 32, and SM89. Native integration is nevertheless blocked by local artifacts and memory:

- The installed wheel has no `csrc` Marlin sources or generated SM89 specializations.
- `_C_stable_libtorch.abi3.so` exports a C++ `marlin::marlin_mm`, not a stable C ABI, and depends on `libtorch`, `libtorch_cpu`, `libtorch_cuda`, and CUDA 13. Calling it would be a vLLM/libtorch fallback, which is prohibited.
- Upstream Marlin requires generated `kernel_selector.h` and generated kernel translation units that are absent from the wheel/source cache.
- One raw W4 representation is **13.081 GiB**. Marlin repacking is a permutation of the same payload size for this checkpoint. Retaining raw weights for prefill plus Marlin weights for decode would project the measured 22.86 GiB footprint to about **35.94 GiB**, 11.94 GiB over the 24 GiB card.
- A compliant implementation therefore requires vendored generated SM89 sources, a maintained raw C ABI, and a Marlin-layout prefill path that allows raw device weights to be released. None exists locally.

No partial Marlin adapter or fallback was added.

## Remaining bottleneck

The all-at-once pass removed most host launch overhead and reduced GDN/selection costs. The remaining performance gap is not addressable by further sub-5-us launch fusions: W4 projections still occupy roughly two-thirds of GPU time, and the exact raw-layout kernel achieves far below the throughput needed for 1.2x vLLM. A new native W4 execution core plus a unified layout for both decode and prefill is required.

## Reproduction

```bash
cargo build --release --features cuda -p apxinf --bin apxinf -j 40
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  ./target/release/apxinf serve --model ../model/qwen --host 127.0.0.1 --port 8028
python3 benchmarks/qwen38_4090/evaluation/run_evaluation.py \
  --dataset benchmarks/qwen38_4090/evaluation/.cache/public \
  --context-dataset benchmarks/qwen38_4090/evaluation/.cache/context-iter3 \
  --model-dir ../model/qwen --base-url http://127.0.0.1:8028 \
  --implementation-name apxinf-iter26-all \
  --implementation-revision f4793ee6d7782c61a55fb2db95cc52d438b5d473 \
  --backend apxinf --profile public_calibration \
  --trajectory-reference <official-reference.json> \
  --run-context --run-id iterate26-all-final \
  --output-dir benchmarks/qwen38_4090/evaluation/runs
```

Profile artifacts: `/tmp/iter26-nsys.nsys-rep`, `/tmp/iter26-nsys.sqlite`.
