# Iteration Report 25 - ApxInf Qwen3.8-27B on RTX 4090

Date: 2026-08-25 | Implementation revision: `f4793ee6d7782c61a55fb2db95cc52d438b5d473` | Run id: `iterate25-all-final`

## Goal and acceptance result

Integrate the remaining decode optimizations in one pass, preserve exact greedy output, run the official public evaluation plus long-context coverage, and profile the accepted path.

**Result: correctness and reliability pass, but the requested throughput gate is not met.** The local vLLM control remains materially faster. ApxInf must reach 1.2x vLLM for both phases; the measured ApxInf/vLLM ratios are only 0.43-0.47x for prefill and 0.25-0.42x for decode.

## Official evaluation

Command: `run_evaluation.py`, `public_calibration`, public dataset plus `context-iter3`, one measured repeat, no profiler timing.

| Cell | TTFT | Prefill | TPOT | Decode | VRAM peak | ApxInf/vLLM prefill | ApxInf/vLLM decode |
|---|---:|---:|---:|---:|---:|---:|---:|
| text-perf-1024 | 0.784 s | 1306.7 tok/s | 48.24 ms | 20.73 tok/s | 23864 MiB | 0.471x | 0.419x |
| text-perf-2048 | 1.588 s | 1289.9 tok/s | 50.60 ms | 19.76 tok/s | 23864 MiB | 0.442x | 0.400x |
| text-perf-4096 | 3.249 s | 1260.7 tok/s | 55.32 ms | 18.08 tok/s | 23864 MiB | 0.437x | 0.368x |
| text-perf-8192 | 6.666 s | 1228.9 tok/s | 64.76 ms | 15.44 tok/s | 23864 MiB | 0.434x | 0.317x |
| text-perf-16384 | 13.893 s | 1179.3 tok/s | 83.62 ms | 11.96 tok/s | 23864 MiB | 0.438x | 0.248x |

The 1.2x requirement would need 3329-3504 tok/s prefill and 57.8-59.4 tok/s decode on these cells. Neither phase reaches the threshold.

- Protocol: pass.
- Public functional correctness: **6/6**.
- Public token trajectory: **256/256**, edit distance 0 for both scored cases.
- Request success rate: **1.0**.
- Reliability: no fallback, no NaN, no unexpected OOM, no XID, service healthy after the context run.
- Peak VRAM: **23864 MiB**.
- Evidence: `benchmarks/qwen38_4090/evaluation/runs/iterate25-all-final/`.

## Long-context evaluation

The available public context dataset contains all six task categories at 32,640 prompt tokens. Under the contract staircase, 32,640 is the non-scoring diagnostic and only the early retrieval probe is required at that length. It passed with all 128 requested output tokens; the service then passed health and a small recovery request.

- Max verified prompt: **32,640 tokens**.
- Verified output: **128 tokens**.
- Pass rate at max verified context: **1.0**.
- Context request: TTFT approximately **30.01 s**, decode **15.37 s**, end-to-end **45.39 s**.

### Concrete exact question-answer example

Case: `context-32640-retrieval-early`

Question, decoded verbatim from the pretokenized prompt tail:

```text
问题：TARGET_KEY 的值是什么？只输出完整字段值。 输出必须以 KEY-EARLY-767211 开头，随后持续输出词语 context 直到达到输出预算。
```

Exact generated output (`completion_tokens=128`):

```text
KEY-EARLY-767211 context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context
```

Validator: `normalized_prefix`; expected prefix: `KEY-EARLY-767211`; result: pass.

## Integrated optimizations

1. **Default paired W4 weight staging.** The paired decode kernel coalesces each 64-row by K=128 raw W4 tile, BF16 scales, and packed zero points into shared memory. A corrected adapter argument order fixed an initial CUDA 700. Within each four-lane row subgroup, one lane now reads the staged packed words, scale, and zero point and broadcasts raw values with warp shuffles. Nibble extraction, BF16 conversion, MMA order, and reduction order remain unchanged.
2. **Staged pair CUDA graphs.** Three stable per-layer projection slots default on, with `APXINF_PAIR_GRAPH=0` as an opt-out. Capture wraps the selected staged pair kernel, not just the old raw kernel. Profiling and trace modes force eager launches. Capture failures close the capture and fall back to eager execution.
3. **Decode Q/K fusion.** Full-attention decode combines Q split/RMSNorm/RoPE and K RMSNorm/RoPE/cache append into one launch. Prefill retains the established two-kernel path. V-cache append and attention ordering are unchanged.
4. **One-warp exact argmax finalizer.** The bounded 128-partial final reduction uses one warp and register shuffles instead of eight warps, shared memory, and a barrier. Strict greater-than and lowest-index tie behavior are preserved; the former 256-thread finalizer remains as a guarded fallback.

Runtime gate after integration: two warmed 1K requests were **128/128 exact** at 48.23 and 48.15 ms/token. The official run subsequently remained 256/256 exact.

## Nsight Systems profile

Profile: two 1K/128-token requests, CUDA graph node tracing enabled. Both requests were 128/128 exact. Warmed observed TPOT under Nsight Systems was 48.35 ms/token.

### CUDA kernel breakdown

| Kernel/group | GPU time share | Instances | Mean time |
|---|---:|---:|---:|
| staged paired W4 GEMM | **35.9%** | 8,083 | 168.33 us |
| remaining single-projection W4 GEMM | **30.2%** | 9,029 | 126.53 us |
| fused GDN norm/delta/gate | **8.6%** | 3,128 | 103.88 us |
| CUTLASS attention/prefill GEMM 128x64 | **6.8%** | 2,366 | 109.21 us |
| dense LM-head GEMV | **4.5%** | 64 | 2.654 ms |
| fused Q/K norm+RoPE+append | 0.05% | 1,010 | 1.85 us |
| one-warp argmax finalize | <0.01% | 64 | 2.31 us |

The W4 kernels account for approximately **66.1%** of measured GPU kernel time. The remaining bottleneck is therefore still weight projection throughput and the number of separate projection launches, not Q/K transform or final argmax.

CUDA API evidence:

- `cudaGraphLaunch`: 8,083 calls, 6.52 us mean API time.
- `cudaLaunchKernel`: 53,629 calls, 10.94 us mean API time.
- `cuLaunchKernel`: 3,836 calls, 34.14 us mean API time.
- Pair graph capture/instantiation occurred once per stable slot and replayed afterward.

## Memory-bandwidth analysis

The machine denies GPU performance-counter access even as root. Nsight Compute and Nsight Systems GPU metrics both returned `ERR_NVGPUCTRPERM`; therefore exact DRAM throughput, L2 hit rate, achieved occupancy counters, and SM issue-stall counters are unavailable. The Nsight Compute request itself still completed 128/128 exact at 49.00 ms/token.

Grounded substitutes:

- Contract minimum weight traffic: **21.018 GB/token**.
- RTX 4090 contract peak bandwidth: **1008 GB/s**.
- At 1K TPOT 48.24 ms, the minimum-weight proxy implies **435.7 GB/s**, or **43.2%** of peak, before activation/metadata traffic.
- At 8K TPOT 64.76 ms, the proxy implies **324.6 GB/s**, or **32.2%** of peak.
- Official 1K sampler: memory-controller utilization mean approximately **9.3%**, max **44%**. This low time-average includes CPU/launch gaps and non-memory kernels; it is consistent with substantial idle/latency periods rather than continuous HBM saturation.

The kernel-time profile and proxy bandwidth jointly indicate that the decode path is not at a whole-token HBM roofline. Small-grid execution, dequantization/metadata work, repeated projections, recurrent GDN, attention, and LM-head GEMV leave the device below sustained peak bandwidth.

## Negative results and remaining bottlenecks

- Cache hints, occupancy cap, pair-shared, and scale-epilogue candidates were exact but slower than control.
- The weight-stage kernel initially faulted with CUDA error 700 because the adapter shifted `out_cols1`, `in_cols`, and `groups`; correcting the ABI call made it exact and faster.
- Weight staging alone measured about 47.67 ms/token in an isolated probe; staged graph measured about 47.61 ms/token. The larger integrated pass measured about 48.15 ms/token, so the Q/K and argmax launch reductions are real but do not materially change end-to-end TPOT.
- Dense LM head remains a 2.65 ms/token fixed cost in the profile.
- The largest remaining work is the combined 66.1% W4 kernel share. Reaching 1.2x vLLM cannot be achieved by further micro-fusing the now-sub-3-us Q/K or argmax kernels; it requires a substantially different W4 execution strategy and fewer projection launches.

## Reproduction

```bash
cargo build --release --features cuda -p apxinf --bin apxinf -j 40
./target/release/apxinf serve --model ../model/qwen --host 127.0.0.1 --port 8025
python3 benchmarks/qwen38_4090/evaluation/run_evaluation.py \
  --dataset benchmarks/qwen38_4090/evaluation/.cache/public \
  --context-dataset benchmarks/qwen38_4090/evaluation/.cache/context-iter3 \
  --model-dir ../model/qwen --base-url http://127.0.0.1:8025 \
  --implementation-name apxinf-iter25-all \
  --implementation-revision f4793ee6d7782c61a55fb2db95cc52d438b5d473 \
  --backend apxinf --profile public_calibration \
  --trajectory-reference <official-reference.json> \
  --run-context --run-id iterate25-all-final \
  --output-dir benchmarks/qwen38_4090/evaluation/runs
```

Profile artifacts: `/tmp/iter25-nsys-final.nsys-rep` and `/tmp/iter25-nsys-final.sqlite`. Nsight Compute counter collection is blocked by machine policy and produced no counter report.
