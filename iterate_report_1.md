# Iteration Report 1 — ApxInf Qwen3.8-27B on RTX 4090

Date: 2026-08-23 · Implementation revision: `e6bfa85` · Run id: `iterate1c`

## Goal

Optimize the ApxInf Rust/CUDA executor for `cyankiwi/Qwen3.8-27B-AWQ-INT4`
on one RTX 4090 toward the reference speeds: **800 tokens/s prefill,
30 tokens/s decode**, while keeping the official evaluation green.

## Results (official evaluation, `run_evaluation.py`, public_calibration)

| Cell | TTFT | Prefill | TPOT | Decode | VRAM peak |
|---|---|---|---|---|---|
| text-perf-1024  | 0.76 s | **1340 tok/s** | 87 ms  | 11.5 tok/s | 23194 MiB |
| text-perf-2048  | 1.55 s | **1320 tok/s** | 89 ms  | 11.2 tok/s | 23194 MiB |
| text-perf-4096  | 3.18 s | **1288 tok/s** | 94 ms  | 10.7 tok/s | 23194 MiB |
| text-perf-8192  | 6.60 s | **1241 tok/s** | 103 ms | 9.7 tok/s  | 23194 MiB |
| text-perf-16384 | 14.09 s | **1163 tok/s** | 122 ms | 8.2 tok/s  | 23194 MiB |

- Correctness: **6/6 public cases**, protocol pass, trajectory **186/256**
  (unchanged from the pre-iteration baseline).
- Reliability: no fallback, no NaN, no unexpected OOM, no XID,
  request success rate 1.0, healthy after failure.
- Baseline before this iteration: 1K 1.02 s, 4K 5.72 s, 8K 16.0 s,
  16K 50.7 s prefill; TPOT ~0.21 s. **Prefill is now > 800 tok/s at every
  measured length; decode improved 2.4× (192 → 81 ms/token).**

## What was done

1. **GEMM-based prefill attention** (replaces a scalar flash kernel that
   ran at ~25 GFLOPS and dominated 1K+ prefills, growing with the visible
   prefix): per kv-group scores = q @ kᵀ via per-head cublas GEMMs with
   f32 accumulation, causal row softmax, then p @ v. Decode steps keep the
   fused flash kernel.
2. **Delta recurrence split**: a parallel q/k norm prepass kernel plus a
   slim serial sweep (128-lane blocks, pre-normalized rows), removing the
   per-token shuffles/syncs that made the old kernel DRAM-latency bound
   (16 ms → ~5.6 ms per call at seq=2048).
3. **Tensor-core decode GEMM** (m16n8k16 bf16 MMA): dequantizes 128-column
   weight tiles into shared memory with coalesced loads and hoisted
   scale/zero-points; the single activation row sits in A's row 0. Decode
   dropped 192 → 81 ms/token.
4. **Fixed cublas tensor-op algo (113)** instead of the per-call heuristic.
5. Numerics locked to the reference: f32 attention scores/weights,
   zeroed post-causal columns, exact bf16 v values, precise `expf` (the
   fast-math variant cost 35 trajectory tokens).

## Obstacles found (and fixed)

- `cublasGemmStridedBatchedEx` computes only batch 0 on this driver
  (580.82.07, cublas 12.x); `cublasGemmBatchedEx` rejects opA=T and some
  asymmetric shapes. Workaround: plain per-head `cublasGemmEx` loops with
  explicit leading dimensions.
- The q buffer is `[seq, heads, dim]`; heads must be addressed as
  col-major slabs with ld = heads×dim.
- The softmax warp merge double-counted its own warp; shuffle offsets
  >16 are invalid across warps.
- In-place softmax leaves raw scores past the causal boundary; they must
  be zeroed (the p @ v GEMM sums all `visible` columns).
- A concurrent session shares this worktree and reverted uncommitted
  changes twice; all work is now committed incrementally.

## Remaining gap

Decode is 81 ms/token (12.3 tok/s) vs the 30 tok/s target. The TC kernel
runs at 0.13–0.43 ms per GEMM against a ~0.05 ms bandwidth floor;
per-GEMM overhead and small grids (80–272 blocks) are the next targets.
The measured per-token weight traffic floor is ~15.6 ms, so 30 tok/s is
reachable with kernel tuning.
