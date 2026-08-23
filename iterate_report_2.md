# Iteration Report 2 — ApxInf Qwen3.8-27B on RTX 4090

Date: 2026-08-24 · Implementation revision: `db8e835` · Run id: `iterate2`

## Goal

Continue toward the reference speeds (800 tokens/s prefill — already met —
and 30 tokens/s decode) while keeping the official evaluation green.

## Results (official evaluation, `run_evaluation.py`, public_calibration)

| Cell | TTFT | Prefill | TPOT | Decode | VRAM peak |
|---|---|---|---|---|---|
| text-perf-1024  | 0.76 s | **1339 tok/s** | 67 ms  | **14.9 tok/s** | 23194 MiB |
| text-perf-2048  | 1.55 s | **1318 tok/s** | 70 ms  | **14.4 tok/s** | 23194 MiB |
| text-perf-4096  | 3.19 s | **1284 tok/s** | 74 ms  | **13.5 tok/s** | 23194 MiB |
| text-perf-8192  | 6.61 s | **1240 tok/s** | 83 ms  | **12.0 tok/s** | 23194 MiB |
| text-perf-16384 | 14.12 s | **1160 tok/s** | 102 ms | **9.8 tok/s**  | 23194 MiB |

- Correctness: **6/6 public cases**, protocol pass, trajectory **186/256**
  (unchanged — the numerics preserved).
- Reliability: all gates pass, request success rate 1.0.
- Decode improved from 87–122 ms/token (8.2–11.5 tok/s) to 67–102 ms/token
  (9.8–14.9 tok/s); the raw per-token decode is ~61 ms.

## What was done

1. **Direct-register B fragments in the tensor-core decode GEMM.** The
   previous kernel dequantized the weight tile into shared memory and
   loaded the MMA B fragments with `ldmatrix`; the shared round-trip plus
   the load/MMA dependency cost ~100 µs per GEMM (measured by ablating the
   pieces in a standalone harness). Each lane now dequantizes its own four
   B elements (two packed words, the scale/zero-point) straight into
   registers, with 4 independent accumulator chains over the k sub-tiles.
   The kernel runs at the DRAM bandwidth floor (~59 µs vs 52 µs for
   5120×17408); decode dropped 81 → 61 ms/token.
2. **Build-time fixes for the iteration loop:**
   - The CUDA build script now skips adapters whose objects are newer than
     their include trees (kernels/custom for the regular adapters, the
     cutlass tree for the cutlass/fa2 sources) and compiles the remaining
     adapters in parallel. A full kernel rebuild went from ~950 s to
     ~295 s; a `.cuh` edit now rebuilds in ~25–30 s.
   - `APXINF_CUDA_ARCH=sm_89` is pinned via `.cargo/config.toml` so the
     tensor-core kernels always build (the multi-arch default also broke
     the bf16 MMA at link time).
   - Worked around the recurrent `rust-lld` futex hang by linking with
     bfd (`RUSTFLAGS=-C link-arg=-fuse-ld=bfd`) and using a private
     `CARGO_TARGET_DIR` to avoid lock contention with the other session
     building in the shared target directory.

## Obstacles

- The standalone kernel harness (which enabled the fast ablations)
  initially misled the B-fragment pair order; the model's token-level
  check is the ground truth and caught it.
- The 4-way accumulator reordering changed the f32 sum order and flipped a
  near-tie; the k-order is preserved within each chain so the trajectory
  stayed exact.

## Remaining gap

Decode is ~61 ms/token against the 33 ms target. The GEMMs are now at the
bandwidth floor (~16 ms/token of weight traffic); the remaining overhead
is the per-GEMM launch/tail (416 launches), the scale/zero-point reads,
and the non-GEMM kernels (delta, norms, attention). Next: fuse the
per-layer GEMM sequence into fewer launches, and consider CUDA graphs for
the decode loop.
