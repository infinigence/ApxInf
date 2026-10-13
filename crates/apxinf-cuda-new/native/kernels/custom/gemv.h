// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>

namespace apxinf::cuda_new::gemv {

// Single-token projections: GEMMs at M=1, specialized for the decode path.
//
// These are not members of the tunable GEMM family in adapters/gemm/ -- they
// have one implementation each and no candidate registry -- but the math is a
// GEMM, so they live beside it rather than in a drawer named after the MLP that
// calls them (doc/cuda-new-design.md section 1.3).

// y[n] = alpha * sum_k weight[n, k] * activation[k], E4M3 operands, BF16 out.
//
// The single-token projection. A general GEMM reaches about half of this
// device's bandwidth at M=1 because its tiling is built for large M; here one
// block owns one output row and reads that row contiguously, which is the
// access pattern the hardware wants.
//
// `weight` is [N, K] -- the checkpoint's own orientation, so this path also
// skips the transpose the [K, N] GEMM contract requires.
int fp8_gemv(const void* weight, const void* activation, void* output, int n,
             int k, float alpha, cudaStream_t stream);

// y[n] = alpha * sum_k dequant(weight[n,k]) * dequant(activation[k]), NVFP4
// operands with one E4M3 scale per 16 elements, BF16 out.
//
// MEASURED 13x SLOWER than the block-scaled GEMM at M=1 (37.0 ms against
// 2.85 ms on the 248320x5120 lm_head). Unlike the FP8 GEMV, which won, this
// one is ALU-bound rather than memory-bound: each 16-byte load unpacks 32
// nibbles through a __constant__ lookup table, and non-uniform indexing into
// constant memory serializes within a warp. A viable version would use the
// hardware FP4 conversion intrinsics instead of a table. Kept for that work;
// do not use it as-is.
//
// Unlike the block-scaled GEMM this reads scales in plain row-major
// [rows, K/16] -- the layout a checkpoint stores. The CUTLASS atom layout
// exists for the tcgen05 MMA; a GEMV indexes scales directly, so the decode
// path needs no relayout at all.
//
//   weight  packed E2M1, [N, K/2] bytes
//   w_scale E4M3, [N, K/16]
//   act     packed E2M1, [K/2] bytes
//   a_scale E4M3, [K/16]
int nvfp4_gemv(const void* weight, const void* weight_scales,
               const void* activation, const void* activation_scales,
               void* output, int n, int k, float alpha, cudaStream_t stream);

}  // namespace apxinf::cuda_new::gemv
