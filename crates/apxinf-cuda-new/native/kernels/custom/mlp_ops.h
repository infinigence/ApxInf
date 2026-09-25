// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>

namespace apxinf::cuda::mlp_ops {

// y[r,c] = x[r,c] / sqrt(mean(x[r,:]^2) + epsilon) * weight[c]
//
// Reduction accumulates in f32: summing thousands of squared BF16 values in
// BF16 drops the small terms outright.
int rms_norm_bf16(const void* input, const void* weight, void* output,
                  int rows, int width, float epsilon, cudaStream_t stream);

// y[r,c] = silu(fused[r,c]) * fused[r,width+c]
//
// `fused_gate_up` is [rows, 2*width] with gate first, which is what one fused
// gate/up GEMM produces. This is SwiGLU, not the GELU-based `gemm_geglu`.
int swiglu_bf16(const void* fused_gate_up, void* output, int rows, int width,
                cudaStream_t stream);

// accumulator += addend, elementwise. Residual connections.
int add_bf16(const void* addend, void* accumulator, long long count,
             cudaStream_t stream);

// Quantize BF16 to E4M3 against a single per-tensor scale.
//
// This is the activation side of the FP8 projections in a ModelOpt checkpoint,
// whose weight_scale and input_scale are scalars rather than the [M]/[N]
// vectors the scaled-FP8 GEMM contract expects. Quantizing against
// input_scale lets the projection run as unit-scale FP8 with
// `alpha = weight_scale * input_scale`, so no new quantization contract is
// needed for attention or GDN.
int quantize_fp8_per_tensor(const void* input, void* output, long long count,
                            float input_scale, cudaStream_t stream);

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

}  // namespace apxinf::cuda::mlp_ops
