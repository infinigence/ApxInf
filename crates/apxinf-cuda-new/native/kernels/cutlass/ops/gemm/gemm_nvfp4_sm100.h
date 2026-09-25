// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>
#include <stddef.h>

namespace apxinf::cuda::cutlass_ops {

// Number of tile/cluster configurations exposed to the autotuner.
int nvfp4_gemm_tactic_count();

// Whether a tactic can serve this shape at all. Configurations that overrun
// TMEM on the target device are excluded here rather than discovered by a
// kernel abort, because a CUDA context that has faulted cannot be reused.
bool nvfp4_gemm_tactic_supported(int tactic, int m, int n, int k, int sf_vec);

// Upper bound on device scratch the chosen tactic needs.
size_t nvfp4_gemm_workspace_bytes(int m, int n, int k, int sf_vec, int tactic);

// Y[M,N] (bf16, row-major) = alpha * sum_k A[M,K] * B[N,K]
//
//   a     packed E2M1, physically [M, K/2], K contiguous
//   b     packed E2M1, physically [N, K/2], K contiguous
//   a_sf  unsigned E4M3 block scales for A, already in the kernel atom layout
//   b_sf  unsigned E4M3 block scales for B, already in the kernel atom layout
//
// The per-tensor scales a ModelOpt checkpoint carries (weight_scale_2,
// input_scale) multiply the whole projection and belong in `alpha`.
int nvfp4_gemm_bf16(const void* a, const void* a_sf, const void* b,
                    const void* b_sf, void* out, void* workspace,
                    size_t workspace_bytes, int m, int n, int k, int sf_vec,
                    float alpha, int tactic, cudaStream_t stream);

// Bytes required for one operand's block-scale buffer in the kernel's atom
// layout. `rows` is M for the activation and N for the weight.
size_t nvfp4_scale_buffer_bytes(int rows, int k, int sf_vec);

// Rewrite row-major `[rows, k/sf_vec]` E4M3 block scales -- the layout a
// checkpoint stores -- into the atom layout the kernel reads.
//
// This is a load-time transform: the atom layout depends only on sf_vec, not
// on the tile configuration, so one conversion serves every tactic and the
// autotuner may switch tactics without invalidating it.
int nvfp4_scatter_block_scales(const void* src_row_major, void* dst_atom,
                               int rows, int k, int sf_vec,
                               cudaStream_t stream);

// Quantize a BF16 activation to the NVFP4 operand pair the GEMM consumes.
//
//   src          BF16 [rows, k], row-major
//   dst_packed   E2M1 pairs [rows, k/2], low nibble first
//   dst_scales   unsigned-E4M3 block scales in the kernel atom layout
//
// Follows the ModelOpt convention: each block's scale is `amax/6` expressed
// relative to the checkpoint's per-tensor `input_scale`, so the GEMM recovers
// absolute values by folding `input_scale` into alpha. Passing the same
// `input_scale` the checkpoint stores is therefore required, not optional --
// it is what makes the activation's scale range match what the weights were
// calibrated against.
//
// `row_major_scales` selects the scale layout: 0 writes the tcgen05 atom
// layout the block-scaled GEMM reads, 1 writes the plain [rows, k/sf_vec]
// grid the GEMV indexes.
int nvfp4_quantize_activation(const void* src_bf16, void* dst_packed,
                              void* dst_scales, int rows, int k, int sf_vec,
                              float input_scale, int row_major_scales,
                              cudaStream_t stream);

// RMSNorm fused with NVFP4 quantization.
//
// Running them separately writes a [rows, k] BF16 tensor and reads it straight
// back. At prefill widths that round trip costs more than either op's
// arithmetic, so the fused form is the one worth having.
int nvfp4_quantize_rms_norm(const void* src_bf16, const void* norm_weight,
                            void* dst_packed, void* dst_scales, int rows,
                            int k, int sf_vec, float epsilon,
                            float input_scale, int row_major_scales,
                            cudaStream_t stream);

// SwiGLU fused with NVFP4 quantization.
//
// `src_bf16` is the [rows, 2*k] fused gate/up projection, gate first. The
// unfused path materializes a [rows, k] BF16 intermediate -- ~36 MB per layer
// at 1024 tokens, written and immediately re-read.
int nvfp4_quantize_swiglu(const void* src_bf16, void* dst_packed,
                          void* dst_scales, int rows, int k, int sf_vec,
                          float input_scale, int row_major_scales,
                          cudaStream_t stream);

}  // namespace apxinf::cuda::cutlass_ops
