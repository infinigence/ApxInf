// Copyright 2026 ApxInf contributors.
// Fixed implementation for this operator family; ported without numerical changes.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace {
#include "../../kernels/custom/gdn_policy.h"
#include "../../kernels/custom/math.cuh"
#include "../../kernels/custom/reduction.cuh"
#include "../../kernels/custom/pillow_bicubic_u8.cuh"
#include "../../kernels/custom/linear_attention.cuh"
#include "../../kernels/custom/gdn_raw_inverse_f1.cuh"
#include "../../kernels/custom/gdn_chunk_state_wmma.cuh"
#include "../../kernels/custom/gdn_chunk_gemm_tri.cuh"
__global__ void row_softmax_f32_bf16_kernel(
    const float* input, __nv_bfloat16* output, uint32_t cols, uint32_t rows) {
  const uint32_t row = blockIdx.x;
  if (row >= rows) return;
  const float* x = input + static_cast<size_t>(row) * cols;
  __shared__ float scratch[32];
  float local = -INFINITY;
  for (uint32_t i = threadIdx.x; i < cols; i += blockDim.x) {
    local = fmaxf(local, x[i]);
  }
  const float max_val = block_max_parallel_unsafe(local, scratch);
  float partial = 0.0f;
  for (uint32_t i = threadIdx.x; i < cols; i += blockDim.x) {
    partial += expf(x[i] - max_val);
  }
  // Finish all maximum reads before the sum overwrites shared scratch.
  __syncthreads();
  const float sum = block_sum_parallel_unsafe(partial, scratch);
  __nv_bfloat16* y = output + static_cast<size_t>(row) * cols;
  for (uint32_t i = threadIdx.x; i < cols; i += blockDim.x) {
    y[i] = __float2bfloat16(expf(x[i] - max_val) / sum);
  }
}


// FIX (implement_final_r20): Option A budget repair for the composed hdim256 text
// path -- tiled block-per-row causal fp32 softmax replacing the thread-per-element
// attention_softmax_f32_kernel (measured ~3.9-4.1s per composed full-attention layer,
// ~31.6s of the ~43s prefill). Row contract identical to the stock kernel
// (kernels/custom/attention.cuh): row = blockIdx.x, seq_pos = row / n_heads,
// valid = min(seq_pos + kv_offset + 1u, cols); valid cells expf(x-max)/sum, masked
// cells exact 0.0f. In-place safe with input == output: every index is owned by
// exactly one thread and the block reductions (block_max/block_sum) synchronize
// between the read and write passes. Revert in the acceptance-bound revision.
__global__ void row_softmax_causal_f32_kernel(
    const float* input, float* output, uint32_t cols, uint32_t rows,
    uint32_t kv_offset, uint32_t n_heads) {
  const uint32_t row = blockIdx.x;
  if (row >= rows) return;
  const float* x = input + static_cast<size_t>(row) * cols;
  float* y = output + static_cast<size_t>(row) * cols;
  const uint32_t seq_pos = row / n_heads;
  const uint32_t valid = min(seq_pos + kv_offset + 1u, cols);
  __shared__ float scratch[32];
  float local = -INFINITY;
  for (uint32_t i = threadIdx.x; i < valid; i += blockDim.x) {
    local = fmaxf(local, x[i]);
  }
  const float max_val = block_max_parallel_unsafe(local, scratch);
  float partial = 0.0f;
  for (uint32_t i = threadIdx.x; i < valid; i += blockDim.x) {
    partial += expf(x[i] - max_val);
  }
  // Finish all maximum reads before the sum overwrites shared scratch.
  __syncthreads();
  const float sum = block_sum_parallel_unsafe(partial, scratch);
  for (uint32_t i = threadIdx.x; i < cols; i += blockDim.x) {
    y[i] = (i < valid) ? (expf(x[i] - max_val) / sum) : 0.0f;
  }
}

}  // namespace

extern "C" cudaError_t apxinf_hybrid_row_softmax_f32_bf16(
    const void* input, void* output, uint32_t cols, uint32_t rows,
    cudaStream_t stream) {
  if (input == nullptr || output == nullptr || cols == 0 || rows == 0) {
    return cudaErrorInvalidValue;
  }
  row_softmax_f32_bf16_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const float*>(input), static_cast<__nv_bfloat16*>(output),
      cols, rows);
  return cudaGetLastError();
}


extern "C" cudaError_t apxinf_hybrid_row_softmax_causal_f32(
    const void* input, void* output, uint32_t cols, uint32_t rows,
    uint32_t kv_offset, uint32_t n_heads, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || cols == 0 || rows == 0 ||
      n_heads == 0 || n_heads > rows) {
    return cudaErrorInvalidValue;
  }
  row_softmax_causal_f32_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const float*>(input), static_cast<float*>(output), cols, rows,
      kv_offset, n_heads);
  return cudaGetLastError();
}


extern "C" cudaError_t apxinf_hybrid_full_attn_prepare_bf16(
    const void* fused, const void* q_norm_w, const void* k_norm_w,
    const void* cos, const void* sin, void* q_out, void* k_cache, void* v_cache,
    int seq, int cache_offset, int q_heads, int kv_heads, int head_dim,
    int rotary_dim, int64_t fused_width, int64_t cache_width, float eps,
    cudaStream_t stream) {
  if (fused == nullptr || q_norm_w == nullptr || k_norm_w == nullptr ||
      cos == nullptr || sin == nullptr || q_out == nullptr ||
      k_cache == nullptr || v_cache == nullptr || seq <= 0 ||
      cache_offset < 0 || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
      rotary_dim <= 0 || (rotary_dim & 1) != 0 || rotary_dim > head_dim ||
      !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  // Opt-in only for the measured D256 BF16 prefill geometry. Keep the
  // original C ABI and generic route for all other inputs and decode.
  if (seq > 1 && q_heads == 16 && kv_heads == 4 && head_dim == 256 &&
      rotary_dim == 64 && fused_width == 10240 && cache_width == 1024) {
    full_attn_prepare_bf16_warp_kernel
        <<<dim3(seq, 6), 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(fused),
        static_cast<const __nv_bfloat16*>(q_norm_w),
        static_cast<const __nv_bfloat16*>(k_norm_w),
        static_cast<const __nv_bfloat16*>(cos),
        static_cast<const __nv_bfloat16*>(sin),
        static_cast<__nv_bfloat16*>(q_out),
        static_cast<__nv_bfloat16*>(k_cache),
        static_cast<__nv_bfloat16*>(v_cache),
        cache_offset, q_heads, kv_heads, fused_width, cache_width, eps);
    return cudaGetLastError();
  }
  const size_t smem = static_cast<size_t>(head_dim) * sizeof(float);
  full_attn_prepare_bf16_kernel
      <<<dim3(seq, q_heads + 2 * kv_heads), 128, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(fused),
      static_cast<const __nv_bfloat16*>(q_norm_w),
      static_cast<const __nv_bfloat16*>(k_norm_w),
      static_cast<const __nv_bfloat16*>(cos),
      static_cast<const __nv_bfloat16*>(sin),
      static_cast<__nv_bfloat16*>(q_out),
      static_cast<__nv_bfloat16*>(k_cache),
      static_cast<__nv_bfloat16*>(v_cache),
      cache_offset, q_heads, kv_heads, head_dim, rotary_dim, fused_width,
      cache_width, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_hybrid_expert_qkv_prepare_bf16(
    const void* fused, const void* q_norm_w, const void* k_norm_w,
    const void* cos, const void* sin, void* q_out, void* gate_out, void* k_out,
    void* v_out, int seq, int q_heads, int kv_heads, int head_dim,
    int rotary_dim, int64_t fused_width, float eps, cudaStream_t stream) {
  if (fused == nullptr || q_norm_w == nullptr || k_norm_w == nullptr ||
      cos == nullptr || sin == nullptr || q_out == nullptr ||
      gate_out == nullptr || k_out == nullptr || v_out == nullptr ||
      seq <= 0 || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
      rotary_dim <= 0 || (rotary_dim & 1) != 0 || rotary_dim > head_dim ||
      q_heads % kv_heads != 0 || !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  const size_t smem = static_cast<size_t>(head_dim) * sizeof(float);
  expert_qkv_prepare_bf16_kernel
      <<<dim3(seq, q_heads + 2 * kv_heads), 128, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(fused),
      static_cast<const __nv_bfloat16*>(q_norm_w),
      static_cast<const __nv_bfloat16*>(k_norm_w),
      static_cast<const __nv_bfloat16*>(cos),
      static_cast<const __nv_bfloat16*>(sin),
      static_cast<__nv_bfloat16*>(q_out),
      static_cast<__nv_bfloat16*>(gate_out),
      static_cast<__nv_bfloat16*>(k_out),
      static_cast<__nv_bfloat16*>(v_out),
      q_heads, kv_heads, head_dim, rotary_dim, fused_width, eps);
  return cudaGetLastError();
}
