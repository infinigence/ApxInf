// Copyright 2026 ApxInf contributors.
// Fixed implementation for this operator family; ported without numerical changes.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {
#include "../custom/reduction.cuh"
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
