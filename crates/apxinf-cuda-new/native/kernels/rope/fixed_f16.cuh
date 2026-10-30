// Copyright 2026 ApxInf contributors.
// Fixed FP16 implementation for this operator family; ported without numerical changes.
#include "../custom/reduction.cuh"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kThreads = 256;

int blocks_for(int64_t count) {
  return static_cast<int>((count + kThreads - 1) / kThreads);
}

int clamped_blocks(int64_t count) {
  int blocks = static_cast<int>((count + 255) / 256);
  return blocks > 1024 ? 1024 : blocks;
}

// ── rope (kernels/custom/rope.cuh) ──────────────────────────────────────────

__global__ void rope_half_split_f16_kernel(
    const half* input, half* output,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    float rope_theta, uint32_t pos_offset)
{
    uint32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head_idx = blockIdx.y;
    uint32_t seq_idx  = blockIdx.z;
    if (pair_idx >= head_dim / 2) return;

    uint32_t pos = seq_idx + pos_offset;
    float frequency = 1.0f / powf(
        rope_theta, 2.0f * static_cast<float>(pair_idx) /
                         static_cast<float>(head_dim));
    float angle = static_cast<float>(pos) * frequency;
    float sine;
    float cosine;
    sincosf(angle, &sine, &cosine);

    uint32_t base = seq_idx * n_heads * head_dim + head_idx * head_dim;
    uint32_t first = base + pair_idx;
    uint32_t second = base + head_dim / 2 + pair_idx;
    float x0 = __half2float(input[first]);
    float x1 = __half2float(input[second]);
    output[first] = __float2half(x0 * cosine - x1 * sine);
    output[second] = __float2half(x0 * sine + x1 * cosine);
}

}  // namespace
