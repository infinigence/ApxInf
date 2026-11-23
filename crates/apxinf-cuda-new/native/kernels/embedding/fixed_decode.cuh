// Copyright 2026 ApxInf contributors.
// Fixed decode kernels, moved without arithmetic changes from the legacy port.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
namespace {
#define FIXED_DECODE_BLOCK_SIZE 256
__global__ void embedding_f32_kernel(
    const float* table, const uint32_t* ids, float* output,
    uint32_t embed_dim, uint32_t seq_len)
{
    uint32_t dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t seq_idx = blockIdx.y;
    if (dim_idx >= embed_dim || seq_idx >= seq_len) return;

    uint32_t token_id = ids[seq_idx];
    output[seq_idx * embed_dim + dim_idx] = table[token_id * embed_dim + dim_idx];
}

__global__ void embedding_bf16_kernel(
    const __nv_bfloat16* table, const uint32_t* ids, __nv_bfloat16* output,
    uint32_t embed_dim, uint32_t seq_len)
{
    uint32_t dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t seq_idx = blockIdx.y;
    if (dim_idx >= embed_dim || seq_idx >= seq_len) return;

    uint32_t token_id = ids[seq_idx];
    output[seq_idx * embed_dim + dim_idx] = table[token_id * embed_dim + dim_idx];
}

// ── RMSNorm — legacy kernels/custom/normalization.cuh ────────────────────


}  // namespace
