// Copyright 2026 ApxInf contributors.
// Fixed decode kernels, moved without arithmetic changes from the legacy port.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
namespace {
#define FIXED_DECODE_BLOCK_SIZE 256
__global__ void rms_norm_f32_kernel(
    const float* input,
    const float* weight,
    float* output,
    uint32_t cols,
    uint32_t rows,
    float eps)
{
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y;
    if (col >= cols || row >= rows) return;

    uint32_t offset = row * cols;
    float sum_sq = 0.0f;
    for (uint32_t i = 0; i < cols; i++) {
        float v = input[offset + i];
        sum_sq += v * v;
    }
    float rms = rsqrtf(sum_sq / (float)cols + eps);
    output[offset + col] = input[offset + col] * rms * weight[col];
}

// Shared-memory reduction: one block per row, strided load, warp-shuffle
// reduction.
__global__ void rms_norm_bf16_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* weight, __nv_bfloat16* output,
    uint32_t cols, uint32_t rows, float eps)
{
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    uint32_t tid = threadIdx.x;
    uint32_t offset = row * cols;

    extern __shared__ float x_buf[];
    __shared__ float s_sum;

    float partial = 0.0f;
    for (uint32_t i = tid; i < cols; i += blockDim.x) {
        float v = __bfloat162float(input[offset + i]);
        x_buf[i] = v;
        partial += v * v;
    }

    for (int off = 16; off > 0; off >>= 1)
        partial += __shfl_xor_sync(0xffffffff, partial, off);
    __shared__ float warp_sums[32];
    uint32_t warp_id = tid / 32;
    uint32_t lane = tid % 32;
    if (lane == 0) warp_sums[warp_id] = partial;
    __syncthreads();
    if (warp_id == 0) {
        float v = (tid < (blockDim.x + 31) / 32) ? warp_sums[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_xor_sync(0xffffffff, v, off);
        if (lane == 0) s_sum = v;
    }
    __syncthreads();
    float rms = rsqrtf(s_sum / (float)cols + eps);

    for (uint32_t i = tid; i < cols; i += blockDim.x) {
        float w = __bfloat162float(weight[i]);
        output[offset + i] = __float2bfloat16(x_buf[i] * rms * w);
    }
}

// ── Fused RMSNorm + residual add (bf16) — legacy kernels/custom/fused.cuh ─

__global__ void rms_norm_add_bf16_kernel(
    __nv_bfloat16* x_inout, const __nv_bfloat16* delta,
    const __nv_bfloat16* weight, __nv_bfloat16* output,
    uint32_t cols, uint32_t rows, float eps)
{
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    uint32_t tid = threadIdx.x;
    uint32_t offset = row * cols;

    extern __shared__ float smem[];
    float* x_new = smem;        // [cols]
    __shared__ float s_sum;

    float partial = 0.0f;
    for (uint32_t i = tid; i < cols; i += blockDim.x) {
        float xv = __bfloat162float(x_inout[offset + i]);
        float dv = __bfloat162float(delta[offset + i]);
        float xn = xv + dv;
        x_new[i] = xn;
        partial += xn * xn;
    }

    for (int off = 16; off > 0; off >>= 1)
        partial += __shfl_xor_sync(0xffffffff, partial, off);
    __shared__ float warp_sums[32];
    uint32_t warp_id = tid / 32;
    uint32_t lane = tid % 32;
    if (lane == 0) warp_sums[warp_id] = partial;
    __syncthreads();
    if (warp_id == 0) {
        float v = (tid < (blockDim.x + 31) / 32) ? warp_sums[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_xor_sync(0xffffffff, v, off);
        if (lane == 0) s_sum = v;
    }
    __syncthreads();
    float rms = rsqrtf(s_sum / (float)cols + eps);

    for (uint32_t i = tid; i < cols; i += blockDim.x) {
        float xn = x_new[i];
        float w  = __bfloat162float(weight[i]);
        x_inout[offset + i] = __float2bfloat16(xn);
        output[offset + i]  = __float2bfloat16(xn * rms * w);
    }
}

// ── Attention softmax — legacy kernels/custom/attention.cuh ─────────────


}  // namespace
