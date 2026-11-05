// Copyright 2026 ApxInf contributors.
// Fixed decode kernels, moved without arithmetic changes from the legacy port.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
namespace {
#define FIXED_DECODE_BLOCK_SIZE 256
__global__ void add_f32_kernel(
    const float* a, const float* b, float* output, uint32_t count)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    output[gid] = a[gid] + b[gid];
}

__global__ void mul_f32_kernel(
    const float* a, const float* b, float* output, uint32_t count)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    output[gid] = a[gid] * b[gid];
}

__global__ void add_bf16_kernel(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* output, uint32_t count)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    float x = __bfloat162float(a[gid]) + __bfloat162float(b[gid]);
    output[gid] = __float2bfloat16(x);
}

__global__ void mul_bf16_kernel(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* output, uint32_t count)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    float x = __bfloat162float(a[gid]) * __bfloat162float(b[gid]);
    output[gid] = __float2bfloat16(x);
}

__global__ void scale_f32_kernel(
    const float* input, float* output, uint32_t count, float scale)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < count) output[gid] = input[gid] * scale;
}

__global__ void scale_bf16_kernel(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    uint32_t count, float scale)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < count)
        output[gid] = __float2bfloat16(__bfloat162float(input[gid]) * scale);
}

// ── Embedding — legacy kernels/custom/embedding.cuh ──────────────────────


}  // namespace
