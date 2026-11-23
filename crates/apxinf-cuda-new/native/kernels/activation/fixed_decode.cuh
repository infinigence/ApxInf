// Copyright 2026 ApxInf contributors.
// Fixed decode kernels, moved without arithmetic changes from the legacy port.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
namespace {
#define FIXED_DECODE_BLOCK_SIZE 256
__global__ void silu_f32_kernel(
    const float* input, float* output, uint32_t count)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    float x = input[gid];
    output[gid] = x / (1.0f + expf(-x));
}

__global__ void silu_bf16_kernel(
    const __nv_bfloat16* input, __nv_bfloat16* output, uint32_t count)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    float x = __bfloat162float(input[gid]);
    float y = x / (1.0f + expf(-x));
    output[gid] = __float2bfloat16(y);
}

// Fused SiLU + Mul (bf16) over a packed [gate | up] buffer.
__global__ void silu_mul_bf16_kernel(
    const __nv_bfloat16* gate_up, __nv_bfloat16* output, uint32_t inter)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= inter) return;
    float g = __bfloat162float(gate_up[gid]);
    float u = __bfloat162float(gate_up[gid + inter]);
    float s = g / (1.0f + expf(-g));
    output[gid] = __float2bfloat16(s * u);
}

// ── Elementwise — legacy kernels/custom/elementwise.cuh ──────────────────


}  // namespace
