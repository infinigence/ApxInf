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

// ── elementwise (kernels/custom/elementwise.cuh) ────────────────────────────

__global__ void bias_f16_kernel(
    const half* input, const half* bias, half* output, int64_t count, int cols) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    float value = __half2float(input[index]);
    if (bias != nullptr) value += __half2float(bias[index % cols]);
    output[index] = __float2half(value);
  }
}

__global__ void euler_update_f16_kernel(
    const half* state, const half* velocity, half* output,
    int64_t count, float dt) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    output[index] = __float2half(
        __half2float(state[index]) + dt * __half2float(velocity[index]));
  }
}

__global__ void add_f16_kernel(
    const half* a, const half* b, half* output, uint32_t count)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    output[gid] = __float2half(__half2float(a[gid]) + __half2float(b[gid]));
}

__global__ void add_f16_vec8_kernel(
    const float4* __restrict__ a, const float4* __restrict__ b,
    float4* __restrict__ output, uint32_t vec_count)
{
    for (uint32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < vec_count;
         v += gridDim.x * blockDim.x) {
        float4 pa = a[v];
        float4 pb = b[v];
        const half* la = reinterpret_cast<const half*>(&pa);
        const half* lb = reinterpret_cast<const half*>(&pb);
        float4 out;
        half* lo = reinterpret_cast<half*>(&out);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            lo[i] = __float2half(__half2float(la[i]) + __half2float(lb[i]));
        }
        output[v] = out;
    }
}

__global__ void scale_f16_kernel(
    const half* input, half* output, uint32_t count, float scale)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    output[gid] = __float2half(__half2float(input[gid]) * scale);
}

__global__ void scale_f16_vec8_kernel(
    const float4* __restrict__ input, float4* __restrict__ output,
    uint32_t vec_count, float scale)
{
    for (uint32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < vec_count;
         v += gridDim.x * blockDim.x) {
        float4 packed = input[v];
        half* lane = reinterpret_cast<half*>(&packed);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            lane[i] = __float2half(__half2float(lane[i]) * scale);
        }
        output[v] = packed;
    }
}

}  // namespace
