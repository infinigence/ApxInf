// Copyright 2026 ApxInf contributors.
// Fixed FP16 implementation for this operator family; ported without numerical changes.
#include "../../include/apxinf_cuda/transformer_f16.h"
#include "../../kernels/custom/reduction.cuh"

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

extern "C" cudaError_t apxinf_transformer_f16_bias_f16(
    const void* input, const void* bias, void* output, int rows, int cols,
    cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
  int64_t count = static_cast<int64_t>(rows) * cols;
  bias_f16_kernel<<<clamped_blocks(count), 256, 0, stream>>>(
      static_cast<const half*>(input), static_cast<const half*>(bias),
      static_cast<half*>(output), count, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_transformer_f16_euler_update_f16(
    const void* state, const void* velocity, void* output, int64_t count,
    float dt, cudaStream_t stream) {
  if (count <= 0) return cudaErrorInvalidValue;
  euler_update_f16_kernel<<<clamped_blocks(count), 256, 0, stream>>>(
      static_cast<const half*>(state), static_cast<const half*>(velocity),
      static_cast<half*>(output), count, dt);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_transformer_f16_add_f16(
    const void* a, const void* b, void* output, uint32_t count,
    cudaStream_t stream) {
  const uintptr_t aa = reinterpret_cast<uintptr_t>(a);
  const uintptr_t ba = reinterpret_cast<uintptr_t>(b);
  const uintptr_t oa = reinterpret_cast<uintptr_t>(output);
  if ((count % 8u) == 0u && (aa % 16u) == 0u && (ba % 16u) == 0u &&
      (oa % 16u) == 0u) {
    add_f16_vec8_kernel<<<(count / 8 + kThreads - 1) / kThreads,
                          kThreads, 0, stream>>>(
        (const float4*)a, (const float4*)b, (float4*)output, count / 8);
  } else {
    dim3 grid((count + kThreads - 1) / kThreads, 1, 1);
    dim3 block(kThreads, 1, 1);
    add_f16_kernel<<<grid, block, 0, stream>>>(
        (const half*)a, (const half*)b, (half*)output, count);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_transformer_f16_scale_f16(
    const void* input, void* output, uint32_t count, float scale,
    cudaStream_t stream) {
  const uintptr_t input_address = reinterpret_cast<uintptr_t>(input);
  const uintptr_t output_address = reinterpret_cast<uintptr_t>(output);
  if ((count % 8u) == 0u && (input_address % 16u) == 0u &&
      (output_address % 16u) == 0u) {
    scale_f16_vec8_kernel<<<(count / 8 + kThreads - 1) / kThreads,
                            kThreads, 0, stream>>>(
        (const float4*)input, (float4*)output, count / 8, scale);
  } else {
    dim3 grid((count + kThreads - 1) / kThreads, 1, 1);
    dim3 block(kThreads, 1, 1);
    scale_f16_kernel<<<grid, block, 0, stream>>>(
        (const half*)input, (half*)output, count, scale);
  }
  return cudaGetLastError();
}
