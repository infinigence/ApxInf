// Copyright 2026 ApxInf contributors.
// Interleaved-pair rotary embedding driven by precomputed cos/sin tables;
// ported verbatim from the legacy speech kernel without numerical changes.
#include "../custom/reduction.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

namespace {

constexpr int kThreads = 256;

template <class T>
__device__ __forceinline__ float rope_to_float(T value);
template <>
__device__ __forceinline__ float rope_to_float<float>(float value) { return value; }
template <>
__device__ __forceinline__ float rope_to_float(__half value) { return __half2float(value); }
template <>
__device__ __forceinline__ float rope_to_float(__nv_bfloat16 value) { return __bfloat162float(value); }

template <class T>
__device__ __forceinline__ T rope_from_float(float value);
template <>
__device__ __forceinline__ float rope_from_float<float>(float value) { return value; }
template <>
__device__ __forceinline__ __half rope_from_float<__half>(float value) { return __float2half(value); }
template <>
__device__ __forceinline__ __nv_bfloat16 rope_from_float<__nv_bfloat16>(float value) { return __float2bfloat16(value); }

template <class T>
__global__ void rope_apply_interleaved_kernel(
    const T* __restrict__ input, const T* __restrict__ cos_values,
    const T* __restrict__ sin_values, T* __restrict__ output, int64_t pairs,
    int64_t time, int64_t heads, int64_t head_dim) {
  const int64_t index = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (index >= pairs) return;
  const int64_t pair = index % (head_dim / 2);
  const int64_t row = index / (head_dim / 2);
  const int64_t head = row % heads;
  const int64_t position = (row / heads) % time;
  const int64_t batch = (row / heads) / time;
  const int64_t element = 2 * pair;
  const int64_t first =
      ((batch * time + position) * heads + head) * head_dim + element;
  const int64_t second = first + 1;
  const T c = cos_values[position * head_dim + element];
  const T s = sin_values[position * head_dim + element];
  const T x1 = input[first];
  const T x2 = input[second];
  // Round products independently; CUDA may otherwise contract them into HFMA.
  const T x1c = rope_from_float<T>(__fmul_rn(rope_to_float(x1), rope_to_float(c)));
  const T x2s = rope_from_float<T>(__fmul_rn(rope_to_float(x2), rope_to_float(s)));
  const T x2c = rope_from_float<T>(__fmul_rn(rope_to_float(x2), rope_to_float(c)));
  const T x1s = rope_from_float<T>(__fmul_rn(rope_to_float(x1), rope_to_float(s)));
  output[first] = rope_from_float<T>(__fsub_rn(rope_to_float(x1c), rope_to_float(x2s)));
  output[second] = rope_from_float<T>(__fadd_rn(rope_to_float(x2c), rope_to_float(x1s)));
}

}  // namespace

template <class T>
cudaError_t launch_rope_apply_interleaved(
    const void* input, const void* cos_values, const void* sin_values,
    void* output, int64_t tokens, int64_t time, int64_t heads,
    int64_t head_dim, cudaStream_t stream) {
  const int64_t pairs = tokens * heads * (head_dim / 2);
  const int64_t blocks = (pairs + kThreads - 1) / kThreads;
  if (blocks > INT_MAX) return cudaErrorInvalidValue;
  rope_apply_interleaved_kernel<T>
      <<<static_cast<int>(blocks), kThreads, 0, stream>>>(
          static_cast<const T*>(input), static_cast<const T*>(cos_values),
          static_cast<const T*>(sin_values), static_cast<T*>(output), pairs,
          time, heads, head_dim);
  return cudaGetLastError();
}

template cudaError_t launch_rope_apply_interleaved<float>(
    const void*, const void*, const void*, void*, int64_t, int64_t, int64_t,
    int64_t, cudaStream_t);
template cudaError_t launch_rope_apply_interleaved<__half>(
    const void*, const void*, const void*, void*, int64_t, int64_t, int64_t,
    int64_t, cudaStream_t);
template cudaError_t launch_rope_apply_interleaved<__nv_bfloat16>(
    const void*, const void*, const void*, void*, int64_t, int64_t, int64_t,
    int64_t, cudaStream_t);
