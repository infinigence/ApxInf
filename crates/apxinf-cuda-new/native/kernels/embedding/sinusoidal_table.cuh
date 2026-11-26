// Copyright 2026 ApxInf contributors.
// Reference sinusoidal table kernel, ported verbatim from the legacy family:
// `scale` multiplies each scalar time with the product rounded to the storage
// dtype before the frequency sweep, unlike the newer float-time variant.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int kTableThreads = 256;

template <class T>
__device__ float table_to_float(T value);

template <>
__device__ inline float table_to_float(float value) {
  return value;
}

template <>
__device__ inline float table_to_float(__half value) {
  return __half2float(value);
}

template <>
__device__ inline float table_to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <class T>
__device__ T table_from_float(float value);

template <>
__device__ inline float table_from_float(float value) {
  return value;
}

template <>
__device__ inline __half table_from_float(float value) {
  return __float2half(value);
}

template <>
__device__ inline __nv_bfloat16 table_from_float(float value) {
  return __float2bfloat16(value);
}

template <class T>
__global__ void sinusoidal_table_kernel(
    const T* times, T* output, int64_t rows, int64_t width,
    float scale, float log_step) {
  const int64_t index = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
  if (index >= rows * width) return;
  const int64_t row = index / width;
  const int64_t column = index % width;
  const int64_t half_width = width / 2;
  const T scaled = table_from_float<T>(table_to_float(times[row]) * scale);
  const float frequency = expf(-static_cast<float>(column % half_width) * log_step);
  const float angle = table_to_float(scaled) * frequency;
  output[index] = table_from_float<T>(column < half_width ? sinf(angle) : cosf(angle));
}

}  // namespace
