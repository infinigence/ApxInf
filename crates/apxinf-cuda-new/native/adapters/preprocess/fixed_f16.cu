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

// ── preprocess (kernels/custom/preprocess.cuh) ──────────────────────────────

__global__ void pixel_shuffle_4_bf16_kernel(
    const __nv_bfloat16* __restrict__ input, __nv_bfloat16* __restrict__ output,
    int views, int tokens_per_view, int width) {
  const int input_side = static_cast<int>(sqrt(static_cast<float>(tokens_per_view)));
  const int output_side = input_side / 4;
  const int output_channels = width * 16;
  const int64_t output_count =
      static_cast<int64_t>(views) * output_side * output_side * output_channels;
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < output_count; index += stride) {
    const int channel = static_cast<int>(index % output_channels);
    const int64_t output_token64 = index / output_channels;
    const int view = static_cast<int>(output_token64 / (output_side * output_side));
    const int output_token =
        static_cast<int>(output_token64 - view * output_side * output_side);
    const int output_y = output_token / output_side;
    const int output_x = output_token - output_y * output_side;
    const int group = channel / width;
    const int input_channel = channel - group * width;
    const int group_y = group / 4;
    const int group_x = group - group_y * 4;
    const int input_y = output_y * 4 + group_y;
    const int input_x = output_x * 4 + group_x;
    const int input_token = input_y * input_side + input_x;
    output[index] =
        input[static_cast<int64_t>(view) * tokens_per_view * width +
              static_cast<int64_t>(input_token) * width + input_channel];
  }
}

__global__ void pixel_shuffle_4_f16_kernel(
    const half* __restrict__ input, half* __restrict__ output,
    int views, int tokens_per_view, int width) {
  const int input_side = static_cast<int>(sqrt(static_cast<float>(tokens_per_view)));
  const int output_side = input_side / 4;
  const int output_channels = width * 16;
  const int64_t output_count =
      static_cast<int64_t>(views) * output_side * output_side * output_channels;
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < output_count; index += stride) {
    const int channel = static_cast<int>(index % output_channels);
    const int64_t output_token64 = index / output_channels;
    const int view = static_cast<int>(output_token64 / (output_side * output_side));
    const int output_token =
        static_cast<int>(output_token64 - view * output_side * output_side);
    const int output_y = output_token / output_side;
    const int output_x = output_token - output_y * output_side;
    const int group = channel / width;
    const int input_channel = channel - group * width;
    const int group_y = group / 4;
    const int group_x = group - group_y * 4;
    const int input_y = output_y * 4 + group_y;
    const int input_x = output_x * 4 + group_x;
    const int input_token = input_y * input_side + input_x;
    output[index] =
        input[static_cast<int64_t>(view) * tokens_per_view * width +
              static_cast<int64_t>(input_token) * width + input_channel];
  }
}

}  // namespace

extern "C" cudaError_t apxinf_transformer_f16_pixel_shuffle_4_bf16(
    const void* input, void* output, int views, int tokens_per_view, int width,
    cudaStream_t stream) {
  if (views <= 0 || tokens_per_view <= 0 || tokens_per_view % 16 != 0 || width <= 0)
    return cudaErrorInvalidValue;
  const int64_t count = static_cast<int64_t>(views) * (tokens_per_view / 16) *
                        static_cast<int64_t>(width) * 16;
  pixel_shuffle_4_bf16_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_bfloat16*>(output), views, tokens_per_view, width);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_transformer_f16_pixel_shuffle_4_f16(
    const void* input, void* output, int views, int tokens_per_view, int width,
    cudaStream_t stream) {
  if (views <= 0 || tokens_per_view <= 0 || tokens_per_view % 16 != 0 || width <= 0)
    return cudaErrorInvalidValue;
  const int64_t count = static_cast<int64_t>(views) * (tokens_per_view / 16) *
                        static_cast<int64_t>(width) * 16;
  pixel_shuffle_4_f16_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
      static_cast<const half*>(input), static_cast<half*>(output),
      views, tokens_per_view, width);
  return cudaGetLastError();
}
