// Copyright 2026 apxinf contributors.
// Stable C ABI and CUDA launch adapter for X-VLA port operators.
//
// Own translation unit (see build.rs) so the shared adapter list stays small;
// declarations live in ../kernels/custom/xvla.cuh.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "../kernels/custom/xvla.cuh"

namespace {

constexpr int kXvlaThreads = 256;

int blocks_for(int64_t count) {
  int64_t blocks = (count + kXvlaThreads - 1) / kXvlaThreads;
  return blocks > INT32_MAX ? INT32_MAX : static_cast<int>(blocks);
}

}  // namespace

extern "C" cudaError_t apxinf_xvla_rgb_u8_to_normalized_tokens_bf16(
    const void* rgb, void* output, int height, int width, const float* mean,
    const float* std, cudaStream_t stream) {
  if (rgb == nullptr || output == nullptr || mean == nullptr || std == nullptr ||
      height <= 0 || width <= 0) {
    return cudaErrorInvalidValue;
  }
  const int64_t count = static_cast<int64_t>(height) * width;
  xvla_rgb_u8_to_normalized_tokens_bf16_kernel<<<blocks_for(count), kXvlaThreads,
                                                 0, stream>>>(
      static_cast<const unsigned char*>(rgb), static_cast<__nv_bfloat16*>(output),
      height, width, mean[0], mean[1], mean[2], std[0], std[1], std[2]);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_xvla_im2col_bf16(
    const void* input, void* columns, int channels, int height, int width,
    int kh, int kw, int stride_h, int stride_w, int pad_h, int pad_w,
    cudaStream_t stream) {
  if (input == nullptr || columns == nullptr || channels <= 0 || height <= 0 ||
      width <= 0 || kh <= 0 || kw <= 0 || stride_h <= 0 || stride_w <= 0 ||
      pad_h < 0 || pad_w < 0) {
    return cudaErrorInvalidValue;
  }
  const int out_h = (height + 2 * pad_h - kh) / stride_h + 1;
  const int out_w = (width + 2 * pad_w - kw) / stride_w + 1;
  if (out_h <= 0 || out_w <= 0) {
    return cudaErrorInvalidValue;
  }
  const int64_t count = static_cast<int64_t>(out_h) * out_w * channels * kh * kw;
  xvla_im2col_bf16_kernel<<<blocks_for(count), kXvlaThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_bfloat16*>(columns), channels, height, width, kh, kw,
      stride_h, stride_w, pad_h, pad_w, out_h, out_w);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_xvla_dwconv3x3_bf16(
    const void* input, const void* weight, const void* bias, void* output,
    int channels, int height, int width, cudaStream_t stream) {
  if (input == nullptr || weight == nullptr || output == nullptr ||
      channels <= 0 || height <= 0 || width <= 0) {
    return cudaErrorInvalidValue;
  }
  const int64_t count = static_cast<int64_t>(channels) * height * width;
  xvla_dwconv3x3_bf16_kernel<<<blocks_for(count), kXvlaThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<__nv_bfloat16*>(output), channels, height, width);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_xvla_gelu_erf_bf16(
    const void* input, void* output, int64_t count, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || count <= 0) {
    return cudaErrorInvalidValue;
  }
  xvla_gelu_erf_bf16_kernel<<<blocks_for(count), kXvlaThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_bfloat16*>(output), count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_xvla_sigmoid_bf16(
    const void* input, void* output, int64_t count, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || count <= 0) {
    return cudaErrorInvalidValue;
  }
  xvla_sigmoid_bf16_kernel<<<blocks_for(count), kXvlaThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_bfloat16*>(output), count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_xvla_pack_action_tokens_bf16(
    const void* action, const void* proprio, const void* time, void* output,
    int rows, int dim_action, int dim_time, int gripper0, int gripper1,
    cudaStream_t stream) {
  if (action == nullptr || proprio == nullptr || time == nullptr ||
      output == nullptr || rows <= 0 || dim_action <= 0 || dim_time <= 0 ||
      gripper0 < 0 || gripper1 < 0 || gripper0 >= dim_action ||
      gripper1 >= dim_action) {
    return cudaErrorInvalidValue;
  }
  const int64_t count = static_cast<int64_t>(rows) *
                        (dim_action * 2 + dim_time);
  xvla_pack_action_tokens_bf16_kernel<<<blocks_for(count), kXvlaThreads, 0,
                                        stream>>>(
      static_cast<const __nv_bfloat16*>(action),
      static_cast<const __nv_bfloat16*>(proprio),
      static_cast<const __nv_bfloat16*>(time),
      static_cast<__nv_bfloat16*>(output), rows, dim_action, dim_time,
      gripper0, gripper1);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_xvla_sigmoid_columns_bf16(
    const void* input, void* output, int rows, int columns, int index0,
    int index1, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || rows <= 0 || columns <= 0 ||
      index0 < 0 || index1 < 0 || index0 >= columns || index1 >= columns) {
    return cudaErrorInvalidValue;
  }
  const int64_t count = static_cast<int64_t>(rows) * columns;
  xvla_sigmoid_columns_bf16_kernel<<<blocks_for(count), kXvlaThreads, 0,
                                     stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_bfloat16*>(output), rows, columns, index0, index1);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_xvla_gather_rows_bf16(
    const void* input, const void* indices, void* output, int64_t out_rows,
    int columns, cudaStream_t stream) {
  if (input == nullptr || indices == nullptr || output == nullptr ||
      out_rows <= 0 || columns <= 0) {
    return cudaErrorInvalidValue;
  }
  const int64_t count = out_rows * columns;
  xvla_gather_rows_bf16_kernel<<<blocks_for(count), kXvlaThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const unsigned int*>(indices),
      static_cast<__nv_bfloat16*>(output), out_rows, columns);
  return cudaGetLastError();
}
