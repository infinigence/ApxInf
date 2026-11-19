#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace apxinf::conv2d::kernels {

template <class T>
__device__ __forceinline__ float to_float(T value);
template <>
__device__ __forceinline__ float to_float<float>(float value) { return value; }
template <>
__device__ __forceinline__ float to_float(__half value) { return __half2float(value); }
template <>
__device__ __forceinline__ float to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <class T>
__device__ __forceinline__ T from_float(float value);
template <>
__device__ __forceinline__ float from_float<float>(float value) { return value; }
template <>
__device__ __forceinline__ __half from_float<__half>(float value) {
  return __float2half(value);
}
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16(value);
}

/* Channel-last direct convolution. One thread per output element, FP32
 * accumulation. groups == 1, dilation == 1.
 * input [B, H, W, Cin], weight [Cout, Cin, KH, KW], bias [Cout],
 * output [B, Hout, Wout, Cout]. */
template <class T>
__global__ void conv2d_forward(const T* __restrict__ input,
                               const T* __restrict__ weight,
                               const T* __restrict__ bias,
                               T* __restrict__ output, int64_t batch,
                               int64_t in_h, int64_t in_w, int64_t in_channels,
                               int64_t out_channels, int64_t kernel_h,
                               int64_t kernel_w, int64_t stride_h,
                               int64_t stride_w, int64_t pad_h, int64_t pad_w,
                               int64_t out_h, int64_t out_w) {
  const int64_t total = batch * out_h * out_w * out_channels;
  const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
       index < total; index += step) {
    const int64_t co = index % out_channels;
    int64_t rem = index / out_channels;
    const int64_t ow = rem % out_w;
    rem /= out_w;
    const int64_t oh = rem % out_h;
    const int64_t b = rem / out_h;

    float acc = bias != nullptr ? to_float<T>(bias[co]) : 0.0f;
    const T* w = weight + co * in_channels * kernel_h * kernel_w;
    const T* in_batch = input + b * in_h * in_w * in_channels;
    for (int64_t ci = 0; ci < in_channels; ++ci) {
      for (int64_t kh = 0; kh < kernel_h; ++kh) {
        const int64_t ih = oh * stride_h - pad_h + kh;
        if (ih < 0 || ih >= in_h) continue;
        for (int64_t kw = 0; kw < kernel_w; ++kw) {
          const int64_t iw = ow * stride_w - pad_w + kw;
          if (iw >= 0 && iw < in_w) {
            acc += to_float<T>(in_batch[(ih * in_w + iw) * in_channels + ci]) *
                   to_float<T>(w[(ci * kernel_h + kh) * kernel_w + kw]);
          }
        }
      }
    }
    output[index] = from_float<T>(acc);
  }
}

/* im2col for the cuBLASLt fast path. Expands the padded input into the
 * row-major matrix COL[rows, K], rows = B*OH*OW, K = Cin*KH*KW, with
 *   COL[((b*OH + oh)*OW + ow), ci*KH*KW + kh*KW + kw] =
 *       input[((b*IH + oh*stride_h - pad_h + kh)*IW + ow*stride_w - pad_w + kw)*Cin + ci]
 * and zero when the sampled coordinate falls outside the input (padding).
 * The k index order matches the innermost dims of weight [Cout, Cin, KH, KW]
 * so the following GEMM consumes the weight exactly as stored. One thread
 * per element with a grid-stride loop; values are copied without any type
 * conversion (im2col is pure data movement, the GEMM accumulates in FP32). */
template <class T>
__global__ void im2col_kernel(const T* __restrict__ input,
                              T* __restrict__ col, int64_t batch,
                              int64_t in_h, int64_t in_w, int64_t in_channels,
                              int64_t out_h, int64_t out_w, int64_t kernel_h,
                              int64_t kernel_w, int64_t stride_h,
                              int64_t stride_w, int64_t pad_h, int64_t pad_w) {
  const int64_t rows = batch * out_h * out_w;
  const int64_t cols = in_channels * kernel_h * kernel_w;
  const int64_t total = rows * cols;
  const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
       index < total; index += step) {
    const int64_t col_index = index % cols;
    int64_t row = index / cols;
    const int64_t ow = row % out_w;
    row /= out_w;
    const int64_t oh = row % out_h;
    const int64_t b = row / out_h;
    const int64_t kw = col_index % kernel_w;
    int64_t rem = col_index / kernel_w;
    const int64_t kh = rem % kernel_h;
    const int64_t ci = rem / kernel_h;
    const int64_t ih = oh * stride_h - pad_h + kh;
    const int64_t iw = ow * stride_w - pad_w + kw;
    if (ih < 0 || ih >= in_h || iw < 0 || iw >= in_w) {
      col[index] = from_float<T>(0.0f);
    } else {
      col[index] = input[((b * in_h + ih) * in_w + iw) * in_channels + ci];
    }
  }
}

}  // namespace apxinf::conv2d::kernels
