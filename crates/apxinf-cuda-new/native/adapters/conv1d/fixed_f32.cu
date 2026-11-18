// Copyright 2026 ApxInf contributors.
// Fixed family launchers of the FP32 1-D convolution operators; kernel bodies live in kernels/custom/conv1d.cuh.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "../../kernels/custom/conv1d.cuh"

namespace {
using apxinf::conv1d::Geometry;

bool valid(const Geometry& g) {
  return g.batch > 0 && g.in_channels > 0 && g.out_channels > 0 && g.in_len > 0 && g.out_len > 0 && g.kernel > 0 &&
         g.stride > 0 && g.dilation > 0 && g.pad_left >= 0 && g.crop_left >= 0 && g.groups > 0 &&
         g.in_channels % g.groups == 0 && g.out_channels % g.groups == 0 && g.replicate >= 0 && g.replicate <= 1;
}

int grid_for(int64_t total) { return (int)std::min<int64_t>((total + 255) / 256, 65535); }
}  // namespace

extern "C" cudaError_t apxinf_conv1d_f32(const float* x, const float* w, const float* bias, float* out,
                                         const Geometry* geometry, cudaStream_t stream) {
  if (!x || !w || !out || !geometry) return cudaErrorInvalidValue;
  const Geometry g = *geometry;
  const bool depthwise = g.groups == g.in_channels && g.in_channels == g.out_channels;
  if (!valid(g) || (g.groups != 1 && !depthwise)) return cudaErrorInvalidValue;
  if (g.groups == 1) {
    dim3 grid((g.out_len + apxinf::conv1d::kBn - 1) / apxinf::conv1d::kBn,
              (g.out_channels + apxinf::conv1d::kBm - 1) / apxinf::conv1d::kBm, g.batch);
    if (grid.y > 65535 || grid.z > 65535) return cudaErrorInvalidValue;
    apxinf::conv1d::conv1d_dense_kernel<<<grid, dim3(16, 16), 0, stream>>>(x, w, bias, out, g);
  } else {
    apxinf::conv1d::conv1d_depthwise_kernel<<<grid_for((int64_t)g.batch * g.out_channels * g.out_len), 256, 0, stream>>>(
        x, w, bias, out, g);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_conv_transpose1d_f32(const float* x, const float* w, const float* bias, float* out,
                                                   const Geometry* geometry, cudaStream_t stream) {
  if (!x || !w || !out || !geometry) return cudaErrorInvalidValue;
  const Geometry g = *geometry;
  if (!valid(g)) return cudaErrorInvalidValue;
  apxinf::conv1d::conv_transpose1d_kernel<<<grid_for((int64_t)g.batch * g.out_channels * g.out_len), 256, 0, stream>>>(
      x, w, bias, out, g);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_snake_beta_f32(const float* x, const float* alpha, const float* beta, float* out,
                                             int channels, int len, int64_t total, cudaStream_t stream) {
  if (!x || !alpha || !beta || !out || channels <= 0 || len <= 0 || total <= 0) return cudaErrorInvalidValue;
  apxinf::conv1d::snake_beta_kernel<<<grid_for(total), 256, 0, stream>>>(x, alpha, beta, out, channels, len, total);
  return cudaGetLastError();
}
