// Copyright 2026 ApxInf contributors.
// Fixed implementation for this operator family; ported without numerical changes.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace {
// Fourier waypoint features: per channel, cat(sin(w*f*2pi), cos(w*f*2pi)).
__global__ void fourier_features_bf16_kernel(
    const __nv_bfloat16* waypoints, const __nv_bfloat16* freqs, __nv_bfloat16* out,
    int point_dim, int num_features) {
  const int token = blockIdx.x;
  const int width = point_dim * num_features * 2;
  const int64_t base = static_cast<int64_t>(token) * width;
  for (int idx = threadIdx.x; idx < point_dim * num_features; idx += blockDim.x) {
    const int c = idx / num_features;
    const int f = idx - c * num_features;
    const float w = __bfloat162float(waypoints[static_cast<int64_t>(token) * point_dim + c]);
    const float freq = __bfloat162float(freqs[f]);
    const float angle = (w * freq) * 6.2831855f;
    out[base + c * 2 * num_features + f] = __float2bfloat16(sinf(angle));
    out[base + c * 2 * num_features + num_features + f] = __float2bfloat16(cosf(angle));
  }
}

__global__ void sinusoidal_embedding_bf16_kernel(const float* positions,__nv_bfloat16* out,
    int dim,float scale,float frequency_step) {
  const int row=blockIdx.x,half=dim/2;
  for(int i=threadIdx.x;i<half;i+=blockDim.x) {
    const float frequency=expf(__fmul_rn(-frequency_step,static_cast<float>(i)));
    const float angle=__fmul_rn(__fmul_rn(scale,positions[row]),frequency);
    const int64_t base=static_cast<int64_t>(row)*dim;
    out[base+i]=__float2bfloat16(sinf(angle));
    out[base+half+i]=__float2bfloat16(cosf(angle));
  }
}


}  // namespace
