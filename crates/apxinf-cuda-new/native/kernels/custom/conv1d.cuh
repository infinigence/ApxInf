// Copyright 2026 ApxInf contributors.
// FP32 one-dimensional convolution family used by waveform decoders: conv1d (zero or replicate padding, stride, dilation, dense or
// depthwise), conv_transpose1d (optional replicate input padding, output crop and scale) and the SnakeBeta activation.
// Layout is contiguous [batch, channels, time]. Accumulation is FP32 in the order (input channel, tap); the kernels are exact FP32 (no
// TF32, no fast math), so results agree with a strict-FP32 reference up to summation order.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace apxinf::conv1d {

struct Geometry {
  int batch, in_channels, out_channels, in_len, out_len;
  int kernel, stride, dilation;
  int pad_left;    // conv1d: left padding; conv_transpose1d: replicate pad on both sides of the input
  int crop_left;   // conv_transpose1d: columns removed from the left of the full transposed output
  int groups;      // 1 or in_channels == out_channels (depthwise)
  int replicate;   // conv1d: 1 pads with the edge value, 0 with zeros
  float scale;     // conv_transpose1d: output multiplier
};

__device__ __forceinline__ int source_index(int t, int tap, const Geometry& g) {
  int s = t * g.stride + tap * g.dilation - g.pad_left;
  if (s < 0 || s >= g.in_len) {
    if (!g.replicate) return -1;
    s = s < 0 ? 0 : g.in_len - 1;
  }
  return s;
}

// Dense convolution as an implicit GEMM: out[co, t] = bias[co] + sum_r w[co, r] * x[r -> (ci, tap)][t], r = ci * K + tap.
constexpr int kBm = 64, kBn = 64, kBk = 16, kTm = 4, kTn = 4;

__global__ void __launch_bounds__(256) conv1d_dense_kernel(
    const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ bias, float* __restrict__ out,
    Geometry g) {
  __shared__ float ws[kBk][kBm + 1];
  __shared__ float xs[kBk][kBn];
  const int t0 = blockIdx.x * kBn, co0 = blockIdx.y * kBm, b = blockIdx.z;
  const int tid = threadIdx.y * 16 + threadIdx.x;
  const int reduce = g.in_channels * g.kernel;
  const float* xb = x + (int64_t)b * g.in_channels * g.in_len;
  float acc[kTm][kTn] = {};
  for (int r0 = 0; r0 < reduce; r0 += kBk) {
    for (int i = tid; i < kBm * kBk; i += 256) {
      const int m = i / kBk, kk = i % kBk;
      const int co = co0 + m, r = r0 + kk;
      ws[kk][m] = (co < g.out_channels && r < reduce) ? w[(int64_t)co * reduce + r] : 0.0f;
    }
    for (int i = tid; i < kBk * kBn; i += 256) {
      const int kk = i / kBn, n = i % kBn;
      const int r = r0 + kk, t = t0 + n;
      float v = 0.0f;
      if (r < reduce && t < g.out_len) {
        const int ci = r / g.kernel, tap = r % g.kernel;
        const int s = source_index(t, tap, g);
        if (s >= 0) v = xb[(int64_t)ci * g.in_len + s];
      }
      xs[kk][n] = v;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < kBk; ++kk) {
      float a[kTm], c[kTn];
#pragma unroll
      for (int i = 0; i < kTm; ++i) a[i] = ws[kk][threadIdx.y * kTm + i];
#pragma unroll
      for (int j = 0; j < kTn; ++j) c[j] = xs[kk][threadIdx.x * kTn + j];
#pragma unroll
      for (int i = 0; i < kTm; ++i)
#pragma unroll
        for (int j = 0; j < kTn; ++j) acc[i][j] = fmaf(a[i], c[j], acc[i][j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < kTm; ++i) {
    const int co = co0 + threadIdx.y * kTm + i;
    if (co >= g.out_channels) continue;
    const float bv = bias ? bias[co] : 0.0f;
#pragma unroll
    for (int j = 0; j < kTn; ++j) {
      const int t = t0 + threadIdx.x * kTn + j;
      if (t < g.out_len) out[((int64_t)b * g.out_channels + co) * g.out_len + t] = acc[i][j] + bv;
    }
  }
}

// Depthwise convolution: one thread per output element.
__global__ void conv1d_depthwise_kernel(
    const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ bias, float* __restrict__ out,
    Geometry g) {
  const int64_t total = (int64_t)g.batch * g.out_channels * g.out_len;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (int64_t)gridDim.x * blockDim.x) {
    const int t = (int)(i % g.out_len);
    const int c = (int)((i / g.out_len) % g.out_channels);
    const float* xc = x + ((i / g.out_len)) * g.in_len;
    float acc = bias ? bias[c] : 0.0f;
    for (int tap = 0; tap < g.kernel; ++tap) {
      const int s = source_index(t, tap, g);
      if (s >= 0) acc = fmaf(w[(int64_t)c * g.kernel + tap], xc[s], acc);
    }
    out[i] = acc;
  }
}

// Transposed convolution, gather form. The (virtual) input is the replicate-padded signal of length in_len + 2 * pad_left; output element
// j of the cropped result is full-output column tf = j + crop_left, fed by taps k with (tf - k) divisible by the stride.
// Weight layout is [in_channels, out_channels / groups, kernel].
__global__ void conv_transpose1d_kernel(
    const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ bias, float* __restrict__ out,
    Geometry g) {
  const int64_t total = (int64_t)g.batch * g.out_channels * g.out_len;
  const int padded = g.in_len + 2 * g.pad_left;
  const int per_group_in = g.in_channels / g.groups, per_group_out = g.out_channels / g.groups;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (int64_t)gridDim.x * blockDim.x) {
    const int j = (int)(i % g.out_len);
    const int co = (int)((i / g.out_len) % g.out_channels);
    const int b = (int)(i / ((int64_t)g.out_len * g.out_channels));
    const int group = co / per_group_out;
    const int tf = j + g.crop_left;
    const int k0 = ((tf % g.stride) + g.stride) % g.stride;
    float acc = 0.0f;
    for (int k = k0; k < g.kernel; k += g.stride) {
      int p = (tf - k) / g.stride;  // index into the padded input
      if (p < 0 || p >= padded) continue;
      int s = p - g.pad_left;
      s = s < 0 ? 0 : (s >= g.in_len ? g.in_len - 1 : s);
      for (int cg = 0; cg < per_group_in; ++cg) {
        const int ci = group * per_group_in + cg;
        acc = fmaf(x[((int64_t)b * g.in_channels + ci) * g.in_len + s],
                   w[((int64_t)ci * per_group_out + (co - group * per_group_out)) * g.kernel + k], acc);
      }
    }
    out[i] = acc * g.scale + (bias ? bias[co] : 0.0f);
  }
}

// SnakeBeta: y = x + sin(alpha_c * x)^2 / (beta_c + 1e-9), alpha and beta already exponentiated. y may alias x exactly.
__global__ void snake_beta_kernel(
    const float* __restrict__ x, const float* __restrict__ alpha, const float* __restrict__ beta, float* out,
    int channels, int len, int64_t total) {
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (int64_t)gridDim.x * blockDim.x) {
    const int c = (int)((i / len) % channels);
    const float v = x[i];
    const float s = sinf(alpha[c] * v);
    out[i] = fmaf(s * s, 1.0f / (beta[c] + 1e-9f), v);
  }
}

}  // namespace apxinf::conv1d
