#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace apxinf::conv1d::kernels {

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

/* Channel-last direct convolution. One thread per output element.
 * input [B, L, Cin], weight [Cout, Cin/groups, K], bias [Cout],
 * output [B, Lout, Cout]. */
template <class T>
__global__ void conv1d_forward(const T* __restrict__ input,
                               const T* __restrict__ weight,
                               const T* __restrict__ bias,
                               T* __restrict__ output, int64_t batch,
                               int64_t in_length, int64_t in_channels,
                               int64_t out_channels, int64_t kernel,
                               int64_t stride, int64_t padding,
                               int64_t dilation, int64_t groups,
                               int64_t out_length) {
  const int64_t total = batch * out_length * out_channels;
  const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
       index < total; index += step) {
    const int64_t co = index % out_channels;
    const int64_t rem = index / out_channels;
    const int64_t to = rem % out_length;
    const int64_t b = rem / out_length;

    const int64_t out_per_group = out_channels / groups;
    const int64_t in_per_group = in_channels / groups;
    const int64_t group = co / out_per_group;

    float acc = bias != nullptr ? to_float<T>(bias[co]) : 0.0f;
    const T* w = weight + co * in_per_group * kernel;
    const T* in_batch = input + b * in_length * in_channels;
    for (int64_t ci = 0; ci < in_per_group; ++ci) {
      const int64_t ic = group * in_per_group + ci;
      const T* in_channel = in_batch + ic;
      for (int64_t k = 0; k < kernel; ++k) {
        const int64_t li = to * stride - padding + k * dilation;
        if (li >= 0 && li < in_length) {
          acc += to_float<T>(in_channel[li * in_channels]) *
                 to_float<T>(w[ci * kernel + k]);
        }
      }
    }
    output[index] = from_float<T>(acc);
  }
}

// Channel-first storage gives adjacent lanes consecutive time samples. Each
// output retains the direct kernel's channel/kernel accumulation order.
template <class T>
__global__ void conv1d_forward_temporal(const T* __restrict__ input,
                               const T* __restrict__ weight,
                               const T* __restrict__ bias,
                               T* __restrict__ output, int64_t batch,
                               int64_t in_length, int64_t in_channels,
                               int64_t out_channels, int64_t kernel,
                               int64_t stride, int64_t padding,
                               int64_t dilation, int64_t groups,
                               int64_t out_length) {
  const int64_t total = batch * out_length * out_channels;
  const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
       index < total; index += step) {
    const int64_t to = index % out_length;
    const int64_t co = (index / out_length) % out_channels;
    const int64_t b = index / (out_length * out_channels);

    const int64_t out_per_group = out_channels / groups;
    const int64_t in_per_group = in_channels / groups;
    const int64_t group = co / out_per_group;

    float acc = bias != nullptr ? to_float<T>(bias[co]) : 0.0f;
    const T* w = weight + co * in_per_group * kernel;
    const T* in_batch = input + b * in_length * in_channels;
    for (int64_t ci = 0; ci < in_per_group; ++ci) {
      const int64_t ic = group * in_per_group + ci;
      const T* in_channel = in_batch + ic * in_length;
      for (int64_t k = 0; k < kernel; ++k) {
        const int64_t li = to * stride - padding + k * dilation;
        if (li >= 0 && li < in_length) {
          acc += to_float<T>(in_channel[li]) *
                 to_float<T>(w[ci * kernel + k]);
        }
      }
    }
    output[index] = from_float<T>(acc);
  }
}

// Each lane evaluates two consecutive warp time tiles. Their independent
// accumulators expose load/compute overlap without reordering either sum.
template <class T, int FixedKernel>
__global__ void conv1d_forward_temporal_pair(const T* __restrict__ input,
                               const T* __restrict__ weight,
                               const T* __restrict__ bias,
                               T* __restrict__ output, int64_t batch,
                               int64_t in_length, int64_t in_channels,
                               int64_t out_channels, int64_t kernel,
                               int64_t stride, int64_t padding,
                               int64_t dilation, int64_t groups,
                               int64_t out_length) {
  const int64_t tiles = (out_length + 63) / 64;
  const int64_t total_tiles = batch * out_channels * tiles;
  const int64_t tile_step = static_cast<int64_t>(gridDim.x) * (blockDim.x / 32);
  for (int64_t tile = static_cast<int64_t>(blockIdx.x) * (blockDim.x / 32) +
           threadIdx.x / 32;
       tile < total_tiles; tile += tile_step) {
    const int64_t to = (tile % tiles) * 64 + threadIdx.x % 32;
    if (to >= out_length) continue;
    const int64_t co = (tile / tiles) % out_channels;
    const int64_t b = tile / (tiles * out_channels);
    const int64_t index = (b * out_channels + co) * out_length + to;
    const int64_t out_per_group = out_channels / groups;
    const int64_t in_per_group = in_channels / groups;
    const int64_t group = co / out_per_group;

    float acc = bias != nullptr ? to_float<T>(bias[co]) : 0.0f;
    const int64_t to2 = to + 32;
    const bool second_valid = to2 < out_length;
    float acc2 = acc;
    const T* w = weight + co * in_per_group * kernel;
    const T* in_batch = input + b * in_length * in_channels;
    const int64_t first = to * stride - padding;
    const int64_t first2 = to2 * stride - padding;
    const bool interior = first >= 0 &&
        first + (FixedKernel - 1) * dilation < in_length;
    for (int64_t ci = 0; ci < in_per_group; ++ci) {
      const int64_t ic = group * in_per_group + ci;
      const T* in_channel = in_batch + ic * in_length;
      #pragma unroll
      for (int k = 0; k < FixedKernel; ++k) {
        const int64_t li = to * stride - padding + k * dilation;
        if (interior || (li >= 0 && li < in_length)) {
          acc += to_float<T>(in_channel[li]) *
                 to_float<T>(w[ci * kernel + k]);
        }
        const int64_t li2 = first2 + k * dilation;
        if (second_valid && li2 >= 0 && li2 < in_length) {
          acc2 += to_float<T>(in_channel[li2]) *
                  to_float<T>(w[ci * kernel + k]);
        }
      }
    }
    output[index] = from_float<T>(acc);
    if (second_valid) output[index + 32] = from_float<T>(acc2);
  }
}


/* Channel-last transposed convolution. One thread per output element.
 * input [B, L, Cin], weight [Cin, Cout/groups, K], bias [Cout],
 * output [B, Lout, Cout]. */
template <class T>
__global__ void conv_transpose1d_forward(
    const T* __restrict__ input, const T* __restrict__ weight,
    const T* __restrict__ bias, T* __restrict__ output, int64_t batch,
    int64_t in_length, int64_t in_channels, int64_t out_channels,
    int64_t kernel, int64_t stride, int64_t padding, int64_t dilation,
    int64_t groups, int64_t out_length) {
  const int64_t total = batch * out_length * out_channels;
  const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
       index < total; index += step) {
    const int64_t co = index % out_channels;
    const int64_t rem = index / out_channels;
    const int64_t to = rem % out_length;
    const int64_t b = rem / out_length;

    const int64_t out_per_group = out_channels / groups;
    const int64_t in_per_group = in_channels / groups;
    const int64_t group = co / out_per_group;
    const int64_t co_local = co % out_per_group;

    float acc = bias != nullptr ? to_float<T>(bias[co]) : 0.0f;
    const T* in_batch = input + b * in_length * in_channels;
    for (int64_t ci_local = 0; ci_local < in_per_group; ++ci_local) {
      const int64_t ic = group * in_per_group + ci_local;
      const T* w = weight + (ic * out_per_group + co_local) * kernel;
      const T* in_channel = in_batch + ic;
      for (int64_t k = 0; k < kernel; ++k) {
        const int64_t numerator = to + padding - k * dilation;
        if (numerator >= 0 && numerator % stride == 0) {
          const int64_t li = numerator / stride;
          if (li < in_length) {
            acc += to_float<T>(in_channel[li * in_channels]) *
                   to_float<T>(w[k]);
          }
        }
      }
    }
    output[index] = from_float<T>(acc);
  }
}

/* Swap the last two axes of a `[batch, rows, cols]` buffer in place-equivalent
 * fashion, optionally adding a per-`row` bias on the way out. The cuDNN
 * provider needs temporal-major operands while the model stores channel-last,
 * so both the input staging and the result copy run through this tile
 * transpose. */
template <class T, bool AddBias>
__global__ void transpose_layout(const T* input, const T* bias, T* output,
                                 int64_t rows, int64_t cols, int64_t batch) {
  __shared__ T tile[32][33];
  const int64_t row_tiles = (rows + 31) / 32;
  const int64_t col_tiles = (cols + 31) / 32;
  const int64_t tiles_per_batch = row_tiles * col_tiles;
  for (int64_t tile_id = blockIdx.x; tile_id < batch * tiles_per_batch;
       tile_id += gridDim.x) {
    const int64_t batch_id = tile_id / tiles_per_batch;
    const int64_t local = tile_id % tiles_per_batch;
    const int64_t row_base = local / col_tiles * 32;
    const int64_t col_base = local % col_tiles * 32;
    for (int j = 0; j < 32; j += 8) {
      const int64_t row = row_base + threadIdx.y + j;
      const int64_t col = col_base + threadIdx.x;
      if (row < rows && col < cols)
        tile[threadIdx.y + j][threadIdx.x] =
            input[(batch_id * rows + row) * cols + col];
    }
    __syncthreads();
    for (int j = 0; j < 32; j += 8) {
      const int64_t row = row_base + threadIdx.x;
      const int64_t col = col_base + threadIdx.y + j;
      if (row < rows && col < cols) {
        T value = tile[threadIdx.x][threadIdx.y + j];
        if constexpr (AddBias) {
          if (bias) value = from_float<T>(to_float<T>(value) + to_float<T>(bias[row]));
        }
        output[(batch_id * cols + col) * rows + row] = value;
      }
    }
    __syncthreads();
  }
}

/* A channel-last filter descriptor indexes `[K, 1, kernel, C]` while the model
 * stores `[K, C, 1, kernel]`; only the storage order changes, so the same
 * permutation serves both semantics. Run once per prepared execution. */
template <class T>
__global__ void permute_filter_krsc(const T* source, T* destination,
                                    int64_t out_channels, int64_t in_channels,
                                    int64_t kernel) {
  const int64_t filter = in_channels * kernel;
  const int64_t total = out_channels * filter;
  for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total; index += int64_t(blockDim.x) * gridDim.x) {
    const int64_t filter_index = index / filter;
    const int64_t rest = index % filter;
    const int64_t tap = rest / in_channels;
    const int64_t channel = rest % in_channels;
    destination[index] = source[filter_index * filter + channel * kernel + tap];
  }
}

}  // namespace apxinf::conv1d::kernels
