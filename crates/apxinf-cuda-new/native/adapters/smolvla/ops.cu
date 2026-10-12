// Copyright 2026 ApxInf contributors.
// SmolVLA FP16 kernel surface plus the BF16 pixel-shuffle, ported verbatim
// from the legacy apxinf-cuda adapters (static_bf16_adapter.cu,
// custom_kernels.cu, core_kernels_adapter.cu, cublas_adapter.cu) and their
// kernels/custom/*.cuh bodies. Symbols carry the apxinf_smolvla_ prefix
// because the legacy crate links into the same binary and its own launchers
// are already present. The copied kernels live in an anonymous namespace so
// their names cannot collide with identically named kernels elsewhere.

#include "../../include/apxinf_cuda/smolvla.h"
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

// ── normalization (kernels/custom/normalization.cuh) ────────────────────────

__global__ void rms_norm_f16_kernel(
    const half* input, const half* weight, half* output,
    uint32_t cols, uint32_t rows, float eps)
{
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    uint32_t tid = threadIdx.x;
    uint32_t offset = row * cols;
    extern __shared__ float x_buf[];
    __shared__ float s_sum;

    float partial = 0.0f;
    for (uint32_t i = tid; i < cols; i += blockDim.x) {
        float v = __half2float(input[offset + i]);
        x_buf[i] = v;
        partial += v * v;
    }
    for (int off = 16; off > 0; off >>= 1)
        partial += __shfl_xor_sync(0xffffffff, partial, off);
    __shared__ float warp_sums[32];
    uint32_t warp_id = tid / 32;
    uint32_t lane = tid % 32;
    if (lane == 0) warp_sums[warp_id] = partial;
    __syncthreads();
    if (warp_id == 0) {
        float v = (tid < (blockDim.x + 31) / 32) ? warp_sums[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_xor_sync(0xffffffff, v, off);
        if (lane == 0) s_sum = v;
    }
    __syncthreads();
    float rms = rsqrtf(s_sum / (float)cols + eps);
    for (uint32_t i = tid; i < cols; i += blockDim.x) {
        float w = __half2float(weight[i]);
        output[offset + i] = __float2half(x_buf[i] * rms * w);
    }
}

__global__ void layer_norm_f16_kernel(
    const half* input, const half* weight,
    const half* bias, half* output,
    int rows, int cols, float eps) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    sum += __half2float(input[static_cast<int64_t>(row) * cols + col]);
  const float mean = block_sum_parallel_unsafe(sum, scratch) / cols;
  float variance_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered =
        __half2float(input[static_cast<int64_t>(row) * cols + col]) - mean;
    variance_sum += centered * centered;
  }
  __syncthreads();
  const float inverse_std =
      rsqrtf(block_sum_parallel_unsafe(variance_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    const float value =
        (__half2float(input[index]) - mean) * inverse_std *
            __half2float(weight[col]) +
        __half2float(bias[col]);
    output[index] = __float2half(value);
  }
}

// ── activations (kernels/custom/activation.cuh) ─────────────────────────────

__device__ __forceinline__ half gelu_tanh_f16_one(half value)
{
    float x = __half2float(value);
    const float kBeta  = 0.7978845608028654f;
    const float kAlpha = 0.044715f;
    float inner = kBeta * (x + kAlpha * x * x * x);
    float y = 0.5f * x * (1.0f + tanhf(inner));
    return __float2half(y);
}

__global__ void bias_gelu_f16_kernel(
    const half* input, const half* bias, half* output,
    int rows, int cols) {
  const int64_t count = static_cast<int64_t>(rows) * cols;
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const int col = static_cast<int>(index % cols);
    output[index] = gelu_tanh_f16_one(
        __float2half(__half2float(input[index]) + __half2float(bias[col])));
  }
}

__global__ void bias_silu_f16_kernel(
    const half* input, const half* bias, half* output, int64_t count, int cols) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    float value = __half2float(input[index]);
    if (bias != nullptr) value += __half2float(bias[index % cols]);
    output[index] = __float2half(value / (1.0f + expf(-value)));
  }
}

__global__ void swiglu_f16_kernel(
    const half* gate_up, half* output, int rows, int inner) {
  const int64_t count = static_cast<int64_t>(rows) * inner;
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const int row = static_cast<int>(index / inner);
    const int col = static_cast<int>(index % inner);
    const float gate = __half2float(gate_up[static_cast<int64_t>(row) * 2 * inner + col]);
    const float up = __half2float(gate_up[static_cast<int64_t>(row) * 2 * inner + inner + col]);
    const float silu = gate / (1.0f + expf(-gate));
    output[index] = __float2half(silu * up);
  }
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

// ── rope (kernels/custom/rope.cuh) ──────────────────────────────────────────

__global__ void rope_half_split_f16_kernel(
    const half* input, half* output,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    float rope_theta, uint32_t pos_offset)
{
    uint32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head_idx = blockIdx.y;
    uint32_t seq_idx  = blockIdx.z;
    if (pair_idx >= head_dim / 2) return;

    uint32_t pos = seq_idx + pos_offset;
    float frequency = 1.0f / powf(
        rope_theta, 2.0f * static_cast<float>(pair_idx) /
                         static_cast<float>(head_dim));
    float angle = static_cast<float>(pos) * frequency;
    float sine;
    float cosine;
    sincosf(angle, &sine, &cosine);

    uint32_t base = seq_idx * n_heads * head_dim + head_idx * head_dim;
    uint32_t first = base + pair_idx;
    uint32_t second = base + head_dim / 2 + pair_idx;
    float x0 = __half2float(input[first]);
    float x1 = __half2float(input[second]);
    output[first] = __float2half(x0 * cosine - x1 * sine);
    output[second] = __float2half(x0 * sine + x1 * cosine);
}

// ── cuBLAS FP16 GQA (adapters/cublas_adapter.cu) ────────────────────────────

constexpr int kSoftmaxMaxCols = 1024;

thread_local cublasHandle_t g_mqa_blas = nullptr;
thread_local half* g_mqa_logits = nullptr;
thread_local size_t g_mqa_logits_bytes = 0;

cublasStatus_t initialize_mqa(size_t logits_bytes) {
  if (g_mqa_blas == nullptr) {
    cublasStatus_t status = cublasCreate(&g_mqa_blas);
    if (status != CUBLAS_STATUS_SUCCESS) return status;
  }
  if (logits_bytes > g_mqa_logits_bytes) {
    if (g_mqa_logits != nullptr) cudaFree(g_mqa_logits);
    cudaError_t cuda_status = cudaMalloc(&g_mqa_logits, logits_bytes);
    if (cuda_status != cudaSuccess) {
      g_mqa_logits = nullptr;
      g_mqa_logits_bytes = 0;
      return CUBLAS_STATUS_ALLOC_FAILED;
    }
    g_mqa_logits_bytes = logits_bytes;
  }
  return CUBLAS_STATUS_SUCCESS;
}

__global__ void gqa_softmax_f16_warp_kernel(
    half* data, int rows, int cols, int batches) {
  const int lane = threadIdx.x;
  const int row = blockIdx.x;
  const int batch = blockIdx.y;
  if (row >= rows || lane >= 32) return;
  half* source =
      data + (static_cast<int64_t>(batch) * rows + row) * cols;
  const int vectors = cols / 2;
  half2* source2 = reinterpret_cast<half2*>(source);
  float maximum = -1.0e30f;

  if ((cols & 1) == 0) {
#pragma unroll 4
    for (int vector = lane; vector < vectors; vector += 32) {
      const float2 pair = __half22float2(source2[vector]);
      maximum = fmaxf(maximum, fmaxf(pair.x, pair.y));
    }
  } else {
    for (int col = lane; col < cols; col += 32) {
      maximum = fmaxf(maximum, __half2float(source[col]));
    }
  }
  maximum = warp_max(maximum);

  float sum = 0.0f;
  if ((cols & 1) == 0) {
#pragma unroll 4
    for (int vector = lane; vector < vectors; vector += 32) {
      const float2 pair = __half22float2(source2[vector]);
      sum += __expf(pair.x - maximum) + __expf(pair.y - maximum);
    }
  } else {
    for (int col = lane; col < cols; col += 32) {
      sum += __expf(__half2float(source[col]) - maximum);
    }
  }
  sum = warp_sum_all(sum);
  const float inverse = 1.0f / sum;

  if ((cols & 1) == 0) {
#pragma unroll 4
    for (int vector = lane; vector < vectors; vector += 32) {
      const float2 pair = __half22float2(source2[vector]);
      source2[vector] = __floats2half2_rn(
          __expf(pair.x - maximum) * inverse,
          __expf(pair.y - maximum) * inverse);
    }
  } else {
    for (int col = lane; col < cols; col += 32) {
      source[col] =
          __float2half(__expf(__half2float(source[col]) - maximum) * inverse);
    }
  }
}

__global__ void gqa_softmax_f16_causal_warp_kernel(
    half* data, int rows, int cols, int batches, int offset) {
  const int lane = threadIdx.x;
  const int row = blockIdx.x;
  const int batch = blockIdx.y;
  if (row >= rows || lane >= 32) return;
  half* source = data + (static_cast<int64_t>(batch) * rows + row) * cols;
  const int valid = offset + row + 1;
  float maximum = -1.0e30f;
  for (int col = lane; col < cols; col += 32) {
    const float value = col < valid ? __half2float(source[col]) : -1.0e30f;
    maximum = fmaxf(maximum, value);
  }
  maximum = warp_max(maximum);

  float sum = 0.0f;
  for (int col = lane; col < cols; col += 32) {
    if (col < valid) {
      sum += __expf(__half2float(source[col]) - maximum);
    }
  }
  sum = warp_sum_all(sum);
  const float inverse = 1.0f / sum;
  for (int col = lane; col < cols; col += 32) {
    const float value =
        col < valid ? __expf(__half2float(source[col]) - maximum) * inverse : 0.0f;
    source[col] = __float2half(value);
  }
}

}  // namespace

// ── launchers ───────────────────────────────────────────────────────────────

extern "C" cudaError_t apxinf_smolvla_rms_norm_f16(
    const void* input, const void* weight, void* output,
    int rows, int cols, float eps, cudaStream_t stream) {
  if (input == nullptr || weight == nullptr || output == nullptr ||
      rows <= 0 || cols <= 0 || !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  const size_t shared_bytes = static_cast<size_t>(cols) * sizeof(float);
  rms_norm_f16_kernel<<<rows, kThreads, shared_bytes, stream>>>(
      static_cast<const half*>(input), static_cast<const half*>(weight),
      static_cast<half*>(output), cols, rows, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_smolvla_layer_norm_f16(
    const void* input, const void* weight, const void* bias, void* output,
    int rows, int cols, float eps, cudaStream_t stream) {
  if (input == nullptr || weight == nullptr || bias == nullptr ||
      output == nullptr || rows <= 0 || cols <= 0 || !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  layer_norm_f16_kernel<<<rows, kThreads, 0, stream>>>(
      static_cast<const half*>(input), static_cast<const half*>(weight),
      static_cast<const half*>(bias), static_cast<half*>(output),
      rows, cols, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_smolvla_swiglu_f16(
    const void* gate_up, void* output, int rows, int inner,
    cudaStream_t stream) {
  const int64_t count = static_cast<int64_t>(rows) * inner;
  if (gate_up == nullptr || output == nullptr || rows <= 0 || inner <= 0)
    return cudaErrorInvalidValue;
  swiglu_f16_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
      static_cast<const half*>(gate_up), static_cast<half*>(output),
      rows, inner);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_smolvla_bias_gelu_f16(
    const void* input, const void* bias, void* output,
    int rows, int cols, cudaStream_t stream) {
  if (input == nullptr || bias == nullptr || output == nullptr ||
      rows <= 0 || cols <= 0) {
    return cudaErrorInvalidValue;
  }
  const int64_t count = static_cast<int64_t>(rows) * cols;
  bias_gelu_f16_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
      static_cast<const half*>(input), static_cast<const half*>(bias),
      static_cast<half*>(output), rows, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_smolvla_bias_silu_f16(
    const void* input, const void* bias, void* output, int rows, int cols,
    cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
  int64_t count = static_cast<int64_t>(rows) * cols;
  bias_silu_f16_kernel<<<clamped_blocks(count), 256, 0, stream>>>(
      static_cast<const half*>(input), static_cast<const half*>(bias),
      static_cast<half*>(output), count, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_smolvla_bias_f16(
    const void* input, const void* bias, void* output, int rows, int cols,
    cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
  int64_t count = static_cast<int64_t>(rows) * cols;
  bias_f16_kernel<<<clamped_blocks(count), 256, 0, stream>>>(
      static_cast<const half*>(input), static_cast<const half*>(bias),
      static_cast<half*>(output), count, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_smolvla_euler_update_f16(
    const void* state, const void* velocity, void* output, int64_t count,
    float dt, cudaStream_t stream) {
  if (count <= 0) return cudaErrorInvalidValue;
  euler_update_f16_kernel<<<clamped_blocks(count), 256, 0, stream>>>(
      static_cast<const half*>(state), static_cast<const half*>(velocity),
      static_cast<half*>(output), count, dt);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_smolvla_add_f16(
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

extern "C" cudaError_t apxinf_smolvla_scale_f16(
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

extern "C" cudaError_t apxinf_smolvla_pixel_shuffle_4_bf16(
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

extern "C" cudaError_t apxinf_smolvla_pixel_shuffle_4_f16(
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

extern "C" cudaError_t apxinf_smolvla_rope_half_split_f16(
    const void* input, void* output,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    float rope_theta, uint32_t pos_offset, cudaStream_t stream) {
  dim3 grid((head_dim / 2 + kThreads - 1) / kThreads, n_heads, seq_len);
  dim3 block(kThreads, 1, 1);
  rope_half_split_f16_kernel<<<grid, block, 0, stream>>>(
      (const half*)input, (half*)output,
      head_dim, n_heads, seq_len, rope_theta, pos_offset);
  return cudaGetLastError();
}

extern "C" int apxinf_smolvla_cublas_gqa_f16(
    const void* q, const void* k, const void* v, void* output,
    int query_tokens, int key_tokens, int q_heads, int kv_heads,
    int head_dim, cudaStream_t stream) {
  if (q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
      query_tokens <= 0 || key_tokens <= 0 || key_tokens > kSoftmaxMaxCols ||
      q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0 ||
      head_dim <= 0) {
    return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  }
  const int group = q_heads / kv_heads;
  const int64_t score_stride =
      static_cast<int64_t>(query_tokens) * key_tokens;
  const size_t logits_bytes =
      static_cast<size_t>(q_heads) * score_stride * sizeof(half);
  cublasStatus_t status = initialize_mqa(logits_bytes);
  if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  status = cublasSetStream(g_mqa_blas, stream);
  if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);

  const float attention_scale = rsqrtf(static_cast<float>(head_dim));
  const float zero = 0.0f;
  const float one = 1.0f;
  for (int kv_head = 0; kv_head < kv_heads; ++kv_head) {
    const half* query = static_cast<const half*>(q) +
                        kv_head * group * head_dim;
    const half* key = static_cast<const half*>(k) + kv_head * head_dim;
    half* scores = static_cast<half*>(g_mqa_logits) +
                   static_cast<int64_t>(kv_head) * group * score_stride;
    status = cublasGemmStridedBatchedEx(
        g_mqa_blas, CUBLAS_OP_T, CUBLAS_OP_N,
        key_tokens, query_tokens, head_dim, &attention_scale,
        key, CUDA_R_16F, kv_heads * head_dim, 0,
        query, CUDA_R_16F, q_heads * head_dim, head_dim,
        &zero, scores, CUDA_R_16F, key_tokens, score_stride,
        group, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  }

  gqa_softmax_f16_warp_kernel<<<dim3(query_tokens, q_heads), 32, 0, stream>>>(
      g_mqa_logits, query_tokens, key_tokens, q_heads);
  if (cudaPeekAtLastError() != cudaSuccess) {
    return static_cast<int>(CUBLAS_STATUS_EXECUTION_FAILED);
  }

  for (int kv_head = 0; kv_head < kv_heads; ++kv_head) {
    const half* value = static_cast<const half*>(v) + kv_head * head_dim;
    const half* scores = static_cast<const half*>(g_mqa_logits) +
                         static_cast<int64_t>(kv_head) * group * score_stride;
    half* destination = static_cast<half*>(output) +
                        kv_head * group * head_dim;
    status = cublasGemmStridedBatchedEx(
        g_mqa_blas, CUBLAS_OP_N, CUBLAS_OP_N,
        head_dim, query_tokens, key_tokens, &one,
        value, CUDA_R_16F, kv_heads * head_dim, 0,
        scores, CUDA_R_16F, key_tokens, score_stride,
        &zero, destination, CUDA_R_16F, q_heads * head_dim, head_dim,
        group, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  }
  return static_cast<int>(status);
}

extern "C" int apxinf_smolvla_cublas_gqa_causal_f16(
    const void* q, const void* k, const void* v, void* output,
    int query_tokens, int key_tokens, int q_heads, int kv_heads,
    int head_dim, int key_offset, cudaStream_t stream) {
  if (q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
      query_tokens <= 0 || key_tokens <= 0 || key_tokens > kSoftmaxMaxCols ||
      q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0 ||
      head_dim <= 0 || key_offset < 0 || query_tokens + key_offset > key_tokens) {
    return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  }
  const int group = q_heads / kv_heads;
  const int64_t score_stride = static_cast<int64_t>(query_tokens) * key_tokens;
  const size_t logits_bytes = static_cast<size_t>(q_heads) * score_stride * sizeof(half);
  cublasStatus_t status = initialize_mqa(logits_bytes);
  if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  status = cublasSetStream(g_mqa_blas, stream);
  if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);

  const float attention_scale = rsqrtf(static_cast<float>(head_dim));
  const float zero = 0.0f;
  const float one = 1.0f;
  for (int kv_head = 0; kv_head < kv_heads; ++kv_head) {
    const half* query = static_cast<const half*>(q) + kv_head * group * head_dim;
    const half* key = static_cast<const half*>(k) + kv_head * head_dim;
    half* scores = static_cast<half*>(g_mqa_logits) +
                   static_cast<int64_t>(kv_head) * group * score_stride;
    status = cublasGemmStridedBatchedEx(
        g_mqa_blas, CUBLAS_OP_T, CUBLAS_OP_N,
        key_tokens, query_tokens, head_dim, &attention_scale,
        key, CUDA_R_16F, kv_heads * head_dim, 0,
        query, CUDA_R_16F, q_heads * head_dim, head_dim,
        &zero, scores, CUDA_R_16F, key_tokens, score_stride,
        group, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  }

  gqa_softmax_f16_causal_warp_kernel<<<dim3(query_tokens, q_heads), 32, 0, stream>>>(
      g_mqa_logits, query_tokens, key_tokens, q_heads, key_offset);
  if (cudaPeekAtLastError() != cudaSuccess) {
    return static_cast<int>(CUBLAS_STATUS_EXECUTION_FAILED);
  }

  for (int kv_head = 0; kv_head < kv_heads; ++kv_head) {
    const half* value = static_cast<const half*>(v) + kv_head * head_dim;
    half* scores = static_cast<half*>(g_mqa_logits) +
                   static_cast<int64_t>(kv_head) * group * score_stride;
    half* destination = static_cast<half*>(output) + kv_head * group * head_dim;
    status = cublasGemmStridedBatchedEx(
        g_mqa_blas, CUBLAS_OP_N, CUBLAS_OP_N,
        head_dim, query_tokens, key_tokens, &one,
        value, CUDA_R_16F, kv_heads * head_dim, 0,
        scores, CUDA_R_16F, key_tokens, score_stride,
        &zero, destination, CUDA_R_16F, q_heads * head_dim, head_dim,
        group, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  }
  return static_cast<int>(status);
}
