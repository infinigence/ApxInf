// Copyright 2026 apxinf contributors.
// Stable C ABI and CUDA launch policy for static-path BF16 operators.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {
#include "../kernels/custom/math.cuh"
#include "../kernels/custom/reduction.cuh"
#include "../kernels/custom/preprocess.cuh"
#include "../kernels/custom/activation.cuh"
#include "../kernels/custom/embedding.cuh"
#include "../kernels/custom/elementwise.cuh"
#include "../kernels/custom/normalization.cuh"
#include "../kernels/custom/fused.cuh"
#include "../kernels/custom/attention.cuh"

int blocks_for(int64_t count) {
  return static_cast<int>((count + kThreads - 1) / kThreads);
}
}  // namespace

extern "C" cudaError_t apxinf_static_rgb_u8_to_patches_bf16(
    const void* images, void* patches, int views, int image_size,
    int patch_size, int layout, cudaStream_t stream) {
  if (views <= 0 || image_size <= 0 || patch_size <= 0 ||
      image_size % patch_size != 0 || (layout != 0 && layout != 1))
    return cudaErrorInvalidValue;
  const int64_t count = static_cast<int64_t>(views) * 3 * image_size * image_size;
  if (layout == 0) {
    rgb_u8_to_patches_bf16_kernel<true><<<blocks_for(count), kThreads, 0, stream>>>(
        static_cast<const uint8_t*>(images), static_cast<__nv_bfloat16*>(patches),
        views, image_size, patch_size);
  } else {
    rgb_u8_to_patches_bf16_kernel<false><<<blocks_for(count), kThreads, 0, stream>>>(
        static_cast<const uint8_t*>(images), static_cast<__nv_bfloat16*>(patches),
        views, image_size, patch_size);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_activation_bf16(
    const void* input, const void* bias, void* output,
    int rows, int cols, int activation, cudaStream_t stream) {
  const int64_t count = static_cast<int64_t>(rows) * cols;
  const bool packed4 = cols % 4 == 0 &&
      reinterpret_cast<uintptr_t>(input) % alignof(Bf16x4) == 0 &&
      reinterpret_cast<uintptr_t>(output) % alignof(Bf16x4) == 0 &&
      (bias == nullptr ||
       reinterpret_cast<uintptr_t>(bias) % alignof(Bf16x4) == 0);
  const bool packed2 = cols % 2 == 0 &&
      reinterpret_cast<uintptr_t>(input) % alignof(__nv_bfloat162) == 0 &&
      reinterpret_cast<uintptr_t>(output) % alignof(__nv_bfloat162) == 0 &&
      (bias == nullptr ||
       reinterpret_cast<uintptr_t>(bias) % alignof(__nv_bfloat162) == 0);
  if (packed4) {
    bias_activation_bf16_packed4_kernel<<<
        blocks_for(count / 4), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<const __nv_bfloat16*>(bias),
        static_cast<__nv_bfloat16*>(output), count / 4, cols, activation);
  } else if (packed2) {
    bias_activation_bf16_packed2_kernel<<<
        blocks_for(count / 2), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<const __nv_bfloat16*>(bias),
        static_cast<__nv_bfloat16*>(output), count / 2, cols, activation);
  } else {
    bias_activation_bf16_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<const __nv_bfloat16*>(bias),
        static_cast<__nv_bfloat16*>(output), count, cols, activation);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_embedding_bf16(
    const void* table, const void* ids, void* output,
    int tokens, int width, int vocab_size, cudaStream_t stream) {
  const int64_t count = static_cast<int64_t>(tokens) * width;
  embedding_bf16_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(table), static_cast<const uint32_t*>(ids),
      static_cast<__nv_bfloat16*>(output), tokens, width, vocab_size);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_concat_rows_bf16(
    const void* first, const void* second, void* output,
    int first_rows, int second_rows, int cols, cudaStream_t stream) {
  const int64_t first_count = static_cast<int64_t>(first_rows) * cols;
  const int64_t total_count = static_cast<int64_t>(first_rows + second_rows) * cols;
  concat_rows_bf16_kernel<<<blocks_for(total_count), kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(first),
      static_cast<const __nv_bfloat16*>(second),
      static_cast<__nv_bfloat16*>(output), first_count, total_count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_euler_update_bf16(
    const void* state, const void* velocity, void* output,
    int64_t count, float dt, cudaStream_t stream) {
  euler_update_bf16_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(state),
      static_cast<const __nv_bfloat16*>(velocity),
      static_cast<__nv_bfloat16*>(output), count, dt);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_geglu_bf16(
    const void* gate_up, void* output, int rows, int inner,
    cudaStream_t stream) {
  const int64_t count = static_cast<int64_t>(rows) * inner;
  const bool packed4 = inner % 4 == 0 &&
      reinterpret_cast<uintptr_t>(gate_up) % alignof(Bf16x4) == 0 &&
      reinterpret_cast<uintptr_t>(output) % alignof(Bf16x4) == 0;
  const bool packed2 = inner % 2 == 0 &&
      reinterpret_cast<uintptr_t>(gate_up) % alignof(__nv_bfloat162) == 0 &&
      reinterpret_cast<uintptr_t>(output) % alignof(__nv_bfloat162) == 0;
  if (packed4) {
    geglu_bf16_packed4_kernel<<<blocks_for(count / 4), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gate_up),
        static_cast<__nv_bfloat16*>(output), rows, inner);
  } else if (packed2) {
    geglu_bf16_packed2_kernel<<<blocks_for(count / 2), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gate_up),
        static_cast<__nv_bfloat16*>(output), rows, inner);
  } else {
    geglu_bf16_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gate_up),
        static_cast<__nv_bfloat16*>(output), rows, inner);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_residual_bf16(
    const void* projection, const void* bias, const void* residual,
    void* output, int rows, int cols, cudaStream_t stream) {
  const int64_t count = static_cast<int64_t>(rows) * cols;
  bias_residual_bf16_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(projection),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<const __nv_bfloat16*>(residual),
      static_cast<__nv_bfloat16*>(output), count, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_rms_norm_bf16(
    const void* input, const void* weight, void* output,
    int rows, int cols, float eps, cudaStream_t stream) {
  rms_norm_bf16_kernel<<<rows, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(output), rows, cols, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_layer_norm_bf16(
    const void* input, const void* weight, const void* bias, void* output,
    int rows, int cols, float eps, cudaStream_t stream) {
  layer_norm_bf16_kernel<<<rows, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<__nv_bfloat16*>(output), rows, cols, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_residual_rms_norm_bf16(
    const void* projection, const void* bias, const void* residual,
    const void* weight, void* hidden, void* normalized,
    int rows, int cols, float eps, cudaStream_t stream) {
  bias_residual_rms_norm_bf16_kernel<<<rows, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(projection),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<const __nv_bfloat16*>(residual),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(hidden),
      static_cast<__nv_bfloat16*>(normalized), rows, cols, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_residual_layer_norm_bf16(
    const void* projection, const void* projection_bias,
    const void* residual, const void* norm_weight, const void* norm_bias,
    void* hidden, void* normalized, int rows, int cols, float eps,
    cudaStream_t stream) {
  bias_residual_layer_norm_bf16_kernel<<<rows, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(projection),
      static_cast<const __nv_bfloat16*>(projection_bias),
      static_cast<const __nv_bfloat16*>(residual),
      static_cast<const __nv_bfloat16*>(norm_weight),
      static_cast<const __nv_bfloat16*>(norm_bias),
      static_cast<__nv_bfloat16*>(hidden),
      static_cast<__nv_bfloat16*>(normalized), rows, cols, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_ada_rms_norm_bf16(
    const void* input, const void* style, void* output,
    int rows, int cols, float eps, cudaStream_t stream) {
  ada_rms_norm_bf16_kernel<<<rows, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(style),
      static_cast<__nv_bfloat16*>(output), rows, cols, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_ada_gate_residual_bf16(
    const void* projection, const void* residual, const void* style,
    void* output, int rows, int cols, cudaStream_t stream) {
  const int64_t count = static_cast<int64_t>(rows) * cols;
  ada_gate_residual_bf16_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(projection),
      static_cast<const __nv_bfloat16*>(residual),
      static_cast<const __nv_bfloat16*>(style),
      static_cast<__nv_bfloat16*>(output), count, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_ada_gate_residual_rms_norm_bf16(
    const void* projection, const void* residual, const void* gate_style,
    const void* norm_style, void* hidden, void* normalized,
    int rows, int cols, float eps, cudaStream_t stream) {
  ada_gate_residual_rms_norm_bf16_kernel<<<rows, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(projection),
      static_cast<const __nv_bfloat16*>(residual),
      static_cast<const __nv_bfloat16*>(gate_style),
      static_cast<const __nv_bfloat16*>(norm_style),
      static_cast<__nv_bfloat16*>(hidden),
      static_cast<__nv_bfloat16*>(normalized), rows, cols, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qkv_rope_bf16(
    const void* qkv, const void* bias, void* q, void* k, void* v,
    int tokens, int q_heads, int kv_heads, int head_dim,
    float theta, int position_offset, int kv_output_offset,
    cudaStream_t stream) {
  dim3 grid(tokens, q_heads + 2 * kv_heads, 1);
  qkv_rope_bf16_kernel<<<grid, head_dim / 2, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(qkv),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<__nv_bfloat16*>(q), static_cast<__nv_bfloat16*>(k),
      static_cast<__nv_bfloat16*>(v), tokens, q_heads, kv_heads, head_dim,
      theta, position_offset, kv_output_offset);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qkv_split_bias_bf16(
    const void* qkv, const void* bias, void* q, void* k, void* v,
    int tokens, int projection_width, cudaStream_t stream) {
  qkv_split_bias_bf16_kernel<<<tokens, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(qkv),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<__nv_bfloat16*>(q), static_cast<__nv_bfloat16*>(k),
      static_cast<__nv_bfloat16*>(v), tokens, projection_width);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gqa_qkv_split_bias_bf16(
    const void* qkv, const void* bias, void* q, void* k, void* v,
    int tokens, int q_width, int kv_width, cudaStream_t stream) {
  gqa_qkv_split_bias_bf16_kernel<<<tokens, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(qkv),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<__nv_bfloat16*>(q), static_cast<__nv_bfloat16*>(k),
      static_cast<__nv_bfloat16*>(v), tokens, q_width, kv_width);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_mqa_bf16(
    const void* q, const void* k, const void* v, void* output,
    int query_tokens, int key_tokens, int heads, int head_dim,
    cudaStream_t stream) {
  dim3 grid(query_tokens, heads, 1);
  const size_t shared = static_cast<size_t>(key_tokens + 8) * sizeof(float);
  mqa_bf16_kernel<<<grid, kThreads, shared, stream>>>(
      static_cast<const __nv_bfloat16*>(q),
      static_cast<const __nv_bfloat16*>(k),
      static_cast<const __nv_bfloat16*>(v),
      static_cast<__nv_bfloat16*>(output), query_tokens, key_tokens,
      heads, head_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_mha_bf16(
    const void* q, const void* k, const void* v, void* output,
    int tokens_per_batch, int batches, int heads, int head_dim,
    cudaStream_t stream) {
  dim3 grid(tokens_per_batch, heads, batches);
  const size_t shared = static_cast<size_t>(tokens_per_batch + 8) * sizeof(float);
  mha_bf16_kernel<<<grid, kThreads, shared, stream>>>(
      static_cast<const __nv_bfloat16*>(q),
      static_cast<const __nv_bfloat16*>(k),
      static_cast<const __nv_bfloat16*>(v),
      static_cast<__nv_bfloat16*>(output), tokens_per_batch, heads, head_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_position_bf16(
    const void* projection, const void* bias, const void* position,
    void* output, int rows, int cols, int tokens_per_view,
    cudaStream_t stream) {
  const int64_t count = static_cast<int64_t>(rows) * cols;
  bias_position_bf16_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(projection),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<const __nv_bfloat16*>(position),
      static_cast<__nv_bfloat16*>(output), count, cols, tokens_per_view);
  return cudaGetLastError();
}
