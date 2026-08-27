// Copyright 2026 apxinf contributors.
// Stable C ABI and CUDA launch adapter for custom static-inference operators.

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {
#include "../kernels/custom/math.cuh"
#include "../kernels/custom/reduction.cuh"
#include "../kernels/custom/quantization.cuh"
#include "../kernels/custom/preprocess.cuh"
#include "../kernels/custom/attention.cuh"
#include "../kernels/custom/normalization.cuh"
#include "../kernels/custom/activation.cuh"
#include "../kernels/custom/embedding.cuh"
#include "../kernels/custom/elementwise.cuh"
#include "../kernels/custom/fused.cuh"
#include "../kernels/custom/cache.cuh"
#include "../kernels/custom/qwen35.cuh"
}  // namespace

extern "C" cudaError_t apxinf_static_evict_l2(
    void* buffer, size_t bytes, uint32_t seed, cudaStream_t stream) {
  if (buffer == nullptr || bytes < sizeof(uint32_t) ||
      bytes % sizeof(uint32_t) != 0) {
    return cudaErrorInvalidValue;

  }
  constexpr int threads = 256;
  int blocks = static_cast<int>((bytes / sizeof(uint32_t) + threads - 1) /
                                threads);
  blocks = blocks > 4096 ? 4096 : blocks;
  l2_cache_evict_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<volatile uint32_t*>(buffer), bytes / sizeof(uint32_t), seed);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_quantize_f16_e4m3(
    const void* input, void* output, int64_t count, float scale,
    cudaStream_t stream) {
  if (input == nullptr || output == nullptr || count <= 0 || !(scale > 0.0f))
    return cudaErrorInvalidValue;
  constexpr int threads = 256;
  const float inverse_scale = 1.0f / scale;
  const bool aligned =
      (reinterpret_cast<uintptr_t>(input) & 3U) == 0 &&
      (reinterpret_cast<uintptr_t>(output) & 3U) == 0;
  int64_t vector_count = aligned ? count & ~int64_t{3} : 0;
  if (vector_count != 0) {
    const int64_t groups = vector_count / 4;
    int blocks = static_cast<int>((groups + threads - 1) / threads);
    blocks = blocks > 1024 ? 1024 : blocks;
    quantize_f16_e4m3_packed4_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const half*>(input),
        static_cast<__nv_fp8_e4m3*>(output), vector_count, inverse_scale);
  }
  const int64_t tail = count - vector_count;
  if (tail != 0) {
    int blocks = static_cast<int>((tail + threads - 1) / threads);
    blocks = blocks > 1024 ? 1024 : blocks;
    quantize_f16_e4m3_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const half*>(input) + vector_count,
        static_cast<__nv_fp8_e4m3*>(output) + vector_count,
        tail, inverse_scale);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_dequantize_e4m3_f16(
    const void* input, void* output, int64_t count, float scale,
    cudaStream_t stream) {
  if (input == nullptr || output == nullptr || count <= 0 ||
      !(scale > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 4096 ? 4096 : blocks;
  dequantize_e4m3_f16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_fp8_e4m3*>(input),
      static_cast<half*>(output), count, scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_rgb_u8_to_patches_e4m3(
    const void* images, void* patches, int views, int image_size,
    int patch_size, int layout, float scale, cudaStream_t stream) {
  if (images == nullptr || patches == nullptr || views <= 0 ||
      image_size <= 0 || patch_size <= 0 || image_size % patch_size != 0 ||
      (layout != 0 && layout != 1) || !(scale > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  const int patches_per_side = image_size / patch_size;
  const int64_t count = static_cast<int64_t>(views) * patches_per_side *
                        patches_per_side * 3 * patch_size * patch_size;
  constexpr int threads = 256;
  int blocks = static_cast<int>((count + threads - 1) / threads);
  blocks = blocks > 1024 ? 1024 : blocks;
  if (layout == 0) {
    rgb_u8_to_patches_e4m3_kernel<true><<<blocks, threads, 0, stream>>>(
        static_cast<const uint8_t*>(images),
        static_cast<__nv_fp8_e4m3*>(patches), views, image_size, patch_size,
        1.0f / scale);
  } else {
    rgb_u8_to_patches_e4m3_kernel<false><<<blocks, threads, 0, stream>>>(
        static_cast<const uint8_t*>(images),
        static_cast<__nv_fp8_e4m3*>(patches), views, image_size, patch_size,
        1.0f / scale);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_mqa_flash_f16(
    const void* q, const void* prefix_k, const void* prefix_v,
    const void* suffix_k, const void* suffix_v, void* output,
    int suffix_tokens, int heads, int head_dim, int prefix_tokens,
    cudaStream_t stream) {
  if (suffix_tokens <= 0 || heads <= 0 || head_dim <= 0 || head_dim > 256 ||
      prefix_tokens < 0) return cudaErrorInvalidValue;
  int threads = 256;
  int warps = threads / 32;
  size_t shared_bytes =
      static_cast<size_t>(prefix_tokens + suffix_tokens + warps) * sizeof(float);
  mqa_flash_f16_kernel<<<dim3(suffix_tokens, heads), threads, shared_bytes, stream>>>(
      static_cast<const half*>(q), static_cast<const half*>(prefix_k),
      static_cast<const half*>(prefix_v), static_cast<const half*>(suffix_k),
      static_cast<const half*>(suffix_v), static_cast<half*>(output),
      suffix_tokens, heads, head_dim, prefix_tokens);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_rms_norm_quant_f16_e4m3(
    const void* input, const void* weight, void* output, int rows, int cols,
    float eps, float scale, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0 || !(scale > 0.0f)) return cudaErrorInvalidValue;
  rms_norm_quant_f16_e4m3_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const half*>(input), static_cast<const half*>(weight),
      static_cast<__nv_fp8_e4m3*>(output), rows, cols, eps, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_layer_norm_quant_f16_e4m3(
    const void* input, const void* weight, const void* bias, void* output,
    int rows, int cols, float eps, float scale, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0 || !(scale > 0.0f)) return cudaErrorInvalidValue;
  layer_norm_quant_f16_e4m3_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const half*>(input), static_cast<const half*>(weight),
      static_cast<const half*>(bias), static_cast<__nv_fp8_e4m3*>(output),
      rows, cols, eps, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_gelu_quant_f16_e4m3(
    const void* input, const void* bias, void* output, int rows, int cols,
    float scale, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0 || !(scale > 0.0f)) return cudaErrorInvalidValue;
  int64_t count = static_cast<int64_t>(rows) * cols;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  bias_gelu_quant_f16_e4m3_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const half*>(input), static_cast<const half*>(bias),
      static_cast<__nv_fp8_e4m3*>(output), count, cols, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_silu_quant_f16_e4m3(
    const void* input, const void* bias, void* output, int rows, int cols,
    float scale, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0 || !(scale > 0.0f)) return cudaErrorInvalidValue;
  int64_t count = static_cast<int64_t>(rows) * cols;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  bias_silu_quant_f16_e4m3_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const half*>(input), static_cast<const half*>(bias),
      static_cast<__nv_fp8_e4m3*>(output), count, cols, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_silu_f16(
    const void* input, const void* bias, void* output, int rows, int cols,
    cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
  int64_t count = static_cast<int64_t>(rows) * cols;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  bias_silu_f16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const half*>(input), static_cast<const half*>(bias),
      static_cast<half*>(output), count, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_f16(
    const void* input, const void* bias, void* output, int rows, int cols,
    cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
  int64_t count = static_cast<int64_t>(rows) * cols;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  bias_f16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const half*>(input), static_cast<const half*>(bias),
      static_cast<half*>(output), count, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_embedding_f16(
    const void* table, const void* ids, void* output, int tokens,
    int width, int vocab_size, cudaStream_t stream) {
  if (tokens <= 0 || width <= 0 || vocab_size <= 0) return cudaErrorInvalidValue;
  int64_t count = static_cast<int64_t>(tokens) * width;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  embedding_f16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const half*>(table), static_cast<const uint32_t*>(ids),
      static_cast<half*>(output), tokens, width, vocab_size);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_concat_rows_f16(
    const void* first, const void* second, void* output, int first_rows,
    int second_rows, int cols, cudaStream_t stream) {
  if (first_rows <= 0 || second_rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
  int64_t first_count = static_cast<int64_t>(first_rows) * cols;
  int64_t total_count = static_cast<int64_t>(first_rows + second_rows) * cols;
  int blocks = static_cast<int>((total_count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  concat_rows_f16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const half*>(first), static_cast<const half*>(second),
      static_cast<half*>(output), first_count, total_count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_euler_update_f16(
    const void* state, const void* velocity, void* output, int64_t count,
    float dt, cudaStream_t stream) {
  if (count <= 0) return cudaErrorInvalidValue;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  euler_update_f16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const half*>(state), static_cast<const half*>(velocity),
      static_cast<half*>(output), count, dt);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_geglu_quant_f16_e4m3(
    const void* gate_up, void* output, int rows, int inner, float scale,
    cudaStream_t stream) {
  if (rows <= 0 || inner <= 0 || !(scale > 0.0f)) return cudaErrorInvalidValue;
  if ((inner & 1) != 0) return cudaErrorInvalidValue;
  const bool packed8 = (inner & 7) == 0 &&
      (reinterpret_cast<uintptr_t>(gate_up) & 7U) == 0 &&
      (reinterpret_cast<uintptr_t>(output) & 7U) == 0;
  if (packed8) {
    int group_count = rows * (inner / 8);
    int blocks = (group_count + 255) / 256;
    blocks = blocks > 1024 ? 1024 : blocks;
    geglu_quant_f16_e4m3_packed8_kernel<<<blocks, 256, 0, stream>>>(
        static_cast<const half*>(gate_up),
        static_cast<__nv_fp8_e4m3*>(output), rows, inner, 1.0f / scale);
    return cudaGetLastError();
  }
  const bool packed4 = (inner & 3) == 0 &&
      (reinterpret_cast<uintptr_t>(gate_up) & 3U) == 0 &&
      (reinterpret_cast<uintptr_t>(output) & 3U) == 0;
  if (packed4) {
    int group_count = rows * (inner / 4);
    int blocks = (group_count + 255) / 256;
    blocks = blocks > 1024 ? 1024 : blocks;
    geglu_quant_f16_e4m3_packed4_kernel<<<blocks, 256, 0, stream>>>(
        static_cast<const half*>(gate_up),
        static_cast<__nv_fp8_e4m3*>(output), rows, inner, 1.0f / scale);
    return cudaGetLastError();
  }
  int pair_count = rows * (inner / 2);
  int blocks = (pair_count + 255) / 256;
  blocks = blocks > 1024 ? 1024 : blocks;
  geglu_quant_f16_e4m3_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const half*>(gate_up), static_cast<__nv_fp8_e4m3*>(output),
      rows, inner, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_residual_f16(
    const void* projection, const void* bias, const void* residual, void* output,
    int rows, int cols, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
  int64_t count = static_cast<int64_t>(rows) * cols;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  bias_residual_f16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const half*>(projection), static_cast<const half*>(bias),
      static_cast<const half*>(residual), static_cast<half*>(output), count, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_residual_rms_norm_quant_f16_e4m3(
    const void* projection, const void* bias, const void* residual,
    const void* weight, void* hidden, void* normalized, int rows, int cols,
    float eps, float scale, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0 || !(scale > 0.0f)) return cudaErrorInvalidValue;
  bias_residual_rms_norm_quant_f16_e4m3_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const half*>(projection), static_cast<const half*>(bias),
      static_cast<const half*>(residual), static_cast<const half*>(weight),
      static_cast<half*>(hidden), static_cast<__nv_fp8_e4m3*>(normalized),
      rows, cols, eps, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_residual_layer_norm_quant_f16_e4m3(
    const void* projection, const void* projection_bias, const void* residual,
    const void* norm_weight, const void* norm_bias, void* hidden,
    void* normalized, int rows, int cols, float eps, float scale,
    cudaStream_t stream) {
  if (rows <= 0 || cols <= 0 || !(scale > 0.0f)) return cudaErrorInvalidValue;
  bias_residual_layer_norm_quant_f16_e4m3_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const half*>(projection), static_cast<const half*>(projection_bias),
      static_cast<const half*>(residual), static_cast<const half*>(norm_weight),
      static_cast<const half*>(norm_bias), static_cast<half*>(hidden),
      static_cast<__nv_fp8_e4m3*>(normalized), rows, cols, eps, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_ada_rms_norm_quant_f16_e4m3(
    const void* input, const void* style, void* output, int rows, int cols,
    float eps, float scale, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0 || !(scale > 0.0f)) return cudaErrorInvalidValue;
  ada_rms_norm_quant_f16_e4m3_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const half*>(input), static_cast<const half*>(style),
      static_cast<__nv_fp8_e4m3*>(output), rows, cols, eps, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_ada_gate_residual_f16(
    const void* projection, const void* residual, const void* style,
    void* output, int rows, int cols, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
  int64_t count = static_cast<int64_t>(rows) * cols;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  ada_gate_residual_f16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const half*>(projection), static_cast<const half*>(residual),
      static_cast<const half*>(style), static_cast<half*>(output), rows, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_ada_gate_residual_rms_norm_quant_f16_e4m3(
    const void* projection, const void* residual, const void* gate_style,
    const void* norm_style, void* hidden, void* normalized, int rows, int cols,
    float eps, float scale, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0 || !(scale > 0.0f)) return cudaErrorInvalidValue;
  ada_gate_residual_rms_norm_quant_f16_e4m3_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const half*>(projection), static_cast<const half*>(residual),
      static_cast<const half*>(gate_style), static_cast<const half*>(norm_style),
      static_cast<half*>(hidden), static_cast<__nv_fp8_e4m3*>(normalized),
      rows, cols, eps, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qkv_rope_f16(
    const void* qkv, const void* bias, void* q, void* k, void* v, int tokens, int q_heads,
    int kv_heads, int head_dim, float theta, int position_offset,
    int kv_output_offset, cudaStream_t stream) {
  if (tokens <= 0 || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
      head_dim > 256 || (head_dim & 1) != 0) return cudaErrorInvalidValue;
  qkv_rope_f16_kernel<<<dim3(tokens, q_heads + 2 * kv_heads), head_dim / 2, 0, stream>>>(
      static_cast<const half*>(qkv), static_cast<const half*>(bias),
      static_cast<half*>(q), static_cast<half*>(k),
      static_cast<half*>(v), tokens, q_heads, kv_heads, head_dim, theta,
      position_offset, kv_output_offset);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qkv_split_bias_f16(
    const void* qkv, const void* bias, void* q, void* k, void* v,
    int tokens, int projection_width, cudaStream_t stream) {
  if (tokens <= 0 || projection_width <= 0) return cudaErrorInvalidValue;
  qkv_split_bias_f16_kernel<<<tokens, 256, 0, stream>>>(
      static_cast<const half*>(qkv), static_cast<const half*>(bias),
      static_cast<half*>(q), static_cast<half*>(k), static_cast<half*>(v),
      tokens, projection_width);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_mha_flash_f16(
    const void* q, const void* k, const void* v, void* output,
    int tokens_per_batch, int batches, int heads, int head_dim, cudaStream_t stream) {
  if (tokens_per_batch <= 0 || batches <= 0 || heads <= 0 ||
      head_dim <= 0 || head_dim > 256)
    return cudaErrorInvalidValue;
  constexpr int threads = 256;
  size_t shared_bytes = static_cast<size_t>(tokens_per_batch + threads / 32) * sizeof(float);
  mha_flash_f16_kernel<<<dim3(tokens_per_batch, heads, batches), threads, shared_bytes, stream>>>(
      static_cast<const half*>(q), static_cast<const half*>(k),
      static_cast<const half*>(v), static_cast<half*>(output),
      tokens_per_batch, heads, head_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_position_f16(
    const void* projection, const void* bias, const void* position,
    void* output, int rows, int cols, int tokens_per_view, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0 || tokens_per_view <= 0 ||
      rows % tokens_per_view != 0) return cudaErrorInvalidValue;
  int64_t count = static_cast<int64_t>(rows) * cols;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  bias_position_f16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const half*>(projection), static_cast<const half*>(bias),
      static_cast<const half*>(position), static_cast<half*>(output),
      count, cols, tokens_per_view);
  return cudaGetLastError();
}

// ── Qwen3.5 launchers (C1 eager path) ───────────────────────────────────

extern "C" cudaError_t apxinf_qwen_dequant_w4a16_bf16(
    const void* packed, const void* scale, const void* zp, void* out,
    int out_dim, int in_dim, cudaStream_t stream) {
  if (out_dim <= 0 || in_dim <= 0) return cudaErrorInvalidValue;
  dim3 dq_block(32, 8);
  dim3 dq_grid((out_dim + 31) / 32, (in_dim + 31) / 32);
  qwen_dequant_w4a16_bf16_kernel_v2<<<dq_grid, dq_block, 0, stream>>>(
      static_cast<const int32_t*>(packed),
      static_cast<const __half*>(scale),
      static_cast<const int32_t*>(zp),
      static_cast<__half*>(out), out_dim, in_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen_mul_bf16(
    const void* a, const void* b, void* out, int64_t n, cudaStream_t stream) {
  if (n <= 0) return cudaErrorInvalidValue;
  int threads = 256;
  int64_t blocks64 = (n + threads - 1) / threads;
  int blocks = blocks64 > (1 << 20) ? (1 << 20) : (int)blocks64;
  qwen_mul_bf16_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __half*>(a),
      static_cast<const __half*>(b),
      static_cast<__half*>(out), n);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen_silu_bf16(
    const void* a, void* out, int64_t n, cudaStream_t stream) {
  if (n <= 0) return cudaErrorInvalidValue;
  int threads = 256;
  int64_t blocks64 = (n + threads - 1) / threads;
  int blocks = blocks64 > (1 << 20) ? (1 << 20) : (int)blocks64;
  qwen_silu_bf16_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __half*>(a),
      static_cast<__half*>(out), n);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen_sigmoid_mul_bf16(
    const void* a, const void* b, void* out, int64_t n, cudaStream_t stream) {
  if (n <= 0) return cudaErrorInvalidValue;
  int threads = 256;
  int64_t blocks64 = (n + threads - 1) / threads;
  int blocks = blocks64 > (1 << 20) ? (1 << 20) : (int)blocks64;
  qwen_sigmoid_mul_bf16_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __half*>(a),
      static_cast<const __half*>(b),
      static_cast<__half*>(out), n);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen_accum_bf16(
    void* dst, const void* src, int64_t n, cudaStream_t stream) {
  if (n <= 0) return cudaErrorInvalidValue;
  int threads = 256;
  int64_t blocks64 = (n + threads - 1) / threads;
  int blocks = blocks64 > (1 << 20) ? (1 << 20) : (int)blocks64;
  qwen_accum_bf16_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<__half*>(dst),
      static_cast<const __half*>(src), n);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen_rms_norm_bf16(
    const void* x, const void* w, void* out,
    int rows, int cols, float eps, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
  int threads = 256;
  size_t shmem = (size_t)threads * sizeof(float);
  qwen_rms_norm_bf16_kernel<<<rows, threads, shmem, stream>>>(
      static_cast<const __half*>(x),
      static_cast<const __half*>(w),
      static_cast<__half*>(out), rows, cols, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen_qg_split_bf16(
    const void* qg, void* q, void* gate, int64_t total,
    int heads, int hd, cudaStream_t stream) {
  if (total <= 0 || heads <= 0 || hd <= 0) return cudaErrorInvalidValue;
  int threads = 256;
  int64_t blocks64 = (total + threads - 1) / threads;
  int blocks = blocks64 > (1 << 20) ? (1 << 20) : (int)blocks64;
  qwen_qg_split_bf16_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __half*>(qg),
      static_cast<__half*>(q),
      static_cast<__half*>(gate), total, heads, hd);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen_partial_rope_bf16(
    void* x, const void* cos, const void* sin, int64_t pairs,
    int heads, int hd, int half, const void* pos0, cudaStream_t stream) {
  if (pairs <= 0) return cudaErrorInvalidValue;
  int threads = 256;
  int64_t blocks64 = (pairs + threads - 1) / threads;
  int blocks = blocks64 > (1 << 20) ? (1 << 20) : (int)blocks64;
  qwen_partial_rope_bf16_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<__half*>(x),
      static_cast<const __half*>(cos),
      static_cast<const __half*>(sin),
      pairs, heads, hd, half, static_cast<const int*>(pos0));
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen_conv_silu_bf16(
    const void* x, const void* w, void* out,
    int L, int conv_dim, cudaStream_t stream) {
  if (L <= 0 || conv_dim <= 0) return cudaErrorInvalidValue;
  int64_t total = (int64_t)L * conv_dim;
  int threads = 256;
  int64_t blocks64 = (total + threads - 1) / threads;
  int blocks = blocks64 > (1 << 20) ? (1 << 20) : (int)blocks64;
  qwen_conv_silu_bf16_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __half*>(x),
      static_cast<const __half*>(w),
      static_cast<__half*>(out), L, conv_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen_delta_recurrence_bf16(
    const void* q, const void* k, const void* v,
    const void* beta, const void* g,
    void* state, void* out,
    int L, int nv, int kd, int vd, cudaStream_t stream) {
  if (L <= 0 || nv <= 0 || kd <= 0 || vd <= 0) return cudaErrorInvalidValue;
  if (vd > 128 || kd % DR_KDCHUNK != 0) return cudaErrorInvalidValue;
  if (kd % 32 == 0 && kd <= 512) {
    dim3 b2(256);
    dim3 g2(nv, (vd + 7) / 8);
    const __half* qh = static_cast<const __half*>(q);
    const __half* kh = static_cast<const __half*>(k);
    const __half* vh = static_cast<const __half*>(v);
    const __half* bh = static_cast<const __half*>(beta);
    const __half* gh = static_cast<const __half*>(g);
    float* st = static_cast<float*>(state);
    __half* oh = static_cast<__half*>(out);
    switch (kd / 32) {
      case 2:
        qwen_delta_recurrence_bf16_kernel_v3<2><<<g2, b2, 0, stream>>>(qh, kh, vh, bh, gh, st, oh, L, nv, vd);
        return cudaGetLastError();
      case 4:
        qwen_delta_recurrence_bf16_kernel_v3<4><<<g2, b2, 0, stream>>>(qh, kh, vh, bh, gh, st, oh, L, nv, vd);
        return cudaGetLastError();
      case 8:
        qwen_delta_recurrence_bf16_kernel_v3<8><<<g2, b2, 0, stream>>>(qh, kh, vh, bh, gh, st, oh, L, nv, vd);
        return cudaGetLastError();
      case 16:
        qwen_delta_recurrence_bf16_kernel_v3<16><<<g2, b2, 0, stream>>>(qh, kh, vh, bh, gh, st, oh, L, nv, vd);
        return cudaGetLastError();
      default:
        break;
    }
    qwen_delta_recurrence_bf16_kernel_v2<<<g2, b2, 0, stream>>>(qh, kh, vh, bh, gh, st, oh, L, nv, kd, vd);
    return cudaGetLastError();
  }
  dim3 block(vd, DR_KDCHUNK);
  qwen_delta_recurrence_bf16_kernel<<<nv, block, 0, stream>>>(
      static_cast<const __half*>(q),
      static_cast<const __half*>(k),
      static_cast<const __half*>(v),
      static_cast<const __half*>(beta),
      static_cast<const __half*>(g),
      static_cast<float*>(state),
      static_cast<__half*>(out),
      L, nv, kd, vd);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen_attention_bf16(
    const void* q, const void* k, const void* v, const void* gate,
    void* out, int L, int heads, int kvheads, int hd,
    cudaStream_t stream) {
  if (L <= 0 || heads <= 0 || kvheads <= 0 || hd <= 0) return cudaErrorInvalidValue;
  if (hd > 256 || (hd & (hd - 1)) != 0) return cudaErrorInvalidValue;
  qwen_attention_bf16_kernel<<<L * heads, hd, 0, stream>>>(
      static_cast<const __half*>(q),
      static_cast<const __half*>(k),
      static_cast<const __half*>(v),
      static_cast<const __half*>(gate),
      static_cast<__half*>(out),
      L, heads, kvheads, hd);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen_conv_split_bf16(
    const void* conv, void* q, void* k, void* v,
    int L, int nk, int nv, int kd, int vd, int conv_dim, cudaStream_t stream) {
  if (L <= 0 || nk <= 0 || nv <= 0 || kd <= 0 || vd <= 0 || conv_dim <= 0)
    return cudaErrorInvalidValue;
  int64_t total = (int64_t)L * nv * kd + (int64_t)L * nv * vd;
  int threads = 256;
  int64_t b64 = (total + threads - 1) / threads;
  int blocks = b64 > (1 << 20) ? (1 << 20) : (int)b64;
  qwen_conv_split_bf16_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __half*>(conv),
      static_cast<__half*>(q),
      static_cast<__half*>(k),
      static_cast<__half*>(v),
      L, nk, nv, kd, vd, conv_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen_l2norm_bf16(
    const void* x, void* out, int rows, int cols, float eps, float scale,
    cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
  int threads = 256;
  size_t shmem = (size_t)threads * sizeof(float);
  qwen_l2norm_bf16_kernel<<<rows, threads, shmem, stream>>>(
      static_cast<const __half*>(x),
      static_cast<__half*>(out), rows, cols, eps, scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen_gemm_w4a16_bf16(
    const void* a, const void* packed, const void* scale, const void* zp,
    void* c, int m, int n, int k, cudaStream_t stream) {
  if (m <= 0 || n <= 0 || k <= 0) return cudaErrorInvalidValue;
  dim3 grid((n + QW4_BN - 1) / QW4_BN, (m + QW4_BM - 1) / QW4_BM);
  qwen_gemm_w4a16_bf16_kernel<<<grid, 256, 0, stream>>>(
      static_cast<const __half*>(a),
      static_cast<const int32_t*>(packed),
      static_cast<const __half*>(scale),
      static_cast<const int32_t*>(zp),
      static_cast<__half*>(c), m, n, k);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen_beta_g_bf16(
    const void* a, const void* b, const void* a_log, const void* dt_bias,
    void* beta, void* g, int total, int nv, cudaStream_t stream) {
  if (total <= 0 || nv <= 0) return cudaErrorInvalidValue;
  int threads = 256;
  int64_t blocks64 = ((int64_t)total + threads - 1) / threads;
  int blocks = blocks64 > (1 << 20) ? (1 << 20) : (int)blocks64;
  qwen_beta_g_bf16_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __half*>(a),
      static_cast<const __half*>(b),
      static_cast<const float*>(a_log),
      static_cast<const float*>(dt_bias),
      static_cast<__half*>(beta),
      static_cast<__half*>(g), total, nv);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen_copy_bf16(
    const void* src, void* dst, int64_t n, cudaStream_t stream) {
  if (n <= 0) return cudaErrorInvalidValue;
  int threads = 256;
  int64_t blocks64 = (n + threads - 1) / threads;
  int blocks = blocks64 > (1 << 20) ? (1 << 20) : (int)blocks64;
  qwen_copy_bf16_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __half*>(src),
      static_cast<__half*>(dst), n);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen_attention_decode_bf16(
    const void* q, const void* kcache, const void* vcache, const void* gate,
    void* out, const void* seq_ptr, int heads, int kvheads, int hd,
    cudaStream_t stream) {
  if (heads <= 0 || kvheads <= 0 || hd <= 0) return cudaErrorInvalidValue;
  if (hd > 256 || (hd & (hd - 1)) != 0) return cudaErrorInvalidValue;
  qwen_attention_decode_bf16_kernel<<<heads, hd, 0, stream>>>(
      static_cast<const __half*>(q),
      static_cast<const __half*>(kcache),
      static_cast<const __half*>(vcache),
      static_cast<const __half*>(gate),
      static_cast<__half*>(out),
      static_cast<const int*>(seq_ptr), heads, kvheads, hd);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen_conv_step_silu_bf16(
    const void* cur, void* hist, const void* w, void* out,
    int conv_dim, cudaStream_t stream) {
  if (conv_dim <= 0) return cudaErrorInvalidValue;
  int threads = 256;
  int blocks = (conv_dim + threads - 1) / threads;
  qwen_conv_step_silu_bf16_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __half*>(cur),
      static_cast<__half*>(hist),
      static_cast<const __half*>(w),
      static_cast<__half*>(out), conv_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen_delta_step_bf16(
    const void* q, const void* k, const void* v,
    const void* beta, const void* g, void* state, void* out,
    int nv, int kd, int vd, cudaStream_t stream) {
  if (nv <= 0 || kd <= 0 || vd <= 0) return cudaErrorInvalidValue;
  if (vd > 128 || kd % DR_KDCHUNK != 0) return cudaErrorInvalidValue;
  dim3 block(vd, DR_KDCHUNK);
  qwen_delta_step_bf16_kernel<<<nv, block, 0, stream>>>(
      static_cast<const __half*>(q),
      static_cast<const __half*>(k),
      static_cast<const __half*>(v),
      static_cast<const __half*>(beta),
      static_cast<const __half*>(g),
      static_cast<float*>(state),
      static_cast<__half*>(out), nv, kd, vd);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen_copy_at_bf16(
    const void* src, void* dst_base, const void* pos_ptr,
    int stride_elems, int n, cudaStream_t stream) {
  if (n <= 0) return cudaErrorInvalidValue;
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  qwen_copy_at_bf16_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __half*>(src),
      static_cast<__half*>(dst_base),
      static_cast<const int*>(pos_ptr), stride_elems, n);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen_gemm_w4a16_m1_bf16(
    const void* a, const void* packed, const void* scale, const void* zp,
    void* c, int n, int k, cudaStream_t stream) {
  if (n <= 0 || k <= 0) return cudaErrorInvalidValue;
  int threads = 128;
  int blocks = (n + threads - 1) / threads;
  qwen_gemm_w4a16_m1_bf16_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __half*>(a),
      static_cast<const int32_t*>(packed),
      static_cast<const __half*>(scale),
      static_cast<const int32_t*>(zp),
      static_cast<__half*>(c), n, k);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen_gemm_f16(
    const void* a, const void* b, void* c,
    int m, int n, int k, cudaStream_t stream) {
  if (m <= 0 || n <= 0 || k <= 0) return cudaErrorInvalidValue;
  dim3 grid((n + FW4_BN - 1) / FW4_BN, (m + FW4_BM - 1) / FW4_BM);
  qwen_gemm_f16_kernel<<<grid, 256, 0, stream>>>(
      static_cast<const __half*>(a), static_cast<const __half*>(b),
      static_cast<__half*>(c), m, n, k);
  return cudaGetLastError();
}


extern "C" cudaError_t apxinf_qwen_embed_gather_f16(
    const void* table, const void* ids, void* out,
    int64_t tokens, int hidden, cudaStream_t stream) {
  if (tokens <= 0 || hidden <= 0) return cudaErrorInvalidValue;
  int64_t count = tokens * hidden;
  int threads = 256;
  int64_t blocks64 = (count + threads - 1) / threads;
  int blocks = blocks64 > (1 << 20) ? (1 << 20) : (int)blocks64;
  qwen_embed_gather_f16_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(table),
      static_cast<const uint32_t*>(ids),
      static_cast<__half*>(out), count, hidden);
  return cudaGetLastError();
}
