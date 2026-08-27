// Copyright 2026 apxinf contributors.
// Stable C ABI and CUDA launch adapter for custom static-inference operators.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <atomic>

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


extern "C" cudaError_t apxinf_static_dequantize_w4a16_asym_bf16(
    const void* weight_packed, const void* weight_scale,
    const void* weight_zero_point, void* dense, int in_cols, int out_cols,
    int groups, cudaStream_t stream) {
  if (weight_packed == nullptr || weight_scale == nullptr ||
      weight_zero_point == nullptr || dense == nullptr || in_cols <= 0 ||
      out_cols <= 0 || groups <= 0) {
    return cudaErrorInvalidValue;
  }
  constexpr int threads = 256;
  const int64_t total = static_cast<int64_t>(in_cols) * out_cols;
  int blocks = static_cast<int>((total + threads - 1) / threads);
  blocks = blocks > 4096 ? 4096 : blocks;
  dequantize_w4a16_asym_bf16_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const int32_t*>(weight_packed),
      static_cast<const __nv_bfloat16*>(weight_scale),
      static_cast<const int32_t*>(weight_zero_point),
      static_cast<__nv_bfloat16*>(dense), in_cols, out_cols, groups);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_matmul_bf16_w4a16_asym(
    const void* activation, const void* weight_packed, const void* weight_scale,
    const void* weight_zero_point, void* output, int rows, int in_cols,
    int out_cols, int groups, cudaStream_t stream) {
  if (activation == nullptr || weight_packed == nullptr ||
      weight_scale == nullptr || weight_zero_point == nullptr ||
      output == nullptr || rows <= 0 || in_cols <= 0 || out_cols <= 0 ||
      groups <= 0) {
    return cudaErrorInvalidValue;
  }
  dim3 block(128);
  dim3 grid((out_cols + block.x - 1) / block.x, rows);
  matmul_bf16_w4a16_asym_kernel<<<grid, block, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(activation),
      static_cast<const int32_t*>(weight_packed),
      static_cast<const __nv_bfloat16*>(weight_scale),
      static_cast<const int32_t*>(weight_zero_point),
      static_cast<__nv_bfloat16*>(output), rows, in_cols, out_cols, groups);
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

// ── Qwen3.5 hybrid linear-attention adapters ───────────────────────────────

extern "C" cudaError_t apxinf_qwen35_conv_silu(
    const void* input, const void* weight, void* output, void* state,
    int seq, int channels, int kernel, cudaStream_t stream) {
  if (input == nullptr || weight == nullptr || output == nullptr ||
      state == nullptr || seq <= 0 || channels <= 0 || kernel <= 1) {
    return cudaErrorInvalidValue;
  }
  constexpr int threads = 256;
  int blocks = (channels + threads - 1) / threads;
  blocks = blocks > 1024 ? 1024 : blocks;
  const size_t shared = static_cast<size_t>(2 * kernel - 1) * threads * sizeof(float);
  qwen35_conv_silu_kernel<<<blocks, threads, shared, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(output), static_cast<float*>(state), seq,
      channels, kernel);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_delta_norm_prepass(
    const void* qkv, void* qk_out, int seq, int k_heads, int v_heads,
    int kdim, int vdim, cudaStream_t stream) {
  if (qkv == nullptr || qk_out == nullptr || seq <= 0 || k_heads <= 0 ||
      v_heads <= 0 || kdim <= 0 || kdim > QWEN35_KMAX || vdim <= 0) {
    return cudaErrorInvalidValue;
  }
  dim3 grid(seq, k_heads);
  qwen35_delta_norm_prepass_kernel<<<grid, QWEN35_KMAX, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(qkv),
      static_cast<__nv_bfloat16*>(qk_out), seq, k_heads, v_heads, kdim, vdim);
  return cudaGetLastError();
}

namespace {
std::atomic<bool> qwen35_delta_step_shared_opted{false};
}

extern "C" cudaError_t apxinf_qwen35_prepare_delta_step() {
  if (qwen35_delta_step_shared_opted.load(std::memory_order_acquire))
    return cudaSuccess;
  const cudaError_t status = cudaFuncSetAttribute(
      qwen35_delta_step_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      98304);
  if (status == cudaSuccess)
    qwen35_delta_step_shared_opted.store(true, std::memory_order_release);
  return status;
}

extern "C" cudaError_t apxinf_qwen35_delta_step(
    const void* qkv, const void* qk_norm, const void* a, const void* b,
    const void* a_log, const void* dt_bias, void* recurrent, void* out,
    int seq, int k_heads, int v_heads, int kdim, int vdim,
    cudaStream_t stream) {
  if (qkv == nullptr || qk_norm == nullptr || a == nullptr || b == nullptr ||
      a_log == nullptr || dt_bias == nullptr || recurrent == nullptr ||
      out == nullptr || seq <= 0 || k_heads <= 0 || v_heads <= 0 ||
      kdim <= 0 || kdim > QWEN35_KMAX || vdim <= 0 ||
      vdim % QWEN35_V_TILE != 0) {
    return cudaErrorInvalidValue;
  }
  dim3 grid(vdim / QWEN35_V_TILE, v_heads);
  const size_t shared =
      static_cast<size_t>(2 * kdim + kdim * QWEN35_V_TILE) * sizeof(float);
  if (!qwen35_delta_step_shared_opted.load(std::memory_order_acquire)) {
    const cudaError_t opt = apxinf_qwen35_prepare_delta_step();
    if (opt != cudaSuccess) return opt;
  }
  qwen35_delta_step_kernel<<<grid, QWEN35_V_TILE, shared, stream>>>(
      static_cast<const __nv_bfloat16*>(qkv),
      static_cast<const __nv_bfloat16*>(qk_norm),
      static_cast<const __nv_bfloat16*>(a),
      static_cast<const __nv_bfloat16*>(b),
      static_cast<const __nv_bfloat16*>(a_log),
      static_cast<const __nv_bfloat16*>(dt_bias),
      static_cast<float*>(recurrent), static_cast<__nv_bfloat16*>(out), seq,
      k_heads, v_heads, kdim, vdim);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_prefill_delta_step(
    const void* qkv, const void* qk_norm, const void* a, const void* b,
    const void* a_log, const void* dt_bias, void* recurrent, void* out,
    int seq, int k_heads, int v_heads, int kdim, int vdim,
    cudaStream_t stream) {
  if (qkv == nullptr || qk_norm == nullptr || a == nullptr || b == nullptr ||
      a_log == nullptr || dt_bias == nullptr || recurrent == nullptr ||
      out == nullptr || seq <= 1 || seq > 512 || k_heads <= 0 ||
      v_heads <= 0 || v_heads % k_heads != 0 || kdim != QWEN35_KMAX ||
      vdim != QWEN35_V_TILE) {
    return cudaErrorInvalidValue;
  }
  qwen35_prefill_delta_step_kernel<<<
      dim3(QWEN35_V_TILE / QWEN35_PREFILL_V_TILE, v_heads),
      QWEN35_PREFILL_V_TILE, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(qkv),
      static_cast<const __nv_bfloat16*>(qk_norm),
      static_cast<const __nv_bfloat16*>(a),
      static_cast<const __nv_bfloat16*>(b),
      static_cast<const __nv_bfloat16*>(a_log),
      static_cast<const __nv_bfloat16*>(dt_bias),
      static_cast<float*>(recurrent), static_cast<__nv_bfloat16*>(out), seq,
      k_heads, v_heads);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_prepare_prefill_delta_step_4w(
    int* supported) {
  if (!supported) return cudaErrorInvalidValue;
  constexpr size_t shared =
      static_cast<size_t>(QWEN35_KMAX * QWEN35_V_TILE + 2 * QWEN35_KMAX + 2) *
      sizeof(float);
  cudaError_t status = cudaFuncSetAttribute(
      qwen35_prefill_delta_step_4w_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared));
  *supported = status == cudaSuccess ? 1 : 0;
  return status == cudaErrorInvalidValue ? cudaSuccess : status;
}

extern "C" cudaError_t apxinf_qwen35_prefill_delta_step_4w(
    const void* qkv, const void* qk_norm, const void* a, const void* b,
    const void* a_log, const void* dt_bias, void* recurrent, void* out,
    int seq, int k_heads, int v_heads, int kdim, int vdim,
    cudaStream_t stream) {
  if (!qkv || !qk_norm || !a || !b || !a_log || !dt_bias || !recurrent ||
      !out || seq <= 1 || seq > 512 || k_heads <= 0 || v_heads <= 0 ||
      v_heads % k_heads != 0 || kdim != QWEN35_KMAX || vdim != QWEN35_V_TILE)
    return cudaErrorInvalidValue;
  constexpr size_t shared =
      static_cast<size_t>(QWEN35_KMAX * QWEN35_V_TILE + 2 * QWEN35_KMAX + 2) *
      sizeof(float);
  qwen35_prefill_delta_step_4w_kernel<<<v_heads, 128, shared, stream>>>(
      static_cast<const __nv_bfloat16*>(qkv),
      static_cast<const __nv_bfloat16*>(qk_norm),
      static_cast<const __nv_bfloat16*>(a),
      static_cast<const __nv_bfloat16*>(b),
      static_cast<const __nv_bfloat16*>(a_log),
      static_cast<const __nv_bfloat16*>(dt_bias), static_cast<float*>(recurrent),
      static_cast<__nv_bfloat16*>(out), seq, k_heads, v_heads);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_prepare_prefill_delta_step_2w(
    int* supported) {
  if (!supported) return cudaErrorInvalidValue;
  constexpr size_t shared =
      static_cast<size_t>(QWEN35_KMAX * 64 + 2 * QWEN35_KMAX + 2) *
      sizeof(float);
  cudaError_t status = cudaFuncSetAttribute(
      qwen35_prefill_delta_step_2w_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared));
  *supported = status == cudaSuccess ? 1 : 0;
  return status == cudaErrorInvalidValue ? cudaSuccess : status;
}

extern "C" cudaError_t apxinf_qwen35_prefill_delta_step_2w(
    const void* qkv, const void* qk_norm, const void* a, const void* b,
    const void* a_log, const void* dt_bias, void* recurrent, void* out,
    int seq, int k_heads, int v_heads, int kdim, int vdim,
    cudaStream_t stream) {
  if (!qkv || !qk_norm || !a || !b || !a_log || !dt_bias || !recurrent ||
      !out || seq <= 1 || seq > 512 || k_heads <= 0 || v_heads <= 0 ||
      v_heads % k_heads != 0 || kdim != QWEN35_KMAX || vdim != QWEN35_V_TILE)
    return cudaErrorInvalidValue;
  constexpr size_t shared =
      static_cast<size_t>(QWEN35_KMAX * 64 + 2 * QWEN35_KMAX + 2) *
      sizeof(float);
  qwen35_prefill_delta_step_2w_kernel<<<dim3(2, v_heads), 64, shared, stream>>>(
      static_cast<const __nv_bfloat16*>(qkv),
      static_cast<const __nv_bfloat16*>(qk_norm),
      static_cast<const __nv_bfloat16*>(a),
      static_cast<const __nv_bfloat16*>(b),
      static_cast<const __nv_bfloat16*>(a_log),
      static_cast<const __nv_bfloat16*>(dt_bias), static_cast<float*>(recurrent),
      static_cast<__nv_bfloat16*>(out), seq, k_heads, v_heads);
  return cudaGetLastError();
}


namespace {
std::atomic<int> qwen35_norm_delta_gated_ready{-1};
}

extern "C" cudaError_t apxinf_qwen35_prepare_norm_delta_gated(
    int* supported) {
  if (supported == nullptr) return cudaErrorInvalidValue;
  int ready = qwen35_norm_delta_gated_ready.load(std::memory_order_acquire);
  if (ready >= 0) {
    *supported = ready;
    return cudaSuccess;
  }
  constexpr size_t shared =
      static_cast<size_t>(QWEN35_KMAX * QWEN35_V_TILE +
                          2 * QWEN35_KMAX) * sizeof(float) +
      static_cast<size_t>(QWEN35_V_TILE) * sizeof(__nv_bfloat16) +
      static_cast<size_t>(QWEN35_V_TILE / 32) * sizeof(float);
  int device = 0;
  int max_shared = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status == cudaSuccess) {
    status = cudaDeviceGetAttribute(
        &max_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  }
  if (status != cudaSuccess || max_shared < static_cast<int>(shared)) {
    qwen35_norm_delta_gated_ready.store(0, std::memory_order_release);
    *supported = 0;
    return cudaSuccess;
  }
  status = cudaFuncSetAttribute(
      qwen35_norm_delta_gated_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared));
  ready = status == cudaSuccess ? 1 : 0;
  qwen35_norm_delta_gated_ready.store(ready, std::memory_order_release);
  *supported = ready;
  return cudaSuccess;
}

extern "C" cudaError_t apxinf_qwen35_norm_delta_gated(
    const void* qkv, void* qk_out, const void* a, const void* b,
    const void* a_log, const void* dt_bias, const void* z,
    const void* norm_weight, void* recurrent, void* delta_out, void* out,
    int seq, int k_heads, int v_heads, int kdim, int vdim, float eps,
    cudaStream_t stream) {
  if (qkv == nullptr || qk_out == nullptr || a == nullptr || b == nullptr ||
      a_log == nullptr || dt_bias == nullptr || z == nullptr ||
      norm_weight == nullptr || recurrent == nullptr || delta_out == nullptr ||
      out == nullptr || seq <= 0 || k_heads <= 0 || v_heads <= 0 ||
      v_heads % k_heads != 0 || kdim != QWEN35_KMAX ||
      vdim != QWEN35_V_TILE || !(eps >= 0.0f) ||
      qwen35_norm_delta_gated_ready.load(std::memory_order_acquire) != 1) {
    return cudaErrorInvalidValue;
  }
  const size_t shared =
      static_cast<size_t>(kdim * vdim + 2 * kdim) * sizeof(float) +
      static_cast<size_t>(vdim) * sizeof(__nv_bfloat16) +
      static_cast<size_t>(vdim / 32) * sizeof(float);
  qwen35_norm_delta_gated_kernel<<<dim3(1, v_heads), QWEN35_V_TILE,
                                   shared, stream>>>(
      static_cast<const __nv_bfloat16*>(qkv),
      static_cast<__nv_bfloat16*>(qk_out),
      static_cast<const __nv_bfloat16*>(a),
      static_cast<const __nv_bfloat16*>(b),
      static_cast<const __nv_bfloat16*>(a_log),
      static_cast<const __nv_bfloat16*>(dt_bias),
      static_cast<const __nv_bfloat16*>(z),
      static_cast<const __nv_bfloat16*>(norm_weight),
      static_cast<float*>(recurrent),
      static_cast<__nv_bfloat16*>(delta_out),
      static_cast<__nv_bfloat16*>(out), seq, k_heads, v_heads, eps);
  return cudaGetLastError();
}

namespace {
std::atomic<int> qwen35_packed_delta_gated_ready{-1};
}

extern "C" cudaError_t apxinf_qwen35_prepare_packed_delta_gated(
    int* supported) {
  if (supported == nullptr) return cudaErrorInvalidValue;
  int ready = qwen35_packed_delta_gated_ready.load(std::memory_order_acquire);
  if (ready >= 0) {
    *supported = ready;
    return cudaSuccess;
  }
  constexpr size_t shared =
      static_cast<size_t>(QWEN35_KMAX * QWEN35_V_TILE +
                          2 * QWEN35_KMAX) * sizeof(float) +
      static_cast<size_t>(QWEN35_V_TILE) * sizeof(__nv_bfloat16) +
      static_cast<size_t>(QWEN35_V_TILE / 32) * sizeof(float);
  int device = 0;
  int max_shared = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status == cudaSuccess) {
    status = cudaDeviceGetAttribute(
        &max_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  }
  if (status != cudaSuccess || max_shared < static_cast<int>(shared)) {
    qwen35_packed_delta_gated_ready.store(0, std::memory_order_release);
    *supported = 0;
    return cudaSuccess;
  }
  status = cudaFuncSetAttribute(
      qwen35_packed_delta_gated_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared));
  ready = status == cudaSuccess ? 1 : 0;
  qwen35_packed_delta_gated_ready.store(ready, std::memory_order_release);
  *supported = ready;
  return cudaSuccess;
}

extern "C" cudaError_t apxinf_qwen35_packed_delta_gated(
    const void* qkv, const void* conv_weight, void* conv_state,
    const void* a, const void* b, const void* a_log, const void* dt_bias,
    const void* z, const void* norm_weight, void* recurrent, void* out,
    int seq, int k_heads, int v_heads, int kdim, int vdim, int conv_kernel,
    float eps, cudaStream_t stream) {
  if (qkv == nullptr || conv_weight == nullptr || conv_state == nullptr ||
      a == nullptr || b == nullptr || a_log == nullptr || dt_bias == nullptr ||
      z == nullptr || norm_weight == nullptr || recurrent == nullptr ||
      out == nullptr || seq != 1 || k_heads <= 0 || v_heads <= 0 ||
      v_heads % k_heads != 0 || kdim != QWEN35_KMAX ||
      vdim != QWEN35_V_TILE || conv_kernel != 4 || !(eps >= 0.0f) ||
      qwen35_packed_delta_gated_ready.load(std::memory_order_acquire) != 1) {
    return cudaErrorInvalidValue;
  }
  const size_t shared =
      static_cast<size_t>(kdim * vdim + 2 * kdim) * sizeof(float) +
      static_cast<size_t>(vdim) * sizeof(__nv_bfloat16) +
      static_cast<size_t>(vdim / 32) * sizeof(float);
  qwen35_packed_delta_gated_kernel<<<dim3(1, v_heads), QWEN35_V_TILE,
                                      shared, stream>>>(
      static_cast<const __nv_bfloat16*>(qkv),
      static_cast<const __nv_bfloat16*>(conv_weight),
      static_cast<float*>(conv_state), static_cast<const __nv_bfloat16*>(a),
      static_cast<const __nv_bfloat16*>(b),
      static_cast<const __nv_bfloat16*>(a_log),
      static_cast<const __nv_bfloat16*>(dt_bias),
      static_cast<const __nv_bfloat16*>(z),
      static_cast<const __nv_bfloat16*>(norm_weight),
      static_cast<float*>(recurrent), static_cast<__nv_bfloat16*>(out), seq,
      k_heads, v_heads, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_gated_norm(
    const void* input, const void* z, const void* weight, void* out,
    int seq, int v_heads, int vdim, float eps, cudaStream_t stream) {
  if (input == nullptr || z == nullptr || weight == nullptr || out == nullptr ||
      seq <= 0 || v_heads <= 0 || vdim <= 0) {
    return cudaErrorInvalidValue;
  }
  dim3 grid(seq, v_heads);
  const int threads = vdim < 256 ? vdim : 256;
  const size_t shared = static_cast<size_t>(vdim) * sizeof(float);
  qwen35_gated_norm_kernel<<<grid, threads, shared, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(z),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(out), seq, v_heads, vdim, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_q_split_norm_rope(
    const void* q_gate, const void* q_norm_w, void* q_out, void* gate_out,
    int seq, int heads, int head_dim, int rotary_dim, float theta,
    uint32_t start_pos, cudaStream_t stream) {
  if (q_gate == nullptr || q_norm_w == nullptr || q_out == nullptr ||
      gate_out == nullptr || seq <= 0 || heads <= 0 || head_dim <= 0 ||
      head_dim > 1024) {
    return cudaErrorInvalidValue;
  }
  dim3 grid(seq, heads);
  qwen35_q_split_norm_rope_kernel<<<grid, head_dim, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(q_gate),
      static_cast<const __nv_bfloat16*>(q_norm_w),
      static_cast<__nv_bfloat16*>(q_out),
      static_cast<__nv_bfloat16*>(gate_out), seq, heads, head_dim, rotary_dim,
      theta, start_pos);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_k_norm_rope_append(
    const void* k_in, const void* k_norm_w, void* k_cache, int seq,
    int n_kv_heads, int head_dim, int rotary_dim, float theta,
    uint32_t start_pos, int max_seq_len, cudaStream_t stream) {
  if (k_in == nullptr || k_norm_w == nullptr || k_cache == nullptr ||
      seq <= 0 || n_kv_heads <= 0 || head_dim <= 0 || head_dim > 1024 ||
      max_seq_len <= 0) {
    return cudaErrorInvalidValue;
  }
  dim3 grid(seq, n_kv_heads);
  qwen35_k_norm_rope_append_kernel<<<grid, head_dim, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(k_in),
      static_cast<const __nv_bfloat16*>(k_norm_w),
      static_cast<__nv_bfloat16*>(k_cache), seq, n_kv_heads, head_dim,
      rotary_dim, theta, start_pos, max_seq_len);
  return cudaGetLastError();
}
extern "C" cudaError_t apxinf_qwen35_qk_norm_rope_append(
    const void* q_gate, const void* q_norm_w, const void* k_in,
    const void* k_norm_w, void* q_out, void* gate_out, void* k_cache,
    int seq, int heads, int n_kv_heads, int head_dim, int rotary_dim,
    float theta, const uint32_t* position, int max_seq_len,
    cudaStream_t stream) {
  if (q_gate == nullptr || q_norm_w == nullptr || k_in == nullptr ||
      k_norm_w == nullptr || q_out == nullptr || gate_out == nullptr ||
      k_cache == nullptr || position == nullptr || seq != 1 || heads <= 0 ||
      n_kv_heads <= 0 || head_dim <= 0 || head_dim > 1024 ||
      max_seq_len <= 0) {
    return cudaErrorInvalidValue;
  }
  qwen35_qk_norm_rope_append_kernel<<<dim3(1, heads + n_kv_heads), head_dim,
                                      0, stream>>>(
      static_cast<const __nv_bfloat16*>(q_gate),
      static_cast<const __nv_bfloat16*>(q_norm_w),
      static_cast<const __nv_bfloat16*>(k_in),
      static_cast<const __nv_bfloat16*>(k_norm_w),
      static_cast<__nv_bfloat16*>(q_out),
      static_cast<__nv_bfloat16*>(gate_out),
      static_cast<__nv_bfloat16*>(k_cache), seq, heads, n_kv_heads,
      head_dim, rotary_dim, theta, position, max_seq_len);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_sigmoid_mul(
    const void* gate, const void* x, void* out, int64_t count,
    cudaStream_t stream) {
  if (gate == nullptr || x == nullptr || out == nullptr || count <= 0) {
    return cudaErrorInvalidValue;
  }
  constexpr int threads = 256;
  int blocks = (count + threads - 1) / threads;
  blocks = blocks > 4096 ? 4096 : blocks;
  qwen35_sigmoid_mul_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<const __nv_bfloat16*>(x),
      static_cast<__nv_bfloat16*>(out), count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_flash_prefill(
    const void* q, const void* k_cache, const void* v_cache, void* out,
    int seq, int heads, int n_kv_heads, int head_dim, float scale,
    uint32_t start_pos, int max_seq_len, cudaStream_t stream) {
  if (q == nullptr || k_cache == nullptr || v_cache == nullptr ||
      out == nullptr || seq <= 0 || heads <= 0 || n_kv_heads <= 0 ||
      head_dim <= 0 || head_dim > 1024 || heads % n_kv_heads != 0) {
    return cudaErrorInvalidValue;
  }
  dim3 grid(seq, heads);
  qwen35_flash_prefill_kernel<<<grid, head_dim, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(q),
      static_cast<const __nv_bfloat16*>(k_cache),
      static_cast<const __nv_bfloat16*>(v_cache),
      static_cast<__nv_bfloat16*>(out), seq, heads, n_kv_heads, head_dim,
      scale, start_pos, max_seq_len);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_flash_decode_gated_256(
    const void* q, const void* k_cache, const void* v_cache,
    const void* gate, void* out, float scale, const uint32_t* position,
    int max_seq_len, cudaStream_t stream) {
  if (q == nullptr || k_cache == nullptr || v_cache == nullptr ||
      gate == nullptr || out == nullptr || position == nullptr ||
      max_seq_len <= 0) {
    return cudaErrorInvalidValue;
  }
  qwen35_flash_decode_gated_256_kernel<<<24, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(q),
      static_cast<const __nv_bfloat16*>(k_cache),
      static_cast<const __nv_bfloat16*>(v_cache),
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<__nv_bfloat16*>(out), scale, position, max_seq_len);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_flash_decode_gated_256_split(
    const void* q, const void* k_cache, const void* v_cache,
    const void* gate, void* out, void* partials, float scale,
    const uint32_t* position, int max_seq_len, cudaStream_t stream) {
  if (!q || !k_cache || !v_cache || !gate || !out || !partials ||
      !position || max_seq_len <= 0)
    return cudaErrorInvalidValue;
  qwen35_flash_decode_gated_256_partial_kernel<<<dim3(24, 2), 128, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(q),
      static_cast<const __nv_bfloat16*>(k_cache),
      static_cast<const __nv_bfloat16*>(v_cache),
      static_cast<float*>(partials), scale, position, max_seq_len);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  qwen35_flash_decode_gated_256_merge_kernel<<<24, 256, 0, stream>>>(
      static_cast<const float*>(partials),
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<__nv_bfloat16*>(out));
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_flash_decode_gated_256_split_2w(
    const void* q, const void* k_cache, const void* v_cache,
    const void* gate, void* out, void* partials, float scale,
    const uint32_t* position, int max_seq_len, cudaStream_t stream) {
  if (!q || !k_cache || !v_cache || !gate || !out || !partials ||
      !position || max_seq_len <= 0)
    return cudaErrorInvalidValue;
  qwen35_flash_decode_gated_256_partial_2w_kernel<<<dim3(24, 4), 64, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(q),
      static_cast<const __nv_bfloat16*>(k_cache),
      static_cast<const __nv_bfloat16*>(v_cache), static_cast<float*>(partials),
      scale, position, max_seq_len);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  qwen35_flash_decode_gated_256_merge_kernel<<<24, 256, 0, stream>>>(
      static_cast<const float*>(partials),
      static_cast<const __nv_bfloat16*>(gate), static_cast<__nv_bfloat16*>(out));
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_flash_decode_gated_256_split_1w(
    const void* q, const void* k_cache, const void* v_cache,
    const void* gate, void* out, void* partials, float scale,
    const uint32_t* position, int max_seq_len, cudaStream_t stream) {
  if (!q || !k_cache || !v_cache || !gate || !out || !partials ||
      !position || max_seq_len <= 0)
    return cudaErrorInvalidValue;
  qwen35_flash_decode_gated_256_partial_1w_kernel<<<dim3(24, 8), 32, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(q),
      static_cast<const __nv_bfloat16*>(k_cache),
      static_cast<const __nv_bfloat16*>(v_cache), static_cast<float*>(partials),
      scale, position, max_seq_len);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  qwen35_flash_decode_gated_256_merge_kernel<<<24, 256, 0, stream>>>(
      static_cast<const float*>(partials),
      static_cast<const __nv_bfloat16*>(gate), static_cast<__nv_bfloat16*>(out));
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_flash_decode_gated_256_gqa(
    const void* q, const void* k_cache, const void* v_cache,
    const void* gate, void* out, void* partials, float scale,
    const uint32_t* position, int max_seq_len, int group,
    cudaStream_t stream) {
  if (!q || !k_cache || !v_cache || !gate || !out || !partials || !position ||
      max_seq_len <= 0 || (group != 2 && group != 3 && group != 6))
    return cudaErrorInvalidValue;
#define LAUNCH_GQA(G) \
  qwen35_flash_decode_gated_256_partial_gqa_kernel<G> \
      <<<dim3(24 / G, 8), G * 32, 0, stream>>>( \
          static_cast<const __nv_bfloat16*>(q), \
          static_cast<const __nv_bfloat16*>(k_cache), \
          static_cast<const __nv_bfloat16*>(v_cache), \
          static_cast<float*>(partials), scale, position, max_seq_len)
  if (group == 2) { LAUNCH_GQA(2); }
  else if (group == 3) { LAUNCH_GQA(3); }
  else { LAUNCH_GQA(6); }
#undef LAUNCH_GQA
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  qwen35_flash_decode_gated_256_merge_kernel<<<24, 256, 0, stream>>>(
      static_cast<const float*>(partials),
      static_cast<const __nv_bfloat16*>(gate), static_cast<__nv_bfloat16*>(out));
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_silu_mul(
    const void* gate, const void* up, void* out, int64_t count,
    cudaStream_t stream) {
  if (gate == nullptr || up == nullptr || out == nullptr || count <= 0) {
    return cudaErrorInvalidValue;
  }
  constexpr int threads = 256;
  int64_t blocks = (count + threads - 1) / threads;
  blocks = blocks > 65535 ? 65535 : blocks;
  qwen35_silu_mul_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<const __nv_bfloat16*>(up), static_cast<__nv_bfloat16*>(out),
      count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_dequant_w4a16_bf16_rows(
    const void* weight_packed, const void* weight_scale,
    const void* weight_zero_point, void* dense, int in_cols, int out_cols,
    int groups, int row_start, int row_count, cudaStream_t stream) {
  if (weight_packed == nullptr || weight_scale == nullptr ||
      weight_zero_point == nullptr || dense == nullptr || in_cols <= 0 ||
      out_cols <= 0 || groups <= 0 || in_cols % 8 != 0 ||
      in_cols % groups != 0 || row_start < 0 || row_count <= 0 ||
      row_start > out_cols - row_count) {
    return cudaErrorInvalidValue;
  }
  qwen35_dequant_w4a16_bf16_rows_kernel<<<row_count, 256, 0, stream>>>(
      static_cast<const int32_t*>(weight_packed),
      static_cast<const __nv_bfloat16*>(weight_scale),
      static_cast<const int32_t*>(weight_zero_point),
      static_cast<__nv_bfloat16*>(dense), in_cols, out_cols, groups,
      row_start, row_count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_dequant_w4a16_bf16_pair_rows(
    const void* weight_packed0, const void* weight_scale0,
    const void* weight_zero_point0, void* dense0, int out_cols0,
    const void* weight_packed1, const void* weight_scale1,
    const void* weight_zero_point1, void* dense1, int out_cols1,
    int in_cols, int groups, cudaStream_t stream) {
  if (weight_packed0 == nullptr || weight_scale0 == nullptr ||
      weight_zero_point0 == nullptr || dense0 == nullptr ||
      weight_packed1 == nullptr || weight_scale1 == nullptr ||
      weight_zero_point1 == nullptr || dense1 == nullptr ||
      out_cols0 <= 0 || out_cols1 <= 0 || in_cols <= 0 || groups <= 0 ||
      in_cols % 8 != 0 || in_cols % groups != 0 ||
      (in_cols / groups) % 8 != 0 ||
      out_cols0 > 0x7fffffff - out_cols1) {
    return cudaErrorInvalidValue;
  }
  qwen35_dequant_w4a16_bf16_pair_rows_kernel
      <<<out_cols0 + out_cols1, 256, 0, stream>>>(
          static_cast<const int32_t*>(weight_packed0),
          static_cast<const __nv_bfloat16*>(weight_scale0),
          static_cast<const int32_t*>(weight_zero_point0),
          static_cast<__nv_bfloat16*>(dense0), out_cols0,
          static_cast<const int32_t*>(weight_packed1),
          static_cast<const __nv_bfloat16*>(weight_scale1),
          static_cast<const int32_t*>(weight_zero_point1),
          static_cast<__nv_bfloat16*>(dense1), out_cols1, in_cols, groups);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_gemm_w4a16_bf16(
    const void* activation, const void* weight_packed, const void* weight_scale,
    const void* weight_zero_point, void* output, int in_cols, int out_cols,
    int groups, cudaStream_t stream) {
  if (activation == nullptr || weight_packed == nullptr ||
      weight_scale == nullptr || weight_zero_point == nullptr ||
      output == nullptr || in_cols <= 0 || in_cols % 256 != 0 ||
      out_cols <= 0 || groups <= 0 || in_cols % groups != 0 ||
      (in_cols / groups) % 8 != 0 ||
      out_cols % QWEN35_GEMM_OUT_TILE != 0) {
    return cudaErrorInvalidValue;
  }
  const size_t shared = static_cast<size_t>(QWEN35_GEMM_OUT_TILE) *
                            QWEN35_GEMM_IN_TILE +
                        QWEN35_GEMM_IN_TILE * sizeof(float);
  qwen35_gemm_w4a16_bf16_kernel<<<out_cols / QWEN35_GEMM_OUT_TILE, 128,
                                 shared, stream>>>(
      static_cast<const __nv_bfloat16*>(activation),
      static_cast<const int32_t*>(weight_packed),
      static_cast<const __nv_bfloat16*>(weight_scale),
      static_cast<const int32_t*>(weight_zero_point),
      static_cast<__nv_bfloat16*>(output), in_cols, out_cols, groups);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_attention_softmax_rows(
    const void* scores, void* l_out, int head_base, int seq, int heads,
    int visible, int row_stride, int start_pos, float scale,
    cudaStream_t stream) {
  if (scores == nullptr || l_out == nullptr || seq <= 0 || heads <= 0 ||
      visible <= 0 || row_stride < visible || start_pos < 0) {
    return cudaErrorInvalidValue;
  }
  dim3 grid(seq, heads);
  qwen35_attention_softmax_rows_kernel<<<grid, 256, 0, stream>>>(
      static_cast<float*>(const_cast<void*>(scores)),
      static_cast<float*>(l_out), head_base, seq, heads, visible, row_stride,
      start_pos, scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_v_to_f32(
    const void* v, void* vf32, int visible, int head_dim,
    cudaStream_t stream) {
  if (v == nullptr || vf32 == nullptr || visible <= 0 || head_dim <= 0) {
    return cudaErrorInvalidValue;
  }
  const int total = visible * head_dim;
  qwen35_v_to_f32_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(v), static_cast<float*>(vf32),
      visible, head_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_scale_out(
    const void* pv, const void* l, void* out, int seq, int heads,
    int head_dim, cudaStream_t stream) {
  if (pv == nullptr || l == nullptr || out == nullptr || seq <= 0 ||
      heads <= 0 || head_dim <= 0 || head_dim > 1024) {
    return cudaErrorInvalidValue;
  }
  dim3 grid(seq, heads);
  qwen35_scale_out_kernel<<<grid, head_dim, 0, stream>>>(
      static_cast<const float*>(pv), static_cast<const float*>(l),
      static_cast<__nv_bfloat16*>(out), seq, heads, head_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_scale_out_gated(
    const void* pv, const void* l, const void* gate, void* out, int seq,
    int heads, int head_dim, cudaStream_t stream) {
  if (pv == nullptr || l == nullptr || gate == nullptr || out == nullptr ||
      seq <= 1 || heads <= 0 || head_dim <= 0 || head_dim > 1024) {
    return cudaErrorInvalidValue;
  }
  dim3 grid(seq, heads);
  qwen35_scale_out_gated_kernel<<<grid, head_dim, 0, stream>>>(
      static_cast<const float*>(pv), static_cast<const float*>(l),
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<__nv_bfloat16*>(out), seq, heads, head_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_transpose_kt(
    const void* k, void* kt, int visible, int head_dim, cudaStream_t stream) {
  if (k == nullptr || kt == nullptr || visible <= 0 || head_dim <= 0) {
    return cudaErrorInvalidValue;
  }
  dim3 grid((head_dim + 31) / 32, (visible + 31) / 32);
  qwen35_transpose_kt_kernel<<<grid, dim3(32, 8), 0, stream>>>(
      static_cast<const __nv_bfloat16*>(k), static_cast<__nv_bfloat16*>(kt),
      visible, head_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_gemm_w4a16_bf16_tc(
    const void* activation, const void* weight_packed, const void* weight_scale,
    const void* weight_zero_point, void* output, int in_cols, int out_cols,
    int groups, cudaStream_t stream) {
  if (activation == nullptr || weight_packed == nullptr ||
      weight_scale == nullptr || weight_zero_point == nullptr ||
      output == nullptr || in_cols <= 0 || in_cols % 128 != 0 ||
      out_cols <= 0 || out_cols % QWEN35_TC_OUT_TILE != 0 || groups <= 0) {
    return cudaErrorInvalidValue;
  }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  qwen35_gemm_w4a16_bf16_tc_kernel<false>
      <<<out_cols / QWEN35_TC_OUT_TILE, 256, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(activation),
          static_cast<const int32_t*>(weight_packed),
          static_cast<const __nv_bfloat16*>(weight_scale),
          static_cast<const int32_t*>(weight_zero_point),
          static_cast<__nv_bfloat16*>(output), out_cols, nullptr, nullptr,
          nullptr, nullptr, in_cols, groups);
#else
  (void)activation; (void)weight_packed; (void)weight_scale;
  (void)weight_zero_point; (void)output; (void)in_cols; (void)out_cols;
  (void)groups; (void)stream;
#endif
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_gemm_w4a16_bf16_tc_cache_hint(
    const void* activation, const void* weight_packed, const void* weight_scale,
    const void* weight_zero_point, void* output, int in_cols, int out_cols,
    int groups, cudaStream_t stream) {
  if (activation == nullptr || weight_packed == nullptr ||
      weight_scale == nullptr || weight_zero_point == nullptr ||
      output == nullptr || in_cols <= 0 || in_cols % 128 != 0 ||
      out_cols <= 0 || out_cols % QWEN35_TC_OUT_TILE != 0 || groups <= 0 ||
      in_cols % groups != 0 || in_cols / groups != 32) {
    return cudaErrorInvalidValue;
  }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  qwen35_gemm_w4a16_bf16_tc_kernel<false, true>
      <<<out_cols / QWEN35_TC_OUT_TILE, 256, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(activation),
          static_cast<const int32_t*>(weight_packed),
          static_cast<const __nv_bfloat16*>(weight_scale),
          static_cast<const int32_t*>(weight_zero_point),
          static_cast<__nv_bfloat16*>(output), out_cols, nullptr, nullptr,
          nullptr, nullptr, in_cols, groups);
#else
  (void)activation; (void)weight_packed; (void)weight_scale;
  (void)weight_zero_point; (void)output; (void)in_cols; (void)out_cols;
  (void)groups; (void)stream;
#endif
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_gemm_w4a16_bf16_tc_scale_epilogue(
    const void* activation, const void* weight_packed, const void* weight_scale,
    const void* weight_zero_point, void* output, int in_cols, int out_cols,
    int groups, cudaStream_t stream) {
  if (activation == nullptr || weight_packed == nullptr ||
      weight_scale == nullptr || weight_zero_point == nullptr ||
      output == nullptr || in_cols <= 0 || in_cols % 128 != 0 ||
      out_cols <= 0 || out_cols % QWEN35_TC_OUT_TILE != 0 || groups <= 0 ||
      in_cols % groups != 0 || in_cols / groups != 32) {
    return cudaErrorInvalidValue;
  }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  qwen35_gemm_w4a16_bf16_tc_scale_epilogue_kernel
      <<<out_cols / QWEN35_TC_OUT_TILE, 256, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(activation),
          static_cast<const int32_t*>(weight_packed),
          static_cast<const __nv_bfloat16*>(weight_scale),
          static_cast<const int32_t*>(weight_zero_point),
          static_cast<__nv_bfloat16*>(output), in_cols, groups);
#else
  (void)activation; (void)weight_packed; (void)weight_scale;
  (void)weight_zero_point; (void)output; (void)in_cols; (void)out_cols;
  (void)groups; (void)stream;
#endif
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_gemm_w4a16_bf16_qkv_10240x5120(
    const void* activation, const void* weight_packed, const void* weight_scale,
    const void* weight_zero_point, void* output, int in_cols, int out_cols,
    int groups, cudaStream_t stream) {
  if (activation == nullptr || weight_packed == nullptr ||
      weight_scale == nullptr || weight_zero_point == nullptr ||
      output == nullptr || in_cols != 5120 || out_cols != 10240 ||
      groups != 160) return cudaErrorInvalidValue;
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  qwen35_gemm_w4a16_bf16_qkv_10240x5120_kernel
      <<<10240 / QWEN35_TC_OUT_TILE, 128, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(activation),
          static_cast<const int32_t*>(weight_packed),
          static_cast<const __nv_bfloat16*>(weight_scale),
          static_cast<const int32_t*>(weight_zero_point),
          static_cast<__nv_bfloat16*>(output));
#else
  (void)stream;
#endif
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_gemm_w4a16_bf16_qkv_sched(
    const void* activation, const void* weight_packed, const void* weight_scale,
    const void* weight_zero_point, void* output, int in_cols, int out_cols,
    int groups, int mode, cudaStream_t stream) {
  if (!activation || !weight_packed || !weight_scale || !weight_zero_point ||
      !output || in_cols != 5120 || out_cols != 10240 || groups != 160 ||
      mode < 0 || mode > 7)
    return cudaErrorInvalidValue;
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define LAUNCH_QKV(R, A, M) \
  qwen35_gemm_w4a16_bf16_qkv_sched_kernel<R, A, M> \
      <<<10240 / R, R * 2, 0, stream>>>( \
          static_cast<const __nv_bfloat16*>(activation), \
          static_cast<const int32_t*>(weight_packed), \
          static_cast<const __nv_bfloat16*>(weight_scale), \
          static_cast<const int32_t*>(weight_zero_point), \
          static_cast<__nv_bfloat16*>(output))
  switch (mode) {
    case 0: LAUNCH_QKV(64, 1, 1); break;
    case 1: LAUNCH_QKV(64, 2, 1); break;
    case 2: LAUNCH_QKV(64, 1, 2); break;
    case 3: LAUNCH_QKV(64, 2, 2); break;
    case 4: LAUNCH_QKV(128, 1, 1); break;
    case 5: LAUNCH_QKV(128, 2, 1); break;
    case 6: LAUNCH_QKV(128, 1, 2); break;
    case 7: LAUNCH_QKV(128, 2, 2); break;
  }
#undef LAUNCH_QKV
  return cudaGetLastError();
#else
  (void)stream;
  return cudaErrorNotSupported;
#endif
}

extern "C" cudaError_t apxinf_qwen35_gemm_w4a16_bf16_tc_tile_alt(

    const void* activation, const void* weight_packed, const void* weight_scale,
    const void* weight_zero_point, void* output, int in_cols, int out_cols,
    int groups, cudaStream_t stream) {
  if (activation == nullptr || weight_packed == nullptr ||
      weight_scale == nullptr || weight_zero_point == nullptr ||
      output == nullptr || in_cols <= 0 || in_cols % 128 != 0 ||
      out_cols <= 0 || out_cols % QWEN35_TC_ALT_OUT_TILE != 0 || groups <= 0 ||
      in_cols % groups != 0 || in_cols / groups != 32) {
    return cudaErrorInvalidValue;
  }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  qwen35_gemm_w4a16_bf16_tc_tile_alt_kernel
      <<<out_cols / QWEN35_TC_ALT_OUT_TILE, 128, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(activation),
          static_cast<const int32_t*>(weight_packed),
          static_cast<const __nv_bfloat16*>(weight_scale),
          static_cast<const int32_t*>(weight_zero_point),
          static_cast<__nv_bfloat16*>(output), in_cols, groups);
#else
  (void)activation; (void)weight_packed; (void)weight_scale;
  (void)weight_zero_point; (void)output; (void)in_cols; (void)out_cols;
  (void)groups; (void)stream;
#endif
  return cudaGetLastError();
}
extern "C" cudaError_t apxinf_qwen35_gemm_w4a16_bf16_gate_up_silu(
    const void* activation,
    const void* gate_packed, const void* gate_scale,
    const void* gate_zero_point,
    const void* up_packed, const void* up_scale,
    const void* up_zero_point, void* output,
    int in_cols, int out_cols, int groups, cudaStream_t stream) {
  if (!activation || !gate_packed || !gate_scale || !gate_zero_point ||
      !up_packed || !up_scale || !up_zero_point || !output ||
      in_cols != 5120 || out_cols != 17408 || groups != 160)
    return cudaErrorInvalidValue;
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  qwen35_gemm_w4a16_bf16_gate_up_silu_kernel
      <<<out_cols / QWEN35_TC_ALT_OUT_TILE, 128, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(activation),
          static_cast<const int32_t*>(gate_packed),
          static_cast<const __nv_bfloat16*>(gate_scale),
          static_cast<const int32_t*>(gate_zero_point),
          static_cast<const int32_t*>(up_packed),
          static_cast<const __nv_bfloat16*>(up_scale),
          static_cast<const int32_t*>(up_zero_point),
          static_cast<__nv_bfloat16*>(output), in_cols, out_cols, groups);
#endif
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_gemm_w4a16_bf16_tc_vector_mma(
    const void* activation, const void* weight_packed, const void* weight_scale,
    const void* weight_zero_point, void* output, int in_cols, int out_cols,
    int groups, cudaStream_t stream) {
  if (activation == nullptr || weight_packed == nullptr ||
      weight_scale == nullptr || weight_zero_point == nullptr ||
      output == nullptr || in_cols <= 0 || in_cols % 128 != 0 ||
      out_cols <= 0 || out_cols % QWEN35_TC_OUT_TILE != 0 || groups <= 0 ||
      in_cols % groups != 0 || in_cols / groups != 32) {
    return cudaErrorInvalidValue;
  }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  qwen35_gemm_w4a16_bf16_tc_vector_mma_kernel<false>
      <<<out_cols / QWEN35_TC_OUT_TILE, 256, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(activation),
          static_cast<const int32_t*>(weight_packed),
          static_cast<const __nv_bfloat16*>(weight_scale),
          static_cast<const int32_t*>(weight_zero_point),
          static_cast<__nv_bfloat16*>(output), out_cols, nullptr, nullptr,
          nullptr, nullptr, in_cols, groups);
#else
  (void)activation; (void)weight_packed; (void)weight_scale;
  (void)weight_zero_point; (void)output; (void)in_cols; (void)out_cols;
  (void)groups; (void)stream;
#endif
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_gemm_w4a16_bf16_tc_meta_shared(
    const void* activation, const void* weight_packed, const void* weight_scale,
    const void* weight_zero_point, void* output, int in_cols, int out_cols,
    int groups, cudaStream_t stream) {
  if (activation == nullptr || weight_packed == nullptr ||
      weight_scale == nullptr || weight_zero_point == nullptr ||
      output == nullptr || in_cols <= 0 || in_cols % 128 != 0 ||
      out_cols <= 0 || out_cols % QWEN35_TC_OUT_TILE != 0 || groups <= 0 ||
      in_cols % groups != 0 || in_cols / groups != 32) {
    return cudaErrorInvalidValue;
  }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  qwen35_gemm_w4a16_bf16_tc_meta_shared_kernel
      <<<out_cols / QWEN35_TC_OUT_TILE, 256, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(activation),
          static_cast<const int32_t*>(weight_packed),
          static_cast<const __nv_bfloat16*>(weight_scale),
          static_cast<const int32_t*>(weight_zero_point),
          static_cast<__nv_bfloat16*>(output), in_cols, groups);
#else
  (void)activation; (void)weight_packed; (void)weight_scale;
  (void)weight_zero_point; (void)output; (void)in_cols; (void)out_cols;
  (void)groups; (void)stream;
#endif
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_gemm_w4a16_bf16_tc_persistent(
    const void* activation, const void* weight_packed, const void* weight_scale,
    const void* weight_zero_point, void* output, int in_cols, int out_cols,
    int groups, cudaStream_t stream) {
  if (activation == nullptr || weight_packed == nullptr ||
      weight_scale == nullptr || weight_zero_point == nullptr ||
      output == nullptr || in_cols <= 0 || in_cols % 128 != 0 ||
      out_cols < 2 * QWEN35_TC_OUT_TILE ||
      out_cols % QWEN35_TC_OUT_TILE != 0 || groups <= 0 ||
      in_cols % groups != 0 || in_cols / groups != 32) {
    return cudaErrorInvalidValue;
  }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  const int output_blocks = out_cols / QWEN35_TC_OUT_TILE;
  const int persistent_blocks = (output_blocks + 1) / 2;
  qwen35_gemm_w4a16_bf16_tc_persistent_kernel
      <<<persistent_blocks, 256, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(activation),
          static_cast<const int32_t*>(weight_packed),
          static_cast<const __nv_bfloat16*>(weight_scale),
          static_cast<const int32_t*>(weight_zero_point),
          static_cast<__nv_bfloat16*>(output), in_cols, out_cols, groups);
#else
  (void)activation; (void)weight_packed; (void)weight_scale;
  (void)weight_zero_point; (void)output; (void)in_cols; (void)out_cols;
  (void)groups; (void)stream;
#endif
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_gemm_w4a16_bf16_tc_repacked_v1(
    const void* activation, const void* weight_packed, const void* weight_scale,
    const void* weight_zero_point, void* output, int in_cols, int out_cols,
    int padded_out_cols, int groups, cudaStream_t stream) {
  if (activation == nullptr || weight_packed == nullptr ||
      weight_scale == nullptr || weight_zero_point == nullptr ||
      output == nullptr || in_cols <= 0 || in_cols % 128 != 0 ||
      out_cols <= 0 || padded_out_cols < out_cols ||
      padded_out_cols % QWEN35_TC_OUT_TILE != 0 || groups <= 0 ||
      in_cols / groups != 32) {
    return cudaErrorInvalidValue;
  }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  qwen35_gemm_w4a16_bf16_tc_repacked_v1_kernel
      <<<(out_cols + QWEN35_TC_OUT_TILE - 1) / QWEN35_TC_OUT_TILE, 256, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(activation),
          static_cast<const int32_t*>(weight_packed),
          static_cast<const __nv_bfloat16*>(weight_scale),
          static_cast<const int32_t*>(weight_zero_point),
          static_cast<__nv_bfloat16*>(output), in_cols, out_cols, groups);
#else
  (void)activation; (void)weight_packed; (void)weight_scale;
  (void)weight_zero_point; (void)output; (void)in_cols; (void)out_cols;
  (void)padded_out_cols; (void)groups; (void)stream;
#endif
  return cudaGetLastError();
}
extern "C" cudaError_t apxinf_qwen35_gemm_w4a16_bf16_tc_w4_transform_cache(
    const void* activation, const void* weight_packed, const void* weight_scale,
    const void* weight_zero_point, void* output, int in_cols, int out_cols,
    int padded_out_cols, int groups, cudaStream_t stream) {
  if (activation == nullptr || weight_packed == nullptr ||
      weight_scale == nullptr || weight_zero_point == nullptr || output == nullptr ||
      in_cols <= 0 || in_cols % 128 != 0 || out_cols <= 0 ||
      padded_out_cols < out_cols || padded_out_cols % QWEN35_TC_OUT_TILE != 0 ||
      groups <= 0 || in_cols / groups != 32) {
    return cudaErrorInvalidValue;
  }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  // Transform-cache storage has explicit N64/K16 physical indexing.  Reuse
  // the exact repacked arithmetic without any per-token transform or copy.
  qwen35_gemm_w4a16_bf16_tc_repacked_v1_kernel
      <<<(out_cols + QWEN35_TC_OUT_TILE - 1) / QWEN35_TC_OUT_TILE, 256, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(activation),
          static_cast<const int32_t*>(weight_packed),
          static_cast<const __nv_bfloat16*>(weight_scale),
          static_cast<const int32_t*>(weight_zero_point),
          static_cast<__nv_bfloat16*>(output), in_cols, out_cols, groups);
#else
  (void)activation; (void)weight_packed; (void)weight_scale;
  (void)weight_zero_point; (void)output; (void)in_cols; (void)out_cols;
  (void)padded_out_cols; (void)groups; (void)stream;
#endif
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_gemm_w4a16_bf16_tc_pair(
    const void* activation, const void* weight_packed0,
    const void* weight_scale0, const void* weight_zero_point0, void* output0,
    int out_cols0, const void* weight_packed1, const void* weight_scale1,
    const void* weight_zero_point1, void* output1, int out_cols1, int in_cols,
    int groups, cudaStream_t stream) {
  if (activation == nullptr || weight_packed0 == nullptr ||
      weight_scale0 == nullptr || weight_zero_point0 == nullptr ||
      output0 == nullptr || weight_packed1 == nullptr ||
      weight_scale1 == nullptr || weight_zero_point1 == nullptr ||
      output1 == nullptr || in_cols <= 0 || in_cols % 128 != 0 ||
      out_cols0 <= 0 || out_cols0 % QWEN35_TC_OUT_TILE != 0 ||
      out_cols1 <= 0 || out_cols1 % QWEN35_TC_OUT_TILE != 0 || groups <= 0) {
    return cudaErrorInvalidValue;
  }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  const int blocks = (out_cols0 + out_cols1) / QWEN35_TC_OUT_TILE;
  qwen35_gemm_w4a16_bf16_tc_kernel<true><<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(activation),
      static_cast<const int32_t*>(weight_packed0),
      static_cast<const __nv_bfloat16*>(weight_scale0),
      static_cast<const int32_t*>(weight_zero_point0),
      static_cast<__nv_bfloat16*>(output0), out_cols0,
      static_cast<const int32_t*>(weight_packed1),
      static_cast<const __nv_bfloat16*>(weight_scale1),
      static_cast<const int32_t*>(weight_zero_point1),
      static_cast<__nv_bfloat16*>(output1), in_cols, groups);
#else
  (void)activation; (void)weight_packed0; (void)weight_scale0;
  (void)weight_zero_point0; (void)output0; (void)out_cols0;
  (void)weight_packed1; (void)weight_scale1; (void)weight_zero_point1;
  (void)output1; (void)out_cols1; (void)in_cols; (void)groups; (void)stream;
#endif
  return cudaGetLastError();
}
extern "C" cudaError_t apxinf_qwen35_gemm_w4a16_bf16_tc_store_alt_single(
    const void* activation, const void* weight_packed,
    const void* weight_scale, const void* weight_zero_point, void* output,
    int in_cols, int out_cols, int groups, cudaStream_t stream) {
  if (activation == nullptr || weight_packed == nullptr ||
      weight_scale == nullptr || weight_zero_point == nullptr || output == nullptr ||
      in_cols <= 0 || in_cols % 128 != 0 || out_cols <= 0 ||
      out_cols % QWEN35_TC_OUT_TILE != 0 || groups <= 0 ||
      in_cols % groups != 0 || in_cols / groups != 32 ||
      (reinterpret_cast<uintptr_t>(output) & 3U) != 0) {
    return cudaErrorInvalidValue;
  }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  qwen35_gemm_w4a16_bf16_tc_store_alt_kernel<false>
      <<<out_cols / QWEN35_TC_OUT_TILE, 256, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(activation),
          static_cast<const int32_t*>(weight_packed),
          static_cast<const __nv_bfloat16*>(weight_scale),
          static_cast<const int32_t*>(weight_zero_point),
          static_cast<__nv_bfloat16*>(output), out_cols, nullptr, nullptr,
          nullptr, nullptr, in_cols, groups);
#else
  (void)activation; (void)weight_packed; (void)weight_scale;
  (void)weight_zero_point; (void)output; (void)in_cols; (void)out_cols;
  (void)groups; (void)stream;
#endif
  return cudaGetLastError();
}
extern "C" cudaError_t apxinf_qwen35_gemm_w4a16_bf16_tc_store_alt(
    const void* activation, const void* weight_packed0,
    const void* weight_scale0, const void* weight_zero_point0, void* output0,
    int out_cols0, const void* weight_packed1, const void* weight_scale1,
    const void* weight_zero_point1, void* output1, int out_cols1, int in_cols,
    int groups, cudaStream_t stream) {
  if (activation == nullptr || weight_packed0 == nullptr ||
      weight_scale0 == nullptr || weight_zero_point0 == nullptr ||
      output0 == nullptr || weight_packed1 == nullptr ||
      weight_scale1 == nullptr || weight_zero_point1 == nullptr ||
      output1 == nullptr || in_cols <= 0 || in_cols % 128 != 0 ||
      out_cols0 <= 0 || out_cols1 <= 0 ||
      out_cols0 % QWEN35_TC_OUT_TILE != 0 ||
      out_cols1 % QWEN35_TC_OUT_TILE != 0 || groups <= 0 ||
      in_cols % groups != 0 || in_cols / groups != 32 ||
      (reinterpret_cast<uintptr_t>(output0) & 3U) != 0 ||
      (reinterpret_cast<uintptr_t>(output1) & 3U) != 0) {
    return cudaErrorInvalidValue;
  }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  const int blocks = (out_cols0 + out_cols1) / QWEN35_TC_OUT_TILE;
  qwen35_gemm_w4a16_bf16_tc_store_alt_kernel<true>
      <<<blocks, 256, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(activation),
          static_cast<const int32_t*>(weight_packed0),
          static_cast<const __nv_bfloat16*>(weight_scale0),
          static_cast<const int32_t*>(weight_zero_point0),
          static_cast<__nv_bfloat16*>(output0), out_cols0,
          static_cast<const int32_t*>(weight_packed1),
          static_cast<const __nv_bfloat16*>(weight_scale1),
          static_cast<const int32_t*>(weight_zero_point1),
          static_cast<__nv_bfloat16*>(output1), in_cols, groups);
#else
  (void)activation; (void)weight_packed0; (void)weight_scale0;
  (void)weight_zero_point0; (void)output0; (void)out_cols0;
  (void)weight_packed1; (void)weight_scale1; (void)weight_zero_point1;
  (void)output1; (void)out_cols1; (void)in_cols; (void)groups; (void)stream;
#endif
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_gemm_w4a16_bf16_tc_pair_coarsen(
    const void* activation, const void* weight_packed0,
    const void* weight_scale0, const void* weight_zero_point0, void* output0,
    int out_cols0, const void* weight_packed1, const void* weight_scale1,
    const void* weight_zero_point1, void* output1, int out_cols1, int in_cols,
    int groups, cudaStream_t stream) {
  if (activation == nullptr || weight_packed0 == nullptr ||
      weight_scale0 == nullptr || weight_zero_point0 == nullptr ||
      output0 == nullptr || weight_packed1 == nullptr ||
      weight_scale1 == nullptr || weight_zero_point1 == nullptr ||
      output1 == nullptr || in_cols <= 0 || in_cols % 128 != 0 ||
      out_cols0 <= 0 || out_cols1 <= 0 || out_cols0 % (2 * QWEN35_TC_OUT_TILE) != 0 ||
      out_cols1 % (2 * QWEN35_TC_OUT_TILE) != 0 || groups <= 0 ||
      in_cols % groups != 0 || in_cols / groups != 32 ||
      out_cols0 > INT32_MAX - (2 * QWEN35_TC_OUT_TILE - 1) ||
      out_cols1 > INT32_MAX - (2 * QWEN35_TC_OUT_TILE - 1)) {
    return cudaErrorInvalidValue;
  }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  const int blocks0 = out_cols0 / (2 * QWEN35_TC_OUT_TILE);
  const int blocks1 = out_cols1 / (2 * QWEN35_TC_OUT_TILE);
  if (blocks0 > INT32_MAX - blocks1) return cudaErrorInvalidValue;
  qwen35_gemm_w4a16_bf16_tc_pair_coarsen_kernel<<<blocks0 + blocks1, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(activation),
      static_cast<const int32_t*>(weight_packed0),
      static_cast<const __nv_bfloat16*>(weight_scale0),
      static_cast<const int32_t*>(weight_zero_point0),
      static_cast<__nv_bfloat16*>(output0), out_cols0,
      static_cast<const int32_t*>(weight_packed1),
      static_cast<const __nv_bfloat16*>(weight_scale1),
      static_cast<const int32_t*>(weight_zero_point1),
      static_cast<__nv_bfloat16*>(output1), out_cols1, in_cols, groups);
#else
  (void)activation; (void)weight_packed0; (void)weight_scale0;
  (void)weight_zero_point0; (void)output0; (void)out_cols0;
  (void)weight_packed1; (void)weight_scale1; (void)weight_zero_point1;
  (void)output1; (void)out_cols1; (void)in_cols; (void)groups; (void)stream;
#endif
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_gemm_w4a16_bf16_tc_pair_warp(
    const void* activation, const void* weight_packed0,
    const void* weight_scale0, const void* weight_zero_point0, void* output0,
    int out_cols0, const void* weight_packed1, const void* weight_scale1,
    const void* weight_zero_point1, void* output1, int out_cols1, int in_cols,
    int groups, cudaStream_t stream) {
  const bool supported_in =
      in_cols == 5120 || in_cols == 6144 || in_cols == 17408;
  const bool supported_out0 = out_cols0 == 1024 || out_cols0 == 5120 ||
      out_cols0 == 6144 || out_cols0 == 10240 || out_cols0 == 12288 ||
      out_cols0 == 17408;
  const bool supported_out1 = out_cols1 == 1024 || out_cols1 == 5120 ||
      out_cols1 == 6144 || out_cols1 == 10240 || out_cols1 == 12288 ||
      out_cols1 == 17408;
  if (activation == nullptr || weight_packed0 == nullptr ||
      weight_scale0 == nullptr || weight_zero_point0 == nullptr ||
      output0 == nullptr || weight_packed1 == nullptr ||
      weight_scale1 == nullptr || weight_zero_point1 == nullptr ||
      output1 == nullptr || !supported_in || !supported_out0 ||
      !supported_out1 || out_cols0 % QWEN35_TC_OUT_TILE != 0 ||
      out_cols1 % QWEN35_TC_OUT_TILE != 0 || groups <= 0 ||
      in_cols % groups != 0 || in_cols / groups != 32) {
    return cudaErrorInvalidValue;
  }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  const int blocks = (out_cols0 + out_cols1) / QWEN35_TC_OUT_TILE;
  qwen35_gemm_w4a16_bf16_tc_pair_warp_kernel<<<blocks, 384, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(activation),
      static_cast<const int32_t*>(weight_packed0),
      static_cast<const __nv_bfloat16*>(weight_scale0),
      static_cast<const int32_t*>(weight_zero_point0),
      static_cast<__nv_bfloat16*>(output0), out_cols0,
      static_cast<const int32_t*>(weight_packed1),
      static_cast<const __nv_bfloat16*>(weight_scale1),
      static_cast<const int32_t*>(weight_zero_point1),
      static_cast<__nv_bfloat16*>(output1), in_cols, groups);
#else
  (void)activation; (void)weight_packed0; (void)weight_scale0;
  (void)weight_zero_point0; (void)output0; (void)out_cols0;
  (void)weight_packed1; (void)weight_scale1; (void)weight_zero_point1;
  (void)output1; (void)out_cols1; (void)in_cols; (void)groups; (void)stream;
#endif
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_gemm_w4a16_bf16_tc_pair_reg(
    const void* activation, const void* weight_packed0,
    const void* weight_scale0, const void* weight_zero_point0, void* output0,
    int out_cols0, const void* weight_packed1, const void* weight_scale1,
    const void* weight_zero_point1, void* output1, int out_cols1, int in_cols,
    int groups, cudaStream_t stream) {
  if (activation == nullptr || weight_packed0 == nullptr ||
      weight_scale0 == nullptr || weight_zero_point0 == nullptr ||
      output0 == nullptr || weight_packed1 == nullptr ||
      weight_scale1 == nullptr || weight_zero_point1 == nullptr ||
      output1 == nullptr || in_cols <= 0 || in_cols % 16 != 0 ||
      out_cols0 <= 0 || out_cols0 % QWEN35_TC_OUT_TILE != 0 ||
      out_cols1 <= 0 || out_cols1 % QWEN35_TC_OUT_TILE != 0 || groups <= 0) {
    return cudaErrorInvalidValue;
  }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  const int blocks = (out_cols0 + out_cols1) / QWEN35_TC_OUT_TILE;
  qwen35_gemm_w4a16_bf16_tc_pair_reg_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(activation),
      static_cast<const int32_t*>(weight_packed0),
      static_cast<const __nv_bfloat16*>(weight_scale0),
      static_cast<const int32_t*>(weight_zero_point0),
      static_cast<__nv_bfloat16*>(output0), out_cols0,
      static_cast<const int32_t*>(weight_packed1),
      static_cast<const __nv_bfloat16*>(weight_scale1),
      static_cast<const int32_t*>(weight_zero_point1),
      static_cast<__nv_bfloat16*>(output1), in_cols, groups);
#else
  (void)activation; (void)weight_packed0; (void)weight_scale0;
  (void)weight_zero_point0; (void)output0; (void)out_cols0;
  (void)weight_packed1; (void)weight_scale1; (void)weight_zero_point1;
  (void)output1; (void)out_cols1; (void)in_cols; (void)groups; (void)stream;
#endif
  return cudaGetLastError();
}
extern "C" cudaError_t apxinf_qwen35_gemm_w4a16_bf16_tc_pair_occupancy(
    const void* activation, const void* weight_packed0,
    const void* weight_scale0, const void* weight_zero_point0, void* output0,
    int out_cols0, const void* weight_packed1, const void* weight_scale1,
    const void* weight_zero_point1, void* output1, int out_cols1, int in_cols,
    int groups, cudaStream_t stream) {
  if (activation == nullptr || weight_packed0 == nullptr ||
      weight_scale0 == nullptr || weight_zero_point0 == nullptr ||
      output0 == nullptr || weight_packed1 == nullptr ||
      weight_scale1 == nullptr || weight_zero_point1 == nullptr ||
      output1 == nullptr || in_cols <= 0 || in_cols % 128 != 0 ||
      out_cols0 <= 0 || out_cols0 % QWEN35_TC_OUT_TILE != 0 ||
      out_cols1 <= 0 || out_cols1 % QWEN35_TC_OUT_TILE != 0 || groups <= 0 ||
      in_cols % groups != 0 || in_cols / groups != 32) {
    return cudaErrorInvalidValue;
  }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  const int blocks = (out_cols0 + out_cols1) / QWEN35_TC_OUT_TILE;
  qwen35_gemm_w4a16_bf16_tc_pair_occupancy_kernel<3><<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(activation),
      static_cast<const int32_t*>(weight_packed0),
      static_cast<const __nv_bfloat16*>(weight_scale0),
      static_cast<const int32_t*>(weight_zero_point0),
      static_cast<__nv_bfloat16*>(output0), out_cols0,
      static_cast<const int32_t*>(weight_packed1),
      static_cast<const __nv_bfloat16*>(weight_scale1),
      static_cast<const int32_t*>(weight_zero_point1),
      static_cast<__nv_bfloat16*>(output1), in_cols, groups);
#else
  (void)activation; (void)weight_packed0; (void)weight_scale0;
  (void)weight_zero_point0; (void)output0; (void)out_cols0;
  (void)weight_packed1; (void)weight_scale1; (void)weight_zero_point1;
  (void)output1; (void)out_cols1; (void)in_cols; (void)groups; (void)stream;
#endif
  return cudaGetLastError();
}
extern "C" cudaError_t apxinf_qwen35_gemm_w4a16_bf16_tc_pair_vector_mma(
    const void* activation, const void* weight_packed0,
    const void* weight_scale0, const void* weight_zero_point0, void* output0,
    int out_cols0, const void* weight_packed1, const void* weight_scale1,
    const void* weight_zero_point1, void* output1, int out_cols1, int in_cols,
    int groups, cudaStream_t stream) {
  if (activation == nullptr || weight_packed0 == nullptr ||
      weight_scale0 == nullptr || weight_zero_point0 == nullptr ||
      output0 == nullptr || weight_packed1 == nullptr ||
      weight_scale1 == nullptr || weight_zero_point1 == nullptr ||
      output1 == nullptr || in_cols <= 0 || in_cols % 128 != 0 ||
      out_cols0 <= 0 || out_cols0 % QWEN35_TC_OUT_TILE != 0 ||
      out_cols1 <= 0 || out_cols1 % QWEN35_TC_OUT_TILE != 0 || groups <= 0 ||
      in_cols % groups != 0 || in_cols / groups != 32) {
    return cudaErrorInvalidValue;
  }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  const int blocks = (out_cols0 + out_cols1) / QWEN35_TC_OUT_TILE;
  qwen35_gemm_w4a16_bf16_tc_vector_mma_kernel<true>
      <<<blocks, 256, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(activation),
          static_cast<const int32_t*>(weight_packed0),
          static_cast<const __nv_bfloat16*>(weight_scale0),
          static_cast<const int32_t*>(weight_zero_point0),
          static_cast<__nv_bfloat16*>(output0), out_cols0,
          static_cast<const int32_t*>(weight_packed1),
          static_cast<const __nv_bfloat16*>(weight_scale1),
          static_cast<const int32_t*>(weight_zero_point1),
          static_cast<__nv_bfloat16*>(output1), in_cols, groups);
#else
  (void)activation; (void)weight_packed0; (void)weight_scale0;
  (void)weight_zero_point0; (void)output0; (void)out_cols0;
  (void)weight_packed1; (void)weight_scale1; (void)weight_zero_point1;
  (void)output1; (void)out_cols1; (void)in_cols; (void)groups; (void)stream;
#endif
  return cudaGetLastError();
}


extern "C" cudaError_t apxinf_qwen35_gemm_w4a16_bf16_tc_pair_meta(
    const void* activation, const void* weight_packed0,
    const void* weight_scale0, const void* weight_zero_point0, void* output0,
    int out_cols0, const void* weight_packed1, const void* weight_scale1,
    const void* weight_zero_point1, void* output1, int out_cols1, int in_cols,
    int groups, cudaStream_t stream) {
  if (activation == nullptr || weight_packed0 == nullptr ||
      weight_scale0 == nullptr || weight_zero_point0 == nullptr ||
      output0 == nullptr || weight_packed1 == nullptr || weight_scale1 == nullptr ||
      weight_zero_point1 == nullptr || output1 == nullptr || in_cols <= 0 ||
      in_cols % 128 != 0 || out_cols0 <= 0 || out_cols1 <= 0 ||
      out_cols0 % QWEN35_TC_OUT_TILE != 0 ||
      out_cols1 % QWEN35_TC_OUT_TILE != 0 || groups <= 0 ||
      in_cols % groups != 0 || in_cols / groups != 32) {
    return cudaErrorInvalidValue;
  }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  const int blocks = (out_cols0 + out_cols1) / QWEN35_TC_OUT_TILE;
  qwen35_gemm_w4a16_bf16_tc_pair_meta_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(activation),
      static_cast<const int32_t*>(weight_packed0),
      static_cast<const __nv_bfloat16*>(weight_scale0),
      static_cast<const int32_t*>(weight_zero_point0),
      static_cast<__nv_bfloat16*>(output0), out_cols0,
      static_cast<const int32_t*>(weight_packed1),
      static_cast<const __nv_bfloat16*>(weight_scale1),
      static_cast<const int32_t*>(weight_zero_point1),
      static_cast<__nv_bfloat16*>(output1), in_cols, groups);
#else
  (void)activation; (void)weight_packed0; (void)weight_scale0;
  (void)weight_zero_point0; (void)output0; (void)out_cols0;
  (void)weight_packed1; (void)weight_scale1; (void)weight_zero_point1;
  (void)output1; (void)out_cols1; (void)in_cols; (void)groups; (void)stream;
#endif
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_gemm_w4a16_bf16_tc_pair_weight_stage(
    const void* activation, const void* weight_packed0,
    const void* weight_scale0, const void* weight_zero_point0, void* output0,
    int out_cols0, const void* weight_packed1, const void* weight_scale1,
    const void* weight_zero_point1, void* output1, int out_cols1, int in_cols,
    int groups, cudaStream_t stream) {
  if (activation == nullptr || weight_packed0 == nullptr ||
      weight_scale0 == nullptr || weight_zero_point0 == nullptr ||
      output0 == nullptr || weight_packed1 == nullptr || weight_scale1 == nullptr ||
      weight_zero_point1 == nullptr || output1 == nullptr || in_cols <= 0 ||
      in_cols % 128 != 0 || out_cols0 <= 0 || out_cols1 <= 0 ||
      out_cols0 % QWEN35_TC_OUT_TILE != 0 ||
      out_cols1 % QWEN35_TC_OUT_TILE != 0 || groups <= 0 ||
      in_cols % groups != 0 || in_cols / groups != 32) {
    return cudaErrorInvalidValue;
  }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  const int blocks0 = out_cols0 / QWEN35_TC_OUT_TILE;
  const int blocks1 = out_cols1 / QWEN35_TC_OUT_TILE;
  if (blocks0 > INT32_MAX - blocks1) return cudaErrorInvalidValue;
  qwen35_gemm_w4a16_bf16_tc_pair_weight_stage_kernel
      <<<blocks0 + blocks1, 256, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(activation),
          static_cast<const int32_t*>(weight_packed0),
          static_cast<const __nv_bfloat16*>(weight_scale0),
          static_cast<const int32_t*>(weight_zero_point0),
          static_cast<__nv_bfloat16*>(output0), out_cols0,
          static_cast<const int32_t*>(weight_packed1),
          static_cast<const __nv_bfloat16*>(weight_scale1),
          static_cast<const int32_t*>(weight_zero_point1),
          static_cast<__nv_bfloat16*>(output1), in_cols, groups);
#else
  (void)activation; (void)weight_packed0; (void)weight_scale0;
  (void)weight_zero_point0; (void)output0; (void)out_cols0;
  (void)weight_packed1; (void)weight_scale1; (void)weight_zero_point1;
  (void)output1; (void)out_cols1; (void)in_cols; (void)groups; (void)stream;
#endif
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_gemm_w4a16_bf16_tc_pair_alt(
    const void* activation, const void* weight_packed0,
    const void* weight_scale0, const void* weight_zero_point0, void* output0,
    int out_cols0, const void* weight_packed1, const void* weight_scale1,
    const void* weight_zero_point1, void* output1, int out_cols1, int in_cols,
    int groups, cudaStream_t stream) {
  if (activation == nullptr || weight_packed0 == nullptr ||
      weight_scale0 == nullptr || weight_zero_point0 == nullptr ||
      output0 == nullptr || weight_packed1 == nullptr ||
      weight_scale1 == nullptr || weight_zero_point1 == nullptr ||
      output1 == nullptr || in_cols <= 0 || in_cols % 128 != 0 ||
      out_cols0 <= 0 || out_cols0 % QWEN35_TC_ALT_OUT_TILE != 0 ||
      out_cols1 <= 0 || out_cols1 % QWEN35_TC_ALT_OUT_TILE != 0 ||
      groups <= 0 || in_cols % groups != 0 || in_cols / groups != 32) {
    return cudaErrorInvalidValue;
  }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  const int blocks = (out_cols0 + out_cols1) / QWEN35_TC_ALT_OUT_TILE;
  qwen35_gemm_w4a16_bf16_tc_pair_alt_kernel<<<blocks, 128, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(activation),
      static_cast<const int32_t*>(weight_packed0),
      static_cast<const __nv_bfloat16*>(weight_scale0),
      static_cast<const int32_t*>(weight_zero_point0),
      static_cast<__nv_bfloat16*>(output0), out_cols0,
      static_cast<const int32_t*>(weight_packed1),
      static_cast<const __nv_bfloat16*>(weight_scale1),
      static_cast<const int32_t*>(weight_zero_point1),
      static_cast<__nv_bfloat16*>(output1), in_cols, groups);
#else
  (void)activation; (void)weight_packed0; (void)weight_scale0;
  (void)weight_zero_point0; (void)output0; (void)out_cols0;
  (void)weight_packed1; (void)weight_scale1; (void)weight_zero_point1;
  (void)output1; (void)out_cols1; (void)in_cols; (void)groups; (void)stream;
#endif
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_gemm_w4a16_bf16_tc_pair_6w(
    const void* activation, const void* weight_packed0,
    const void* weight_scale0, const void* weight_zero_point0, void* output0,
    int out_cols0, const void* weight_packed1, const void* weight_scale1,
    const void* weight_zero_point1, void* output1, int out_cols1, int in_cols,
    int groups, cudaStream_t stream) {
  if (activation == nullptr || weight_packed0 == nullptr ||
      weight_scale0 == nullptr || weight_zero_point0 == nullptr ||
      output0 == nullptr || weight_packed1 == nullptr ||
      weight_scale1 == nullptr || weight_zero_point1 == nullptr ||
      output1 == nullptr || in_cols <= 0 || in_cols % 128 != 0 ||
      out_cols0 <= 0 || out_cols0 % 8 != 0 ||
      out_cols1 <= 0 || out_cols1 % 8 != 0 || groups <= 0 ||
      in_cols % groups != 0 || in_cols / groups != 32 ||
      out_cols0 > INT32_MAX - (QWEN35_TC_PAIR_6W_OUT_TILE - 1) ||
      out_cols1 > INT32_MAX - (QWEN35_TC_PAIR_6W_OUT_TILE - 1)) {
    return cudaErrorInvalidValue;
  }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  const int blocks0 =
      (out_cols0 + QWEN35_TC_PAIR_6W_OUT_TILE - 1) /
      QWEN35_TC_PAIR_6W_OUT_TILE;
  const int blocks1 =
      (out_cols1 + QWEN35_TC_PAIR_6W_OUT_TILE - 1) /
      QWEN35_TC_PAIR_6W_OUT_TILE;
  if (blocks0 > INT32_MAX - blocks1) return cudaErrorInvalidValue;
  qwen35_gemm_w4a16_bf16_tc_pair_6w_kernel<<<blocks0 + blocks1, 192, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(activation),
      static_cast<const int32_t*>(weight_packed0),
      static_cast<const __nv_bfloat16*>(weight_scale0),
      static_cast<const int32_t*>(weight_zero_point0),
      static_cast<__nv_bfloat16*>(output0), out_cols0,
      static_cast<const int32_t*>(weight_packed1),
      static_cast<const __nv_bfloat16*>(weight_scale1),
      static_cast<const int32_t*>(weight_zero_point1),
      static_cast<__nv_bfloat16*>(output1), out_cols1, in_cols, groups);
#else
  (void)activation; (void)weight_packed0; (void)weight_scale0;
  (void)weight_zero_point0; (void)output0; (void)out_cols0;
  (void)weight_packed1; (void)weight_scale1; (void)weight_zero_point1;
  (void)output1; (void)out_cols1; (void)in_cols; (void)groups; (void)stream;
#endif
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_gemm_w4a16_bf16_tc_pair_prefetch(
    const void* activation, const void* weight_packed0,
    const void* weight_scale0, const void* weight_zero_point0, void* output0,
    int out_cols0, const void* weight_packed1, const void* weight_scale1,
    const void* weight_zero_point1, void* output1, int out_cols1, int in_cols,
    int groups, cudaStream_t stream) {
  if (activation == nullptr || weight_packed0 == nullptr ||
      weight_scale0 == nullptr || weight_zero_point0 == nullptr ||
      output0 == nullptr || weight_packed1 == nullptr ||
      weight_scale1 == nullptr || weight_zero_point1 == nullptr ||
      output1 == nullptr ||
      (reinterpret_cast<uintptr_t>(activation) % alignof(uint4)) != 0 ||
      in_cols <= 0 || in_cols % 128 != 0 ||
      out_cols0 <= 0 || out_cols0 % QWEN35_TC_OUT_TILE != 0 ||
      out_cols1 <= 0 || out_cols1 % QWEN35_TC_OUT_TILE != 0 || groups <= 0 ||
      in_cols % groups != 0 || in_cols / groups != 32) {
    return cudaErrorInvalidValue;
  }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  const int blocks = (out_cols0 + out_cols1) / QWEN35_TC_OUT_TILE;
  qwen35_gemm_w4a16_bf16_tc_pair_prefetch_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(activation),
      static_cast<const int32_t*>(weight_packed0),
      static_cast<const __nv_bfloat16*>(weight_scale0),
      static_cast<const int32_t*>(weight_zero_point0),
      static_cast<__nv_bfloat16*>(output0), out_cols0,
      static_cast<const int32_t*>(weight_packed1),
      static_cast<const __nv_bfloat16*>(weight_scale1),
      static_cast<const int32_t*>(weight_zero_point1),
      static_cast<__nv_bfloat16*>(output1), in_cols, groups);
#else
  (void)activation; (void)weight_packed0; (void)weight_scale0;
  (void)weight_zero_point0; (void)output0; (void)out_cols0;
  (void)weight_packed1; (void)weight_scale1; (void)weight_zero_point1;
  (void)output1; (void)out_cols1; (void)in_cols; (void)groups; (void)stream;
#endif
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_gemm_w4a16_bf16_tc_pair_act(
    const void* activation, const void* weight_packed0,
    const void* weight_scale0, const void* weight_zero_point0, void* output0,
    int out_cols0, const void* weight_packed1, const void* weight_scale1,
    const void* weight_zero_point1, void* output1, int out_cols1, int in_cols,
    int groups, cudaStream_t stream) {
  if (activation == nullptr || weight_packed0 == nullptr ||
      weight_scale0 == nullptr || weight_zero_point0 == nullptr ||
      output0 == nullptr || weight_packed1 == nullptr ||
      weight_scale1 == nullptr || weight_zero_point1 == nullptr ||
      output1 == nullptr ||
      (reinterpret_cast<uintptr_t>(activation) % alignof(uint4)) != 0 ||
      in_cols <= 0 || in_cols % 128 != 0 ||
      out_cols0 <= 0 || out_cols0 % QWEN35_TC_OUT_TILE != 0 ||
      out_cols1 <= 0 || out_cols1 % QWEN35_TC_OUT_TILE != 0 ||
      groups <= 0 || in_cols % groups != 0 || in_cols / groups != 32) {
    return cudaErrorInvalidValue;
  }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  const int blocks = (out_cols0 + out_cols1) / QWEN35_TC_OUT_TILE;
  qwen35_gemm_w4a16_bf16_tc_pair_act_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(activation),
      static_cast<const int32_t*>(weight_packed0),
      static_cast<const __nv_bfloat16*>(weight_scale0),
      static_cast<const int32_t*>(weight_zero_point0),
      static_cast<__nv_bfloat16*>(output0), out_cols0,
      static_cast<const int32_t*>(weight_packed1),
      static_cast<const __nv_bfloat16*>(weight_scale1),
      static_cast<const int32_t*>(weight_zero_point1),
      static_cast<__nv_bfloat16*>(output1), in_cols, groups);
#else
  (void)activation; (void)weight_packed0; (void)weight_scale0;
  (void)weight_zero_point0; (void)output0; (void)out_cols0;
  (void)weight_packed1; (void)weight_scale1; (void)weight_zero_point1;
  (void)output1; (void)out_cols1; (void)in_cols; (void)groups; (void)stream;
#endif
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_gemm_w4a16_bf16_tc_pair_shared(
    const void* activation, const void* weight_packed0,
    const void* weight_scale0, const void* weight_zero_point0, void* output0,
    int out_cols0, const void* weight_packed1, const void* weight_scale1,
    const void* weight_zero_point1, void* output1, int out_cols1, int in_cols,
    int groups, cudaStream_t stream) {
  if (activation == nullptr || weight_packed0 == nullptr ||
      weight_scale0 == nullptr || weight_zero_point0 == nullptr ||
      output0 == nullptr || weight_packed1 == nullptr ||
      weight_scale1 == nullptr || weight_zero_point1 == nullptr ||
      output1 == nullptr || in_cols <= 0 || in_cols % 128 != 0 ||
      out_cols0 <= 0 || out_cols0 % QWEN35_TC_OUT_TILE != 0 ||
      out_cols1 <= 0 || out_cols1 % QWEN35_TC_OUT_TILE != 0 || groups <= 0 ||
      in_cols % groups != 0 || in_cols / groups != 32) {
    return cudaErrorInvalidValue;
  }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  const int blocks = (out_cols0 > out_cols1 ? out_cols0 : out_cols1) /
      QWEN35_TC_OUT_TILE;
  qwen35_gemm_w4a16_bf16_tc_pair_shared_kernel<<<blocks, 512, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(activation),
      static_cast<const int32_t*>(weight_packed0),
      static_cast<const __nv_bfloat16*>(weight_scale0),
      static_cast<const int32_t*>(weight_zero_point0),
      static_cast<__nv_bfloat16*>(output0), out_cols0,
      static_cast<const int32_t*>(weight_packed1),
      static_cast<const __nv_bfloat16*>(weight_scale1),
      static_cast<const int32_t*>(weight_zero_point1),
      static_cast<__nv_bfloat16*>(output1), out_cols1, in_cols, groups);
#else
  (void)activation; (void)weight_packed0; (void)weight_scale0;
  (void)weight_zero_point0; (void)output0; (void)out_cols0;
  (void)weight_packed1; (void)weight_scale1; (void)weight_zero_point1;
  (void)output1; (void)out_cols1; (void)in_cols; (void)groups; (void)stream;
#endif
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_gemm_w4a16_bf16_tc_pair_reuse(
    const void* activation, const void* weight_packed0,
    const void* weight_scale0, const void* weight_zero_point0, void* output0,
    int out_cols0, const void* weight_packed1, const void* weight_scale1,
    const void* weight_zero_point1, void* output1, int out_cols1, int in_cols,
    int groups, cudaStream_t stream) {
  if (activation == nullptr || weight_packed0 == nullptr ||
      weight_scale0 == nullptr || weight_zero_point0 == nullptr ||
      output0 == nullptr || weight_packed1 == nullptr ||
      weight_scale1 == nullptr || weight_zero_point1 == nullptr ||
      output1 == nullptr || in_cols <= 0 || in_cols % 128 != 0 ||
      out_cols0 <= 0 || out_cols0 % QWEN35_TC_OUT_TILE != 0 ||
      out_cols1 <= 0 || out_cols1 % QWEN35_TC_OUT_TILE != 0 || groups <= 0 ||
      in_cols % groups != 0 || in_cols / groups != 32) {
    return cudaErrorInvalidValue;
  }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  const int blocks = (out_cols0 > out_cols1 ? out_cols0 : out_cols1) /
      QWEN35_TC_OUT_TILE;
  qwen35_gemm_w4a16_bf16_tc_pair_reuse_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(activation),
      static_cast<const int32_t*>(weight_packed0),
      static_cast<const __nv_bfloat16*>(weight_scale0),
      static_cast<const int32_t*>(weight_zero_point0),
      static_cast<__nv_bfloat16*>(output0), out_cols0,
      static_cast<const int32_t*>(weight_packed1),
      static_cast<const __nv_bfloat16*>(weight_scale1),
      static_cast<const int32_t*>(weight_zero_point1),
      static_cast<__nv_bfloat16*>(output1), out_cols1, in_cols, groups);
#else
  (void)activation; (void)weight_packed0; (void)weight_scale0;
  (void)weight_zero_point0; (void)output0; (void)out_cols0;
  (void)weight_packed1; (void)weight_scale1; (void)weight_zero_point1;
  (void)output1; (void)out_cols1; (void)in_cols; (void)groups; (void)stream;
#endif
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_gemm_w4a16_bf16_tc_pair_cache(
    const void* activation, const void* weight_packed0,
    const void* weight_scale0, const void* weight_zero_point0, void* output0,
    int out_cols0, const void* weight_packed1, const void* weight_scale1,
    const void* weight_zero_point1, void* output1, int out_cols1, int in_cols,
    int groups, cudaStream_t stream) {
  if (activation == nullptr || weight_packed0 == nullptr ||
      weight_scale0 == nullptr || weight_zero_point0 == nullptr ||
      output0 == nullptr || weight_packed1 == nullptr ||
      weight_scale1 == nullptr || weight_zero_point1 == nullptr ||
      output1 == nullptr ||
      (reinterpret_cast<uintptr_t>(activation) % alignof(__nv_bfloat16)) != 0 ||
      (reinterpret_cast<uintptr_t>(weight_packed0) % alignof(int32_t)) != 0 ||
      (reinterpret_cast<uintptr_t>(weight_packed1) % alignof(int32_t)) != 0 ||
      (reinterpret_cast<uintptr_t>(weight_scale0) % alignof(__nv_bfloat16)) != 0 ||
      (reinterpret_cast<uintptr_t>(weight_scale1) % alignof(__nv_bfloat16)) != 0 ||
      (reinterpret_cast<uintptr_t>(weight_zero_point0) % alignof(int32_t)) != 0 ||
      (reinterpret_cast<uintptr_t>(weight_zero_point1) % alignof(int32_t)) != 0 ||
      in_cols <= 0 || in_cols % 128 != 0 || out_cols0 <= 0 ||
      out_cols1 <= 0 || out_cols0 % QWEN35_TC_OUT_TILE != 0 ||
      out_cols1 % QWEN35_TC_OUT_TILE != 0 || groups <= 0 ||
      in_cols % groups != 0 || in_cols / groups != 32) {
    return cudaErrorInvalidValue;
  }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  const int blocks = (out_cols0 + out_cols1) / QWEN35_TC_OUT_TILE;
  qwen35_gemm_w4a16_bf16_tc_pair_cache_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(activation),
      static_cast<const int32_t*>(weight_packed0),
      static_cast<const __nv_bfloat16*>(weight_scale0),
      static_cast<const int32_t*>(weight_zero_point0),
      static_cast<__nv_bfloat16*>(output0), out_cols0,
      static_cast<const int32_t*>(weight_packed1),
      static_cast<const __nv_bfloat16*>(weight_scale1),
      static_cast<const int32_t*>(weight_zero_point1),
      static_cast<__nv_bfloat16*>(output1), in_cols, groups);
#else
  (void)activation; (void)weight_packed0; (void)weight_scale0;
  (void)weight_zero_point0; (void)output0; (void)out_cols0;
  (void)weight_packed1; (void)weight_scale1; (void)weight_zero_point1;
  (void)output1; (void)out_cols1; (void)in_cols; (void)groups; (void)stream;
#endif
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_gemm_w4a16_bf16_tc_pair_2w(
    const void* activation, const void* weight_packed0,
    const void* weight_scale0, const void* weight_zero_point0, void* output0,
    int out_cols0, const void* weight_packed1, const void* weight_scale1,
    const void* weight_zero_point1, void* output1, int out_cols1, int in_cols,
    int groups, cudaStream_t stream) {
  if (activation == nullptr || weight_packed0 == nullptr ||
      weight_scale0 == nullptr || weight_zero_point0 == nullptr ||
      output0 == nullptr || weight_packed1 == nullptr ||
      weight_scale1 == nullptr || weight_zero_point1 == nullptr ||
      output1 == nullptr || in_cols <= 0 || in_cols % 128 != 0 ||
      out_cols0 <= 0 || out_cols0 % QWEN35_TC_PAIR_2W_OUT_TILE != 0 ||
      out_cols1 <= 0 || out_cols1 % QWEN35_TC_PAIR_2W_OUT_TILE != 0 ||
      groups <= 0 || in_cols % groups != 0 || in_cols / groups != 32) {
    return cudaErrorInvalidValue;
  }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  const int blocks =
      (out_cols0 + out_cols1) / QWEN35_TC_PAIR_2W_OUT_TILE;
  qwen35_gemm_w4a16_bf16_tc_pair_2w_kernel<<<blocks, 64, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(activation),
      static_cast<const int32_t*>(weight_packed0),
      static_cast<const __nv_bfloat16*>(weight_scale0),
      static_cast<const int32_t*>(weight_zero_point0),
      static_cast<__nv_bfloat16*>(output0), out_cols0,
      static_cast<const int32_t*>(weight_packed1),
      static_cast<const __nv_bfloat16*>(weight_scale1),
      static_cast<const int32_t*>(weight_zero_point1),
      static_cast<__nv_bfloat16*>(output1), in_cols, groups);
#else
  (void)activation; (void)weight_packed0; (void)weight_scale0;
  (void)weight_zero_point0; (void)output0; (void)out_cols0;
  (void)weight_packed1; (void)weight_scale1; (void)weight_zero_point1;
  (void)output1; (void)out_cols1; (void)in_cols; (void)groups; (void)stream;
#endif
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_gemm_w4a16_bf16_tc_multi(
    const void* activation,
    const void* weight_packed0, const void* weight_scale0,
    const void* weight_zero_point0, void* output0, int out_cols0,
    const void* weight_packed1, const void* weight_scale1,
    const void* weight_zero_point1, void* output1, int out_cols1,
    const void* weight_packed2, const void* weight_scale2,
    const void* weight_zero_point2, void* output2, int out_cols2,
    const void* weight_packed3, const void* weight_scale3,
    const void* weight_zero_point3, void* output3, int out_cols3,
    int projection_count, int in_cols, int groups, cudaStream_t stream) {
  if (activation == nullptr || weight_packed0 == nullptr ||
      weight_scale0 == nullptr || weight_zero_point0 == nullptr || output0 == nullptr ||
      weight_packed1 == nullptr || weight_scale1 == nullptr ||
      weight_zero_point1 == nullptr || output1 == nullptr ||
      weight_packed2 == nullptr || weight_scale2 == nullptr ||
      weight_zero_point2 == nullptr || output2 == nullptr ||
      (projection_count == 4 && (weight_packed3 == nullptr || weight_scale3 == nullptr ||
       weight_zero_point3 == nullptr || output3 == nullptr)) ||
      (projection_count != 3 && projection_count != 4) || in_cols <= 0 ||
      in_cols % 128 != 0 || groups <= 0 || in_cols % groups != 0 ||
      in_cols / groups != 32 ||
      out_cols0 <= 0 || out_cols0 % 8 != 0 || out_cols1 <= 0 ||
      out_cols1 % 8 != 0 || out_cols2 <= 0 || out_cols2 % 8 != 0 ||
      (projection_count == 4 && (out_cols3 <= 0 || out_cols3 % 8 != 0))) {
    return cudaErrorInvalidValue;
  }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  const int blocks = (out_cols0 + QWEN35_TC_OUT_TILE - 1) / QWEN35_TC_OUT_TILE +
      (out_cols1 + QWEN35_TC_OUT_TILE - 1) / QWEN35_TC_OUT_TILE +
      (out_cols2 + QWEN35_TC_OUT_TILE - 1) / QWEN35_TC_OUT_TILE +
      (projection_count == 4
          ? (out_cols3 + QWEN35_TC_OUT_TILE - 1) / QWEN35_TC_OUT_TILE : 0);
  qwen35_gemm_w4a16_bf16_tc_multi_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(activation),
      static_cast<const int32_t*>(weight_packed0),
      static_cast<const __nv_bfloat16*>(weight_scale0),
      static_cast<const int32_t*>(weight_zero_point0),
      static_cast<__nv_bfloat16*>(output0), out_cols0,
      static_cast<const int32_t*>(weight_packed1),
      static_cast<const __nv_bfloat16*>(weight_scale1),
      static_cast<const int32_t*>(weight_zero_point1),
      static_cast<__nv_bfloat16*>(output1), out_cols1,
      static_cast<const int32_t*>(weight_packed2),
      static_cast<const __nv_bfloat16*>(weight_scale2),
      static_cast<const int32_t*>(weight_zero_point2),
      static_cast<__nv_bfloat16*>(output2), out_cols2,
      static_cast<const int32_t*>(weight_packed3),
      static_cast<const __nv_bfloat16*>(weight_scale3),
      static_cast<const int32_t*>(weight_zero_point3),
      static_cast<__nv_bfloat16*>(output3), out_cols3,
      projection_count, in_cols, groups);
#else
  (void)activation; (void)weight_packed0; (void)weight_scale0;
  (void)weight_zero_point0; (void)output0; (void)out_cols0;
  (void)weight_packed1; (void)weight_scale1; (void)weight_zero_point1;
  (void)output1; (void)out_cols1; (void)weight_packed2; (void)weight_scale2;
  (void)weight_zero_point2; (void)output2; (void)out_cols2;
  (void)weight_packed3; (void)weight_scale3; (void)weight_zero_point3;
  (void)output3; (void)out_cols3; (void)projection_count;
  (void)in_cols; (void)groups; (void)stream;
#endif
  return cudaGetLastError();
}

extern "C" cudaError_t
apxinf_qwen35_gemm_w4a16_bf16_prefill_fast_raw(
    const void* activation, const void* weight_packed,
    const void* weight_scale, const void* weight_zero_point, void* output,
    int rows, int in_cols, int out_cols, int groups, cudaStream_t stream) {
  if (activation == nullptr || weight_packed == nullptr ||
      weight_scale == nullptr || weight_zero_point == nullptr ||
      output == nullptr || rows <= 0 || rows > 256 || in_cols <= 0 ||
      in_cols % QWEN35_PREFILL_FAST_K_TILE != 0 || out_cols <= 0 ||
      groups <= 0 || in_cols % groups != 0 ||
      in_cols / groups != QWEN35_PREFILL_FAST_K_TILE) {
    return cudaErrorInvalidValue;
  }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  dim3 grid((out_cols + QWEN35_PREFILL_FAST_N_TILE - 1) /
                QWEN35_PREFILL_FAST_N_TILE,
            (rows + QWEN35_PREFILL_FAST_M_TILE - 1) /
                QWEN35_PREFILL_FAST_M_TILE);
  qwen35_gemm_w4a16_bf16_prefill_fast_kernel<false>
      <<<grid, 256, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(activation),
          static_cast<const int32_t*>(weight_packed),
          static_cast<const __nv_bfloat16*>(weight_scale),
          static_cast<const int32_t*>(weight_zero_point),
          static_cast<__nv_bfloat16*>(output), rows, in_cols, out_cols, groups);
  return cudaGetLastError();
#else
  (void)stream;
  return cudaErrorNotSupported;
#endif
}

extern "C" cudaError_t
apxinf_qwen35_gemm_w4a16_bf16_prefill_native(
    const void* activation, const void* weight_packed,
    const void* weight_scale, const void* weight_zero_point, void* output,
    int rows, int in_cols, int out_cols, int groups, cudaStream_t stream) {
  if (activation == nullptr || weight_packed == nullptr ||
      weight_scale == nullptr || weight_zero_point == nullptr ||
      output == nullptr || rows <= 1 ||
      rows > QWEN35_PREFILL_NATIVE_MAX_ROWS || in_cols <= 0 ||
      in_cols % QWEN35_PREFILL_PACKED_K_TILE != 0 || out_cols <= 0 ||
      groups <= 0 || in_cols % groups != 0 ||
      in_cols / groups != QWEN35_PREFILL_PACKED_K_TILE) {
    return cudaErrorInvalidValue;
  }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  dim3 grid((out_cols + QWEN35_PREFILL_PACKED_N_TILE - 1) /
                QWEN35_PREFILL_PACKED_N_TILE,
            (rows + QWEN35_PREFILL_PACKED_M_TILE - 1) /
                QWEN35_PREFILL_PACKED_M_TILE);
  qwen35_gemm_w4a16_bf16_prefill_packed_kernel<false>
      <<<grid, 256, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(activation),
          static_cast<const int32_t*>(weight_packed),
          static_cast<const __nv_bfloat16*>(weight_scale),
          static_cast<const int32_t*>(weight_zero_point),
          static_cast<__nv_bfloat16*>(output), rows, in_cols, out_cols, groups);
  return cudaGetLastError();
#else
  (void)stream;
  return cudaErrorNotSupported;
#endif
}

extern "C" cudaError_t
apxinf_qwen35_gemm_w4a16_bf16_prefill_fast_repacked_v1(
    const void* activation, const void* weight_qwords,
    const void* weight_scale, const void* weight_zero_point, void* output,
    int rows, int in_cols, int out_cols, int padded_out_cols, int groups,
    cudaStream_t stream) {
  if (activation == nullptr || weight_qwords == nullptr ||
      weight_scale == nullptr || weight_zero_point == nullptr ||
      output == nullptr || rows <= 0 || rows > 256 || in_cols <= 0 ||
      in_cols % 128 != 0 || out_cols <= 0 || padded_out_cols < out_cols ||
      padded_out_cols % 64 != 0 || groups <= 0 || in_cols % groups != 0 ||
      in_cols / groups != QWEN35_PREFILL_FAST_K_TILE) {
    return cudaErrorInvalidValue;
  }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  dim3 grid((out_cols + QWEN35_PREFILL_FAST_N_TILE - 1) /
                QWEN35_PREFILL_FAST_N_TILE,
            (rows + QWEN35_PREFILL_FAST_M_TILE - 1) /
                QWEN35_PREFILL_FAST_M_TILE);
  qwen35_gemm_w4a16_bf16_prefill_fast_kernel<true>
      <<<grid, 256, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(activation),
          static_cast<const int32_t*>(weight_qwords),
          static_cast<const __nv_bfloat16*>(weight_scale),
          static_cast<const int32_t*>(weight_zero_point),
          static_cast<__nv_bfloat16*>(output), rows, in_cols, out_cols, groups);
  return cudaGetLastError();
#else
  (void)stream;
  return cudaErrorNotSupported;
#endif
}

extern "C" cudaError_t
apxinf_qwen35_gemm_w4a16_bf16_prefill_packed_raw(
    const void* activation, const void* weight_packed,
    const void* weight_scale, const void* weight_zero_point, void* output,
    int rows, int in_cols, int out_cols, int groups, cudaStream_t stream) {
  if (activation == nullptr || weight_packed == nullptr ||
      weight_scale == nullptr || weight_zero_point == nullptr ||
      output == nullptr || rows <= 1 || rows > 256 || in_cols <= 0 ||
      in_cols % QWEN35_PREFILL_PACKED_K_TILE != 0 || out_cols <= 0 ||
      groups <= 0 || in_cols % groups != 0 ||
      in_cols / groups != QWEN35_PREFILL_PACKED_K_TILE) {
    return cudaErrorInvalidValue;
  }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  dim3 grid((out_cols + QWEN35_PREFILL_PACKED_N_TILE - 1) /
                QWEN35_PREFILL_PACKED_N_TILE,
            (rows + QWEN35_PREFILL_PACKED_M_TILE - 1) /
                QWEN35_PREFILL_PACKED_M_TILE);
  qwen35_gemm_w4a16_bf16_prefill_packed_kernel<false>
      <<<grid, 256, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(activation),
          static_cast<const int32_t*>(weight_packed),
          static_cast<const __nv_bfloat16*>(weight_scale),
          static_cast<const int32_t*>(weight_zero_point),
          static_cast<__nv_bfloat16*>(output), rows, in_cols, out_cols, groups);
  return cudaGetLastError();
#else
  (void)stream;
  return cudaErrorNotSupported;
#endif
}

extern "C" cudaError_t
apxinf_qwen35_gemm_w4a16_bf16_prefill_packed_repacked_v1(
    const void* activation, const void* weight_qwords,
    const void* weight_scale, const void* weight_zero_point, void* output,
    int rows, int in_cols, int out_cols, int padded_out_cols, int groups,
    cudaStream_t stream) {
  if (activation == nullptr || weight_qwords == nullptr ||
      weight_scale == nullptr || weight_zero_point == nullptr ||
      output == nullptr || rows <= 1 || rows > 256 || in_cols <= 0 ||
      in_cols % 128 != 0 || out_cols <= 0 || padded_out_cols < out_cols ||
      padded_out_cols % 64 != 0 || groups <= 0 || in_cols % groups != 0 ||
      in_cols / groups != QWEN35_PREFILL_PACKED_K_TILE) {
    return cudaErrorInvalidValue;
  }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  dim3 grid((out_cols + QWEN35_PREFILL_PACKED_N_TILE - 1) /
                QWEN35_PREFILL_PACKED_N_TILE,
            (rows + QWEN35_PREFILL_PACKED_M_TILE - 1) /
                QWEN35_PREFILL_PACKED_M_TILE);
  qwen35_gemm_w4a16_bf16_prefill_packed_kernel<true>
      <<<grid, 256, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(activation),
          static_cast<const int32_t*>(weight_qwords),
          static_cast<const __nv_bfloat16*>(weight_scale),
          static_cast<const int32_t*>(weight_zero_point),
          static_cast<__nv_bfloat16*>(output), rows, in_cols, out_cols, groups);
  return cudaGetLastError();
#else
  (void)stream;
  return cudaErrorNotSupported;
#endif
}

extern "C" cudaError_t
apxinf_qwen35_gemm_w4a16_bf16_prefill_repacked_v1(
    const void* activation, const void* weight_qwords,
    const void* weight_scale, const void* weight_zero_point, void* output,
    int rows, int in_cols, int out_cols, int padded_out_cols, int groups,
    cudaStream_t stream) {
  if (activation == nullptr || weight_qwords == nullptr ||
      weight_scale == nullptr || weight_zero_point == nullptr ||
      output == nullptr || rows != QWEN35_PREFILL_W4_M || in_cols <= 0 ||
      in_cols % 128 != 0 || out_cols <= 0 || padded_out_cols < out_cols ||
      padded_out_cols % 64 != 0 || groups <= 0 || in_cols / groups != 32) {
    return cudaErrorInvalidValue;
  }
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
  dim3 grid(padded_out_cols / QWEN35_PREFILL_W4_N_TILE,
            rows / QWEN35_PREFILL_W4_K_TILE);
  qwen35_gemm_w4a16_bf16_prefill_repacked_v1_kernel<<<grid, 128, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(activation),
      static_cast<const int32_t*>(weight_qwords),
      static_cast<const __nv_bfloat16*>(weight_scale),
      static_cast<const int32_t*>(weight_zero_point),
      static_cast<__nv_bfloat16*>(output), out_cols, in_cols, groups);
  return cudaGetLastError();
#else
  (void)stream;
  return cudaErrorNotSupported;
#endif
}

extern "C" cudaError_t
apxinf_qwen35_dequant_w4a16_bf16_rows_repacked_v1(
    const void* weight_qwords, const void* weight_scale,
    const void* weight_zero_point, void* dense, int in_cols, int out_cols,
    int groups, int row_start, int row_count, int padded_out_cols,
    cudaStream_t stream) {
  if (weight_qwords == nullptr || weight_scale == nullptr ||
      weight_zero_point == nullptr || dense == nullptr || in_cols <= 0 ||
      in_cols % 128 != 0 || out_cols <= 0 || padded_out_cols < out_cols ||
      padded_out_cols % 64 != 0 || groups <= 0 || in_cols / groups != 32 ||
      row_start < 0 || row_count <= 0 || row_start > out_cols - row_count) {
    return cudaErrorInvalidValue;
  }
  qwen35_dequant_w4a16_bf16_rows_repacked_v1_kernel
      <<<row_count, 256, 0, stream>>>(
          static_cast<const int32_t*>(weight_qwords),
          static_cast<const __nv_bfloat16*>(weight_scale),
          static_cast<const int32_t*>(weight_zero_point),
          static_cast<__nv_bfloat16*>(dense), in_cols, out_cols, groups,
          row_start, row_count);
  return cudaGetLastError();
}
