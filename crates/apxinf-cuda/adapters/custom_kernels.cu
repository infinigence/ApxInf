// Copyright 2026 apxinf contributors.
// Stable C ABI and CUDA launch adapter for custom static-inference operators.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// Reuse the exact U4 conversion without importing third-party headers into
// the adapter's anonymous namespace.
#define MARLIN_NAMESPACE_NAME apxinf_decode_pair
#include "../kernels/marlin/csrc/moe/marlin_moe_wna16/kernel.h"
#include "../kernels/marlin/csrc/quantization/gptq_marlin/dequant.h"

namespace {
#include "../kernels/custom/math.cuh"
#include "../kernels/custom/reduction.cuh"
#include "../kernels/custom/quantization.cuh"
#include "../kernels/custom/w4a16_blocked.cuh"
#include "../kernels/custom/w4a16_magic.cuh"
#include "../kernels/custom/w4a16_pair.cuh"
#include "../kernels/custom/grouped_gemm.cuh"
#include "../kernels/custom/qk_norm_rope.cuh"
#include "../kernels/custom/decode_epilogues.cuh"
#include "../kernels/custom/moe_permutation.cuh"
#include "../kernels/custom/preprocess.cuh"
#include "../kernels/custom/attention.cuh"
#include "../kernels/custom/gqa_decode.cuh"
#include "../kernels/custom/gqa_decode_mma.cuh"
#include "../kernels/custom/normalization.cuh"
#include "../kernels/custom/activation.cuh"
#include "../kernels/custom/embedding.cuh"
#include "../kernels/custom/elementwise.cuh"
#include "../kernels/custom/fused.cuh"
#include "../kernels/custom/cache.cuh"
#include "../kernels/custom/selection.cuh"
}  // namespace

namespace {

// Resolve the validated action Ada packed8 route before CUDA graph capture.
// Auto enables it only for the exact supported shape.
const int kActionAdaPacked8Mode = [] {
    const char* value = std::getenv("APXINF_PI05_ACTION_ADA_PACKED8");
    if (value == nullptr || std::strcmp(value, "auto") == 0) {
      return 2;
    }
    if (std::strcmp(value, "0") == 0 || std::strcmp(value, "off") == 0) {
      return 0;
    }
    if (std::strcmp(value, "1") == 0 || std::strcmp(value, "on") == 0) {
      return 1;
    }
    return -1;
  }();

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
  const int packed8_mode = kActionAdaPacked8Mode;
  if (packed8_mode < 0) return cudaErrorInvalidValue;
  const bool packed8_exact_shape = rows == 10 && cols == 1024;
  if (packed8_mode == 1 && !packed8_exact_shape) return cudaErrorInvalidValue;
  if (packed8_mode != 0 && packed8_exact_shape) {
    if (!std::isfinite(scale) ||
        projection == nullptr || residual == nullptr || gate_style == nullptr ||
        norm_style == nullptr || hidden == nullptr || normalized == nullptr) {
      return cudaErrorInvalidValue;
    }
    ada_gate_residual_rms_norm_quant_f16_e4m3_packed8_kernel
        <<<rows, 256, 0, stream>>>(
            static_cast<const half*>(projection),
            static_cast<const half*>(residual),
            static_cast<const half*>(gate_style),
            static_cast<const half*>(norm_style), static_cast<half*>(hidden),
            static_cast<__nv_fp8_e4m3*>(normalized), eps, 1.0f / scale);
    return cudaGetLastError();
  }
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

// ── AutoAWQ INT4 weights and MoE routing ─────────────────────────────────

namespace {
int grid_stride_blocks(int64_t count, int threads, int cap) {
  int64_t blocks = (count + threads - 1) / threads;
  if (blocks > cap) blocks = cap;
  return blocks < 1 ? 1 : static_cast<int>(blocks);
}
}  // namespace

extern "C" cudaError_t apxinf_rope_table_128(void* table,int positions,float theta,cudaStream_t stream) {
  if(positions<=0 || positions>INT32_MAX/64 || !std::isfinite(theta) || theta<=0)return cudaErrorInvalidValue;
  rope_table_f32_kernel<<<256,256,0,stream>>>(static_cast<float*>(table),positions,theta);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qk_norm_rope_append_f16(const void* q,const void* k,
    const void* v,const void* qw,const void* kw,void* oq,void* ok,void* ov,
    void* cache_k,void* cache_v,int tokens,int qheads,int kvheads,int capacity,
    int offset,float eps,const void* rope,cudaStream_t stream) {
  if(tokens<=0 || qheads<=0 || kvheads<=0 || offset<0 || tokens>capacity-offset
      || int64_t(tokens)*(int64_t(qheads)+kvheads)>INT32_MAX)return cudaErrorInvalidValue;
  qk_norm_rope_append_f16_kernel<<<(tokens*(qheads+kvheads)+3)/4,128,0,stream>>>(
      static_cast<const __nv_bfloat16*>(q),static_cast<const __nv_bfloat16*>(k),
      static_cast<const __nv_bfloat16*>(v),static_cast<const __nv_bfloat16*>(qw),
      static_cast<const __nv_bfloat16*>(kw),static_cast<half*>(oq),static_cast<half*>(ok),
      static_cast<half*>(ov),static_cast<__nv_bfloat16*>(cache_k),static_cast<__nv_bfloat16*>(cache_v),
      tokens,qheads,kvheads,capacity,offset,eps,static_cast<const float*>(rope));
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qk_norm_rope_append_cached_f16(const void* q,const void* k,
    const void* v,const void* qw,const void* kw,void* oq,void* ok,void* ov,
    void* cache_k,void* cache_v,int tokens,int qheads,int kvheads,int capacity,
    const void* used_k,float eps,const void* rope,cudaStream_t stream) {
  if(tokens<=0 || qheads<=0 || kvheads<=0 || used_k==nullptr || tokens>capacity
      || int64_t(tokens)*(int64_t(qheads)+kvheads)>INT32_MAX)return cudaErrorInvalidValue;
  qk_norm_rope_append_f16_kernel<true><<<(tokens*(qheads+kvheads)+3)/4,128,0,stream>>>(
      static_cast<const __nv_bfloat16*>(q),static_cast<const __nv_bfloat16*>(k),
      static_cast<const __nv_bfloat16*>(v),static_cast<const __nv_bfloat16*>(qw),
      static_cast<const __nv_bfloat16*>(kw),static_cast<half*>(oq),static_cast<half*>(ok),
      static_cast<half*>(ov),static_cast<__nv_bfloat16*>(cache_k),static_cast<__nv_bfloat16*>(cache_v),
      tokens,qheads,kvheads,capacity,0,eps,static_cast<const float*>(rope),static_cast<const int32_t*>(used_k));
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_moe_permute_marlin(const void* ids,void* counts,void* offsets,
    void* sorted,void* expert_ids,void* padded,int slots,int experts,int tile,cudaStream_t stream) {
  if((tile!=32 && tile!=64) || slots<=0 || slots>INT32_MAX-128*64 || experts<=0 || experts>128)return cudaErrorInvalidValue;
  moe_count_slots_kernel<<<experts,256,0,stream>>>(static_cast<const int32_t*>(ids),static_cast<int*>(counts),slots);
  moe_scan_tiles_kernel<<<1,128,0,stream>>>(static_cast<const int*>(counts),static_cast<int*>(offsets),
      static_cast<int32_t*>(expert_ids),static_cast<int*>(padded),experts,tile);
  moe_scatter_slots_kernel<<<experts,128,0,stream>>>(static_cast<const int32_t*>(ids),
      static_cast<const int*>(counts),static_cast<const int*>(offsets),static_cast<int32_t*>(sorted),slots);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_w4a16_grouped_bf16(
    const void* input, const void* qweight, const void* qzeros, const void* scales,
    const void* tiles, void* output, int tile_count, int k, int n, int group,
    int64_t sq, int64_t sz, int64_t ss, cudaStream_t stream) {
  if (tile_count <= 0 || k <= 0 || n <= 0 || n % 8 || group != 128 || k % 128)
    return cudaErrorInvalidValue;
  w4a16_grouped_bf16_kernel<<<dim3(tile_count, (n + 127) / 128), 128, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input), static_cast<const int32_t*>(qweight),
      static_cast<const int32_t*>(qzeros), static_cast<const half*>(scales),
      static_cast<const int32_t*>(tiles), static_cast<__nv_bfloat16*>(output),
      k, n, group, sq, sz, ss);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_awq_dequant_bf16(
    const void* qweight, const void* qzeros, const void* scales, void* output,
    int rows, int packed_cols, int group_size, int experts, int64_t stride_q,
    int64_t stride_z, int64_t stride_s, int64_t stride_out, cudaStream_t stream) {
  if (rows <= 0 || packed_cols <= 0 || group_size <= 0 || experts <= 0 ||
      rows % group_size != 0)
    return cudaErrorInvalidValue;
  const int64_t words = static_cast<int64_t>(rows) * packed_cols;
  const dim3 grid(grid_stride_blocks(words, 256, 4096), experts);
  awq_dequant_bf16_kernel<<<grid, 256, 0, stream>>>(
      static_cast<const int32_t*>(qweight), static_cast<const int32_t*>(qzeros),
      static_cast<const half*>(scales), static_cast<__nv_bfloat16*>(output), rows,
      packed_cols, group_size, stride_q, stride_z, stride_s, stride_out);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_w4a16_gemv_partial_bf16(
    const void* x, int64_t x_slot_stride, const void* qweight, const void* qzeros,
    const void* scales, const void* expert_ids, int64_t stride_q, int64_t stride_z,
    int64_t stride_s, const void* slot_scale, void* partial, int rows,
    int packed_cols, int group_size, int splits, int slots, cudaStream_t stream) {
  if (rows <= 0 || packed_cols <= 0 || group_size <= 0 || splits <= 0 ||
      slots <= 0 || rows % group_size != 0)
    return cudaErrorInvalidValue;
  // Rows per block: ceil(rows/splits) rounded up to a multiple of 8 warps.
  int rows_per_block = (rows + splits - 1) / splits;
  rows_per_block = (rows_per_block + 7) / 8 * 8;
  const int n_tiles = (packed_cols * 8 + 255) / 256;
  const dim3 grid(n_tiles, splits, slots);
  const size_t shared = static_cast<size_t>(rows_per_block) * sizeof(float) +
                        8 * 256 * sizeof(float);
  w4a16_gemv_partial_kernel<<<grid, 256, shared, stream>>>(
      static_cast<const __nv_bfloat16*>(x), x_slot_stride,
      static_cast<const int32_t*>(qweight), static_cast<const int32_t*>(qzeros),
      static_cast<const half*>(scales), static_cast<const int32_t*>(expert_ids),
      stride_q, stride_z, stride_s, static_cast<const float*>(slot_scale),
      static_cast<float*>(partial), rows, packed_cols, group_size, rows_per_block);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_w4a16_gemv_blocked_partial_bf16(
    const void* x, int64_t x_slot_stride, const void* qweight, const void* qzeros,
    const void* scales, const void* expert_ids, int64_t stride_q, int64_t stride_z,
    int64_t stride_s, const void* slot_scale, void* partial, int rows,
    int packed_cols, int group_size, int splits, int slots, cudaStream_t stream) {
  if (rows <= 0 || packed_cols <= 0 || group_size <= 0 || splits <= 0 ||
      slots <= 0 || rows % group_size != 0 || rows % 4 || packed_cols % 32)
    return cudaErrorInvalidValue;
  // Rows per block: ceil(rows/splits) rounded up to a multiple of 8 warps.
  int rows_per_block = (rows + splits - 1) / splits;
  rows_per_block = (rows_per_block + 7) / 8 * 8;
  const int n_tiles = (packed_cols * 8 + 255) / 256;
  const dim3 grid(n_tiles, splits, slots);
  const size_t shared = static_cast<size_t>(rows_per_block) * sizeof(float) +
                        8 * 256 * sizeof(float);
  w4a16_gemv_blocked_partial_kernel<<<grid, 256, shared, stream>>>(
      static_cast<const __nv_bfloat16*>(x), x_slot_stride,
      static_cast<const int32_t*>(qweight), static_cast<const int32_t*>(qzeros),
      static_cast<const half*>(scales), static_cast<const int32_t*>(expert_ids),
      stride_q, stride_z, stride_s, static_cast<const float*>(slot_scale),
      static_cast<float*>(partial), rows, packed_cols, group_size, rows_per_block);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_w4a16_gemv_magic_partial_bf16(
    const void* x, int64_t x_slot_stride, const void* qweight, const void* qzeros,
    const void* scales, const void* expert_ids, int64_t stride_q, int64_t stride_z,
    int64_t stride_s, const void* slot_scale, void* partial, int rows,
    int packed_cols, int group_size, int splits, int slots, cudaStream_t stream) {
  if (rows <= 0 || packed_cols <= 0 || group_size <= 0 || splits <= 0 ||
      slots <= 0 || rows % group_size != 0)
    return cudaErrorInvalidValue;
  // Rows per block: ceil(rows/splits) rounded up to a multiple of 8 warps.
  int rows_per_block = (rows + splits - 1) / splits;
  rows_per_block = (rows_per_block + 7) / 8 * 8;
  const int n_tiles = (packed_cols * 8 + 255) / 256;
  const dim3 grid(n_tiles, splits, slots);
  const size_t shared = static_cast<size_t>(rows_per_block) * sizeof(float) +
                        8 * 256 * sizeof(float);
  w4a16_gemv_magic_kernel<false><<<grid, 256, shared, stream>>>(
      static_cast<const __nv_bfloat16*>(x), x_slot_stride,
      static_cast<const int32_t*>(qweight), static_cast<const int32_t*>(qzeros),
      static_cast<const half*>(scales), static_cast<const int32_t*>(expert_ids),
      stride_q, stride_z, stride_s, static_cast<const float*>(slot_scale),
      static_cast<float*>(partial), rows, packed_cols, group_size, rows_per_block);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_w4a16_gemv_magic_blocked_partial_bf16(
    const void* x, int64_t x_slot_stride, const void* qweight, const void* qzeros,
    const void* scales, const void* expert_ids, int64_t stride_q, int64_t stride_z,
    int64_t stride_s, const void* slot_scale, void* partial, int rows,
    int packed_cols, int group_size, int splits, int slots, cudaStream_t stream) {
  if (rows <= 0 || packed_cols <= 0 || group_size <= 0 || splits <= 0 ||
      slots <= 0 || rows % group_size != 0 || rows % 4 || packed_cols % 32)
    return cudaErrorInvalidValue;
  // Rows per block: ceil(rows/splits) rounded up to a multiple of 8 warps.
  int rows_per_block = (rows + splits - 1) / splits;
  rows_per_block = (rows_per_block + 7) / 8 * 8;
  const int n_tiles = (packed_cols * 8 + 255) / 256;
  const dim3 grid(n_tiles, splits, slots);
  const size_t shared = static_cast<size_t>(rows_per_block) * sizeof(float) +
                        8 * 256 * sizeof(float);
  w4a16_gemv_magic_kernel<true><<<grid, 256, shared, stream>>>(
      static_cast<const __nv_bfloat16*>(x), x_slot_stride,
      static_cast<const int32_t*>(qweight), static_cast<const int32_t*>(qzeros),
      static_cast<const half*>(scales), static_cast<const int32_t*>(expert_ids),
      stride_q, stride_z, stride_s, static_cast<const float*>(slot_scale),
      static_cast<float*>(partial), rows, packed_cols, group_size, rows_per_block);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_w4a16_gemv_pair_blocked_partial_bf16(
    const void* x, int64_t x_slot_stride, const void* qweight, const void* qzeros,
    const void* scales, const void* expert_ids, int64_t stride_q, int64_t stride_z,
    int64_t stride_s, const void* slot_scale, void* partial, int rows,
    int packed_cols, int group_size, int splits, int slots, cudaStream_t stream) {
  if (rows <= 0 || packed_cols <= 0 || group_size <= 0 || splits <= 0 ||
      slots <= 0 || rows % group_size != 0 || rows % 4 || packed_cols % 32)
    return cudaErrorInvalidValue;
  // Rows per block: ceil(rows/splits) rounded up to a multiple of 8 warps.
  int rows_per_block = (rows + splits - 1) / splits;
  rows_per_block = (rows_per_block + 7) / 8 * 8;
  const int n_tiles = (packed_cols * 8 + 255) / 256;
  const dim3 grid(n_tiles, splits, slots);
  const size_t shared = static_cast<size_t>(rows_per_block) * sizeof(float) +
                        8 * 256 * sizeof(float);
  // Group 128 is the checkpoint layout. Keep a generic specialization for
  // other valid groups; both preserve the original FP32 reduction order.
  const auto kernel = group_size == 128
      ? w4a16_gemv_pair_kernel<true, false, 128, true>
      : w4a16_gemv_pair_kernel<true, false, 0, true>;
  kernel<<<grid, 256, shared, stream>>>(
      static_cast<const __nv_bfloat16*>(x), x_slot_stride,
      static_cast<const int32_t*>(qweight), static_cast<const int32_t*>(qzeros),
      static_cast<const half*>(scales), static_cast<const int32_t*>(expert_ids),
      stride_q, stride_z, stride_s, static_cast<const float*>(slot_scale),
      static_cast<float*>(partial), rows, packed_cols, group_size, rows_per_block);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_w4a16_blocked_repack(const void* source,void* output,
    int rows,int packed_cols,int experts,cudaStream_t stream) {
  if(rows<=0 || rows%128 || packed_cols<=0 || packed_cols%32 || experts<=0)
    return cudaErrorInvalidValue;
  w4a16_blocked_repack_kernel<<<256,256,0,stream>>>(static_cast<const int32_t*>(source),
      static_cast<int4*>(output),rows,packed_cols,experts);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_partial_sum_bf16(
    const void* partial, void* output, int cols, int count, cudaStream_t stream) {
  if (cols <= 0 || count <= 0) return cudaErrorInvalidValue;
  partial_sum_bf16_kernel<<<(cols + 255) / 256, 256, 0, stream>>>(
      static_cast<const float*>(partial), static_cast<__nv_bfloat16*>(output), cols,
      count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_partial_silu_mul_bf16(
    const void* partial, void* output, int inter, int splits, int slots,
    cudaStream_t stream) {
  if (inter <= 0 || splits <= 0 || slots <= 0) return cudaErrorInvalidValue;
  const int64_t count = static_cast<int64_t>(inter) * slots;
  partial_silu_mul_bf16_kernel<<<static_cast<int>((count + 255) / 256), 256, 0, stream>>>(
      static_cast<const float*>(partial), static_cast<__nv_bfloat16*>(output), inter,
      splits, slots);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_moe_router_topk_bf16(
    const void* logits, void* topk_idx, void* topk_weight, int tokens, int experts,
    int k, int renormalize, cudaStream_t stream) {
  if (tokens <= 0 || experts <= 0 || experts > 256 || k <= 0 || k > experts ||
      k > 32)
    return cudaErrorInvalidValue;
  constexpr int warps_per_block = 4;
  const int blocks = (tokens + warps_per_block - 1) / warps_per_block;
  moe_router_topk_bf16_kernel<<<blocks, warps_per_block * 32, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(logits), static_cast<int32_t*>(topk_idx),
      static_cast<float*>(topk_weight), tokens, experts, k, renormalize);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gather_rows_bf16(
    const void* x, const void* source_rows, void* output, int rows, int cols,
    cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
  const int64_t count = static_cast<int64_t>(rows) * cols;
  gather_rows_bf16_kernel<<<grid_stride_blocks(count, 256, 4096), 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x), static_cast<const int32_t*>(source_rows),
      static_cast<__nv_bfloat16*>(output), rows, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_weighted_gather_sum_bf16(
    const void* y, const void* slot_rows, const void* weight, void* output,
    int tokens, int k, int cols, cudaStream_t stream) {
  if (tokens <= 0 || k <= 0 || cols <= 0) return cudaErrorInvalidValue;
  const int64_t count = static_cast<int64_t>(tokens) * cols;
  weighted_gather_sum_bf16_kernel<<<grid_stride_blocks(count, 256, 4096), 256, 0,
                                    stream>>>(
      static_cast<const __nv_bfloat16*>(y), static_cast<const int32_t*>(slot_rows),
      static_cast<const float*>(weight), static_cast<__nv_bfloat16*>(output), tokens,
      k, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_silu_mul_rows_bf16(
    const void* gate_up, void* output, int rows, int inter, cudaStream_t stream) {
  if (rows <= 0 || inter <= 0) return cudaErrorInvalidValue;
  const int64_t count = static_cast<int64_t>(rows) * inter;
  silu_mul_rows_bf16_kernel<<<grid_stride_blocks(count, 256, 4096), 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(gate_up), static_cast<__nv_bfloat16*>(output),
      rows, inter);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_convert_bf16_f16(
    const void* input, void* output, int64_t count, cudaStream_t stream) {
  if (count <= 0) return cudaErrorInvalidValue;
  convert_bf16_to_f16_kernel<<<grid_stride_blocks(count, 256, 4096), 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input), static_cast<half*>(output), count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_convert_f16_bf16(
    const void* input, void* output, int64_t count, cudaStream_t stream) {
  if (count <= 0) return cudaErrorInvalidValue;
  convert_f16_to_bf16_kernel<<<grid_stride_blocks(count, 256, 4096), 256, 0, stream>>>(
      static_cast<const half*>(input), static_cast<__nv_bfloat16*>(output), count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gqa_decode_bf16(const void* q,const void* k,const void* v,
    void* partial,void* out,const void* position,int capacity,int splits,int kv_heads,float scale,cudaStream_t stream) {
  if(capacity<=0 || capacity==INT32_MAX || splits<=0 || splits>128 || kv_heads<=0 || kv_heads>8191)return cudaErrorInvalidValue;
  gqa_decode_partial_bf16_kernel<<<dim3(kv_heads,splits),256,0,stream>>>(
    static_cast<const __nv_bfloat16*>(q),static_cast<const __nv_bfloat16*>(k),static_cast<const __nv_bfloat16*>(v),
    static_cast<float*>(partial),static_cast<const uint32_t*>(position),capacity,splits,kv_heads,scale);
  gqa_decode_combine_bf16_kernel<<<kv_heads*8,128,0,stream>>>(static_cast<const float*>(partial),
    static_cast<__nv_bfloat16*>(out),kv_heads*8,splits);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gqa_decode_vector_bf16(const void* q,const void* k,const void* v,
    void* partial,void* out,const void* position,int capacity,int splits,int kv_heads,float scale,cudaStream_t stream) {
  if(capacity<=0 || capacity>=INT32_MAX-128 || splits<=0 || splits>128 || kv_heads<=0 || kv_heads>8191)return cudaErrorInvalidValue;
  if((reinterpret_cast<uintptr_t>(q)|reinterpret_cast<uintptr_t>(k)|reinterpret_cast<uintptr_t>(v))&15)return cudaErrorInvalidValue;
  gqa_decode_vector_bf16_kernel<<<dim3(kv_heads,splits),256,0,stream>>>(
    static_cast<const __nv_bfloat16*>(q),static_cast<const __nv_bfloat16*>(k),static_cast<const __nv_bfloat16*>(v),
    static_cast<float*>(partial),static_cast<const uint32_t*>(position),capacity,splits,kv_heads,scale);
  gqa_decode_combine_bf16_kernel<<<kv_heads*8,128,0,stream>>>(static_cast<const float*>(partial),
    static_cast<__nv_bfloat16*>(out),kv_heads*8,splits);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gqa_decode_balanced_bf16(const void* q,const void* k,const void* v,
    void* partial,void* out,const void* position,int capacity,int splits,int kv_heads,float scale,cudaStream_t stream) {
  if(capacity<=0 || capacity>=INT32_MAX-128 || splits<=0 || splits>128 || kv_heads<=0 || kv_heads>8191)return cudaErrorInvalidValue;
  if((reinterpret_cast<uintptr_t>(q)|reinterpret_cast<uintptr_t>(k)|reinterpret_cast<uintptr_t>(v))&15)return cudaErrorInvalidValue;
  gqa_decode_vector_bf16_kernel<true><<<dim3(kv_heads,splits),256,0,stream>>>(
    static_cast<const __nv_bfloat16*>(q),static_cast<const __nv_bfloat16*>(k),static_cast<const __nv_bfloat16*>(v),
    static_cast<float*>(partial),static_cast<const uint32_t*>(position),capacity,splits,kv_heads,scale);
  gqa_decode_combine_bf16_kernel<<<kv_heads*8,128,0,stream>>>(static_cast<const float*>(partial),
    static_cast<__nv_bfloat16*>(out),kv_heads*8,splits);
  return cudaGetLastError();
}

// Tile32/64 use at most 44,672 dynamic shared bytes, below the default
// 48-KiB limit. No attribute mutation occurs inside graph capture.
extern "C" cudaError_t apxinf_gqa_decode_mma_bf16(const void* q,const void* k,const void* v,
    void* partial,void* out,const void* position,int capacity,int splits,int kv_heads,int tile,float scale,cudaStream_t stream) {
  if(capacity<=0 || capacity>=INT32_MAX-128 || splits<=0 || splits>128 || kv_heads<=0 || kv_heads>8191 || (tile!=32 && tile!=64))return cudaErrorInvalidValue;
  const int shared=8*136*2+tile*136*2+tile*128*2+8*(tile+4)*4+2*8*(tile+8)*2+8*132*4;
  auto kernel=tile==64?gqa_decode_mma_bf16_kernel<64>:gqa_decode_mma_bf16_kernel<32>;
  kernel<<<dim3(kv_heads,splits),256,shared,stream>>>(
    static_cast<const __nv_bfloat16*>(q),static_cast<const __nv_bfloat16*>(k),static_cast<const __nv_bfloat16*>(v),
    static_cast<float*>(partial),static_cast<const uint32_t*>(position),capacity,splits,kv_heads,scale);
  auto status=cudaGetLastError();if(status!=cudaSuccess)return status;
  gqa_decode_combine_bf16_kernel<<<kv_heads*8,128,0,stream>>>(static_cast<const float*>(partial),
    static_cast<__nv_bfloat16*>(out),kv_heads*8,splits);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qkv_partial_norm_rope_cache_bf16(const void* partial,
    const void* qw,const void* kw,void* oq,void* ck,void* cv,const void* position,
    int qheads,int kvheads,int capacity,int splits,float eps,const void* table,cudaStream_t stream) {
  if(qheads<=0 || kvheads<=0 || int64_t(qheads)+2*kvheads>INT32_MAX/128 || capacity<=0 || splits<=0)return cudaErrorInvalidValue;
  qkv_partial_norm_rope_cache_bf16_kernel<<<(qheads+kvheads+3)/4,128,0,stream>>>(
    static_cast<const float*>(partial),static_cast<const __nv_bfloat16*>(qw),static_cast<const __nv_bfloat16*>(kw),
    static_cast<__nv_bfloat16*>(oq),static_cast<__nv_bfloat16*>(ck),static_cast<__nv_bfloat16*>(cv),
    static_cast<const uint32_t*>(position),qheads,kvheads,capacity,splits,eps,static_cast<const float*>(table));
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_partial_residual_rms_bf16(void* x,const void* partial,
    const void* weight,void* output,int cols,int count,float eps,cudaStream_t stream) {
  if(cols<=0 || cols>8192 || count<=0)return cudaErrorInvalidValue;
  if(cols>=2048) {
  partial_residual_rms_bf16_kernel<true><<<1,1024,cols*4,stream>>>(static_cast<__nv_bfloat16*>(x),
    static_cast<const float*>(partial),static_cast<const __nv_bfloat16*>(weight),
    static_cast<__nv_bfloat16*>(output),cols,1,count,eps);
  } else {
  partial_residual_rms_bf16_kernel<<<1,256,cols*4,stream>>>(static_cast<__nv_bfloat16*>(x),
    static_cast<const float*>(partial),static_cast<const __nv_bfloat16*>(weight),
    static_cast<__nv_bfloat16*>(output),cols,1,count,eps);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_partial_residual_rms_rows_bf16(void* x,const void* partial,
    const void* weight,void* output,int cols,int rows,int count,float eps,cudaStream_t stream) {
  if(cols<=0 || cols>8192 || rows<=0 || count<=0 || !(eps>0) || !isfinite(eps) ||
     int64_t(rows)*cols>UINT32_MAX)return cudaErrorInvalidValue;
  // Multi-row prefill favors one 256-thread block per row. Decode keeps its
  // separately tuned 1024-thread load and original 256-thread reduction tree.
  partial_residual_rms_bf16_kernel<<<rows,256,cols*4,stream>>>(static_cast<__nv_bfloat16*>(x),
      static_cast<const float*>(partial),static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(output),cols,rows,count,eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_routed_residual_rms_bf16(void* x,const void* y,const void* ids,const void* rw,
    const void* weight,void* output,int cols,int rows,int topk,float eps,int input_f16,cudaStream_t stream) {
  if(cols<=0 || cols>8192 || rows<=0 || topk<=0)return cudaErrorInvalidValue;
  const bool vector = (cols==2048 || cols==4096)
      && (reinterpret_cast<uintptr_t>(x)&15)==0 && (reinterpret_cast<uintptr_t>(y)&15)==0;
  auto selected = input_f16 ? (vector ? routed_residual_rms_vector_kernel<true> : routed_residual_rms_bf16_kernel<true>)
      : (vector ? routed_residual_rms_vector_kernel<false> : routed_residual_rms_bf16_kernel<false>);
  selected<<<rows,256,cols*4,stream>>>(static_cast<__nv_bfloat16*>(x),
      static_cast<const __nv_bfloat16*>(y),static_cast<const int32_t*>(ids),static_cast<const float*>(rw),
      static_cast<const __nv_bfloat16*>(weight),static_cast<__nv_bfloat16*>(output),cols,rows,topk,eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_silu_mul_rows_f16_rounded(const void* gu,void* output,int rows,int inter,cudaStream_t stream) {
  if(rows<=0 || inter<=0)return cudaErrorInvalidValue;
  int64_t count=(int64_t(rows)*inter+255)/256;int blocks=int(count<256?count:256);
  silu_mul_rows_f16_rounded_kernel<<<blocks,256,0,stream>>>(static_cast<const half*>(gu),static_cast<half*>(output),rows,inter);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_silu_bf16_table(void* table,cudaStream_t stream) {
  silu_bf16_table_kernel<<<256,256,0,stream>>>(static_cast<float*>(table));return cudaGetLastError();
}
extern "C" cudaError_t apxinf_silu_mul_rows_f16_lut(const void* gu,void* output,
    const void* table,int rows,int inter,cudaStream_t stream) {
  if(rows<=0 || inter<=0)return cudaErrorInvalidValue;
  int blocks=int(std::min<int64_t>(256,(int64_t(rows)*inter+255)/256));
  if(inter==768 && rows<=(INT32_MAX-65536)/96
      && (reinterpret_cast<uintptr_t>(gu)&15)==0 && (reinterpret_cast<uintptr_t>(output)&15)==0) {
    silu_mul_rows_f16_lut_768_kernel<<<blocks,256,0,stream>>>(static_cast<const half*>(gu),
        static_cast<half*>(output),static_cast<const float*>(table),rows);
  } else {
    silu_mul_rows_f16_lut_kernel<<<blocks,256,0,stream>>>(static_cast<const half*>(gu),
        static_cast<half*>(output),static_cast<const float*>(table),rows,inter);
  }
  return cudaGetLastError();
}

// BF16 activations with checkpoint FP16 RMSNorm weights.
extern "C" cudaError_t apxinf_partial_residual_rms_bf16_f16_weight(void* x,const void* partial,
    const void* weight,void* output,int cols,int count,float eps,cudaStream_t stream) {
  if(cols<=0 || cols>8192 || count<=0)return cudaErrorInvalidValue;
  if(cols>=2048) {
  partial_residual_rms_bf16_kernel<true,false,half><<<1,1024,cols*4,stream>>>(static_cast<__nv_bfloat16*>(x),
    static_cast<const float*>(partial),static_cast<const half*>(weight),
    static_cast<__nv_bfloat16*>(output),cols,1,count,eps);
  } else {
  partial_residual_rms_bf16_kernel<false,false,half><<<1,256,cols*4,stream>>>(static_cast<__nv_bfloat16*>(x),
    static_cast<const float*>(partial),static_cast<const half*>(weight),
    static_cast<__nv_bfloat16*>(output),cols,1,count,eps);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_partial_residual_rms_rows_bf16_f16_weight(void* x,const void* partial,
    const void* weight,void* output,int cols,int rows,int count,float eps,cudaStream_t stream) {
  if(cols<=0 || cols>8192 || rows<=0 || count<=0 || !(eps>0) || !isfinite(eps) ||
     int64_t(rows)*cols>UINT32_MAX)return cudaErrorInvalidValue;
  // Multi-row prefill favors one 256-thread block per row. Decode keeps its
  // separately tuned 1024-thread load and original 256-thread reduction tree.
  partial_residual_rms_bf16_kernel<false,false,half><<<rows,256,cols*4,stream>>>(static_cast<__nv_bfloat16*>(x),
      static_cast<const float*>(partial),static_cast<const half*>(weight),
      static_cast<__nv_bfloat16*>(output),cols,rows,count,eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_routed_residual_rms_bf16_f16_weight(void* x,const void* y,const void* ids,const void* rw,
    const void* weight,void* output,int cols,int rows,int topk,float eps,int input_f16,cudaStream_t stream) {
  if(cols<=0 || cols>8192 || rows<=0 || topk<=0)return cudaErrorInvalidValue;
  const bool vector = (cols==2048 || cols==4096)
      && (reinterpret_cast<uintptr_t>(x)&15)==0 && (reinterpret_cast<uintptr_t>(y)&15)==0;
  auto selected = input_f16 ? (vector ? routed_residual_rms_vector_kernel<true,half> : routed_residual_rms_bf16_kernel<true,half>)
      : (vector ? routed_residual_rms_vector_kernel<false,half> : routed_residual_rms_bf16_kernel<false,half>);
  selected<<<rows,256,cols*4,stream>>>(static_cast<__nv_bfloat16*>(x),
      static_cast<const __nv_bfloat16*>(y),static_cast<const int32_t*>(ids),static_cast<const float*>(rw),
      static_cast<const half*>(weight),static_cast<__nv_bfloat16*>(output),cols,rows,topk,eps);
  return cudaGetLastError();
}
