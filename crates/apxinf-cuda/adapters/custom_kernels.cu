// Copyright 2026 apxinf contributors.
// Stable C ABI and CUDA launch adapter for custom static-inference operators.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {
#include "../kernels/custom/math.cuh"
#include "../kernels/custom/reduction.cuh"
#include "../kernels/custom/quantization.cuh"
#include "../kernels/custom/qwen35_w4.cuh"
#include "../kernels/custom/qwen35_gdn.cuh"
#include "../kernels/custom/preprocess.cuh"
#include "../kernels/custom/attention.cuh"
#include "../kernels/custom/normalization.cuh"
#include "../kernels/custom/activation.cuh"
#include "../kernels/custom/embedding.cuh"
#include "../kernels/custom/elementwise.cuh"
#include "../kernels/custom/fused.cuh"
#include "../kernels/custom/cache.cuh"
}  // namespace

extern "C" cudaError_t apxinf_qwen35_gdn_check_finite_bf16(
    const void* input, void* error_flags, int elements, cudaStream_t stream) {
  if (input == nullptr || error_flags == nullptr || elements <= 0) {
    return cudaErrorInvalidValue;
  }
  const int blocks = (elements + 255) / 256;
  qwen35_gdn_check_finite_bf16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<uint32_t*>(error_flags), elements);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_w4_project_bf16(
    const void* activation, const void* weight_packed, const void* scales,
    const void* zero_points, void* output, void* error_flags, int rows,
    int out_features, int in_features, int group_size, cudaStream_t stream) {
  if (activation == nullptr || weight_packed == nullptr || scales == nullptr ||
      zero_points == nullptr || output == nullptr || error_flags == nullptr ||
      rows <= 0 || out_features <= 0 || in_features <= 0 || group_size != 32) {
    return cudaErrorInvalidValue;
  }
  const int64_t blocks = static_cast<int64_t>(rows) * out_features;
  if (blocks > INT32_MAX) return cudaErrorInvalidConfiguration;
  qwen35_w4_project_bf16_kernel<<<static_cast<int>(blocks), 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(activation),
      static_cast<const uint32_t*>(weight_packed),
      static_cast<const __nv_bfloat16*>(scales),
      static_cast<const uint32_t*>(zero_points),
      static_cast<__nv_bfloat16*>(output),
      static_cast<uint32_t*>(error_flags), rows, out_features, in_features,
      group_size);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_w4_project_bf16_packed(
    const void* activation, const void* weight_packed, const void* scales,
    const void* zero_points, void* output, void* error_flags, int rows,
    int out_features, int in_features, int group_size, cudaStream_t stream) {
  if (activation == nullptr || weight_packed == nullptr || scales == nullptr ||
      zero_points == nullptr || output == nullptr || error_flags == nullptr ||
      rows <= 0 || out_features <= 0 || in_features <= 0 || group_size != 32) {
    return cudaErrorInvalidValue;
  }
  const int64_t blocks = static_cast<int64_t>(rows) * out_features;
  if (blocks > INT32_MAX) return cudaErrorInvalidConfiguration;
  qwen35_w4_project_bf16_packed_kernel<<<static_cast<int>(blocks), 256, 0,
                                        stream>>>(
      static_cast<const __nv_bfloat16*>(activation),
      static_cast<const uint32_t*>(weight_packed),
      static_cast<const __nv_bfloat16*>(scales),
      static_cast<const uint32_t*>(zero_points),
      static_cast<__nv_bfloat16*>(output),
      static_cast<uint32_t*>(error_flags), rows, out_features, in_features,
      group_size);
  return cudaGetLastError();
}

// Diagnostic-only launchers. `variant` selects: 1 = no-dequant (wrong math),
// 2 = stream-only (meaningless math), 3 = vec4 (correct math, vectorized
// loads). Only variant 3 is a production candidate.
extern "C" cudaError_t apxinf_static_qwen35_w4_project_bf16_diag(
    const void* activation, const void* weight_packed, const void* scales,
    const void* zero_points, void* output, void* error_flags, int rows,
    int out_features, int in_features, int group_size, int variant,
    cudaStream_t stream) {
  if (activation == nullptr || weight_packed == nullptr || scales == nullptr ||
      zero_points == nullptr || output == nullptr || error_flags == nullptr ||
      rows <= 0 || out_features <= 0 || in_features <= 0 || group_size != 32) {
    return cudaErrorInvalidValue;
  }
  const int64_t blocks = static_cast<int64_t>(rows) * out_features;
  if (blocks > INT32_MAX) return cudaErrorInvalidConfiguration;
  const int grid = static_cast<int>(blocks);
  switch (variant) {
    case 1:
      qwen35_w4_diag_nodequant_kernel<<<grid, 256, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(activation),
          static_cast<const uint32_t*>(weight_packed),
          static_cast<const __nv_bfloat16*>(scales),
          static_cast<const uint32_t*>(zero_points),
          static_cast<__nv_bfloat16*>(output),
          static_cast<uint32_t*>(error_flags), rows, out_features, in_features,
          group_size);
      break;
    case 2:
      qwen35_w4_diag_streamonly_kernel<<<grid, 256, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(activation),
          static_cast<const uint32_t*>(weight_packed),
          static_cast<const __nv_bfloat16*>(scales),
          static_cast<const uint32_t*>(zero_points),
          static_cast<__nv_bfloat16*>(output),
          static_cast<uint32_t*>(error_flags), rows, out_features, in_features,
          group_size);
      break;
    case 3:
      qwen35_w4_project_bf16_vec4_kernel<<<grid, 256, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(activation),
          static_cast<const uint32_t*>(weight_packed),
          static_cast<const __nv_bfloat16*>(scales),
          static_cast<const uint32_t*>(zero_points),
          static_cast<__nv_bfloat16*>(output),
          static_cast<uint32_t*>(error_flags), rows, out_features, in_features,
          group_size);
      break;
    case 4:
      qwen35_w4_project_bf16_fast_kernel<false><<<grid, 256, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(activation),
          static_cast<const uint32_t*>(weight_packed),
          static_cast<const __nv_bfloat16*>(scales),
          static_cast<const uint32_t*>(zero_points),
          static_cast<__nv_bfloat16*>(output),
          static_cast<uint32_t*>(error_flags), rows, out_features, in_features,
          group_size);
      break;
    case 5:
      qwen35_w4_project_bf16_fast_kernel<true><<<grid, 256, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(activation),
          static_cast<const uint32_t*>(weight_packed),
          static_cast<const __nv_bfloat16*>(scales),
          static_cast<const uint32_t*>(zero_points),
          static_cast<__nv_bfloat16*>(output),
          static_cast<uint32_t*>(error_flags), rows, out_features, in_features,
          group_size);
      break;
    case 6:
      qwen35_w4_project_bf16_marlin_kernel<<<grid, 256, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(activation),
          static_cast<const uint32_t*>(weight_packed),
          static_cast<const __nv_bfloat16*>(scales),
          static_cast<const uint32_t*>(zero_points),
          static_cast<__nv_bfloat16*>(output),
          static_cast<uint32_t*>(error_flags), rows, out_features, in_features,
          group_size);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_w4_project_bf16_marlin(
    const void* activation, const void* weight_packed, const void* scales,
    const void* zero_points, void* output, void* error_flags, int rows,
    int out_features, int in_features, int group_size, cudaStream_t stream) {
  if (activation == nullptr || weight_packed == nullptr || scales == nullptr ||
      zero_points == nullptr || output == nullptr || error_flags == nullptr ||
      rows <= 0 || out_features <= 0 || in_features <= 0 || group_size != 32) {
    return cudaErrorInvalidValue;
  }
  const int64_t blocks = static_cast<int64_t>(rows) * out_features;
  if (blocks > INT32_MAX) return cudaErrorInvalidConfiguration;
  qwen35_w4_project_bf16_marlin_kernel<<<static_cast<int>(blocks), 256, 0,
                                        stream>>>(
      static_cast<const __nv_bfloat16*>(activation),
      static_cast<const uint32_t*>(weight_packed),
      static_cast<const __nv_bfloat16*>(scales),
      static_cast<const uint32_t*>(zero_points),
      static_cast<__nv_bfloat16*>(output),
      static_cast<uint32_t*>(error_flags), rows, out_features, in_features,
      group_size);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_w4_project_bf16_warp(
    const void* activation, const void* weight_packed, const void* scales,
    const void* zero_points, void* output, void* error_flags, int rows,
    int out_features, int in_features, int group_size, cudaStream_t stream) {
  if (activation == nullptr || weight_packed == nullptr || scales == nullptr ||
      zero_points == nullptr || output == nullptr || error_flags == nullptr ||
      rows <= 0 || out_features <= 0 || in_features <= 0 || group_size != 32) {
    return cudaErrorInvalidValue;
  }
  constexpr int kWarpsPerBlock = 8;
  constexpr int kThreads = kWarpsPerBlock * 32;
  const size_t shared_bytes = static_cast<size_t>(in_features) * sizeof(float);
  // The activation row must fit in the 48 KB default shared-memory budget.
  if (shared_bytes > 48u * 1024u) return cudaErrorInvalidConfiguration;
  const int blocks_x = (out_features + kWarpsPerBlock - 1) / kWarpsPerBlock;
  if (blocks_x <= 0 || rows > 65535) return cudaErrorInvalidConfiguration;
  dim3 grid(static_cast<unsigned>(blocks_x), static_cast<unsigned>(rows));
  qwen35_w4_project_bf16_warp_kernel<kWarpsPerBlock>
      <<<grid, kThreads, shared_bytes, stream>>>(
          static_cast<const __nv_bfloat16*>(activation),
          static_cast<const uint32_t*>(weight_packed),
          static_cast<const __nv_bfloat16*>(scales),
          static_cast<const uint32_t*>(zero_points),
          static_cast<__nv_bfloat16*>(output),
          static_cast<uint32_t*>(error_flags), rows, out_features, in_features,
          group_size);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_w4_dequantize_bf16(
    const void* weight_packed, const void* scales, const void* zero_points,
    void* dequantized, void* error_flags, int out_features, int in_features,
    int group_size, cudaStream_t stream) {
  if (weight_packed == nullptr || scales == nullptr ||
      zero_points == nullptr || dequantized == nullptr ||
      error_flags == nullptr || out_features <= 0 || in_features <= 0 ||
      group_size != 32) {
    return cudaErrorInvalidValue;
  }
  const int64_t total =
      static_cast<int64_t>(out_features) * static_cast<int64_t>(in_features);
  const int threads = 256;
  int64_t blocks = (total + threads - 1) / threads;
  // Grid-stride loop: cap the grid and let each thread walk the remainder.
  const int64_t max_blocks = 65535;
  if (blocks > max_blocks) blocks = max_blocks;
  qwen35_w4_dequantize_bf16_kernel<<<static_cast<int>(blocks), threads, 0,
                                     stream>>>(
      static_cast<const uint32_t*>(weight_packed),
      static_cast<const __nv_bfloat16*>(scales),
      static_cast<const uint32_t*>(zero_points),
      static_cast<__nv_bfloat16*>(dequantized),
      static_cast<uint32_t*>(error_flags), out_features, in_features,
      group_size);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_gdn_conv_bf16(
    const void* ring_in, void* ring_out, const void* input,
    const void* weights, void* output, void* error_flags, int channels,
    int kernel, int cursor, cudaStream_t stream) {
  if (ring_in == nullptr || ring_out == nullptr || input == nullptr ||
      weights == nullptr || output == nullptr || error_flags == nullptr ||
      channels <= 0 || kernel <= 0 || cursor < 0 || cursor >= kernel) {
    return cudaErrorInvalidValue;
  }
  int blocks = (channels + 255) / 256;
  qwen35_gdn_conv_bf16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(ring_in),
      static_cast<__nv_bfloat16*>(ring_out),
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(weights),
      static_cast<__nv_bfloat16*>(output),
      static_cast<uint32_t*>(error_flags), channels, kernel, cursor);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_gdn_conv_prefill_bf16(
    const void* ring_in, void* ring_out, const void* input,
    const void* weights, void* output, void* error_flags, int rows,
    int channels, int kernel, int cursor, cudaStream_t stream) {
  if (ring_in == nullptr || ring_out == nullptr || input == nullptr ||
      weights == nullptr || output == nullptr || error_flags == nullptr ||
      rows <= 0 || channels <= 0 || kernel <= 0 || cursor < 0 ||
      cursor >= kernel) {
    return cudaErrorInvalidValue;
  }
  int blocks = (channels + 255) / 256;
  qwen35_gdn_conv_prefill_bf16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(ring_in),
      static_cast<__nv_bfloat16*>(ring_out),
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(weights),
      static_cast<__nv_bfloat16*>(output), static_cast<uint32_t*>(error_flags),
      rows, channels, kernel, cursor);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_gdn_conv_batch_bf16(
    const void* ring_in_ptrs, const void* ring_out_ptrs, const void* input,
    const void* weights, void* output, const void* cursors, void* error_flags,
    int batch, int channels, int kernel, cudaStream_t stream) {
  if (ring_in_ptrs == nullptr || ring_out_ptrs == nullptr ||
      input == nullptr || weights == nullptr || output == nullptr ||
      cursors == nullptr || error_flags == nullptr || batch <= 0 ||
      channels <= 0 || kernel <= 0) {
    return cudaErrorInvalidValue;
  }
  dim3 grid((channels + 255) / 256, batch, 1);
  qwen35_gdn_conv_batch_bf16_kernel<<<grid, 256, 0, stream>>>(
      static_cast<const unsigned long long*>(ring_in_ptrs),
      static_cast<const unsigned long long*>(ring_out_ptrs),
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(weights),
      static_cast<__nv_bfloat16*>(output),
      static_cast<const int*>(cursors),
      static_cast<uint32_t*>(error_flags), channels, kernel);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_gdn_recurrent_batch_bf16_f32(
    const void* state_in_ptrs, const void* state_out_ptrs, const void* query,
    const void* key, const void* value, const void* a, const void* b,
    const void* a_log, const void* dt_bias, void* output, void* error_flags,
    int batch, int key_heads, int value_heads, int key_dim, int value_dim,
    cudaStream_t stream) {
  if (state_in_ptrs == nullptr || state_out_ptrs == nullptr ||
      query == nullptr || key == nullptr || value == nullptr ||
      a == nullptr || b == nullptr || a_log == nullptr ||
      dt_bias == nullptr || output == nullptr || error_flags == nullptr ||
      batch <= 0 || key_heads <= 0 || value_heads <= 0 ||
      value_heads % key_heads != 0 || key_dim <= 0 || value_dim <= 0) {
    return cudaErrorInvalidValue;
  }
  dim3 grid(value_heads, batch, 1);
  qwen35_gdn_recurrent_batch_bf16_f32_kernel<<<grid, 256, 0, stream>>>(
      static_cast<const unsigned long long*>(state_in_ptrs),
      static_cast<const unsigned long long*>(state_out_ptrs),
      static_cast<const __nv_bfloat16*>(query),
      static_cast<const __nv_bfloat16*>(key),
      static_cast<const __nv_bfloat16*>(value),
      static_cast<const __nv_bfloat16*>(a),
      static_cast<const __nv_bfloat16*>(b),
      static_cast<const __nv_bfloat16*>(a_log),
      static_cast<const __nv_bfloat16*>(dt_bias),
      static_cast<__nv_bfloat16*>(output),
      static_cast<uint32_t*>(error_flags), key_heads, value_heads, key_dim,
      value_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_gdn_recurrent_bf16_f32(
    const void* state_in, void* state_out, const void* query, const void* key,
    const void* value, const void* a, const void* b, const void* a_log,
    const void* dt_bias, void* output, void* error_flags, int key_heads,
    int value_heads, int key_dim, int value_dim, cudaStream_t stream) {
  if (state_in == nullptr || state_out == nullptr || query == nullptr ||
      key == nullptr || value == nullptr || a == nullptr || b == nullptr ||
      a_log == nullptr || dt_bias == nullptr || output == nullptr ||
      error_flags == nullptr || key_heads <= 0 || value_heads <= 0 ||
      value_heads % key_heads != 0 || key_dim <= 0 || value_dim <= 0) {
    return cudaErrorInvalidValue;
  }
  qwen35_gdn_recurrent_bf16_f32_kernel<<<value_heads, 256, 0, stream>>>(
      static_cast<const float*>(state_in), static_cast<float*>(state_out),
      static_cast<const __nv_bfloat16*>(query),
      static_cast<const __nv_bfloat16*>(key),
      static_cast<const __nv_bfloat16*>(value),
      static_cast<const __nv_bfloat16*>(a),
      static_cast<const __nv_bfloat16*>(b),
      static_cast<const __nv_bfloat16*>(a_log),
      static_cast<const __nv_bfloat16*>(dt_bias),
      static_cast<__nv_bfloat16*>(output),
      static_cast<uint32_t*>(error_flags), key_heads, value_heads, key_dim,
      value_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_gdn_sequence_recurrent_bf16_f32(
    const void* state_in, void* state_out, const void* query, const void* key,
    const void* value, const void* a, const void* b, const void* a_log,
    const void* dt_bias, void* output, void* error_flags, int rows,
    int key_heads, int value_heads, int key_dim, int value_dim,
    void* workspace, int64_t workspace_stride, void* qk_scores,
    void* transition_scores,
    int chunk_index, int phase,
    cudaStream_t stream) {
  if (state_in == nullptr || state_out == nullptr || query == nullptr ||
      key == nullptr || value == nullptr || a == nullptr || b == nullptr ||
      a_log == nullptr || dt_bias == nullptr || output == nullptr ||
      error_flags == nullptr || workspace == nullptr || qk_scores == nullptr ||
      transition_scores == nullptr ||
      workspace_stride <= 0 || chunk_index < 0 || phase < 0 || phase > 2 ||
      rows <= 0 || key_heads <= 0 ||
      value_heads <= 0 || value_heads % key_heads != 0 || key_dim <= 0 ||
      value_dim <= 0) {
    return cudaErrorInvalidValue;
  }
  // One block per value head. 256 lanes cover the (row, dimension) work of a
  // 64-token chunk; every reduction that feeds a stored value still runs in a
  // single lane so the FP32 accumulation order is unchanged.
  qwen35_gdn_sequence_recurrent_bf16_f32_kernel<<<value_heads, 256, 0, stream>>>(
      static_cast<const float*>(state_in), static_cast<float*>(state_out),
      static_cast<const __nv_bfloat16*>(query),
      static_cast<const __nv_bfloat16*>(key),
      static_cast<const __nv_bfloat16*>(value),
      static_cast<const __nv_bfloat16*>(a),
      static_cast<const __nv_bfloat16*>(b),
      static_cast<const __nv_bfloat16*>(a_log),
      static_cast<const __nv_bfloat16*>(dt_bias),
      static_cast<__nv_bfloat16*>(output), static_cast<uint32_t*>(error_flags),
      rows, key_heads, value_heads, key_dim, value_dim,
      static_cast<float*>(workspace), workspace_stride,
      static_cast<float*>(qk_scores), static_cast<float*>(transition_scores),
      chunk_index, phase);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qwen35_gdn_gated_rms_norm_bf16(
    const void* input, const void* gate, const void* weight, void* output,
    void* error_flags, int rows, int heads, int head_dim, float eps,
    cudaStream_t stream) {
  if (input == nullptr || gate == nullptr || weight == nullptr ||
      output == nullptr || error_flags == nullptr || rows <= 0 || heads <= 0 ||
      head_dim <= 0 || !(eps > 0.0f) || !isfinite(eps)) {
    return cudaErrorInvalidValue;
  }
  qwen35_gdn_gated_rms_norm_bf16_kernel<<<rows * heads, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(output),
      static_cast<uint32_t*>(error_flags), rows, heads, head_dim, eps);
  return cudaGetLastError();
}

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
