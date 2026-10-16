// Copyright 2026 apxinf contributors.
// Stable C ABI and CUDA launch policy for static-path BF16 operators.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <cstdint>
#include <cstdint>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
__device__ __forceinline__ float2 __bfloat1622float2(__nv_bfloat162 value) {
    return make_float2(__bfloat162float(value.x), __bfloat162float(value.y));
}
#endif

namespace {
#include "../kernels/custom/math.cuh"
#include "../kernels/custom/reduction.cuh"
#include "../kernels/custom/preprocess.cuh"
#include "../kernels/custom/activation.cuh"
#include "../kernels/custom/embedding.cuh"
#include "../kernels/custom/elementwise.cuh"
#include "../kernels/custom/normalization.cuh"
#include "../kernels/custom/quantization.cuh"
#include "../kernels/custom/fused.cuh"
#include "../kernels/custom/attention.cuh"

int blocks_for(int64_t count) {
  return static_cast<int>((count + kThreads - 1) / kThreads);
}
// A load-ahead CTA reads the whole projection/residual row before writing
// hidden. The __restrict__ specialization also requires both output ranges
// to be disjoint from all read-only input ranges and from each other. Inputs
// may alias one another, because none is written by this kernel.
bool byte_ranges_disjoint(const void* first, size_t first_bytes,
                          const void* second, size_t second_bytes) {
  if (first == nullptr || second == nullptr) return false;
  const uintptr_t a = reinterpret_cast<uintptr_t>(first);
  const uintptr_t b = reinterpret_cast<uintptr_t>(second);
  if (first_bytes > std::numeric_limits<uintptr_t>::max() - a ||
      second_bytes > std::numeric_limits<uintptr_t>::max() - b)
    return false;
  return a <= b ? b - a >= first_bytes : a - b >= second_bytes;
}

bool layer_norm_1024_load_ahead_noalias(
    const void* projection, const void* residual,
    const void* norm_weight, const void* norm_bias,
    const void* hidden, const void* normalized, int rows) {
  constexpr size_t kRowBytes = 1024 * sizeof(__nv_bfloat16);
  if (rows <= 0 || static_cast<size_t>(rows) >
                       std::numeric_limits<size_t>::max() / kRowBytes)
    return false;
  const size_t matrix_bytes = static_cast<size_t>(rows) * kRowBytes;
  const void* inputs[] = {projection, residual, norm_weight, norm_bias};
  const size_t input_bytes[] = {matrix_bytes, matrix_bytes,
                                kRowBytes, kRowBytes};
  const void* outputs[] = {hidden, normalized};
  if (!byte_ranges_disjoint(hidden, matrix_bytes, normalized, matrix_bytes))
    return false;
  for (const void* output : outputs) {
    for (int i = 0; i < 4; ++i) {
      if (!byte_ranges_disjoint(output, matrix_bytes, inputs[i], input_bytes[i]))
        return false;
    }
  }
  return true;
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

extern "C" cudaError_t apxinf_static_rgb_u8_to_patches_f16(
    const void* images, void* patches, int views, int image_size,
    int patch_size, int layout, cudaStream_t stream) {
  if (views <= 0 || image_size <= 0 || patch_size <= 0 ||
      image_size % patch_size != 0 || (layout != 0 && layout != 1))
    return cudaErrorInvalidValue;
  const int64_t count = static_cast<int64_t>(views) * 3 * image_size * image_size;
  if (layout == 0) {
    rgb_u8_to_patches_f16_kernel<true><<<blocks_for(count), kThreads, 0, stream>>>(
        static_cast<const uint8_t*>(images), static_cast<half*>(patches),
        views, image_size, patch_size);
  } else {
    rgb_u8_to_patches_f16_kernel<false><<<blocks_for(count), kThreads, 0, stream>>>(
        static_cast<const uint8_t*>(images), static_cast<half*>(patches),
        views, image_size, patch_size);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_pixel_shuffle_4_bf16(
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

extern "C" cudaError_t apxinf_static_pixel_shuffle_4_f16(
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

// PaliGemma keeps the SigLIP patch embedding in FP32, so the raw-image path
// produces an FP32 patch tensor for an FP32 patch projection.
extern "C" cudaError_t apxinf_static_rgb_u8_to_patches_f32(
    const void* images, void* patches, int views, int image_size,
    int patch_size, int layout, cudaStream_t stream) {
  if (views <= 0 || image_size <= 0 || patch_size <= 0 ||
      image_size % patch_size != 0 || (layout != 0 && layout != 1))
    return cudaErrorInvalidValue;
  const int64_t count = static_cast<int64_t>(views) * 3 * image_size * image_size;
  if (layout == 0) {
    rgb_u8_to_patches_f32_kernel<true><<<blocks_for(count), kThreads, 0, stream>>>(
        static_cast<const uint8_t*>(images), static_cast<float*>(patches),
        views, image_size, patch_size);
  } else {
    rgb_u8_to_patches_f32_kernel<false><<<blocks_for(count), kThreads, 0, stream>>>(
        static_cast<const uint8_t*>(images), static_cast<float*>(patches),
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

extern "C" cudaError_t apxinf_bias_activation_bf16_packed8(
    const void* input, const void* bias, void* output,
    int rows, int cols, int activation, cudaStream_t stream) {
  if (input == nullptr || bias == nullptr || output == nullptr || rows <= 0 ||
      cols <= 0 || cols % 8 != 0 || (activation != 0 && activation != 1))
    return cudaErrorInvalidValue;
  if (reinterpret_cast<uintptr_t>(input) % alignof(Bf16x8) != 0 ||
      reinterpret_cast<uintptr_t>(bias) % alignof(Bf16x8) != 0 ||
      reinterpret_cast<uintptr_t>(output) % alignof(Bf16x8) != 0)
    return cudaErrorInvalidValue;
  const int64_t oct_count = static_cast<int64_t>(rows) * cols / 8;
  bias_activation_bf16_packed8_kernel<<<
      blocks_for(oct_count), kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<__nv_bfloat16*>(output), oct_count, cols, activation);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_relu_bf16(
    const void* input, const void* bias, void* output,
    int rows, int cols, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || rows <= 0 || cols <= 0)
    return cudaErrorInvalidValue;
  const int64_t count = static_cast<int64_t>(rows) * cols;
  bias_relu_bf16_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<__nv_bfloat16*>(output), count, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_qkv_in_place_bf16(
    void* query, void* key, void* value, const void* query_bias,
    const void* key_bias, const void* value_bias, int rows, int cols,
    cudaStream_t stream) {
  if (!query || !key || !value || !query_bias || !key_bias || !value_bias ||
      rows <= 0 || cols <= 0 || cols % 4 != 0)
    return cudaErrorInvalidValue;
  constexpr int threads = 256;
  const int64_t groups = static_cast<int64_t>(rows) * cols / 4;
  int blocks = static_cast<int>((groups + threads - 1) / threads);
  blocks = blocks > 1024 ? 1024 : blocks;
  bias_qkv_in_place_bf16_packed4_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<__nv_bfloat16*>(query),
      static_cast<__nv_bfloat16*>(key),
      static_cast<__nv_bfloat16*>(value),
      static_cast<const __nv_bfloat16*>(query_bias),
      static_cast<const __nv_bfloat16*>(key_bias),
      static_cast<const __nv_bfloat16*>(value_bias), groups, cols / 4);
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

extern "C" cudaError_t apxinf_static_gather_rows_bf16(
    const void* input, const void* indices, void* output,
    int rows, int cols, cudaStream_t stream) {
  if (input == nullptr || indices == nullptr || output == nullptr ||
      rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
  const int64_t count = static_cast<int64_t>(rows) * cols;
  gather_rows_bf16_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const uint32_t*>(indices),
      static_cast<__nv_bfloat16*>(output), rows, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_scatter_rows_bf16(
    const void* source, const void* rows, void* output,
    int row_count, int cols, int add, cudaStream_t stream) {
  if (source == nullptr || rows == nullptr || output == nullptr ||
      row_count <= 0 || cols <= 0 || (add != 0 && add != 1))
    return cudaErrorInvalidValue;
  const int64_t count = static_cast<int64_t>(row_count) * cols;
  scatter_rows_bf16_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(source),
      static_cast<const uint32_t*>(rows),
      static_cast<__nv_bfloat16*>(output), count, cols, add != 0);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_replace_rows_bf16(
    const void* base, const void* replacement, const void* row_map,
    void* output, int rows, int cols, cudaStream_t stream) {
  if (base == nullptr || replacement == nullptr || row_map == nullptr ||
      output == nullptr || rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
  const int64_t count = static_cast<int64_t>(rows) * cols;
  replace_rows_bf16_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(base),
      static_cast<const __nv_bfloat16*>(replacement),
      static_cast<const uint32_t*>(row_map),
      static_cast<__nv_bfloat16*>(output), rows, cols);
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

extern "C" cudaError_t apxinf_static_swiglu_bf16(
    const void* gate_up, void* output, int rows, int inner,
    cudaStream_t stream) {
  const int64_t count = static_cast<int64_t>(rows) * inner;
  if (gate_up == nullptr || output == nullptr || rows <= 0 || inner <= 0)
    return cudaErrorInvalidValue;
  if (swiglu_vec8_ok(gate_up, output, inner)) {
    swiglu_bf16_vec8_kernel<<<blocks_for(count / 8), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gate_up),
        static_cast<__nv_bfloat16*>(output), rows, inner);
    return cudaGetLastError();
  }
  swiglu_bf16_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(gate_up),
      static_cast<__nv_bfloat16*>(output), rows, inner);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_swiglu_f16(
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

extern "C" cudaError_t apxinf_static_swiglu_quant_f16_e4m3(
    const void* gate_up, const void* bias, void* output,
    int rows, int inner, float scale, cudaStream_t stream) {
  if (gate_up == nullptr || output == nullptr || rows <= 0 || inner <= 0 ||
      !std::isfinite(scale) || scale <= 0.0f)
    return cudaErrorInvalidValue;
  const int64_t count = static_cast<int64_t>(rows) * inner;
  const bool packed4 =
      inner % 4 == 0 &&
      reinterpret_cast<uintptr_t>(gate_up) % alignof(half2) == 0 &&
      (bias == nullptr ||
       reinterpret_cast<uintptr_t>(bias) % alignof(__nv_bfloat162) == 0) &&
      reinterpret_cast<uintptr_t>(output) % alignof(Fp8x4) == 0;
  if (packed4) {
    swiglu_quant_f16_e4m3_packed4_kernel
        <<<blocks_for(count / 4), kThreads, 0, stream>>>(
            static_cast<const half*>(gate_up),
            static_cast<const __nv_bfloat16*>(bias),
            static_cast<__nv_fp8_e4m3*>(output), rows, inner, 1.0f / scale);
  } else {
    swiglu_quant_f16_e4m3_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
        static_cast<const half*>(gate_up),
        static_cast<const __nv_bfloat16*>(bias),
        static_cast<__nv_fp8_e4m3*>(output), rows, inner, 1.0f / scale);
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

extern "C" cudaError_t apxinf_static_bias_residual_bf16_packed4(
    const void* projection, const void* bias, const void* residual,
    void* output, int rows, int cols, cudaStream_t stream) {
  if (projection == nullptr || bias == nullptr || residual == nullptr ||
      output == nullptr || rows <= 0 || cols <= 0 || (cols & 3) != 0)
    return cudaErrorInvalidValue;
  const uintptr_t pointers = reinterpret_cast<uintptr_t>(projection) |
      reinterpret_cast<uintptr_t>(residual) |
      reinterpret_cast<uintptr_t>(output) |
      reinterpret_cast<uintptr_t>(bias);
  if ((pointers & (alignof(Bf16x4) - 1)) != 0)
    return cudaErrorInvalidValue;
  const int64_t packed_count = static_cast<int64_t>(rows) * (cols / 4);
  bias_residual_bf16_packed4_kernel<<<
      blocks_for(packed_count), kThreads, 0, stream>>>(
      static_cast<const Bf16x4*>(projection),
      static_cast<const Bf16x4*>(bias),
      static_cast<const Bf16x4*>(residual),
      static_cast<Bf16x4*>(output), packed_count, cols / 4);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_then_residual_bf16_packed4(
    const void* projection, const void* bias, const void* residual,
    void* output, int rows, int cols, cudaStream_t stream) {
  if (projection == nullptr || residual == nullptr || output == nullptr ||
      rows <= 0 || cols <= 0 || (cols & 3) != 0)
    return cudaErrorInvalidValue;
  const uintptr_t pointers = reinterpret_cast<uintptr_t>(projection) |
      reinterpret_cast<uintptr_t>(residual) |
      reinterpret_cast<uintptr_t>(output) |
      reinterpret_cast<uintptr_t>(bias);
  if ((pointers & (alignof(Bf16x4) - 1)) != 0)
    return cudaErrorInvalidValue;
  const int64_t packed_count = static_cast<int64_t>(rows) * (cols / 4);
  bias_then_residual_bf16_packed4_kernel<<<
      blocks_for(packed_count), kThreads, 0, stream>>>(
      static_cast<const Bf16x4*>(projection),
      static_cast<const Bf16x4*>(bias),
      static_cast<const Bf16x4*>(residual),
      static_cast<Bf16x4*>(output), packed_count, cols / 4);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_then_residual_bf16(
    const void* projection, const void* bias, const void* residual,
    void* output, int rows, int cols, cudaStream_t stream) {
  const int64_t count = static_cast<int64_t>(rows) * cols;
  bias_then_residual_bf16_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(projection),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<const __nv_bfloat16*>(residual),
      static_cast<__nv_bfloat16*>(output), count, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_residual_f16_bf16(
    const void* projection, const void* bias, const void* residual,
    void* output, int rows, int cols, cudaStream_t stream) {
  if (projection == nullptr || residual == nullptr || output == nullptr ||
      rows <= 0 || cols <= 0)
    return cudaErrorInvalidValue;
  const int64_t count = static_cast<int64_t>(rows) * cols;
  bias_residual_f16_bf16_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
      static_cast<const half*>(projection),
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

extern "C" cudaError_t apxinf_static_rms_norm_f16(
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

extern "C" cudaError_t apxinf_static_rms_norm_quant_bf16_e4m3(
    const void* input, const void* weight, void* output,
    int rows, int cols, float eps, float scale, cudaStream_t stream) {
  if (input == nullptr || weight == nullptr || output == nullptr ||
      rows <= 0 || cols <= 0 || !std::isfinite(scale) || scale <= 0.0f)
    return cudaErrorInvalidValue;
  rms_norm_quant_bf16_e4m3_kernel<<<rows, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_fp8_e4m3*>(output), rows, cols, eps, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_layer_norm_bf16(
    const void* input, const void* weight, const void* bias, void* output,
    int rows, int cols, float eps, cudaStream_t stream) {
  // Cache the row in registers when it divides evenly into the block; the
  // three-pass kernel below covers everything else.
  if (cols == kThreads * 2 || cols == kThreads * 4 || cols == kThreads * 8) {
    const int per_thread = cols / kThreads;
    if (per_thread == 2) {
      layer_norm_bf16_cached_kernel<2><<<rows, kThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(input),
          static_cast<const __nv_bfloat16*>(weight),
          static_cast<const __nv_bfloat16*>(bias),
          static_cast<__nv_bfloat16*>(output), rows, cols, eps);
    } else if (per_thread == 4) {
      layer_norm_bf16_cached_kernel<4><<<rows, kThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(input),
          static_cast<const __nv_bfloat16*>(weight),
          static_cast<const __nv_bfloat16*>(bias),
          static_cast<__nv_bfloat16*>(output), rows, cols, eps);
    } else {
      layer_norm_bf16_cached_kernel<8><<<rows, kThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(input),
          static_cast<const __nv_bfloat16*>(weight),
          static_cast<const __nv_bfloat16*>(bias),
          static_cast<__nv_bfloat16*>(output), rows, cols, eps);
    }
    return cudaGetLastError();
  }
  layer_norm_bf16_kernel<<<rows, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<__nv_bfloat16*>(output), rows, cols, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_layer_norm_f16(
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

extern "C" cudaError_t apxinf_static_bias_residual_rms_norm_quant_f16_bf16_e4m3(
    const void* projection, const void* bias, const void* residual,
    const void* weight, void* hidden, void* normalized,
    int rows, int cols, float eps, float scale, cudaStream_t stream) {
  if (projection == nullptr || residual == nullptr || weight == nullptr ||
      hidden == nullptr || normalized == nullptr || rows <= 0 || cols <= 0 ||
      !std::isfinite(scale) || scale <= 0.0f)
    return cudaErrorInvalidValue;
  bias_residual_rms_norm_quant_f16_bf16_e4m3_kernel<<<
      rows, kThreads, 0, stream>>>(
      static_cast<const half*>(projection),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<const __nv_bfloat16*>(residual),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(hidden),
      static_cast<__nv_fp8_e4m3*>(normalized), rows, cols, eps, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_residual_layer_norm_bf16(
    const void* projection, const void* projection_bias,
    const void* residual, const void* norm_weight, const void* norm_bias,
    void* hidden, void* normalized, int rows, int cols, float eps,
    cudaStream_t stream) {
  // The 1024-column, no-projection-bias path may use the disjoint-buffer
  // load-ahead specialization. Aliased buffers use the generic implementation.
  if (cols == 1024 && projection_bias == nullptr) {
    if (layer_norm_1024_load_ahead_noalias(
            projection, residual, norm_weight, norm_bias,
            hidden, normalized, rows)) {
      bias_residual_layer_norm_bf16_carry_1024_load_ahead_kernel<<<
          rows, kThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(projection), nullptr,
          static_cast<const __nv_bfloat16*>(residual),
          static_cast<const __nv_bfloat16*>(norm_weight),
          static_cast<const __nv_bfloat16*>(norm_bias),
          static_cast<__nv_bfloat16*>(hidden),
          static_cast<__nv_bfloat16*>(normalized), rows, cols, eps);
      return cudaGetLastError();
    }
  }

  // Qwen-Drive vision uses 1024 columns and no projection bias. Retain the
  // original generic kernel for every other call shape or bias mode.
  if (cols == 1024 && projection_bias == nullptr) {
    bias_residual_layer_norm_bf16_carry_1024_kernel<<<rows, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(projection),
        nullptr,
        static_cast<const __nv_bfloat16*>(residual),
        static_cast<const __nv_bfloat16*>(norm_weight),
        static_cast<const __nv_bfloat16*>(norm_bias),
        static_cast<__nv_bfloat16*>(hidden),
        static_cast<__nv_bfloat16*>(normalized), rows, cols, eps);
    return cudaGetLastError();
  }
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

extern "C" cudaError_t apxinf_static_bias_residual_layer_norm_quant_bf16_e4m3(
    const void* projection, const void* projection_bias,
    const void* residual, const void* norm_weight, const void* norm_bias,
    void* hidden, void* normalized, int rows, int cols, float eps,
    float scale, cudaStream_t stream) {
  if (projection == nullptr || residual == nullptr || norm_weight == nullptr ||
      norm_bias == nullptr || hidden == nullptr || normalized == nullptr ||
      rows <= 0 || cols <= 0 || !std::isfinite(scale) || scale <= 0.0f)
    return cudaErrorInvalidValue;
  bias_residual_layer_norm_quant_bf16_e4m3_kernel<<<
      rows, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(projection),
      static_cast<const __nv_bfloat16*>(projection_bias),
      static_cast<const __nv_bfloat16*>(residual),
      static_cast<const __nv_bfloat16*>(norm_weight),
      static_cast<const __nv_bfloat16*>(norm_bias),
      static_cast<__nv_bfloat16*>(hidden),
      static_cast<__nv_fp8_e4m3*>(normalized), rows, cols, eps, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t
apxinf_static_bias_then_residual_adaptive_layer_norm_bf16_cached_1536(
    const void* projection, const void* projection_bias,
    const void* residual, const void* modulation,
    void* hidden, void* normalized, int rows, int cols, float eps,
    cudaStream_t stream) {
  if (projection == nullptr || projection_bias == nullptr ||
      residual == nullptr || modulation == nullptr || hidden == nullptr ||
      normalized == nullptr || rows <= 0 || cols != 1536)
    return cudaErrorInvalidValue;
  bias_then_residual_adaptive_layer_norm_bf16_cached_1536_kernel
      <<<rows, kThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(projection),
          static_cast<const __nv_bfloat16*>(projection_bias),
          static_cast<const __nv_bfloat16*>(residual),
          static_cast<const __nv_bfloat16*>(modulation),
          static_cast<__nv_bfloat16*>(hidden),
          static_cast<__nv_bfloat16*>(normalized), rows, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t
apxinf_static_bias_then_residual_layer_norm_bf16_cached_1536(
    const void* projection, const void* projection_bias,
    const void* residual, const void* norm_weight, const void* norm_bias,
    void* hidden, void* normalized, int rows, int cols, float eps,
    cudaStream_t stream) {
  if (projection == nullptr || projection_bias == nullptr ||
      residual == nullptr || norm_weight == nullptr || norm_bias == nullptr ||
      hidden == nullptr || normalized == nullptr || rows <= 0 || cols != 1536)
    return cudaErrorInvalidValue;
  bias_then_residual_layer_norm_bf16_cached_1536_kernel
      <<<rows, kThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(projection),
          static_cast<const __nv_bfloat16*>(projection_bias),
          static_cast<const __nv_bfloat16*>(residual),
          static_cast<const __nv_bfloat16*>(norm_weight),
          static_cast<const __nv_bfloat16*>(norm_bias),
          static_cast<__nv_bfloat16*>(hidden),
          static_cast<__nv_bfloat16*>(normalized), rows, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_gelu_bf16_packed8(
    const void* input, const void* bias, void* output,
    int rows, int cols, cudaStream_t stream) {
  if (input == nullptr || bias == nullptr || output == nullptr || rows <= 0 ||
      cols <= 0 || cols % 8 != 0 ||
      reinterpret_cast<uintptr_t>(input) % alignof(Bf16Pairx8) != 0 ||
      reinterpret_cast<uintptr_t>(bias) % alignof(Bf16Pairx8) != 0 ||
      reinterpret_cast<uintptr_t>(output) % alignof(Bf16Pairx8) != 0)
    return cudaErrorInvalidValue;
  const int64_t octet_count = static_cast<int64_t>(rows) * cols / 8;
  const int threads = rows >= 512 ? 256 : 128;
  const int blocks = static_cast<int>((octet_count + threads - 1) / threads);
  bias_gelu_bf16_packed8_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const Bf16Pairx8*>(input), static_cast<const Bf16Pairx8*>(bias),
      static_cast<Bf16Pairx8*>(output), octet_count, cols / 8);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_gelu_f16(
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

extern "C" cudaError_t apxinf_static_bias_residual_layer_norm_bf16_cached_1024(
    const void* projection, const void* projection_bias,
    const void* residual, const void* norm_weight, const void* norm_bias,
    void* hidden, void* normalized, int rows, int cols, float eps,
    cudaStream_t stream) {
  if (projection == nullptr || residual == nullptr || norm_weight == nullptr ||
      norm_bias == nullptr || hidden == nullptr || normalized == nullptr ||
      rows <= 0 || cols != 1024)
    return cudaErrorInvalidValue;
  bias_residual_layer_norm_bf16_cached_1024_kernel
      <<<rows, kThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(projection),
          static_cast<const __nv_bfloat16*>(projection_bias),
          static_cast<const __nv_bfloat16*>(residual),
          static_cast<const __nv_bfloat16*>(norm_weight),
          static_cast<const __nv_bfloat16*>(norm_bias),
          static_cast<__nv_bfloat16*>(hidden),
          static_cast<__nv_bfloat16*>(normalized), rows, eps);
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

extern "C" cudaError_t apxinf_static_gqa_qkv_mrope_cache_bf16(
    const void* qkv, const void* bias, const uint32_t* position_ids,
    void* q, void* k_cache, void* v_cache, int tokens,
    int q_heads, int kv_heads, int head_dim, float theta,
    int section_h, int section_w, int cache_offset,
    cudaStream_t stream) {
  if (qkv == nullptr || position_ids == nullptr || q == nullptr ||
      k_cache == nullptr || v_cache == nullptr || tokens <= 0 ||
      q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0 ||
      head_dim <= 0 || head_dim > 256 || head_dim % 2 != 0 ||
      !(theta > 0.0f) || section_h < 0 || section_w < 0 ||
      section_h + section_w > head_dim / 2 || cache_offset < 0) {
    return cudaErrorInvalidValue;
  }
  dim3 grid(tokens, q_heads + 2 * kv_heads, 1);
  gqa_qkv_mrope_cache_kernel<__nv_bfloat16>
      <<<grid, head_dim / 2, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(qkv),
      static_cast<const __nv_bfloat16*>(bias), position_ids,
      static_cast<__nv_bfloat16*>(q),
      static_cast<__nv_bfloat16*>(k_cache),
      static_cast<__nv_bfloat16*>(v_cache), tokens, q_heads, kv_heads,
      head_dim, theta, section_h, section_w, cache_offset);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_gqa_qkv_mrope_cache_f16(
    const void* qkv, const void* bias, const uint32_t* position_ids,
    void* q, void* k_cache, void* v_cache, int tokens,
    int q_heads, int kv_heads, int head_dim, float theta,
    int section_h, int section_w, int cache_offset,
    cudaStream_t stream) {
  if (qkv == nullptr || position_ids == nullptr || q == nullptr ||
      k_cache == nullptr || v_cache == nullptr || tokens <= 0 ||
      q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0 ||
      head_dim <= 0 || head_dim > 256 || head_dim % 2 != 0 ||
      !(theta > 0.0f) || section_h < 0 || section_w < 0 ||
      section_h + section_w > head_dim / 2 || cache_offset < 0) {
    return cudaErrorInvalidValue;
  }
  dim3 grid(tokens, q_heads + 2 * kv_heads, 1);
  gqa_qkv_mrope_cache_kernel<half><<<grid, head_dim / 2, 0, stream>>>(
      static_cast<const half*>(qkv),
      static_cast<const __nv_bfloat16*>(bias), position_ids,
      static_cast<__nv_bfloat16*>(q),
      static_cast<__nv_bfloat16*>(k_cache),
      static_cast<__nv_bfloat16*>(v_cache), tokens, q_heads, kv_heads,
      head_dim, theta, section_h, section_w, cache_offset);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_vision_qkv_rope_bf16(
    const void* qkv, const void* bias, const uint32_t* position_ids,
    void* q, void* k, void* v, int tokens, int heads, int head_dim,
    float theta, cudaStream_t stream) {
  if (qkv == nullptr || position_ids == nullptr || q == nullptr ||
      k == nullptr || v == nullptr || tokens <= 0 || heads <= 0 ||
      head_dim <= 0 || head_dim > 256 || head_dim % 4 != 0 ||
      !(theta > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  // Two axes of head_dim/4 rotations, held as sine then cosine.
  const size_t rope_smem = static_cast<size_t>(head_dim) * sizeof(float);
  vision_qkv_rope_kernel<__nv_bfloat16><<<tokens, kThreads, rope_smem, stream>>>(
      static_cast<const __nv_bfloat16*>(qkv),
      static_cast<const __nv_bfloat16*>(bias), position_ids,
      static_cast<__nv_bfloat16*>(q), static_cast<__nv_bfloat16*>(k),
      static_cast<__nv_bfloat16*>(v), tokens, heads, head_dim, theta);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_vision_qkv_rope_f16(
    const void* qkv, const void* bias, const uint32_t* position_ids,
    void* q, void* k, void* v, int tokens, int heads, int head_dim,
    float theta, cudaStream_t stream) {
  if (qkv == nullptr || position_ids == nullptr || q == nullptr ||
      k == nullptr || v == nullptr || tokens <= 0 || heads <= 0 ||
      head_dim <= 0 || head_dim > 256 || head_dim % 4 != 0 ||
      !(theta > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  const size_t rope_smem = static_cast<size_t>(head_dim) * sizeof(float);
  vision_qkv_rope_kernel<half><<<tokens, kThreads, rope_smem, stream>>>(
      static_cast<const half*>(qkv),
      static_cast<const __nv_bfloat16*>(bias), position_ids,
      static_cast<__nv_bfloat16*>(q), static_cast<__nv_bfloat16*>(k),
      static_cast<__nv_bfloat16*>(v), tokens, heads, head_dim, theta);
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

extern "C" cudaError_t apxinf_static_segmented_mha_bf16(
    const void* q, const void* k, const void* v, const void* offsets,
    void* output, int segments, int max_tokens, int heads, int head_dim,
    cudaStream_t stream) {
  if (q == nullptr || k == nullptr || v == nullptr || offsets == nullptr ||
      output == nullptr || segments <= 0 || max_tokens <= 0 || heads <= 0 ||
      head_dim <= 0 || head_dim > kThreads) {
    return cudaErrorInvalidValue;
  }
  dim3 grid(max_tokens, heads, segments);
  const size_t shared = static_cast<size_t>(max_tokens + 8) * sizeof(float);
  segmented_mha_bf16_kernel<<<grid, kThreads, shared, stream>>>(
      static_cast<const __nv_bfloat16*>(q),
      static_cast<const __nv_bfloat16*>(k),
      static_cast<const __nv_bfloat16*>(v),
      static_cast<const uint32_t*>(offsets),
      static_cast<__nv_bfloat16*>(output), heads, head_dim);
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

// FP32 projection + bias + learned position embedding, rounded to BF16 for the
// vision encoder. Mirrors PaliGemma's `embeddings` block, which runs in FP32
// and only then casts into the BF16 transformer stack.
extern "C" cudaError_t apxinf_static_bias_position_f32_bf16(
    const void* projection, const void* bias, const void* position,
    void* output, int rows, int cols, int tokens_per_view,
    cudaStream_t stream) {
  const int64_t count = static_cast<int64_t>(rows) * cols;
  bias_position_f32_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
      static_cast<const float*>(projection),
      static_cast<const float*>(bias),
      static_cast<const float*>(position),
      static_cast<__nv_bfloat16*>(output), count, cols, tokens_per_view);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_vision_qk_rope_bf16_no_v(
    const void* qkv, const uint32_t* position_ids,
    void* q, void* k, int tokens, int heads, int head_dim,
    float theta, cudaStream_t stream) {
  if (qkv == nullptr || position_ids == nullptr || q == nullptr ||
      k == nullptr || tokens <= 0 || heads <= 0 ||
      head_dim <= 0 || head_dim > 256 || head_dim % 4 != 0 ||
      !(theta > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  const size_t rope_smem = static_cast<size_t>(head_dim) * sizeof(float);
  vision_qkv_rope_kernel<__nv_bfloat16, true><<<tokens, kThreads, rope_smem, stream>>>(
      static_cast<const __nv_bfloat16*>(qkv), nullptr, position_ids,
      static_cast<__nv_bfloat16*>(q), static_cast<__nv_bfloat16*>(k),
      nullptr, tokens, heads, head_dim, theta);
  return cudaGetLastError();
}
