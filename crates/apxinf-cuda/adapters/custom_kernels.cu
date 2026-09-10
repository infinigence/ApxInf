// Copyright 2026 apxinf contributors.
// Stable C ABI and CUDA launch adapter for custom static-inference operators.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

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
#include "../kernels/custom/linear_attention.cuh"
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
  if (input == nullptr || output == nullptr || count <= 0 || !(scale > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 4096 ? 4096 : blocks;
  dequantize_e4m3_f16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_fp8_e4m3*>(input),
      static_cast<half*>(output), count, scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_quantize_bf16_e4m3(
    const void* input, void* output, int64_t count, float scale,
    cudaStream_t stream) {
  if (input == nullptr || output == nullptr || count <= 0 || !(scale > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 4096 ? 4096 : blocks;
  quantize_bf16_e4m3_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_fp8_e4m3*>(output), count, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_dynamic_quantize_rows_bf16_e4m3(
    const void* input, void* output, void* scales, int rows,
    int input_cols, int output_cols, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || scales == nullptr ||
      rows <= 0 || input_cols <= 0 || output_cols < input_cols) {
    return cudaErrorInvalidValue;
  }
  constexpr int threads = 256;
  if (input_cols % 8 == 0 && output_cols % 8 == 0) {
    constexpr int rows_per_block = threads / 32;
    const int blocks = (rows + rows_per_block - 1) / rows_per_block;
    quantize_rows_bf16_e4m3_vec8_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<__nv_fp8_e4m3*>(output), static_cast<float*>(scales),
        rows, input_cols, output_cols);
  } else {
    quantize_rows_bf16_e4m3_kernel<<<rows, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<__nv_fp8_e4m3*>(output), static_cast<float*>(scales),
        rows, input_cols, output_cols);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_dynamic_rms_norm_quantize_rows_bf16_e4m3(
    const void* input, const void* weight, void* output, void* scales,
    int rows, int input_cols, int output_cols, float eps,
    cudaStream_t stream) {
  if (input == nullptr || weight == nullptr || output == nullptr ||
      scales == nullptr || rows <= 0 || input_cols <= 0 ||
      output_cols < input_cols || !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  constexpr int threads = 256;
  constexpr int rows_per_block = threads / 32;
  const int blocks = (rows + rows_per_block - 1) / rows_per_block;
  if (input_cols % 8 == 0 && output_cols % 8 == 0) {
    rms_norm_quantize_rows_bf16_e4m3_vec8_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_fp8_e4m3*>(output), static_cast<float*>(scales),
        rows, input_cols, output_cols, eps);
  } else {
    rms_norm_quantize_rows_bf16_e4m3_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_fp8_e4m3*>(output), static_cast<float*>(scales),
        rows, input_cols, output_cols, eps);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_dynamic_swiglu_quantize_rows_bf16_e4m3(
    const void* gate_up, const void* bias, void* output, void* scales,
    int rows, int input_cols, int inner, int output_cols,
    cudaStream_t stream) {
  if (gate_up == nullptr || output == nullptr || scales == nullptr ||
      rows <= 0 || inner <= 0 || input_cols < 2 * inner ||
      output_cols < inner) {
    return cudaErrorInvalidValue;
  }
  constexpr int threads = 256;
  const size_t shared_bytes = static_cast<size_t>(inner) * sizeof(float);
  if (input_cols % 8 == 0 && inner % 8 == 0 && output_cols % 8 == 0) {
    swiglu_quantize_rows_bf16_e4m3_vec8_kernel
        <<<rows, threads, shared_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(gate_up),
        static_cast<const __nv_bfloat16*>(bias),
        static_cast<__nv_fp8_e4m3*>(output), static_cast<float*>(scales),
        rows, input_cols, inner, output_cols);
  } else if (input_cols % 4 == 0 && inner % 4 == 0 &&
             output_cols % 4 == 0) {
    swiglu_quantize_rows_bf16_e4m3_vec4_kernel
        <<<rows, threads, shared_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(gate_up),
        static_cast<const __nv_bfloat16*>(bias),
        static_cast<__nv_fp8_e4m3*>(output), static_cast<float*>(scales),
        rows, input_cols, inner, output_cols);
  } else {
    swiglu_quantize_rows_bf16_e4m3_kernel
        <<<rows, threads, shared_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(gate_up),
        static_cast<const __nv_bfloat16*>(bias),
        static_cast<__nv_fp8_e4m3*>(output), static_cast<float*>(scales),
        rows, input_cols, inner, output_cols);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t
apxinf_dynamic_bias_residual_rms_norm_quantize_rows_bf16_e4m3(
    const void* projection, const void* bias, const void* residual,
    const void* weight, void* hidden, void* normalized, void* scales,
    int rows, int cols, int output_cols, float eps, cudaStream_t stream) {
  if (projection == nullptr || residual == nullptr || weight == nullptr ||
      hidden == nullptr || normalized == nullptr || scales == nullptr ||
      rows <= 0 || cols <= 0 || output_cols < cols || !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  constexpr int threads = 256;
  if (rows <= 64) {
    constexpr int rows_per_block = threads / 32;
    const int blocks = (rows + rows_per_block - 1) / rows_per_block;
    if (cols % 8 == 0 && output_cols % 8 == 0) {
      bias_residual_rms_norm_quantize_rows_bf16_e4m3_vec8_kernel
          <<<blocks, threads, 0, stream>>>(
              static_cast<const __nv_bfloat16*>(projection),
              static_cast<const __nv_bfloat16*>(bias),
              static_cast<const __nv_bfloat16*>(residual),
              static_cast<const __nv_bfloat16*>(weight),
              static_cast<__nv_bfloat16*>(hidden),
              static_cast<__nv_fp8_e4m3*>(normalized),
              static_cast<float*>(scales), rows, cols, output_cols, eps);
    } else {
      bias_residual_rms_norm_quantize_rows_bf16_e4m3_kernel
          <<<blocks, threads, 0, stream>>>(
              static_cast<const __nv_bfloat16*>(projection),
              static_cast<const __nv_bfloat16*>(bias),
              static_cast<const __nv_bfloat16*>(residual),
              static_cast<const __nv_bfloat16*>(weight),
              static_cast<__nv_bfloat16*>(hidden),
              static_cast<__nv_fp8_e4m3*>(normalized),
              static_cast<float*>(scales), rows, cols, output_cols, eps);
    }
  } else {
    if (cols % 8 == 0 && output_cols % 8 == 0) {
      bias_residual_rms_norm_quantize_rows_bf16_e4m3_large_vec8_kernel
          <<<rows, threads, 0, stream>>>(
              static_cast<const __nv_bfloat16*>(projection),
              static_cast<const __nv_bfloat16*>(bias),
              static_cast<const __nv_bfloat16*>(residual),
              static_cast<const __nv_bfloat16*>(weight),
              static_cast<__nv_bfloat16*>(hidden),
              static_cast<__nv_fp8_e4m3*>(normalized),
              static_cast<float*>(scales), rows, cols, output_cols, eps);
    } else {
      bias_residual_rms_norm_quantize_rows_bf16_e4m3_large_kernel
          <<<rows, threads, 0, stream>>>(
              static_cast<const __nv_bfloat16*>(projection),
              static_cast<const __nv_bfloat16*>(bias),
              static_cast<const __nv_bfloat16*>(residual),
              static_cast<const __nv_bfloat16*>(weight),
              static_cast<__nv_bfloat16*>(hidden),
              static_cast<__nv_fp8_e4m3*>(normalized),
              static_cast<float*>(scales), rows, cols, output_cols, eps);
    }
  }
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_slice_columns_bf16(
    const void* input, void* output, int rows, int input_cols,
    int output_cols, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || rows <= 0 ||
      output_cols <= 0 || output_cols > input_cols) {
    return cudaErrorInvalidValue;
  }
  const int64_t count = static_cast<int64_t>(rows) * output_cols;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 4096 ? 4096 : blocks;
  slice_columns_bf16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_bfloat16*>(output), rows, input_cols, output_cols);
  return cudaGetLastError();
}

// FIX (implement_r8): block-per-row tiled softmax for the route-3 composed
// vision attention: fp32 scores in, bf16 probabilities out (round-to-nearest,
// matching the cast kernels). The stock thread-per-element softmax is
// structurally unviable at the [16*3472, 3472] vision-segment geometry.
__global__ void row_softmax_f32_bf16_kernel(
    const float* input, __nv_bfloat16* output, uint32_t cols, uint32_t rows) {
  const uint32_t row = blockIdx.x;
  if (row >= rows) return;
  const float* x = input + static_cast<size_t>(row) * cols;
  __shared__ float scratch[32];
  float local = -INFINITY;
  for (uint32_t i = threadIdx.x; i < cols; i += blockDim.x) {
    local = fmaxf(local, x[i]);
  }
  const float max_val = block_max(local, scratch);
  float partial = 0.0f;
  for (uint32_t i = threadIdx.x; i < cols; i += blockDim.x) {
    partial += expf(x[i] - max_val);
  }
  const float sum = block_sum(partial, scratch);
  __nv_bfloat16* y = output + static_cast<size_t>(row) * cols;
  for (uint32_t i = threadIdx.x; i < cols; i += blockDim.x) {
    y[i] = __float2bfloat16(expf(x[i] - max_val) / sum);
  }
}

extern "C" cudaError_t apxinf_static_row_softmax_f32_bf16(
    const void* input, void* output, uint32_t cols, uint32_t rows,
    cudaStream_t stream) {
  if (input == nullptr || output == nullptr || cols == 0 || rows == 0) {
    return cudaErrorInvalidValue;
  }
  row_softmax_f32_bf16_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const float*>(input), static_cast<__nv_bfloat16*>(output),
      cols, rows);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_cast_f16_bf16(
    const void* input, void* output, int64_t count, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || count <= 0)
    return cudaErrorInvalidValue;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 4096 ? 4096 : blocks;
  cast_f16_bf16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const half*>(input),
      static_cast<__nv_bfloat16*>(output), count);
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

// ── linear_attention.cuh launch adapters ───────────────────────────────────

extern "C" cudaError_t apxinf_static_cast_f32_bf16(
    const void* input, void* output, int64_t count, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || count <= 0)
    return cudaErrorInvalidValue;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 4096 ? 4096 : blocks;
  cast_f32_to_bf16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const float*>(input),
      static_cast<__nv_bfloat16*>(output), count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_cast_bf16_f32(
    const void* input, void* output, int64_t count, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || count <= 0)
    return cudaErrorInvalidValue;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 4096 ? 4096 : blocks;
  cast_bf16_to_f32_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<float*>(output), count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_causal_conv1d_silu_bf16(
    const void* x, const void* weight, const void* state, void* out,
    void* new_state, int channels, int seq, int kernel_size,
    int64_t x_row_stride, cudaStream_t stream) {
  if (x == nullptr || weight == nullptr || out == nullptr ||
      new_state == nullptr || channels <= 0 || seq <= 0 || kernel_size <= 0 ||
      kernel_size > 8 || x_row_stride < channels) {
    return cudaErrorInvalidValue;
  }
  const int channel_blocks = (channels + 255) / 256;
  causal_conv1d_silu_bf16_kernel<<<dim3(seq, channel_blocks), 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<const __nv_bfloat16*>(state),
      static_cast<__nv_bfloat16*>(out),
      static_cast<__nv_bfloat16*>(new_state),
      channels, seq, kernel_size, x_row_stride);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_gdn_qk_prep_bf16(
    const void* conv_out, void* q_out, void* k_out,
    int seq, int seq_pad, int conv_dim, int key_dim,
    int num_v_heads, int head_k_dim, float scale, float eps,
    cudaStream_t stream) {
  if (conv_out == nullptr || q_out == nullptr || k_out == nullptr ||
      seq <= 0 || seq_pad < seq || conv_dim < 2 * key_dim || key_dim <= 0 ||
      num_v_heads <= 0 || head_k_dim <= 0 ||
      (head_k_dim & (head_k_dim - 1)) != 0 || !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  if (key_dim % head_k_dim != 0) return cudaErrorInvalidValue;
  const int num_k_heads = key_dim / head_k_dim;
  if (num_v_heads % num_k_heads != 0) return cudaErrorInvalidValue;
  const size_t smem = static_cast<size_t>(2 * head_k_dim) * sizeof(float);
  gdn_qk_prep_kernel<<<dim3(seq, num_k_heads), head_k_dim, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(conv_out),
      static_cast<float*>(q_out), static_cast<float*>(k_out),
      seq, seq_pad, conv_dim, key_dim, num_v_heads, head_k_dim, scale, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_gdn_vb_prep_bf16(
    const void* conv_out, const void* b_proj, const void* a_proj,
    const void* dt_bias, const void* a_log,
    void* v_out, void* beta_out, void* g_out,
    int seq, int seq_pad, int conv_dim, int v_offset,
    int num_v_heads, int ba_row_stride, int head_v_dim,
    cudaStream_t stream) {
  if (conv_out == nullptr || b_proj == nullptr || a_proj == nullptr ||
      dt_bias == nullptr || a_log == nullptr || v_out == nullptr ||
      beta_out == nullptr || g_out == nullptr || seq <= 0 || seq_pad < seq ||
      num_v_heads <= 0 || head_v_dim <= 0 || ba_row_stride < num_v_heads) {
    return cudaErrorInvalidValue;
  }
  gdn_vb_prep_kernel<<<dim3(seq, num_v_heads), head_v_dim, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(conv_out),
      static_cast<const __nv_bfloat16*>(b_proj),
      static_cast<const __nv_bfloat16*>(a_proj),
      static_cast<const float*>(dt_bias), static_cast<const float*>(a_log),
      static_cast<float*>(v_out), static_cast<float*>(beta_out),
      static_cast<float*>(g_out),
      seq, seq_pad, conv_dim, v_offset, ba_row_stride, head_v_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_gdn_cumsum_f32(
    const void* g, void* g_cum, int seq_pad, int num_v_heads, int chunk_size,
    cudaStream_t stream) {
  if (g == nullptr || g_cum == nullptr || seq_pad <= 0 ||
      num_v_heads <= 0 || chunk_size <= 0 || seq_pad % chunk_size != 0) {
    return cudaErrorInvalidValue;
  }
  const int chunks = seq_pad / chunk_size;
  gdn_cumsum_kernel<<<dim3(chunks, num_v_heads), 32, 0, stream>>>(
      static_cast<const float*>(g), static_cast<float*>(g_cum),
      seq_pad, chunk_size);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_gdn_attn_raw_f32(
    const void* q, const void* k, const void* beta, const void* g_cum,
    void* a_out, void* t_out, int seq_pad, int num_v_heads, int head_k_dim,
    int chunk_size, cudaStream_t stream) {
  if (q == nullptr || k == nullptr || beta == nullptr || g_cum == nullptr ||
      a_out == nullptr || t_out == nullptr || seq_pad <= 0 ||
      num_v_heads <= 0 || head_k_dim <= 0 || chunk_size <= 0 ||
      seq_pad % chunk_size != 0) {
    return cudaErrorInvalidValue;
  }
  const int chunks = seq_pad / chunk_size;
  gdn_attn_raw_kernel<<<dim3(chunks, num_v_heads), 256, 0, stream>>>(
      static_cast<const float*>(q), static_cast<const float*>(k),
      static_cast<const float*>(beta), static_cast<const float*>(g_cum),
      static_cast<float*>(a_out), static_cast<float*>(t_out),
      seq_pad, head_k_dim, chunk_size);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_gdn_tri_solve_f32(
    void* a, int matrices, int chunk_size, cudaStream_t stream) {
  if (a == nullptr || matrices <= 0 || chunk_size <= 0 || chunk_size > 128) {
    return cudaErrorInvalidValue;
  }
  const size_t smem =
      (static_cast<size_t>(chunk_size) * chunk_size + chunk_size) * sizeof(float);
  gdn_tri_solve_kernel<<<matrices, 64, smem, stream>>>(
      static_cast<float*>(a), chunk_size);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_gdn_chunk_gemm_f32(
    const void* a, const void* v, const void* k, const void* beta,
    const void* g_cum, void* vt_out, void* kcd_out,
    int seq_pad, int num_v_heads, int head_k_dim, int head_v_dim,
    int chunk_size, cudaStream_t stream) {
  if (a == nullptr || v == nullptr || k == nullptr || beta == nullptr ||
      g_cum == nullptr || vt_out == nullptr || kcd_out == nullptr ||
      seq_pad <= 0 || num_v_heads <= 0 || head_k_dim <= 0 ||
      head_v_dim <= 0 || chunk_size <= 0 || seq_pad % chunk_size != 0) {
    return cudaErrorInvalidValue;
  }
  const int chunks = seq_pad / chunk_size;
  gdn_chunk_gemm_kernel<<<dim3(chunks, num_v_heads), 256, 0, stream>>>(
      static_cast<const float*>(a), static_cast<const float*>(v),
      static_cast<const float*>(k), static_cast<const float*>(beta),
      static_cast<const float*>(g_cum),
      static_cast<float*>(vt_out), static_cast<float*>(kcd_out),
      seq_pad, head_k_dim, head_v_dim, chunk_size);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_gdn_chunk_state_f32(
    const void* q, const void* k, const void* g_cum, const void* t_in,
    const void* vt_in, const void* kcd_in, void* state, void* out,
    int seq, int seq_pad, int num_v_heads, int head_k_dim, int head_v_dim,
    int chunk_size, int total_chunks, int out_row_width,
    cudaStream_t stream) {
  if (q == nullptr || k == nullptr || g_cum == nullptr || t_in == nullptr ||
      vt_in == nullptr || kcd_in == nullptr || state == nullptr ||
      out == nullptr || seq <= 0 || seq_pad < seq || num_v_heads <= 0 ||
      head_k_dim <= 0 || head_v_dim <= 0 || chunk_size <= 0 ||
      total_chunks <= 0 || seq_pad != total_chunks * chunk_size ||
      out_row_width != num_v_heads * head_v_dim) {
    return cudaErrorInvalidValue;
  }
  const size_t smem =
      static_cast<size_t>(chunk_size) * head_v_dim * sizeof(float);
  gdn_chunk_state_kernel<<<num_v_heads, 256, smem, stream>>>(
      static_cast<const float*>(q), static_cast<const float*>(k),
      static_cast<const float*>(g_cum), static_cast<const float*>(t_in),
      static_cast<const float*>(vt_in), static_cast<const float*>(kcd_in),
      static_cast<float*>(state), static_cast<__nv_bfloat16*>(out),
      seq, seq_pad, head_k_dim, head_v_dim, chunk_size, total_chunks,
      out_row_width);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_gdn_recurrent_f32(
    const void* q, const void* k, const void* v, const void* beta,
    const void* g, void* state, void* out,
    int num_v_heads, int head_k_dim, int head_v_dim, cudaStream_t stream) {
  if (q == nullptr || k == nullptr || v == nullptr || beta == nullptr ||
      g == nullptr || state == nullptr || out == nullptr ||
      num_v_heads <= 0 || head_k_dim <= 0 || head_v_dim <= 0) {
    return cudaErrorInvalidValue;
  }
  const size_t smem = static_cast<size_t>(2 * head_k_dim) * sizeof(float);
  gdn_recurrent_kernel<<<num_v_heads, head_v_dim, smem, stream>>>(
      static_cast<const float*>(q), static_cast<const float*>(k),
      static_cast<const float*>(v), static_cast<const float*>(beta),
      static_cast<const float*>(g), static_cast<float*>(state),
      static_cast<__nv_bfloat16*>(out), head_k_dim, head_v_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_gated_rms_silu_bf16(
    const void* x, const void* z, const void* weight, void* out,
    int rows, int cols, int z_heads, int64_t z_row_stride,
    int64_t z_col_offset, float eps, cudaStream_t stream) {
  if (x == nullptr || z == nullptr || weight == nullptr || out == nullptr ||
      rows <= 0 || cols <= 0 || z_heads <= 0 || rows % z_heads != 0 ||
      !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  const size_t smem = static_cast<size_t>(cols) * sizeof(float);
  gated_rms_silu_bf16_kernel<<<rows, 128, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(x),
      static_cast<const __nv_bfloat16*>(z),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(out),
      cols, z_heads, z_row_stride, z_col_offset, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_rms_norm_plus1_bf16(
    const void* input, const void* weight, void* output,
    int rows, int cols, float eps, cudaStream_t stream) {
  if (input == nullptr || weight == nullptr || output == nullptr ||
      rows <= 0 || cols <= 0 || !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  const size_t smem = static_cast<size_t>(cols) * sizeof(float);
  rms_norm_plus1_bf16_kernel<<<rows, 256, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(output), cols, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_partial_rope_table_bf16(
    const void* x, const void* cos, const void* sin, void* out,
    int rows, int heads, int head_dim, int rotary_dim, cudaStream_t stream) {
  if (x == nullptr || cos == nullptr || sin == nullptr || out == nullptr ||
      rows <= 0 || heads <= 0 || head_dim <= 0 || rotary_dim <= 0 ||
      (rotary_dim & 1) != 0 || rotary_dim > head_dim) {
    return cudaErrorInvalidValue;
  }
  partial_rope_table_bf16_kernel<<<dim3(rows, heads), 32, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x),
      static_cast<const __nv_bfloat16*>(cos),
      static_cast<const __nv_bfloat16*>(sin),
      static_cast<__nv_bfloat16*>(out), heads, head_dim, rotary_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_full_attn_prepare_bf16(
    const void* fused, const void* q_norm_w, const void* k_norm_w,
    const void* cos, const void* sin, void* q_out, void* k_cache, void* v_cache,
    int seq, int cache_offset, int q_heads, int kv_heads, int head_dim,
    int rotary_dim, int64_t fused_width, int64_t cache_width, float eps,
    cudaStream_t stream) {
  if (fused == nullptr || q_norm_w == nullptr || k_norm_w == nullptr ||
      cos == nullptr || sin == nullptr || q_out == nullptr ||
      k_cache == nullptr || v_cache == nullptr || seq <= 0 ||
      cache_offset < 0 || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
      rotary_dim <= 0 || (rotary_dim & 1) != 0 || rotary_dim > head_dim ||
      !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  const size_t smem = static_cast<size_t>(head_dim) * sizeof(float);
  full_attn_prepare_bf16_kernel
      <<<dim3(seq, q_heads + 2 * kv_heads), 128, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(fused),
      static_cast<const __nv_bfloat16*>(q_norm_w),
      static_cast<const __nv_bfloat16*>(k_norm_w),
      static_cast<const __nv_bfloat16*>(cos),
      static_cast<const __nv_bfloat16*>(sin),
      static_cast<__nv_bfloat16*>(q_out),
      static_cast<__nv_bfloat16*>(k_cache),
      static_cast<__nv_bfloat16*>(v_cache),
      cache_offset, q_heads, kv_heads, head_dim, rotary_dim, fused_width,
      cache_width, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_sigmoid_gate_mul_bf16(
    void* attn, const void* fused, int rows, int heads, int head_dim,
    int64_t fused_width, cudaStream_t stream) {
  if (attn == nullptr || fused == nullptr || rows <= 0 || heads <= 0 ||
      head_dim <= 0) {
    return cudaErrorInvalidValue;
  }
  sigmoid_gate_mul_bf16_kernel<<<rows, 256, 0, stream>>>(
      static_cast<__nv_bfloat16*>(attn),
      static_cast<const __nv_bfloat16*>(fused), heads, head_dim, fused_width);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_adaln_rms_norm_bf16(
    const void* x, const void* weight, const void* scale, const void* shift,
    void* out, int rows, int cols, float eps, cudaStream_t stream) {
  if (x == nullptr || weight == nullptr || scale == nullptr ||
      shift == nullptr || out == nullptr || rows <= 0 || cols <= 0 ||
      !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  const size_t smem = static_cast<size_t>(cols) * sizeof(float);
  adaln_rms_norm_bf16_kernel<<<rows, 256, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(x),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<const __nv_bfloat16*>(scale),
      static_cast<const __nv_bfloat16*>(shift),
      static_cast<__nv_bfloat16*>(out), cols, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_adaln_gate_residual_bf16(
    const void* proj, const void* residual, const void* gate, void* out,
    int64_t count, int cols, cudaStream_t stream) {
  if (proj == nullptr || residual == nullptr || gate == nullptr ||
      out == nullptr || count <= 0 || cols <= 0) {
    return cudaErrorInvalidValue;
  }
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  adaln_gate_residual_bf16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(proj),
      static_cast<const __nv_bfloat16*>(residual),
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<__nv_bfloat16*>(out), count, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_expert_qkv_prepare_bf16(
    const void* fused, const void* q_norm_w, const void* k_norm_w,
    const void* cos, const void* sin, void* q_out, void* gate_out, void* k_out,
    void* v_out, int seq, int q_heads, int kv_heads, int head_dim,
    int rotary_dim, int64_t fused_width, float eps, cudaStream_t stream) {
  if (fused == nullptr || q_norm_w == nullptr || k_norm_w == nullptr ||
      cos == nullptr || sin == nullptr || q_out == nullptr ||
      gate_out == nullptr || k_out == nullptr || v_out == nullptr ||
      seq <= 0 || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
      rotary_dim <= 0 || (rotary_dim & 1) != 0 || rotary_dim > head_dim ||
      q_heads % kv_heads != 0 || !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  const size_t smem = static_cast<size_t>(head_dim) * sizeof(float);
  expert_qkv_prepare_bf16_kernel
      <<<dim3(seq, q_heads + 2 * kv_heads), 128, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(fused),
      static_cast<const __nv_bfloat16*>(q_norm_w),
      static_cast<const __nv_bfloat16*>(k_norm_w),
      static_cast<const __nv_bfloat16*>(cos),
      static_cast<const __nv_bfloat16*>(sin),
      static_cast<__nv_bfloat16*>(q_out),
      static_cast<__nv_bfloat16*>(gate_out),
      static_cast<__nv_bfloat16*>(k_out),
      static_cast<__nv_bfloat16*>(v_out),
      q_heads, kv_heads, head_dim, rotary_dim, fused_width, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_expert_sigmoid_gate_mul_bf16(
    void* attn, const void* gate, int64_t count, cudaStream_t stream) {
  if (attn == nullptr || gate == nullptr || count <= 0) {
    return cudaErrorInvalidValue;
  }
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  expert_sigmoid_gate_mul_bf16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<__nv_bfloat16*>(attn),
      static_cast<const __nv_bfloat16*>(gate), count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_fourier_features_bf16(
    const void* waypoints, const void* freqs, void* out,
    int rows, int point_dim, int num_features, cudaStream_t stream) {
  if (waypoints == nullptr || freqs == nullptr || out == nullptr ||
      rows <= 0 || point_dim <= 0 || num_features <= 0) {
    return cudaErrorInvalidValue;
  }
  fourier_features_bf16_kernel<<<rows, 64, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(waypoints),
      static_cast<const __nv_bfloat16*>(freqs),
      static_cast<__nv_bfloat16*>(out), point_dim, num_features);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_concat7_cols_bf16(
    const void* s0, const void* s1, const void* s2, const void* s3,
    const void* s4, const void* s5, const void* s6, void* dst,
    int rows, int cols, int broadcast_mask, cudaStream_t stream) {
  if (s0 == nullptr || s1 == nullptr || s2 == nullptr || s3 == nullptr ||
      s4 == nullptr || s5 == nullptr || s6 == nullptr || dst == nullptr ||
      rows <= 0 || cols <= 0) {
    return cudaErrorInvalidValue;
  }
  concat7_cols_bf16_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(s0),
      static_cast<const __nv_bfloat16*>(s1),
      static_cast<const __nv_bfloat16*>(s2),
      static_cast<const __nv_bfloat16*>(s3),
      static_cast<const __nv_bfloat16*>(s4),
      static_cast<const __nv_bfloat16*>(s5),
      static_cast<const __nv_bfloat16*>(s6),
      static_cast<__nv_bfloat16*>(dst), rows, cols, broadcast_mask);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_flow_update_f32(
    void* w, const void* endpoint, float remaining, float step, int64_t count,
    cudaStream_t stream) {
  if (w == nullptr || endpoint == nullptr || count <= 0 ||
      !(remaining > 0.0f) || !std::isfinite(step)) {
    return cudaErrorInvalidValue;
  }
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  flow_update_f32_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<float*>(w), static_cast<const float*>(endpoint),
      remaining, step, count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_suppress_logits_bf16(
    void* logits, const uint32_t* ids, int count, cudaStream_t stream) {
  if (logits == nullptr || ids == nullptr || count <= 0) {
    return cudaErrorInvalidValue;
  }
  const int blocks = (count + 63) / 64;
  suppress_logits_bf16_kernel<<<blocks, 64, 0, stream>>>(
      static_cast<__nv_bfloat16*>(logits), ids, count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_gelu_exact_bf16(
    const void* input, void* output, int64_t count, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || count <= 0) {
    return cudaErrorInvalidValue;
  }
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  gelu_exact_bf16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_bfloat16*>(output), count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_merge_rows_bf16(
    const void* input, void* output, int64_t out_count, int cols, int factor,
    cudaStream_t stream) {
  if (input == nullptr || output == nullptr || out_count <= 0 ||
      cols <= 0 || factor <= 0) {
    return cudaErrorInvalidValue;
  }
  int blocks = static_cast<int>((out_count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  merge_rows_bf16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_bfloat16*>(output), out_count, cols, factor);
  return cudaGetLastError();
}
