#include "../include/apxinf_cuda/quantization.h"
#include "../framework/runtime_internal.h"

#include <climits>
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <string>

namespace {

#include "../kernels/primitives/math.cuh"
#include "../kernels/primitives/reduction.cuh"
#include "../kernels/primitives/quantization.cuh"

using apxinf::framework::Failure;

cudaError_t launch_quantize_f16_e4m3(
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

cudaError_t launch_quantize_bf16_e4m3(
    const void* input, void* output, int64_t count, float scale,
    cudaStream_t stream) {
  if (input == nullptr || output == nullptr || count <= 0 || !(scale > 0.0f))
    return cudaErrorInvalidValue;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 4096 ? 4096 : blocks;
  quantize_bf16_e4m3_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_fp8_e4m3*>(output), count, 1.0f / scale);
  return cudaGetLastError();
}

cudaError_t launch_quantize_rows_bf16_e4m3(
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

cudaError_t launch_cast_f16_bf16(
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

cudaError_t launch_slice_columns_bf16(
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

cudaError_t launch_quantize_rows_bf16_int8(
    const void* input, void* output, void* scales,
    int rows, int cols, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || scales == nullptr ||
      rows <= 0 || cols <= 0) {
    return cudaErrorInvalidValue;
  }
  quantize_rows_bf16_int8_kernel<<<rows, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<int8_t*>(output), static_cast<float*>(scales), rows, cols);
  return cudaGetLastError();
}

bool valid_alignment(uint32_t alignment) {
  return alignment <= 256 && alignment != 0 &&
         (alignment & (alignment - 1)) == 0;
}

bool has_row_scales(uint32_t semantic) {
  return semantic == APXINF_QUANTIZATION_SEMANTIC_ROWWISE_E4M3 ||
         semantic == APXINF_QUANTIZATION_SEMANTIC_ROWWISE_I8;
}

uint32_t dtype_bytes(uint32_t dtype) {
  switch (dtype) {
    case APXINF_DTYPE_F32:
      return 4;
    case APXINF_DTYPE_F16:
    case APXINF_DTYPE_BF16:
      return 2;
    case APXINF_DTYPE_E4M3:
    case APXINF_DTYPE_I8:
      return 1;
    default:
      return 0;
  }
}

void validate_recorded_alignment(const void* pointer, uint32_t alignment,
                                 const char* name) {
  if (pointer != nullptr &&
      reinterpret_cast<uintptr_t>(pointer) % alignment != 0) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  std::string(name) + " violates its alignment class");
  }
}

void validate_spec(const apxinf_quantization_spec_t& spec) {
  if (spec.version != APXINF_QUANTIZATION_SPEC_VERSION ||
      spec.semantic > APXINF_QUANTIZATION_SEMANTIC_ROWWISE_I8 ||
      spec.scale_dtype != APXINF_DTYPE_F32 || spec.rows <= 0 ||
      spec.input_cols <= 0 || spec.output_cols <= 0 ||
      spec.rows > INT32_MAX || spec.input_cols > INT32_MAX ||
      spec.output_cols > INT32_MAX ||
      !valid_alignment(spec.input_alignment) ||
      !valid_alignment(spec.output_alignment) ||
      !valid_alignment(spec.scales_alignment)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "invalid Quantization Spec");
  }
  if (spec.input_alignment < dtype_bytes(spec.input_dtype) ||
      spec.output_alignment < dtype_bytes(spec.output_dtype) ||
      (has_row_scales(spec.semantic) &&
       spec.scales_alignment < sizeof(float))) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Quantization binding violates dtype alignment");
  }
  switch (spec.semantic) {
    case APXINF_QUANTIZATION_SEMANTIC_FIXED_E4M3:
      if ((spec.input_dtype != APXINF_DTYPE_F16 &&
           spec.input_dtype != APXINF_DTYPE_BF16) ||
          spec.output_dtype != APXINF_DTYPE_E4M3 ||
          spec.input_cols != spec.output_cols) {
        throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                      "invalid fixed-scale E4M3 Spec");
      }
      break;
    case APXINF_QUANTIZATION_SEMANTIC_ROWWISE_E4M3:
      if (spec.input_dtype != APXINF_DTYPE_BF16 ||
          spec.output_dtype != APXINF_DTYPE_E4M3 ||
          spec.output_cols < spec.input_cols) {
        throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                      "invalid rowwise E4M3 Spec");
      }
      break;
    case APXINF_QUANTIZATION_SEMANTIC_CAST_F16_BF16:
      if (spec.input_dtype != APXINF_DTYPE_F16 ||
          spec.output_dtype != APXINF_DTYPE_BF16 ||
          spec.input_cols != spec.output_cols) {
        throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                      "invalid F16-to-BF16 cast Spec");
      }
      break;
    case APXINF_QUANTIZATION_SEMANTIC_SLICE_BF16:
      if (spec.input_dtype != APXINF_DTYPE_BF16 ||
          spec.output_dtype != APXINF_DTYPE_BF16 ||
          spec.output_cols > spec.input_cols) {
        throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                      "invalid BF16 slice Spec");
      }
      break;
    case APXINF_QUANTIZATION_SEMANTIC_ROWWISE_I8:
      if (spec.input_dtype != APXINF_DTYPE_BF16 ||
          spec.output_dtype != APXINF_DTYPE_I8 ||
          spec.input_cols != spec.output_cols) {
        throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                      "invalid rowwise INT8 Spec");
      }
      break;
    default:
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "unknown Quantization semantic");
  }
}

void validate_bindings(const apxinf_quantization_spec_t& spec,
                       const apxinf_quantization_bindings_t& bindings) {
  if (bindings.input == nullptr || bindings.output == nullptr) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "missing Quantization binding");
  }
  if (has_row_scales(spec.semantic) != (bindings.scales != nullptr)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Quantization scales binding disagrees with semantic");
  }
  if (spec.semantic == APXINF_QUANTIZATION_SEMANTIC_FIXED_E4M3) {
    if (!(bindings.scale > 0.0f) || !std::isfinite(bindings.scale)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "fixed E4M3 scale must be finite and positive");
    }
  } else if (bindings.scale != 1.0f) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "unused Quantization scale must be one");
  }
  validate_recorded_alignment(bindings.input, spec.input_alignment,
                              "Quantization input");
  validate_recorded_alignment(bindings.output, spec.output_alignment,
                              "Quantization output");
  validate_recorded_alignment(bindings.scales, spec.scales_alignment,
                              "Quantization scales");
}

cudaError_t launch(const apxinf_quantization_spec_t& spec,
                   const apxinf_quantization_bindings_t& bindings) {
  auto stream = static_cast<cudaStream_t>(bindings.stream);
  const int rows = static_cast<int>(spec.rows);
  const int input_cols = static_cast<int>(spec.input_cols);
  const int output_cols = static_cast<int>(spec.output_cols);
  const int64_t input_count = spec.rows * spec.input_cols;

  switch (spec.semantic) {
    case APXINF_QUANTIZATION_SEMANTIC_FIXED_E4M3:
      return spec.input_dtype == APXINF_DTYPE_F16
                 ? launch_quantize_f16_e4m3(
                       bindings.input, bindings.output, input_count,
                       bindings.scale, stream)
                 : launch_quantize_bf16_e4m3(
                       bindings.input, bindings.output, input_count,
                       bindings.scale, stream);
    case APXINF_QUANTIZATION_SEMANTIC_ROWWISE_E4M3:
      return launch_quantize_rows_bf16_e4m3(
          bindings.input, bindings.output, bindings.scales, rows, input_cols,
          output_cols, stream);
    case APXINF_QUANTIZATION_SEMANTIC_CAST_F16_BF16:
      return launch_cast_f16_bf16(
          bindings.input, bindings.output, input_count, stream);
    case APXINF_QUANTIZATION_SEMANTIC_SLICE_BF16:
      return launch_slice_columns_bf16(bindings.input, bindings.output, rows,
                                       input_cols, output_cols, stream);
    case APXINF_QUANTIZATION_SEMANTIC_ROWWISE_I8:
      return launch_quantize_rows_bf16_int8(
          bindings.input, bindings.output, bindings.scales, rows, input_cols,
          stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace

extern "C" apxinf_status_t apxinf_quantization_launch(
    apxinf_runtime_t runtime, const apxinf_quantization_spec_t* spec,
    const apxinf_quantization_bindings_t* bindings) {
  return apxinf::framework::abi_boundary([&] {
    if (runtime == nullptr || spec == nullptr || bindings == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "null Quantization argument");
    }
    validate_spec(*spec);
    validate_bindings(*spec, *bindings);
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    apxinf::framework::check_cuda(launch(*spec, *bindings));
  });
}
