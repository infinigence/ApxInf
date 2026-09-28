#include "../include/apxinf_cuda/pointwise.h"
#include "../framework/runtime_internal.h"
#include "../kernels/custom/pointwise.cuh"

#include <climits>
#include <cmath>
#include <cstdint>
#include <cuda_fp8.h>

namespace {
#include "../kernels/primitives/math.cuh"
#include "../kernels/primitives/activation.cuh"

using apxinf::framework::Failure;
namespace kernels = apxinf::pointwise::kernels;

constexpr int kDirectThreads = 256;
constexpr int kMaxBlocks = 4096;

bool valid_alignment(uint32_t alignment) {
  return alignment <= 256 && alignment != 0 &&
         (alignment & (alignment - 1)) == 0;
}

bool reads_secondary(uint32_t semantic) {
  return semantic == APXINF_POINTWISE_SEMANTIC_EULER_UPDATE;
}

bool may_have_bias(uint32_t semantic) {
  return semantic == APXINF_POINTWISE_SEMANTIC_BIAS_ACTIVATION;
}

void validate_spec(const apxinf_pointwise_spec_t& spec) {
  const bool quantized_geglu =
      spec.semantic == APXINF_POINTWISE_SEMANTIC_GEGLU &&
      spec.dtype == APXINF_DTYPE_F16 &&
      spec.output_dtype == APXINF_DTYPE_E4M3;
  if (spec.version != APXINF_POINTWISE_SPEC_VERSION ||
      spec.semantic > APXINF_POINTWISE_SEMANTIC_EULER_UPDATE ||
      (spec.dtype != APXINF_DTYPE_F16 && spec.dtype != APXINF_DTYPE_BF16) ||
      (!quantized_geglu && spec.output_dtype != spec.dtype) ||
      spec.activation > APXINF_POINTWISE_ACTIVATION_SILU ||
      spec.has_bias > 1 || spec.rows <= 0 || spec.cols <= 0 ||
      spec.rows > INT32_MAX || spec.cols > INT32_MAX ||
      spec.output_scale_is_unit > 1 ||
      !valid_alignment(spec.input_alignment) ||
      !valid_alignment(spec.secondary_alignment) ||
      !valid_alignment(spec.bias_alignment) ||
      !valid_alignment(spec.output_alignment)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid Pointwise Spec");
  }
  if ((!quantized_geglu && spec.output_scale_is_unit != 1) ||
      (quantized_geglu &&
       (spec.cols % 2 != 0 || spec.input_alignment < 4 ||
        spec.output_alignment < 2 || spec.rows * spec.cols > INT32_MAX))) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "invalid Pointwise quantized GeGLU contract");
  }
  if (spec.has_bias != 0 && !may_have_bias(spec.semantic)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Pointwise semantic does not take a bias");
  }
  if (spec.activation != APXINF_POINTWISE_ACTIVATION_NONE &&
      spec.semantic != APXINF_POINTWISE_SEMANTIC_BIAS_ACTIVATION) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Pointwise semantic does not take an activation");
  }
  if (spec.semantic == APXINF_POINTWISE_SEMANTIC_BIAS_ACTIVATION &&
      spec.has_bias == 0 &&
      spec.activation == APXINF_POINTWISE_ACTIVATION_NONE) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Pointwise bias-activation is a no-op");
  }
}

void validate_bindings(const apxinf_pointwise_spec_t& spec,
                       const apxinf_pointwise_bindings_t& bindings) {
  if (bindings.input == nullptr || bindings.output == nullptr ||
      (reads_secondary(spec.semantic) && bindings.secondary == nullptr)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "missing Pointwise binding for semantic");
  }
  if ((spec.has_bias != 0) != (bindings.bias != nullptr)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Pointwise bias binding disagrees with Spec.has_bias");
  }
  if (!std::isfinite(bindings.output_scale) || bindings.output_scale <= 0.0f ||
      ((bindings.output_scale == 1.0f) !=
       (spec.output_scale_is_unit != 0)) ||
      !std::isfinite(bindings.dt)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid Pointwise scalar");
  }
  if (spec.semantic == APXINF_POINTWISE_SEMANTIC_GEGLU &&
      spec.dtype == APXINF_DTYPE_F16 &&
      spec.output_dtype == APXINF_DTYPE_E4M3) {
    const auto input = reinterpret_cast<uintptr_t>(bindings.input);
    const auto output = reinterpret_cast<uintptr_t>(bindings.output);
    const size_t count = static_cast<size_t>(spec.rows) * spec.cols;
    const size_t input_bytes = count * 2 * sizeof(half);
    const size_t output_bytes = count * sizeof(__nv_fp8_e4m3);
    const bool range_overflow = input_bytes > UINTPTR_MAX - input ||
                                output_bytes > UINTPTR_MAX - output;
    const bool overlap = !range_overflow && input < output + output_bytes &&
                         output < input + input_bytes;
    if (input % 4 != 0 || output % 2 != 0 || range_overflow || overlap) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid F16-to-E4M3 GeGLU storage");
    }
  }
}

int blocks_for(int64_t count) {
  const int64_t blocks = (count + kDirectThreads - 1) / kDirectThreads;
  if (blocks < 1) return 1;
  return static_cast<int>(blocks < kMaxBlocks ? blocks : kMaxBlocks);
}

int elementwise_blocks_for(int64_t count) {
  return static_cast<int>((count + kThreads - 1) / kThreads);
}

cudaError_t launch_f16_e4m3_geglu(
    const half* input, __nv_fp8_e4m3* output, int rows, int cols,
    float output_scale, cudaStream_t stream) {
  const float inverse_scale = 1.0f / output_scale;
  const bool packed8 = cols % 8 == 0 &&
      reinterpret_cast<uintptr_t>(input) % 8 == 0 &&
      reinterpret_cast<uintptr_t>(output) % 8 == 0;
  if (packed8) {
    const int64_t groups = static_cast<int64_t>(rows) * (cols / 8);
    geglu_quant_f16_e4m3_packed8_kernel<<<
        blocks_for(groups), kDirectThreads, 0, stream>>>(
        input, output, rows, cols, inverse_scale);
    return cudaGetLastError();
  }
  const bool packed4 = cols % 4 == 0 &&
      reinterpret_cast<uintptr_t>(input) % 4 == 0 &&
      reinterpret_cast<uintptr_t>(output) % 4 == 0;
  if (packed4) {
    const int64_t groups = static_cast<int64_t>(rows) * (cols / 4);
    geglu_quant_f16_e4m3_packed4_kernel<<<
        blocks_for(groups), kDirectThreads, 0, stream>>>(
        input, output, rows, cols, inverse_scale);
    return cudaGetLastError();
  }
  const int64_t pairs = static_cast<int64_t>(rows) * (cols / 2);
  geglu_quant_f16_e4m3_kernel<<<
      blocks_for(pairs), kDirectThreads, 0, stream>>>(
      input, output, rows, cols, inverse_scale);
  return cudaGetLastError();
}

cudaError_t launch_bf16_geglu(const __nv_bfloat16* input,
                                     __nv_bfloat16* output, int rows, int cols,
                                     cudaStream_t stream) {
  const int64_t count = static_cast<int64_t>(rows) * cols;
  const bool packed4 =
      cols % 4 == 0 &&
      reinterpret_cast<uintptr_t>(input) % alignof(Bf16x4) == 0 &&
      reinterpret_cast<uintptr_t>(output) % alignof(Bf16x4) == 0;
  const bool packed2 =
      cols % 2 == 0 &&
      reinterpret_cast<uintptr_t>(input) % alignof(__nv_bfloat162) == 0 &&
      reinterpret_cast<uintptr_t>(output) % alignof(__nv_bfloat162) == 0;
  if (packed4) {
    geglu_bf16_packed4_kernel<<<elementwise_blocks_for(count / 4), kThreads, 0,
                                stream>>>(input, output, rows, cols);
  } else if (packed2) {
    geglu_bf16_packed2_kernel<<<elementwise_blocks_for(count / 2), kThreads, 0,
                                stream>>>(input, output, rows, cols);
  } else {
    geglu_bf16_kernel<<<elementwise_blocks_for(count), kThreads, 0, stream>>>(
        input, output, rows, cols);
  }
  return cudaGetLastError();
}

cudaError_t launch_bf16_bias_activation(
    const __nv_bfloat16* input, const __nv_bfloat16* bias,
    __nv_bfloat16* output, int rows, int cols, int activation,
    cudaStream_t stream) {
  const int64_t count = static_cast<int64_t>(rows) * cols;
  const bool packed4 =
      cols % 4 == 0 &&
      reinterpret_cast<uintptr_t>(input) % alignof(Bf16x4) == 0 &&
      reinterpret_cast<uintptr_t>(output) % alignof(Bf16x4) == 0 &&
      (bias == nullptr ||
       reinterpret_cast<uintptr_t>(bias) % alignof(Bf16x4) == 0);
  const bool packed2 =
      cols % 2 == 0 &&
      reinterpret_cast<uintptr_t>(input) % alignof(__nv_bfloat162) == 0 &&
      reinterpret_cast<uintptr_t>(output) % alignof(__nv_bfloat162) == 0 &&
      (bias == nullptr ||
       reinterpret_cast<uintptr_t>(bias) % alignof(__nv_bfloat162) == 0);
  if (packed4) {
    bias_activation_bf16_packed4_kernel<<<
        elementwise_blocks_for(count / 4), kThreads, 0, stream>>>(
        input, bias, output, count / 4, cols, activation);
  } else if (packed2) {
    bias_activation_bf16_packed2_kernel<<<
        elementwise_blocks_for(count / 2), kThreads, 0, stream>>>(
        input, bias, output, count / 2, cols, activation);
  } else {
    bias_activation_bf16_kernel<<<elementwise_blocks_for(count), kThreads, 0,
                                  stream>>>(
        input, bias, output, count, cols, activation);
  }
  return cudaGetLastError();
}

template <class T>
cudaError_t launch(const apxinf_pointwise_spec_t& spec,
                   const apxinf_pointwise_bindings_t& bindings) {
  const auto* input = static_cast<const T*>(bindings.input);
  const auto* secondary = static_cast<const T*>(bindings.secondary);
  const auto* bias = static_cast<const T*>(bindings.bias);
  auto* output = static_cast<T*>(bindings.output);
  const int rows = static_cast<int>(spec.rows);
  const int cols = static_cast<int>(spec.cols);
  const int64_t count = spec.rows * spec.cols;
  auto stream = static_cast<cudaStream_t>(bindings.stream);
  const int grid = blocks_for(count);

  switch (spec.semantic) {
    case APXINF_POINTWISE_SEMANTIC_GEGLU:
      kernels::geglu<T><<<grid, kDirectThreads, 0, stream>>>(
          input, output, rows, cols);
      break;
    case APXINF_POINTWISE_SEMANTIC_BIAS_ACTIVATION:
      kernels::bias_activation<T><<<grid, kDirectThreads, 0, stream>>>(
          input, bias, output, count, cols, static_cast<int>(spec.activation));
      break;
    case APXINF_POINTWISE_SEMANTIC_EULER_UPDATE:
      kernels::euler_update<T><<<grid, kDirectThreads, 0, stream>>>(
          input, secondary, output, count, bindings.dt);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

cudaError_t launch_bf16(const apxinf_pointwise_spec_t& spec,
                        const apxinf_pointwise_bindings_t& bindings) {
  const auto* input = static_cast<const __nv_bfloat16*>(bindings.input);
  const auto* bias = static_cast<const __nv_bfloat16*>(bindings.bias);
  auto* output = static_cast<__nv_bfloat16*>(bindings.output);
  auto stream = static_cast<cudaStream_t>(bindings.stream);
  const int rows = static_cast<int>(spec.rows);
  const int cols = static_cast<int>(spec.cols);
  if (spec.semantic == APXINF_POINTWISE_SEMANTIC_GEGLU) {
    return launch_bf16_geglu(input, output, rows, cols, stream);
  }
  if (spec.semantic == APXINF_POINTWISE_SEMANTIC_BIAS_ACTIVATION) {
    return launch_bf16_bias_activation(
        input, bias, output, rows, cols, static_cast<int>(spec.activation),
        stream);
  }
  return launch<__nv_bfloat16>(spec, bindings);
}

}  // namespace

extern "C" apxinf_status_t apxinf_pointwise_launch(
    apxinf_runtime_t runtime, const apxinf_pointwise_spec_t* spec,
    const apxinf_pointwise_bindings_t* bindings) {
  return apxinf::framework::abi_boundary([&] {
    if (runtime == nullptr || spec == nullptr || bindings == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "null Pointwise argument");
    }
    validate_spec(*spec);
    validate_bindings(*spec, *bindings);
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    if (spec->semantic == APXINF_POINTWISE_SEMANTIC_GEGLU &&
        spec->dtype == APXINF_DTYPE_F16 &&
        spec->output_dtype == APXINF_DTYPE_E4M3) {
      apxinf::framework::check_cuda(launch_f16_e4m3_geglu(
          static_cast<const half*>(bindings->input),
          static_cast<__nv_fp8_e4m3*>(bindings->output),
          static_cast<int>(spec->rows), static_cast<int>(spec->cols),
          bindings->output_scale,
          static_cast<cudaStream_t>(bindings->stream)));
      return;
    }
    apxinf::framework::check_cuda(
        spec->dtype == APXINF_DTYPE_BF16
            ? launch_bf16(*spec, *bindings)
            : launch<__half>(*spec, *bindings));
  });
}
