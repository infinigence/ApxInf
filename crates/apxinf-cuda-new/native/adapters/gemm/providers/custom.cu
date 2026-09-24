#include "vendor.h"
#include "../../../kernels/custom/gemm.cuh"

#include <cstdint>
#include <cuda_fp8.h>

namespace {
#include "../../../kernels/primitives/math.cuh"
#include "../../../kernels/primitives/activation.cuh"

int elementwise_blocks_for(int64_t count) {
  return static_cast<int>((count + kThreads - 1) / kThreads);
}

cudaError_t launch_bf16_geglu(
    const void* projection, void* output, int rows, int inner,
    cudaStream_t stream) {
  const int64_t count = static_cast<int64_t>(rows) * inner;
  const bool packed4 =
      inner % 4 == 0 &&
      reinterpret_cast<uintptr_t>(projection) % alignof(Bf16x4) == 0 &&
      reinterpret_cast<uintptr_t>(output) % alignof(Bf16x4) == 0;
  const bool packed2 =
      inner % 2 == 0 &&
      reinterpret_cast<uintptr_t>(projection) % alignof(__nv_bfloat162) == 0 &&
      reinterpret_cast<uintptr_t>(output) % alignof(__nv_bfloat162) == 0;
  if (packed4) {
    geglu_bf16_packed4_kernel<<<elementwise_blocks_for(count / 4), kThreads, 0,
                                stream>>>(
        static_cast<const __nv_bfloat16*>(projection),
        static_cast<__nv_bfloat16*>(output), rows, inner);
  } else if (packed2) {
    geglu_bf16_packed2_kernel<<<elementwise_blocks_for(count / 2), kThreads, 0,
                                stream>>>(
        static_cast<const __nv_bfloat16*>(projection),
        static_cast<__nv_bfloat16*>(output), rows, inner);
  } else {
    geglu_bf16_kernel<<<elementwise_blocks_for(count), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(projection),
        static_cast<__nv_bfloat16*>(output), rows, inner);
  }
  return cudaGetLastError();
}
}  // namespace

namespace apxinf::gemm::vendor {

uint32_t common_projection_dtype(const Spec& spec) {
  return spec.a_dtype == APXINF_DTYPE_I8
             ? APXINF_DTYPE_I32
             : has_row_channel_scales(spec)
                   ? APXINF_DTYPE_F32
                   : (spec.a_dtype == APXINF_DTYPE_E4M3 ||
                      spec.b_dtype == APXINF_DTYPE_E4M3)
                         ? (spec.output_dtype == APXINF_DTYPE_F16
                                ? APXINF_DTYPE_F16
                                : APXINF_DTYPE_F32)
                         : spec.a_dtype;
}

CommonResources::~CommonResources() {
  release();
}

void CommonResources::release() noexcept {
  if (projection != nullptr) cudaFree(projection);
  if (unpack_a != nullptr) cudaFree(unpack_a);
  if (unpack_b != nullptr) cudaFree(unpack_b);
  projection = nullptr;
  unpack_a = nullptr;
  unpack_b = nullptr;
}

void allocate_common_resources(const Spec& spec,
                               CommonResources& resources,
                               bool native_fp8) {
  const bool unpack =
      ((spec.a_dtype == APXINF_DTYPE_E4M3 ||
        spec.b_dtype == APXINF_DTYPE_E4M3) &&
       !native_fp8) ||
      (has_row_channel_scales(spec) && !native_fp8 &&
       spec.a_dtype != APXINF_DTYPE_I8);

  resources.projection_dtype = common_projection_dtype(spec);

  if (unpack) {
    const size_t a_bytes =
        spec.m * spec.k * dtype_bytes(resources.projection_dtype);
    const size_t b_bytes =
        spec.k * spec.n * dtype_bytes(resources.projection_dtype);
    check_cuda(cudaMalloc(&resources.unpack_a, a_bytes));
    check_cuda(cudaMalloc(&resources.unpack_b, b_bytes));
    resources.resource_bytes += a_bytes + b_bytes;
  }

  const bool native_bias =
      native_fp8 &&
      (spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_BIAS ||
       spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_BIAS_RESIDUAL) &&
      spec.quantization == APXINF_GEMM_QUANT_FP8_UNIT_SCALE &&
      spec.output_scale_is_unit != 0;
  const bool needs_postprocess =
      (spec.semantic != APXINF_GEMM_SEMANTIC_GEMM && !native_bias) ||
      has_row_channel_scales(spec) ||
      spec.output_dtype != resources.projection_dtype ||
      spec.output_scale_is_unit == 0;
  if (needs_postprocess) {
    const size_t bytes =
        spec.m * spec.n * dtype_bytes(resources.projection_dtype);
    check_cuda(cudaMalloc(&resources.projection, bytes));
    resources.resource_bytes += bytes;
  }
}

size_t common_resource_requirements(const Spec& spec, bool native_fp8) {
  const bool unpack =
      ((spec.a_dtype == APXINF_DTYPE_E4M3 ||
        spec.b_dtype == APXINF_DTYPE_E4M3) &&
       !native_fp8) ||
      (has_row_channel_scales(spec) && !native_fp8 &&
       spec.a_dtype != APXINF_DTYPE_I8);
  const uint32_t projection_dtype = common_projection_dtype(spec);
  size_t bytes = 0;
  if (unpack) {
    bytes += static_cast<size_t>(spec.m * spec.k) *
             dtype_bytes(projection_dtype);
    bytes += static_cast<size_t>(spec.k * spec.n) *
             dtype_bytes(projection_dtype);
  }
  const bool native_bias =
      native_fp8 &&
      (spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_BIAS ||
       spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_BIAS_RESIDUAL) &&
      spec.quantization == APXINF_GEMM_QUANT_FP8_UNIT_SCALE &&
      spec.output_scale_is_unit != 0;
  const bool needs_postprocess =
      (spec.semantic != APXINF_GEMM_SEMANTIC_GEMM && !native_bias) ||
      has_row_channel_scales(spec) || spec.output_dtype != projection_dtype ||
      spec.output_scale_is_unit == 0;
  if (needs_postprocess) {
    bytes += static_cast<size_t>(spec.m * spec.n) *
             dtype_bytes(projection_dtype);
  }
  return bytes;
}

cudaError_t launch_postprocess(const Spec& spec,
                               CommonResources& resources,
                               const apxinf_gemm_bindings_t& bindings,
                               void* projection) {
  if (resources.projection == nullptr) {
    return cudaSuccess;
  }
  if (spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_GEGLU &&
      spec.quantization == APXINF_GEMM_QUANT_NONE &&
      resources.projection_dtype == APXINF_DTYPE_BF16 &&
      spec.output_dtype == APXINF_DTYPE_BF16 && spec.alpha_is_unit != 0 &&
      spec.output_scale_is_unit != 0 && bindings.bias == nullptr) {
    return launch_bf16_geglu(
        projection, bindings.output, static_cast<int>(spec.m),
        static_cast<int>(spec.n / 2),
        static_cast<cudaStream_t>(bindings.stream));
  }
  const int64_t output_width =
      spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_GEGLU ? spec.n / 2 : spec.n;
  const int64_t count = spec.m * output_width;
  const int blocks = static_cast<int>(
      std::min<int64_t>((count + 255) / 256, 4096));
  const uint32_t bias_dtype =
      spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_BIAS_GELU &&
              spec.a_dtype == APXINF_DTYPE_E4M3 &&
              spec.output_dtype == APXINF_DTYPE_E4M3
          ? APXINF_DTYPE_F16
          : has_row_channel_scales(spec) ? spec.output_dtype
                                         : resources.projection_dtype;
  apxinf::cuda_new::custom::finish<<<blocks, 256, 0,
                                 static_cast<cudaStream_t>(bindings.stream)>>>(
      projection, resources.projection_dtype, bindings.output,
      spec.output_dtype,
      bindings.bias,
      bias_dtype,
      bindings.residual, resources.projection_dtype,
      bindings.a_scales, bindings.b_scales, spec.m, spec.n,
      static_cast<int>(spec.semantic),
      has_row_channel_scales(spec) ? 1 : 0,
      has_row_channel_scales(spec) ? bindings.alpha : 1.0F,
      bindings.output_scale);
  return cudaGetLastError();
}

}  // namespace apxinf::gemm::vendor
