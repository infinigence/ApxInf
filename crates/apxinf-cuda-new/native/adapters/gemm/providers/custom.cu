#include "vendor.h"
#include "../../../kernels/custom/gemm.cuh"

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

  const bool needs_postprocess =
      spec.semantic != APXINF_GEMM_SEMANTIC_GEMM ||
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
  const bool needs_postprocess =
      spec.semantic != APXINF_GEMM_SEMANTIC_GEMM ||
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
  const int64_t output_width =
      spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_GEGLU ? spec.n / 2 : spec.n;
  const int64_t count = spec.m * output_width;
  const int blocks = static_cast<int>(
      std::min<int64_t>((count + 255) / 256, 4096));
  apxinf::cuda::custom::finish<<<blocks, 256, 0,
                                 static_cast<cudaStream_t>(bindings.stream)>>>(
      projection, resources.projection_dtype, bindings.output,
      spec.output_dtype,
      bindings.bias,
      has_row_channel_scales(spec)
          ? spec.output_dtype
          : resources.projection_dtype,
      bindings.a_scales, bindings.b_scales, spec.m, spec.n,
      static_cast<int>(spec.semantic),
      has_row_channel_scales(spec) ? 1 : 0,
      has_row_channel_scales(spec) ? bindings.alpha : 1.0F,
      bindings.output_scale);
  return cudaGetLastError();
}

}  // namespace apxinf::gemm::vendor
