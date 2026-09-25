#include "vendor.h"
#include "../../../kernels/custom/gemm.cuh"

#include <cstdlib>

namespace apxinf::gemm::vendor {
namespace {

// Escape hatch for the pre-change behaviour, kept so the two projection
// dtypes can be compared inside one binary.
bool fp8_f32_projection_forced() {
  static const bool forced =
      std::getenv("APXINF_GEMM_FP8_F32_PROJECTION") != nullptr;
  return forced;
}

}  // namespace

uint32_t common_projection_dtype(const Spec& spec) {
  if (spec.a_dtype == APXINF_DTYPE_I8) {
    return APXINF_DTYPE_I32;
  }
  if (has_row_channel_scales(spec)) {
    return APXINF_DTYPE_F32;
  }
  if (spec.a_dtype != APXINF_DTYPE_E4M3 && spec.b_dtype != APXINF_DTYPE_E4M3) {
    return spec.a_dtype;
  }
  if (spec.output_dtype == APXINF_DTYPE_F16) {
    return APXINF_DTYPE_F16;
  }
  // A 16-bit output needs no wider intermediate. Every vendor FP8 path already
  // accumulates in F32 and converts in its own epilogue, so projecting through
  // F32 buys no precision and costs a full-size write, read and convert pass
  // per call -- plus, at prefill widths, an unpack budget large enough to
  // prefilter every candidate that is not native FP8.
  if (spec.output_dtype == APXINF_DTYPE_BF16 && !fp8_f32_projection_forced()) {
    return APXINF_DTYPE_BF16;
  }
  return APXINF_DTYPE_F32;
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
