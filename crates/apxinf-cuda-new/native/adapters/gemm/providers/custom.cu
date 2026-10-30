#include "vendor.h"
#include "../../../kernels/custom/gemm.cuh"

#include <cstdlib>
#include <cstdint>
#include <cuda_fp8.h>

extern "C" cudaError_t apxinf_activation_swiglu_bf16_rounded_pair_interleaved(
    const void* gate_up, void* output, int rows, int inner, int pair,
    cudaStream_t stream);

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
namespace {

// Escape hatch for the pre-change behaviour, kept so the two projection
// dtypes can be compared inside one binary.
bool fp8_f32_projection_forced() {
  static const bool forced =
      std::getenv("APXINF_GEMM_FP8_F32_PROJECTION") != nullptr;
  return forced;
}

// Dtype of caller-owned fused operands. This mirrors the Rust L3 contract's
// `projection_dtype` calculation; it is deliberately separate from
// `common_projection_dtype`, which describes an internal provider buffer and
// may be widened for final-rounding reasons.
uint32_t binding_projection_dtype(const Spec& spec) {
  if (has_row_channel_scales(spec)) return spec.output_dtype;
  if (spec.a_dtype == APXINF_DTYPE_E4M3) {
    return spec.output_dtype == APXINF_DTYPE_F16 ? APXINF_DTYPE_F16
                                                 : APXINF_DTYPE_F32;
  }
  return spec.a_dtype;
}

}  // namespace

uint32_t common_projection_dtype(const Spec& spec) {
  if (spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_BIAS &&
      spec.a_dtype == APXINF_DTYPE_BF16 &&
      spec.b_dtype == APXINF_DTYPE_BF16 &&
      spec.output_dtype == APXINF_DTYPE_BF16 &&
      spec.quantization == APXINF_GEMM_QUANT_NONE) {
    // The cuBLAS fallback has no fused bias epilogue. Keep its accumulator in
    // F32 until finish() adds the BF16 bias, matching the L3 final-rounding
    // contract instead of rounding the GEMM result before bias.
    return APXINF_DTYPE_F32;
  }
  if (spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_GEGLU &&
      spec.quantization == APXINF_GEMM_QUANT_FP8_UNIT_SCALE &&
      spec.output_dtype == APXINF_DTYPE_E4M3) {
    return APXINF_DTYPE_F16;
  }
  if (spec.a_dtype == APXINF_DTYPE_I8) {
    return APXINF_DTYPE_I32;
  }
  if (has_row_channel_scales(spec)) {
    // cuBLASLt native-FP8 matmul supports half-precision outputs, not an F32
    // projection. Accumulation remains F32 inside the GEMM; finish() applies
    // row/channel scales in F32 before writing the requested 16-bit result.
    if (spec.output_dtype == APXINF_DTYPE_F16 ||
        spec.output_dtype == APXINF_DTYPE_BF16) {
      return spec.output_dtype;
    }
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
                               bool native_fp8,
                               bool native_bias_epilogue) {
  const bool unpack =
      ((spec.a_dtype == APXINF_DTYPE_E4M3 ||
        spec.b_dtype == APXINF_DTYPE_E4M3) &&
       !native_fp8) ||
      (has_row_channel_scales(spec) && !native_fp8 &&
       spec.a_dtype != APXINF_DTYPE_I8);

  resources.projection_dtype = native_bias_epilogue
                                   ? spec.output_dtype
                                   : common_projection_dtype(spec);

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
      native_bias_epilogue ||
      (native_fp8 &&
       (spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_BIAS ||
        spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_BIAS_RESIDUAL) &&
       spec.quantization == APXINF_GEMM_QUANT_FP8_UNIT_SCALE &&
       spec.output_scale_is_unit != 0);
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

size_t common_resource_requirements(const Spec& spec,
                                    bool native_fp8,
                                    bool native_bias_epilogue) {
  const bool unpack =
      ((spec.a_dtype == APXINF_DTYPE_E4M3 ||
        spec.b_dtype == APXINF_DTYPE_E4M3) &&
       !native_fp8) ||
      (has_row_channel_scales(spec) && !native_fp8 &&
       spec.a_dtype != APXINF_DTYPE_I8);
  const uint32_t projection_dtype =
      native_bias_epilogue ? spec.output_dtype : common_projection_dtype(spec);
  size_t bytes = 0;
  if (unpack) {
    bytes += static_cast<size_t>(spec.m * spec.k) *
             dtype_bytes(projection_dtype);
    bytes += static_cast<size_t>(spec.k * spec.n) *
             dtype_bytes(projection_dtype);
  }
  const bool native_bias =
      native_bias_epilogue ||
      (native_fp8 &&
       (spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_BIAS ||
        spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_BIAS_RESIDUAL) &&
       spec.quantization == APXINF_GEMM_QUANT_FP8_UNIT_SCALE &&
       spec.output_scale_is_unit != 0);
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
      resources.projection_dtype == APXINF_DTYPE_F16 &&
      spec.output_dtype == APXINF_DTYPE_E4M3 && spec.n % 16 == 0 &&
      spec.m * (spec.n / 16) <= INT32_MAX &&
      reinterpret_cast<uintptr_t>(projection) % 8 == 0 &&
      reinterpret_cast<uintptr_t>(bindings.output) % 8 == 0) {
    const int64_t groups = spec.m * (spec.n / 16);
    const int blocks = static_cast<int>(
        std::min<int64_t>((groups + 255) / 256, 4096));
    geglu_quant_f16_e4m3_packed8_kernel<<<
        blocks, 256, 0, static_cast<cudaStream_t>(bindings.stream)>>>(
        static_cast<const half*>(projection),
        static_cast<__nv_fp8_e4m3*>(bindings.output), spec.m, spec.n / 2,
        1.0F / bindings.output_scale);
    return cudaGetLastError();
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
  if (spec.semantic ==
          APXINF_GEMM_SEMANTIC_GEMM_SWIGLU &&
      spec.quantization == APXINF_GEMM_QUANT_NONE &&
      resources.projection_dtype == APXINF_DTYPE_BF16 &&
      spec.output_dtype == APXINF_DTYPE_BF16 && spec.alpha_is_unit != 0 &&
      spec.output_scale_is_unit != 0 && bindings.bias == nullptr) {
    return apxinf_activation_swiglu_bf16_rounded_pair_interleaved(
        projection, bindings.output, static_cast<int>(spec.m),
        static_cast<int>(spec.n / 2), 256,
        static_cast<cudaStream_t>(bindings.stream));
  }
  const int64_t output_width =
      (spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_GEGLU ||
       spec.semantic ==
           APXINF_GEMM_SEMANTIC_GEMM_SWIGLU)
          ? spec.n / 2
          : spec.n;
  const int64_t count = spec.m * output_width;
  const int blocks = static_cast<int>(
      std::min<int64_t>((count + 255) / 256, 4096));
  const uint32_t bias_dtype =
      spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_BIAS_GELU &&
              spec.a_dtype == APXINF_DTYPE_E4M3 &&
              spec.output_dtype == APXINF_DTYPE_E4M3
          ? APXINF_DTYPE_F16
          : binding_projection_dtype(spec);
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
