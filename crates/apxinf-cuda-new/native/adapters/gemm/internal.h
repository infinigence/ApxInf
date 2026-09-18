#pragma once

#include "../../include/apxinf_cuda/gemm.h"
#include "../../framework/registry.h"
#include "../../framework/runtime_internal.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "apxinf_cuda_arches.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace apxinf::gemm {

struct Spec : apxinf_gemm_spec_t {};

using apxinf::framework::Failure;
using apxinf::framework::abi_boundary;
using apxinf::framework::check_cublas;
using apxinf::framework::check_cuda;

inline size_t dtype_bytes(uint32_t dtype) {
  if (dtype == APXINF_DTYPE_F32 || dtype == APXINF_DTYPE_I32) {
    return 4;
  }
  if (dtype == APXINF_DTYPE_E4M3 || dtype == APXINF_DTYPE_I8) {
    return 1;
  }
  return 2;
}

inline bool has_row_channel_scales(const Spec& spec) {
  return spec.quantization == APXINF_GEMM_QUANT_FP8_ROW_CHANNEL ||
         spec.quantization == APXINF_GEMM_QUANT_W8A8_ROW_CHANNEL;
}

struct Execution;
using PrepareExecutionFn = void (*)(Execution&);
using EnqueueFn = cudaError_t (*)(Execution&);
using DestroyExecutionFn = void (*)(Execution&) noexcept;

struct AlignmentRequirements {
  uint32_t a = 1;
  uint32_t b = 1;
  uint32_t bias = 1;
  uint32_t a_scales = 1;
  uint32_t b_scales = 1;
  uint32_t output = 1;
};

using AlignmentFn = AlignmentRequirements (*)(const Spec&);
using ResourceRequirementsFn = size_t (*)(const Spec&);

struct Implementation {
  uint32_t provider_id;
  uint32_t implementation_id;
  uint32_t implementation_version;
  const char* name;
  uint64_t required_device_features;
  bool graph_safe;
  bool deterministic;
  bool (*supports)(const Spec&);
  AlignmentFn alignment_requirements;
  ResourceRequirementsFn resource_requirements;
  void (*enumerate_configs)(const Spec&, std::vector<int>&);
  PrepareExecutionFn prepare;
  EnqueueFn enqueue;
  DestroyExecutionFn destroy;
};

bool supports_device(const Implementation& implementation,
                     int device,
                     std::string* reason = nullptr);

inline bool supports_alignment(const Implementation& implementation,
                               const Spec& spec) {
  const auto required = implementation.alignment_requirements(spec);
  return spec.a_alignment >= required.a &&
         spec.b_alignment >= required.b &&
         spec.bias_alignment >=
             (spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_BIAS ||
                      spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_BIAS_GELU
                                     ? required.bias
                                     : 0) &&
         spec.a_scales_alignment >=
             (has_row_channel_scales(spec) ? required.a_scales : 0) &&
         spec.b_scales_alignment >=
             (has_row_channel_scales(spec) ? required.b_scales : 0) &&
         spec.output_alignment >= required.output;
}

using Recipe = apxinf::framework::Recipe;

struct TuningKeys {
  // A recipe under this key is fully tuned for one exact equivalence class.
  std::string key;
};

struct AccuracyMetrics {
  bool finite = false;
  bool valid = false;
  double max_absolute_error = 0.0;
  double max_scaled_element_error = 0.0;
  double relative_l2 = 0.0;
  double cosine = 0.0;
};

struct Execution {
  Spec spec{};
  apxinf_gemm_bindings_t bindings{};
  int configuration = 0;
  int device = 0;
  const Implementation* implementation = nullptr;
  // Opaque provider-owned state. The common execution framework never knows which
  // handles, descriptors, algorithms or temporary buffers a provider needs.
  void* provider_state = nullptr;
  size_t resource_bytes = 0;
  // Upper bound for provider-owned device scratch. Providers may query
  // algorithms and construct descriptors first, but must not allocate device
  // memory beyond this limit.
  size_t resource_limit = 0;
  std::string summary;

  ~Execution();
};

using ImplementationRegistry = apxinf::framework::Registry<Implementation>;
const ImplementationRegistry& registry(uint32_t semantic);
void prepare_cublas(Execution& execution);
size_t cublas_resource_requirements(const Spec& spec);
void destroy_cublas(Execution& execution) noexcept;
cudaError_t launch_cublas(Execution& execution);
void prepare_cublaslt(Execution& execution);
void prepare_cublaslt_native_fp8(Execution& execution);
size_t cublaslt_resource_requirements(const Spec& spec);
size_t cublaslt_native_fp8_resource_requirements(const Spec& spec);
void destroy_cublaslt(Execution& execution) noexcept;
cudaError_t launch_cublaslt(Execution& execution);
void prepare_cutlass_fp8_gemm(Execution& execution);
void prepare_cutlass_geglu(Execution& execution);
size_t cutlass_fp8_resource_requirements(const Spec& spec);
size_t cutlass_geglu_resource_requirements(const Spec& spec);
void destroy_cutlass(Execution& execution) noexcept;
cudaError_t launch_cutlass_fp8_gemm(Execution& execution);
cudaError_t launch_cutlass_fp8_geglu(Execution& execution);
cudaError_t launch_cutlass_bf16_geglu(Execution& execution);
uint64_t cutlass_weight_prepack_count(const Execution& execution);

TuningKeys tuning_keys(const Spec& spec,
                       const apxinf_gemm_policy_t& policy,
                       int device);
using apxinf::framework::read_recipe;
using apxinf::framework::write_recipe;
std::unique_ptr<Execution> prepare(const Implementation& implementation,
                                   int configuration, const Spec& spec,
                                   const apxinf_gemm_policy_t& policy,
                                   const apxinf_gemm_bindings_t& bindings,
                                   int device);
Recipe tune(
    const Spec& spec, const apxinf_gemm_policy_t& policy,
    const apxinf_gemm_bindings_t& bindings, int device,
    std::string& report);
AccuracyMetrics compare_reference(const std::vector<float>& expected,
                                  const std::vector<float>& actual);
std::string format_accuracy(const AccuracyMetrics& metrics);
void validate_candidates(const Spec& spec,
                         const apxinf_gemm_policy_t& policy,
                         const apxinf_gemm_bindings_t& bindings, int device,
                         const float* expected, size_t expected_len);

}  // namespace apxinf::gemm
