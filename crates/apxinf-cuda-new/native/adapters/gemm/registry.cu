#include "internal.h"

namespace apxinf::gemm {
namespace {

constexpr uint32_t kProviderCublas = 1;
constexpr uint32_t kProviderCublasLt = 2;
constexpr uint32_t kProviderCutlass = 3;

bool supports_vendor(const Spec& spec) {
  return spec.a_dtype == spec.b_dtype &&
         ((spec.a_dtype == APXINF_DTYPE_I8 &&
           spec.accumulation_dtype == APXINF_DTYPE_I32) ||
          (spec.a_dtype != APXINF_DTYPE_I8 &&
           spec.accumulation_dtype == APXINF_DTYPE_F32));
}

bool supports_native_fp8(const Spec& spec) {
  return supports_vendor(spec) && spec.a_dtype == APXINF_DTYPE_E4M3 &&
         spec.k % 16 == 0 && spec.n % 16 == 0;
}

AlignmentRequirements vendor_alignment(const Spec&) {
  return {};
}

AlignmentRequirements cublaslt_alignment(const Spec&) {
  // Algorithms returned by the heuristic API may use vectorized accesses;
  // cuBLASLt does not accept the operand pointers when enumerating them, so a
  // conservative 16-byte contract is required for a reusable recipe.
  AlignmentRequirements requirements{};
  requirements.a = 16;
  requirements.b = 16;
  requirements.output = 16;
  return requirements;
}

#ifdef APXINF_GEMM_CUTLASS
AlignmentRequirements cutlass_fp8_alignment(const Spec&) {
  AlignmentRequirements requirements{};
  requirements.a = 16;
  requirements.b = 16;
  requirements.output = 16;
  return requirements;
}

AlignmentRequirements cutlass_geglu_alignment(const Spec& spec) {
  AlignmentRequirements requirements{};
  if (spec.a_dtype == APXINF_DTYPE_E4M3) {
    requirements.a = 16;
    requirements.output = 8;
  } else {
    requirements.a = 32;
    requirements.output = 16;
  }
  return requirements;
}
#endif

void one_configuration(const Spec&, std::vector<int>& configs) {
  configs.push_back(0);
}

void cublaslt_configurations(const Spec&,
                             std::vector<int>& configs) {
  for (int rank = 0; rank < 8; ++rank) {
    configs.push_back(rank);
  }
}

#ifdef APXINF_GEMM_CUTLASS
bool supports_cutlass_fp8(const Spec& spec) {
  return spec.semantic == APXINF_GEMM_SEMANTIC_GEMM &&
         spec.a_dtype == APXINF_DTYPE_E4M3 &&
         spec.b_dtype == APXINF_DTYPE_E4M3 &&
         spec.output_dtype == APXINF_DTYPE_F16 &&
         spec.quantization == APXINF_GEMM_QUANT_FP8_UNIT_SCALE &&
         spec.n % 16 == 0 &&
         spec.k % 16 == 0 && spec.output_scale_is_unit != 0;
}

bool supports_cutlass_fp8_geglu(const Spec& spec) {
  const bool exact_shape = (spec.m == 522 || spec.m == 533) &&
                           spec.n == 32768 && spec.k == 2048;
  return exact_shape && spec.a_dtype == APXINF_DTYPE_E4M3 &&
         spec.b_dtype == APXINF_DTYPE_E4M3 &&
         spec.output_dtype == APXINF_DTYPE_E4M3 &&
         spec.quantization == APXINF_GEMM_QUANT_FP8_UNIT_SCALE &&
         spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_GEGLU &&
         spec.output_scale_is_unit != 0;
}

bool supports_cutlass_bf16_geglu(const Spec& spec) {
  const bool exact_shape = (spec.m == 522 || spec.m == 533) &&
                           spec.n == 32768 && spec.k == 2048;
  return exact_shape && spec.a_dtype == APXINF_DTYPE_BF16 &&
         spec.b_dtype == APXINF_DTYPE_BF16 &&
         spec.output_dtype == APXINF_DTYPE_BF16 &&
         // This kernel has no alpha epilogue at all, so a non-unit alpha is a
         // contract mismatch rather than a slower path.
         spec.alpha_is_unit != 0 && spec.output_scale_is_unit != 0 &&
         spec.quantization == APXINF_GEMM_QUANT_NONE &&
         spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_GEGLU;
}

void cutlass_configurations(const Spec&,
                            std::vector<int>& configs) {
  for (int configuration = 0; configuration < 4; ++configuration) {
    configs.push_back(configuration);
  }
}
#endif

}  // namespace

bool supports_device(const Implementation& implementation,
                     int device,
                     std::string* reason) {
  cudaDeviceProp properties{};
  const auto status = cudaGetDeviceProperties(&properties, device);
  if (status != cudaSuccess) {
    if (reason != nullptr) {
      *reason = std::string("cannot query CUDA device: ") +
                cudaGetErrorString(status);
    }
    return false;
  }
  const int sm = properties.major * 10 + properties.minor;
  const auto* target = compiled_target(sm);
  if (target == nullptr) {
    if (reason != nullptr) {
      *reason = "SM " + std::to_string(sm) +
                " is not present in this GEMM build";
    }
    return false;
  }
  if ((target->features & implementation.required_device_features) !=
      implementation.required_device_features) {
    if (reason != nullptr) {
      *reason = "SM " + std::to_string(sm) +
                " lacks a capability required by this candidate";
    }
    return false;
  }
  return true;
}

const std::vector<Implementation>& registry(uint32_t semantic) {
  static const std::vector<Implementation> vendor_entries = {
      {kProviderCublas, 1, 1, "cublas+custom-epilogue", 0, true, true,
       supports_vendor, vendor_alignment, cublas_resource_requirements,
       one_configuration, prepare_cublas, launch_cublas, destroy_cublas},
      {kProviderCublasLt, 1, 1, "cublasLt+custom-epilogue", 0, true, false,
       supports_vendor, cublaslt_alignment, cublaslt_resource_requirements,
       cublaslt_configurations, prepare_cublaslt, launch_cublaslt,
       destroy_cublaslt},
  };
  // Keep GEMM+bias as a separate L3 tuning domain even though its current L1
  // candidates happen to be the same vendor implementations.
  static const std::vector<Implementation> gemm_bias_entries = {
      {kProviderCublas, 1, 1, "cublas+custom-epilogue", 0, true, true,
       supports_vendor, vendor_alignment, cublas_resource_requirements,
       one_configuration, prepare_cublas, launch_cublas, destroy_cublas},
      {kProviderCublasLt, 1, 1, "cublasLt+custom-epilogue", 0, true, false,
       supports_vendor, cublaslt_alignment, cublaslt_resource_requirements,
       cublaslt_configurations, prepare_cublaslt, launch_cublaslt,
       destroy_cublaslt},
  };
  static const std::vector<Implementation> gemm_entries = {
      {kProviderCublas, 1, 1, "cublas+custom-epilogue", 0, true, true,
       supports_vendor, vendor_alignment, cublas_resource_requirements,
       one_configuration, prepare_cublas, launch_cublas, destroy_cublas},
      {kProviderCublasLt, 1, 1, "cublasLt+custom-epilogue", 0, true, false,
       supports_vendor, cublaslt_alignment, cublaslt_resource_requirements,
       cublaslt_configurations, prepare_cublaslt, launch_cublaslt,
       destroy_cublaslt},
      {kProviderCublasLt, 2, 1, "cublasLt-native-fp8+custom-epilogue",
       kDeviceFeatureNativeFp8, true, false, supports_native_fp8,
       cublaslt_alignment, cublaslt_native_fp8_resource_requirements,
       cublaslt_configurations, prepare_cublaslt_native_fp8, launch_cublaslt,
       destroy_cublaslt},
#ifdef APXINF_GEMM_CUTLASS
      {kProviderCutlass, 1, 1, "cutlass-fp8", kDeviceFeatureCutlassSm100, true, true,
       supports_cutlass_fp8, cutlass_fp8_alignment,
       cutlass_fp8_resource_requirements, cutlass_configurations,
       prepare_cutlass_fp8_gemm, launch_cutlass_fp8_gemm, destroy_cutlass},
#endif
  };
  static const std::vector<Implementation> gemm_geglu_entries = {
      {kProviderCublas, 1, 1, "cublas+custom-epilogue", 0, true, true,
       supports_vendor, vendor_alignment, cublas_resource_requirements,
       one_configuration, prepare_cublas, launch_cublas, destroy_cublas},
      {kProviderCublasLt, 1, 1, "cublasLt+custom-epilogue", 0, true, false,
       supports_vendor, cublaslt_alignment, cublaslt_resource_requirements,
       cublaslt_configurations, prepare_cublaslt, launch_cublaslt,
       destroy_cublaslt},
#ifdef APXINF_GEMM_CUTLASS
      {kProviderCutlass, 2, 2, "cutlass-dual-geglu", kDeviceFeatureCutlassSm100, true, true,
       supports_cutlass_fp8_geglu, cutlass_geglu_alignment,
       cutlass_geglu_resource_requirements, one_configuration,
       prepare_cutlass_geglu, launch_cutlass_fp8_geglu, destroy_cutlass},
      {kProviderCutlass, 3, 2, "cutlass-bf16-dual-geglu",
       kDeviceFeatureCutlassSm100, true, true, supports_cutlass_bf16_geglu,
       cutlass_geglu_alignment, cutlass_geglu_resource_requirements,
       one_configuration, prepare_cutlass_geglu,
       launch_cutlass_bf16_geglu, destroy_cutlass},
#endif
  };
  switch (semantic) {
    case APXINF_GEMM_SEMANTIC_GEMM:
      return gemm_entries;
    case APXINF_GEMM_SEMANTIC_GEMM_BIAS_GELU:
      return vendor_entries;
    case APXINF_GEMM_SEMANTIC_GEMM_GEGLU:
      return gemm_geglu_entries;
    case APXINF_GEMM_SEMANTIC_GEMM_BIAS:
      return gemm_bias_entries;
  }
  throw Failure(APXINF_STATUS_INTERNAL_ERROR, "unknown GEMM semantic registry");
}

Execution::~Execution() {
  cudaSetDevice(device);
  if (implementation != nullptr && implementation->destroy != nullptr) {
    implementation->destroy(*this);
  }
}

std::unique_ptr<Execution> prepare(
    const Implementation& implementation, int configuration, const Spec& spec,
    const apxinf_gemm_policy_t& policy,
    const apxinf_gemm_bindings_t& bindings, int device) {
  std::string device_reason;
  if (!supports_device(implementation, device, &device_reason)) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "candidate is incompatible with this device: " +
                      device_reason);
  }
  if (!implementation.supports(spec)) {
    throw Failure(APXINF_STATUS_UNSUPPORTED, "candidate contract mismatch");
  }
  if (!supports_alignment(implementation, spec)) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "candidate binding alignment mismatch");
  }
  if (policy.graph_safe && !implementation.graph_safe) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "candidate is not CUDA Graph safe");
  }
  if (policy.deterministic && !implementation.deterministic) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "candidate is not deterministic");
  }

  const size_t required = implementation.resource_requirements(spec);
  if (required > policy.workspace_limit) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "workspace policy exceeded before allocation: requires at least " +
                      std::to_string(required) + " bytes");
  }

  auto execution = std::make_unique<Execution>();
  execution->spec = spec;
  execution->bindings = bindings;
  execution->configuration = configuration;
  execution->implementation = &implementation;
  execution->device = device;
  execution->resource_limit = policy.workspace_limit;
  check_cuda(cudaSetDevice(device));
  implementation.prepare(*execution);
  if (execution->resource_bytes > policy.workspace_limit) {
    throw Failure(APXINF_STATUS_UNSUPPORTED, "workspace policy exceeded");
  }
  return execution;
}

}  // namespace apxinf::gemm

namespace {
bool test_create_called = false;

bool test_supports(const apxinf::gemm::Spec&) { return true; }
apxinf::gemm::AlignmentRequirements test_alignment(
    const apxinf::gemm::Spec&) {
  return {};
}
size_t test_resource_requirements(const apxinf::gemm::Spec&) { return 4096; }
void test_configs(const apxinf::gemm::Spec&, std::vector<int>& values) {
  values.push_back(0);
}
void test_create(apxinf::gemm::Execution&) {
  test_create_called = true;
}
void test_destroy(apxinf::gemm::Execution&) noexcept {}
cudaError_t test_launch(apxinf::gemm::Execution&) {
  return cudaSuccess;
}
}  // namespace

// Private regression hook: proves fixed resource requirements are rejected
// before provider construction (and therefore before provider allocation).
extern "C" int apxinf_gemm_test_resource_prefilter(int device) {
  using namespace apxinf::gemm;
  test_create_called = false;
  const Implementation implementation = {
      999,
      1,
      1,
      "resource-prefilter-test",
      0,
      true,
      true,
      test_supports,
      test_alignment,
      test_resource_requirements,
      test_configs,
      test_create,
      test_launch,
      test_destroy};
  Spec spec{};
  spec.semantic = APXINF_GEMM_SEMANTIC_GEMM;
  apxinf_gemm_policy_t policy{};
  policy.workspace_limit = 4095;
  apxinf_gemm_bindings_t bindings{};
  try {
    prepare(implementation, 0, spec, policy, bindings, device);
  } catch (const Failure& failure) {
    return failure.status == APXINF_STATUS_UNSUPPORTED && !test_create_called;
  }
  return 0;
}
