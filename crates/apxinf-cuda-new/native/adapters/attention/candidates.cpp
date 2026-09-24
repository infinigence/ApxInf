#include "internal.h"

#include <algorithm>

namespace apxinf::attention {
namespace {

constexpr uint32_t kProviderCustom = 1;
#if defined(APXINF_ATTENTION_FA2)
constexpr uint32_t kProviderFa2 = 2;
#endif
#if defined(APXINF_ATTENTION_CUTLASS)
constexpr uint32_t kProviderCutlass = 3;
#endif

bool supports_custom(const Spec& spec) {
  return (spec.dtype == APXINF_DTYPE_F16 ||
          spec.dtype == APXINF_DTYPE_BF16) &&
         spec.output_dtype == spec.dtype;
}

bool supports_dense_custom(const Spec& spec) {
  return spec.semantic == APXINF_ATTENTION_SEMANTIC_DENSE &&
         supports_custom(spec);
}

bool supports_kv_cache_custom(const Spec& spec) {
  return spec.semantic == APXINF_ATTENTION_SEMANTIC_KV_CACHE &&
         supports_custom(spec);
}

bool supports_segmented_custom(const Spec& spec) {
  return spec.semantic == APXINF_ATTENTION_SEMANTIC_SEGMENTED &&
         supports_custom(spec);
}

AlignmentRequirements natural_alignment(const Spec&) { return {}; }

void one_configuration(const Spec&, std::vector<int>& configurations) {
  configurations.push_back(0);
}

#if defined(APXINF_ATTENTION_FA2)
bool supports_fa2(const Spec& spec) {
  const bool layout_supported =
      spec.semantic == APXINF_ATTENTION_SEMANTIC_DENSE ||
      (spec.semantic == APXINF_ATTENTION_SEMANTIC_KV_CACHE &&
       spec.key_capacity == spec.key_tokens &&
       (spec.mask == APXINF_ATTENTION_MASK_NONE ||
        spec.query_start == spec.key_tokens - spec.query_tokens));
  const bool bf16_supported =
      spec.dtype == APXINF_DTYPE_BF16 && spec.head_dim <= 256;
  const bool f16_supported = spec.dtype == APXINF_DTYPE_F16 &&
                             spec.mask == APXINF_ATTENTION_MASK_NONE &&
                             spec.head_dim <= 256;
  return layout_supported && (bf16_supported || f16_supported) &&
         spec.output_dtype == spec.dtype;
}

bool supports_fa2_packed_qkv(const Spec& spec) {
  return spec.semantic == APXINF_ATTENTION_SEMANTIC_PACKED_QKV &&
         spec.dtype == APXINF_DTYPE_F16 &&
         spec.output_dtype == APXINF_DTYPE_F16 && spec.query_tokens == 256 &&
         spec.query_heads == 16 && spec.kv_heads == 16 &&
         spec.head_dim == 72;
}

bool supports_fa2_splitkv(const Spec& spec) {
  const int64_t key_tile = spec.head_dim == 128 ? 128 : 64;
  return spec.semantic == APXINF_ATTENTION_SEMANTIC_DENSE &&
         spec.dtype == APXINF_DTYPE_BF16 &&
         spec.output_dtype == APXINF_DTYPE_BF16 &&
         spec.query_tokens <= 64 && spec.key_tokens > spec.query_tokens &&
         spec.query_heads > spec.kv_heads &&
         spec.query_heads % spec.kv_heads == 0 &&
         (spec.head_dim == 128 || spec.head_dim == 256) &&
         spec.key_tokens > key_tile;
}

void splitkv_configurations(const Spec& spec,
                            std::vector<int>& configurations) {
  const uint64_t key_tile = spec.head_dim == 128 ? 128 : 64;
  const uint64_t key_tiles =
      (static_cast<uint64_t>(spec.key_tokens) + key_tile - 1) / key_tile;
  const int limit = static_cast<int>(std::min<uint64_t>(128, key_tiles));
  for (int splits = 2; splits <= limit; ++splits) {
    const uint64_t current = (key_tiles + splits - 1) / splits;
    const uint64_t previous = (key_tiles + splits - 2) / (splits - 1);
    if (current != previous) configurations.push_back(splits);
  }
}

#if defined(APXINF_ATTENTION_FA2_E4M3)
bool supports_fa2_direct_e4m3_522(const Spec& spec) {
  return spec.semantic == APXINF_ATTENTION_SEMANTIC_DENSE &&
         spec.mask == APXINF_ATTENTION_MASK_NONE && spec.batch == 1 &&
         spec.query_tokens == 522 && spec.key_tokens == 522 &&
         spec.key_capacity == 522 && spec.query_heads == 8 &&
         spec.kv_heads == 1 && spec.head_dim == 256 &&
         spec.dtype == APXINF_DTYPE_F16 &&
         spec.output_dtype == APXINF_DTYPE_E4M3;
}
#endif

AlignmentRequirements fa2_alignment(const Spec&) {
  return {16, 16, 16, 16};
}
#endif

#if defined(APXINF_ATTENTION_CUTLASS)
bool supports_cutlass(const Spec& spec) {
  return spec.semantic == APXINF_ATTENTION_SEMANTIC_DENSE &&
         spec.mask == APXINF_ATTENTION_MASK_NONE &&
         spec.query_tokens == 256 && spec.key_tokens == 256 &&
         spec.query_heads == 16 && spec.kv_heads == 16 &&
         spec.head_dim == 72 &&
         spec.dtype == APXINF_DTYPE_BF16 &&
         spec.output_dtype == spec.dtype;
}

AlignmentRequirements cutlass_alignment(const Spec&) {
  return {16, 16, 16, 16};
}
#endif

}  // namespace

const ImplementationRegistry& registry(uint32_t semantic) {
  static const ImplementationRegistry dense_entries = {
#if defined(APXINF_ATTENTION_CUTLASS)
      {kProviderCutlass, 1, 1, "cutlass-fmha-sm100",
       apxinf::gemm::kDeviceFeatureCutlassSm100, true, true, false,
       supports_cutlass, cutlass_alignment, cutlass_resource_requirements,
       one_configuration, prepare_cutlass, launch_cutlass, destroy_cutlass},
#endif
#if defined(APXINF_ATTENTION_FA2)
#if defined(APXINF_ATTENTION_FA2_E4M3)
      {kProviderFa2, 2, 1, "flash-attention-2-f16-e4m3-522",
       apxinf::gemm::kDeviceFeatureCutlassSm100, true, true, false,
       supports_fa2_direct_e4m3_522, fa2_alignment,
       fa2_resource_requirements, one_configuration, prepare_fa2, launch_fa2,
       destroy_fa2},
#endif
      {kProviderFa2, 3, 1, "flash-attention-2-split-kv",
       apxinf::gemm::kDeviceFeatureFa2, true, true, false,
       supports_fa2_splitkv, fa2_alignment,
       fa2_splitkv_resource_requirements, splitkv_configurations,
       prepare_fa2_splitkv, launch_fa2_splitkv, destroy_fa2_splitkv},
      {kProviderFa2, 1, 1, "flash-attention-2",
       apxinf::gemm::kDeviceFeatureFa2, true, true, false, supports_fa2,
       fa2_alignment, fa2_resource_requirements, one_configuration,
       prepare_fa2, launch_fa2, destroy_fa2},
#endif
      {kProviderCustom, 1, 1, "custom-attention-fallback", 0, true, true,
       true, supports_dense_custom, natural_alignment,
       custom_resource_requirements, one_configuration, prepare_custom,
       launch_custom, destroy_custom},
  };
  static const ImplementationRegistry kv_cache_entries = {
#if defined(APXINF_ATTENTION_FA2)
      {kProviderFa2, 1, 1, "flash-attention-2-kv-cache",
       apxinf::gemm::kDeviceFeatureFa2, true, true, false, supports_fa2,
       fa2_alignment, fa2_resource_requirements, one_configuration,
       prepare_fa2, launch_fa2, destroy_fa2},
#endif
      {kProviderCustom, 2, 1, "custom-kv-cache-attention", 0, true, true,
       true, supports_kv_cache_custom, natural_alignment,
       custom_resource_requirements, one_configuration, prepare_custom,
       launch_custom, destroy_custom},
  };
  static const ImplementationRegistry segmented_entries = {
      {kProviderCustom, 3, 1, "custom-segmented-attention", 0, true, true,
       true, supports_segmented_custom, natural_alignment,
       custom_resource_requirements, one_configuration, prepare_custom,
       launch_custom, destroy_custom},
  };
  static const ImplementationRegistry packed_qkv_entries = {
#if defined(APXINF_ATTENTION_FA2)
      {kProviderFa2, 4, 1, "flash-attention-2-packed-qkv",
       apxinf::gemm::kDeviceFeatureCutlassSm100, true, true, true,
       supports_fa2_packed_qkv, fa2_alignment, fa2_resource_requirements,
       one_configuration, prepare_fa2, launch_fa2, destroy_fa2},
#endif
  };
  switch (semantic) {
    case APXINF_ATTENTION_SEMANTIC_DENSE:
      return dense_entries;
    case APXINF_ATTENTION_SEMANTIC_KV_CACHE:
      return kv_cache_entries;
    case APXINF_ATTENTION_SEMANTIC_SEGMENTED:
      return segmented_entries;
    case APXINF_ATTENTION_SEMANTIC_PACKED_QKV:
      return packed_qkv_entries;
  }
  throw Failure(APXINF_STATUS_INTERNAL_ERROR,
                "unknown Attention semantic registry");
}

bool supports_device(const Implementation& implementation, int device,
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
  const auto* target = apxinf::gemm::compiled_target(sm);
  if (target == nullptr) {
    if (reason != nullptr) {
      *reason = "SM " + std::to_string(sm) +
                " is not present in this CUDA build";
    }
    return false;
  }
  if ((target->features & implementation.required_device_features) !=
      implementation.required_device_features) {
    if (reason != nullptr) {
      *reason = "device lacks a capability required by this candidate";
    }
    return false;
  }
  return true;
}

bool supports_alignment(const Implementation& implementation,
                        const Spec& spec) {
  const auto required = implementation.alignment_requirements(spec);
  return spec.q_alignment >= required.query &&
         spec.k_alignment >= required.key &&
         spec.v_alignment >= required.value &&
         spec.output_alignment >= required.output;
}

Execution::~Execution() {
  cudaSetDevice(device);
  if (implementation != nullptr && implementation->destroy != nullptr) {
    implementation->destroy(*this);
  }
}

std::unique_ptr<Execution> prepare(
    const Implementation& implementation, int configuration, const Spec& spec,
    const apxinf_attention_policy_t& policy,
    const apxinf_attention_bindings_t& bindings, int device) {
  std::string reason;
  if (!supports_device(implementation, device, &reason)) {
    throw Failure(APXINF_STATUS_UNSUPPORTED, reason);
  }
  if (!implementation.supports(spec)) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "Attention candidate contract mismatch");
  }
  if (!supports_alignment(implementation, spec)) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "Attention candidate binding alignment mismatch");
  }
  if (policy.graph_safe && !implementation.graph_safe) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "Attention candidate is not CUDA Graph safe");
  }
  if (policy.deterministic && !implementation.deterministic) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "Attention candidate is not deterministic");
  }
  const size_t required =
      implementation.resource_requirements(spec, configuration);
  if (required > policy.workspace_limit) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "Attention workspace policy exceeded before allocation");
  }
  auto execution = std::make_unique<Execution>();
  execution->spec = spec;
  execution->bindings = bindings;
  execution->configuration = configuration;
  execution->device = device;
  execution->implementation = &implementation;
  execution->resource_limit = policy.workspace_limit;
  check_cuda(cudaSetDevice(device));
  implementation.prepare(*execution);
  return execution;
}

}  // namespace apxinf::attention
