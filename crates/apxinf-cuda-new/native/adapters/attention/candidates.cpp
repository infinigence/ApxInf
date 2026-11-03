#include "internal.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

namespace apxinf::attention {
namespace {

constexpr uint32_t kProviderCustom = 1;
constexpr uint32_t kProviderCublas = 5;
#if defined(APXINF_ATTENTION_FA2)
constexpr uint32_t kProviderFa2 = 2;
#endif
#if defined(APXINF_ATTENTION_CUTLASS)
constexpr uint32_t kProviderCutlass = 3;
#endif
#if defined(APXINF_SEGMENTED_ATTENTION_SM110)
constexpr uint32_t kProviderSpecializedSm110 = 4;
#endif

bool supports_custom(const Spec& spec) {
  return (spec.dtype == APXINF_DTYPE_F16 ||
          spec.dtype == APXINF_DTYPE_BF16) &&
         spec.output_dtype == spec.dtype &&
         spec.qkv_layout == APXINF_ATTENTION_QKV_LAYOUT_SEPARATE &&
         spec.numerics == APXINF_ATTENTION_NUMERICS_FLEXIBLE;
}

bool supports_dense_custom(const Spec& spec) {
  return spec.semantic == APXINF_ATTENTION_SEMANTIC_DENSE &&
         supports_custom(spec);
}

bool supports_rounded(const Spec& spec) {
  return spec.semantic == APXINF_ATTENTION_SEMANTIC_ROUNDED_ADDITIVE &&
         supports_custom(spec) && spec.mask == APXINF_ATTENTION_MASK_NONE &&
         spec.query_heads == spec.kv_heads &&
         spec.query_heads <= INT32_MAX / spec.head_dim &&
         spec.batch <= INT32_MAX / spec.query_heads / spec.query_tokens;
}

bool supports_kv_cache_custom(const Spec& spec) {
  return spec.semantic == APXINF_ATTENTION_SEMANTIC_KV_CACHE &&
         supports_custom(spec);
}

bool supports_segmented_custom(const Spec& spec) {
  return spec.semantic == APXINF_ATTENTION_SEMANTIC_SEGMENTED &&
         supports_custom(spec);
}

bool supports_cublas_f16(const Spec& spec) {
  return spec.semantic == APXINF_ATTENTION_SEMANTIC_DENSE &&
         spec.qkv_layout == APXINF_ATTENTION_QKV_LAYOUT_SEPARATE &&
         spec.dtype == APXINF_DTYPE_F16 &&
         spec.output_dtype == APXINF_DTYPE_F16 &&
         spec.batch == 1 && spec.key_tokens <= 1024 &&
         spec.scale_is_default != 0 &&
         spec.numerics == APXINF_ATTENTION_NUMERICS_STABLE;
}

#if defined(APXINF_ATTENTION_INDEXED_SM110)
bool supports_indexed(const Spec& s){
 return s.semantic==APXINF_ATTENTION_SEMANTIC_DENSE &&
  s.qkv_layout==APXINF_ATTENTION_QKV_LAYOUT_SEPARATE &&
  s.mask==APXINF_ATTENTION_MASK_NONE && s.dtype==APXINF_DTYPE_BF16 &&
  s.output_dtype==s.dtype && s.numerics==APXINF_ATTENTION_NUMERICS_FLEXIBLE &&
  s.batch==1 && s.head_dim==128 && s.query_heads==40 && s.kv_heads==40 &&
  s.query_tokens==1785 && (s.key_tokens==2665 || s.key_tokens==512 || s.key_tokens==257) &&
  s.scale_is_default!=0;
}
AlignmentRequirements indexed_alignment(const Spec&){return {16,16,16,16};}
void indexed_configurations(const Spec&,std::vector<int>& out){out.push_back(0);out.push_back(1);}
#endif

AlignmentRequirements natural_alignment(const Spec&) { return {}; }

void one_configuration(const Spec&, std::vector<int>& configurations) {
  configurations.push_back(0);
}

#if defined(APXINF_ATTENTION_FA2)
bool supports_efficient(const Spec& s) {
  return s.semantic == APXINF_ATTENTION_SEMANTIC_DENSE &&
      s.mask == APXINF_ATTENTION_MASK_NONE && supports_custom(s) &&
      s.head_dim == 384 && s.query_heads == s.kv_heads &&
      s.query_heads <= INT32_MAX / s.head_dim && s.query_heads <= 65535 &&
      s.batch <= 65535 && s.query_tokens <= INT32_MAX - 32 &&
      s.batch <= INT32_MAX / s.query_tokens &&
      s.query_tokens <= INT32_MAX / (s.query_heads * s.head_dim) &&
      s.key_tokens <= INT32_MAX / (s.query_heads * s.head_dim);
}
bool fa2_head_dim_supported(const Spec& spec) {
  return spec.head_dim > 0 && spec.head_dim <= 256 &&
         spec.head_dim % 8 == 0;
}

int multiprocessor_count() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) return 0;
  static std::mutex mutex;
  static std::unordered_map<int, int> counts;
  std::lock_guard<std::mutex> lock(mutex);
  const auto cached = counts.find(device);
  if (cached != counts.end()) return cached->second;
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) return 0;
  counts.emplace(device, properties.multiProcessorCount);
  return properties.multiProcessorCount;
}

bool splitkv_shape_supported(const Spec& spec) {
  const int64_t key_tile = spec.head_dim == 128 ? 128 : 64;
  const bool layout_supported =
      spec.semantic == APXINF_ATTENTION_SEMANTIC_DENSE ||
      (spec.semantic == APXINF_ATTENTION_SEMANTIC_KV_CACHE &&
       spec.dynamic_decode == 0 && spec.key_capacity >= spec.key_tokens &&
       (spec.mask == APXINF_ATTENTION_MASK_NONE ||
        spec.query_start == spec.key_tokens - spec.query_tokens));
  const bool bf16_supported = spec.dtype == APXINF_DTYPE_BF16;
  const bool f16_supported = spec.dtype == APXINF_DTYPE_F16 &&
                             spec.mask == APXINF_ATTENTION_MASK_NONE;
  return layout_supported &&
         spec.qkv_layout == APXINF_ATTENTION_QKV_LAYOUT_SEPARATE &&
         (bf16_supported || f16_supported) &&
         spec.output_dtype == spec.dtype && spec.query_tokens <= 64 &&
         spec.key_tokens > spec.query_tokens &&
         spec.query_heads > spec.kv_heads &&
         spec.query_heads % spec.kv_heads == 0 &&
         (spec.head_dim == 128 || spec.head_dim == 256) &&
         spec.key_tokens > key_tile;
}

// Native copy of the legacy FA2 occupancy policy. Stable numerics uses this
// single configuration instead of timing reduction variants with different
// rounding behavior.
int stable_split_count(const Spec& spec) {
  const int sm_count = multiprocessor_count();
  if (sm_count <= 0) return 1;
  const uint64_t key_tile = spec.head_dim <= 128 ? 128 : 64;
  const uint64_t key_tiles =
      (static_cast<uint64_t>(spec.key_tokens) + key_tile - 1) / key_tile;
  const uint64_t row_tiles =
      (static_cast<uint64_t>(spec.query_tokens) + 63) / 64;
  const uint64_t base_jobs = static_cast<uint64_t>(spec.batch) *
                             static_cast<uint64_t>(spec.query_heads) *
                             row_tiles;
  const uint64_t execution_slots =
      static_cast<uint64_t>(sm_count) * 2;
  if (key_tiles == 0 || execution_slots == 0 ||
      base_jobs * 5 >= execution_slots * 4) {
    return 1;
  }
  const uint64_t limit = std::min<uint64_t>(
      128, std::min<uint64_t>(execution_slots, key_tiles));
  const auto eligible = [key_tiles](uint64_t splits) {
    if (splits == 1) return true;
    return (key_tiles + splits - 1) / splits !=
           (key_tiles + splits - 2) / (splits - 1);
  };
  const auto utilization = [base_jobs, execution_slots](uint64_t splits) {
    const double waves = static_cast<double>(base_jobs * splits) /
                         static_cast<double>(execution_slots);
    return waves / std::ceil(waves);
  };
  double peak = 0.0;
  for (uint64_t splits = 1; splits <= limit; ++splits) {
    if (eligible(splits)) peak = std::max(peak, utilization(splits));
  }
  for (uint64_t splits = 1; splits <= limit; ++splits) {
    if (eligible(splits) && utilization(splits) >= 0.85 * peak) {
      return static_cast<int>(splits);
    }
  }
  return 1;
}

bool supports_fa2(const Spec& spec) {
  const bool layout_supported =
      spec.semantic == APXINF_ATTENTION_SEMANTIC_DENSE ||
      (spec.semantic == APXINF_ATTENTION_SEMANTIC_KV_CACHE &&
       spec.dynamic_decode == 0 && spec.key_capacity >= spec.key_tokens &&
       (spec.mask == APXINF_ATTENTION_MASK_NONE ||
        spec.query_start == spec.key_tokens - spec.query_tokens));
  const bool bf16_supported =
      spec.dtype == APXINF_DTYPE_BF16 && fa2_head_dim_supported(spec);
  const bool f16_supported = spec.dtype == APXINF_DTYPE_F16 &&
                             spec.mask == APXINF_ATTENTION_MASK_NONE &&
                             fa2_head_dim_supported(spec);
  const bool supported = layout_supported &&
                         spec.qkv_layout ==
                             APXINF_ATTENTION_QKV_LAYOUT_SEPARATE &&
                         (bf16_supported || f16_supported) &&
                         spec.output_dtype == spec.dtype;
  return supported &&
         !(spec.dtype == APXINF_DTYPE_F16 &&
           spec.numerics == APXINF_ATTENTION_NUMERICS_STABLE) &&
         !(spec.numerics == APXINF_ATTENTION_NUMERICS_STABLE &&
           splitkv_shape_supported(spec) && stable_split_count(spec) > 1);
}

bool supports_fa2_segmented(const Spec& spec) {
  return spec.semantic == APXINF_ATTENTION_SEMANTIC_SEGMENTED &&
         spec.qkv_layout == APXINF_ATTENTION_QKV_LAYOUT_SEPARATE &&
         (spec.dtype == APXINF_DTYPE_BF16 ||
          spec.dtype == APXINF_DTYPE_F16) &&
         spec.output_dtype == spec.dtype && spec.batch == 1 &&
         spec.query_heads == spec.kv_heads &&
         fa2_head_dim_supported(spec) &&
         spec.numerics == APXINF_ATTENTION_NUMERICS_FLEXIBLE;
}

bool supports_fa2_packed_qkv(const Spec& spec) {
  return spec.semantic == APXINF_ATTENTION_SEMANTIC_DENSE &&
         spec.qkv_layout == APXINF_ATTENTION_QKV_LAYOUT_PACKED_TOKEN &&
         spec.dtype == APXINF_DTYPE_F16 &&
         spec.output_dtype == APXINF_DTYPE_F16 && spec.query_tokens == 256 &&
         spec.query_heads == 16 && spec.kv_heads == 16 &&
         spec.head_dim == 72 &&
         spec.numerics == APXINF_ATTENTION_NUMERICS_FLEXIBLE;
}

bool supports_fa2_splitkv(const Spec& spec) {
  if (std::getenv("APXINF_DISABLE_FA2_SPLITKV") != nullptr) return false;
  return splitkv_shape_supported(spec) &&
         (spec.numerics == APXINF_ATTENTION_NUMERICS_FLEXIBLE ||
          stable_split_count(spec) > 1);
}

void splitkv_configurations(const Spec& spec,
                            std::vector<int>& configurations) {
  if (spec.numerics == APXINF_ATTENTION_NUMERICS_STABLE) {
    const int splits = stable_split_count(spec);
    if (splits > 1) configurations.push_back(splits);
    return;
  }
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
         spec.qkv_layout == APXINF_ATTENTION_QKV_LAYOUT_SEPARATE &&
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
         spec.qkv_layout == APXINF_ATTENTION_QKV_LAYOUT_SEPARATE &&
         spec.mask == APXINF_ATTENTION_MASK_NONE &&
         spec.query_tokens == 256 && spec.key_tokens == 256 &&
         spec.query_heads == 16 && spec.kv_heads == 16 &&
         spec.head_dim == 72 &&
         spec.dtype == APXINF_DTYPE_BF16 &&
         spec.output_dtype == spec.dtype &&
         spec.numerics == APXINF_ATTENTION_NUMERICS_FLEXIBLE;
}

AlignmentRequirements cutlass_alignment(const Spec&) {
  return {16, 16, 16, 16};
}
#endif

#if defined(APXINF_SEGMENTED_ATTENTION_SM110)
bool supports_d256_causal_sm110(const Spec& spec) {
  return spec.semantic == APXINF_ATTENTION_SEMANTIC_DENSE &&
         spec.qkv_layout == APXINF_ATTENTION_QKV_LAYOUT_SEPARATE &&
         spec.dtype == APXINF_DTYPE_BF16 &&
         spec.output_dtype == APXINF_DTYPE_BF16 &&
         spec.mask == APXINF_ATTENTION_MASK_CAUSAL && spec.batch == 1 &&
         spec.query_tokens == 3387 && spec.key_tokens == 3387 &&
         spec.key_capacity == 3387 && spec.query_heads == 16 &&
         spec.kv_heads == 4 && spec.head_dim == 256 &&
         spec.query_start == 0 && spec.scale_is_default != 0 &&
         spec.numerics == APXINF_ATTENTION_NUMERICS_FLEXIBLE;
}

bool supports_d256_kv_split_sm110(const Spec& spec) {
  return spec.semantic == APXINF_ATTENTION_SEMANTIC_KV_CACHE &&
         spec.qkv_layout == APXINF_ATTENTION_QKV_LAYOUT_SEPARATE &&
         spec.dtype == APXINF_DTYPE_BF16 &&
         spec.output_dtype == APXINF_DTYPE_BF16 &&
         spec.mask == APXINF_ATTENTION_MASK_NONE && spec.batch == 1 &&
         spec.query_tokens == 50 && spec.key_tokens >= 3433 &&
         spec.key_tokens <= 3437 && spec.key_capacity == 3438 &&
         spec.query_heads == 16 && spec.kv_heads == 4 &&
         spec.head_dim == 256 && spec.query_start == 0 &&
         spec.dynamic_decode == 0 && spec.scale_is_default != 0 &&
         spec.numerics == APXINF_ATTENTION_NUMERICS_FLEXIBLE;
}

bool supports_segmented_vision_sm110(const Spec& spec) {
  // README 12-frame fixture: each camera contributes three 624-token frames
  // and one 2200-token frame, in camera-major order. Keep the fingerprint in
  // sync with the host_offsets consumed by the fixed-stride launcher below.
  constexpr uint64_t kOffsetsHash = UINT64_C(0x5febecb608d78843);
  return spec.semantic == APXINF_ATTENTION_SEMANTIC_SEGMENTED &&
         spec.qkv_layout == APXINF_ATTENTION_QKV_LAYOUT_SEPARATE &&
         spec.dtype == APXINF_DTYPE_BF16 &&
         spec.output_dtype == APXINF_DTYPE_BF16 &&
         spec.mask == APXINF_ATTENTION_MASK_NONE && spec.batch == 1 &&
         spec.query_tokens == 12216 && spec.key_tokens == 12216 &&
         spec.key_capacity == 12216 && spec.query_heads == 16 &&
         spec.kv_heads == 16 && spec.head_dim == 64 &&
         spec.segments == 12 && spec.max_segment_tokens == 2200 &&
         spec.offsets_hash == kOffsetsHash && spec.scale_is_default != 0 &&
         spec.numerics == APXINF_ATTENTION_NUMERICS_FLEXIBLE;
}

AlignmentRequirements specialized_sm110_alignment(const Spec&) {
  return {16, 16, 16, 16};
}
#endif

}  // namespace

// Test-only metadata checks; no GPU allocation or provider forcing.
extern "C" int apxinf_attention_test_indexed_support_domain() {
#if defined(APXINF_ATTENTION_INDEXED_SM110)
  Spec base{};
  base.semantic=APXINF_ATTENTION_SEMANTIC_DENSE;
  base.qkv_layout=APXINF_ATTENTION_QKV_LAYOUT_SEPARATE;
  base.mask=APXINF_ATTENTION_MASK_NONE;
  base.dtype=base.output_dtype=APXINF_DTYPE_BF16;
  base.numerics=APXINF_ATTENTION_NUMERICS_FLEXIBLE;
  base.batch=1; base.head_dim=128; base.query_heads=base.kv_heads=40;
  base.query_tokens=1785; base.scale_is_default=1;
  for (int keys : {257,512,2665}) {
    base.key_tokens=keys;
    if (!supports_indexed(base) || indexed_resource_requirements(base,0)!=1143844 ||
        indexed_resource_requirements(base,1)!=1143844)
      return 0;
    const auto alignment=indexed_alignment(base);
    if (alignment.query!=16 || alignment.key!=16 || alignment.value!=16 || alignment.output!=16)
      return 0;
  }
  auto rejected=[&](auto mutate){auto s=base;mutate(s);return !supports_indexed(s);};
  if (!rejected([](Spec& s){s.semantic=APXINF_ATTENTION_SEMANTIC_KV_CACHE;}) ||
      !rejected([](Spec& s){s.qkv_layout=APXINF_ATTENTION_QKV_LAYOUT_PACKED_TOKEN;}) ||
      !rejected([](Spec& s){s.mask=APXINF_ATTENTION_MASK_CAUSAL;}) ||
      !rejected([](Spec& s){s.dtype=APXINF_DTYPE_F16;}) ||
      !rejected([](Spec& s){s.output_dtype=APXINF_DTYPE_F16;}) ||
      !rejected([](Spec& s){s.numerics=APXINF_ATTENTION_NUMERICS_STABLE;}) ||
      !rejected([](Spec& s){s.batch=2;}) ||
      !rejected([](Spec& s){s.head_dim=64;}) ||
      !rejected([](Spec& s){s.query_heads=39;}) ||
      !rejected([](Spec& s){s.kv_heads=39;}) ||
      !rejected([](Spec& s){s.query_tokens=1784;}) ||
      !rejected([](Spec& s){s.key_tokens=2664;}) ||
      !rejected([](Spec& s){s.scale_is_default=0;})) return 0;
  std::vector<int> configs;indexed_configurations(base,configs);
  if (configs!=std::vector<int>{0,1}) return 0;
  try {
    indexed_resource_requirements(base,2);
    return 0;
  } catch (const Failure& failure) {
    if (failure.status!=APXINF_STATUS_INVALID_ARGUMENT) return 0;
  }
  // A one-byte-short policy must reject before allocating any device state.
  for(int configuration:{0,1}){
    Execution execution{};execution.spec=base;execution.configuration=configuration;
    execution.resource_limit=1143843;
    try {
      prepare_indexed(execution);
      return 0;
    } catch (const Failure& failure) {
      if (failure.status!=APXINF_STATUS_UNSUPPORTED || execution.provider_state!=nullptr)
        return 0;
    }
  }
  return 2; // Compiled and checked; 1 means absent on this build target.
#else
  return 1;
#endif
}

extern "C" int apxinf_attention_test_efficient_support_domain() {
#if defined(APXINF_ATTENTION_FA2)
  Spec s{};
  s.semantic=APXINF_ATTENTION_SEMANTIC_DENSE;
  s.dtype=s.output_dtype=APXINF_DTYPE_BF16;
  s.mask=APXINF_ATTENTION_MASK_NONE;
  s.batch=1; s.head_dim=384; s.query_heads=s.kv_heads=2;
  s.query_tokens=35; s.key_tokens=137;
  if (!supports_efficient(s)) return 0;
  const int64_t limit=INT32_MAX/(s.query_heads*s.head_dim);
  s.key_tokens=limit;
  if (!supports_efficient(s)) return 0;
  s.key_tokens=limit+1;
  if (supports_efficient(s)) return 0;
  s.key_tokens=137; s.query_tokens=limit;
  if (!supports_efficient(s)) return 0;
  s.query_tokens=limit+1;
  if (supports_efficient(s)) return 0;
  s.query_tokens=35; s.query_heads=3;
  if (supports_efficient(s)) return 0;
#endif
  return 1;
}

const ImplementationRegistry& registry(uint32_t semantic) {
  static const ImplementationRegistry dense_entries = {
#if defined(APXINF_ATTENTION_INDEXED_SM110)
      {3,3,1,"cutlass-indexed-fa2-sm110",apxinf::gemm::kDeviceFeatureFixedSm110,
       true,true,false,supports_indexed,indexed_alignment,indexed_resource_requirements,
       indexed_configurations,prepare_indexed,launch_indexed,destroy_indexed},
#endif
#if defined(APXINF_SEGMENTED_ATTENTION_SM110)
      {kProviderSpecializedSm110, 1, 1, "specialized-d256-causal-sm110",
       apxinf::gemm::kDeviceFeatureFixedSm110, true, true, false,
       supports_d256_causal_sm110, specialized_sm110_alignment,
       d256_causal_sm110_resource_requirements, one_configuration,
       prepare_d256_causal_sm110, launch_d256_causal_sm110,
       destroy_d256_causal_sm110},
#endif
#if defined(APXINF_ATTENTION_CUTLASS)
      {kProviderCutlass, 1, 1, "cutlass-fmha-sm100",
       apxinf::gemm::kDeviceFeatureCutlassSm100, true, true, false,
       supports_cutlass, cutlass_alignment, cutlass_resource_requirements,
       one_configuration, prepare_cutlass, launch_cutlass, destroy_cutlass},
#endif
#if defined(APXINF_ATTENTION_FA2)
      {kProviderFa2, 4, 1, "flash-attention-2-packed-qkv",
       apxinf::gemm::kDeviceFeatureCutlassSm100, true, true, true,
       supports_fa2_packed_qkv, fa2_alignment, fa2_resource_requirements,
       one_configuration, prepare_fa2, launch_fa2, destroy_fa2},
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
#endif
#if defined(APXINF_ATTENTION_FA2)
      {5, 1, 1, "cutlass-memory-efficient-attention", apxinf::gemm::kDeviceFeatureFa2,
       true, true, true, supports_efficient, fa2_alignment,
       efficient_resource_requirements, one_configuration,
       prepare_efficient, launch_efficient, destroy_efficient},
#endif
      {kProviderCublas, 1, 1, "cublas-f16-gqa-stable", 0, true, true, true,
       supports_cublas_f16, natural_alignment,
       cublas_f16_resource_requirements, one_configuration,
       prepare_cublas_f16, launch_cublas_f16, destroy_cublas_f16},
      {kProviderCustom, 1, 1, "custom-attention-fallback", 0, true, true,
       true, supports_dense_custom, natural_alignment,
       custom_resource_requirements, one_configuration, prepare_custom,
       launch_custom, destroy_custom},
#if defined(APXINF_ATTENTION_FA2)
      // Keep the existing custom cold-cache baseline when it fits. Large
      // sequences can exceed its quadratic workspace budget; FA2 is then a
      // deterministic, graph-safe fallback without requiring online tuning.
      {kProviderFa2, 1, 1, "flash-attention-2",
       apxinf::gemm::kDeviceFeatureFa2, true, true, true, supports_fa2,
       fa2_alignment, fa2_resource_requirements, one_configuration,
       prepare_fa2, launch_fa2, destroy_fa2},
#endif
  };
  static const ImplementationRegistry kv_cache_entries = {
#if defined(APXINF_SEGMENTED_ATTENTION_SM110)
      {kProviderSpecializedSm110, 2, 1, "specialized-d256-kv-split-sm110",
       apxinf::gemm::kDeviceFeatureFixedSm110, true, true, false,
       supports_d256_kv_split_sm110, specialized_sm110_alignment,
       d256_kv_split_sm110_resource_requirements, one_configuration,
       prepare_d256_kv_split_sm110, launch_d256_kv_split_sm110,
       destroy_d256_kv_split_sm110},
#endif
#if defined(APXINF_ATTENTION_FA2)
      {kProviderFa2, 3, 1, "flash-attention-2-split-kv",
       apxinf::gemm::kDeviceFeatureFa2, true, true, false,
       supports_fa2_splitkv, fa2_alignment,
       fa2_splitkv_resource_requirements, splitkv_configurations,
       prepare_fa2_splitkv, launch_fa2_splitkv, destroy_fa2_splitkv},
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
#if defined(APXINF_SEGMENTED_ATTENTION_SM110)
      {kProviderSpecializedSm110, 3, 1, "specialized-segmented-vision-sm110",
       apxinf::gemm::kDeviceFeatureFixedSm110, true, true, false,
       supports_segmented_vision_sm110, specialized_sm110_alignment,
       segmented_vision_sm110_resource_requirements, one_configuration,
       prepare_segmented_vision_sm110, launch_segmented_vision_sm110,
       destroy_segmented_vision_sm110},
#endif
#if defined(APXINF_ATTENTION_FA2)
      {kProviderFa2, 5, 1, "flash-attention-2-segmented",
       apxinf::gemm::kDeviceFeatureFa2, true, true, true,
       supports_fa2_segmented, fa2_alignment,
       fa2_segmented_resource_requirements, one_configuration,
       prepare_fa2_segmented, launch_fa2_segmented,
       destroy_fa2_segmented},
#endif
      {kProviderCustom, 3, 1, "custom-segmented-attention", 0, true, true,
       true, supports_segmented_custom, natural_alignment,
       custom_resource_requirements, one_configuration, prepare_custom,
       launch_custom, destroy_custom},
  };
  switch (semantic) {
    case APXINF_ATTENTION_SEMANTIC_ROUNDED_ADDITIVE: {
      static const ImplementationRegistry rounded_entries = {
          {4, 1, 1, "cublas-rounded-additive-attention", 0, true, true, true,
           supports_rounded, natural_alignment, rounded_resource_requirements,
           one_configuration, prepare_rounded, launch_rounded, destroy_rounded},
      };
      return rounded_entries;
    }
    case APXINF_ATTENTION_SEMANTIC_DENSE:
      return dense_entries;
    case APXINF_ATTENTION_SEMANTIC_KV_CACHE:
      return kv_cache_entries;
    case APXINF_ATTENTION_SEMANTIC_SEGMENTED:
      return segmented_entries;
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
