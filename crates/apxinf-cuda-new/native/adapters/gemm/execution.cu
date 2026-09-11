#include "internal.h"

#include <cmath>
#include <limits>

namespace {

using apxinf::gemm::Failure;
using apxinf::gemm::Recipe;
using apxinf::gemm::Execution;
bool valid_alignment_class(uint32_t alignment) {
  return alignment <= 256 &&
         (alignment == 0 || (alignment & (alignment - 1)) == 0);
}

void validate_recorded_alignment(const void* pointer, uint32_t alignment,
                                 const char* name) {
  if (pointer != nullptr &&
      (alignment == 0 ||
       reinterpret_cast<uintptr_t>(pointer) % alignment != 0)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  std::string(name) + " does not satisfy execution alignment");
  }
}

void validate_spec(const apxinf::gemm::Spec& spec) {
  if (spec.version != 4 || spec.semantic > APXINF_GEMM_SEMANTIC_GEMM_BIAS ||
      spec.a_dtype > APXINF_DTYPE_I8 || spec.b_dtype > APXINF_DTYPE_I8 ||
      spec.accumulation_dtype > APXINF_DTYPE_I32 ||
      spec.output_dtype > APXINF_DTYPE_E4M3 ||
      spec.quantization > APXINF_GEMM_QUANT_W8A8_ROW_CHANNEL || spec.m <= 0 ||
      spec.n <= 0 || spec.k <= 0 || spec.m > INT32_MAX ||
      spec.n > INT32_MAX || spec.k > INT32_MAX ||
      spec.alpha_is_unit > 1 || spec.output_scale_is_unit > 1 ||
      spec.b_is_immutable > 1) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid GEMM Spec");
  }
  for (uint32_t alignment : {
           spec.a_alignment, spec.b_alignment, spec.bias_alignment,
           spec.a_scales_alignment, spec.b_scales_alignment,
           spec.output_alignment}) {
    if (!valid_alignment_class(alignment)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid GEMM binding alignment class");
    }
  }
  if (spec.a_alignment < apxinf::gemm::dtype_bytes(spec.a_dtype) ||
      spec.b_alignment < apxinf::gemm::dtype_bytes(spec.b_dtype) ||
      spec.output_alignment < apxinf::gemm::dtype_bytes(spec.output_dtype)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "GEMM binding violates dtype alignment");
  }
  if ((spec.a_dtype == APXINF_DTYPE_I8 ||
       spec.b_dtype == APXINF_DTYPE_I8) &&
      (spec.a_dtype != APXINF_DTYPE_I8 ||
       spec.b_dtype != APXINF_DTYPE_I8 ||
       spec.accumulation_dtype != APXINF_DTYPE_I32 ||
       spec.quantization != APXINF_GEMM_QUANT_W8A8_ROW_CHANNEL ||
       spec.output_dtype != APXINF_DTYPE_BF16 || spec.k > 131071 ||
       (spec.semantic != APXINF_GEMM_SEMANTIC_GEMM &&
        spec.semantic != APXINF_GEMM_SEMANTIC_GEMM_BIAS))) {
    throw Failure(APXINF_STATUS_UNSUPPORTED, "invalid INT8 GEMM contract");
  }
  if (spec.a_dtype != APXINF_DTYPE_I8 &&
      spec.accumulation_dtype != APXINF_DTYPE_F32) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "floating-point GEMM requires F32 accumulation");
  }
  if ((spec.quantization == APXINF_GEMM_QUANT_FP8_UNIT_SCALE ||
       spec.quantization == APXINF_GEMM_QUANT_FP8_ROW_CHANNEL) &&
      (spec.a_dtype != APXINF_DTYPE_E4M3 ||
       spec.b_dtype != APXINF_DTYPE_E4M3)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "FP8 quantization requires two E4M3 inputs");
  }
  if (spec.quantization == APXINF_GEMM_QUANT_NONE &&
      (spec.a_dtype == APXINF_DTYPE_E4M3 ||
       spec.a_dtype == APXINF_DTYPE_I8 ||
       spec.a_dtype != spec.b_dtype)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "plain GEMM requires matching non-quantized inputs");
  }
  if (spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_GEGLU && spec.n % 2 != 0) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "GEMM+GeGLU requires an even projection width");
  }
  if (apxinf::gemm::has_row_channel_scales(spec) &&
      spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_GEGLU) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "rowwise GEMM+GeGLU is not implemented");
  }
  constexpr int64_t kMaximumElementBytes = 4;
  if (spec.m > INT64_MAX / spec.n / kMaximumElementBytes ||
      spec.m > INT64_MAX / spec.k / kMaximumElementBytes ||
      spec.k > INT64_MAX / spec.n / kMaximumElementBytes) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "GEMM size overflow");
  }
}

void validate_policy(const apxinf_gemm_policy_t& policy) {
  if (policy.online_tune > 1 || policy.allow_fallback > 1 ||
      policy.graph_safe > 1 || policy.deterministic > 1) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid GEMM Policy");
  }
}

void validate_bindings(const apxinf::gemm::Spec& spec,
                       const apxinf_gemm_bindings_t& bindings,
                       bool require_output) {
  const bool needs_bias =
      spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_BIAS ||
      spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_BIAS_GELU;
  const bool needs_scales =
      apxinf::gemm::has_row_channel_scales(spec);
  if (bindings.a == nullptr || bindings.b == nullptr ||
      (require_output && bindings.output == nullptr) ||
      (needs_bias && bindings.bias == nullptr) ||
      (needs_scales &&
       (bindings.a_scales == nullptr || bindings.b_scales == nullptr))) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "missing required GEMM bindings");
  }
  if (!needs_bias && bindings.bias != nullptr) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "unexpected GEMM bias");
  }
  if (!std::isfinite(bindings.alpha) ||
      !std::isfinite(bindings.output_scale) ||
      bindings.output_scale <= 0.0F) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid GEMM scale binding");
  }
  // The Spec only records whether each scale is unit. A binding that
  // contradicts that predicate would silently execute on a candidate selected
  // for the other case, so it is rejected instead.
  if ((bindings.alpha == 1.0F) != (spec.alpha_is_unit != 0) ||
      (bindings.output_scale == 1.0F) != (spec.output_scale_is_unit != 0)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "GEMM scale binding contradicts the Spec scale predicate");
  }
  if (bindings.b_is_immutable > 1 ||
      bindings.b_is_immutable != spec.b_is_immutable ||
      (bindings.b_is_immutable == 0 && bindings.b_version != 0)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "invalid GEMM immutable-weight identity");
  }
  validate_recorded_alignment(bindings.a, spec.a_alignment, "GEMM A binding");
  validate_recorded_alignment(bindings.b, spec.b_alignment, "GEMM B binding");
  validate_recorded_alignment(bindings.bias, spec.bias_alignment,
                              "GEMM bias binding");
  validate_recorded_alignment(bindings.a_scales, spec.a_scales_alignment,
                              "GEMM A scales binding");
  validate_recorded_alignment(bindings.b_scales, spec.b_scales_alignment,
                              "GEMM B scales binding");
  validate_recorded_alignment(bindings.output, spec.output_alignment,
                              "GEMM output binding");
}

const apxinf::gemm::Implementation* find_implementation(const Recipe& recipe,
                                                        const apxinf::gemm::Spec& spec,
                                                        int device) {
  for (const auto& implementation :
       apxinf::gemm::registry(spec.semantic)) {
    if (implementation.provider_id == recipe.provider_id &&
        implementation.implementation_id == recipe.implementation_id &&
        implementation.implementation_version ==
            recipe.implementation_version &&
        apxinf::gemm::supports_device(implementation, device) &&
        implementation.supports(spec)) {
      std::vector<int> configurations;
      implementation.enumerate_configs(spec, configurations);
      if (std::find(configurations.begin(), configurations.end(),
                    recipe.configuration) != configurations.end()) {
        return &implementation;
      }
    }
  }
  return nullptr;
}

void warmup(Execution& execution);

std::unique_ptr<Execution> fallback(
    const apxinf::gemm::Spec& spec, const apxinf_gemm_policy_t& policy,
    const apxinf_gemm_bindings_t& bindings, int device) {
  for (const auto& implementation :
       apxinf::gemm::registry(spec.semantic)) {
    if (!apxinf::gemm::supports_device(implementation, device) ||
        !implementation.supports(spec) ||
        (policy.graph_safe && !implementation.graph_safe) ||
        (policy.deterministic && !implementation.deterministic)) {
      continue;
    }
    std::vector<int> configurations;
    implementation.enumerate_configs(spec, configurations);
    for (int configuration : configurations) {
      try {
        auto execution = apxinf::gemm::prepare(
            implementation, configuration, spec, policy, bindings, device);
        warmup(*execution);
        return execution;
      } catch (const Failure&) {
        cudaGetLastError();
      }
    }
  }
  throw Failure(APXINF_STATUS_UNSUPPORTED,
                "no GEMM fallback satisfies the Spec and Policy");
}

void warmup(Execution& execution) {
  // An Execution is immutable with respect to its bindings after prepare.
  // Validate the exact serving addresses instead of substituting an aligned
  // scratch output that a provider may not have prepared for.
  apxinf::gemm::check_cuda(execution.implementation->enqueue(execution));
  apxinf::gemm::check_cuda(cudaStreamSynchronize(
      static_cast<cudaStream_t>(execution.bindings.stream)));
}

}  // namespace

extern "C" apxinf_status_t apxinf_gemm_prepare(
    apxinf_runtime_t runtime,
    const apxinf_gemm_spec_t* spec,
    const apxinf_gemm_policy_t* policy,
    const apxinf_gemm_tuning_bindings_t* tuning_bindings,
    apxinf_gemm_execution_t* output) {
  if (output != nullptr) {
    *output = nullptr;
  }
  return apxinf::gemm::abi_boundary([&] {
    if (runtime == nullptr || spec == nullptr || policy == nullptr ||
        tuning_bindings == nullptr || output == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "null GEMM execution argument");
    }
    apxinf::gemm::Spec normalized_spec{};
    static_cast<apxinf_gemm_spec_t&>(normalized_spec) = *spec;
    validate_spec(normalized_spec);
    validate_policy(*policy);
    validate_bindings(normalized_spec, tuning_bindings->execution, true);
    if (tuning_bindings->reference_kind >
        APXINF_GEMM_REFERENCE_TORCH_OUTPUT) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid GEMM reference kind");
    }
    if (tuning_bindings->reference_kind ==
        APXINF_GEMM_REFERENCE_TORCH_OUTPUT) {
      const uint64_t output_width =
          normalized_spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_GEGLU
              ? static_cast<uint64_t>(normalized_spec.n / 2)
              : static_cast<uint64_t>(normalized_spec.n);
      const uint64_t output_count =
          static_cast<uint64_t>(normalized_spec.m) * output_width;
      if (tuning_bindings->expected_output == nullptr ||
          tuning_bindings->expected_output_len != output_count) {
        throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                      "invalid Torch GEMM reference output");
      }
    } else if (tuning_bindings->expected_output != nullptr ||
               tuning_bindings->expected_output_len != 0) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "unexpected Torch GEMM reference output");
    }
    cudaStreamCaptureStatus capture_status;
    apxinf::gemm::check_cuda(cudaStreamIsCapturing(
        static_cast<cudaStream_t>(tuning_bindings->execution.stream),
        &capture_status));
    if (capture_status != cudaStreamCaptureStatusNone) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "prepare GEMM executions before graph capture");
    }

    apxinf::gemm::check_cuda(cudaSetDevice(runtime->device));
    const auto keys =
        apxinf::gemm::tuning_keys(normalized_spec, *policy, runtime->device);
    const std::string& key = keys.performance;
    std::lock_guard<std::mutex> lock(runtime->gemm_mutex);

    Recipe recipe{};
    bool recipe_found = false;
    Recipe compatible_hint{};
    bool compatible_hint_found = false;
    std::string source = "recipe";
    if (const auto cached = runtime->gemm_recipes.find(key);
        cached != runtime->gemm_recipes.end()) {
      recipe = cached->second;
      recipe_found = true;
      source = "memory-recipe";
    } else {
      const std::string serialized = apxinf::gemm::read_recipe(
          policy->cache_dir != nullptr ? policy->cache_dir : "", key);
      std::istringstream input(serialized);
      recipe_found = static_cast<bool>(
          input >> recipe.provider_id >> recipe.implementation_id >>
          recipe.implementation_version >> recipe.configuration);
    }

    // A compatibility hint may come from a device with a different
    // performance profile. It is never restored as a fully tuned recipe: it is
    // only moved to the front of the next complete tuning pass.
    if (!recipe_found) {
      const std::string serialized = apxinf::gemm::read_recipe(
          policy->cache_dir != nullptr ? policy->cache_dir : "",
          keys.compatible_hint);
      std::istringstream input(serialized);
      compatible_hint_found = static_cast<bool>(
          input >> compatible_hint.provider_id >>
          compatible_hint.implementation_id >>
          compatible_hint.implementation_version >>
          compatible_hint.configuration);
      if (compatible_hint_found &&
          find_implementation(compatible_hint, normalized_spec,
                              runtime->device) == nullptr) {
        compatible_hint_found = false;
      }
    }

    std::unique_ptr<Execution> execution;
    if (recipe_found) {
      if (const auto* implementation =
              find_implementation(recipe, normalized_spec, runtime->device)) {
        try {
          execution = apxinf::gemm::prepare(
              *implementation, recipe.configuration, normalized_spec, *policy,
              tuning_bindings->execution, runtime->device);
          warmup(*execution);
        } catch (const Failure&) {
          execution.reset();
          cudaGetLastError();
        }
        if (execution != nullptr) {
          runtime->gemm_recipes[key] = recipe;
        }
      }
      if (execution == nullptr) {
        compatible_hint = recipe;
        compatible_hint_found = true;
      }
    }

    bool persist_recipe = false;
    if (execution == nullptr) {
      if (policy->online_tune) {
        const Recipe tuned_recipe = apxinf::gemm::tune(
            normalized_spec, *policy, *tuning_bindings, runtime->device,
            source, compatible_hint_found ? &compatible_hint : nullptr);
        const auto* implementation =
            find_implementation(tuned_recipe, normalized_spec, runtime->device);
        if (implementation == nullptr) {
          throw Failure(APXINF_STATUS_INTERNAL_ERROR,
                        "tuned GEMM Recipe is not restorable");
        }
        try {
          execution = apxinf::gemm::prepare(
              *implementation, tuned_recipe.configuration, normalized_spec,
              *policy, tuning_bindings->execution, runtime->device);
          warmup(*execution);
          recipe = tuned_recipe;
          persist_recipe = true;
        } catch (const Failure&) {
          execution.reset();
          cudaGetLastError();
          if (!policy->allow_fallback) {
            throw;
          }
          execution = fallback(normalized_spec, *policy,
                               tuning_bindings->execution, runtime->device);
          source = "fallback-after-tune";
        }
      } else if (policy->allow_fallback) {
        execution = fallback(normalized_spec, *policy,
                             tuning_bindings->execution, runtime->device);
        source = "fallback";
      } else {
        throw Failure(APXINF_STATUS_CACHE_MISS,
                      "GEMM recipe miss and tuning/fallback are disabled");
      }

      if (!persist_recipe) {
        recipe = {execution->implementation->provider_id,
                  execution->implementation->implementation_id,
                  execution->implementation->implementation_version,
                  execution->configuration};
      }
      if (persist_recipe) {
        runtime->gemm_recipes[key] = recipe;
        std::ostringstream serialized;
        serialized << recipe.provider_id << ' ' << recipe.implementation_id
                   << ' ' << recipe.implementation_version << ' '
                   << recipe.configuration;
        apxinf::gemm::write_recipe(
            policy->cache_dir != nullptr ? policy->cache_dir : "", key,
            serialized.str());
        apxinf::gemm::write_recipe(
            policy->cache_dir != nullptr ? policy->cache_dir : "",
            keys.compatible_hint, serialized.str());
      }
    }

    execution->summary =
        std::string(execution->implementation->name) +
        " config=" + std::to_string(execution->configuration) +
        " workspace=" + std::to_string(execution->resource_bytes) +
        " source=" + source;
    *output = reinterpret_cast<apxinf_gemm_execution_t>(execution.release());
  });
}

extern "C" apxinf_status_t apxinf_gemm_enqueue(
    apxinf_gemm_execution_t handle) {
  return apxinf::gemm::abi_boundary([&] {
    auto* execution = reinterpret_cast<Execution*>(handle);
    if (execution == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "null GEMM execution");
    }
    apxinf::gemm::check_cuda(cudaSetDevice(execution->device));
    apxinf::gemm::check_cuda(execution->implementation->enqueue(*execution));
  });
}

extern "C" void apxinf_gemm_destroy(apxinf_gemm_execution_t handle) {
  delete reinterpret_cast<Execution*>(handle);
}

extern "C" const char* apxinf_gemm_summary(
    apxinf_gemm_execution_t handle) {
  const auto* execution = reinterpret_cast<const Execution*>(handle);
  return execution != nullptr ? execution->summary.c_str()
                              : "null GEMM execution";
}

extern "C" uint64_t apxinf_gemm_execution_weight_prepack_count(
    apxinf_gemm_execution_t handle) {
  const auto* execution = reinterpret_cast<const Execution*>(handle);
  if (execution == nullptr || execution->implementation->provider_id != 3) {
    return 0;
  }
  return apxinf::gemm::cutlass_weight_prepack_count(*execution);
}

// Private test hook used to select a provider without running the very large
// provider-independent CPU reference required by production GeGLU shapes.
extern "C" apxinf_status_t apxinf_gemm_test_seed_recipe(
    const apxinf_gemm_spec_t* spec,
    const apxinf_gemm_policy_t* policy,
    int device,
    uint32_t provider_id,
    uint32_t implementation_id,
    uint32_t implementation_version,
    int32_t configuration) {
  return apxinf::gemm::abi_boundary([&] {
    if (spec == nullptr || policy == nullptr || policy->cache_dir == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid test recipe seed arguments");
    }
    apxinf::gemm::Spec normalized{};
    static_cast<apxinf_gemm_spec_t&>(normalized) = *spec;
    validate_spec(normalized);
    const Recipe recipe{provider_id, implementation_id,
                        implementation_version, configuration};
    if (find_implementation(recipe, normalized, device) == nullptr) {
      throw Failure(APXINF_STATUS_UNSUPPORTED,
                    "test recipe candidate is unavailable");
    }
    const auto keys =
        apxinf::gemm::tuning_keys(normalized, *policy, device);
    std::ostringstream serialized;
    serialized << provider_id << ' ' << implementation_id << ' '
               << implementation_version << ' ' << configuration;
    apxinf::gemm::write_recipe(policy->cache_dir, keys.performance,
                               serialized.str());
  });
}
