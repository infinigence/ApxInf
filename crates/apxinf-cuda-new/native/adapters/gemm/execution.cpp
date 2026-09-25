#include "internal.h"

#ifdef APXINF_GEMM_CUTLASS
#include "../../kernels/cutlass/ops/gemm/gemm_nvfp4_sm100.h"
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
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
  if (spec.version != 5 || spec.semantic > APXINF_GEMM_SEMANTIC_GEMM_BIAS ||
      spec.a_dtype > APXINF_DTYPE_E2M1_PAIR ||
      spec.b_dtype > APXINF_DTYPE_E2M1_PAIR ||
      spec.accumulation_dtype > APXINF_DTYPE_I32 ||
      spec.output_dtype > APXINF_DTYPE_E4M3 ||
      spec.quantization > APXINF_GEMM_QUANT_NVFP4_BLOCK || spec.m <= 0 ||
      spec.n <= 0 || spec.k <= 0 || spec.m > INT32_MAX ||
      spec.n > INT32_MAX || spec.k > INT32_MAX ||
      spec.alpha_is_unit > 1 || spec.output_scale_is_unit > 1 ||
      spec.b_is_immutable > 1) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid GEMM Spec");
  }
  for (uint32_t alignment : {
           spec.a_alignment, spec.b_alignment, spec.bias_alignment,
           spec.a_scales_alignment, spec.b_scales_alignment,
           spec.a_block_scales_alignment, spec.b_block_scales_alignment,
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
       spec.a_dtype == APXINF_DTYPE_E2M1_PAIR ||
       spec.a_dtype != spec.b_dtype)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "plain GEMM requires matching non-quantized inputs");
  }
  if (spec.quantization == APXINF_GEMM_QUANT_NVFP4_BLOCK) {
    if (spec.a_dtype != APXINF_DTYPE_E2M1_PAIR ||
        spec.b_dtype != APXINF_DTYPE_E2M1_PAIR ||
        spec.semantic != APXINF_GEMM_SEMANTIC_GEMM) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "NVFP4 quantization requires two packed E2M1 inputs");
    }
    if (spec.sf_vec_size == 0 || spec.k % spec.sf_vec_size != 0) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "NVFP4 K must be a multiple of the block size");
    }
  } else if (spec.a_dtype == APXINF_DTYPE_E2M1_PAIR ||
             spec.b_dtype == APXINF_DTYPE_E2M1_PAIR ||
             spec.sf_vec_size != 0) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "packed E2M1 operands require the NVFP4 contract");
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
  const bool needs_block_scales = apxinf::gemm::has_block_scales(spec);
  if (bindings.a == nullptr || bindings.b == nullptr ||
      (require_output && bindings.output == nullptr) ||
      (needs_bias && bindings.bias == nullptr) ||
      (needs_scales &&
       (bindings.a_scales == nullptr || bindings.b_scales == nullptr)) ||
      (needs_block_scales && (bindings.a_block_scales == nullptr ||
                              bindings.b_block_scales == nullptr))) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "missing required GEMM bindings");
  }
  if (!needs_block_scales && (bindings.a_block_scales != nullptr ||
                              bindings.b_block_scales != nullptr)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "unexpected GEMM block scales");
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
  validate_recorded_alignment(bindings.a_block_scales,
                              spec.a_block_scales_alignment,
                              "GEMM A block scales binding");
  validate_recorded_alignment(bindings.b_block_scales,
                              spec.b_block_scales_alignment,
                              "GEMM B block scales binding");
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

std::unique_ptr<Execution> fallback(
    const apxinf::gemm::Spec& spec, const apxinf_gemm_policy_t& policy,
    const apxinf_gemm_bindings_t& bindings, int device) {
  for (const auto& implementation :
       apxinf::gemm::registry(spec.semantic)) {
    if (!implementation.fallback ||
        !apxinf::gemm::supports_device(implementation, device) ||
        !implementation.supports(spec) ||
        (policy.graph_safe && !implementation.graph_safe) ||
        (policy.deterministic && !implementation.deterministic)) {
      continue;
    }
    std::vector<int> configurations;
    implementation.enumerate_configs(spec, configurations);
    for (int configuration : configurations) {
      try {
        return apxinf::gemm::prepare(
            implementation, configuration, spec, policy, bindings, device);
      } catch (const Failure&) {
        cudaGetLastError();
      }
    }
  }
  throw Failure(APXINF_STATUS_UNSUPPORTED,
                "no GEMM fallback satisfies the Spec and Policy");
}

}  // namespace

extern "C" apxinf_status_t apxinf_gemm_prepare(
    apxinf_runtime_t runtime,
    const apxinf_gemm_spec_t* spec,
    const apxinf_gemm_policy_t* policy,
    const apxinf_gemm_bindings_t* bindings,
    apxinf_gemm_execution_t* output) {
  if (output != nullptr) {
    *output = nullptr;
  }
  return apxinf::gemm::abi_boundary([&] {
    if (runtime == nullptr || spec == nullptr || policy == nullptr ||
        bindings == nullptr || output == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "null GEMM execution argument");
    }
    apxinf::gemm::Spec normalized_spec{};
    static_cast<apxinf_gemm_spec_t&>(normalized_spec) = *spec;
    validate_spec(normalized_spec);
    validate_policy(*policy);
    validate_bindings(normalized_spec, *bindings, true);
    cudaStreamCaptureStatus capture_status;
    apxinf::gemm::check_cuda(cudaStreamIsCapturing(
        static_cast<cudaStream_t>(bindings->stream),
        &capture_status));
    if (capture_status != cudaStreamCaptureStatusNone) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "prepare GEMM executions before graph capture");
    }

    apxinf::gemm::check_cuda(cudaSetDevice(runtime->device));
    const auto keys =
        apxinf::gemm::tuning_keys(normalized_spec, *policy, runtime->device);
    const std::string& key = keys.key;
    std::lock_guard<std::mutex> lock(runtime->gemm_mutex);

    Recipe recipe{};
    bool recipe_found = false;
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
    if (recipe_found) {
      input >> std::ws;
      recipe_found = input.eof();
    }
    }

    std::unique_ptr<Execution> execution;
    if (recipe_found) {
      if (const auto* implementation =
              find_implementation(recipe, normalized_spec, runtime->device)) {
        try {
          execution = apxinf::gemm::prepare(
              *implementation, recipe.configuration, normalized_spec, *policy,
              *bindings, runtime->device);
        } catch (const Failure&) {
          execution.reset();
          cudaGetLastError();
        }
        if (execution != nullptr) {
          runtime->gemm_recipes[key] = recipe;
        }
      }
    }

    bool persist_recipe = false;
    if (execution == nullptr) {
      if (policy->online_tune) {
        const Recipe tuned_recipe = apxinf::gemm::tune(
            normalized_spec, *policy, *bindings, runtime->device,
            source);
        const auto* implementation =
            find_implementation(tuned_recipe, normalized_spec, runtime->device);
        if (implementation == nullptr) {
          throw Failure(APXINF_STATUS_INTERNAL_ERROR,
                        "tuned GEMM Recipe is not restorable");
        }
        try {
          execution = apxinf::gemm::prepare(
              *implementation, tuned_recipe.configuration, normalized_spec,
              *policy, *bindings, runtime->device);
          recipe = tuned_recipe;
          persist_recipe = true;
        } catch (const Failure&) {
          execution.reset();
          cudaGetLastError();
          if (!policy->allow_fallback) {
            throw;
          }
          execution = fallback(normalized_spec, *policy, *bindings,
                               runtime->device);
          source = "fallback-after-tune";
        }
      } else if (policy->allow_fallback) {
        execution = fallback(normalized_spec, *policy, *bindings,
                             runtime->device);
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
      }
    }

    execution->summary =
        std::string(execution->implementation->name) +
        " config=" + std::to_string(execution->configuration) +
        " workspace=" + std::to_string(execution->resource_bytes) +
        " source=" + source;
    // The per-attempt autotune report is otherwise only reachable from unit
    // tests inside the library crate, which is exactly where an integration
    // run cannot see it. APXINF_GEMM_REPORT=1 puts it on stderr.
    static const bool report = std::getenv("APXINF_GEMM_REPORT") != nullptr;
    if (report) {
      std::fprintf(stderr,
                   "[apxinf-gemm] sem=%u m=%lld n=%lld k=%lld a=%u b=%u "
                   "out=%u quant=%u alpha_unit=%u | %s\n",
                   normalized_spec.semantic,
                   static_cast<long long>(normalized_spec.m),
                   static_cast<long long>(normalized_spec.n),
                   static_cast<long long>(normalized_spec.k),
                   normalized_spec.a_dtype, normalized_spec.b_dtype,
                   normalized_spec.output_dtype, normalized_spec.quantization,
                   normalized_spec.alpha_is_unit,
                   execution->summary.c_str());
    }
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

extern "C" uint64_t apxinf_gemm_nvfp4_scale_buffer_bytes(int64_t rows,
                                                         int64_t k,
                                                         uint32_t sf_vec_size) {
#ifdef APXINF_GEMM_CUTLASS
  if (rows <= 0 || k <= 0 || rows > INT32_MAX || k > INT32_MAX) return 0;
  return apxinf::cuda::cutlass_ops::nvfp4_scale_buffer_bytes(
      static_cast<int>(rows), static_cast<int>(k),
      static_cast<int>(sf_vec_size));
#else
  (void)rows;
  (void)k;
  (void)sf_vec_size;
  return 0;
#endif
}

extern "C" apxinf_status_t apxinf_gemm_nvfp4_pack_block_scales(
    const void* source_row_major, void* destination, int64_t rows, int64_t k,
    uint32_t sf_vec_size, apxinf_cuda_stream_t stream) {
#ifdef APXINF_GEMM_CUTLASS
  return apxinf::framework::abi_boundary([&] {
    if (source_row_major == nullptr || destination == nullptr || rows <= 0 ||
        k <= 0 || rows > INT32_MAX || k > INT32_MAX) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid NVFP4 block-scale packing arguments");
    }
    const int status = apxinf::cuda::cutlass_ops::nvfp4_scatter_block_scales(
        source_row_major, destination, static_cast<int>(rows),
        static_cast<int>(k), static_cast<int>(sf_vec_size),
        static_cast<cudaStream_t>(stream));
    if (status != 0) {
      throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                    "NVFP4 block-scale packing failed with status " +
                        std::to_string(status));
    }
  });
#else
  (void)source_row_major;
  (void)destination;
  (void)rows;
  (void)k;
  (void)sf_vec_size;
  (void)stream;
  return APXINF_STATUS_UNSUPPORTED;
#endif
}

extern "C" apxinf_status_t apxinf_gemm_nvfp4_quantize_activation(
    const void* source_bf16, void* destination_packed,
    void* destination_scales, int64_t rows, int64_t k, uint32_t sf_vec_size,
    float input_scale, int32_t row_major_scales,
    apxinf_cuda_stream_t stream) {
#ifdef APXINF_GEMM_CUTLASS
  return apxinf::framework::abi_boundary([&] {
    if (source_bf16 == nullptr || destination_packed == nullptr ||
        destination_scales == nullptr || rows <= 0 || k <= 0 ||
        rows > INT32_MAX || k > INT32_MAX) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid NVFP4 activation quantization arguments");
    }
    const int status = apxinf::cuda::cutlass_ops::nvfp4_quantize_activation(
        source_bf16, destination_packed, destination_scales,
        static_cast<int>(rows), static_cast<int>(k),
        static_cast<int>(sf_vec_size), input_scale, row_major_scales,
        static_cast<cudaStream_t>(stream));
    if (status != 0) {
      throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                    "NVFP4 activation quantization failed with status " +
                        std::to_string(status));
    }
  });
#else
  (void)source_bf16;
  (void)destination_packed;
  (void)destination_scales;
  (void)rows;
  (void)k;
  (void)sf_vec_size;
  (void)input_scale;
  (void)row_major_scales;
  (void)stream;
  return APXINF_STATUS_UNSUPPORTED;
#endif
}

extern "C" apxinf_status_t apxinf_gemm_nvfp4_quantize_rms_norm(
    const void* source_bf16, const void* norm_weight,
    void* destination_packed, void* destination_scales, int64_t rows,
    int64_t k, uint32_t sf_vec_size, float epsilon, float input_scale,
    int32_t row_major_scales, apxinf_cuda_stream_t stream) {
#ifdef APXINF_GEMM_CUTLASS
  return apxinf::framework::abi_boundary([&] {
    if (source_bf16 == nullptr || norm_weight == nullptr ||
        destination_packed == nullptr || destination_scales == nullptr ||
        rows <= 0 || k <= 0 || rows > INT32_MAX || k > INT32_MAX) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid fused RMSNorm quantization arguments");
    }
    const int status = apxinf::cuda::cutlass_ops::nvfp4_quantize_rms_norm(
        source_bf16, norm_weight, destination_packed, destination_scales,
        static_cast<int>(rows), static_cast<int>(k),
        static_cast<int>(sf_vec_size), epsilon, input_scale, row_major_scales,
        static_cast<cudaStream_t>(stream));
    if (status != 0) {
      throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                    "fused RMSNorm quantization failed with status " +
                        std::to_string(status));
    }
  });
#else
  (void)source_bf16; (void)norm_weight; (void)destination_packed;
  (void)destination_scales; (void)rows; (void)k; (void)sf_vec_size;
  (void)epsilon; (void)input_scale; (void)row_major_scales; (void)stream;
  return APXINF_STATUS_UNSUPPORTED;
#endif
}

extern "C" apxinf_status_t apxinf_gemm_nvfp4_quantize_swiglu(
    const void* source_bf16, void* destination_packed,
    void* destination_scales, int64_t rows, int64_t k, uint32_t sf_vec_size,
    float input_scale, int32_t row_major_scales,
    apxinf_cuda_stream_t stream) {
#ifdef APXINF_GEMM_CUTLASS
  return apxinf::framework::abi_boundary([&] {
    if (source_bf16 == nullptr || destination_packed == nullptr ||
        destination_scales == nullptr || rows <= 0 || k <= 0 ||
        rows > INT32_MAX || k > INT32_MAX) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid fused SwiGLU quantization arguments");
    }
    const int status = apxinf::cuda::cutlass_ops::nvfp4_quantize_swiglu(
        source_bf16, destination_packed, destination_scales,
        static_cast<int>(rows), static_cast<int>(k),
        static_cast<int>(sf_vec_size), input_scale, row_major_scales,
        static_cast<cudaStream_t>(stream));
    if (status != 0) {
      throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                    "fused SwiGLU quantization failed with status " +
                        std::to_string(status));
    }
  });
#else
  (void)source_bf16; (void)destination_packed; (void)destination_scales;
  (void)rows; (void)k; (void)sf_vec_size; (void)input_scale;
  (void)row_major_scales; (void)stream;
  return APXINF_STATUS_UNSUPPORTED;
#endif
}

extern "C" uint64_t apxinf_gemm_execution_weight_prepack_count(
    apxinf_gemm_execution_t handle) {
  const auto* execution = reinterpret_cast<const Execution*>(handle);
  if (execution == nullptr || execution->implementation->provider_id != 3) {
    return 0;
  }
  return apxinf::gemm::cutlass_weight_prepack_count(*execution);
}

extern "C" apxinf_status_t apxinf_gemm_test_validate_candidates(
    apxinf_runtime_t runtime, const apxinf_gemm_spec_t* spec,
    const apxinf_gemm_policy_t* policy,
    const apxinf_gemm_bindings_t* bindings, const float* expected_output,
    uint64_t expected_output_len) {
  return apxinf::gemm::abi_boundary([&] {
    if (runtime == nullptr || spec == nullptr || policy == nullptr ||
        bindings == nullptr || expected_output == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid candidate validation arguments");
    }
    apxinf::gemm::Spec normalized{};
    static_cast<apxinf_gemm_spec_t&>(normalized) = *spec;
    validate_spec(normalized);
    validate_policy(*policy);
    validate_bindings(normalized, *bindings, true);
    apxinf::gemm::check_cuda(cudaSetDevice(runtime->device));
    apxinf::gemm::validate_candidates(normalized, *policy, *bindings,
                                      runtime->device, expected_output,
                                      expected_output_len);
  });
}

// Private test hook used to select a provider for focused lifecycle tests.
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
    apxinf::gemm::write_recipe(policy->cache_dir, keys.key,
                               serialized.str());
  });
}
