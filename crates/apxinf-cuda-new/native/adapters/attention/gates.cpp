#include "internal.h"
#include "../../kernels/custom/attention_gates.h"

#include <cstdint>
#include <limits>
#include <string>

namespace {
using apxinf::framework::Failure;
using apxinf::framework::abi_boundary;
bool extent(int64_t value) { return value > 0 && value <= INT32_MAX; }
bool valid_alignment(uint32_t alignment) {
  return alignment >= 2 && alignment <= 256 &&
         (alignment & (alignment - 1)) == 0;
}
bool gate_geometry(const apxinf_attention_gate_spec_t& spec,
                   int64_t& elements, int64_t& minimum_row_width) {
  if (!extent(spec.tokens) || !extent(spec.heads) ||
      !extent(spec.head_dim) ||
      spec.heads > std::numeric_limits<int64_t>::max() / spec.head_dim) {
    return false;
  }
  const int64_t width = spec.heads * spec.head_dim;
  if (width > std::numeric_limits<int64_t>::max() / spec.tokens ||
      width > std::numeric_limits<int64_t>::max() / 2) {
    return false;
  }
  elements = spec.tokens * width;
  minimum_row_width = 2 * width;
  return true;
}
void check(int status, const char* what) {
  if (status != 0)
    throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                  std::string(what) + " failed with status " +
                      std::to_string(status));
}

void validate_spec(const apxinf_attention_gate_spec_t& spec) {
  if (spec.version != APXINF_ATTENTION_GATE_SPEC_VERSION ||
      spec.semantic > APXINF_ATTENTION_GATE_SEMANTIC_APPLY_OUTPUT_GATE ||
      spec.dtype != APXINF_DTYPE_BF16 ||
      spec.source > APXINF_ATTENTION_GATE_SOURCE_PROJECTION ||
      spec.numerics > APXINF_ATTENTION_GATE_NUMERICS_ROUNDED_SIGMOID ||
      !valid_alignment(spec.input_alignment) ||
      !valid_alignment(spec.output_alignment) ||
      !valid_alignment(spec.auxiliary_alignment) || spec.elements <= 0) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "invalid Attention gate Spec");
  }
  if (spec.semantic == APXINF_ATTENTION_GATE_SEMANTIC_SPLIT_QUERY_GATE) {
    int64_t elements = 0;
    int64_t minimum_row_width = 0;
    if (spec.source != APXINF_ATTENTION_GATE_SOURCE_PROJECTION ||
        spec.numerics != APXINF_ATTENTION_GATE_NUMERICS_FLEXIBLE ||
        !gate_geometry(spec, elements, minimum_row_width) ||
        spec.elements != elements ||
        spec.source_row_width != minimum_row_width) {
      throw Failure(APXINF_STATUS_UNSUPPORTED,
                    "unsupported query/gate split Spec");
    }
    return;
  }
  if (spec.source == APXINF_ATTENTION_GATE_SOURCE_TENSOR) {
    if (spec.tokens != 0 || spec.heads != 0 || spec.head_dim != 0 ||
        spec.source_row_width != 0) {
      throw Failure(APXINF_STATUS_UNSUPPORTED,
                    "dense output gate has projection geometry");
    }
    return;
  }
  int64_t elements = 0;
  int64_t minimum_row_width = 0;
  if (spec.numerics != APXINF_ATTENTION_GATE_NUMERICS_ROUNDED_SIGMOID ||
      !gate_geometry(spec, elements, minimum_row_width) ||
      spec.elements != elements ||
      spec.source_row_width < minimum_row_width) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "unsupported projection output gate Spec");
  }
}

void validate_bindings(const apxinf_attention_gate_spec_t& spec,
                       const apxinf_attention_gate_bindings_t& bindings) {
  const bool split =
      spec.semantic == APXINF_ATTENTION_GATE_SEMANTIC_SPLIT_QUERY_GATE;
  if (bindings.input == nullptr || bindings.output == nullptr ||
      (split != (bindings.auxiliary != nullptr))) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "invalid Attention gate bindings");
  }
  for (const auto [pointer, alignment] : {
           std::pair{bindings.input, spec.input_alignment},
           std::pair{static_cast<const void*>(bindings.output),
                     spec.output_alignment},
           std::pair{static_cast<const void*>(bindings.auxiliary),
                     spec.auxiliary_alignment}}) {
    if (pointer != nullptr &&
        reinterpret_cast<uintptr_t>(pointer) % alignment != 0) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "Attention gate pointer contradicts alignment");
    }
  }
}
}  // namespace

extern "C" apxinf_status_t apxinf_attention_gate_launch(
    apxinf_runtime_t runtime, const apxinf_attention_gate_spec_t* spec,
    const apxinf_attention_gate_bindings_t* bindings) {
  return abi_boundary([&] {
    if (runtime == nullptr || spec == nullptr || bindings == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "null Attention gate family argument");
    }
    validate_spec(*spec);
    validate_bindings(*spec, *bindings);
    const auto stream = static_cast<cudaStream_t>(bindings->stream);
    if (spec->semantic == APXINF_ATTENTION_GATE_SEMANTIC_SPLIT_QUERY_GATE) {
      check(apxinf::cuda_new::attention::split_query_and_gate(
                bindings->input, bindings->output, bindings->auxiliary,
                static_cast<int>(spec->tokens), static_cast<int>(spec->heads),
                static_cast<int>(spec->head_dim), stream),
            "query/gate split");
      return;
    }
    if (spec->source == APXINF_ATTENTION_GATE_SOURCE_PROJECTION) {
      apxinf::attention::check_cuda(apxinf::attention::launch_output_gate_projection(
          bindings->output, bindings->input, static_cast<int>(spec->tokens),
          static_cast<int>(spec->heads), static_cast<int>(spec->head_dim),
          spec->source_row_width, stream));
      return;
    }
    if (spec->numerics ==
        APXINF_ATTENTION_GATE_NUMERICS_ROUNDED_SIGMOID) {
      apxinf::attention::check_cuda(
          apxinf::attention::launch_output_gate_tensor_rounded(
              bindings->output, bindings->input, spec->elements, stream));
      return;
    }
    check(apxinf::cuda_new::attention::apply_output_gate(
              bindings->output, bindings->input, spec->elements, stream),
          "attention output gate");
  });
}
