#include "../include/apxinf_cuda/norm.h"
#include "../framework/runtime_internal.h"
#include "../kernels/custom/norm.cuh"
#include "../kernels/custom/norm_head.h"
#include "../kernels/custom/norm_offset.h"

#include <climits>
#include <cmath>
#include <cstdint>
#include <cuda_fp8.h>
#include <string>

namespace {

using apxinf::framework::Failure;
namespace kernels = apxinf::norm::kernels;
namespace offset_kernels = apxinf::cuda_new::norm_offset;

constexpr int kThreads = 256;
constexpr int kMaxElementwiseBlocks = 4096;

bool valid_alignment(uint32_t alignment) {
  return alignment <= 256 && alignment != 0 &&
         (alignment & (alignment - 1)) == 0;
}

// Binding participation reads straight off the Spec fields. There is no
// per-semantic table to keep in step with the Rust contract any more: each
// predicate names the one field that decides it.

bool writes_hidden(const apxinf_norm_spec_t& spec) {
  return spec.semantic != APXINF_NORM_SEMANTIC_NORM;
}

bool writes_normalized(const apxinf_norm_spec_t& spec) {
  return spec.semantic != APXINF_NORM_SEMANTIC_RESIDUAL_ADD;
}

bool reads_residual(const apxinf_norm_spec_t& spec) {
  return writes_hidden(spec);
}

bool reads_weight(const apxinf_norm_spec_t& spec) {
  return spec.affine_kind == APXINF_NORM_AFFINE_WEIGHT ||
         spec.affine_kind == APXINF_NORM_AFFINE_WEIGHT_OFFSET;
}

bool reads_norm_bias(const apxinf_norm_spec_t& spec) {
  return spec.norm_kind == APXINF_NORM_KIND_LAYER;
}

bool reads_norm_style(const apxinf_norm_spec_t& spec) {
  return spec.affine_kind == APXINF_NORM_AFFINE_ADA_SCALE_SHIFT;
}

bool reads_gate_style(const apxinf_norm_spec_t& spec) {
  return spec.has_gate != 0;
}

// The gate replaces the bias in the combine, so a gated pipeline never takes
// one, and a pipeline with no combine has nothing to add it to.
bool may_have_bias(const apxinf_norm_spec_t& spec) {
  return writes_hidden(spec) && spec.has_gate == 0;
}

const char* semantic_name(uint32_t semantic) {
  switch (semantic) {
    case APXINF_NORM_SEMANTIC_NORM: return "norm";
    case APXINF_NORM_SEMANTIC_FUSED_ADD_NORM: return "fused_add_norm";
    case APXINF_NORM_SEMANTIC_RESIDUAL_ADD: return "residual_add";
    default: return "?";
  }
}

const char* norm_kind_name(uint32_t kind) {
  switch (kind) {
    case APXINF_NORM_KIND_NONE: return "none";
    case APXINF_NORM_KIND_RMS: return "rms";
    case APXINF_NORM_KIND_LAYER: return "layer";
    default: return "?";
  }
}

const char* affine_name(uint32_t affine) {
  switch (affine) {
    case APXINF_NORM_AFFINE_NONE: return "none";
    case APXINF_NORM_AFFINE_WEIGHT: return "weight";
    case APXINF_NORM_AFFINE_WEIGHT_OFFSET: return "weight_offset";
    case APXINF_NORM_AFFINE_ADA_SCALE_SHIFT: return "ada";
    default: return "?";
  }
}

const char* bias_order_name(uint32_t order) {
  switch (order) {
    case APXINF_NORM_BIAS_ORDER_SINGLE_ROUND: return "single_round";
    case APXINF_NORM_BIAS_ORDER_ROUND_AFTER_BIAS: return "round_after_bias";
    default: return "?";
  }
}

const char* quant_name(uint32_t quant) {
  switch (quant) {
    case APXINF_NORM_QUANT_NONE: return "none";
    case APXINF_NORM_QUANT_E4M3: return "e4m3";
    default: return "?";
  }
}

const char* dtype_name(uint32_t dtype) {
  switch (dtype) {
    case APXINF_DTYPE_F16: return "f16";
    case APXINF_DTYPE_BF16: return "bf16";
    default: return "?";
  }
}

// A field combination with no kernel is an explicit refusal: dispatching to a
// neighbouring kernel would silently change the arithmetic.
[[noreturn]] void unsupported(const apxinf_norm_spec_t& spec) {
  throw Failure(APXINF_STATUS_UNSUPPORTED,
                std::string("no norm kernel for semantic=") +
                    semantic_name(spec.semantic) + " norm_kind=" +
                    norm_kind_name(spec.norm_kind) + " affine=" +
                    affine_name(spec.affine_kind) + " gate=" +
                    (spec.has_gate != 0 ? "1" : "0") + " bias_order=" +
                    bias_order_name(spec.bias_order) + " quant=" +
                    quant_name(spec.quant_kind) + " dtype=" +
                    dtype_name(spec.dtype));
}

void validate_spec(const apxinf_norm_spec_t& spec) {
  if (spec.version != APXINF_NORM_SPEC_VERSION ||
      spec.semantic > APXINF_NORM_SEMANTIC_RESIDUAL_ADD ||
      spec.norm_kind > APXINF_NORM_KIND_LAYER ||
      spec.affine_kind > APXINF_NORM_AFFINE_ADA_SCALE_SHIFT ||
      spec.bias_order > APXINF_NORM_BIAS_ORDER_ROUND_AFTER_BIAS ||
      spec.quant_kind > APXINF_NORM_QUANT_E4M3 || spec.has_gate > 1 ||
      (spec.dtype != APXINF_DTYPE_F16 && spec.dtype != APXINF_DTYPE_BF16) ||
      spec.rows <= 0 || spec.cols <= 0 ||
      spec.rows > INT32_MAX || spec.cols > INT32_MAX || spec.has_bias > 1 ||
      spec.output_scale_is_unit > 1 ||
      !valid_alignment(spec.input_alignment) ||
      !valid_alignment(spec.weight_alignment) ||
      !valid_alignment(spec.bias_alignment) ||
      !valid_alignment(spec.residual_alignment) ||
      !valid_alignment(spec.style_alignment) ||
      !valid_alignment(spec.hidden_alignment) ||
      !valid_alignment(spec.normalized_alignment)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid Norm Spec");
  }
  // Stage coherence. These used to be implied by the single semantic code;
  // with independent fields they have to be stated, otherwise the binding
  // check below would accept a Spec that describes no pipeline at all.
  const bool normalizes = writes_normalized(spec);
  if ((spec.norm_kind != APXINF_NORM_KIND_NONE) != normalizes ||
      (spec.affine_kind != APXINF_NORM_AFFINE_NONE) != normalizes) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Norm semantic disagrees with norm_kind/affine_kind");
  }
  if (spec.norm_kind == APXINF_NORM_KIND_LAYER &&
      spec.affine_kind != APXINF_NORM_AFFINE_WEIGHT) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "LayerNorm takes a plain weight and a norm bias");
  }
  if (spec.affine_kind == APXINF_NORM_AFFINE_ADA_SCALE_SHIFT &&
      spec.norm_kind != APXINF_NORM_KIND_RMS) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "adaptive scale/shift conditioning is an RMS form");
  }
  if (spec.has_gate != 0 && !writes_hidden(spec)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "a gate needs a residual combine to act on");
  }
  if (spec.bias_order == APXINF_NORM_BIAS_ORDER_ROUND_AFTER_BIAS) {
    if (!may_have_bias(spec)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "bias_order describes a bias this Norm cannot take");
    }
    // Unconditional in the dtype: the intermediate rounding *is* the contract,
    // so the field is meaningless outside BF16 no matter which semantic
    // carries it. Scoping this to RESIDUAL_ADD would make the guard depend on
    // the Rust layer never setting the field on a fused pipeline, which is the
    // kind of cross-layer assumption this validation exists to not make.
    if (spec.dtype != APXINF_DTYPE_BF16) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "BiasThenResidual requires BF16 input and output");
    }
  }
  if (spec.quant_kind == APXINF_NORM_QUANT_E4M3) {
    if (spec.dtype != APXINF_DTYPE_F16 || !normalizes ||
        spec.output_dtype != APXINF_DTYPE_E4M3) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "E4M3 Norm output requires an F16 normalization pipeline");
    }
  } else if (spec.output_dtype != spec.dtype) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "unquantized Norm output must preserve the input dtype");
  }
  if (spec.has_bias != 0 && !may_have_bias(spec)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "this Norm pipeline does not take a bias");
  }
}

void validate_bindings(const apxinf_norm_spec_t& spec,
                       const apxinf_norm_bindings_t& bindings) {
  const bool missing =
      bindings.input == nullptr ||
      (writes_hidden(spec) && bindings.hidden == nullptr) ||
      (writes_normalized(spec) && bindings.normalized == nullptr) ||
      (reads_residual(spec) && bindings.residual == nullptr) ||
      (reads_weight(spec) && bindings.weight == nullptr) ||
      (reads_norm_bias(spec) && bindings.norm_bias == nullptr) ||
      (reads_norm_style(spec) && bindings.norm_style == nullptr) ||
      (reads_gate_style(spec) && bindings.gate_style == nullptr);
  if (missing) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "missing Norm binding for the requested pipeline");
  }
  if ((spec.has_bias != 0) != (bindings.bias != nullptr)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Norm bias binding disagrees with Spec.has_bias");
  }
  const bool scale_matches_spec =
      (bindings.output_scale == 1.0f) ==
      (spec.output_scale_is_unit != 0);
  const bool valid_output_scale =
      std::isfinite(bindings.output_scale) && bindings.output_scale > 0.0f &&
      scale_matches_spec &&
      (spec.output_dtype == APXINF_DTYPE_E4M3 ||
       bindings.output_scale == 1.0f);
  if (!valid_output_scale || !(bindings.eps > 0.0f) ||
      !std::isfinite(bindings.eps)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid Norm scalar");
  }
}

int elementwise_blocks(int64_t count) {
  const int64_t blocks = (count + kThreads - 1) / kThreads;
  if (blocks < 1) return 1;
  return static_cast<int>(blocks < kMaxElementwiseBlocks
                              ? blocks
                              : kMaxElementwiseBlocks);
}

bool plain_weight_rms(const apxinf_norm_spec_t& spec) {
  return spec.norm_kind == APXINF_NORM_KIND_RMS &&
         spec.affine_kind == APXINF_NORM_AFFINE_WEIGHT;
}

bool plain_weight_layer(const apxinf_norm_spec_t& spec) {
  return spec.norm_kind == APXINF_NORM_KIND_LAYER &&
         spec.affine_kind == APXINF_NORM_AFFINE_WEIGHT;
}

bool ada_rms(const apxinf_norm_spec_t& spec) {
  return spec.norm_kind == APXINF_NORM_KIND_RMS &&
         spec.affine_kind == APXINF_NORM_AFFINE_ADA_SCALE_SHIFT;
}

bool single_round_bias(const apxinf_norm_spec_t& spec) {
  return spec.has_gate == 0 &&
         spec.bias_order == APXINF_NORM_BIAS_ORDER_SINGLE_ROUND;
}

bool gated(const apxinf_norm_spec_t& spec) {
  return spec.has_gate != 0 &&
         spec.bias_order == APXINF_NORM_BIAS_ORDER_SINGLE_ROUND;
}

// The `(1 + weight)` affine. Its kernel lives in kernels/custom/norm_offset.cu
// rather than norm.cuh, and launches from there, because it reduces over a
// width-derived block of up to 1024 threads through its own warp tree instead
// of norm.cuh's fixed 256 threads and `block_sum_parallel_unsafe`. Routing it
// through `launch<T>` below would mean rebuilding it on those helpers, which
// would reorder the summation and move the Qwen3.8 output bits -- so it gets
// its own launch path, and exactly one field combination.
cudaError_t launch_weight_offset(const apxinf_norm_spec_t& spec,
                                 const apxinf_norm_bindings_t& bindings) {
  // One kernel, one shape: pure normalization, RMS reduction, BF16 in and out,
  // no fused quantization. It writes the storage dtype directly, so a
  // quantized output has nowhere to go.
  if (spec.semantic != APXINF_NORM_SEMANTIC_NORM ||
      spec.norm_kind != APXINF_NORM_KIND_RMS ||
      spec.dtype != APXINF_DTYPE_BF16 ||
      spec.quant_kind != APXINF_NORM_QUANT_NONE) {
    unsupported(spec);
  }
  // The moved host function calls cudaGetLastError() itself, which clears the
  // error, so the caller's own check would see success. Return its status
  // verbatim rather than re-deriving one -- a synthesized code would report an
  // illegal address or a bad configuration as a generic launch failure.
  return offset_kernels::rms_norm_bf16(
      bindings.input, bindings.weight, bindings.normalized,
      static_cast<int>(spec.rows), static_cast<int>(spec.cols), bindings.eps,
      static_cast<cudaStream_t>(bindings.stream));
}

cudaError_t launch_quantized_f16(const apxinf_norm_spec_t& spec,
                                 const apxinf_norm_bindings_t& bindings) {
  const auto* input = static_cast<const half*>(bindings.input);
  const auto* bias = static_cast<const half*>(bindings.bias);
  const auto* residual = static_cast<const half*>(bindings.residual);
  const auto* weight = static_cast<const half*>(bindings.weight);
  const auto* norm_bias = static_cast<const half*>(bindings.norm_bias);
  const auto* norm_style = static_cast<const half*>(bindings.norm_style);
  const auto* gate_style = static_cast<const half*>(bindings.gate_style);
  auto* hidden = static_cast<half*>(bindings.hidden);
  auto* normalized = static_cast<__nv_fp8_e4m3*>(bindings.normalized);
  const int rows = static_cast<int>(spec.rows);
  const int cols = static_cast<int>(spec.cols);
  const float inverse_scale = 1.0f / bindings.output_scale;
  auto stream = static_cast<cudaStream_t>(bindings.stream);

  switch (spec.semantic) {
    case APXINF_NORM_SEMANTIC_NORM:
      if (plain_weight_rms(spec)) {
        kernels::rms_norm_quant_f16_e4m3<<<rows, kThreads, 0, stream>>>(
            input, weight, normalized, rows, cols, bindings.eps,
            inverse_scale);
      } else if (plain_weight_layer(spec)) {
        kernels::layer_norm_quant_f16_e4m3<<<rows, kThreads, 0, stream>>>(
            input, weight, norm_bias, normalized, rows, cols, bindings.eps,
            inverse_scale);
      } else if (ada_rms(spec)) {
        kernels::ada_rms_norm_quant_f16_e4m3<<<rows, kThreads, 0, stream>>>(
            input, norm_style, normalized, rows, cols, bindings.eps,
            inverse_scale);
      } else {
        unsupported(spec);
      }
      break;
    case APXINF_NORM_SEMANTIC_FUSED_ADD_NORM:
      if (single_round_bias(spec) && plain_weight_rms(spec)) {
        kernels::bias_residual_rms_norm_quant_f16_e4m3
            <<<rows, kThreads, 0, stream>>>(
                input, bias, residual, weight, hidden, normalized, rows, cols,
                bindings.eps, inverse_scale);
      } else if (single_round_bias(spec) && plain_weight_layer(spec)) {
        kernels::bias_residual_layer_norm_quant_f16_e4m3
            <<<rows, kThreads, 0, stream>>>(
                input, bias, residual, weight, norm_bias, hidden, normalized,
                rows, cols, bindings.eps, inverse_scale);
      } else if (gated(spec) && ada_rms(spec)) {
        // The 10x1024 specialization performs uint2 loads on norm_style and a
        // uint32 store on normalized; gate on the actual pointer alignment.
        if (rows == 10 && cols == 1024 &&
            reinterpret_cast<uintptr_t>(norm_style) % 8 == 0 &&
            reinterpret_cast<uintptr_t>(normalized) % 4 == 0) {
          kernels::ada_gate_residual_rms_norm_quant_f16_e4m3_10x1024
              <<<rows, kThreads, 0, stream>>>(
                  input, residual, gate_style, norm_style, hidden, normalized,
                  bindings.eps, inverse_scale);
        } else {
          kernels::ada_gate_residual_rms_norm_quant_f16_e4m3
              <<<rows, kThreads, 0, stream>>>(
                  input, residual, gate_style, norm_style, hidden, normalized,
                  rows, cols, bindings.eps, inverse_scale);
        }
      } else {
        unsupported(spec);
      }
      break;
    default:
      // RESIDUAL_ADD writes no normalized tensor, so there is nothing to
      // quantize; validate_spec already refuses it.
      unsupported(spec);
  }
  return cudaGetLastError();
}

template <class T>
cudaError_t launch(const apxinf_norm_spec_t& spec,
                   const apxinf_norm_bindings_t& bindings) {
  const auto* input = static_cast<const T*>(bindings.input);
  const auto* bias = static_cast<const T*>(bindings.bias);
  const auto* residual = static_cast<const T*>(bindings.residual);
  const auto* weight = static_cast<const T*>(bindings.weight);
  const auto* norm_bias = static_cast<const T*>(bindings.norm_bias);
  const auto* norm_style = static_cast<const T*>(bindings.norm_style);
  const auto* gate_style = static_cast<const T*>(bindings.gate_style);
  auto* hidden = static_cast<T*>(bindings.hidden);
  auto* normalized = static_cast<T*>(bindings.normalized);
  const int rows = static_cast<int>(spec.rows);
  const int cols = static_cast<int>(spec.cols);
  const int64_t count = spec.rows * spec.cols;
  auto stream = static_cast<cudaStream_t>(bindings.stream);

  switch (spec.semantic) {
    case APXINF_NORM_SEMANTIC_NORM:
      if (plain_weight_rms(spec)) {
        kernels::rms_norm<T><<<rows, kThreads, 0, stream>>>(
            input, weight, normalized, rows, cols, bindings.eps);
      } else if (plain_weight_layer(spec)) {
        kernels::layer_norm<T><<<rows, kThreads, 0, stream>>>(
            input, weight, norm_bias, normalized, rows, cols, bindings.eps);
      } else if (ada_rms(spec)) {
        kernels::ada_rms_norm<T><<<rows, kThreads, 0, stream>>>(
            input, norm_style, normalized, rows, cols, bindings.eps);
      } else {
        unsupported(spec);
      }
      break;
    case APXINF_NORM_SEMANTIC_FUSED_ADD_NORM:
      if (single_round_bias(spec) && plain_weight_rms(spec)) {
        kernels::bias_residual_rms_norm<T><<<rows, kThreads, 0, stream>>>(
            input, bias, residual, weight, hidden, normalized, rows, cols,
            bindings.eps);
      } else if (single_round_bias(spec) && plain_weight_layer(spec)) {
        kernels::bias_residual_layer_norm<T><<<rows, kThreads, 0, stream>>>(
            input, bias, residual, weight, norm_bias, hidden, normalized, rows,
            cols, bindings.eps);
      } else if (gated(spec) && ada_rms(spec)) {
        kernels::ada_gate_residual_rms_norm<T><<<rows, kThreads, 0, stream>>>(
            input, residual, gate_style, norm_style, hidden, normalized, rows,
            cols, bindings.eps);
      } else {
        unsupported(spec);
      }
      break;
    case APXINF_NORM_SEMANTIC_RESIDUAL_ADD:
      if (single_round_bias(spec)) {
        kernels::bias_residual<T>
            <<<elementwise_blocks(count), kThreads, 0, stream>>>(
                input, bias, residual, hidden, count, cols);
      } else if (spec.has_gate == 0 &&
                 spec.bias_order ==
                     APXINF_NORM_BIAS_ORDER_ROUND_AFTER_BIAS) {
        kernels::bias_then_residual<T>
            <<<elementwise_blocks(count), kThreads, 0, stream>>>(
                input, bias, residual, hidden, count, cols);
      } else if (gated(spec)) {
        kernels::ada_gate_residual<T>
            <<<elementwise_blocks(count), kThreads, 0, stream>>>(
                input, residual, gate_style, hidden, count, cols);
      } else {
        unsupported(spec);
      }
      break;
    default:
      unsupported(spec);
  }
  return cudaGetLastError();
}

}  // namespace

extern "C" apxinf_status_t apxinf_norm_launch(
    apxinf_runtime_t runtime, const apxinf_norm_spec_t* spec,
    const apxinf_norm_bindings_t* bindings) {
  return apxinf::framework::abi_boundary([&] {
    if (runtime == nullptr || spec == nullptr || bindings == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null Norm argument");
    }
    validate_spec(*spec);
    validate_bindings(*spec, *bindings);
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    // weight_offset is checked first: it is the one affine whose kernel is not
    // a member of the norm.cuh family, so neither the quantized nor the
    // templated dispatch below has a branch for it.
    apxinf::framework::check_cuda(
        spec->affine_kind == APXINF_NORM_AFFINE_WEIGHT_OFFSET
            ? launch_weight_offset(*spec, *bindings)
            : spec->quant_kind == APXINF_NORM_QUANT_E4M3
                  ? launch_quantized_f16(*spec, *bindings)
                  : (spec->dtype == APXINF_DTYPE_BF16
                         ? launch<__nv_bfloat16>(*spec, *bindings)
                         : launch<__half>(*spec, *bindings)));
  });
}

// The family's per-head member. It writes its input in place and reduces over
// the last axis of a [tokens, heads, head_dim] activation, neither of which the
// [rows, cols] Spec can express, so it forwards straight to
// kernels/custom/norm_head.cu. That is also the family's third reduction tree;
// see norm_head.cu for why it is not rebuilt on the other two.

namespace {

bool norm_extent(int64_t value) { return value > 0 && value <= INT32_MAX; }

}  // namespace

extern "C" apxinf_status_t apxinf_norm_head_rms(
    void* data, const void* weight, int64_t rows, int64_t head_dim,
    float epsilon, apxinf_cuda_stream_t stream) {
  return apxinf::framework::abi_boundary([&] {
    if (data == nullptr || weight == nullptr || !norm_extent(rows) ||
        !norm_extent(head_dim)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid head RMSNorm arguments");
    }
    const int status = apxinf::cuda_new::norm_head::head_rms_norm(
        data, weight, static_cast<int>(rows), static_cast<int>(head_dim),
        epsilon, static_cast<cudaStream_t>(stream));
    if (status != 0) {
      throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                    "head RMSNorm failed with status " +
                        std::to_string(status));
    }
  });
}
