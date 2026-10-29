// Linear-attention (GDN) kernels: recurrent step, causal conv, gated norm, L2 norm, decay/beta.
//
// These ops have a single fixed implementation and no persisted selection,
// so per doc/adding-new-kernels.md §6 they carry no candidate registry,
// tuning key, or autotuner; they are direct C-ABI forwarders.

#include "../../include/apxinf_cuda/gdn.h"

#include "../../framework/runtime_internal.h"
#include "../../kernels/custom/gdn_ops.h"
#include "../../kernels/flashinfer_gdn/flashinfer_gdn.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

extern "C" cudaError_t apxinf_norm_gated_rms_silu_bf16(
    const void* x, const void* z, const void* weight, void* out, int rows,
    int cols, int z_heads, int64_t z_row_stride, int64_t z_col_offset,
    float eps, cudaStream_t stream);

namespace {

using apxinf::framework::Failure;
using apxinf::framework::abi_boundary;

void check(int status, const char* what) {
  if (status != 0) {
    throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                  std::string(what) + " failed with status " +
                      std::to_string(status));
  }
}

bool extent(int64_t value) { return value > 0 && value <= INT32_MAX; }

int as_int(int64_t value) { return static_cast<int>(value); }

void require(bool condition, const char* message) {
  if (!condition) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, message);
  }
}

void validate_spec(const apxinf_gdn_spec_t& spec) {
  require(spec.version == APXINF_GDN_SPEC_VERSION,
          "unsupported GDN Spec version");
  require(spec.operation <= APXINF_GDN_OPERATION_CONV_PREPARE,
          "invalid GDN operation");
  require(spec.input_dtype <= APXINF_DTYPE_I32 &&
              spec.output_dtype <= APXINF_DTYPE_I32,
          "invalid GDN dtype");
  require(spec.rank >= 1 && spec.rank <= 3, "invalid GDN rank");
  require(spec.layout <= APXINF_GDN_LAYOUT_INTERLEAVED_QKV,
          "invalid GDN layout");
  require(spec.quantization <= APXINF_GDN_QUANTIZATION_STATIC_E4M3,
          "invalid GDN quantization");
  require(spec.has_window <= 1, "invalid GDN window predicate");
}

void validate_finite_positive(float value, const char* message) {
  require(std::isfinite(value) && value > 0.0F, message);
}

}  // namespace

extern "C" apxinf_status_t apxinf_gdn_launch(
    apxinf_runtime_t runtime, const apxinf_gdn_spec_t* spec,
    const apxinf_gdn_bindings_t* bindings) {
  return abi_boundary([&] {
    require(runtime != nullptr && spec != nullptr && bindings != nullptr,
            "null GDN launch argument");
    validate_spec(*spec);
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    const auto stream = static_cast<cudaStream_t>(bindings->stream);

    switch (spec->operation) {
      case APXINF_GDN_OPERATION_RECURRENT_STEP:
        require(bindings->state != nullptr && bindings->q != nullptr &&
                    bindings->k != nullptr && bindings->v != nullptr &&
                    bindings->decay != nullptr && bindings->beta != nullptr &&
                    bindings->output != nullptr && extent(spec->value_heads) &&
                    extent(spec->key_heads) && extent(spec->value_dim) &&
                    extent(spec->key_dim),
                "invalid GDN recurrent-step contract");
        check(apxinf::cuda_new::gdn_ops::gdn_recurrent_step(
                  bindings->state, bindings->q, bindings->k, bindings->v,
                  bindings->decay, bindings->beta, bindings->output,
                  as_int(spec->value_heads), as_int(spec->key_heads),
                  as_int(spec->value_dim), as_int(spec->key_dim), stream),
              "GDN recurrent step");
        return;

      case APXINF_GDN_OPERATION_GATED_NORM: {
        require(bindings->input != nullptr && bindings->gate != nullptr &&
                    bindings->weight != nullptr && bindings->output != nullptr &&
                    extent(spec->heads) && extent(spec->head_dim),
                "invalid GDN gated-norm contract");
        validate_finite_positive(bindings->epsilon,
                                 "invalid GDN gated-norm epsilon");
        if (spec->layout == APXINF_GDN_LAYOUT_STRIDED_GATE) {
          require(spec->rank == 2 && spec->input_dtype == APXINF_DTYPE_BF16 &&
                      spec->output_dtype == APXINF_DTYPE_BF16 &&
                      spec->quantization == APXINF_GDN_QUANTIZATION_NONE &&
                      extent(spec->tokens) && extent(spec->gate_row_stride) &&
                      spec->gate_column_offset >= 0,
                  "invalid strided GDN gated-norm Spec");
          apxinf::framework::check_cuda(apxinf_norm_gated_rms_silu_bf16(
              bindings->input, bindings->gate, bindings->weight,
              bindings->output, as_int(spec->tokens),
              as_int(spec->head_dim), as_int(spec->heads),
              spec->gate_row_stride, spec->gate_column_offset,
              bindings->epsilon, stream));
          return;
        }
        require(spec->layout == APXINF_GDN_LAYOUT_DENSE,
                "invalid dense GDN gated-norm layout");
        if (spec->quantization == APXINF_GDN_QUANTIZATION_STATIC_E4M3) {
          require(spec->rank == 3 && extent(spec->tokens) &&
                      spec->head_dim == 128 &&
                      (spec->input_dtype == APXINF_DTYPE_F16 ||
                       spec->input_dtype == APXINF_DTYPE_BF16) &&
                      spec->output_dtype == APXINF_DTYPE_BF16 &&
                      bindings->quantized != nullptr &&
                      spec->tokens <=
                          std::numeric_limits<int>::max() / spec->heads,
                  "invalid quantized GDN gated-norm Spec");
          validate_finite_positive(bindings->input_scale,
                                   "invalid GDN quantization scale");
          check(apxinf::cuda_new::gdn_ops::gdn_gated_norm_quantize(
                    bindings->input, bindings->gate, bindings->weight,
                    bindings->output, bindings->quantized,
                    as_int(spec->tokens * spec->heads),
                    spec->input_dtype == APXINF_DTYPE_F16,
                    bindings->epsilon, bindings->input_scale, stream),
                "GDN norm quantization");
          return;
        }
        require(spec->quantization == APXINF_GDN_QUANTIZATION_NONE &&
                    spec->output_dtype == APXINF_DTYPE_BF16,
                "invalid unquantized GDN gated-norm Spec");
        if (spec->rank == 2) {
          require(spec->input_dtype == APXINF_DTYPE_BF16,
                  "GDN step gated norm requires BF16");
          check(apxinf::cuda_new::gdn_ops::gdn_gated_norm(
                    bindings->input, bindings->gate, bindings->weight,
                    bindings->output, as_int(spec->heads),
                    as_int(spec->head_dim), bindings->epsilon, stream),
                "GDN gated norm");
          return;
        }
        require(spec->rank == 3 && extent(spec->tokens) &&
                    (spec->input_dtype == APXINF_DTYPE_F16 ||
                     spec->input_dtype == APXINF_DTYPE_BF16),
                "invalid sequence GDN gated-norm Spec");
        check(spec->input_dtype == APXINF_DTYPE_F16
                  ? apxinf::cuda_new::gdn_ops::gdn_gated_norm_seq_f16(
                        bindings->input, bindings->gate, bindings->weight,
                        bindings->output, as_int(spec->tokens),
                        as_int(spec->heads), as_int(spec->head_dim),
                        bindings->epsilon, stream)
                  : apxinf::cuda_new::gdn_ops::gdn_gated_norm_seq(
                        bindings->input, bindings->gate, bindings->weight,
                        bindings->output, as_int(spec->tokens),
                        as_int(spec->heads), as_int(spec->head_dim),
                        bindings->epsilon, stream),
              "GDN sequence gated norm");
        return;
      }

      case APXINF_GDN_OPERATION_CAUSAL_CONV:
        require(bindings->input != nullptr && bindings->weight != nullptr &&
                    bindings->output != nullptr && extent(spec->channels) &&
                    extent(spec->kernel_width),
                "invalid GDN causal-conv contract");
        if (spec->rank == 1) {
          require(bindings->window != nullptr && spec->has_window != 0,
                  "GDN conv step requires a window");
          check(apxinf::cuda_new::gdn_ops::gdn_causal_conv_step(
                    bindings->window, bindings->input, bindings->weight,
                    bindings->output, as_int(spec->channels),
                    as_int(spec->kernel_width), stream),
                "GDN conv step");
          return;
        }
        require(spec->rank == 2 && extent(spec->tokens) &&
                    (spec->has_window != 0) == (bindings->window != nullptr),
                "invalid GDN conv-forward Spec");
        check(apxinf::cuda_new::gdn_ops::gdn_causal_conv_forward(
                  bindings->input, bindings->weight, bindings->output,
                  bindings->window, as_int(spec->tokens),
                  as_int(spec->channels), as_int(spec->kernel_width), stream),
              "GDN conv forward");
        return;

      case APXINF_GDN_OPERATION_DECAY_AND_BETA:
        require(bindings->a != nullptr && bindings->b != nullptr &&
                    bindings->a_log != nullptr && bindings->dt_bias != nullptr &&
                    bindings->decay != nullptr && bindings->beta != nullptr &&
                    extent(spec->heads),
                "invalid GDN decay/beta contract");
        if (spec->rank == 1) {
          check(apxinf::cuda_new::gdn_ops::gdn_decay_and_beta(
                    bindings->a, bindings->b, bindings->a_log,
                    bindings->dt_bias, bindings->decay, bindings->beta,
                    as_int(spec->heads),
                    stream),
                "GDN gates");
          return;
        }
        require(spec->rank == 2 && extent(spec->tokens),
                "invalid sequence GDN decay/beta Spec");
        check(apxinf::cuda_new::gdn_ops::gdn_decay_and_beta_seq(
                  bindings->a, bindings->b, bindings->a_log,
                  bindings->dt_bias, bindings->decay, bindings->beta,
                  as_int(spec->tokens),
                  as_int(spec->heads), stream),
              "GDN sequence gates");
        return;

      case APXINF_GDN_OPERATION_L2_NORMALIZE:
        require(bindings->output != nullptr && spec->rank == 2 &&
                    extent(spec->heads) && extent(spec->head_dim),
                "invalid GDN L2-normalize contract");
        validate_finite_positive(bindings->epsilon,
                                 "invalid GDN L2-normalize epsilon");
        check(apxinf::cuda_new::gdn_ops::gdn_l2_normalize_heads(
                  bindings->output, as_int(spec->heads),
                  as_int(spec->head_dim), bindings->epsilon, stream),
              "GDN head normalization");
        return;

      case APXINF_GDN_OPERATION_CHUNK_SCAN:
        require(bindings->q != nullptr && bindings->k != nullptr &&
                    bindings->v != nullptr && bindings->gate_log != nullptr &&
                    bindings->beta != nullptr && bindings->output != nullptr &&
                    bindings->state != nullptr && extent(spec->tokens) &&
                    extent(spec->value_heads) && extent(spec->key_heads) &&
                    extent(spec->chunk_size) && extent(spec->key_dim) &&
                    extent(spec->num_chunks) && extent(spec->q_row_stride) &&
                    extent(spec->k_row_stride) && extent(spec->v_row_stride),
                "invalid GDN chunk-scan contract");
        check(apxinf::cuda_new::gdn_ops::gdn_chunk_scan(
                  bindings->q, bindings->k, bindings->v,
                  bindings->gate_log, bindings->beta, bindings->output,
                  bindings->state, as_int(spec->tokens),
                  as_int(spec->value_heads), as_int(spec->key_heads),
                  as_int(spec->chunk_size), as_int(spec->key_dim),
                  as_int(spec->num_chunks), as_int(spec->q_row_stride),
                  as_int(spec->k_row_stride), as_int(spec->v_row_stride),
                  stream),
              "GDN chunk scan");
        return;

      case APXINF_GDN_OPERATION_PREFILL:
        require(bindings->q != nullptr && bindings->k != nullptr &&
                    bindings->v != nullptr && bindings->output != nullptr &&
                    bindings->gate_log != nullptr && bindings->beta != nullptr &&
                    bindings->cu_seqlens != nullptr && bindings->state != nullptr &&
                    bindings->workspace != nullptr && extent(spec->tokens) &&
                    extent(spec->key_heads) && extent(spec->value_heads) &&
                    extent(spec->num_sequences),
                "invalid GDN prefill contract");
        check(apxinf::cuda_new::flashinfer_gdn::prefill(
                  bindings->q, bindings->k, bindings->v, bindings->output,
                  bindings->gate_log, bindings->beta, bindings->cu_seqlens,
                  bindings->state, bindings->workspace, as_int(spec->tokens),
                  as_int(spec->key_heads), as_int(spec->value_heads),
                  as_int(spec->num_sequences), bindings->scale, stream),
              "FlashInfer GDN prefill");
        return;

      case APXINF_GDN_OPERATION_PREPARE_PREFILL:
        require(bindings->input != nullptr && bindings->q_output != nullptr &&
                    bindings->k_output != nullptr &&
                    bindings->v_output != nullptr &&
                    bindings->gate_log != nullptr && bindings->alpha != nullptr &&
                    extent(spec->tokens) && extent(spec->row_width) &&
                    extent(spec->key_heads) && extent(spec->value_heads) &&
                    extent(spec->head_dim),
                "invalid GDN prefill-preparation contract");
        validate_finite_positive(bindings->epsilon,
                                 "invalid GDN prefill-preparation epsilon");
        check(apxinf::cuda_new::gdn_ops::gdn_prepare_prefill(
                  bindings->input, bindings->q_output, bindings->k_output,
                  bindings->v_output, bindings->gate_log, bindings->alpha,
                  as_int(spec->tokens), as_int(spec->row_width),
                  as_int(spec->key_heads), as_int(spec->value_heads),
                  as_int(spec->head_dim), bindings->epsilon, stream),
              "GDN FlashInfer preparation");
        return;

      case APXINF_GDN_OPERATION_CONV_PREPARE:
        require(bindings->input != nullptr && bindings->weight != nullptr &&
                    bindings->window != nullptr && bindings->q_output != nullptr &&
                    bindings->k_output != nullptr &&
                    bindings->v_output != nullptr &&
                    bindings->decay != nullptr && bindings->alpha != nullptr &&
                    extent(spec->tokens) && extent(spec->key_heads) &&
                    extent(spec->value_heads),
                "invalid fused GDN conv-preparation contract");
        validate_finite_positive(bindings->epsilon,
                                 "invalid GDN conv-preparation epsilon");
        check(apxinf::cuda_new::gdn_ops::gdn_conv_prepare(
                  bindings->input, bindings->weight, bindings->window,
                  bindings->q_output, bindings->k_output,
                  bindings->v_output, bindings->decay, bindings->alpha,
                  as_int(spec->tokens), as_int(spec->key_heads),
                  as_int(spec->value_heads), bindings->epsilon, stream),
              "fused GDN convolution preparation");
        return;
    }
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "unhandled GDN operation");
  });
}

extern "C" apxinf_status_t apxinf_gdn_chunk_scan_prepare(int32_t device) {
  return abi_boundary([&] {
    if (device < 0) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid GDN chunk-scan device");
    }
    check(apxinf::cuda_new::gdn_ops::gdn_chunk_scan_prepare(device),
          "GDN chunk-scan preparation");
  });
}

extern "C" int64_t apxinf_gdn_prefill_workspace_bytes(int64_t v_heads,
                                                         int64_t num_seqs) {
  if (v_heads <= 0 || num_seqs <= 0) return 0;
  return static_cast<int64_t>(apxinf::cuda_new::flashinfer_gdn::
                                  tensor_map_workspace_bytes(
                                      static_cast<int>(v_heads),
                                      static_cast<int>(num_seqs)));
}
