// Linear-attention (GDN) kernels: recurrent step, causal conv, gated norm, L2 norm, decay/beta.
//
// These ops have a single fixed implementation and no persisted selection,
// so per doc/adding-new-kernels.md §6 they carry no candidate registry,
// tuning key, or autotuner; they are direct C-ABI forwarders.

#include "../../include/apxinf_cuda/gdn.h"

#include "../../framework/runtime_internal.h"
#include "../../kernels/custom/gdn_ops.h"
#include "../../kernels/flashinfer_gdn/flashinfer_gdn.h"

#include <cstdint>
#include <string>

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

}  // namespace

extern "C" apxinf_status_t apxinf_gdn_recurrent_step(
    void* state, const void* q, const void* k, const void* v,
    const void* decay, const void* beta, void* output, int64_t v_heads,
    int64_t k_heads, int64_t v_dim, int64_t k_dim,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (state == nullptr || q == nullptr || k == nullptr || v == nullptr ||
        decay == nullptr || beta == nullptr || output == nullptr ||
        !extent(v_heads) || !extent(k_heads) || !extent(v_dim) ||
        !extent(k_dim)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid GDN recurrent step arguments");
    }
    check(apxinf::cuda::gdn_ops::gdn_recurrent_step(
              state, q, k, v, decay, beta, output, static_cast<int>(v_heads),
              static_cast<int>(k_heads), static_cast<int>(v_dim),
              static_cast<int>(k_dim), static_cast<cudaStream_t>(stream)),
          "GDN recurrent step");
  });
}

extern "C" apxinf_status_t apxinf_gdn_gated_norm(
    const void* input, const void* gate, const void* weight, void* output,
    int64_t heads, int64_t head_dim, float epsilon,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (input == nullptr || gate == nullptr || weight == nullptr ||
        output == nullptr || !extent(heads) || !extent(head_dim)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid GDN gated norm arguments");
    }
    check(apxinf::cuda::gdn_ops::gdn_gated_norm(
              input, gate, weight, output, static_cast<int>(heads),
              static_cast<int>(head_dim), epsilon,
              static_cast<cudaStream_t>(stream)),
          "GDN gated norm");
  });
}

extern "C" apxinf_status_t apxinf_gdn_causal_conv_step(
    void* window, const void* input, const void* weight, void* output,
    int64_t channels, int64_t kernel_width, apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (window == nullptr || input == nullptr || weight == nullptr ||
        output == nullptr || !extent(channels) || !extent(kernel_width)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid GDN conv step arguments");
    }
    check(apxinf::cuda::gdn_ops::gdn_causal_conv_step(
              window, input, weight, output, static_cast<int>(channels),
              static_cast<int>(kernel_width),
              static_cast<cudaStream_t>(stream)),
          "GDN conv step");
  });
}

extern "C" apxinf_status_t apxinf_gdn_widen_f16_to_bf16(
    const void* input, void* output, int64_t count,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (input == nullptr || output == nullptr || !extent(count)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid FP16 widening arguments");
    }
    check(apxinf::cuda::gdn_ops::gdn_widen_f16_to_bf16(
              input, output, static_cast<long long>(count),
              static_cast<cudaStream_t>(stream)),
          "FP16 widening");
  });
}

extern "C" apxinf_status_t apxinf_gdn_prepare_flashinfer(
    const void* fused, void* q_out, void* k_out, void* v_out, const void* g,
    void* alpha, int64_t tokens, int64_t row_width, int64_t k_heads,
    int64_t v_heads, int64_t dim, float epsilon, apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (fused == nullptr || q_out == nullptr || k_out == nullptr ||
        v_out == nullptr || g == nullptr || alpha == nullptr ||
        !extent(tokens) || !extent(row_width) || !extent(k_heads) ||
        !extent(v_heads) || !extent(dim)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid GDN FlashInfer preparation arguments");
    }
    check(apxinf::cuda::gdn_ops::gdn_prepare_flashinfer(
              fused, q_out, k_out, v_out, g, alpha, static_cast<int>(tokens),
              static_cast<int>(row_width), static_cast<int>(k_heads),
              static_cast<int>(v_heads), static_cast<int>(dim), epsilon,
              static_cast<cudaStream_t>(stream)),
          "GDN FlashInfer preparation");
  });
}

extern "C" apxinf_status_t apxinf_flashinfer_gdn_prefill(
    const void* q, const void* k, const void* v, void* out,
    const void* gate_log, const void* beta, const void* cu_seqlens,
    void* state, void* tensor_map_workspace, int64_t tokens, int64_t q_heads,
    int64_t v_heads, int64_t num_seqs, float scale,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (q == nullptr || k == nullptr || v == nullptr || out == nullptr ||
        gate_log == nullptr || beta == nullptr || cu_seqlens == nullptr ||
        state == nullptr || tensor_map_workspace == nullptr ||
        !extent(tokens) || !extent(q_heads) || !extent(v_heads) ||
        !extent(num_seqs)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid FlashInfer GDN prefill arguments");
    }
    check(apxinf::cuda::flashinfer_gdn::prefill(
              q, k, v, out, gate_log, beta, cu_seqlens, state,
              tensor_map_workspace, static_cast<int>(tokens),
              static_cast<int>(q_heads), static_cast<int>(v_heads),
              static_cast<int>(num_seqs), scale,
              static_cast<cudaStream_t>(stream)),
          "FlashInfer GDN prefill");
  });
}

extern "C" int64_t apxinf_flashinfer_gdn_workspace_bytes(int64_t v_heads,
                                                         int64_t num_seqs) {
  if (v_heads <= 0 || num_seqs <= 0) return 0;
  return static_cast<int64_t>(apxinf::cuda::flashinfer_gdn::
                                  tensor_map_workspace_bytes(
                                      static_cast<int>(v_heads),
                                      static_cast<int>(num_seqs)));
}

extern "C" apxinf_status_t apxinf_gdn_causal_conv_forward(
    const void* input, const void* weight, void* output, void* window,
    int64_t tokens, int64_t channels, int64_t kernel_width,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    // window is optional: a caller that will not decode afterwards passes null.
    if (input == nullptr || weight == nullptr || output == nullptr ||
        !extent(tokens) || !extent(channels) || !extent(kernel_width)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid GDN conv forward arguments");
    }
    check(apxinf::cuda::gdn_ops::gdn_causal_conv_forward(
              input, weight, output, window, static_cast<int>(tokens),
              static_cast<int>(channels), static_cast<int>(kernel_width),
              static_cast<cudaStream_t>(stream)),
          "GDN conv forward");
  });
}

extern "C" apxinf_status_t apxinf_gdn_decay_and_beta_seq(
    const void* a, const void* b, const void* a_log, const void* dt_bias,
    void* decay, void* beta, int64_t tokens, int64_t heads,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (a == nullptr || b == nullptr || a_log == nullptr ||
        dt_bias == nullptr || decay == nullptr || beta == nullptr ||
        !extent(tokens) || !extent(heads)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid GDN sequence gate arguments");
    }
    check(apxinf::cuda::gdn_ops::gdn_decay_and_beta_seq(
              a, b, a_log, dt_bias, decay, beta, static_cast<int>(tokens),
              static_cast<int>(heads), static_cast<cudaStream_t>(stream)),
          "GDN sequence gates");
  });
}

extern "C" apxinf_status_t apxinf_gdn_gated_norm_seq(
    const void* input, const void* gate, const void* weight, void* output,
    int64_t tokens, int64_t heads, int64_t head_dim, float epsilon,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (input == nullptr || gate == nullptr || weight == nullptr ||
        output == nullptr || !extent(tokens) || !extent(heads) ||
        !extent(head_dim)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid GDN sequence gated norm arguments");
    }
    check(apxinf::cuda::gdn_ops::gdn_gated_norm_seq(
              input, gate, weight, output, static_cast<int>(tokens),
              static_cast<int>(heads), static_cast<int>(head_dim), epsilon,
              static_cast<cudaStream_t>(stream)),
          "GDN sequence gated norm");
  });
}

extern "C" apxinf_status_t apxinf_gdn_l2_normalize_heads(
    void* data, int64_t heads, int64_t head_dim, float epsilon,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (data == nullptr || !extent(heads) || !extent(head_dim)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid GDN normalization arguments");
    }
    check(apxinf::cuda::gdn_ops::gdn_l2_normalize_heads(
              data, static_cast<int>(heads), static_cast<int>(head_dim),
              epsilon, static_cast<cudaStream_t>(stream)),
          "GDN head normalization");
  });
}

extern "C" apxinf_status_t apxinf_gdn_decay_and_beta(
    const void* a, const void* b, const void* a_log, const void* dt_bias,
    void* decay, void* beta, int64_t heads, apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (a == nullptr || b == nullptr || a_log == nullptr ||
        dt_bias == nullptr || decay == nullptr || beta == nullptr ||
        !extent(heads)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid GDN gate arguments");
    }
    check(apxinf::cuda::gdn_ops::gdn_decay_and_beta(
              a, b, a_log, dt_bias, decay, beta, static_cast<int>(heads),
              static_cast<cudaStream_t>(stream)),
          "GDN gates");
  });
}


extern "C" apxinf_status_t apxinf_gdn_chunk_scan(
    const void* q, const void* k, const void* v, const void* g,
    const void* beta, void* out, void* state, int64_t seq_padded,
    int64_t v_heads, int64_t k_heads, int64_t chunk_size, int64_t k_dim,
    int64_t num_chunks, int64_t q_row_stride, int64_t k_row_stride,
    int64_t v_row_stride, apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (q == nullptr || k == nullptr || v == nullptr || g == nullptr ||
        beta == nullptr || out == nullptr || state == nullptr ||
        !extent(seq_padded) || !extent(v_heads) || !extent(k_heads) ||
        !extent(chunk_size) || !extent(k_dim) || !extent(num_chunks) ||
        !extent(q_row_stride) || !extent(k_row_stride) ||
        !extent(v_row_stride)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid GDN chunk scan arguments");
    }
    check(apxinf::cuda::gdn_ops::gdn_chunk_scan(
              q, k, v, g, beta, out, state, static_cast<int>(seq_padded),
              static_cast<int>(v_heads), static_cast<int>(k_heads),
              static_cast<int>(chunk_size), static_cast<int>(k_dim),
              static_cast<int>(num_chunks), static_cast<int>(q_row_stride),
              static_cast<int>(k_row_stride), static_cast<int>(v_row_stride),
              static_cast<cudaStream_t>(stream)),
          "GDN chunk scan");
  });
}
