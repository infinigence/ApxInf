// Copyright 2025 ApxInf contributors.
// SPDX-License-Identifier: Apache-2.0
//
// Causal D256 attention: BF16 head_dim 256, 16 query heads over 4 KV heads,
// CAUSAL, one sequence of `APXINF_FIXED_SCENE_TOKENS` tokens. In-tree CUDA
// translated from the validated CuTe DSL implementation.
//
// Numeric contract (`sm100_hd256_2cta_fmha_forward.py`): ASCENDING KV with the
// mask LAST, NO rescale threshold, ex2 polynomial emulation 14/6/0, `div.rn`
// normalisation with 0/NaN -> 0.0, row_sum over CONTIGUOUS QUARTERS seeded
// (s, s) together with `acc_scale = ex2(..) * 0.5`, and a two-slot S pipeline.
// All of it is selected by this translation unit's `-DAPXINF_*` flag line,
// which `crates/apxinf-cuda-new/build.rs` keeps in step with
// `devlocal/aot2cuda/harness/MERGED_FLAGS.sh` (authoritative).

#define APXINF_D256_NS apxinf_attention_d256_causal
#define APXINF_D256_BUILD_LANGUAGE 1
#define APXINF_D256_BUILD_ACTION 0
#include "fmha_d256_sm110.cuh"

namespace APXINF_D256_NS {

int run_causal(const void* q, const void* k, const void* v, void* output,
               int seq, int query_heads, int kv_heads, float scale,
               cudaStream_t stream, State& state, int prepare_only) {
  if (q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
      seq <= 0 || kv_heads <= 0 || query_heads <= 0 ||
      query_heads % kv_heads != 0) {
    return -1;
  }
  int device = -1;
  int multiprocessors = 0;
  if (int rc = device_properties(device, multiprocessors); rc != 0) return rc;
  if (state.device != -1 && state.device != device) return -4;
  state.device = device;

  const int head_dim = 256;
  const int q_per_kv = query_heads / kv_heads;
  auto problem = cute::make_tuple(
      seq, seq, head_dim,
      cute::make_tuple(cute::make_tuple(q_per_kv, kv_heads), 1));

  const int q_token_stride = query_heads * head_dim;
  const int kv_token_stride = kv_heads * head_dim;
  StrideQ q_stride = make_stride(
      q_token_stride, _1{},
      make_stride(make_stride(head_dim, q_per_kv * head_dim),
                  q_token_stride * seq));
  StrideO o_stride = q_stride;
  StrideK kv_stride = make_stride(
      kv_token_stride, _1{},
      make_stride(make_stride(_0{}, head_dim), kv_token_stride * seq));
  StrideV v_stride = kv_stride;

  const int rounded_query = ((seq + 127) / 128) * 128;
  StrideLSE lse_stride = make_stride(
      _1{}, make_stride(make_stride(rounded_query, rounded_query * q_per_kv),
                        rounded_query * query_heads));
  // `ptr_LSE` stays null (see below), so nothing is allocated here and this
  // path can never `cudaMalloc` inside a CUDA graph capture.

  typename CausalCfg::Op::Arguments arguments{
      problem,
      {{static_cast<Element const*>(q), q_stride,
        static_cast<Element const*>(k), kv_stride,
        static_cast<Element const*>(v), v_stride},
       scale,
       1.0f,
       1.0f,
       1.0f,
       1.0f},
      // The language AOT emits no LSE, so neither do we: a non-null `ptr_LSE`
      // costs a `cutlass::fast_log` (libdevice: a polynomial plus a CALL) and a
      // global store per row.
      {static_cast<Element*>(output), o_stride, nullptr, lse_stride},
      {0, multiprocessors}};
  return finish<CausalCfg>(arguments, state, stream, prepare_only);
}

}  // namespace APXINF_D256_NS

extern "C" size_t apxinf_attention_d256_causal_state_size() {
  return sizeof(APXINF_D256_NS::State);
}

extern "C" void apxinf_attention_d256_causal_state_destroy(void* state_ptr) {
  using APXINF_D256_NS::State;
  if (state_ptr == nullptr) return;
  auto& state = *static_cast<State*>(state_ptr);
  if (state.device >= 0) cudaSetDevice(state.device);
  if (state.lse != nullptr) cudaFree(state.lse);
  if (state.workspace != nullptr) cudaFree(state.workspace);
  state = State{};
}

extern "C" int apxinf_attention_d256_causal_forward(
    const void* q, const void* k, const void* v, void* output, int seq,
    int query_heads, int kv_heads, float scale, cudaStream_t stream,
    void* state_ptr, int prepare_only) {
  if (state_ptr == nullptr) return -1;
  return APXINF_D256_NS::run_causal(
      q, k, v, output, seq, query_heads, kv_heads, scale, stream,
      *static_cast<APXINF_D256_NS::State*>(state_ptr), prepare_only);
}
