// Copyright 2025 ApxInf contributors.
// SPDX-License-Identifier: Apache-2.0
//
// Split-batch D256 attention: BF16 head_dim 256, 16 query heads over 4 KV heads,
// NON-causal, 50 query tokens against a split-batch KV view. In-tree CUDA
// translated from the validated CuTe DSL implementation.
//
// Numeric contract (`flash_fwd_sm100.py`): fully DESCENDING KV with the mask
// FIRST, `rescale_threshold = 8.0`, ex2 polynomial emulation OFF
// (`APXINF_EX2_EMU_RES=0`), `rcp.approx.f32` normalisation, row_sum stride-8
// seeded into LANE 0 only, `ResidualMask` through FA4's R2P bitmask is NOT
// used here (`APXINF_MASK_R2P` stays at its upstream default). All of it comes
// from this translation unit's `-DAPXINF_*` flag line, which
// `crates/apxinf-cuda-new/build.rs` keeps in step with
// `devlocal/aot2cuda/harness/MERGED_FLAGS.sh` (authoritative).
//
// Pack-GQA (`run_pack`) folds `q_per_kv` into the M mode, which is what makes
// one launch of 16 CTAs reproduce the AOT grid (2,4,2) instead of two launches
// of 8. The stock mapping puts `q_per_kv` in the HEAD mode, so every
// (q_head, m_tile) pair gets its own CTA: for 50 query rows in a 128-row tile
// that is 39% row-slot utilisation and exactly 2.000x the AOT's
// `sm__inst_executed_pipe_tensor` (measured). FA4 instead makes the M mode the
// HIERARCHICAL coordinate (q_per_kv, seq_q) with strides
// (head_dim, q_token_stride) -- the row set of one KV head really is a 2-level
// coordinate in gmem, there is no affine row stride that covers it -- so a
// 128-row tile holds 4 q heads x 32 tokens and 4*50 = 200 real rows land in 2
// tiles (78% utilisation). `q_per_kv` has to be a STATIC extent because
// `make_tma_copy`'s `CUTE_STATIC_ASSERT_V(size(slayout) == size(cta_v_map))`
// needs the tile-M size to be a compile-time constant.
//
// It is numerically inert: which rows share a tile does not change any row's
// own reduction (non-causal, every row in the tile sees the same KV range) and
// the trip count is still `ceil_div(seq_k, 128)`.

#define APXINF_D256_NS apxinf_attention_d256_split_batch
#define APXINF_D256_BUILD_LANGUAGE 0
#define APXINF_D256_BUILD_ACTION 1
#include "fmha_d256_sm110.cuh"

namespace APXINF_D256_NS {

// `cu_seqlens_k` is a DEVICE array of `batch + 1` ints: `seqused_k[b] =
// cu_seqlens_k[b+1] - cu_seqlens_k[b]` and the KV base for batch b is
// `cu_seqlens_k[b]` rows -- exactly the AOT's split-batch view, in which the
// two batch views overlap physically and `kv_batch_stride` is the *used* row
// count of batch 0, not the viewed one.
int run_pack(const void* q, const void* k, const void* v, void* output,
             float* lse, int seq_q, int max_seq_k, int total_seq_k,
             int* cu_seqlens_k, int query_heads, int kv_heads, int batch,
             int q_batch_stride, int kv_batch_stride, float scale,
             cudaStream_t stream, State& state, int prepare_only) {
  if (q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
      cu_seqlens_k == nullptr || lse == nullptr || seq_q <= 0 || batch <= 0 ||
      kv_heads <= 0 || query_heads % kv_heads != 0) {
    return -1;
  }
  const int head_dim = 256;
  const int q_per_kv = query_heads / kv_heads;
  if (q_per_kv != QPerKv::value) return -1;
  int device = -1;
  int multiprocessors = 0;
  if (int rc = device_properties(device, multiprocessors); rc != 0) return rc;
  if (state.device != -1 && state.device != device) return -4;
  state.device = device;

  auto problem = cute::make_tuple(
      cute::make_tuple(QPerKv{}, seq_q),
      cutlass::fmha::collective::VariableLength{max_seq_k, cu_seqlens_k,
                                                total_seq_k},
      head_dim, cute::make_tuple(cute::make_tuple(_1{}, kv_heads), batch));

  const int q_token_stride = query_heads * head_dim;
  const int kv_token_stride = kv_heads * head_dim;
  StrideQPack q_stride = make_stride(
      make_stride(head_dim, q_token_stride), _1{},
      make_stride(make_stride(_0{}, q_per_kv * head_dim), q_batch_stride));
  StrideOPack o_stride = q_stride;
  StrideK kv_stride = make_stride(
      kv_token_stride, _1{},
      make_stride(make_stride(_0{}, head_dim), kv_batch_stride));
  StrideV v_stride = kv_stride;

  // The LSE row index inside one (kv_head, batch) block is `q * q_per_kv + j`
  // -- the packed M mode's own linearisation -- over `rounded_query` = 256
  // slots of which 4*50 = 200 are written. The caller owns the buffer;
  // `apxinf_attention_d256_split_batch_lse_elements()` is the single source of truth for its
  // size and `fa4_d256_split_batch_cuda_adapter.cu` owns the repack into the
  // merge kernel's dense [batch, query_head, token] layout.
  const int rounded_query = ((seq_q * q_per_kv + 127) / 128) * 128;
  StrideLSE lse_stride = make_stride(
      _1{}, make_stride(make_stride(rounded_query, rounded_query),
                        rounded_query * kv_heads));

  typename ResidualPackCfg::Op::Arguments arguments{
      problem,
      {{static_cast<Element const*>(q), q_stride,
        static_cast<Element const*>(k), kv_stride,
        static_cast<Element const*>(v), v_stride},
       scale,
       1.0f,
       1.0f,
       1.0f,
       1.0f},
      // The split-batch AOT emits an LSE tensor and the two batches are
      // combined from it, so unlike vision and language this arm has to emit
      // one too. `cutlass::fast_log(row_sum) + scale_softmax*row_max` is NOT
      // bit-identical to FA4's log-sum-exp; `-DAPXINF_LSE_FA4=1` on this
      // translation unit switches the epilogue to FA4's own
      // `(row_max*scale_log2 + lg2.approx.ftz(row_sum)) * float(ln 2)`, which
      // makes the LSE 1600/1600 bit-equal and therefore the MERGED output
      // 100.000000% at all five `key_tokens` -- see report 27, and note the
      // merge weight is `sigmoid(lse0 - lse1)`, which at `key_tokens = 3436`
      // has no leading term left to absorb a 1-fp32-ULP log difference.
      {static_cast<Element*>(output), o_stride, lse, lse_stride},
      {0, multiprocessors}};
  return finish<ResidualPackCfg>(arguments, state, stream, prepare_only);
}

}  // namespace APXINF_D256_NS

extern "C" size_t apxinf_attention_d256_split_batch_state_size() {
  return sizeof(APXINF_D256_NS::State);
}

// Elements of the native (packed-M) LSE buffer for the action geometry:
// batch * kv_heads * round_up(q_per_kv * seq_q, 128).
extern "C" size_t apxinf_attention_d256_split_batch_lse_elements(int seq_q, int kv_heads,
                                                  int batch) {
  const int rounded_query =
      ((seq_q * APXINF_PACK_GQA_QPK + 127) / 128) * 128;
  return size_t(batch) * size_t(kv_heads) * size_t(rounded_query);
}

// Row index inside the (kv_head, batch) block for query head `q_head` and
// token `token`. Exported so the repack in the adapter and the test cannot
// drift from the stride the kernel was given.
extern "C" size_t apxinf_attention_d256_split_batch_lse_index(int seq_q, int kv_heads,
                                               int batch_index, int q_head,
                                               int token) {
  const int rounded_query =
      ((seq_q * APXINF_PACK_GQA_QPK + 127) / 128) * 128;
  const int kv_head = q_head / APXINF_PACK_GQA_QPK;
  const int lane = q_head % APXINF_PACK_GQA_QPK;
  return size_t(batch_index) * kv_heads * rounded_query +
         size_t(kv_head) * rounded_query +
         size_t(token) * APXINF_PACK_GQA_QPK + size_t(lane);
}

extern "C" void apxinf_attention_d256_split_batch_state_destroy(void* state_ptr) {
  using APXINF_D256_NS::State;
  if (state_ptr == nullptr) return;
  auto& state = *static_cast<State*>(state_ptr);
  if (state.device >= 0) cudaSetDevice(state.device);
  if (state.lse != nullptr) cudaFree(state.lse);
  if (state.workspace != nullptr) cudaFree(state.workspace);
  state = State{};
}

extern "C" int apxinf_attention_d256_split_batch_forward(
    const void* q, const void* k, const void* v, void* output, float* lse,
    int seq_q, int max_seq_k, int total_seq_k, int* cu_seqlens_k,
    int query_heads, int kv_heads, int batch, int q_batch_stride,
    int kv_batch_stride, float scale, cudaStream_t stream, void* state_ptr,
    int prepare_only) {
  if (state_ptr == nullptr) return -1;
  return APXINF_D256_NS::run_pack(
      q, k, v, output, lse, seq_q, max_seq_k, total_seq_k, cu_seqlens_k,
      query_heads, kv_heads, batch, q_batch_stride, kv_batch_stride, scale,
      stream, *static_cast<APXINF_D256_NS::State*>(state_ptr), prepare_only);
}
