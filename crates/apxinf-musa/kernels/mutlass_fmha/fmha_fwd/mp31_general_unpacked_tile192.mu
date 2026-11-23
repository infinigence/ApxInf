// Frozen mp_31 D256 FMHA kernel, vendored from MATE.
//
// Do not edit by hand: regenerate with `tools/freeze_kernel.py`. The config
// block below `// FMHA config` is the render's, verbatim; the host entry is
// ours and `tools/fmha_fwd_host.cpp.in` is where it is written.
//
//   rendered from  fmha_fwd_be3f51421622725ce739b37c33f3c52b82fcd2a4010f833c3e0e26846a17650c.mu
//   render sha256  946c354ecb826ee8cba7660f38d347f7c30618a93215c23827f4253c02525c6f
//   MATE           0.2.7
//
// See kernels/mutlass_fmha/README.md for provenance, the edit list, and the
// launch contract.

#include <mute/tensor.hpp>
#include <mutlass/mutlass.h>
#include <mutlass/device_kernel.h>
#include "mate/attention/fmha/fmha.hpp"
#include "mate/attention/fmha/fmha_fwd_kernel.hpp"
#include "mate/attention/fmha/mp31_fmha_fwd_tme_warpspecialized.hpp"
#include "mate/attention/fmha/epilogue_fwd.hpp"
#include "mate/attention/fmha/tile_scheduler.hpp"
#include "mate/attention/fmha/metadata.hpp"

using namespace mute;

// Host entry for one frozen mp_31 FMHA instantiation.
//
// `tools/freeze_kernel.py` puts this after the render's includes and its
// `constexpr` config block. That block is the render's, verbatim, and it is what
// makes the instantiation what it is; this file is the host half, which the
// render wrote against tvm_ffi and this port writes against its own contract.
//
// What this replaces
// ------------------
// The render's entry was a `tvm_ffi` function of 36 arguments, one
// `ffi::TensorView` per tensor, and it derived about thirty numbers from those
// views. `attention::mqa_bf16` calls it one way, so the entry is narrowed to
// the values it actually reads and the rest are named from the contract:
//
//   q, out   bf16 [tokens, 8, 256] contiguous
//   k, v     bf16 [key_tokens, 256] contiguous   (head_dim 256, one kv head)
//   lse      f32  [8, tokens]                    the epilogue writes it
//
// Those shapes fix every stride the render read:
//
//   q_row_stride = 8*256 = 2048   q_head_stride = 256   q_batch_stride = 0
//   k,v row      = 256            k,v head      = 256   k,v batch      = 0
//   o_row_stride = 2048           o_head_stride = 256   o_batch_stride = 0
//   lse_head_stride = tokens      ([8, tokens] f32 is token-major)
//
// Two consequences of the contract that are worth stating because the render
// could not assume them:
//
// * K and V are contiguous `[key_tokens, 256]`, so row stride and head stride
//   are the same number. A `[key_tokens, 1, 256]` view gives the same two
//   values, which is why the bridge's reshape is not reproduced here.
// * `key_tokens` bounds the KV rows that take part, and the kernel reads no row
//   past it: the render set `total_k = k.size(0)`, and the instance here passes
//   exactly `key_tokens`. A reserved prefix past that is never read, so a NaN
//   there cannot reach the softmax.
//
// Dropped from the render's entry
// -------------------------------
// The `tvm_ffi` device guard (the caller already owns the device); the
// `query_device_multiprocessor_count()` probe and `mp_margin` (see below); the
// two `o_accum`/`lse_accum` `FromEnvAlloc`s, which exist for the split-KV
// `fmha_fwd_combine` step and are never read when `IsSplit` is false — this
// instantiation's `HasMetadata` is false, so nothing on this path allocates,
// which is what makes the launch capturable; the `dry_run` branch; and
// `MATE_CHECK_MUSA_KERNEL_LAUNCH`, whose `TVM_FFI_CHECK_MUSA_ERROR` reporting is
// replaced by a returned message.
//
// `SingleTileScheduler::get_grid_shape` takes the SM count and does not read it
// (`tile_scheduler.hpp`), and `MATE_MUSA_DRIVER_CHECK` is not reached either, so
// the probe is not replaced by a cheaper query: it is simply gone, and
// `params.num_mp`, which nothing reads, is 0.

#include "fmha_fwd_args.hpp"

// `musaStream_t`, `musaGetErrorString` and `musaSuccess`. The render reached
// these through `tvm/ffi/extra/musa/*`; the vendored headers do not.
#include <musa_runtime.h>

#include <cstdint>
#include <type_traits>

extern "C" const char* apxinf_musa_mp31_fmha_fwd_tile192_general(
    const void* q,
    const void* k,
    const void* v,
    void* out,
    float* lse,
    const uint32_t* cu_seqlens_q,
    const uint32_t* cu_seqlens_k,
    int32_t tokens,
    int32_t key_tokens,
    void* stream) {
  if (tokens <= 0 || key_tokens <= 0) {
    return "the mp_31 FMHA launcher needs tokens > 0 and key_tokens > 0";
  }

  // The render's `num_splits` argument. `HasMetadata` is false in every frozen
  // instantiation, which makes `num_split` below 1 whatever this is and skips
  // the `get_num_splits` heuristic; MATE's own caller passed 0 here.
  int const num_splits = 0;

  // The launcher takes the scale rather than a parameter because the render's
  // callers all use `1 / sqrt(head_dim)`, and `apxinf-musa`'s `mqa_bf16`
  // contract fixes it there too. Passing it would invite a second value that no
  // caller produces.
  float const softmax_scale = 1.0f / 16.0f;  // 1/sqrt(256)

  constexpr int kNumHead = 8;
  constexpr int kNumHeadKv = 1;
  constexpr int kHeadDim = 256;
  constexpr int kBatch = 1;

  // FMHA config
  constexpr bool HasQv = false;
  constexpr bool OnlyQv = false;
  constexpr bool HasCuseqlensQ = true;
  constexpr bool HasCuseqlensK = true;
  constexpr bool HasSequsedQ = false;
  constexpr bool HasSequsedK = false;
  constexpr bool HasCuseqlensKNew = false;
  constexpr bool HasKvBatchIdx = false;
  constexpr bool HasLeftpadK = false;
  constexpr bool HasQDescale = false;
  constexpr bool HasKDescale = false;
  constexpr bool HasVDescale = false;
  constexpr bool PagedKV = false;
  constexpr bool HasSoftcap = false;
  constexpr bool IsAppendKV = false;
  constexpr bool HasLearnableSink = false;
  constexpr bool IsCausal = false;
  constexpr bool IsLocal = false;
  constexpr int HeadRatio = 8;
  constexpr bool UseLsuLoadK = true;
  constexpr bool UseLsuLoadV = false;
  constexpr bool IsRotary = false;
  constexpr bool IsRotaryInterleaved = false;
  constexpr bool HasSeqlensRotary = false;

  constexpr int TileM = 192;
  constexpr int TileN = 64;
  constexpr int HeadDimQK = 256;
  constexpr int HeadDimVO = 256;
  constexpr int StagesK = 1;
  constexpr int StagesV = 1;
  constexpr int NumQKConsumers = 3;
  constexpr int NumPVConsumers = 3;
  constexpr int TileHeadDim = 0;
  constexpr bool IsPackGQA = false;
  constexpr bool IsEvenHeadDim = true;
  constexpr bool HasMetadata = false;
  constexpr bool EnableCP = false;
  constexpr bool HasAttentionChunk = false;

  // If no metadata, fallback to SingleTileScheduler without split.
  constexpr bool IsSplit = HasMetadata;
  int num_split = IsSplit ? num_splits : 1;

  using Element = mutlass::bfloat16_t;
  using ElementOut = std::conditional_t<
      std::is_same_v<Element, mutlass::float_e4m3_t> || std::is_same_v<Element, mutlass::float_e5m2_t>,
      mutlass::bfloat16_t,
      Element>;
  using ElementAccumulator = float;
  using ElementPartial = float;
  using ElementSink = mutlass::bfloat16_t;

  using TileShape = Shape<Int<TileM>, Int<TileN>, Int<HeadDimQK>>;

  using CollectiveMainloop =
    mate::attention::fmha::Mp31FmhaFwdTmeWarpSpecialized<Element,
                                                         ElementAccumulator,
                                                         TileShape,
                                                         StagesK,
                                                         StagesV,
                                                         HeadDimVO,
                                                         NumQKConsumers,
                                                         HasCuseqlensQ,
                                                         HasCuseqlensK,
                                                         HasCuseqlensKNew,
                                                         HasKvBatchIdx,
                                                         HasSequsedQ,
                                                         HasSequsedK,
                                                         HasLeftpadK,
                                                         HasQDescale,
                                                         HasKDescale,
                                                         HasVDescale,
                                                         PagedKV,
                                                         UseLsuLoadK,
                                                         UseLsuLoadV,
                                                         IsCausal,
                                                         IsLocal,
                                                         HasLearnableSink,
                                                         HasSoftcap,
                                                         IsAppendKV,
                                                         HasQv,
                                                         OnlyQv,
                                                         HeadRatio,
                                                         IsPackGQA,
                                                         IsSplit /*Split*/,
                                                         IsRotary,
                                                         IsRotaryInterleaved,
                                                         HasSeqlensRotary,
                                                         EnableCP,
                                                         HasAttentionChunk,
                                                         NumPVConsumers,
                                                         TileHeadDim>;
  using CollectiveEpilogue =
    mate::attention::fmha::CollectiveEpilogueFwd<ElementOut,
                                                 CollectiveMainloop::TileShapePDVFull,
                                                 HasCuseqlensQ,
                                                 HasSequsedQ,
                                                 IsPackGQA,
                                                 CollectiveMainloop::NumMmaThreads,
                                                 IsEvenHeadDim,
                                                 HeadRatio>;

  using TileScheduler = std::conditional_t<HasMetadata,
    mate::attention::fmha::StaticPersistentTileScheduler<IsSplit>,
    mate::attention::fmha::SingleTileScheduler<HasCuseqlensQ,
                                               HasSequsedQ,
                                               IsPackGQA,
                                               TileM,
                                               IsSplit>
  >;

  using Kernel = mate::attention::fmha::FmhaFwdKernelWarpSpecialized<
    CollectiveMainloop,
    CollectiveEpilogue,
    TileScheduler>;

  int const seqlen_q = tokens;
  int const seqlen_k = key_tokens;
  int const total_q = tokens;
  int const total_k = key_tokens;
  int const headdim = kHeadDim;
  int const headdim_rounded = mate::attention::fmha::round_up_headdim(headdim);

  // `HasMetadata` is false, so the three metadata pointers the render wired
  // through `int32_ptr_sentinel` are null.
  int32_t* num_splits_dynamic_ptr = nullptr;
  int32_t* batch_table_ptr = nullptr;
  int32_t* num_m_blocks_ptr = nullptr;

  mate::attention::fmha::FmhaFwdParams params {
    // QKV params
    .q_ptr = const_cast<void*>(q),
    .k_ptr = const_cast<void*>(k),
    .v_ptr = const_cast<void*>(v),
    .q_batch_stride = 0,
    .k_batch_stride = 0,
    .v_batch_stride = 0,
    .q_row_stride = kNumHead * kHeadDim,
    .k_row_stride = kHeadDim,
    .v_row_stride = kHeadDim,
    .q_head_stride = kHeadDim,
    .k_head_stride = kHeadDim,
    .v_head_stride = kHeadDim,
    .v_dim_stride = 1,
    .h = kNumHead,
    .h_k = kNumHeadKv,

    // Output
    .o_ptr = out,
    .o_batch_stride = 0,
    .o_row_stride = kNumHead * kHeadDim,
    .o_head_stride = kHeadDim,

    // LSE
    .softmax_lse_ptr = lse,
    .lse_batch_stride = 0,
    .lse_head_stride = static_cast<int64_t>(tokens),

    // FP8
    .q_descale_ptr = nullptr,
    .k_descale_ptr = nullptr,
    .v_descale_ptr = nullptr,
    .q_descale_batch_stride = 0,
    .q_descale_head_stride = 0,
    .k_descale_batch_stride = 0,
    .k_descale_head_stride = 0,
    .v_descale_batch_stride = 0,
    .v_descale_head_stride = 0,

    // Shape params
    .b = kBatch,
    .seqlen_q = seqlen_q,
    .seqlen_k = seqlen_k,
    .seqlen_knew = 0,
    .total_q = total_q,
    .total_k = total_k,
    .total_knew = 0,
    .d = headdim,
    .d_rounded = headdim_rounded,
    .dv = headdim,
    .dv_rounded = headdim_rounded,
    .rotary_dim = 0,

    .b_k = kBatch,

    // Scale factor
    .scale_softmax = softmax_scale,
    .softcap = 0.0f,

    // Ragged params. `FmhaFwdParams` holds these as non-const pointers, which is
    // why they are cast: the kernel reads them, and the caller's arrays are
    // device metadata it owns.
    .cu_seqlens_q = const_cast<uint32_t*>(cu_seqlens_q),
    .cu_seqlens_k = const_cast<uint32_t*>(cu_seqlens_k),
    .cu_seqlens_knew = nullptr,
    .leftpad_k = nullptr,
    .seqused_q = nullptr,
    .seqused_k = nullptr,

    // AppendKV
    .knew_ptr = nullptr,
    .vnew_ptr = nullptr,
    .knew_batch_stride = 0,
    .vnew_batch_stride = 0,
    .knew_row_stride = 0,
    .vnew_row_stride = 0,
    .knew_head_stride = 0,
    .vnew_head_stride = 0,

    // Qv
    .qv_ptr = nullptr,
    .qv_batch_stride = 0,
    .qv_row_stride = 0,
    .qv_head_stride = 0,

    // Rotary
    .rotary_cos_ptr = nullptr,
    .rotary_sin_ptr = nullptr,
    .seqlens_rotary = nullptr,

    .kv_batch_idx = nullptr,

    // Learnable Sink
    .learnable_sink_ptr = nullptr,

    // Paged params
    .page_table = nullptr,
    .page_table_batch_stride = 0,
    .page_size = 1,
    .num_pages = 0,

    .num_mp = 0,
    .num_splits = num_split,
    .num_splits_dynamic_ptr = num_splits_dynamic_ptr,
    .batch_table_ptr = batch_table_ptr,

    .is_causal = IsCausal,
    .is_local = IsLocal,
    .is_rotary_interleaved = IsRotaryInterleaved,
    .window_size_left = -1,
    .window_size_right = -1,
    .attention_chunk = 0,

    .is_fp8 = std::is_same_v<Element, mutlass::float_e4m3_t> || std::is_same_v<Element, mutlass::float_e5m2_t>,
  };
  // `num_split` is 1 here (`HasMetadata` false), so the split heuristic below
  // never runs; the branch is kept because `params.num_splits` feeds the
  // scheduler arguments and deleting it would be a second difference from the
  // render for no gain.
  if (params.num_splits <= 0) {
    params.num_splits = mate::attention::fmha::get_num_splits(params, TileM, TileN, false);
  }

  int const seqlen_q_mainloop = params.total_q;
  int const batch_q_mainloop = 1;
  int const batch_k_mainloop = 1;

  typename CollectiveMainloop::Arguments mainloop_args {
    // QKV Args
    .ptr_Q = static_cast<Element const*>(params.q_ptr),
    .shape_Q = {seqlen_q_mainloop, params.d, params.h, batch_q_mainloop},
    .stride_Q = {params.q_row_stride, _1{}, params.q_head_stride, 0},
    .ptr_K = static_cast<Element*>(params.k_ptr),
    .shape_K = {params.total_k, params.d, params.h_k, batch_k_mainloop},
    .stride_K = {params.k_row_stride, _1{}, params.k_head_stride, 0},
    .ptr_V = static_cast<Element*>(params.v_ptr),
    .headdim_V = params.dv,
    .stride_V = {params.v_row_stride, _1{}, params.v_head_stride, 0},

    // AppendKV
    .ptr_K_new = static_cast<Element const*>(params.knew_ptr),
    .shape_K_new = {0, params.d, params.h_k, params.b},
    .stride_K_new = {params.knew_row_stride, _1{}, params.knew_head_stride, 0},
    .ptr_V_new = static_cast<Element*>(params.vnew_ptr),
    .stride_V_new = {params.vnew_row_stride, _1{}, params.vnew_head_stride, 0},

    // QV
    .ptr_Qv = static_cast<Element const*>(params.qv_ptr),
    .shape_Qv = {0, 0, 0, 0},
    .stride_Qv = {0, _1{}, 0, 0},

    // Rotary
    .ptr_rotary_cos = static_cast<Element const*>(params.rotary_cos_ptr),
    .shape_rotary = {0, params.rotary_dim / 2},
    .stride_rotary_cos = {params.rotary_dim / 2, _1{}},
    .ptr_rotary_sin = static_cast<Element const*>(params.rotary_sin_ptr),
    .stride_rotary_sin = {params.rotary_dim / 2, _1{}},

    // PageTable Args
    .ptr_pagetable = params.page_table,
    .shape_pagetable = {params.b, 0},
    .stride_pagetable = {params.page_table_batch_stride, _1{}},

    // Scale
    .softmax_scale = params.scale_softmax,
    .ptr_q_descale = static_cast<float const*>(params.q_descale_ptr),
    .stride_q_descale = {params.q_descale_batch_stride, params.q_descale_head_stride},
    .ptr_k_descale = static_cast<float const*>(params.k_descale_ptr),
    .stride_k_descale = {params.k_descale_batch_stride, params.k_descale_head_stride},
    .ptr_v_descale = static_cast<float const*>(params.v_descale_ptr),
    .stride_v_descale = {params.v_descale_batch_stride, params.v_descale_head_stride},

    // LocalMask
    .window_size_left = params.window_size_left,
    .window_size_right = params.window_size_right,
    .attention_chunk = params.attention_chunk,

    // Sink Tensor
    .ptr_learnable_sink = static_cast<ElementSink const*>(params.learnable_sink_ptr),

    .softcap_val = params.softcap,

    .num_splits = params.num_splits,
    .kv_batch_idx = static_cast<uint32_t const*>(params.kv_batch_idx),
    .cu_seqlens_q = static_cast<uint32_t const*>(params.cu_seqlens_q),
    .cu_seqlens_k = static_cast<uint32_t const*>(params.cu_seqlens_k),
    .cu_seqlens_k_new = static_cast<uint32_t const*>(params.cu_seqlens_knew),
    .seqused_q = static_cast<uint32_t const*>(params.seqused_q),
    .seqused_k = static_cast<uint32_t const*>(params.seqused_k),
    .leftpad_k = static_cast<uint32_t const*>(params.leftpad_k),
    .seqlens_rotary = static_cast<uint32_t const*>(params.seqlens_rotary),

    // CP params — the render's `cp_world_size` is 1 and `cp_rank` 0 for every
    // caller here, and `EnableCP` is false in the instantiation.
    .cp_world_size = 1,
    .cp_rank = 0,
    .cp_tot_seqused_k = nullptr,
  };

  typename CollectiveEpilogue::Arguments epilogue_args {
    .ptr_O = static_cast<ElementOut*>(params.o_ptr),
    .shape_O = {params.total_q, params.dv, params.h, 1, params.num_splits},
    .stride_O = {params.o_row_stride, _1{}, params.o_head_stride, 0, 0},
    .ptr_O_partial = nullptr,
    .stride_O_partial = {0, _1{}, 0, 0, 0},

    .ptr_LSE = static_cast<float*>(params.softmax_lse_ptr),
    .stride_LSE = {_1{}, params.lse_head_stride, 0, 0},
    .ptr_LSE_partial = nullptr,
    .stride_LSE_partial = {_1{}, 0, 0, 0},

    .nheads_kv = params.h_k,
    .cu_seqlens = static_cast<uint32_t const*>(params.cu_seqlens_q),
    .seqused = static_cast<uint32_t const*>(params.seqused_q),
  };

  typename TileScheduler::Arguments scheduler_args {
    .num_blocks = mutlass::ceil_div(params.seqlen_q * (IsPackGQA ? HeadRatio : 1), TileM),
    .num_head = IsPackGQA ? params.h_k : params.h,
    .num_batch = params.b,
    .num_splits = params.num_splits,
    .qhead_per_khead = HeadRatio,
    .seqlen_q = params.seqlen_q,
    .seqlen_k = params.seqlen_k,
    .headdim = params.d,
    .headdim_v = params.dv,
    .element_size = sizeof(Element),
    .cu_seqlens = params.cu_seqlens_q,
    .seqused = params.seqused_q,

    .num_splits_dynamic_ptr = num_splits_dynamic_ptr,
    .batch_table_ptr = batch_table_ptr,
    .num_m_blocks_ptr = num_m_blocks_ptr,
  };

  typename Kernel::Arguments kernel_arguments {
    .mainloop = mainloop_args,
    .epilogue = epilogue_args,
    .scheduler = scheduler_args,
  };

  typename Kernel::Params kernel_params = Kernel::to_underlying_arguments(kernel_arguments);

  musaStream_t musa_stream = static_cast<musaStream_t>(stream);
  // The render passed the queried SM count here; this instantiation's
  // `SingleTileScheduler` does not read it.
  dim3 grid_dim = TileScheduler::get_grid_shape(kernel_params.scheduler, /*num_mp*/ 0);
  mutlass::device_kernel<Kernel><<<grid_dim, Kernel::MaxThreadsPerBlock,
                                   Kernel::SharedStorageSize, musa_stream>>>(kernel_params);

  musaError_t launched = musaGetLastError();
  if (launched != musaSuccess) {
    return musaGetErrorString(launched);
  }
  return nullptr;
}
