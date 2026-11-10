#pragma once

#include <mutlass/fast_math.h>
#include <mutlass/mutlass.h>

#include <mute/tensor.hpp>
#include <mutlass/gemm/collective/collective_builder.hpp>

#include "../../common/mma_mp31_sqmma.hpp"
#include "../../common/numeric_conversion.hpp"
#include "block_info.hpp"
#include "load_primitive_builder.hpp"
#include "mask.hpp"
#include "pack_gqa.hpp"
#include "paged_kv.hpp"
#include "pipeline_ws.hpp"
#include "rotary.hpp"
#include "seqlen.hpp"
#include "softmax.hpp"
#include "utils.hpp"

#define SHOW(x)   \
  print(#x ": "); \
  print(x);       \
  print("\n")

namespace mate::attention::fmha {

using namespace mute;

template <class Element_,
          class ElementAccumulator_,
          class TileShape_,
          int  StagesK_,
          int  StagesV_,
          int  HeadDimV_,
          int  NumQKConsumers_,
          bool HasCuseqlensQ_,
          bool HasCuseqlensK_,
          bool HasCuseqlensKNew_,
          bool HasKvBatchIdx_,
          bool HasSequsedQ_,
          bool HasSequsedK_,
          bool HasLeftpadK_,
          bool HasQDscale_,
          bool HasKDscale_,
          bool HasVDscale_,
          bool IsPagedKV_,
          bool UseLSULoadK_,
          bool UseLSULoadV_,
          bool IsCausal_,
          bool IsLocal_,
          bool HasLearnableSink_,
          bool HasSoftcap_,
          bool IsAppendKV_,
          bool HasQv_,
          bool OnlyQv_,
          int  HeadRatio_,
          bool IsPackGQA_,
          bool Split_,
          bool IsRotary_,
          bool IsRotaryInterleaved_,
          bool HasSeqlensRotary_,
          bool EnableCP_,
          bool HasAttentionChunk_,
          int  NumPVConsumers_ = NumQKConsumers_,
          int  TileHeadDim_    = 0>
struct Mp31FmhaFwdTmeWarpSpecialized {
  using Element            = Element_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementSink        = mutlass::bfloat16_t;

  using TileShape                    = TileShape_;
  static constexpr int TileM         = get<0>(TileShape{});
  static constexpr int TileN         = get<1>(TileShape{});
  static constexpr int TileK         = get<2>(TileShape{});  // Entry for future headdim tiling
  static constexpr int HeadDimQK     = get<2>(TileShape{});
  static constexpr int HeadDimVO     = HeadDimV_;
  static constexpr int HeadRatio     = HeadRatio_;
  static constexpr int TileHeadRatio = std::min(HeadRatio, TileM);

  static constexpr int NumLoadWarpSquads = 1;
  static constexpr int NumQKConsumers    = NumQKConsumers_;
  static constexpr int NumPVConsumers    = NumPVConsumers_;
  // For Pinghu, the consumer granularity is WarpSquad
  static constexpr int NumMmaWarpSquads = std::max(NumQKConsumers, NumPVConsumers);
  static constexpr int NumMmaThreads    = NumMmaWarpSquads * mutlass::NumThreadsPerWarpSquad;
  static constexpr int NumQKMmaThreads  = NumQKConsumers * mutlass::NumThreadsPerWarpSquad;
  static constexpr int NumPVMmaThreads  = NumPVConsumers * mutlass::NumThreadsPerWarpSquad;
  static constexpr int NumTransWarps    = mutlass::NumWarpsPerWarpSquad;
  static constexpr int NumTransThreads  = NumTransWarps * mutlass::NumThreadsPerWarp;

  static_assert(TileM % NumQKConsumers == 0);
  static_assert(TileM % NumPVConsumers == 0);

  static constexpr bool HasCuseqlensQ    = HasCuseqlensQ_;
  static constexpr bool HasCuseqlensK    = HasCuseqlensK_;
  static constexpr bool HasCuseqlensKNew = HasCuseqlensKNew_;
  static constexpr bool HasKvBatchIdx    = HasKvBatchIdx_;
  static constexpr bool HasSequsedQ      = HasSequsedQ_;
  static constexpr bool HasSequsedK      = HasSequsedK_;
  static constexpr bool HasLeftpadK      = HasLeftpadK_;
  static constexpr bool HasQv            = HasQv_;
  static constexpr bool OnlyQv           = OnlyQv_;
  static_assert(!OnlyQv || HasQv, "OnlyQv requires HasQv");

  static constexpr bool HasQDescale = HasQDscale_;
  static constexpr bool HasKDescale = HasKDscale_;
  static constexpr bool HasVDescale = HasVDscale_;

  static constexpr bool IsPackGQA = IsPackGQA_;
  static constexpr bool IsPagedKV = IsPagedKV_;
  static constexpr bool IsCausal  = IsCausal_;
  static constexpr bool IsLocal   = IsLocal_;
  static constexpr bool Split     = Split_;

  static constexpr bool HasLearnableSink = HasLearnableSink_;
  static constexpr bool HasSoftcap       = HasSoftcap_;

  static constexpr bool EnableCP          = EnableCP_;
  static constexpr bool HasAttentionChunk = HasAttentionChunk_;

  static constexpr bool IsAppendKV          = IsAppendKV_;
  static constexpr bool IsRotary            = IsRotary_;
  static constexpr bool IsRotaryInterleaved = IsRotaryInterleaved_;
  static constexpr bool HasSeqlensRotary    = HasSeqlensRotary_;
  static_assert(!EnableCP || !IsRotary || HasSeqlensRotary,
                "CP AppendKV rotary requires global rotary sequence offsets");

  static constexpr int TileHeadDim = TileHeadDim_;
  static_assert(TileHeadDim >= 0, "TileHeadDim must be non-negative");
  static constexpr int TileHeadDimQK = TileHeadDim > 0 && !HasQv ? TileHeadDim : HeadDimQK;
  static constexpr int TileHeadDimVO = TileHeadDim > 0 ? TileHeadDim : HeadDimVO;
  static_assert(HeadDimQK % TileHeadDimQK == 0 && HeadDimVO % TileHeadDimVO == 0);
  static constexpr int  QKIterations      = HeadDimQK / TileHeadDimQK;
  static constexpr int  PVIterations      = HeadDimVO / TileHeadDimVO;
  static constexpr bool IsHeadDimTiled    = QKIterations > 1 || PVIterations > 1;
  static constexpr bool InKernelTranspose = HasQv && !IsHeadDimTiled;
  static_assert(TileHeadDim == 0 || IsHeadDimTiled, "TileHeadDim must split at least one operand");
  static_assert(!HasQv || QKIterations == 1, "QV keeps K whole and chunks only the V dimension");

  static constexpr bool IsFP8 =
      mute::is_same_v<Element, mutlass::float_e4m3_t> || mute::is_same_v<Element, mutlass::float_e5m2_t>;
  static constexpr int MaxOffset = IsFP8 ? 8 : 0;

  static constexpr int UsePackGQATMELoad =
      IsPackGQA_ && mutlass::is_pow2<TileHeadRatio>::value && TileM % TileHeadRatio == 0;
  static constexpr bool UseTMELoadQ = !IsPackGQA_ || UsePackGQATMELoad;
  static constexpr bool UseLSULoadQ = !UseTMELoadQ;

  static constexpr bool UseLSULoadK    = UseLSULoadK_;
  static constexpr bool UseLSULoadV    = UseLSULoadV_;
  static constexpr bool ReuseKPStorage = HasQv && IsHeadDimTiled && !IsFP8;

  static_assert(!IsPagedKV || (UseLSULoadK && UseLSULoadV) || (!UseLSULoadK && !UseLSULoadV),
                "KV Load methods must be same if paged KV is enabled!");
  static_assert(IsPagedKV || ((HasCuseqlensK || HasLeftpadK) && UseLSULoadK) || !HasCuseqlensK,
                "Load K support LSU Only if ragged KV or leftpad_k is enabled!");

  static constexpr int NumProducerThreads =
      UseLSULoadQ || UseLSULoadK || UseLSULoadV ? mutlass::NumThreadsPerWarpSquad : mutlass::NumThreadsPerWarp;
  static constexpr bool SingleProducerWarp = NumProducerThreads == mutlass::NumThreadsPerWarp;

  static constexpr bool IntraWarpSquadOverlap = !HasQv || (InKernelTranspose && StagesV_ >= 2);

  static constexpr bool IsMmaPvRS = false;

  static constexpr int StagesQ         = 1;
  static constexpr int StagesK         = StagesK_;
  static constexpr int StagesV         = StagesV_;
  static constexpr int StagesKPipeline = ReuseKPStorage ? 1 : StagesK;
  static constexpr int StagesVt        = StagesV;
  static constexpr int StagesQv        = 1;
  static constexpr int StagesKNew      = IsHeadDimTiled ? 1 : StagesK;
  static constexpr int StagesVNew      = HasQv && IsHeadDimTiled ? 1 : StagesV;

  static_assert(HasQv || !IsHeadDimTiled || StagesK == StagesV);
  static_assert(!IsHeadDimTiled || StagesV >= 2,
                "A reused operand generation pipeline requires at least two stages on PH1.");
  static_assert(HasQv || !IsHeadDimTiled || UseLSULoadK == UseLSULoadV,
                "A shared K/V pipeline requires the same K/V load mechanism.");
  static_assert(NumQKConsumers == NumPVConsumers, "FMHA QK and PV pipelines require the same consumer arrival count.");
  static constexpr int MaxBarPerStageRatio = IsHeadDimTiled ? 2 : 4;

  static constexpr int AdditionalBarrier = static_cast<int>(FwdNamedBarriers::NumFwdNamedBarriers) +
                                           static_cast<int>(mutlass::arch::AsyncBarrier::ReservedAsyncBarrierCount);

  using BarPerStageRatioHelper =
      mutlass::Mp31PipelineWarpSpecializedBarrierRatio<MaxBarPerStageRatio,
                                                       AdditionalBarrier,
                                                       StagesQ,
                                                       HasQv ? StagesQv : 0,
                                                       StagesKPipeline,
                                                       IsHeadDimTiled && !HasQv ? 0 : StagesV,
                                                       InKernelTranspose ? StagesVt : 0,
                                                       IsAppendKV ? StagesKNew + StagesVNew : 0>;

  static constexpr int BarPerStageRatio = BarPerStageRatioHelper::value;
  static_assert(BarPerStageRatio > 0, "FMHA async barrier storage exceeds hardware limit even with BarPerStageRatio=1");

  static constexpr int Alignment = 32 / sizeof_bits_v<Element>;

  using ShapeQKV  = Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
  using StrideQKV = Stride<int64_t, _1, int64_t, int64_t>;

  // ((head_ratio, seqlen_q), d, nheads_kv, batch)
  using ShapeQPacked_  = Shape<Shape<int32_t, int32_t>, int32_t, int32_t, int32_t>;
  using StrideQPacked_ = Stride<Stride<int64_t, int64_t>, _1, int64_t, int64_t>;

  using ShapeQPacked  = std::conditional_t<!IsPackGQA, ShapeQKV, ShapeQPacked_>;
  using StrideQPacked = std::conditional_t<!IsPackGQA, StrideQKV, StrideQPacked_>;

  using ShapePageTable  = Shape<int32_t, int32_t>;  // (batch, max_num_pages_per_seq)
  using StridePageTable = Stride<int64_t, _1>;

  using ShapeRotary  = Shape<int32_t, int32_t>;
  using StrideRotary = Stride<int64_t, _1>;

  using StrideDescale = Stride<int64_t, int64_t>;

  using NamedBarrier = mutlass::arch::AsyncBarrier;

  using SeqlenInfo = SeqlenInfoQK<HasCuseqlensQ,
                                  HasSequsedQ,
                                  HasCuseqlensK,
                                  HasSequsedK,
                                  HasLeftpadK,
                                  IsAppendKV,
                                  HasCuseqlensKNew,
                                  HasSeqlensRotary,
                                  EnableCP>;
  using BlockInfo =
      BlockInfo<SeqlenInfo, TileM, TileN, HeadRatio, IsCausal, IsLocal, IsPackGQA, HasAttentionChunk, Split, EnableCP>;

  // NOTE: RoPE not implemented yet.
  // using Rotary = Rotary<TileN, TileK, NumMmaThreads, Element>;

  using PackGQAManager   = PackGQAManager<Element, HeadRatio, TileM, TileHeadDimQK, NumProducerThreads>;
  using TileMPack        = Shape<Int<TileHeadRatio>, Int<TileM / TileHeadRatio>>;
  using LayoutMPack      = decltype(make_layout(TileMPack{}));
  using PackGQATileShape = Shape<TileMPack, Int<TileHeadDimQK>>;
  using PackGQvManager =
      ::mate::attention::fmha::PackGQAManager<Element, HeadRatio, TileM, TileHeadDimVO, NumProducerThreads>;
  using PackGQvTileShape = Shape<TileMPack, Int<TileHeadDimVO>>;

  // Tile View
  using TileShapeQKD = Shape<Int<TileM>, Int<TileN>, Int<TileHeadDimQK>>;
  using TileShapeQvD = Shape<Int<TileM>, Int<TileN>, Int<TileHeadDimVO>>;
  using TileShapePDV = Shape<Int<TileM>, Int<TileHeadDimVO>, Int<TileN>>;

  using TileShapeQKDFull = Shape<Int<TileM>, Int<TileN>, Int<HeadDimQK>>;
  using TileShapePDVFull = Shape<Int<TileM>, Int<HeadDimVO>, Int<TileN>>;

  using AtomLayoutQK = Layout<Shape<Int<NumQKConsumers>, _1, _1>>;

  using TiledMmaQK = decltype(mute::make_tiled_mma(mute::MP31::SQMMA::ss_op_selector<Element,
                                                                                     Element,
                                                                                     ElementAccumulator,
                                                                                     TileShapeQKD,
                                                                                     TCE::Major::K,
                                                                                     TCE::Major::K,
                                                                                     Int<TileM / NumQKConsumers>>(),
                                                   AtomLayoutQK{}));
  using TiledMmaQv = decltype(mute::make_tiled_mma(mute::MP31::SQMMA::ss_op_selector<Element,
                                                                                     Element,
                                                                                     ElementAccumulator,
                                                                                     TileShapeQvD,
                                                                                     TCE::Major::K,
                                                                                     TCE::Major::K,
                                                                                     Int<TileM / NumQKConsumers>>(),
                                                   AtomLayoutQK{}));

  using AtomLayoutPV = Layout<Shape<Int<NumPVConsumers>, _1, _1>>;

  using TiledMmaPV = decltype(mute::make_tiled_mma(mute::MP31::SQMMA::ss_op_selector<Element,
                                                                                     Element,
                                                                                     ElementAccumulator,
                                                                                     TileShapePDV,
                                                                                     TCE::Major::K,
                                                                                     TCE::Major::MN,
                                                                                     Int<TileM / NumPVConsumers>>(),
                                                   AtomLayoutPV{}));

  static_assert(!HasQv || mute::is_same_v<typename TiledMmaQK::AtomLayoutC_TV, typename TiledMmaQv::AtomLayoutC_TV>,
                "QK and QV accumulator thread-value mappings must match");
  static_assert(!HasQv || mute::is_same_v<typename TiledMmaQK::ThrLayoutVMNK, typename TiledMmaQv::ThrLayoutVMNK>,
                "QK and QV tiled thread mappings must match");

  using AccPvStorage = decltype(partition_fragment_C(
      TiledMmaPV{}, make_shape(shape<0>(TileShapePDV{}), shape<1>(TileShapePDV{}), Int<PVIterations>{})));

  static_assert(StagesK > 0 && StagesV > 0, "FMHA schedule stage count must be positive");
  static_assert((QKIterations == 1 || TileHeadDimQK % size<2>(typename TiledMmaQK::AtomShape_MNK{}) == 0) &&
                    (!HasQv || PVIterations == 1 ||
                     TileHeadDimVO % size<2>(typename TiledMmaQv::AtomShape_MNK{}) == 0) &&
                    (PVIterations == 1 || TileHeadDimVO % size<1>(typename TiledMmaPV::AtomShape_MNK{}) == 0),
                "FMHA schedule chunk is not integral in selected SQMMA atoms");
  // TODO: refine ss_smem_selector
  using SmemAtomLayoutQ =
      decltype(mutlass::gemm::collective::detail::
                   ss_smem_selector_A<TCE::Major::K, Element, typename TiledMmaQK::Atom::MMA_Op, TileShapeQKD>());
  using SmemAtomLayoutK =
      decltype(mutlass::gemm::collective::detail::
                   ss_smem_selector_B<TCE::Major::K, Element, typename TiledMmaQK::Atom::MMA_Op, TileShapeQKD>());

  using SmemAtomLayoutQv =
      decltype(mutlass::gemm::collective::detail::
                   ss_smem_selector_A<TCE::Major::K, Element, typename TiledMmaQv::Atom::MMA_Op, TileShapeQvD>());
  // K-major for dot(Qv, V)
  using SmemAtomLayoutVMmaQV =
      decltype(mutlass::gemm::collective::detail::
                   ss_smem_selector_B<TCE::Major::K, Element, typename TiledMmaQv::Atom::MMA_Op, TileShapeQvD>());

  using SmemAtomLayoutP =
      decltype(mutlass::gemm::collective::detail::
                   ss_smem_selector_A<TCE::Major::K, Element, typename TiledMmaPV::Atom::MMA_Op, TileShapePDV>());

  // MN-major for dot(P, V)
  using SmemAtomLayoutVMmaPV =
      decltype(mutlass::gemm::collective::detail::
                   ss_smem_selector_B<TCE::Major::MN, Element, typename TiledMmaPV::Atom::MMA_Op, TileShapePDV>());

  using SmemLayoutQ        = decltype(tile_to_shape(SmemAtomLayoutQ{}, select<0, 2>(TileShapeQKD{})));
  using SmemLayoutQPack_   = decltype(composition(SmemLayoutQ{}, make_tile(LayoutMPack{}, Underscore{})));
  using SmemLayoutTMELoadQ = std::conditional_t<!IsPackGQA, SmemLayoutQ, SmemLayoutQPack_>;
  using SmemLayoutQFull =
      decltype(tile_to_shape(SmemAtomLayoutQ{}, make_shape(Int<TileM>{}, Int<TileHeadDimQK>{}, Int<QKIterations>{})));
  using SmemLayoutQRotary = decltype(tile_to_shape(SmemAtomLayoutQ{}, make_shape(Int<TileM>{}, Int<HeadDimQK>{})));
  using SmemLayoutTMELoadQFull =
      std::conditional_t<!IsPackGQA,
                         SmemLayoutQFull,
                         decltype(composition(SmemLayoutQFull{},
                                              make_tile(LayoutMPack{}, Underscore{}, Underscore{})))>;
  using SmemLayoutQv        = decltype(tile_to_shape(SmemAtomLayoutQv{}, select<0, 2>(TileShapeQvD{})));
  using SmemLayoutQvPack_   = decltype(composition(SmemLayoutQv{}, make_tile(LayoutMPack{}, Underscore{})));
  using SmemLayoutTMELoadQv = std::conditional_t<!IsPackGQA, SmemLayoutQv, SmemLayoutQvPack_>;
  using SmemLayoutQvFull =
      decltype(tile_to_shape(SmemAtomLayoutQv{}, make_shape(Int<TileM>{}, Int<TileHeadDimVO>{}, Int<PVIterations>{})));
  using SmemLayoutTMELoadQvFull =
      std::conditional_t<!IsPackGQA,
                         SmemLayoutQvFull,
                         decltype(composition(SmemLayoutQvFull{},
                                              make_tile(LayoutMPack{}, Underscore{}, Underscore{})))>;
  using SmemLayoutK    = decltype(tile_to_shape(
      SmemAtomLayoutK{}, make_shape(shape<1>(TileShapeQKD{}), shape<2>(TileShapeQKD{}), Int<StagesKPipeline>{})));
  using SmemLayoutKNew = decltype(tile_to_shape(
      SmemAtomLayoutK{}, make_shape(shape<1>(TileShapeQKDFull{}), shape<2>(TileShapeQKDFull{}), Int<StagesKNew>{})));
  using SmemLayoutP    = decltype(tile_to_shape(SmemAtomLayoutP{}, select<0, 2>(TileShapePDV{})));

  /* For dot(Qv, V), we need K-Major SmemLayout */
  using SmemLayoutVMmaQV = decltype(tile_to_shape(
      SmemAtomLayoutVMmaQV{}, make_shape(shape<1>(TileShapeQvD{}), shape<2>(TileShapeQvD{}), Int<StagesV>{})));

  /* For dot(P, V), we need MN-Major SmemLayout */
  using SmemLayoutVMmaPV = decltype(tile_to_shape(
      SmemAtomLayoutVMmaPV{}, make_shape(shape<1>(TileShapePDV{}), shape<2>(TileShapePDV{}), Int<StagesVt>{})));
  static_assert(HasQv || !IsHeadDimTiled || cosize_v<SmemLayoutK> == cosize_v<SmemLayoutVMmaPV>,
                "Shared K/V pipeline requires equal per-stage storage");

  // SmemLayoutV is used for TME Load
  using SmemLayoutV = std::conditional_t<HasQv, SmemLayoutVMmaQV, SmemLayoutVMmaPV>;

  // Transpose for dot(P, V)
  using SmemLayoutVStore = decltype(composition(
      SmemLayoutVMmaPV{},
      make_ordered_layout(
          make_shape(size<1>(SmemLayoutVMmaPV{}), size<0>(SmemLayoutVMmaPV{}), size<2>(SmemLayoutVMmaPV{})),
          Step<_2, _1, _3>{})));

  // Dealing with LSU load V
  using SmemAtomLayoutVLsu = std::conditional_t<
      !IsFP8,
      decltype(mute::MP31::SQMMA::Layout_SL256_SS256_SG32_Atom<get<2>(typename TiledMmaPV::AtomShape_MNK{}),
                                                               get<1>(typename TiledMmaPV::AtomShape_MNK{}),
                                                               Element,
                                                               TCE::Major::K>{}),
      decltype(mute::MP31::SQMMA::Layout_SL256_SS256_SG16_Atom<get<2>(typename TiledMmaPV::AtomShape_MNK{}),
                                                               get<1>(typename TiledMmaPV::AtomShape_MNK{}),
                                                               Element,
                                                               TCE::Major::K>{})>;

  // Match the stride but without PermuteLoad
  using SmemLayoutVLsu = decltype(tile_to_shape(
      SmemAtomLayoutVLsu{}, make_shape(shape<2>(TileShapePDV{}), shape<1>(TileShapePDV{}), Int<StagesV>{})));
  static_assert(HasQv || !IsHeadDimTiled || cosize_v<SmemLayoutK> == cosize_v<SmemLayoutVLsu>,
                "Shared K/V LSU pipeline requires equal per-stage storage");

  static constexpr TME::CacheHint TmeQInnerHint    = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeQOuterHint    = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeQvInnerHint   = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeQvOuterHint   = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeKInnerHint    = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeKOuterHint    = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeVInnerHint    = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeVOuterHint    = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeKNewInnerHint = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeKNewOuterHint = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeVNewInnerHint = TME::CacheHint::CACHE_NORMAL;
  static constexpr TME::CacheHint TmeVNewOuterHint = TME::CacheHint::CACHE_NORMAL;

  static constexpr int KLoadVectorBits = IsFP8 && TileN == 64 ? 64 : 128;
  static constexpr int MmaPvAtomM      = get<0>(typename TiledMmaPV::AtomShape_MNK{});
  using TmeLoadKeyBuilder =
      Mp31FmhaTmeLoadKeyBuilder<Element, SmemLayoutK, StrideQKV, TmeKInnerHint, TmeKOuterHint, KLoadVectorBits>;

  // Only used for dot(Qv, V)
  using TmeLoadVLayout  = std::conditional_t<HasQv, SmemLayoutVMmaQV, /* We don't need it actually */ SmemLayoutK>;
  using TmeLoadVBuilder = Mp31FmhaTmeLoadKeyBuilder<Element, TmeLoadVLayout, StrideQKV, TmeVInnerHint, TmeVOuterHint>;

  static constexpr int FragmentSize = TmeLoadKeyBuilder::Fragment;
#if defined(MATE_FMHA_USE_SCALAR_FP16_CVT)
  static constexpr int CvtFragmentSize = mute::is_same_v<Element, mutlass::half_t> ? 1 : FragmentSize;
#else
  static constexpr int CvtFragmentSize = FragmentSize;
#endif

  using FragmentTypeR2S = typename TmeLoadKeyBuilder::FragmentType;
  using PermuteTileR2S  = Tile<Underscore, typename TmeLoadKeyBuilder::PermuteTileN, Underscore>;

  using PermutedShapeK = decltype(TmeLoadKeyBuilder::get_permuted_shape(make_tensor(
      make_gmem_ptr(static_cast<Element const*>(nullptr)), repeat_like(StrideQKV{}, int32_t(0)), StrideQKV{})));
  using PermutedShapeV = decltype(TmeLoadVBuilder::get_permuted_shape(make_tensor(
      make_gmem_ptr(static_cast<Element const*>(nullptr)), repeat_like(StrideQKV{}, int32_t(0)), StrideQKV{})));

  using TmeKTileShape = typename TmeLoadKeyBuilder::TmeKTileShape;
  using TmeVTileShape = typename TmeLoadVBuilder::TmeKTileShape;

  using BarrierQ = mutlass::arch::AsyncTransactionBarrier;

  using TileShapeQ  = std::conditional_t<!IsPackGQA, Shape<Int<TileM>, Int<TileHeadDimQK>>, PackGQATileShape>;
  using TileShapeQv = std::conditional_t<!IsPackGQA, Shape<Int<TileM>, Int<TileHeadDimVO>>, PackGQvTileShape>;
  using TME_Q       = decltype(make_tme_copy<TmeQInnerHint, TmeQOuterHint>(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
                  repeat_like(StrideQPacked{}, int32_t(0)),
                  StrideQPacked{}),
      SmemLayoutTMELoadQ{},
      TileShapeQ{}));
  using TME_Qv      = decltype(make_tme_copy<TmeQvInnerHint, TmeQvOuterHint>(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
                  repeat_like(StrideQPacked{}, int32_t(0)),
                  StrideQPacked{}),
      SmemLayoutTMELoadQv{},
      TileShapeQv{}));
  using TME_K       = typename TmeLoadKeyBuilder::TME_K;
  // V TME TiledCopy for MmaPV
  using TME_V_ = decltype(make_tme_copy<TmeVInnerHint, TmeVOuterHint>(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
                  repeat_like(StrideQKV{}, int32_t(0)),
                  select<1, 0, 2, 3>(StrideQKV{})),
      take<0, 2>(SmemLayoutVMmaPV{})));
  // If HasQv, using TmeBuilder to get a permuted tme load, otherwise using trivial version
  using TME_V = std::conditional_t<HasQv, typename TmeLoadVBuilder::TME_K, TME_V_>;
  // Append V_new is stored back to the KV cache, so keep it in the store-friendly layout even in Qv mode.
  using TME_VNew = TME_V_;
  using TME_KNew = decltype(make_tme_copy<TmeKNewInnerHint, TmeKNewOuterHint>(
      MP31_TME_LOAD{},
      make_tensor(
          make_gmem_ptr(static_cast<Element const*>(nullptr)), repeat_like(StrideQKV{}, int32_t(0)), StrideQKV{}),
      take<0, 2>(SmemLayoutK{})));

  using PermuteTileForQK  = Tile<Underscore, typename TmeLoadKeyBuilder::MmaPermuteTile, Underscore>;
  using PermuteTiledMmaQK = decltype(convert_to_permuted_mma(TiledMmaQK{}, PermuteTileForQK{}));
  using R2SCopyAtom       = Copy_Atom<UniversalCopy<FragmentTypeR2S>, Element>;
  using R2STiledCopy      = decltype(make_tiled_copy_C(R2SCopyAtom{}, PermuteTiledMmaQK{}));

  // LSU Load Key (Always permuted load)
  using LsuLoadKeyBuilder =
      Mp31FmhaLsuLoadKeyBuilder<Element, TileShapeQKD, SmemAtomLayoutK, NumProducerThreads, StrideQKV, KLoadVectorBits>;
  using GmemTiledCopyK = typename LsuLoadKeyBuilder::GmemTiledCopy;
  using LsuKTile       = typename LsuLoadKeyBuilder::LsuKTile;
  using PermuteTileK   = Tile<Underscore, typename LsuLoadKeyBuilder::PermuteTileN, Underscore>;
  using FragmentTypeK  = typename LsuLoadKeyBuilder::FragmentType;

  // LSU Value for QV(Always permuted load)
  using LsuLoadValueBuilder = Mp31FmhaLsuLoadKeyBuilder<Element,
                                                        TileShapeQvD,
                                                        SmemAtomLayoutVMmaQV,
                                                        NumProducerThreads,
                                                        StrideQKV,
                                                        KLoadVectorBits>;
  using GmemTiledCopyV      = typename LsuLoadValueBuilder::GmemTiledCopy;
  using LsuVTile            = typename LsuLoadValueBuilder::LsuKTile;

  // InKernelTranspose
  using TransTiledCopy =
      decltype(mutlass::gemm::collective::detail::make_simt_tiled_copy<NumTransThreads,
                                                                       Element,
                                                                       FragmentSize,
                                                                       StrideQKV,
                                                                       size<0>(SmemAtomLayoutVMmaQV{}),
                                                                       size<1>(SmemAtomLayoutVMmaQV{}),
                                                                       UniversalCopy<FragmentTypeR2S>>());

  static constexpr int GmemElemsPerLoad = 128 / sizeof_bits_v<Element>;
  static constexpr int HeadDimGCD       = mute::gcd(HeadDimQK, HeadDimVO);
  static constexpr int KHalfRowBytes    = HeadDimQK / 2 * sizeof(Element);
  static constexpr int BlockKBytes =
      KHalfRowBytes % 128 == 0 && HeadDimGCD % (128 / sizeof(Element)) == 0
          ? 128
          : (KHalfRowBytes % 64 == 0 && HeadDimGCD % (64 / sizeof(Element)) == 0 ? 64 : 32);
  static constexpr int BlockKGmem          = BlockKBytes / sizeof(Element);
  static constexpr int GmemThreadsPerRow   = BlockKGmem / GmemElemsPerLoad;
  static constexpr int AppendKVLoadsPerRow = HeadDimGCD / BlockKGmem;
  static_assert(KHalfRowBytes % BlockKBytes == 0);
  static_assert(HeadDimGCD % BlockKGmem == 0);
  using GmemCopyAtomAppendKV = mute::Copy_Atom<MP31_ROBUST_STORE<mute::uint128_t>, Element>;
  using GmemLayoutAtomAppendKV =
      Layout<Shape<Int<NumMmaThreads / GmemThreadsPerRow>, Int<GmemThreadsPerRow>>, Stride<Int<GmemThreadsPerRow>, _1>>;
  using GmemTiledCopyAppendKV = decltype(make_tiled_copy(
      GmemCopyAtomAppendKV{}, GmemLayoutAtomAppendKV{}, Layout<Shape<_1, Int<GmemElemsPerLoad>>>{}));

  using MainloopPipelineQ = std::conditional_t<!UseTMELoadQ,
                                               mutlass::Mp31PipelineAsyncWarpSpecialized<StagesQ, BarPerStageRatio>,
                                               mutlass::Mp31PipelineTmeAsyncWarpSpecialized<StagesQ, BarPerStageRatio>>;
  using MainloopPipelineQv =
      std::conditional_t<!UseTMELoadQ,
                         mutlass::Mp31PipelineAsyncWarpSpecialized<HasQv ? StagesQv : 0, BarPerStageRatio>,
                         mutlass::Mp31PipelineTmeAsyncWarpSpecialized<HasQv ? StagesQv : 0, BarPerStageRatio>>;
  using MainloopPipelineK = std::conditional_t<
      ReuseKPStorage && !UseLSULoadK,
      mutlass::Mp31PipelineTmeAsync<StagesKPipeline>,
      std::conditional_t<UseLSULoadK,
                         mutlass::Mp31PipelineAsyncWarpSpecialized<StagesKPipeline, BarPerStageRatio>,
                         mutlass::Mp31PipelineTmeAsyncWarpSpecialized<StagesKPipeline, BarPerStageRatio>>>;
  using MainloopPipelineV =
      std::conditional_t<IsHeadDimTiled && !HasQv,
                         MainloopPipelineK,
                         std::conditional_t<UseLSULoadV,
                                            mutlass::Mp31PipelineAsyncWarpSpecialized<StagesV, BarPerStageRatio>,
                                            mutlass::Mp31PipelineTmeAsyncWarpSpecialized<StagesV, BarPerStageRatio>>>;
  using MainloopPipelineVt = std::conditional_t<InKernelTranspose,
                                                mutlass::Mp31PipelineAsyncWarpSpecialized<StagesVt, BarPerStageRatio>,
                                                MainloopPipelineV>;
  using MainloopPipelineKNew =
      mutlass::Mp31PipelineTmeAsyncWarpSpecialized<IsAppendKV ? StagesKNew : 0, BarPerStageRatio>;
  using MainloopPipelineVNew =
      mutlass::Mp31PipelineTmeAsyncWarpSpecialized<IsAppendKV ? StagesVNew : 0, BarPerStageRatio>;

  using PipelineQState    = typename MainloopPipelineQ::PipelineState;
  using PipelineQvState   = typename MainloopPipelineQv::PipelineState;
  using PipelineKState    = typename MainloopPipelineK::PipelineState;
  using PipelineVState    = typename MainloopPipelineV::PipelineState;
  using PipelineVtState   = typename MainloopPipelineVt::PipelineState;
  using PipelineKNewState = typename MainloopPipelineKNew::PipelineState;
  using PipelineVNewState = typename MainloopPipelineVNew::PipelineState;

  using KStorage = mute::array_aligned<Element, cosize_v<SmemLayoutK>>;
  using VStorage = mute::array_aligned<Element, cosize_v<SmemLayoutV>>;
  using PStorage = mute::array_aligned<Element, cosize_v<SmemLayoutP>>;

  struct SharedStorageHeadDimTiled {
    union {
      KStorage smem_k;
      VStorage smem_v;
    };
    mute::array_aligned<Element, cosize_v<SmemLayoutQFull>> smem_q;
    PStorage                                                smem_p;
  };

  struct SharedStorageHeadDimTiledWithQv {
    mute::array_aligned<Element, cosize_v<SmemLayoutQvFull>> smem_qv;
    mute::array_aligned<Element, cosize_v<SmemLayoutQFull>>  smem_q;
    KStorage                                                 smem_k;
    PStorage                                                 smem_p;
    VStorage                                                 smem_v;
  };

  struct SharedStorageHeadDimTiledReusedKPWithQv {
    mute::array_aligned<Element, cosize_v<SmemLayoutQvFull>> smem_qv;
    mute::array_aligned<Element, cosize_v<SmemLayoutQFull>>  smem_q;
    union {
      KStorage smem_k;
      PStorage smem_p;
    };
    VStorage smem_v;
  };

  struct SharedStorageIndependentKV {
    mute::array_aligned<Element, cosize_v<SmemLayoutQFull>> smem_q;
    KStorage                                                smem_k;
    PStorage                                                smem_p;
    VStorage                                                smem_v;
  };

  struct SharedStorageIndependentKVWithQv : SharedStorageIndependentKV {
    mute::array_aligned<Element, cosize_v<SmemLayoutQvFull>> smem_qv;
    mute::array_aligned<Element, cosize_v<SmemLayoutVStore>> smem_vt;
  };

  using SharedStorage = std::conditional_t<
      ReuseKPStorage,
      SharedStorageHeadDimTiledReusedKPWithQv,
      std::conditional_t<
          HasQv && IsHeadDimTiled,
          SharedStorageHeadDimTiledWithQv,
          std::conditional_t<IsHeadDimTiled,
                             SharedStorageHeadDimTiled,
                             std::conditional_t<HasQv, SharedStorageIndependentKVWithQv, SharedStorageIndependentKV>>>>;

  static constexpr bool UseQStorageForKNew =
      (IsHeadDimTiled && !HasQv) || cosize_v<SmemLayoutKNew> > cosize_v<SmemLayoutK>;

  static_assert(!IsAppendKV || !UseQStorageForKNew || cosize_v<SmemLayoutKNew> <= cosize_v<SmemLayoutQFull>,
                "KNew staging does not fit the selected storage.");
  static_assert(sizeof(SharedStorage) <= 192 * 1024, "FMHA shared storage exceeds the PH1 192 KiB limit");
  static_assert(cosize_v<SmemLayoutQRotary> == cosize_v<SmemLayoutQFull>);

  static constexpr int TmeTransactionBytesQ = mutlass::bits_to_bytes(size(SmemLayoutQFull{}) * sizeof_bits_v<Element>);
  static constexpr int TmeTransactionBytesK =
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutK{})) * sizeof_bits_v<Element>);
  static constexpr int TmeTransactionBytesKNew =
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutKNew{})) * sizeof_bits_v<Element>);
  static constexpr int TmeTransactionBytesQv =
      mutlass::bits_to_bytes(size(SmemLayoutQvFull{}) * sizeof_bits_v<Element>);
  static constexpr int TmeTransactionBytesV =
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutV{})) * sizeof_bits_v<Element>);
  static_assert(HasQv || !IsHeadDimTiled || TmeTransactionBytesK == TmeTransactionBytesV,
                "Shared K/V pipeline requires equal transaction sizes");
  static_assert(!HasQv || !IsHeadDimTiled ||
                TmeTransactionBytesV ==
                    mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutVMmaPV{})) * sizeof_bits_v<Element>));

  struct Arguments {
    Element const* const ptr_Q;
    ShapeQKV const       shape_Q;
    StrideQKV const      stride_Q;
    Element* const       ptr_K;
    ShapeQKV const       shape_K;
    StrideQKV const      stride_K;
    Element* const       ptr_V;
    int32_t const        headdim_V;
    StrideQKV const      stride_V;

    // AppendKV
    Element const* const ptr_K_new;
    ShapeQKV const       shape_K_new;
    StrideQKV const      stride_K_new;
    Element const* const ptr_V_new;
    StrideQKV const      stride_V_new;

    // Qv
    Element const* const ptr_Qv;
    ShapeQKV const       shape_Qv;
    StrideQKV const      stride_Qv;

    // Rotary
    Element const* const ptr_rotary_cos;
    ShapeRotary const    shape_rotary;
    StrideRotary const   stride_rotary_cos;
    Element const* const ptr_rotary_sin;
    StrideRotary const   stride_rotary_sin;
    // bool const is_rotary_interleaved;  // Use IsRotaryInterleaved instead

    // PageTable
    int const* const      ptr_pagetable;
    ShapePageTable const  shape_pagetable;
    StridePageTable const stride_pagetable;

    // Scale
    float const         softmax_scale;
    float const*        ptr_q_descale;
    StrideDescale const stride_q_descale;
    float const*        ptr_k_descale;
    StrideDescale const stride_k_descale;
    float const*        ptr_v_descale;
    StrideDescale const stride_v_descale;

    // Local
    int const window_size_left  = -1;
    int const window_size_right = -1;

    // Chunk
    int const attention_chunk = 0;

    // Learnable Sink
    ElementSink const* ptr_learnable_sink;

    // Softcap
    float const softcap_val;

    int const num_splits;

    // Aux tensors
    uint32_t const* const kv_batch_idx     = nullptr;
    uint32_t const* const cu_seqlens_q     = nullptr;
    uint32_t const* const cu_seqlens_k     = nullptr;
    uint32_t const* const cu_seqlens_k_new = nullptr;
    uint32_t const* const seqused_q        = nullptr;
    uint32_t const* const seqused_k        = nullptr;
    uint32_t const* const leftpad_k        = nullptr;
    uint32_t const* const seqlens_rotary   = nullptr;

    // CP
    int             cp_world_size    = 1;
    int             cp_rank          = 0;
    uint32_t const* cp_tot_seqused_k = nullptr;
  };

  struct Params {
    Element const* const ptr_Q;
    ShapeQKV const       shape_Q;
    StrideQKV const      stride_Q;
    ShapeQPacked const   shape_Q_packed;
    StrideQPacked const  stride_Q_packed;
    Element* const       ptr_K;
    ShapeQKV const       shape_K;
    PermutedShapeK const permuted_shape_K;
    StrideQKV const      stride_K;
    Element* const       ptr_V;
    ShapeQKV const       shape_V;
    PermutedShapeV const permuted_shape_V;
    int32_t const        headdim_V;
    StrideQKV const      stride_V;

    // AppendKV
    Element const* const ptr_K_new;
    ShapeQKV const       shape_K_new;
    StrideQKV const      stride_K_new;
    Element const* const ptr_V_new;
    StrideQKV const      stride_V_new;

    // Qv
    Element const* const ptr_Qv;
    StrideQKV const      stride_Qv;
    ShapeQPacked const   shape_Qv_packed;
    StrideQPacked const  stride_Qv_packed;

    // Rotary
    Element const* const ptr_rotary_cos;
    ShapeRotary const    shape_rotary;
    StrideRotary const   stride_rotary_cos;
    Element const* const ptr_rotary_sin;
    StrideRotary const   stride_rotary_sin;
    // bool const is_rotary_interleaved;  // Use IsRotaryInterleaved instead

    // PageTable
    int const* const      ptr_pagetable;
    ShapePageTable const  shape_pagetable;
    StridePageTable const stride_pagetable;

    mutlass::FastDivmod const page_size_divmod;
    // mutlass::FastDivmod const blockN_per_page_size_divmod;
    // mutlass::FastDivmod const qhead_per_khead_divmod;

    // TiledCopy
    TME_Q    tme_load_Q;
    TME_Qv   tme_load_Qv;
    TME_K    tme_load_K;
    TME_V    tme_load_V;
    TME_V_   tme_load_V_pv;
    TME_KNew tme_load_K_new;
    TME_VNew tme_load_V_new;

    // Robust Desc
    RobustDescriptor const desc_Q;
    RobustDescriptor const desc_Qv;
    RobustDescriptor const desc_K;
    RobustDescriptor const desc_V;
    RobustDescriptor const desc_page_table;
    RobustDescriptor const desc_K_new;
    RobustDescriptor const desc_V_new;
    RobustDescriptor const desc_Cos;
    RobustDescriptor const desc_Sin;

    // Scale
    float const         softmax_scale;
    float const         softmax_scale_log2;
    float const*        ptr_q_descale;
    StrideDescale const stride_q_descale;
    float const*        ptr_k_descale;
    StrideDescale const stride_k_descale;
    float const*        ptr_v_descale;
    StrideDescale const stride_v_descale;

    // Local
    int const window_size_left  = -1;
    int const window_size_right = -1;

    // Sink
    // Unused
    int const sink_token_length = 0;

    // Learnable Sink
    ElementSink const* ptr_learnable_sink = nullptr;

    float const softcap_val = 0.f;

    // Chunk
    mutlass::FastDivmod attention_chunk_divmod;

    // Aux tensors
    uint32_t const* const kv_batch_idx     = nullptr;
    uint32_t const* const cu_seqlens_q     = nullptr;
    uint32_t const* const cu_seqlens_k     = nullptr;
    uint32_t const* const cu_seqlens_k_new = nullptr;
    uint32_t const* const seqused_q        = nullptr;
    uint32_t const* const seqused_k        = nullptr;
    uint32_t const* const leftpad_k        = nullptr;
    uint32_t const* const seqlens_rotary   = nullptr;

    // CP
    int             cp_world_size    = 1;
    int             cp_rank          = 0;
    uint32_t const* cp_tot_seqused_k = nullptr;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    // If IsPackGQA, reshape Q to be ((head_ratio, seqlen_q), head_size, num_head_kv, batch_size)

    // cosize() evaluates size(shape) before applying strides, so promote shape leaves to avoid
    // int32 overflow when the logical tensor contains more than INT32_MAX elements.
    auto const cosize_64 = [](auto const& layout) -> uint64_t {
      auto const shape_64 =
          transform_leaf(layout.shape(), [](auto const& extent) { return static_cast<int64_t>(extent); });
      return static_cast<uint64_t>(cosize(make_layout(shape_64, layout.stride())));
    };

    int const  qhead_per_k_head = !IsPackGQA ? 1 : HeadRatio;
    auto const shape_Q_packed_  = make_shape(make_shape(qhead_per_k_head, get<0>(args.shape_Q)),
                                            get<1>(args.shape_Q),
                                            get<2>(args.shape_K),
                                            get<3>(args.shape_Q));
    auto const shape_Q_packed   = mute::conditional_return<!IsPackGQA>(args.shape_Q, shape_Q_packed_);

    auto const stride_Q_packed_  = make_stride(make_stride(get<2>(args.stride_Q), get<0>(args.stride_Q)),
                                              get<1>(args.stride_Q),
                                              get<2>(args.stride_Q) * qhead_per_k_head,
                                              get<3>(args.stride_Q));
    auto const stride_Q_packed   = mute::conditional_return<!IsPackGQA>(args.stride_Q, stride_Q_packed_);
    auto const shape_Qv_packed_  = make_shape(make_shape(qhead_per_k_head, get<0>(args.shape_Qv)),
                                             get<1>(args.shape_Qv),
                                             get<2>(args.shape_K),
                                             get<3>(args.shape_Qv));
    auto const shape_Qv_packed   = mute::conditional_return<!IsPackGQA>(args.shape_Qv, shape_Qv_packed_);
    auto const stride_Qv_packed_ = make_stride(make_stride(get<2>(args.stride_Qv), get<0>(args.stride_Qv)),
                                               get<1>(args.stride_Qv),
                                               get<2>(args.stride_Qv) * qhead_per_k_head,
                                               get<3>(args.stride_Qv));
    auto const stride_Qv_packed  = mute::conditional_return<!IsPackGQA>(args.stride_Qv, stride_Qv_packed_);

    Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), shape_Q_packed, stride_Q_packed);
    TME_Q  tme_load_Q =
        make_tme_copy<TmeQInnerHint, TmeQOuterHint>(MP31_TME_LOAD{}, mQ, SmemLayoutTMELoadQ{}, TileShapeQ{});
    Tensor mQv = make_tensor(make_gmem_ptr(args.ptr_Qv), shape_Qv_packed, stride_Qv_packed);
    TME_Qv tme_load_Qv =
        make_tme_copy<TmeQvInnerHint, TmeQvOuterHint>(MP31_TME_LOAD{}, mQv, SmemLayoutTMELoadQv{}, TileShapeQv{});

    uint64_t const cosize_q = get<0>(args.shape_Q) == 0 ? 0 : cosize_64(make_layout(shape_Q_packed, stride_Q_packed));
    auto           desc_Q   = make_robust_desc(args.ptr_Q, cosize_q);
    uint64_t const cosize_qv =
        get<0>(args.shape_Qv) == 0 ? 0 : cosize_64(make_layout(shape_Qv_packed, stride_Qv_packed));
    auto desc_Qv = make_robust_desc(args.ptr_Qv, cosize_qv);

    // print("TME_Q:");
    // print(tme_load_Q);
    // print("\n");

    Tensor         mK               = make_tensor(make_gmem_ptr(args.ptr_K), args.shape_K, args.stride_K);
    TME_K          tme_load_K       = TmeLoadKeyBuilder::make_tme_copy(mK);
    PermutedShapeK permuted_shape_K = TmeLoadKeyBuilder::get_permuted_shape(mK);
    uint64_t const cosize_k = get<0>(args.shape_K) == 0 ? 0 : cosize_64(make_layout(args.shape_K, args.stride_K));

    RobustDescriptor desc_K = make_robust_desc(args.ptr_K, cosize_k);

    // print("TME_K:");
    // print(tme_load_K);
    // print("\n");

    auto shape_V_gmem = make_shape(get<0>(args.shape_K), args.headdim_V, get<2>(args.shape_K), get<3>(args.shape_K));
    auto shape_V      = conditional_return<HasQv>(shape_V_gmem, select<1, 0, 2, 3>(shape_V_gmem));
    auto stride_V     = conditional_return<HasQv>(args.stride_V, select<1, 0, 2, 3>(args.stride_V));
    auto mV           = make_tensor(make_gmem_ptr(args.ptr_V), shape_V, stride_V);

    TME_V tme_load_V = [&]() {
      if constexpr (HasQv) {
        return TmeLoadVBuilder::make_tme_copy(mV);
      } else {
        return make_tme_copy<TmeVInnerHint, TmeVOuterHint>(MP31_TME_LOAD{}, mV, take<0, 2>(SmemLayoutVMmaPV{}));
      }
    }();

    // Used for permute tme load
    PermutedShapeV permuted_shape_V = TmeLoadVBuilder::get_permuted_shape(mV);
    uint64_t const cosize_v = get<0>(shape_V_gmem) == 0 ? 0 : cosize_64(make_layout(shape_V_gmem, args.stride_V));

    RobustDescriptor desc_V = make_robust_desc(args.ptr_V, cosize_v);

    // print("TME_V:");
    // print(tme_load_V);
    // print("\n");

    // AppendKV
    Tensor   mKnew          = make_tensor(make_gmem_ptr(args.ptr_K_new), args.shape_K_new, args.stride_K_new);
    TME_KNew tme_load_K_new = make_tme_copy<TmeKNewInnerHint, TmeKNewOuterHint>(
        MP31_TME_LOAD{}, conditional_return<IsAppendKV>(mKnew, mK), take<0, 2>(SmemLayoutK{}));
    uint64_t const cosize_k_new =
        get<0>(args.shape_K_new) == 0 ? 0 : cosize_64(make_layout(args.shape_K_new, args.stride_K_new));

    RobustDescriptor desc_K_new = make_robust_desc(args.ptr_K_new, cosize_k_new);

    Tensor mV_store =
        make_tensor(make_gmem_ptr(args.ptr_V), select<1, 0, 2, 3>(shape_V_gmem), select<1, 0, 2, 3>(args.stride_V));
    TME_V_ tme_load_V_pv =
        make_tme_copy<TmeVInnerHint, TmeVOuterHint>(MP31_TME_LOAD{}, mV_store, take<0, 2>(SmemLayoutVMmaPV{}));
    Tensor mVnew = make_tensor(
        make_gmem_ptr(args.ptr_V_new),
        make_shape(args.headdim_V, get<0>(args.shape_K_new), get<2>(args.shape_K_new), get<3>(args.shape_K_new)),
        select<1, 0, 2, 3>(args.stride_V_new));
    TME_VNew tme_load_V_new = make_tme_copy<TmeVInnerHint, TmeVOuterHint>(
        MP31_TME_LOAD{}, conditional_return<IsAppendKV>(mVnew, mV_store), take<0, 2>(SmemLayoutVMmaPV{}));
    uint64_t const cosize_v_new = get<0>(args.shape_K_new) == 0 ? 0 : cosize_64(mVnew.layout());

    RobustDescriptor desc_V_new = make_robust_desc(args.ptr_V_new, cosize_v_new);

    RobustDescriptor desc_Cos = make_robust_desc(
        args.ptr_rotary_cos,
        get<0>(args.shape_rotary) == 0 ? 0 : cosize_64(make_layout(args.shape_rotary, args.stride_rotary_cos)));
    RobustDescriptor desc_Sin = make_robust_desc(
        args.ptr_rotary_sin,
        get<0>(args.shape_rotary) == 0 ? 0 : cosize_64(make_layout(args.shape_rotary, args.stride_rotary_sin)));

    // Qv

    float const log2e = std::log2(std::exp(1.0f));
    // float const effective_softmax_scale = HasSoftcap ? args.softcap_val : args.softmax_scale;

    int const           page_size = IsPagedKV ? get<0>(args.shape_K) : 1;
    mutlass::FastDivmod attention_chunk_divmod(args.attention_chunk >= 1 ? args.attention_chunk : 1);
    attention_chunk_divmod.divisor = args.attention_chunk;

    RobustDescriptor desc_page_table =
        make_robust_desc(args.ptr_pagetable, cosize_64(make_layout(args.shape_pagetable)));

    // SHOW(shape_Q_packed);
    // SHOW(stride_Q_packed);

    // SHOW(shape_Q_packed);
    // SHOW(stride_Q_packed);

    // SHOW(TiledMmaQK{});
    // SHOW(TiledMmaPV{});

    // SHOW(SmemLayoutQ{});
    // SHOW(SmemLayoutTMELoadQ{});
    // SHOW(SmemLayoutK{});
    // SHOW(SmemLayoutP{});
    // SHOW(SmemLayoutV{});
    // SHOW(R2STiledCopy{});

    // SHOW(UseLSULoadQ);
    // SHOW(UseLSULoadK);
    // SHOW(UseLSULoadV);

    return Params{
        .ptr_Q            = args.ptr_Q,
        .shape_Q          = args.shape_Q,
        .stride_Q         = args.stride_Q,
        .shape_Q_packed   = shape_Q_packed,
        .stride_Q_packed  = stride_Q_packed,
        .ptr_K            = args.ptr_K,
        .shape_K          = args.shape_K,
        .permuted_shape_K = permuted_shape_K,
        .stride_K         = args.stride_K,
        .ptr_V            = args.ptr_V,
        .shape_V          = shape_V,
        .permuted_shape_V = permuted_shape_V,
        .headdim_V        = args.headdim_V,
        .stride_V         = args.stride_V,

        // AppendKV
        .ptr_K_new    = args.ptr_K_new,
        .shape_K_new  = args.shape_K_new,
        .stride_K_new = args.stride_K_new,
        .ptr_V_new    = args.ptr_V_new,
        .stride_V_new = args.stride_V_new,

        // Qv
        .ptr_Qv           = args.ptr_Qv,
        .stride_Qv        = args.stride_Qv,
        .shape_Qv_packed  = shape_Qv_packed,
        .stride_Qv_packed = stride_Qv_packed,

        // Rotary
        .ptr_rotary_cos    = args.ptr_rotary_cos,
        .shape_rotary      = args.shape_rotary,
        .stride_rotary_cos = args.stride_rotary_cos,
        .ptr_rotary_sin    = args.ptr_rotary_sin,
        .stride_rotary_sin = args.stride_rotary_sin,
        // .is_rotary_interleaved;  // Use IsRotaryInterleaved instead

        // PageTable
        .ptr_pagetable    = args.ptr_pagetable,
        .shape_pagetable  = args.shape_pagetable,
        .stride_pagetable = args.stride_pagetable,
        .page_size_divmod = mutlass::FastDivmod(page_size),

        // TiledCopy
        .tme_load_Q     = tme_load_Q,
        .tme_load_Qv    = tme_load_Qv,
        .tme_load_K     = tme_load_K,
        .tme_load_V     = tme_load_V,
        .tme_load_V_pv  = tme_load_V_pv,
        .tme_load_K_new = tme_load_K_new,
        .tme_load_V_new = tme_load_V_new,

        .desc_Q          = desc_Q,
        .desc_Qv         = desc_Qv,
        .desc_K          = desc_K,
        .desc_V          = desc_V,
        .desc_page_table = desc_page_table,
        .desc_K_new      = desc_K_new,
        .desc_V_new      = desc_V_new,
        .desc_Cos        = desc_Cos,
        .desc_Sin        = desc_Sin,

        .softmax_scale      = args.softmax_scale,
        .softmax_scale_log2 = args.softmax_scale * log2e,
        .ptr_q_descale      = args.ptr_q_descale,
        .stride_q_descale   = args.stride_q_descale,
        .ptr_k_descale      = args.ptr_k_descale,
        .stride_k_descale   = args.stride_k_descale,
        .ptr_v_descale      = args.ptr_v_descale,
        .stride_v_descale   = args.stride_v_descale,

        .window_size_left  = args.window_size_left,
        .window_size_right = args.window_size_right,

        .ptr_learnable_sink = args.ptr_learnable_sink,

        .softcap_val = args.softcap_val,

        // Chunk
        .attention_chunk_divmod = attention_chunk_divmod,

        // Aux tensors
        .kv_batch_idx     = args.kv_batch_idx,
        .cu_seqlens_q     = args.cu_seqlens_q,
        .cu_seqlens_k     = args.cu_seqlens_k,
        .cu_seqlens_k_new = args.cu_seqlens_k_new,
        .seqused_q        = args.seqused_q,
        .seqused_k        = args.seqused_k,
        .leftpad_k        = args.leftpad_k,
        .seqlens_rotary   = args.seqlens_rotary,

        // CP
        .cp_world_size    = args.cp_world_size,
        .cp_rank          = args.cp_rank,
        .cp_tot_seqused_k = args.cp_tot_seqused_k,
    };
  }

  template <class BarrierStorage, class BlockCoord, class PipelineQv>
  MUTE_DEVICE void load(Params const&      params,
                        MainloopPipelineQ& pipeline_q,
                        PipelineQv&        pipeline_qv,
                        MainloopPipelineK& pipeline_k,
                        MainloopPipelineV& pipeline_v,
                        PipelineQState&    smem_pipe_write_q,
                        PipelineQvState&   smem_pipe_write_qv,
                        PipelineKState&    smem_pipe_write_k,
                        PipelineVState&    smem_pipe_write_v,
                        SharedStorage&     shared_storage,
                        BarrierStorage*    barrier_storage,
                        SeqlenInfo const&  seqlen_info,
                        BlockCoord         blk_coord,
                        int&               work_idx,
                        int const          num_splits) {
    int const m_block   = get<0>(blk_coord);
    int const bidh      = get<1>(blk_coord);
    int const bidb      = get<2>(blk_coord);
    int const split_idx = get<3>(blk_coord);

    auto [n_block_min, n_block_max] = BlockInfo::get_n_block_min_max(seqlen_info,
                                                                     m_block,
                                                                     split_idx,
                                                                     /*splits*/ num_splits,
                                                                     params.window_size_left,
                                                                     params.window_size_right,
                                                                     params.attention_chunk_divmod);

    // If no valid block, no need to load.
    if (n_block_min >= n_block_max) {
      return;
    }

    Tensor sQ    = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutTMELoadQFull{});
    Tensor sK    = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sV    = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});
    Tensor sV_pv = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVMmaPV{});

    // Used for LSU
    Tensor sV_lsu = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVLsu{});

    int const thread_idx             = threadIdx.x % NumProducerThreads;
    int const bidh_kv                = !IsPackGQA ? bidh / HeadRatio : bidh;
    int const bidb_kv                = !HasKvBatchIdx ? bidb : params.kv_batch_idx[bidb];
    int const warp_idx               = mutlass::canonical_warp_idx();
    int const warp_idx_in_warp_squad = warp_idx % mutlass::NumWarpsPerWarpSquad;
    int const seqlen_k               = seqlen_info.seqlen_k;

    auto offset_coord_q = [&]() {
      if constexpr (!IsPackGQA) {
        return make_coord(seqlen_info.offset_q, _0{});
      } else {
        return make_coord(make_coord(_0{}, seqlen_info.offset_q), _0{});
      }
    }();
    Tensor mQ = params.tme_load_Q.get_tme_tensor(params.shape_Q_packed)(_, _, bidh, HasCuseqlensQ ? 0 : bidb);
    Tensor mK = params.tme_load_K.get_tme_tensor(params.permuted_shape_K)(_, _, bidh_kv, _);
    auto   mV = [&]() {
      if constexpr (HasQv) {
        return params.tme_load_V.get_tme_tensor(params.permuted_shape_V)(_, _, bidh_kv, _);
      } else {
        return params.tme_load_V.get_tme_tensor(params.shape_V)(_, _, bidh_kv, _);
      }
    }();
    auto mV_pv = params.tme_load_V_pv.get_tme_tensor(
        conditional_return<HasQv>(select<1, 0, 2, 3>(params.shape_V), params.shape_V))(_, _, bidh_kv, _);

    Tensor gQ = local_tile(domain_offset(offset_coord_q, mQ), TileShapeQ{}, make_coord(m_block, _));
    Tensor gK = local_tile(
        domain_offset(make_coord(seqlen_info.offset_k, _0{}, _0{}), mK), TmeKTileShape{}, make_coord(_, _, _));

    auto gV = [&]() {
      if constexpr (HasQv) {
        return local_tile(
            domain_offset(make_coord(seqlen_info.offset_k, _0{}, _0{}), mV), TmeVTileShape{}, make_coord(_, _, _));
      } else {
        return local_tile(domain_offset(make_coord(_0{}, seqlen_info.offset_k, _0{}), mV),
                          select<1, 2>(TileShapePDV{}),
                          make_coord(_, _, _));
      }
    }();
    Tensor gV_pv = local_tile(domain_offset(make_coord(_0{}, seqlen_info.offset_k, _0{}), mV_pv),
                              select<1, 2>(TileShapePDV{}),
                              make_coord(_, _, _));

    auto   cta_tme_Q = params.tme_load_Q.get_slice(0);
    Tensor tQgQ      = group_modes<0, 3>(cta_tme_Q.partition_S(gQ));
    Tensor tQsQ      = group_modes<0, 3>(cta_tme_Q.partition_D(sQ));

    auto   cta_tme_K = params.tme_load_K.get_slice(0);
    Tensor tKgK      = group_modes<0, 3>(cta_tme_K.partition_S(gK));  // (TME, n, l)
    Tensor tKsK      = group_modes<0, 3>(cta_tme_K.partition_D(sK));  // (TME, pipe)

    auto   cta_tme_V    = params.tme_load_V.get_slice(0);
    Tensor tVgV         = group_modes<0, 3>(cta_tme_V.partition_S(gV));  // (TME, n, b)
    Tensor tVsV         = group_modes<0, 3>(cta_tme_V.partition_D(sV));  // (TME, pipe)
    auto   cta_tme_V_pv = params.tme_load_V_pv.get_slice(0);
    Tensor tVgV_pv      = group_modes<0, 3>(cta_tme_V_pv.partition_S(gV_pv));
    Tensor tVsV_pv      = group_modes<0, 3>(cta_tme_V_pv.partition_D(sV_pv));

    int const bidb_kv_idx = !HasCuseqlensK && !IsPagedKV ? bidb_kv : 0;

    using KVManager = PagedKVManager<IsPagedKV,
                                     Element,
                                     NumProducerThreads,
                                     TileN,
                                     TileHeadDimQK,
                                     TileHeadDimVO,
                                     !IntraWarpSquadOverlap,
                                     1,
                                     KLoadVectorBits>;
    GmemTiledCopyK tiled_copy_k;
    auto           thr_copy_k = tiled_copy_k.get_thread_slice(threadIdx.x);

    Tensor mK_lsu = make_tensor(make_gmem_ptr(params.ptr_K), params.shape_K, params.stride_K)(
        _, _, bidh_kv, HasCuseqlensK ? 0 : bidb_kv);
    auto gK_lsu = [&]() {
      if constexpr (!IsHeadDimTiled) {
        return local_tile(
            domain_offset(make_coord(seqlen_info.offset_k, _0{}), mK_lsu), LsuKTile{}, make_coord(_, _0{}));
      } else {
        return local_tile(domain_offset(make_coord(seqlen_info.offset_k, _0{}), mK_lsu), LsuKTile{}, make_coord(_, _));
      }
    }();

    GmemTiledCopyV                    tiled_copy_v;
    auto                              thr_copy_v = tiled_copy_v.get_thread_slice(threadIdx.x);
    typename KVManager::GmemTiledCopy tiled_copy_v_pv;
    auto                              thr_copy_v_pv = tiled_copy_v_pv.get_thread_slice(threadIdx.x);
    Tensor mV_lsu = make_tensor(make_gmem_ptr(params.ptr_V), params.shape_V, params.stride_V)(
        _, _, bidh_kv, HasCuseqlensK ? 0 : bidb_kv);
    Tensor gV_lsu =
        local_tile(domain_offset(make_coord(seqlen_info.offset_k, _0{}), mV_lsu), LsuVTile{}, make_coord(_, _));
    Tensor gV_pv_lsu = local_tile(
        domain_offset(make_coord(seqlen_info.offset_k, _0{}), mV_lsu), select<2, 1>(TileShapePDV{}), make_coord(_, _));

    KVManager paged_kv_manager{params.ptr_pagetable,
                               params.shape_pagetable,
                               params.stride_pagetable,
                               params.desc_page_table,
                               params.ptr_K,
                               params.shape_K,
                               params.stride_K,
                               params.desc_K,
                               params.ptr_V,
                               params.headdim_V,
                               params.stride_V,
                               params.desc_V,
                               params.page_size_divmod,
                               seqlen_k,
                               static_cast<int>(seqlen_info.leftpad_k),
                               thread_idx,
                               bidb_kv,
                               bidh_kv,
                               bidb_kv_idx};

    auto load_K = [&](int const n_block, auto& smem_pipe_write) {
      MUTLASS_PRAGMA_UNROLL
      for (int qk_iter = 0; qk_iter < QKIterations; ++qk_iter) {
        pipeline_k.producer_acquire(smem_pipe_write);
        if constexpr (IsPagedKV) {
          if constexpr (UseLSULoadK) {
            paged_kv_manager.load_K(n_block, sK(_, _, smem_pipe_write.index()), qk_iter * TileHeadDimQK);
            mute::ldgsts_wait();
            pipeline_k.producer_commit(smem_pipe_write);
          } else {
            uint32_t bar_id                         = pipeline_k.producer_get_barrier_id(smem_pipe_write);
            auto [tme_n_block_idx, tme_bidb_kv_idx] = paged_kv_manager.get_indices_for_tme_k();
            copy(params.tme_load_K.with(bar_id),
                 tKgK(_, tme_n_block_idx, qk_iter, tme_bidb_kv_idx),
                 tKsK(_, smem_pipe_write.index()));
          }
        } else {
          // NOT Paged KV
          // Got BSHD or Ragged KV
          if constexpr (UseLSULoadK) {
            auto tKgK_lsu = [&]() {
              if constexpr (!IsHeadDimTiled) {
                return group_modes<0, 3>(thr_copy_k.partition_S(gK_lsu(_, _, n_block)));
              } else {
                return group_modes<0, 3>(thr_copy_k.partition_S(gK_lsu(_, _, n_block, qk_iter)));
              }
            }();
            Tensor tKsK_lsu = group_modes<0, 3>(thr_copy_k.partition_D(sK));
            copy(tiled_copy_k.with(params.desc_K), tKgK_lsu, tKsK_lsu(_, smem_pipe_write.index()));

            mute::ldgsts_wait();
            pipeline_k.producer_commit(smem_pipe_write);
          } else {
            uint32_t bar_id = pipeline_k.producer_get_barrier_id(smem_pipe_write);
            copy(params.tme_load_K.with(bar_id),
                 tKgK(_, n_block, qk_iter, bidb_kv_idx),
                 tKsK(_, smem_pipe_write.index()));
          }
        }
        ++smem_pipe_write;
      }
    };

    mute::tuple<int, int> next_page_indices{0, 0};
    auto load_V = [&](int const n_block, auto& smem_pipe_write, auto load_for_pv, bool prefetch_next_page = false) {
      static constexpr bool LoadForPV     = decltype(load_for_pv)::value;
      auto [v_n_block_idx, v_bidb_kv_idx] = [&]() {
        if constexpr (IsPagedKV && !UseLSULoadV) {
          return paged_kv_manager.get_indices_for_tme_v();
        } else {
          return mute::make_tuple(0, 0);
        }
      }();
      if constexpr (IsPagedKV && UseLSULoadV && !IntraWarpSquadOverlap) {
        paged_kv_manager.compute_V_ptr();
      }
      MUTLASS_PRAGMA_UNROLL
      for (int pv_iter = 0; pv_iter < PVIterations; ++pv_iter) {
        pipeline_v.producer_acquire(smem_pipe_write);
        if constexpr (IsPagedKV) {
          if constexpr (UseLSULoadV) {
            if constexpr (HasQv && !LoadForPV) {
              paged_kv_manager.load_V(n_block, sV(_, _, smem_pipe_write.index()), pv_iter * TileHeadDimVO);
            } else {
              paged_kv_manager.load_V(n_block, sV_lsu(_, _, smem_pipe_write.index()), pv_iter * TileHeadDimVO);
            }
            mute::ldgsts_wait();
            pipeline_v.producer_commit(smem_pipe_write);
          } else {
            uint32_t bar_id = pipeline_v.producer_get_barrier_id(smem_pipe_write);
            if constexpr (LoadForPV) {
              copy(params.tme_load_V_pv.with(bar_id),
                   tVgV_pv(_, pv_iter, v_n_block_idx, v_bidb_kv_idx),
                   tVsV_pv(_, smem_pipe_write.index()));
              if constexpr (ReuseKPStorage) {
                if (prefetch_next_page) {
                  prefetch(params.tme_load_V, tVgV(_, get<0>(next_page_indices), pv_iter, get<1>(next_page_indices)));
                }
              }
            } else if constexpr (HasQv) {
              copy(params.tme_load_V.with(bar_id),
                   tVgV(_, v_n_block_idx, pv_iter, v_bidb_kv_idx),
                   tVsV(_, smem_pipe_write.index()));
            } else {
              copy(params.tme_load_V.with(bar_id),
                   tVgV(_, pv_iter, v_n_block_idx, v_bidb_kv_idx),
                   tVsV(_, smem_pipe_write.index()));
            }
          }
        } else {
          // NOT Paged KV
          // Got BSHD or Ragged KV

          if constexpr (UseLSULoadV) {
            if constexpr (HasQv && !LoadForPV) {
              Tensor tVgV_lsu = group_modes<0, 3>(thr_copy_v.partition_S(gV_lsu(_, _, n_block, pv_iter)));
              Tensor tVsV_lsu = group_modes<0, 3>(thr_copy_v.partition_D(sV));
              copy(tiled_copy_v.with(params.desc_V), tVgV_lsu, tVsV_lsu(_, smem_pipe_write.index()));
            } else {
              Tensor tVgV_lsu = group_modes<0, 3>(thr_copy_v_pv.partition_S(gV_pv_lsu(_, _, n_block, pv_iter)));
              Tensor tVsV_lsu = group_modes<0, 3>(thr_copy_v_pv.partition_D(sV_lsu));
              copy(tiled_copy_v_pv.with(params.desc_V), tVgV_lsu, tVsV_lsu(_, smem_pipe_write.index()));
            }
            mute::ldgsts_wait();
            pipeline_v.producer_commit(smem_pipe_write);
          } else {
            uint32_t bar_id = pipeline_v.producer_get_barrier_id(smem_pipe_write);
            if constexpr (LoadForPV) {
              copy(params.tme_load_V_pv.with(bar_id),
                   tVgV_pv(_, pv_iter, n_block, bidb_kv_idx),
                   tVsV_pv(_, smem_pipe_write.index()));
            } else if constexpr (HasQv) {
              copy(params.tme_load_V.with(bar_id),
                   tVgV(_, n_block, pv_iter, bidb_kv_idx),
                   tVsV(_, smem_pipe_write.index()));
            } else {
              copy(params.tme_load_V.with(bar_id),
                   tVgV(_, pv_iter, n_block, bidb_kv_idx),
                   tVsV(_, smem_pipe_write.index()));
            }
          }
        }
        ++smem_pipe_write;
      }
      if constexpr (IsPagedKV && UseLSULoadV && IntraWarpSquadOverlap) {
        paged_kv_manager.compute_V_ptr();
      }
    };

    bool should_load_K = UseLSULoadK || SingleProducerWarp || warp_idx_in_warp_squad == 0;
    bool should_load_V = UseLSULoadV || SingleProducerWarp || warp_idx_in_warp_squad == 0;

    int n_block = n_block_max - 1;

    if constexpr (UseTMELoadQ) {
      // (Non-)PackGQA TME load Q
      if (SingleProducerWarp || warp_idx_in_warp_squad == 0) {
        pipeline_q.producer_acquire(smem_pipe_write_q);
        uint32_t bar_id = pipeline_q.producer_get_barrier_id(smem_pipe_write_q);
        MUTLASS_PRAGMA_UNROLL
        for (int qk_iter = 0; qk_iter < QKIterations; ++qk_iter) {
          copy(params.tme_load_Q.with(bar_id), tQgQ(_, qk_iter), tQsQ(_, qk_iter));
        }
        ++smem_pipe_write_q;

        if constexpr (HasQv) {
          Tensor sQv = make_tensor(make_smem_ptr(shared_storage.smem_qv.data()), SmemLayoutTMELoadQvFull{});
          Tensor mQv = params.tme_load_Qv.get_tme_tensor(params.shape_Qv_packed)(_, _, bidh, HasCuseqlensQ ? 0 : bidb);
          Tensor gQv = local_tile(domain_offset(offset_coord_q, mQv), TileShapeQv{}, make_coord(m_block, _));
          auto   cta_tme_Qv = params.tme_load_Qv.get_slice(0);
          Tensor tQvgQv     = group_modes<0, 3>(cta_tme_Qv.partition_S(gQv));
          Tensor tQvsQv     = group_modes<0, 3>(cta_tme_Qv.partition_D(sQv));

          pipeline_qv.producer_acquire(smem_pipe_write_qv);
          uint32_t qv_bar_id = pipeline_qv.producer_get_barrier_id(smem_pipe_write_qv);
          MUTLASS_PRAGMA_UNROLL
          for (int pv_iter = 0; pv_iter < PVIterations; ++pv_iter) {
            copy(params.tme_load_Qv.with(qv_bar_id), tQvgQv(_, pv_iter), tQvsQv(_, pv_iter));
          }
          ++smem_pipe_write_qv;
        }
      }
    } else {
      // PackGQA LSU Load Q
      pipeline_q.producer_acquire(smem_pipe_write_q);
      Tensor mQPack =
          make_tensor(params.ptr_Q + seqlen_info.offset_q * get<0>(params.stride_Q),
                      make_layout(params.shape_Q_packed, params.stride_Q_packed))(_, _, bidh, HasCuseqlensQ ? 0 : bidb);
      Tensor sQPack = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQFull{});
      MUTLASS_PRAGMA_UNROLL
      for (int qk_iter = 0; qk_iter < QKIterations; ++qk_iter) {
        Tensor mQPackCur = domain_offset(make_coord(_0{}, qk_iter * TileHeadDimQK), mQPack);
        Tensor sQPackCur = sQPack(_, _, qk_iter);
        PackGQAManager::load_Q(params, mQPackCur, sQPackCur, thread_idx, m_block);
      }

      if constexpr (HasQv) {
        pipeline_qv.producer_acquire(smem_pipe_write_qv);
        Tensor mQvPack = make_tensor(params.ptr_Qv + seqlen_info.offset_q * get<0>(params.stride_Qv),
                                     make_layout(params.shape_Qv_packed, params.stride_Qv_packed))(
            _, _, bidh, HasCuseqlensQ ? 0 : bidb);
        Tensor sQvPack = make_tensor(make_smem_ptr(shared_storage.smem_qv.data()), SmemLayoutQvFull{});
        MUTLASS_PRAGMA_UNROLL
        for (int pv_iter = 0; pv_iter < PVIterations; ++pv_iter) {
          Tensor mQvPackCur = domain_offset(make_coord(_0{}, pv_iter * TileHeadDimVO), mQvPack);
          Tensor sQvPackCur = sQvPack(_, _, pv_iter);
          PackGQvManager::template load_Q</*IsQv*/ true>(params, mQvPackCur, sQvPackCur, thread_idx, m_block);
        }
      }

      // Publish Q before filling a downstream pipeline: a multi-chunk K load can block on consumer release.
      mute::ldgsts_wait();
      pipeline_q.producer_commit(smem_pipe_write_q);
      ++smem_pipe_write_q;
      if constexpr (HasQv) {
        pipeline_qv.producer_commit(smem_pipe_write_qv);
        ++smem_pipe_write_qv;
      }
    }

    // Initialize the first page-table state; non-reload paths also load the first K block.
    if (should_load_K) {
      if constexpr (IsPagedKV) {
        // NOTE: force use same load method (LSU or TME) for both K and V if IsPagedKV
        if constexpr (UseLSULoadK || UseLSULoadV) {
          paged_kv_manager.template load_page_table_for_lsu</* FirstIter */ true,
                                                            /* PermuteK */ true,
                                                            /* PermuteV */ HasQv>(n_block);
        } else {
          paged_kv_manager.template load_page_table_for_tme</* FirstIter */ true>(n_block);
        }
      }
      if constexpr (!HasQv || !IsHeadDimTiled) {
        load_K(n_block, smem_pipe_write_k);
      }
    }

    if constexpr (HasQv && IsHeadDimTiled) {
      for (; n_block >= n_block_min; --n_block) {
        if (should_load_V) {
          load_V(n_block, smem_pipe_write_v, mute::false_type{});
        }
        if (should_load_K) {
          load_K(n_block, smem_pipe_write_k);
        }

        bool const has_next = n_block - 1 >= n_block_min;
        if constexpr (IsPagedKV && !UseLSULoadK) {
          if (should_load_K) {
            if (has_next) {
              next_page_indices = paged_kv_manager.load_page_table_indices_for_tme(n_block - 1);
              prefetch(params.tme_load_K, tKgK(_, get<0>(next_page_indices), _0{}, get<1>(next_page_indices)));
            }
          }
        }
        if (should_load_V) {
          if constexpr (IsPagedKV && UseLSULoadV) {
            paged_kv_manager.template load_page_table_for_lsu</* FirstIter */ false,
                                                              /* PermuteK */ true,
                                                              /* PermuteV */ false>(n_block);
          }
          load_V(n_block, smem_pipe_write_v, mute::true_type{}, has_next);
        }

        if constexpr (IsPagedKV) {
          if (has_next && should_load_K) {
            if constexpr (UseLSULoadK) {
              paged_kv_manager.template load_page_table_for_lsu</* FirstIter */ false,
                                                                /* PermuteK */ true,
                                                                /* PermuteV */ true>(n_block - 1);
            } else {
              paged_kv_manager.set_indices_for_tme(next_page_indices);
            }
          }
        }
      }
      return;
    }

    if constexpr (!IntraWarpSquadOverlap) {
      if (should_load_V) {
        load_V(n_block, smem_pipe_write_v, mute::false_type{});
      }
    }

    int n_block_prev = n_block;
    --n_block;
    for (; n_block >= n_block_min; --n_block) {
      if constexpr (IsPagedKV) {
        // NOTE: using same load method (LSU or TME) for both K and V if IsPagedKV
        if (should_load_K) {
          if constexpr (UseLSULoadK || UseLSULoadV) {
            paged_kv_manager.template load_page_table_for_lsu</* FirstIter */ false,
                                                              /* PermuteK */ true,
                                                              /* PermuteV */ HasQv>(n_block);
          } else {
            paged_kv_manager.template load_page_table_for_tme</* FirstIter */ false>(n_block);
          }
        }
      }
      if (should_load_K) {
        load_K(n_block, smem_pipe_write_k);
      }

      if (should_load_V) {
        if constexpr (IntraWarpSquadOverlap) {
          load_V(n_block_prev, smem_pipe_write_v, mute::false_type{});
        } else {
          load_V(n_block, smem_pipe_write_v, mute::false_type{});
        }
      }
      n_block_prev = n_block;
    }

    if constexpr (IntraWarpSquadOverlap) {
      if (should_load_V) {
        load_V(n_block_prev, smem_pipe_write_v, mute::false_type{});
      }
    }
  }

  template <class BlockCoord>
  MUTLASS_DEVICE void transpose(BlockCoord const&   blk_coord,
                                Params const&       params,
                                MainloopPipelineV&  pipeline_v,
                                PipelineVState&     smem_pipe_v_read,
                                MainloopPipelineVt& pipeline_vt,
                                PipelineVtState&    smem_pipe_vt_write,
                                SharedStorage&      shared_storage,
                                SeqlenInfo const&   seqlen_info,
                                int const           thread_idx,
                                int const           num_splits) {
    static_assert(InKernelTranspose);

    TransTiledCopy tiled_copy_trans;
    ThrCopy        thr_copy_trans = tiled_copy_trans.get_thread_slice(thread_idx);

    Tensor smem_src = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), typename TmeLoadVBuilder::SmemLayoutK{});
    Tensor smem_dst = make_tensor(make_smem_ptr(shared_storage.smem_vt.data()), SmemLayoutVStore{});

    Tensor tVsV  = thr_copy_trans.partition_S(smem_src);
    Tensor tVsVt = thr_copy_trans.partition_D(smem_dst);
    Tensor tVrV  = make_fragment_like(tVsV(_, _, _, 0));

    int const m_block   = get<0>(blk_coord);
    int const split_idx = get<3>(blk_coord);

    auto [n_block_min, n_block_max] = BlockInfo::get_n_block_min_max(seqlen_info,
                                                                     m_block,
                                                                     split_idx,
                                                                     /*splits*/ num_splits,
                                                                     params.window_size_left,
                                                                     params.window_size_right,
                                                                     params.attention_chunk_divmod);

    for (int n_block = n_block_min; n_block < n_block_max; ++n_block) {
      pipeline_vt.producer_acquire(smem_pipe_vt_write);
      pipeline_v.consumer_wait(smem_pipe_v_read);

      copy(tiled_copy_trans, tVsV(_, _, _, smem_pipe_v_read.index()), tVrV);
      __syncwarp();

      pipeline_v.consumer_release(smem_pipe_v_read);
      copy(tiled_copy_trans, tVrV, tVsVt(_, _, _, smem_pipe_vt_write.index()));
      __syncwarp();

      pipeline_vt.producer_commit(smem_pipe_vt_write);
      ++smem_pipe_v_read;
      ++smem_pipe_vt_write;
    }
  }

  template <class BarrierStorage, class BlockCoord, class PipelineQv>
  MUTE_DEVICE auto mma(Params const&       params,
                       MainloopPipelineQ&  pipeline_q,
                       PipelineQv&         pipeline_qv,
                       MainloopPipelineK&  pipeline_k,
                       MainloopPipelineV&  pipeline_v,
                       MainloopPipelineVt& pipeline_vt,
                       PipelineQState&     smem_pipe_read_q,
                       PipelineQvState&    smem_pipe_read_qv,
                       PipelineKState&     smem_pipe_read_k,
                       PipelineVState&     smem_pipe_read_v,
                       PipelineVtState&    smem_pipe_read_vt,
                       AccPvStorage&       acc_pv_storage,
                       SharedStorage&      shared_storage,
                       BarrierStorage*     barrier_storage,
                       SeqlenInfo const&   seqlen_info,
                       BlockCoord          blk_coord,
                       int const           thread_idx,
                       int&                work_idx,
                       int const           num_splits) {
    int const m_block   = get<0>(blk_coord);
    int const bidh      = get<1>(blk_coord);
    int const bidb      = get<2>(blk_coord);
    int const split_idx = get<3>(blk_coord);
    int const bidh_kv   = !IsPackGQA ? bidh / HeadRatio : bidh;

    auto [n_block_min, n_block_max] = BlockInfo::get_n_block_min_max(seqlen_info,
                                                                     m_block,
                                                                     split_idx,
                                                                     /*splits*/ num_splits,
                                                                     params.window_size_left,
                                                                     params.window_size_right,
                                                                     params.attention_chunk_divmod);

    TiledMmaQK tiled_mma_qk;
    TiledMmaPV tiled_mma_pv;
    TiledMmaQv tiled_mma_qv;

    Tensor acc_pv = make_tensor(acc_pv_storage.data(), partition_shape_C(tiled_mma_pv, take<0, 2>(TileShapePDVFull{})));

    constexpr int Rows = size<0>(layout_acc_mn(tiled_mma_pv, acc_pv.layout()));

    // If invalid, return empty result.
    if (n_block_min >= n_block_max) {
      auto lse = make_tensor<float>(Shape<Int<Rows>>{});
      return mute::make_tuple(false, mute::make_tuple(acc_pv, lse));
    }

    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQFull{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sP = make_tensor(make_smem_ptr(shared_storage.smem_p.data()), SmemLayoutP{});
    Tensor sV = [&]() {
      if constexpr (InKernelTranspose) {
        return make_tensor(make_smem_ptr(shared_storage.smem_vt.data()), SmemLayoutVMmaPV{});
      } else {
        return make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVMmaPV{});
      }
    }();

    Tensor sQv = [&]() {
      if constexpr (HasQv) {
        return make_tensor(make_smem_ptr(shared_storage.smem_qv.data()), SmemLayoutQvFull{});
      } else {
        return make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQv{});
      }
    }();
    Tensor sVMmaQV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVMmaQV{});

    ThrMMA thr_mma_qk = tiled_mma_qk.get_thread_slice(thread_idx);
    ThrMMA thr_mma_pv = tiled_mma_pv.get_thread_slice(thread_idx);
    ThrMMA thr_mma_qv = tiled_mma_qv.get_thread_slice(thread_idx);

    Tensor tSrQ = thr_mma_qk.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma_qk.partition_fragment_B(sK);
    Tensor tOrP = thr_mma_pv.partition_fragment_A(sP);
    Tensor tOrV = thr_mma_pv.partition_fragment_B(sV);

    Tensor tSrQv = thr_mma_qv.partition_fragment_A(sQv);
    Tensor tSrV  = thr_mma_qv.partition_fragment_B(sVMmaQV);

    // Keep adjacent pipeline phases on different physical barriers so their waiters cannot overlap on PH1.
    auto sync_pipeline_wrap = [&](auto state) {
      if (state.barrier_index() == 0) {
        auto const barrier_id =
            state.phase() == 0 ? FwdNamedBarriers::PipelineWrapPhase0 : FwdNamedBarriers::PipelineWrapPhase1;
        named_barrier_sync(static_cast<uint32_t>(barrier_id));
      }
    };

    auto wait_k = [&](auto state) {
      if constexpr (IsHeadDimTiled && !HasQv) {
        sync_pipeline_wrap(state);
      }
      pipeline_k.consumer_wait(state);
    };

    static constexpr bool RequiresVPhaseGuard =
        IsHeadDimTiled &&
        (!HasQv || PipelineVState::BarPerStageRatio == 1 || (2 * PVIterations) % PipelineVState::BarrierRingSize != 0);
    auto wait_v = [&](auto state) {
      if constexpr (RequiresVPhaseGuard) {
        sync_pipeline_wrap(state);
      }
      pipeline_v.consumer_wait(state);
    };

    auto wait_pv = [&](auto state) {
      if constexpr (InKernelTranspose) {
        pipeline_vt.consumer_wait(state);
      } else {
        wait_v(state);
      }
    };

    auto gemm_pv = [&](auto state, auto pv_iter) {
      mute::gemm(tiled_mma_pv, tOrP, tOrV(_, _, _, state.index()), acc_pv_storage(_, _, _, pv_iter));
    };

    auto release_pv = [&](auto state) { pipeline_vt.consumer_release(state); };

    R2STiledCopy tiled_copy_r2s;
    ThrCopy      thr_copy_r2s = tiled_copy_r2s.get_thread_slice(thread_idx);
    Tensor       tPsP         = thr_copy_r2s.partition_D(sP);

    float effective_scale      = HasSoftcap ? 1.0f : params.softmax_scale;
    float effective_scale_log2 = HasSoftcap ? static_cast<float>(M_LOG2E) : params.softmax_scale_log2;
    float softcap_val          = params.softcap_val;
    float softcap_scale        = params.softmax_scale / params.softcap_val;
    float qk_descale           = 1.f;
    float qv_descale           = 1.f;

    if constexpr (IsFP8) {
      float const q_descale = [&] {
        if constexpr (!HasQDescale) {
          return 1.f;
        } else {
          auto q_index = bidb * get<0>(params.stride_q_descale) + bidh_kv * get<1>(params.stride_q_descale);
          return params.ptr_q_descale[q_index];
        }
      }();
      float const k_descale = [&] {
        if constexpr (!HasKDescale || OnlyQv) {
          return 1.f;
        } else {
          auto k_index = bidb * get<0>(params.stride_k_descale) + bidh_kv * get<1>(params.stride_k_descale);
          return params.ptr_k_descale[k_index];
        }
      }();
      float const v_descale = [&] {
        if constexpr (!HasVDescale) {
          return 1.f;
        } else {
          auto v_index = bidb * get<0>(params.stride_v_descale) + bidh_kv * get<1>(params.stride_v_descale);
          return params.ptr_v_descale[v_index];
        }
      }();

      qk_descale = q_descale * k_descale;
      qv_descale = q_descale * v_descale;

      if constexpr (!HasQv) {
        if constexpr (HasSoftcap) {
          softcap_scale *= qk_descale;
        } else {
          effective_scale *= qk_descale;
          effective_scale_log2 *= qk_descale;
        }
      }
    }

    Softmax<Rows, HasLearnableSink, MaxOffset> softmax{effective_scale, effective_scale_log2};

    auto write_P_to_smem = [&](auto& accum_cvt) {
      if constexpr (ReuseKPStorage) {
        // K and P alias storage; all QK consumers must finish reading K before P overwrites it.
        named_barrier_sync(static_cast<uint32_t>(FwdNamedBarriers::ReuseP));
      }
      Tensor tPrP = thr_copy_r2s.retile_S(accum_cvt);
      copy(tiled_copy_r2s, tPrP, tPsP);
      // TODO: remote sync
    };

    auto arrive_on_P_write_barrier = [&] {
      __syncwarp();
      // TODO: remote sync
    };

    auto gemm_qv = [&](auto& acc_score, auto state, auto pv_iter) {
      wait_v(state);
      if constexpr (IsFP8 && HasQv && IsHeadDimTiled) {
        mute::gemm(tiled_mma_qv, tSrQv(_, _, _, pv_iter), tSrV(_, _, _, state.index()), acc_score);
      } else if constexpr (IsFP8) {
        Tensor acc_qv = make_fragment_like(acc_score);
        clear(acc_qv);
        mute::gemm(tiled_mma_qv, tSrQv(_, _, _, pv_iter), tSrV(_, _, _, state.index()), acc_qv);

        MUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(acc_score); ++i) {
          if constexpr (!OnlyQv) {
            acc_score(i) *= qk_descale;
          }
          acc_score(i) += acc_qv(i) * qv_descale;
        }
      } else {
        mute::gemm(tiled_mma_qv, tSrQv(_, _, _, pv_iter), tSrV(_, _, _, state.index()), acc_score);
      }
    };

    auto gemm_qk = [&](auto& acc_qk, auto state, auto qk_iter) {
      if constexpr (!OnlyQv) {
        mute::gemm(tiled_mma_qk, tSrQ(_, _, _, qk_iter), tSrK(_, _, _, state.index()), acc_qk);
      }
    };

    constexpr auto qk_iter_seq = make_seq<QKIterations>{};
    constexpr auto pv_iter_seq = make_seq<PVIterations>{};

    constexpr auto reserve_k_before_release = bool_constant<IsHeadDimTiled>{};
    constexpr auto reserve_v_before_release = bool_constant<(PVIterations > 1)>{};

    auto take_next_state = [](auto& cursor, auto reserve_before_release) {
      auto state = cursor;
      if constexpr (decltype(reserve_before_release)::value) {
        ++cursor;
      }
      return state;
    };

    auto advance_cursor_after_release = [](auto& cursor, auto reserve_before_release) {
      if constexpr (!decltype(reserve_before_release)::value) {
        ++cursor;
      }
    };

    auto gemm_qv_chunks = [&](auto& acc_score) {
      for_each(pv_iter_seq, [&](auto pv_iter) {
        auto v_state = take_next_state(smem_pipe_read_v, reserve_v_before_release);
        gemm_qv(acc_score, v_state, pv_iter);
        mate::warpsquad_commit_batch();
        mate::warpsquad_wait<0>();
        pipeline_v.consumer_release(v_state);
        advance_cursor_after_release(smem_pipe_read_v, reserve_v_before_release);
      });
    };

    auto gemm_qk_chunks = [&](auto& acc_qk) {
      PipelineKState retained_kp_state;
      for_each(qk_iter_seq, [&](auto qk_iter) {
        auto state = take_next_state(smem_pipe_read_k, reserve_k_before_release);
        wait_k(state);
        gemm_qk(acc_qk, state, qk_iter);
        if constexpr (InKernelTranspose) {
          auto qv_v_state = take_next_state(smem_pipe_read_v, reserve_v_before_release);
          gemm_qv(acc_qk, qv_v_state, _0{});
          mate::warpsquad_commit_batch();
          mate::warpsquad_wait<0>();
          pipeline_k.consumer_release(state);
          pipeline_v.consumer_release(qv_v_state);
          advance_cursor_after_release(smem_pipe_read_v, reserve_v_before_release);
        } else {
          if constexpr (!OnlyQv) {
            mate::warpsquad_commit_batch();
            mate::warpsquad_wait<0>();
          }
          if constexpr (ReuseKPStorage) {
            retained_kp_state = state;
          } else {
            pipeline_k.consumer_release(state);
          }
        }
        advance_cursor_after_release(smem_pipe_read_k, reserve_k_before_release);
      });
      return retained_kp_state;
    };

    auto gemm_pv_chunks = [&]() {
      for_each(pv_iter_seq, [&](auto pv_iter) {
        auto state = take_next_state(smem_pipe_read_vt, reserve_v_before_release);
        wait_pv(state);
        gemm_pv(state, pv_iter);
        mate::warpsquad_commit_batch();
        mate::warpsquad_wait<0>();
        release_pv(state);
        advance_cursor_after_release(smem_pipe_read_vt, reserve_v_before_release);
      });
    };

    auto apply_softcap = [&](auto& acc_qk) {
      if constexpr (HasSoftcap) {
        // float const scale_times_inv_softcap = params.softmax_scale / softcap_val;
        MUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(acc_qk); ++i) {
          acc_qk(i) = softcap_val * mutlass::fast_tanh(acc_qk(i) * softcap_scale);
        }
      }
    };

    int const seqlen_q = seqlen_info.seqlen_q;
    int const seqlen_k = seqlen_info.seqlen_k;

    // Q is ready
    if constexpr (!IsAppendKV || !IsRotary) {
      pipeline_q.consumer_wait(smem_pipe_read_q);
    } else {  // Rotary
      using Rotary_t = Rotary<TileM,
                              HeadDimQK,
                              NumQKMmaThreads,
                              Element,
                              FragmentSize,
                              !(IsCausal || IsLocal) /*FixedPosition*/,
                              HeadRatio>;
      Rotary_t rotary{params.ptr_rotary_cos,
                      params.shape_rotary,
                      params.stride_rotary_cos,
                      params.ptr_rotary_sin,
                      params.stride_rotary_sin,
                      thread_idx,
                      seqlen_q,
                      seqlen_info.seqlen_rotary};

      Tensor sQ_pi = mute::as_position_independent_swizzle_tensor(
          make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQRotary{}));

      auto [tRrCos, tRrSin] =
          conditional_return<!IsPackGQA>(rotary.template load_cos_sin<IsRotaryInterleaved /*IsInterleaved*/>(
                                             m_block, params.desc_Cos, params.desc_Sin),
                                         rotary.template load_cos_sin_packgqa<IsRotaryInterleaved /*IsInterleaved*/>(
                                             m_block, params.desc_Cos, params.desc_Sin));
      pipeline_q.consumer_wait(smem_pipe_read_q);
      if constexpr (IsRotaryInterleaved) {
        rotary.apply_Q_interleaved(sQ_pi, tRrCos, tRrSin, m_block);
      } else {
        rotary.apply_Q_contiguous(sQ_pi, tRrCos, tRrSin, m_block);
      }
      __syncwarp();
      named_barrier_sync(
          static_cast<uint32_t>(FwdNamedBarriers::RotaryQ));  // Ensure rotated Q is visible to all consumers
    }
    if constexpr (HasQv) {
      pipeline_qv.consumer_wait(smem_pipe_read_qv);
    }

    // Tensor acc_pv = partition_fragment_C(tiled_mma_pv, take<0, 2>(TileShapePDV{}));

    clear(acc_pv);

    // if (thread_idx == 0) {
    //   SHOW(tPsP);
    // }

    Mask<PermuteTiledMmaQK,
         SeqlenInfo,
         TileM,
         TileN,
         HeadRatio,
         IsPackGQA,
         IsCausal,
         IsLocal,
         HasAttentionChunk,
         EnableCP>
        mask(thread_idx,
             seqlen_info,
             params.window_size_left,
             params.window_size_right,
             params.sink_token_length,
             params.attention_chunk_divmod);

    int n_block = n_block_max - 1;

    if constexpr (IntraWarpSquadOverlap) {
      constexpr bool UseDeferredRescale = PVIterations > 1 && PVIterations <= 4;
      Tensor         acc_qk             = partition_fragment_C(tiled_mma_qk, take<0, 2>(TileShapeQKD{}));
      // trigger zero init sqmma
      clear(acc_qk);

      gemm_qk_chunks(acc_qk);

      apply_softcap(acc_qk);

      // Mask Mode
      mask.template apply</*SeqlenMask*/ true>(acc_qk, m_block, n_block);

      // Softmax
      Tensor correction_scales = softmax.template online_softmax<true, true>(acc_qk, tiled_mma_qk);

      Tensor accum_cvt = make_fragment_like<Element>(acc_qk);
      convert_type<CvtFragmentSize>(acc_qk, accum_cvt);

      if constexpr (!IsMmaPvRS) {
        write_P_to_smem(accum_cvt);
      }
      if constexpr (!IsMmaPvRS) {
        arrive_on_P_write_barrier();
      }

      --n_block;

      bool has_pending_rescale = false;
      auto rescale_before_pv   = [&](auto pv_iter) {
        if (has_pending_rescale) {
          auto acc_pv_chunk = acc_pv_storage(_, _, _, pv_iter);
          softmax.rescale_o(acc_pv_chunk, tiled_mma_pv, correction_scales);
        }
      };
      // Overlap QK for the current block with PV from the previous block.
      auto fwd_step = [&](int const n_block, auto mask_fn, auto check_inf_type) {
        static constexpr bool CheckInf = decltype(check_inf_type)::value;

        Tensor acc_qk = partition_fragment_C(tiled_mma_qk, take<0, 2>(TileShapeQKD{}));
        // trigger zero init sqmma
        clear(acc_qk);

        PipelineVState previous_pv_state;
        PipelineVState final_pv_state;
        auto           state = take_next_state(smem_pipe_read_k, reserve_k_before_release);
        wait_k(state);
        for_each(take<0, QKIterations - 1>(qk_iter_seq), [&](auto qk_iter) {
          auto current_state = state;
          gemm_qk(acc_qk, current_state, qk_iter);
          mate::warpsquad_commit_batch();
          state = take_next_state(smem_pipe_read_k, reserve_k_before_release);
          wait_k(state);
          mate::warpsquad_wait<0>();
          pipeline_k.consumer_release(current_state);
        });

        gemm_qk(acc_qk, state, back(qk_iter_seq));
        PipelineVState qv_v_state;
        if constexpr (InKernelTranspose) {
          qv_v_state = take_next_state(smem_pipe_read_v, reserve_v_before_release);
          gemm_qv(acc_qk, qv_v_state, _0{});
        }
        mate::warpsquad_commit_batch();

        auto first_pv_state = take_next_state(smem_pipe_read_vt, reserve_v_before_release);
        wait_pv(first_pv_state);
        if constexpr (UseDeferredRescale) {
          rescale_before_pv(get<0>(pv_iter_seq));
        }
        gemm_pv(first_pv_state, get<0>(pv_iter_seq));
        mate::warpsquad_commit_batch();

        mate::warpsquad_wait<1>();
        pipeline_k.consumer_release(state);
        advance_cursor_after_release(smem_pipe_read_k, reserve_k_before_release);
        if constexpr (InKernelTranspose) {
          pipeline_v.consumer_release(qv_v_state);
          advance_cursor_after_release(smem_pipe_read_v, reserve_v_before_release);
        }

        previous_pv_state = first_pv_state;
        final_pv_state    = first_pv_state;
        for_each(take<1, PVIterations>(pv_iter_seq), [&](auto pv_iter) {
          auto current_pv_state = take_next_state(smem_pipe_read_vt, reserve_v_before_release);
          wait_pv(current_pv_state);
          if constexpr (UseDeferredRescale) {
            rescale_before_pv(pv_iter);
          }
          gemm_pv(current_pv_state, pv_iter);
          mate::warpsquad_commit_batch();
          if constexpr (decltype(pv_iter)::value + 1 < PVIterations) {
            mate::warpsquad_wait<1>();
            release_pv(previous_pv_state);
            previous_pv_state = current_pv_state;
          } else {
            final_pv_state = current_pv_state;
          }
        });

        apply_softcap(acc_qk);

        // Mask Mode
        mask_fn(acc_qk, n_block);

        // Softmax
        mute::copy(softmax.template online_softmax<false, CheckInf>(acc_qk, tiled_mma_qk), correction_scales);

        if constexpr (PVIterations > 1) {
          mate::warpsquad_wait<1>();
          release_pv(previous_pv_state);
        }
        mate::warpsquad_wait<0>();
        release_pv(final_pv_state);
        advance_cursor_after_release(smem_pipe_read_vt, reserve_v_before_release);

        Tensor accum_cvt = make_fragment_like<Element>(acc_qk);
        convert_type<CvtFragmentSize>(acc_qk, accum_cvt);

        if constexpr (!IsMmaPvRS) {
          write_P_to_smem(accum_cvt);
        }
        if constexpr (!UseDeferredRescale) {
          softmax.rescale_o(acc_pv, tiled_mma_pv, correction_scales);
        } else {
          has_pending_rescale = true;
        }
        if constexpr (!IsMmaPvRS) {
          arrive_on_P_write_barrier();
        }
      };

      // Causal/Local Masking
      if constexpr (IsCausal || IsLocal) {
        auto mask_fn = [&](auto& tSrS, int n_block) {
          mask.template apply</*seqlenk mask*/ false>(tSrS, m_block, n_block);
        };
        int const n_block_min_causal_local_mask = BlockInfo::get_n_block_min_causal_local_mask(
            seqlen_info, m_block, n_block_min, params.window_size_right, params.attention_chunk_divmod);

        for (; n_block >= n_block_min_causal_local_mask; --n_block) {
          fwd_step(n_block, mask_fn, /* CheckInf */ mute::true_type{});
        }
      }
      // No mask iterations
      int const n_block_min_before_local_mask = BlockInfo::get_n_block_min_before_local_mask(
          seqlen_info, m_block, n_block_min, params.window_size_left, params.attention_chunk_divmod);
      auto no_mask_fn = [](auto& tSrS, int n_block) {};
      for (; n_block >= n_block_min_before_local_mask; --n_block) {
        fwd_step(n_block, no_mask_fn, /* CheckInf */ mute::false_type{});
      }

      // Local mask iterations
      if constexpr (IsLocal) {
        auto local_mask_fn = [&](auto& tSrS, int n_block) {
          mask.template apply</*SeqlenKMask*/ false>(tSrS, m_block, n_block);
        };
        for (; n_block >= n_block_min; --n_block) {
          fwd_step(n_block, local_mask_fn, /* CheckInf */ mute::true_type{});
        }
      }

      if constexpr (UseDeferredRescale) {
        softmax.rescale_o(acc_pv, tiled_mma_pv, correction_scales);
      }

      pipeline_q.consumer_release(smem_pipe_read_q);
      ++smem_pipe_read_q;

      // Last PV MMA
      gemm_pv_chunks();
    } else {
      auto fwd_step = [&](int const n_block, auto mask_fn, auto is_first_iter_type, auto check_inf_type) {
        static constexpr bool IsFirstIter = decltype(is_first_iter_type)::value;
        static constexpr bool CheckInf    = decltype(check_inf_type)::value;

        Tensor acc_qk = partition_fragment_C(tiled_mma_qk, take<0, 2>(TileShapeQKD{}));
        // trigger zero init sqmma
        clear(acc_qk);

        PipelineKState retained_kp_state;
        if constexpr (IsFP8 && HasQv && IsHeadDimTiled) {
          Tensor acc_qv = make_fragment_like(acc_qk);
          clear(acc_qv);
          gemm_qv_chunks(acc_qv);
          retained_kp_state = gemm_qk_chunks(acc_qk);
          MUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(acc_qk); ++i) {
            if constexpr (OnlyQv) {
              acc_qk(i) = acc_qv(i) * qv_descale;
            } else {
              acc_qk(i) = acc_qk(i) * qk_descale + acc_qv(i) * qv_descale;
            }
          }
        } else {
          if constexpr (HasQv && IsHeadDimTiled) {
            gemm_qv_chunks(acc_qk);
          }
          retained_kp_state = gemm_qk_chunks(acc_qk);
        }

        apply_softcap(acc_qk);

        // Mask Mode
        mask_fn(acc_qk, n_block);

        // Softmax
        Tensor correction_scales = softmax.template online_softmax<IsFirstIter, CheckInf>(acc_qk, tiled_mma_qk);

        Tensor accum_cvt = make_fragment_like<Element>(acc_qk);
        convert_type<CvtFragmentSize>(acc_qk, accum_cvt);
        if constexpr (!IsMmaPvRS) {
          write_P_to_smem(accum_cvt);
        }
        if constexpr (!IsFirstIter) {
          softmax.rescale_o(acc_pv, tiled_mma_pv, correction_scales);
        }
        if constexpr (!IsMmaPvRS) {
          arrive_on_P_write_barrier();
        }

        gemm_pv_chunks();
        if constexpr (ReuseKPStorage) {
          pipeline_k.consumer_release(retained_kp_state);
        }
      };

      auto first_iter_mask_fn = [&](auto& tSrS, int n_block) {
        mask.template apply</*seqlenk mask*/ true>(tSrS, m_block, n_block);
      };
      fwd_step(n_block, first_iter_mask_fn, /*IsFirstIter*/ mute::true_type{}, /*CheckInf*/ mute::true_type{});
      --n_block;

      // Causal/Local Masking
      if constexpr (IsCausal || IsLocal) {
        auto mask_fn = [&](auto& tSrS, int n_block) {
          mask.template apply</*seqlenk mask*/ false>(tSrS, m_block, n_block);
        };

        int const n_block_min_causal_local_mask = BlockInfo::get_n_block_min_causal_local_mask(
            seqlen_info, m_block, n_block_min, params.window_size_right, params.attention_chunk_divmod);

        for (; n_block >= n_block_min_causal_local_mask; --n_block) {
          fwd_step(n_block, mask_fn, /* IsFirstIter */ mute::false_type{}, /* CheckInf */ mute::true_type{});
        }
      }

      // No mask iterations
      int const n_block_min_before_local_mask = BlockInfo::get_n_block_min_before_local_mask(
          seqlen_info, m_block, n_block_min, params.window_size_left, params.attention_chunk_divmod);
      auto no_mask_fn = [](auto& tSrS, int n_block) {};
      for (; n_block >= n_block_min_before_local_mask; --n_block) {
        fwd_step(n_block, no_mask_fn, /* IsFirstIter */ mute::false_type{}, /* CheckInf */ mute::false_type{});
      }

      // Local mask iterations
      if constexpr (IsLocal) {
        auto local_mask_fn = [&](auto& tSrS, int n_block) {
          mask.template apply</*SeqlenKMask*/ false>(tSrS, m_block, n_block);
        };
        for (; n_block >= n_block_min; --n_block) {
          fwd_step(n_block, local_mask_fn, /*IsFirstIter*/ mute::false_type{}, /* CheckInf */ mute::true_type{});
        }
      }

      // release q
      pipeline_q.consumer_release(smem_pipe_read_q);
      ++smem_pipe_read_q;
    }
    if constexpr (HasQv) {
      pipeline_qv.consumer_release(smem_pipe_read_qv);
      ++smem_pipe_read_qv;
    }
    ++work_idx;

    auto sink_vals = [&]() {
      if constexpr (IsPackGQA) {
        return make_tensor<ElementAccumulator>(Shape<Int<Rows>>{});
      } else {
        return make_tensor<ElementAccumulator>(Layout<Shape<Int<Rows>>, Stride<Int<0>>>{});
      }
    }();
    if constexpr (HasLearnableSink) {
      if constexpr (IsPackGQA) {
        int const  head_idx_qo_base = bidh * HeadRatio;
        auto const tScS_m           = mask.tScS_mn(_, _0{});
        MUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < Rows; ++i) {
          int const head_idx_qo = head_idx_qo_base + (m_block * TileM + get<0>(tScS_m(i))) % HeadRatio;
          sink_vals(i)          = static_cast<ElementAccumulator>(params.ptr_learnable_sink[head_idx_qo]);
        }
      } else {
        int const head_idx_qo = bidh;
        sink_vals(0)          = static_cast<ElementAccumulator>(params.ptr_learnable_sink[head_idx_qo]);
      }
    }

    float const v_descale =
        !IsFP8 || !HasVDescale
            ? 1.0f
            : params.ptr_v_descale[bidb * get<0>(params.stride_v_descale) + bidh_kv * get<1>(params.stride_v_descale)];
    Tensor lse = softmax.tail(acc_pv, tiled_mma_pv, sink_vals, v_descale);

    return mute::make_tuple(true, mute::make_tuple(acc_pv, lse));
  }

  template <class BlockCoord>
  MUTE_DEVICE bool load_kv_new(Params const&         params,
                               MainloopPipelineKNew& pipeline_k_new,
                               MainloopPipelineVNew& pipeline_v_new,
                               PipelineKNewState&    smem_pipe_write_k_new,
                               PipelineVNewState&    smem_pipe_write_v_new,
                               SharedStorage&        shared_storage,
                               SeqlenInfo const&     seqlen_info,
                               BlockCoord            blk_coord,
                               int const             warp_idx_in_warp_squad,
                               int&                  work_idx,
                               int const             num_splits) {
    int const m_block   = get<0>(blk_coord);
    int const bidh      = get<1>(blk_coord);
    int const bidb      = get<2>(blk_coord);
    int const split_idx = get<3>(blk_coord);

    auto [n_block_new_min, n_block_new_max] = BlockInfo::get_n_block_k_new_min_max(seqlen_info,
                                                                                   m_block,
                                                                                   bidb,
                                                                                   split_idx,
                                                                                   num_splits,
                                                                                   params.window_size_left,
                                                                                   params.window_size_right,
                                                                                   params.attention_chunk_divmod);
    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //   printf("MP=%d, bidm=%d, bidb=%d, bidh=%d, bids=%d, num_splits=%d, n_block_new_min=%d, n_block_new_max=%d\n",
    //          blockIdx.x,
    //          m_block,
    //          bidb,
    //          bidh,
    //          split_idx,
    //          num_splits,
    //          n_block_new_min,
    //          n_block_new_max);
    // }
    if (n_block_new_max <= n_block_new_min) {
      return false;
    }

    // AppendKV runs before the main Q load, so Q storage can stage K_new when K aliases V or is too small.
    auto smem_k_new_ptr =
        conditional_return<UseQStorageForKNew>(shared_storage.smem_q.data(), shared_storage.smem_k.data());
    Tensor sK = make_tensor(make_smem_ptr(smem_k_new_ptr), SmemLayoutKNew{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVMmaPV{});

    int const bidh_kv = !IsPackGQA ? bidh / HeadRatio : bidh;

    Tensor mKnew =
        params.tme_load_K_new.get_tme_tensor(params.shape_K_new)(_, _, bidh_kv, !HasCuseqlensKNew ? bidb : 0);
    auto shape_Vnew = make_shape(
        params.headdim_V, get<0>(params.shape_K_new), get<2>(params.shape_K_new), get<3>(params.shape_K_new));
    Tensor mVnew = params.tme_load_V_new.get_tme_tensor(shape_Vnew)(_, _, bidh_kv, !HasCuseqlensKNew ? bidb : 0);

    Tensor gKnew = local_tile(domain_offset(make_coord(seqlen_info.offset_k_new, _0{}), mKnew),
                              select<1, 2>(TileShapeQKD{}),
                              make_coord(_, _));  // (N, K_chunk, n_block, qk_iter)
    Tensor gVnew = local_tile(domain_offset(make_coord(_0{}, seqlen_info.offset_k_new), mVnew),
                              select<1, 2>(TileShapePDV{}),
                              make_coord(_, _));  // (K_v_chunk, N, pv_iter, n_block)

    auto   cta_tme_K_new = params.tme_load_K_new.get_slice(0);
    Tensor tKgKnew       = group_modes<0, 3>(cta_tme_K_new.partition_S(gKnew));

    auto   cta_tme_V_new = params.tme_load_V_new.get_slice(0);
    Tensor tVgVnew       = group_modes<0, 3>(cta_tme_V_new.partition_S(gVnew));
    Tensor tVsVnew       = group_modes<0, 3>(cta_tme_V_new.partition_D(sV));

    auto load_K_new = [&](int const n_block, auto const& smem_pipe_write) {
      pipeline_k_new.producer_acquire(smem_pipe_write);
      auto bar_id = pipeline_k_new.producer_get_barrier_id(smem_pipe_write);
      MUTLASS_PRAGMA_UNROLL
      for (int qk_iter = 0; qk_iter < QKIterations; ++qk_iter) {
        Tensor sK_chunk =
            local_tile(sK(_, _, smem_pipe_write.index()), select<1, 2>(TileShapeQKD{}), make_coord(_0{}, qk_iter));
        Tensor tKsKnew = group_modes<0, 3>(cta_tme_K_new.partition_D(sK_chunk));
        copy(params.tme_load_K_new.with(bar_id), tKgKnew(_, n_block, qk_iter), tKsKnew);
      }
    };

    auto load_V_new = [&](int const n_block, int const pv_iter, auto const& smem_pipe_write) {
      pipeline_v_new.producer_acquire(smem_pipe_write);
      auto bar_id = pipeline_v_new.producer_get_barrier_id(smem_pipe_write);
      copy(params.tme_load_V_new.with(bar_id), tVgVnew(_, pv_iter, n_block), tVsVnew(_, smem_pipe_write.index()));
    };

    bool should_load_kv = SingleProducerWarp || warp_idx_in_warp_squad == 0;

    // pipeline_kv_guard.producer_acquire(smem_pipe_write_kv_guard);

    // Unlike the Hopper kernel, we don't need barrier_O here.
    // This kernel doesn't have the async O-side epilogue / cluster handoff that keeps
    // shared memory alive across stages, so load_kv_new doesn't need an extra recycle
    // barrier before reusing smem_k and smem_v.
    // Note: TME copies are issued by a producer warp, not a single elected thread,
    // so we intentionally don't use elect_one_sync() here.
    for (int n_block = n_block_new_max - 1; n_block >= n_block_new_min; --n_block) {
      if (should_load_kv) {
        load_K_new(n_block, smem_pipe_write_k_new);
      }
      ++smem_pipe_write_k_new;
      MUTLASS_PRAGMA_UNROLL
      for (int pv_iter = 0; pv_iter < PVIterations; ++pv_iter) {
        if (should_load_kv) {
          load_V_new(n_block, pv_iter, smem_pipe_write_v_new);
        }
        ++smem_pipe_write_v_new;
      }
    }

    return true;
  }

  template <class BlockCoord>
  MUTLASS_DEVICE bool store_kv_new(Params const&         params,
                                   MainloopPipelineKNew& pipeline_k_new,
                                   MainloopPipelineVNew& pipeline_v_new,
                                   PipelineKNewState&    smem_pipe_read_k_new,
                                   PipelineVNewState&    smem_pipe_read_v_new,
                                   int const             thread_idx,
                                   SharedStorage&        shared_storage,
                                   SeqlenInfo const&     seqlen_info,
                                   BlockCoord            blk_coord,
                                   int const             num_splits) {
    int const m_block                       = get<0>(blk_coord);
    int const bidh                          = get<1>(blk_coord);
    int const bidb                          = get<2>(blk_coord);
    int const split_idx                     = get<3>(blk_coord);
    auto [n_block_new_min, n_block_new_max] = BlockInfo::get_n_block_k_new_min_max(seqlen_info,
                                                                                   m_block,
                                                                                   bidb,
                                                                                   split_idx,
                                                                                   num_splits,
                                                                                   params.window_size_left,
                                                                                   params.window_size_right,
                                                                                   params.attention_chunk_divmod);

    if (n_block_new_max <= n_block_new_min) {
      return false;
    }

    // This aliases the same storage selected by load_kv_new.
    auto smem_k_new_ptr =
        conditional_return<UseQStorageForKNew>(shared_storage.smem_q.data(), shared_storage.smem_k.data());
    Tensor sK =
        mute::as_position_independent_swizzle_tensor(make_tensor(make_smem_ptr(smem_k_new_ptr), SmemLayoutKNew{}));
    Tensor sV = mute::as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVLsu{}));

    int const bidh_kv = !IsPackGQA ? bidh / HeadRatio : bidh;
    int const bidb_kv = !HasKvBatchIdx ? bidb : params.kv_batch_idx[bidb];

    Tensor mK = make_tensor(make_gmem_ptr(params.ptr_K), params.shape_K, params.stride_K)(
        _, _, bidh_kv, !HasCuseqlensK ? bidb_kv : 0);
    auto shape_V = make_shape(get<0>(params.shape_K), params.headdim_V, get<2>(params.shape_K), get<3>(params.shape_K));
    Tensor mV =
        make_tensor(make_gmem_ptr(params.ptr_V), shape_V, params.stride_V)(_, _, bidh_kv, !HasCuseqlensK ? bidb_kv : 0);

    int const offset_k = seqlen_info.offset_k + seqlen_info.seqlen_k_og;

    Tensor gK = local_tile(domain_offset(make_coord(offset_k, _0{}), mK),
                           select<1, 2>(TileShapeQKDFull{}),
                           make_coord(_, _0{}));  // (N, K, _)
    Tensor gV = local_tile(domain_offset(make_coord(offset_k, _0{}), mV),
                           select<2, 1>(TileShapePDV{}),
                           make_coord(_, _));  // (N, K_v_chunk, n_block, pv_iter)

    int const seqlen_k_new = seqlen_info.seqlen_k_new;

    using Rotary_t = Rotary<TileN, HeadDimQK, NumMmaThreads, Element, FragmentSize>;

    auto     rotary_cos_ptr    = params.ptr_rotary_cos;
    auto     rotary_sin_ptr    = params.ptr_rotary_sin;
    auto     rotary_shape      = params.shape_rotary;
    auto     rotary_cos_stride = params.stride_rotary_cos;
    auto     rotary_sin_stride = params.stride_rotary_sin;
    uint32_t rotary_start      = seqlen_info.seqlen_rotary;
    if constexpr (EnableCP && IsRotary) {
      int const first_position =
          ((static_cast<int>(rotary_start) + seqlen_info.cp_world_size - 1 - seqlen_info.cp_rank) /
           seqlen_info.cp_world_size) *
              seqlen_info.cp_world_size +
          seqlen_info.cp_rank;
      rotary_cos_ptr += first_position * get<0>(params.stride_rotary_cos);
      rotary_sin_ptr += first_position * get<0>(params.stride_rotary_sin);
      rotary_shape = make_shape(seqlen_k_new, get<1>(params.shape_rotary));
      rotary_cos_stride =
          make_stride(get<0>(params.stride_rotary_cos) * seqlen_info.cp_world_size, get<1>(params.stride_rotary_cos));
      rotary_sin_stride =
          make_stride(get<0>(params.stride_rotary_sin) * seqlen_info.cp_world_size, get<1>(params.stride_rotary_sin));
      rotary_start = 0;
    }

    Rotary_t rotary{rotary_cos_ptr,
                    rotary_shape,
                    rotary_cos_stride,
                    rotary_sin_ptr,
                    rotary_sin_stride,
                    thread_idx,
                    seqlen_k_new,
                    rotary_start};

    // This is used to index into the batch dimension of mK and mV
    int const bidb_kv_idx = !HasCuseqlensKNew && !IsPagedKV ? bidb_kv : 0;

    using KVManager = PagedKVManager<IsPagedKV,
                                     Element,
                                     NumMmaThreads,
                                     TileN,
                                     HeadDimQK,
                                     HeadDimVO,
                                     true /* IsKVSameIter */,
                                     AppendKVLoadsPerRow>;

    // passing offset_k instead of leftpad_k will move the PageTable pointer to the right position
    KVManager paged_kv_manager{params.ptr_pagetable,
                               params.shape_pagetable,
                               params.stride_pagetable,
                               params.desc_page_table,
                               params.ptr_K,
                               params.shape_K,
                               params.stride_K,
                               params.desc_K,
                               params.ptr_V,
                               params.headdim_V,
                               params.stride_V,
                               params.desc_V,
                               params.page_size_divmod,
                               seqlen_k_new,
                               offset_k,  // seqlen_info.offset_k + seqlen_info.seqlen_k_og
                               thread_idx,
                               bidb_kv,
                               bidh_kv,
                               bidb_kv_idx};

    GmemTiledCopyAppendKV gmem_tiled_copy_kv;

    auto gmem_thr_copy_kv = gmem_tiled_copy_kv.get_thread_slice(thread_idx);

    Tensor tKgK = gmem_thr_copy_kv.partition_D(gK);
    Tensor tKsK = gmem_thr_copy_kv.partition_S(sK);  // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tVgV = gmem_thr_copy_kv.partition_D(gV);
    Tensor tVsV = gmem_thr_copy_kv.partition_S(sV);  // ((Atom,AtomNum),ATOM_M,ATOM_N)

    Tensor cK   = make_identity_tensor(select<1, 2>(TileShapeQKDFull{}));  // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tKcK = gmem_thr_copy_kv.partition_D(cK);
    Tensor tKpK = make_tensor<bool>(make_shape(size<2>(tKgK)));
    MUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size(tKpK); ++k) {
      tKpK(k) = get<1>(tKcK(_0{}, _0{}, k)) < get<1>(params.shape_K);
    }

    Tensor cV   = make_identity_tensor(select<2, 1>(TileShapePDV{}));  // (BLK_N,BLK_K_V) -> (blk_n,blk_k_v)
    Tensor tVcV = gmem_thr_copy_kv.partition_D(cV);
    static_assert(std::is_same_v<GmemLayoutAtomAppendKV, typename Rotary_t::LayoutAtom>);
    static_assert(!IsPagedKV || std::is_same_v<GmemLayoutAtomAppendKV, typename KVManager::GmemLayoutAtom>);

    auto store_K = [&](int const n_block, auto const& smem_pipe_read) {
      int const n_limit = std::min(seqlen_k_new - n_block * TileN, TileN);

      if constexpr (!IsRotary) {  // No rotary, smem -> rmem -> gmem directly
        pipeline_k_new.consumer_wait(smem_pipe_read);
        Tensor tKsK_cur = tKsK(_, _, _, smem_pipe_read.index());
        Tensor tKrK     = make_fragment_like(tKsK_cur);  // ((_8,_1),_4,_2):((_1,_0),_8,_32)
        Tensor tKrK_src = gmem_thr_copy_kv.retile_S(tKrK);
        copy(tKsK_cur, tKrK);
        if constexpr (!IsPagedKV) {
          Tensor tKgK_cur = tKgK(_, _, _, n_block);
          MUTLASS_PRAGMA_UNROLL
          for (int m = 0; m < size<1>(tKgK_cur); ++m) {
            bool row_valid = get<0>(tKcK(_0{}, m, _0{})) < n_limit;
            MUTLASS_PRAGMA_UNROLL
            for (int k = 0; k < size<2>(tKgK_cur); ++k) {
              bool pred = row_valid && tKpK(k);
              copy(gmem_tiled_copy_kv.with(params.desc_K).with(pred), tKrK_src(_, m, k), tKgK_cur(_, m, k));
            }
          }
        } else {
          paged_kv_manager.store_K(n_block, tKrK_src);
        }
      } else {
        Tensor gK_cur  = gK(_, _, n_block);
        auto   tPrKPtr = conditional_return<IsPagedKV>(paged_kv_manager.compute_K_ptr(), nullptr);

        auto [tRrCos, tRrSin] = rotary.template load_cos_sin<IsRotaryInterleaved /*kInterleaved*/>(
            n_block, params.desc_Cos, params.desc_Sin);
        pipeline_k_new.consumer_wait(smem_pipe_read);
        if constexpr (IsRotaryInterleaved) {
          rotary.template apply_K_interleaved<IsPagedKV>(sK(_, _, smem_pipe_read.index()),
                                                         gK_cur,
                                                         tKpK,
                                                         tRrCos,
                                                         tRrSin,
                                                         tPrKPtr,
                                                         n_block,
                                                         get<1>(params.shape_K),
                                                         params.desc_K);
        } else {
          rotary.template apply_K_contiguous<IsPagedKV>(sK(_, _, smem_pipe_read.index()),
                                                        gK_cur,
                                                        tKpK,
                                                        tRrCos,
                                                        tRrSin,
                                                        tPrKPtr,
                                                        n_block,
                                                        get<1>(params.shape_K),
                                                        params.desc_K);
        }
      }

      pipeline_k_new.consumer_release(smem_pipe_read);
    };

    auto store_V = [&](int const n_block, int const pv_iter, auto const& smem_pipe_read) {
      int const n_limit        = std::min(seqlen_k_new - n_block * TileN, TileN);
      int const pv_head_offset = pv_iter * TileHeadDimVO;

      pipeline_v_new.consumer_wait(smem_pipe_read);
      Tensor tVsV_cur = tVsV(_, _, _, smem_pipe_read.index());
      Tensor tVrV     = make_fragment_like(tVsV_cur);
      Tensor tVrV_src = gmem_thr_copy_kv.retile_S(tVrV);
      copy(tVsV_cur, tVrV);
      if constexpr (!IsPagedKV) {
        Tensor tVgV_cur = tVgV(_, _, _, n_block, pv_iter);
        MUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < size<1>(tVgV_cur); ++m) {
          bool row_valid = get<0>(tVcV(_0{}, m, _0{})) < n_limit;
          MUTLASS_PRAGMA_UNROLL
          for (int k = 0; k < size<2>(tVgV_cur); ++k) {
            bool pred = row_valid && get<1>(tVcV(_0{}, _0{}, k)) + pv_head_offset < params.headdim_V;
            copy(gmem_tiled_copy_kv.with(params.desc_V).with(pred), tVrV_src(_, m, k), tVgV_cur(_, m, k));
          }
        }
      } else {
        paged_kv_manager.store_V(n_block, tVrV_src, pv_head_offset);
      }
      pipeline_v_new.consumer_release(smem_pipe_read);
    };

    for (int n_block = n_block_new_max - 1; n_block >= n_block_new_min; --n_block) {
      if constexpr (IsPagedKV) {
        if constexpr (IsRotary) {
          paged_kv_manager.template load_page_table_for_lsu<false /* FirstIter */,
                                                            false /* PermuteK */,
                                                            false /* PermuteV */,
                                                            Rotary_t::GmemThreadsPerRow>(n_block);
        } else {
          paged_kv_manager
              .template load_page_table_for_lsu<false /* FirstIter */, false /* PermuteK */, false /* PermuteV */>(
                  n_block);
        }
      }
      store_K(n_block, smem_pipe_read_k_new);
      ++smem_pipe_read_k_new;
      MUTLASS_PRAGMA_UNROLL
      for (int pv_iter = 0; pv_iter < PVIterations; ++pv_iter) {
        store_V(n_block, pv_iter, smem_pipe_read_v_new);
        ++smem_pipe_read_v_new;
      }
    }

    return true;
  }
};

}  // namespace mate::attention::fmha
