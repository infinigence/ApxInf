// apxinf: hook for the shape used to TYPE an SM100 collective's TMA atoms.
//
// A collective types its `Params::TMA_*` from a dummy tensor
//   make_tensor(nullptr, repeat_like(StrideA{}, int32_t(0)), StrideA{})
// i.e. a shape congruent to the stride with every extent DYNAMIC, while
// to_underlying_arguments() builds the real tensor from the problem shape --
// which keeps whatever static extents that shape has.  The two must be the
// same type.
//
// For flat problem shapes that is automatic (all extents dynamic on both
// sides).  pack-GQA needs a HIERARCHICAL M mode (q_per_kv, seq_q) whose leading
// extent MUST be static: make_tma_copy does
//   CUTE_STATIC_ASSERT_V(size(slayout) == size(cta_v_map))
// and derives the gmem vectorisation rank from a static_assert, so a dynamic
// tile-M size does not compile.  Specialise this trait for such a stride type
// to pin the static extents on the dummy shape as well.
//
// The PRIMARY template is exactly upstream's expression, so including this
// header and routing a dummy shape through it changes nothing for any stride
// type that is not explicitly specialised.
#pragma once
#include "cute/layout.hpp"

namespace cutlass::gemm::collective::detail {

template <class Stride>
struct ApxinfTmaDummyShape {
  CUTE_HOST_DEVICE static constexpr auto get() {
    return cute::repeat_like(Stride{}, int32_t(0));
  }
};

}  // namespace cutlass::gemm::collective::detail
