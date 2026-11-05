// Copyright 2026 apxinf contributors.
// Native launch policy for the gated-delta-net fixed candidates.
//
// Every field here was swept on a board and came out different on the next
// one, so none of them is a portable semantic parameter. The native adapter
// selects this policy during explicit per-device preparation; Rust L3 and the
// public ABI do not expose these fields.
#pragma once

#include <cstdint>

extern "C" {

// Which form of a kernel that has a tensor-core implementation to run.
enum ApxinfGdnWmma : int32_t {
  APXINF_GDN_WMMA_OFF = 0,    // the scalar fp32 kernel
  APXINF_GDN_WMMA_LOSSY = 1,  // one BF16 pass; exact when the operands are BF16
};

struct ApxinfGdnPolicy {
  int32_t chunk_state_tile;
  int32_t chunk_state_threads;
  int32_t chunk_state_wmma;
  int32_t chunk_gemm_tile;
  int32_t recurrent_split;
  int32_t chunk_state_v_split;
};

}  // extern "C"
