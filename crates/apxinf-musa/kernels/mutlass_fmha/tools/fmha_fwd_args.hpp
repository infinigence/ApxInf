// The C ABI of the two frozen mp_31 FMHA instantiations.
//
// One declaration per frozen kernel, because each is defined in its own
// translation unit: `doc/musa-backend.md` records that a recompiled TU can move
// an unrelated kernel's bits, and MATE's own build compiles one `.mu` per
// shared object. Keeping the split reproduces that arrangement exactly, which
// is what makes the replay against MATE's output a meaningful comparison.
//
// Both return null on success and a static message otherwise, so the caller
// reads the reason through the return value rather than a second call.
#pragma once

#include <cstdint>

extern "C" const char* apxinf_musa_mp31_fmha_fwd_tile128_packed(
    const void* q,
    const void* k,
    const void* v,
    void* out,
    float* lse,
    const uint32_t* cu_seqlens_q,
    const uint32_t* cu_seqlens_k,
    int32_t tokens,
    int32_t key_tokens,
    void* stream);

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
    void* stream);
