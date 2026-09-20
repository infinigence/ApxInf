#pragma once

// Copyright 2026 apxinf contributors.
// Pure CUDA operators grouped by physical operation; launch policy lives under adapters/.

// ── Argmax over [vocab] bf16 logits → u32 token id ─────────────────────────
//
// One block, many threads. Strided load, warp-shuffle max-reduction that
// also carries the argmax index (Fletcher's variant: pack value+index into
// a 64-bit lane where the high bits hold the value so an unsigned 64-bit
// max gives both the max value and its index). Writes the winning index to
// `out` (typically a host-mapped u32, so the CPU reads it zero-copy).
__global__ void argmax_bf16_kernel(
    const __nv_bfloat16* logits, uint32_t n, uint32_t* out)
{
    uint32_t tid = threadIdx.x;
    // Pack (value, index) as uint64: value in the high 32 bits (reinterpreted
    // from float bits via -value so larger float → larger uint), index low.
    auto pack = [](float v, uint32_t i) -> uint64_t {
        uint32_t bits = __float_as_uint(v);
        // Flip the sign bit for positive, invert all bits for negative, so the
        // uint ordering matches float ordering. Then bias to non-negative.
        uint32_t ordered = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
        return ((uint64_t)ordered << 32) | (uint64_t)i;
    };
    uint64_t best = 0;
    float best_v = -INFINITY;
    uint32_t best_i = 0;
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        float v = __bfloat162float(logits[i]);
        if (v > best_v) { best_v = v; best_i = i; }
    }
    best = pack(best_v, best_i);
    // Warp reduce: keep the (max value, its index).
    for (int off = 16; off > 0; off >>= 1) {
        uint64_t other = __shfl_xor_sync(0xffffffff, best, off);
        if (other > best) best = other;
    }
    uint32_t warp_id = tid / 32;
    uint32_t lane = tid % 32;
    __shared__ uint64_t warp_best[32];
    if (lane == 0) warp_best[warp_id] = best;
    __syncthreads();
    if (warp_id == 0) {
        uint64_t v = (tid < (blockDim.x + 31) / 32) ? warp_best[tid] : 0;
        for (int off = 16; off > 0; off >>= 1)
            v = max(v, __shfl_xor_sync(0xffffffff, v, off));
        if (lane == 0) *out = (uint32_t)v;   // low 32 bits = index
    }
}

// ── Argmax over a pruned [columns] bf16 logit slice → global token id ───────
//
// Identical reduction to `argmax_bf16_kernel`; the only difference is the final
// store, which maps the winning column through `remap` to the global token id
// that the embedding table, the decode loop and the caller's stop token all
// speak. A pruned LM head emits a slice of the vocabulary, so a raw column index
// would be meaningless outside the head.
//
// This is a separate operator rather than an extra argument on the full-vocab
// one so that every other family keeps the unchanged, branch-free kernel.
__global__ void argmax_remap_bf16_kernel(
    const __nv_bfloat16* logits, uint32_t n, const uint32_t* remap,
    uint32_t* out)
{
    uint32_t tid = threadIdx.x;
    auto pack = [](float v, uint32_t i) -> uint64_t {
        uint32_t bits = __float_as_uint(v);
        uint32_t ordered = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
        return ((uint64_t)ordered << 32) | (uint64_t)i;
    };
    uint64_t best = 0;
    float best_v = -INFINITY;
    uint32_t best_i = 0;
    for (uint32_t i = tid; i < n; i += blockDim.x) {
        float v = __bfloat162float(logits[i]);
        if (v > best_v) { best_v = v; best_i = i; }
    }
    best = pack(best_v, best_i);
    for (int off = 16; off > 0; off >>= 1) {
        uint64_t other = __shfl_xor_sync(0xffffffff, best, off);
        if (other > best) best = other;
    }
    uint32_t warp_id = tid / 32;
    uint32_t lane = tid % 32;
    __shared__ uint64_t warp_best[32];
    if (lane == 0) warp_best[warp_id] = best;
    __syncthreads();
    if (warp_id == 0) {
        uint64_t v = (tid < (blockDim.x + 31) / 32) ? warp_best[tid] : 0;
        for (int off = 16; off > 0; off >>= 1)
            v = max(v, __shfl_xor_sync(0xffffffff, v, off));
        if (lane == 0) *out = remap[(uint32_t)v];   // low 32 bits = column
    }
}



