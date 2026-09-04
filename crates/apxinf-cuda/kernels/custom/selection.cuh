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





// ── MoE router: softmax + top-k + optional renormalisation ───────────────
//
// One warp per token. `logits` is [tokens, experts] BF16 with experts <= 256.
// Writes `topk_idx[token][k]` (int32) and `topk_weight[token][k]` (fp32) in
// descending probability order, matching `torch.topk(softmax(logits), k)`
// followed by `w /= w.sum()` when `renormalize` is set (Qwen3-MoE
// `norm_topk_prob`).
__global__ void moe_router_topk_bf16_kernel(
    const __nv_bfloat16* __restrict__ logits, int32_t* __restrict__ topk_idx,
    float* __restrict__ topk_weight, int tokens, int experts, int k,
    int renormalize) {
  const int lane = threadIdx.x & 31;
  const int token = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (token >= tokens) return;
  constexpr int kPerLane = 8;  // experts <= 256
  float value[kPerLane];
  float row_max = -INFINITY;
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) {
    const int e = lane + 32 * i;
    value[i] = e < experts
                   ? __bfloat162float(logits[static_cast<int64_t>(token) * experts + e])
                   : -INFINITY;
    row_max = fmaxf(row_max, value[i]);
  }
  row_max = warp_max(row_max);
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) {
    value[i] = lane + 32 * i < experts ? expf(value[i] - row_max) : 0.0f;
    sum += value[i];
  }
  sum = warp_sum_all(sum);
  const float inv = 1.0f / sum;
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) value[i] *= inv;

  float selected_sum = 0.0f;
  for (int round = 0; round < k; ++round) {
    float best = -1.0f;
    int best_e = experts;
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      const int e = lane + 32 * i;
      if (e < experts && value[i] > best) {
        best = value[i];
        best_e = e;
      }
    }
    // Warp argmax: larger probability wins, ties go to the smaller index.
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float other = __shfl_xor_sync(0xffffffff, best, offset);
      const int other_e = __shfl_xor_sync(0xffffffff, best_e, offset);
      if (other > best || (other == best && other_e < best_e)) {
        best = other;
        best_e = other_e;
      }
    }
    if (lane == 0) {
      topk_idx[static_cast<int64_t>(token) * k + round] = best_e;
      topk_weight[static_cast<int64_t>(token) * k + round] = best;
    }
    selected_sum += best;
    if (best_e < experts && (best_e & 31) == lane) value[best_e >> 5] = -1.0f;
  }
  if (renormalize && lane < k) {
    topk_weight[static_cast<int64_t>(token) * k + lane] /= selected_sum;
  }
}
