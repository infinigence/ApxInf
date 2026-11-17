// Folded flash-decoding attention for `apxinf_hip::native`, specialized for
// the geometry pi05's action expert runs 180 times per forward: one batch,
// few query rows whose heads all share one KV head, head_dim 256, many keys,
// no mask.
//
//   q   [q_rows, 256]        q_rows = q_tokens * q_heads, head-major fold
//   k,v [kv_rows, 256]       rows `kv_row_stride` elements apart
//   out [q_rows, 256]
//
// The kernel pair (see native/attention.rs, `AttentionKernel::Folded`):
//
//   `folded_partial`  grid (q_rows/8, 8): one row per warp, lane l owns dims
//                     [8l, 8l+8). The warp's 32 uint4 loads cover one 512-B
//                     K (or V) row exactly once -- zero load redundancy, no
//                     LDS staging, no barriers. Two keys per iteration keep
//                     two independent butterfly chains in flight. The
//                     256-dim dot is a 5-step `ds_bpermute` butterfly (the
//                     `__shfl_xor_sync` wrapper costs ~11 instructions per
//                     call on this stack; the raw intrinsic is one, with the
//                     float bit-cast done explicitly so no `v_cvt` round
//                     trip creeps in). F32 online softmax throughout.
//   `folded_combine`  grid q_rows: merges the 8 splits' (max, sum, acc)
//                     partials from scratch [q_rows][8][258].
//
// K/V is read about once per 8-row group through L1/L2, so the pair runs at
// a fraction of the CK split-KV + combine it replaces (probe
// devlocal/pi05-hip/perf/attention_probe.cpp: 19.9 us against 27.5, warm L2).
// Scores, probabilities and the PV sum stay F32 and the output is rounded
// once -- the reference kernel's numerics, closer to the CUDA port than
// CK's BF16-rounded probabilities.

#include "common.h"

namespace {

using namespace apxinf_hip;
using bf16 = uint16_t;

// One-instruction lane exchange. AMD-only by construction: this translation
// unit only compiles for AMD targets, and the dispatch in attention.rs gates
// on RDNA (wave32) architectures.
__device__ __forceinline__ float bshuf(float v, int mask) {
  const int r = __builtin_amdgcn_ds_bpermute((__lane_id() ^ mask) << 2,
                                             __float_as_int(v));
  return __int_as_float(r);
}

__device__ __forceinline__ float lowf(uint32_t w) {
  return __uint_as_float(w << 16);
}
__device__ __forceinline__ float highf(uint32_t w) {
  return __uint_as_float(w & 0xffff0000u);
}

constexpr int kDim = 256;   // head_dim; this kernel is D-specialized
constexpr int kSplits = 8;  // K-splits, measured fastest (probe, 2026-11-17)
constexpr int kRowsPerBlock = 8;
constexpr int kStats = kDim + 2;  // acc + max + sum per (row, split)

__global__ void folded_partial(const bf16* q, const bf16* k, const bf16* v,
                               float* scratch, int kv_tokens,
                               int64_t kv_stride, float scale) {
  const int row = blockIdx.x * kRowsPerBlock + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const int split = blockIdx.y;
  const int lane8 = lane * 8;
  uint32_t q4[4];
  {
    const uint4 t = *reinterpret_cast<const uint4*>(q + (int64_t)row * kDim + lane8);
    q4[0] = t.x; q4[1] = t.y; q4[2] = t.z; q4[3] = t.w;
  }
  const int keys_per_split = (kv_tokens + kSplits - 1) / kSplits;
  const int j0 = split * keys_per_split;
  const int j1 = min(kv_tokens, j0 + keys_per_split);
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float run_max = -INFINITY;
  float run_sum = 0.0f;
  const bf16* kp = k + (int64_t)j0 * kv_stride + lane8;
  const bf16* vp = v + (int64_t)j0 * kv_stride + lane8;
  int j = j0;
  // Two keys per iteration: the two butterfly chains interleave and hide
  // each other's LDS round-trip latency, and the V loads issue early.
  for (; j + 1 < j1; j += 2, kp += 2 * kv_stride, vp += 2 * kv_stride) {
    const uint4 ka = *reinterpret_cast<const uint4*>(kp);
    const uint4 va = *reinterpret_cast<const uint4*>(vp);
    const uint4 kb = *reinterpret_cast<const uint4*>(kp + kv_stride);
    const uint4 vb = *reinterpret_cast<const uint4*>(vp + kv_stride);
    const uint32_t kwa[4] = {ka.x, ka.y, ka.z, ka.w};
    const uint32_t kwb[4] = {kb.x, kb.y, kb.z, kb.w};
    const uint32_t vwa[4] = {va.x, va.y, va.z, va.w};
    const uint32_t vwb[4] = {vb.x, vb.y, vb.z, vb.w};
    float da = 0.0f, db = 0.0f;
#pragma unroll
    for (int h = 0; h < 4; ++h) {
      da += lowf(q4[h]) * lowf(kwa[h]) + highf(q4[h]) * highf(kwa[h]);
      db += lowf(q4[h]) * lowf(kwb[h]) + highf(q4[h]) * highf(kwb[h]);
    }
    da += bshuf(da, 16); da += bshuf(da, 8); da += bshuf(da, 4); da += bshuf(da, 2); da += bshuf(da, 1);
    db += bshuf(db, 16); db += bshuf(db, 8); db += bshuf(db, 4); db += bshuf(db, 2); db += bshuf(db, 1);
    const float sa = da * scale, sb = db * scale;
    const float ma = fmaxf(run_max, sa);
    const float ca = __expf(run_max - ma), pa = __expf(sa - ma);
    const float mb = fmaxf(ma, sb);
    const float cb = __expf(ma - mb), pb = __expf(sb - mb);
    run_sum = (run_sum * ca + pa) * cb + pb;
    run_max = mb;
    const float c2 = ca * cb, pac = pa * cb;
#pragma unroll
    for (int h = 0; h < 4; ++h) {
      const float v0 = lowf(vwa[h]), v1 = highf(vwa[h]);
      const float u0 = lowf(vwb[h]), u1 = highf(vwb[h]);
      acc[2 * h] = acc[2 * h] * c2 + pac * v0 + pb * u0;
      acc[2 * h + 1] = acc[2 * h + 1] * c2 + pac * v1 + pb * u1;
    }
  }
  for (; j < j1; ++j, kp += kv_stride, vp += kv_stride) {
    const uint4 ka = *reinterpret_cast<const uint4*>(kp);
    const uint4 va = *reinterpret_cast<const uint4*>(vp);
    const uint32_t kwa[4] = {ka.x, ka.y, ka.z, ka.w};
    const uint32_t vwa[4] = {va.x, va.y, va.z, va.w};
    float da = 0.0f;
#pragma unroll
    for (int h = 0; h < 4; ++h) da += lowf(q4[h]) * lowf(kwa[h]) + highf(q4[h]) * highf(kwa[h]);
    da += bshuf(da, 16); da += bshuf(da, 8); da += bshuf(da, 4); da += bshuf(da, 2); da += bshuf(da, 1);
    const float sa = da * scale;
    const float ma = fmaxf(run_max, sa);
    const float ca = __expf(run_max - ma), pa = __expf(sa - ma);
    run_sum = run_sum * ca + pa;
    run_max = ma;
#pragma unroll
    for (int h = 0; h < 4; ++h) {
      acc[2 * h] = acc[2 * h] * ca + pa * lowf(vwa[h]);
      acc[2 * h + 1] = acc[2 * h + 1] * ca + pa * highf(vwa[h]);
    }
  }
  float* slot = scratch + ((int64_t)row * kSplits + split) * kStats;
  if (lane == 0) {
    slot[kDim] = run_max;
    slot[kDim + 1] = run_sum;
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) slot[lane8 + i] = acc[i];
}

__global__ void folded_combine(const float* scratch, bf16* out) {
  const int row = blockIdx.x;
  const int d = threadIdx.x;
  const float* base = scratch + (int64_t)row * kSplits * kStats;
  float m = -INFINITY;
  for (int s = 0; s < kSplits; ++s) m = fmaxf(m, base[s * kStats + kDim]);
  float sum = 0.0f, value = 0.0f;
  for (int s = 0; s < kSplits; ++s) {
    const float w = __expf(base[s * kStats + kDim] - m);
    sum += base[s * kStats + kDim + 1] * w;
    value += base[s * kStats + d] * w;
  }
  out[(int64_t)row * kDim + d] = f32_to_bf16(value / sum);
}

}  // namespace

extern "C" int apxinf_hip_folded_attention(const void* q, const void* k,
                                           const void* v, void* out,
                                           void* scratch, int q_rows,
                                           int kv_tokens, int64_t kv_row_stride,
                                           float scale, void* stream) {
  if (q_rows <= 0 || q_rows % kRowsPerBlock != 0 || kv_tokens <= 0 ||
      kv_row_stride < kDim || kv_row_stride % 8 != 0) {
    return kInvalidArgument;
  }
  hipStream_t s = as_stream(stream);
  const dim3 grid(q_rows / kRowsPerBlock, kSplits);
  folded_partial<<<grid, kRowsPerBlock * 32, 0, s>>>(
      static_cast<const bf16*>(q), static_cast<const bf16*>(k),
      static_cast<const bf16*>(v), static_cast<float*>(scratch), kv_tokens,
      kv_row_stride, scale);
  const hipError_t partial = hipGetLastError();
  if (partial != hipSuccess) return static_cast<int>(partial);
  folded_combine<<<q_rows, kDim, 0, s>>>(static_cast<const float*>(scratch),
                                          static_cast<bf16*>(out));
  return last_launch_status();
}
