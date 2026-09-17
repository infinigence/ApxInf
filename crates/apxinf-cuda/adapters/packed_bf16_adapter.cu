// Copyright 2026 apxinf contributors.
// Bit-exact repacking of BF16 weights, and the batch-1 GEMV that reads it.
//
// Decode is weight-bandwidth-bound: this model reads 8.41 GB of weights per
// generated token and the cuBLAS GEMVs already run at 96% of the device's
// measured 259.7 GB/s, so the only lever left is moving fewer bytes without
// changing a single one of them.
//
// A BF16 weight is sign(1) exponent(8) mantissa(7). Across this checkpoint the
// exponent field carries 2.58 bits of entropy while sign+mantissa carries
// 7.97, and in every tensor sampled 100.0000% of 128-weight blocks have an
// exponent range that fits in five bits. So store
//
//   lo      1 byte per weight    sign in bit 7, the 7 mantissa bits below
//   off_lo  4 bits per weight    low nibble of (exponent - block base)
//   off_hi  1 bit per weight     its fifth bit, in a separate plane
//   base    1 byte per 128       the block's minimum exponent
//
// 13.0625 bits against 16. The reconstruction
//   bf16 = (lo & 0x80) << 8 | (base + off) << 7 | (lo & 0x7F)
// returns the original sixteen bits, so a GEMV over the packed form produces
// the same output as one over the BF16 form, bit for bit.
//
// Packing refuses any block whose exponent range exceeds 31 rather than
// rounding it: the caller keeps the BF16 path for that weight.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kPackBlock = 128;

// Source is the loader's [k, n] row-major weight; the packed planes are [n, k]
// so a warp walking one output column reads contiguous bytes. One block per
// output column, blockDim threads striding k.
__global__ void pack_bf16_transposed_kernel(
    const __nv_bfloat16* __restrict__ src, uint8_t* __restrict__ lo,
    uint8_t* __restrict__ off_lo, uint8_t* __restrict__ off_hi,
    uint8_t* __restrict__ base, int32_t* __restrict__ reject, int k, int n) {
  const int col = blockIdx.x;
  if (col >= n) return;
  const int64_t out_base = static_cast<int64_t>(col) * k;
  for (int blk = threadIdx.x; blk < k / kPackBlock; blk += blockDim.x) {
    const int start = blk * kPackBlock;
    uint32_t mn = 255u;
    uint32_t mx = 0u;
    for (int t = 0; t < kPackBlock; ++t) {
      const __nv_bfloat16 value = src[static_cast<int64_t>(start + t) * n + col];
      uint16_t bits;
      memcpy(&bits, &value, 2);
      const uint32_t e = (bits >> 7) & 0xFFu;
      mn = e < mn ? e : mn;
      mx = e > mx ? e : mx;
    }
    if (mx - mn > 31u) {
      atomicExch(reject, 1);
      return;
    }
    base[out_base / kPackBlock + blk] = static_cast<uint8_t>(mn);
    for (int t = 0; t < kPackBlock; ++t) {
      const __nv_bfloat16 value = src[static_cast<int64_t>(start + t) * n + col];
      uint16_t bits;
      memcpy(&bits, &value, 2);
      const int64_t i = out_base + start + t;
      lo[i] = static_cast<uint8_t>(((bits >> 8) & 0x80u) | (bits & 0x7Fu));
      const uint32_t off = ((bits >> 7) & 0xFFu) - mn;
      // Two weights share a byte here and eight share a byte in the bit plane,
      // and a thread owns a whole 128-block, so neither write races.
      if (t & 1) {
        off_lo[i / 2] = static_cast<uint8_t>((off_lo[i / 2] & 0x0Fu) | ((off & 0xFu) << 4));
      } else {
        off_lo[i / 2] = static_cast<uint8_t>((off_lo[i / 2] & 0xF0u) | (off & 0xFu));
      }
      if ((t & 7) == 0) off_hi[i / 8] = 0;
      off_hi[i / 8] = static_cast<uint8_t>(off_hi[i / 8] | (((off >> 4) & 1u) << (t & 7)));
    }
  }
}

// One warp per output column, eight weights per lane per step: one uint2 of
// lo, one uint32 of nibbles, one byte of the bit plane. Thirteen bytes per
// eight weights against sixteen. The value is rebuilt straight into fp32,
// since a BF16 datum is its own fp32 pattern shifted left by 16.
__global__ void packed_gemv_bf16_kernel(
    const uint8_t* __restrict__ lo, const uint8_t* __restrict__ off_lo,
    const uint8_t* __restrict__ off_hi, const uint8_t* __restrict__ base,
    const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ y,
    int n, int k) {
  const int lane = threadIdx.x & 31;
  const int col = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (col >= n) return;
  const uint2* lo_r = reinterpret_cast<const uint2*>(lo + static_cast<int64_t>(col) * k);
  const uint32_t* ol_r =
      reinterpret_cast<const uint32_t*>(off_lo + static_cast<int64_t>(col) * (k / 2));
  const uint8_t* oh_r = off_hi + static_cast<int64_t>(col) * (k / 8);
  const uint8_t* bs_r = base + static_cast<int64_t>(col) * (k / kPackBlock);
  const float4* xr = reinterpret_cast<const float4*>(x);
  float acc = 0.0f;
  for (int i = lane; i < k / 8; i += 32) {
    const uint2 lov = lo_r[i];
    const uint32_t olv = ol_r[i];
    const uint32_t ohv = oh_r[i];
    const uint32_t b = bs_r[i / (kPackBlock / 8)];
    const float4 xv = xr[i];
    const __nv_bfloat16* xl = reinterpret_cast<const __nv_bfloat16*>(&xv);
#define APXINF_PACKED_ONE(WORD, J, BIT)                                        \
  {                                                                            \
    const uint32_t lb = ((WORD) >> ((J) * 8)) & 0xFFu;                          \
    const uint32_t off =                                                        \
        ((olv >> ((BIT) * 4)) & 0xFu) | (((ohv >> (BIT)) & 1u) << 4);            \
    const float w = __int_as_float(((lb & 0x80u) << 24) | ((b + off) << 23) |    \
                                   ((lb & 0x7Fu) << 16));                       \
    acc = fmaf(w, __bfloat162float(xl[BIT]), acc);                              \
  }
    APXINF_PACKED_ONE(lov.x, 0, 0)
    APXINF_PACKED_ONE(lov.x, 1, 1)
    APXINF_PACKED_ONE(lov.x, 2, 2)
    APXINF_PACKED_ONE(lov.x, 3, 3)
    APXINF_PACKED_ONE(lov.y, 0, 4)
    APXINF_PACKED_ONE(lov.y, 1, 5)
    APXINF_PACKED_ONE(lov.y, 2, 6)
    APXINF_PACKED_ONE(lov.y, 3, 7)
#undef APXINF_PACKED_ONE
  }
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, offset);
  }
  if (lane == 0) y[col] = __float2bfloat16(acc);
}

}  // namespace

extern "C" cudaError_t apxinf_pack_bf16_transposed(
    const void* src, void* lo, void* off_lo, void* off_hi, void* base,
    void* reject, int k, int n, cudaStream_t stream) {
  if (src == nullptr || lo == nullptr || off_lo == nullptr || off_hi == nullptr ||
      base == nullptr || reject == nullptr || k <= 0 || n <= 0 ||
      k % kPackBlock != 0) {
    return cudaErrorInvalidValue;
  }
  pack_bf16_transposed_kernel<<<n, 128, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(src), static_cast<uint8_t*>(lo),
      static_cast<uint8_t*>(off_lo), static_cast<uint8_t*>(off_hi),
      static_cast<uint8_t*>(base), static_cast<int32_t*>(reject), k, n);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_packed_gemv_bf16(
    const void* lo, const void* off_lo, const void* off_hi, const void* base,
    const void* x, void* y, int n, int k, cudaStream_t stream) {
  if (lo == nullptr || off_lo == nullptr || off_hi == nullptr ||
      base == nullptr || x == nullptr || y == nullptr || n <= 0 || k <= 0 ||
      k % kPackBlock != 0) {
    return cudaErrorInvalidValue;
  }
  const int threads = 256;
  const int warps = threads / 32;
  const int blocks = (n + warps - 1) / warps;
  packed_gemv_bf16_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const uint8_t*>(lo), static_cast<const uint8_t*>(off_lo),
      static_cast<const uint8_t*>(off_hi), static_cast<const uint8_t*>(base),
      static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(y), n, k);
  return cudaGetLastError();
}
