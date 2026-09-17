// Is a bit-exact repacking of BF16 weights worth its decode cost in a batch-1
// GEMV?
//
// Decode on this model is 8.41 GB of weight traffic per token and the cuBLAS
// GEMVs already run at 96% of the measured 259.7 GB/s read roofline, so the
// only lever left is moving fewer bytes. BF16 weights of this checkpoint carry
// 2.58 bits of exponent entropy against 7.97 for sign+mantissa, and across
// every tensor sampled, 100.0000% of 128-weight blocks have an exponent range
// that fits in 5 bits. So each weight can be stored as
//
//   lo      1 byte   sign in bit 7, the 7 mantissa bits below it
//   off_lo  4 bits   low nibble of exponent - block base
//   off_hi  1 bit    its high bit, in a separate bit plane
//   base    1 byte per 128 weights
//
// = 13.0625 bits against 16, a 1.2249x cut, and the reconstruction
//   bf16 = (lo & 0x80) << 8 | (base + off) << 7 | (lo & 0x7F)
// returns the original 16 bits exactly. This measures whether the arithmetic
// to do that fits under the bytes it saves.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#define CK(x) do{cudaError_t r=(x); if(r){printf("cuda %s @%d\n",cudaGetErrorString(r),__LINE__);exit(1);} }while(0)

static const int BLK = 128;

// Baseline: one warp per output column, float4 loads, shuffle reduce. This is
// the kernel that measured 219 GB/s at this shape, against cuBLAS's 207.
__global__ void gemv_bf16(const __nv_bfloat16* __restrict__ w,
                          const __nv_bfloat16* __restrict__ x,
                          __nv_bfloat16* __restrict__ y, int n, int k) {
  const int lane = threadIdx.x & 31;
  const int col = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (col >= n) return;
  const float4* wr = reinterpret_cast<const float4*>(w + (size_t)col * k);
  const float4* xr = reinterpret_cast<const float4*>(x);
  float acc = 0.f;
  for (int i = lane; i < k / 8; i += 32) {
    float4 wv = wr[i], xv = xr[i];
    const __nv_bfloat16* wl = reinterpret_cast<const __nv_bfloat16*>(&wv);
    const __nv_bfloat16* xl = reinterpret_cast<const __nv_bfloat16*>(&xv);
#pragma unroll
    for (int j = 0; j < 8; ++j) acc += __bfloat162float(wl[j]) * __bfloat162float(xl[j]);
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
  if (lane == 0) y[col] = __float2bfloat16(acc);
}

// Packed: same warp-per-column shape, eight weights per lane per step.
//   lo      8 bytes  (one uint2)
//   off_lo  4 bytes  (one uint32, eight nibbles)
//   off_hi  1 byte   (eight bits)
// 13 bytes per eight weights against 16.
__global__ void gemv_packed(const uint8_t* __restrict__ lo,
                            const uint8_t* __restrict__ off_lo,
                            const uint8_t* __restrict__ off_hi,
                            const uint8_t* __restrict__ base,
                            const __nv_bfloat16* __restrict__ x,
                            __nv_bfloat16* __restrict__ y, int n, int k) {
  const int lane = threadIdx.x & 31;
  const int col = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (col >= n) return;
  const uint2* lo_r = reinterpret_cast<const uint2*>(lo + (size_t)col * k);
  const uint32_t* ol_r = reinterpret_cast<const uint32_t*>(off_lo + (size_t)col * (k / 2));
  const uint8_t* oh_r = off_hi + (size_t)col * (k / 8);
  const uint8_t* bs_r = base + (size_t)col * (k / BLK);
  const float4* xr = reinterpret_cast<const float4*>(x);
  float acc = 0.f;
  for (int i = lane; i < k / 8; i += 32) {
    const uint2 lov = lo_r[i];
    const uint32_t olv = ol_r[i];
    const uint32_t ohv = oh_r[i];
    const uint32_t b = bs_r[i / (BLK / 8)];
    const float4 xv = xr[i];
    const __nv_bfloat16* xl = reinterpret_cast<const __nv_bfloat16*>(&xv);
    // Reconstruct straight into fp32: a BF16 value is its own fp32 bit pattern
    // shifted left by 16, so the BF16 type never has to appear.
    //   f32 = sign<<31 | (base+off)<<23 | mantissa<<16
    // Eight weights, unrolled with constant shifts so nothing is indexed and
    // nothing spills.
#define PACK_ONE(WORD, J, BIT)                                                 \
    {                                                                          \
      const uint32_t lb = ((WORD) >> ((J) * 8)) & 0xFFu;                        \
      const uint32_t off = ((olv >> ((BIT) * 4)) & 0xFu) |                      \
                           (((ohv >> (BIT)) & 1u) << 4);                        \
      const float w = __int_as_float(((lb & 0x80u) << 24) |                     \
                                     ((b + off) << 23) | ((lb & 0x7Fu) << 16)); \
      acc = fmaf(w, __bfloat162float(xl[BIT]), acc);                            \
    }
    PACK_ONE(lov.x, 0, 0) PACK_ONE(lov.x, 1, 1) PACK_ONE(lov.x, 2, 2) PACK_ONE(lov.x, 3, 3)
    PACK_ONE(lov.y, 0, 4) PACK_ONE(lov.y, 1, 5) PACK_ONE(lov.y, 2, 6) PACK_ONE(lov.y, 3, 7)
#undef PACK_ONE
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
  if (lane == 0) y[col] = __float2bfloat16(acc);
}

int main(int argc, char** argv) {
  const int n = argc > 1 ? atoi(argv[1]) : 18432;
  const int k = argc > 2 ? atoi(argv[2]) : 2560;
  const size_t count = (size_t)n * k;

  // Real weights. A synthetic draw is not good enough here: the whole question
  // is whether a 128-weight block's exponents span 5 bits, and that is a
  // property of the trained tensor, not of any distribution one writes down.
  std::vector<uint16_t> host(count);
  {
    const char* path = argc > 3 ? argv[3] : "/tmp/realw/gate_up.bin";
    FILE* fh = fopen(path, "rb");
    if (!fh) { printf("cannot open %s\n", path); return 1; }
    if (fread(host.data(), 2, count, fh) != count) {
      printf("%s is shorter than %zu weights\n", path, count); return 1;
    }
    fclose(fh);
    printf("weights from %s\n", path);
  }
  // Pack, and verify the reconstruction is the identity before timing it.
  std::vector<uint8_t> lo(count), ol(count / 2), oh(count / 8), bs(count / BLK);
  size_t misfit = 0;
  for (size_t blk = 0; blk < count / BLK; ++blk) {
    int mn = 255, mx = 0;
    for (int t = 0; t < BLK; ++t) { int e = (host[blk * BLK + t] >> 7) & 0xFF; mn = mn < e ? mn : e; mx = mx > e ? mx : e; }
    if (mx - mn > 31) { ++misfit; }
    bs[blk] = (uint8_t)mn;
    for (int t = 0; t < BLK; ++t) {
      const uint16_t h = host[blk * BLK + t];
      const size_t i = blk * BLK + t;
      lo[i] = (uint8_t)(((h >> 8) & 0x80) | (h & 0x7F));
      const uint32_t off = (uint32_t)(((h >> 7) & 0xFF) - mn);
      if (t & 1) ol[i / 2] |= (off & 0xF) << 4; else ol[i / 2] = (off & 0xF);
      if ((t & 7) == 0) oh[i / 8] = 0;
      oh[i / 8] |= ((off >> 4) & 1) << (t & 7);
    }
  }
  size_t bad = 0;
  for (size_t i = 0; i < count; ++i) {
    const uint32_t off = ((ol[i / 2] >> ((i & 1) * 4)) & 0xF) | (((oh[i / 8] >> (i & 7)) & 1) << 4);
    const uint32_t bits = ((lo[i] & 0x80u) << 8) | ((bs[i / BLK] + off) << 7) | (lo[i] & 0x7Fu);
    if ((uint16_t)bits != host[i]) ++bad;
  }
  printf("pack: %zu weights, %zu blocks over 5-bit range, %zu reconstruction mismatches\n",
         count, misfit, bad);
  if (bad) return 1;

  __nv_bfloat16 *dw, *dx, *dy;
  uint8_t *dlo, *dol, *doh, *dbs;
  CK(cudaMalloc(&dw, count * 2)); CK(cudaMalloc(&dx, (size_t)k * 2)); CK(cudaMalloc(&dy, (size_t)n * 2));
  CK(cudaMalloc(&dlo, count)); CK(cudaMalloc(&dol, count / 2));
  CK(cudaMalloc(&doh, count / 8)); CK(cudaMalloc(&dbs, count / BLK));
  CK(cudaMemcpy(dw, host.data(), count * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dlo, lo.data(), count, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dol, ol.data(), count / 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(doh, oh.data(), count / 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dbs, bs.data(), count / BLK, cudaMemcpyHostToDevice));
  CK(cudaMemset(dx, 0x3c, (size_t)k * 2));

  const int threads = 256, warps = threads / 32, blocks = (n + warps - 1) / warps;
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  const int reps = 20; float ms;
  auto run_plain  = [&]{ gemv_bf16<<<blocks, threads>>>(dw, dx, dy, n, k); };
  auto run_packed = [&]{ gemv_packed<<<blocks, threads>>>(dlo, dol, doh, dbs, dx, dy, n, k); };

  // Same y from both, before timing either.
  std::vector<uint16_t> y0(n), y1(n);
  run_plain();  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(y0.data(), dy, (size_t)n * 2, cudaMemcpyDeviceToHost));
  run_packed(); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(y1.data(), dy, (size_t)n * 2, cudaMemcpyDeviceToHost));
  size_t ydiff = 0; for (int i = 0; i < n; ++i) if (y0[i] != y1[i]) ++ydiff;
  printf("output: %zu of %d columns differ\n", ydiff, n);

  for (int i = 0; i < 3; ++i) run_plain(); CK(cudaDeviceSynchronize());
  cudaEventRecord(a); for (int i = 0; i < reps; ++i) run_plain(); cudaEventRecord(b);
  CK(cudaEventSynchronize(b)); cudaEventElapsedTime(&ms, a, b);
  const double plain_us = ms * 1000.0 / reps;
  for (int i = 0; i < 3; ++i) run_packed(); CK(cudaDeviceSynchronize());
  cudaEventRecord(a); for (int i = 0; i < reps; ++i) run_packed(); cudaEventRecord(b);
  CK(cudaEventSynchronize(b)); cudaEventElapsedTime(&ms, a, b);
  const double packed_us = ms * 1000.0 / reps;

  const double plain_mb = count * 2 / 1048576.0;
  const double packed_mb = (count + count / 2.0 + count / 8.0 + count / (double)BLK) / 1048576.0;
  printf("shape n=%d k=%d\n", n, k);
  printf("  plain  %9.1f us  %7.1f MB  %6.1f GB/s\n", plain_us, plain_mb, plain_mb * 1048576 / (plain_us * 1e-6) / 1e9);
  printf("  packed %9.1f us  %7.1f MB  %6.1f GB/s   bytes %.4fx, time %.4fx\n",
         packed_us, packed_mb, packed_mb * 1048576 / (packed_us * 1e-6) / 1e9,
         plain_mb / packed_mb, plain_us / packed_us);
  return 0;
}
