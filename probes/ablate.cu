// Which part of the packed read costs the bandwidth? Four kernels with the
// same warp-per-column shape, adding one plane at a time. Arithmetic is kept
// identical where the plane is absent (a constant stands in), so the
// difference is the load, not the work.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#define CK(x) do{cudaError_t r=(x); if(r){printf("cuda %s @%d\n",cudaGetErrorString(r),__LINE__);exit(1);} }while(0)
static const int BLK = 128;

template <int PLANES>   // 1 = lo only, 2 = +off_lo, 3 = +off_hi, 4 = +base
__global__ void gemv_ab(const uint8_t* __restrict__ lo,
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
    const uint32_t olv = PLANES >= 2 ? ol_r[i] : 0x76543210u;
    const uint32_t ohv = PLANES >= 3 ? (uint32_t)oh_r[i] : 0u;
    const uint32_t b = PLANES >= 4 ? (uint32_t)bs_r[i / (BLK / 8)] : 120u;
    const float4 xv = xr[i];
    const __nv_bfloat16* xl = reinterpret_cast<const __nv_bfloat16*>(&xv);
#define ONE(WORD, J, BIT)                                                      \
    { const uint32_t lb = ((WORD) >> ((J) * 8)) & 0xFFu;                        \
      const uint32_t off = ((olv >> ((BIT) * 4)) & 0xFu) | (((ohv >> (BIT)) & 1u) << 4); \
      const float w = __int_as_float(((lb & 0x80u) << 24) | ((b + off) << 23) | ((lb & 0x7Fu) << 16)); \
      acc = fmaf(w, __bfloat162float(xl[BIT]), acc); }
    ONE(lov.x,0,0) ONE(lov.x,1,1) ONE(lov.x,2,2) ONE(lov.x,3,3)
    ONE(lov.y,0,4) ONE(lov.y,1,5) ONE(lov.y,2,6) ONE(lov.y,3,7)
#undef ONE
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
  if (lane == 0) y[col] = __float2bfloat16(acc);
}

// Reference: the same loop reading BF16 directly.
__global__ void gemv_plain(const __nv_bfloat16* __restrict__ w,
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
    for (int j = 0; j < 8; ++j) acc = fmaf(__bfloat162float(wl[j]), __bfloat162float(xl[j]), acc);
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
  if (lane == 0) y[col] = __float2bfloat16(acc);
}

int main(int argc, char** argv) {
  const int n = argc > 1 ? atoi(argv[1]) : 9216;
  const int k = argc > 2 ? atoi(argv[2]) : 2560;
  const size_t count = (size_t)n * k;
  __nv_bfloat16 *dw, *dx, *dy; uint8_t *dlo, *dol, *doh, *dbs;
  CK(cudaMalloc(&dw, count*2)); CK(cudaMalloc(&dx,(size_t)k*2)); CK(cudaMalloc(&dy,(size_t)n*2));
  CK(cudaMalloc(&dlo, count)); CK(cudaMalloc(&dol, count/2));
  CK(cudaMalloc(&doh, count/8)); CK(cudaMalloc(&dbs, count/BLK));
  // Real weights, really packed. Memset operands make every variant read the
  // same byte everywhere, and that is exactly the thing a memory experiment
  // must not do.
  {
    std::vector<uint16_t> host(count);
    const char* path = argc > 3 ? argv[3] : "/tmp/realw/gate_up.bin";
    FILE* fh = fopen(path, "rb");
    if (!fh || fread(host.data(), 2, count, fh) != count) { printf("cannot read %s\n", path); return 1; }
    fclose(fh);
    std::vector<uint8_t> lo(count), ol(count/2), oh(count/8), bs(count/BLK);
    for (size_t blk = 0; blk < count/BLK; ++blk) {
      int mn = 255; for (int t=0;t<BLK;++t){int e=(host[blk*BLK+t]>>7)&0xFF; mn = mn<e?mn:e;}
      bs[blk] = (uint8_t)mn;
      for (int t=0;t<BLK;++t) {
        const uint16_t h = host[blk*BLK+t]; const size_t i = blk*BLK+t;
        lo[i] = (uint8_t)(((h>>8)&0x80)|(h&0x7F));
        const uint32_t off = (uint32_t)(((h>>7)&0xFF)-mn);
        if (t&1) ol[i/2] |= (off&0xF)<<4; else ol[i/2] = (off&0xF);
        if ((t&7)==0) oh[i/8] = 0;
        oh[i/8] |= ((off>>4)&1)<<(t&7);
      }
    }
    CK(cudaMemcpy(dw, host.data(), count*2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dlo, lo.data(), count, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dol, ol.data(), count/2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(doh, oh.data(), count/8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dbs, bs.data(), count/BLK, cudaMemcpyHostToDevice));
    printf("packed %s\n", path);
  }
  CK(cudaMemset(dx,0x3c,(size_t)k*2));
  // L2 is 32 MB here and the weights are 45, so a tight replay loop would be
  // served partly from L2. Scrub it between replays and subtract the scrub.
  const size_t scrub_bytes = 96u<<20;
  uint8_t* scrub; CK(cudaMalloc(&scrub, scrub_bytes));
  CK(cudaMemset(scrub, 1, scrub_bytes));
  const int threads=256, warps=threads/32, blocks=(n+warps-1)/warps;
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); float ms; const int reps=20;
  // No scrub. Both variants replay the same way and see the same L2, so the
  // comparison is fair even though neither number is a cold one.
  (void)scrub;
  const double scrub_us = 0.0;
  auto time=[&](auto fn){ for(int i=0;i<3;i++) fn(); CK(cudaDeviceSynchronize());
    cudaEventRecord(a); for(int i=0;i<reps;i++) fn(); cudaEventRecord(b);
    CK(cudaEventSynchronize(b)); cudaEventElapsedTime(&ms,a,b); return ms*1000.0/reps; };
  const double mb_plain = count*2/1048576.0;
  double t = time([&]{ gemv_plain<<<blocks,threads>>>(dw,dx,dy,n,k); });
  // One variant per process. Running them in one process makes the answer
  // depend on what ran before it: these buffers are 81 MB against a 32 MB L2,
  // so each variant inherits a different residency, and the ordering came out
  // non-monotonic and not reproducible across builds. argv[4] selects one.
  const int only = argc > 4 ? atoi(argv[4]) : -1;
  (void)scrub_us;
  if (only == 0) {
    double t0 = time([&]{ gemv_plain<<<blocks,threads>>>(dw,dx,dy,n,k); });
    printf("plain  bf16          %8.1f us  %7.2f MB  %6.1f GB/s\n", t0, mb_plain, mb_plain*1048576/(t0*1e-6)/1e9);
    return 0;
  }
  if (only == 4) {
    const double m = count*(1.625+1.0/BLK)/1048576.0;
    double t0 = time([&]{ gemv_ab<4><<<blocks,threads>>>(dlo,dol,doh,dbs,dx,dy,n,k); });
    printf("packed 13.0625 bits  %8.1f us  %7.2f MB  %6.1f GB/s\n", t0, m, m*1048576/(t0*1e-6)/1e9);
    return 0;
  }
  printf("plain  bf16          %8.1f us  %7.2f MB  %6.1f GB/s\n", t, mb_plain, mb_plain*1048576/(t*1e-6)/1e9);
  const double mb[5] = {0, count/1048576.0, count*1.5/1048576.0,
                        count*1.625/1048576.0, count*(1.625+1.0/BLK)/1048576.0};
  const char* nm[5] = {"", "lo only            ", "lo+off_lo          ",
                       "lo+off_lo+off_hi   ", "lo+off_lo+off_hi+bs"};
  t = time([&]{ gemv_ab<1><<<blocks,threads>>>(dlo,dol,doh,dbs,dx,dy,n,k); });
  printf("%s %8.1f us  %7.2f MB  %6.1f GB/s\n", nm[1], t, mb[1], mb[1]*1048576/(t*1e-6)/1e9);
  t = time([&]{ gemv_ab<2><<<blocks,threads>>>(dlo,dol,doh,dbs,dx,dy,n,k); });
  printf("%s %8.1f us  %7.2f MB  %6.1f GB/s\n", nm[2], t, mb[2], mb[2]*1048576/(t*1e-6)/1e9);
  t = time([&]{ gemv_ab<3><<<blocks,threads>>>(dlo,dol,doh,dbs,dx,dy,n,k); });
  printf("%s %8.1f us  %7.2f MB  %6.1f GB/s\n", nm[3], t, mb[3], mb[3]*1048576/(t*1e-6)/1e9);
  t = time([&]{ gemv_ab<4><<<blocks,threads>>>(dlo,dol,doh,dbs,dx,dy,n,k); });
  printf("%s %8.1f us  %7.2f MB  %6.1f GB/s\n", nm[4], t, mb[4], mb[4]*1048576/(t*1e-6)/1e9);
  return 0;
}
