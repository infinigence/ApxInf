// The packed GEMV is 2.7x slower than the BF16 one at n=9216, k=2560, and the
// byte stream alone is not the reason: reading 22.5 MB of lo bytes takes
// 137.7 us where reading 45 MB of BF16 takes 184.8. So add the rest back one
// step at a time, one variant per process, and see which one costs.
//
//   0 stream    read lo, sum the bytes
//   1 +x        also read the x vector and sum it
//   2 +rebuild  reconstruct the fp32 weight from lo and a constant exponent
//   3 +fma      multiply by x
//   4 +planes   read off_lo, off_hi and base for the real exponent
//   5 bf16      the BF16 GEMV, for reference
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#define CK(x) do{cudaError_t r=(x); if(r){printf("cuda %s @%d\n",cudaGetErrorString(r),__LINE__);exit(1);} }while(0)
static const int BLK = 128;

template <int STEP>
__global__ void step_kernel(const uint8_t* __restrict__ lo,
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
    const uint32_t olv = STEP >= 4 ? ol_r[i] : 0x76543210u;
    const uint32_t ohv = STEP >= 4 ? (uint32_t)oh_r[i] : 0u;
    const uint32_t b = STEP >= 4 ? (uint32_t)bs_r[i / (BLK / 8)] : 120u;
    float4 xv;
    if (STEP >= 1) xv = xr[i];
    const __nv_bfloat16* xl = reinterpret_cast<const __nv_bfloat16*>(&xv);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const uint32_t word = (j < 4) ? lov.x : lov.y;
      const uint32_t lb = (word >> ((j & 3) * 8)) & 0xFFu;
      if (STEP == 0) { acc += (float)lb; continue; }
      if (STEP == 1) { acc += (float)lb + __bfloat162float(xl[j]); continue; }
      const uint32_t off = ((olv >> (j * 4)) & 0xFu) | (((ohv >> j) & 1u) << 4);
      const float w = __int_as_float(((lb & 0x80u) << 24) | ((b + off) << 23) | ((lb & 0x7Fu) << 16));
      if (STEP == 2) { acc += w; continue; }
      acc = fmaf(w, __bfloat162float(xl[j]), acc);
    }
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
  if (lane == 0) y[col] = __float2bfloat16(acc);
}

__global__ void bf16_ref(const __nv_bfloat16* __restrict__ w,
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
  const int n = atoi(argv[1]), k = atoi(argv[2]), step = atoi(argv[3]);
  const size_t count = (size_t)n * k;
  uint8_t *lo, *ol, *oh, *bs; __nv_bfloat16 *w, *x, *y;
  CK(cudaMalloc(&lo, count)); CK(cudaMalloc(&ol, count/2));
  CK(cudaMalloc(&oh, count/8)); CK(cudaMalloc(&bs, count/BLK));
  CK(cudaMalloc(&w, count*2)); CK(cudaMalloc(&x, (size_t)k*2)); CK(cudaMalloc(&y, (size_t)n*2));
  CK(cudaMemset(lo, 0x21, count)); CK(cudaMemset(ol, 0x21, count/2));
  CK(cudaMemset(oh, 0, count/8)); CK(cudaMemset(bs, 120, count/BLK));
  CK(cudaMemset(w, 0x3c, count*2)); CK(cudaMemset(x, 0x3c, (size_t)k*2));
  const int threads = 256, warps = threads/32, blocks = (n+warps-1)/warps;
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); float ms; const int reps = 20;
  auto run = [&] {
    switch (step) {
      case 0: step_kernel<0><<<blocks,threads>>>(lo,ol,oh,bs,x,y,n,k); break;
      case 1: step_kernel<1><<<blocks,threads>>>(lo,ol,oh,bs,x,y,n,k); break;
      case 2: step_kernel<2><<<blocks,threads>>>(lo,ol,oh,bs,x,y,n,k); break;
      case 3: step_kernel<3><<<blocks,threads>>>(lo,ol,oh,bs,x,y,n,k); break;
      case 4: step_kernel<4><<<blocks,threads>>>(lo,ol,oh,bs,x,y,n,k); break;
      default: bf16_ref<<<blocks,threads>>>(w,x,y,n,k); break;
    }
  };
  for (int i = 0; i < 3; ++i) run(); CK(cudaDeviceSynchronize());
  cudaEventRecord(a); for (int i = 0; i < reps; ++i) run(); cudaEventRecord(b);
  CK(cudaEventSynchronize(b)); cudaEventElapsedTime(&ms, a, b);
  const char* nm[6] = {"0 stream  ","1 +x      ","2 +rebuild","3 +fma    ","4 +planes ","5 bf16 ref"};
  printf("%s n=%-7d k=%-5d %8.1f us\n", nm[step<6?step:5], n, k, ms*1000.0/reps);
  return 0;
}
