// The packed GEMV reads half the bytes of the BF16 one and is slower at
// n=9216. Plane-by-plane ablation put it in the lo plane alone, so this takes
// everything else away: no x vector, no reconstruction, no reduction that
// matters -- just a warp per column walking one stream and summing it.
//
//   bf16_16B   float4 per lane, 2 bytes per weight  (what the BF16 GEMV reads)
//   byte_16B   uint4  per lane, 1 byte per weight   (what the packed one reads)
//   byte_8B    uint2  per lane, 1 byte per weight, half the request size
//
// Same column count, same warp geometry, same trip structure. One per process.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#define CK(x) do{cudaError_t r=(x); if(r){printf("cuda %s @%d\n",cudaGetErrorString(r),__LINE__);exit(1);} }while(0)

__global__ void bf16_16B(const __nv_bfloat16* __restrict__ w, float* __restrict__ y, int n, int k) {
  const int lane = threadIdx.x & 31;
  const int col = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (col >= n) return;
  const float4* r = reinterpret_cast<const float4*>(w + (size_t)col * k);
  float acc = 0.f;
  for (int i = lane; i < k / 8; i += 32) {
    float4 v = r[i];
    const __nv_bfloat16* l = reinterpret_cast<const __nv_bfloat16*>(&v);
#pragma unroll
    for (int j = 0; j < 8; ++j) acc += __bfloat162float(l[j]);
  }
  for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
  if (lane == 0 && acc == 1e30f) y[col] = acc;
}

__global__ void byte_16B(const uint8_t* __restrict__ w, float* __restrict__ y, int n, int k) {
  const int lane = threadIdx.x & 31;
  const int col = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (col >= n) return;
  const uint4* r = reinterpret_cast<const uint4*>(w + (size_t)col * k);
  float acc = 0.f;
  for (int i = lane; i < k / 16; i += 32) {
    uint4 v = r[i];
    const uint32_t word[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int q = 0; q < 4; ++q)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc += (float)((word[q] >> (j * 8)) & 0xFFu);
  }
  for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
  if (lane == 0 && acc == 1e30f) y[col] = acc;
}

__global__ void byte_8B(const uint8_t* __restrict__ w, float* __restrict__ y, int n, int k) {
  const int lane = threadIdx.x & 31;
  const int col = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (col >= n) return;
  const uint2* r = reinterpret_cast<const uint2*>(w + (size_t)col * k);
  float acc = 0.f;
  for (int i = lane; i < k / 8; i += 32) {
    uint2 v = r[i];
    const uint32_t word[2] = {v.x, v.y};
#pragma unroll
    for (int q = 0; q < 2; ++q)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc += (float)((word[q] >> (j * 8)) & 0xFFu);
  }
  for (int o = 16; o; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
  if (lane == 0 && acc == 1e30f) y[col] = acc;
}

int main(int argc, char** argv) {
  const int n = atoi(argv[1]), k = atoi(argv[2]), which = atoi(argv[3]);
  const size_t count = (size_t)n * k;
  void* w; float* y;
  const size_t bytes = which == 0 ? count * 2 : count;
  CK(cudaMalloc(&w, bytes)); CK(cudaMalloc(&y, (size_t)n * 4));
  CK(cudaMemset(w, 0x3c, bytes));
  const int threads = 256, warps = threads / 32, blocks = (n + warps - 1) / warps;
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); float ms; const int reps = 20;
  auto run = [&] {
    if (which == 0) bf16_16B<<<blocks, threads>>>((const __nv_bfloat16*)w, y, n, k);
    else if (which == 1) byte_16B<<<blocks, threads>>>((const uint8_t*)w, y, n, k);
    else byte_8B<<<blocks, threads>>>((const uint8_t*)w, y, n, k);
  };
  for (int i = 0; i < 3; ++i) run(); CK(cudaDeviceSynchronize());
  cudaEventRecord(a); for (int i = 0; i < reps; ++i) run(); cudaEventRecord(b);
  CK(cudaEventSynchronize(b)); cudaEventElapsedTime(&ms, a, b);
  const double us = ms * 1000.0 / reps;
  const char* nm[3] = {"bf16_16B", "byte_16B", "byte_8B "};
  printf("%s n=%-7d k=%-5d %8.1f us  %7.2f MB  %6.1f GB/s\n", nm[which], n, k, us,
         bytes / 1048576.0, bytes / (us * 1e-6) / 1e9);
  return 0;
}
