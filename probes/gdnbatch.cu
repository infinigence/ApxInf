// What would the GDN chunk-state scan cost if its inner products were a
// batched tensor-core GEMM instead of scalar fp32 FMAs?
//
// The kernel is 452.3 ms per scene, 38.2% of prefill and the largest single
// kernel in the run, and it is fp32 on CUDA cores where Thor is only 1.59x
// Orin. Its two inner loops are GEMM-shaped:
//   vp, ai   [64,128] x [128,128]   per head per chunk, twice
//   intra    [64,64]  x [64,128]    per head per chunk
// at 32 heads x 49 chunks x 24 layers. Before rewriting the kernel around
// WMMA, this asks cuBLAS what those batches cost at all -- if the shapes are
// too small to reach the tensor cores, the rewrite cannot pay.
//
// fp32 rows are what the kernel has today; bf16 rows are what a tensor-core
// form would run, and an exact-enough version needs two or three bf16 GEMMs
// per fp32 one (splitting the fp32 operand; the other side is already on the
// BF16 grid), so multiply the bf16 time by that before comparing.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#define CK(x) do{cudaError_t r=(x); if(r){printf("cuda %s @%d\n",cudaGetErrorString(r),__LINE__);exit(1);} }while(0)

static double bench(cublasHandle_t h, cudaDataType_t dt, int m, int n, int k,
                    int batch, int reps) {
  const size_t es = (dt == CUDA_R_32F) ? 4 : 2;
  void *A, *B, *C;
  CK(cudaMalloc(&A, (size_t)m * k * batch * es));
  CK(cudaMalloc(&B, (size_t)k * n * batch * es));
  CK(cudaMalloc(&C, (size_t)m * n * batch * es));
  CK(cudaMemset(A, 0x3c, (size_t)m * k * batch * es));
  CK(cudaMemset(B, 0x3c, (size_t)k * n * batch * es));
  float al = 1.f, be = 0.f;
  auto run = [&] {
    return cublasGemmStridedBatchedEx(
        h, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &al,
        B, dt, n, (long long)k * n, A, dt, k, (long long)m * k, &be,
        C, dt, n, (long long)m * n, batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  };
  double us = -1;
  if (run() == CUBLAS_STATUS_SUCCESS) {
    for (int i = 0; i < 3; ++i) run();
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); float ms;
    cudaEventRecord(a); for (int i = 0; i < reps; ++i) run(); cudaEventRecord(b);
    CK(cudaEventSynchronize(b)); cudaEventElapsedTime(&ms, a, b);
    us = ms * 1000.0 / reps;
    cudaEventDestroy(a); cudaEventDestroy(b);
  }
  cudaFree(A); cudaFree(B); cudaFree(C);
  return us;
}

int main() {
  cublasHandle_t h; cublasCreate(&h);
  const int heads = 32, chunks = 49, layers = 24;
  // The chunk loop is a recurrence -- the state leaving chunk c enters chunk
  // c+1 -- so chunks cannot be batched. The batch is the 32 heads, and the
  // launches are 49 chunks x 24 layers of it.
  const int batch = heads;
  struct S { const char* name; int m, n, k; int per_layer; } cases[] = {
    {"vp/ai  [64,128]x[128,128]", 64, 128, 128, 2},
    {"intra  [64,128]x[64,128] ", 64, 128,  64, 1},
  };
  printf("batch %d (heads), %d chunks x %d layers of sequential launches\n", batch, chunks, layers);
  printf("%-28s %10s %10s %12s %12s\n", "shape", "fp32 us", "bf16 us",
         "fp32 scene", "bf16 scene");
  double fp32_total = 0, bf16_total = 0;
  for (auto& c : cases) {
    const double f = bench(h, CUDA_R_32F, c.m, c.n, c.k, batch, 20);
    const double b = bench(h, CUDA_R_16BF, c.m, c.n, c.k, batch, 20);
    const double fs = f * c.per_layer * layers * chunks / 1000.0;
    const double bs = b * c.per_layer * layers * chunks / 1000.0;
    fp32_total += fs; bf16_total += bs;
    printf("%-28s %10.1f %10.1f %10.2f ms %10.2f ms\n", c.name, f, b, fs, bs);
  }
  printf("%-28s %10s %10s %10.2f ms %10.2f ms\n", "total for the scene", "", "",
         fp32_total, bf16_total);
  printf("hand-written kernel today: 452.30 ms\n");
  printf("bf16 x2 (near-fp32): %.2f ms   bf16 x3 (fp32-class): %.2f ms\n",
         bf16_total * 2, bf16_total * 3);
  return 0;
}
