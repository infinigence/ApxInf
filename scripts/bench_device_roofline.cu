// Measured roofline for a CUDA device: achievable read/copy bandwidth, BF16
// tensor-core GEMM throughput, and FP32 FMA throughput against instruction-level
// parallelism.
//
// A spec sheet is not a roofline. On LPDDR5 parts the quoted pin rate and what a
// kernel actually reaches differ by a third, which is enough to change whether a
// workload reads as "at the roof" or "26% short of it". The FP32 sweep is here
// because kernels built out of recurrences carry few independent accumulators,
// and it is worth knowing from measurement rather than assumption whether a
// part penalises that.
//
//   nvcc -O3 -arch=native -o bench_device_roofline scripts/bench_device_roofline.cu -lcublas
//   ./bench_device_roofline
//
// On Drive OS layouts cuBLAS lives beside the toolkit rather than in it:
//   T=/usr/local/cuda/thor/targets/aarch64-linux
//   nvcc -O3 -arch=native -I$T/include -L$T/lib -Xlinker -rpath -Xlinker $T/lib ...
//
// Note for Tegra: cudaDevAttrMemoryClockRate does not report the LPDDR rate, so
// a "theoretical" bandwidth computed from it is wrong there. Use the measured
// figures.
//
// Every section warms the GPU before timing, and that is not a formality. On a
// cold process a Jetson starts at a low DVFS point and takes on the order of a
// second to reach its ceiling, which is long enough to cover several timed
// kernels. Measuring the FP32 sweep cold produces a clean-looking but entirely
// false result -- 1.18 TFLOPS at four chains rising to 5.15 at thirty-two,
// which reads as a dependency cliff and is really just the clock ramping. The
// same sweep after a warm-up is flat at 4.8 to 5.2 across every chain count.
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#define CK(x)                                                            \
  do {                                                                   \
    cudaError_t e = (x);                                                 \
    if (e != cudaSuccess) {                                              \
      printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); \
      return 1;                                                          \
    }                                                                    \
  } while (0)

// Weight streaming during decode is read-only, so the read figure is the one
// that bounds it; the copy figure is here for kernels that also write.
__global__ void read_kernel(const float4* __restrict__ src, size_t n4, float* sink) {
  float acc = 0.0f;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4;
       i += (size_t)gridDim.x * blockDim.x) {
    float4 v = src[i];
    acc += v.x + v.y + v.z + v.w;
  }
  if (acc == 1.2345e-30f) *sink = acc;  // never true; keeps the loads live
}

__global__ void copy_kernel(const float4* __restrict__ src, float4* __restrict__ dst,
                            size_t n4) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4;
       i += (size_t)gridDim.x * blockDim.x) {
    dst[i] = src[i];
  }
}

// CHAINS independent accumulators per thread. Sweeping it separates the rate a
// dependent chain reaches from the rate the cores can actually sustain.
template <int CHAINS>
__global__ void fma_kernel(float* out, float a, float b, int iters) {
  float x[CHAINS];
#pragma unroll
  for (int i = 0; i < CHAINS; ++i) x[i] = a + i;
  for (int t = 0; t < iters; ++t) {
#pragma unroll
    for (int i = 0; i < CHAINS; ++i) x[i] = fmaf(x[i], a, b);
  }
  float s = 0.0f;
#pragma unroll
  for (int i = 0; i < CHAINS; ++i) s += x[i];
  if (s == 1.2345e-30f) *out = s;
}

template <int CHAINS>
static void fma_rate(float* sink, int sms, cudaEvent_t t0, cudaEvent_t t1) {
  const int blocks = sms * 6, threads = 256, iters = 4096, reps = 10;
  fma_kernel<CHAINS><<<blocks, threads>>>(sink, 1.0001f, 0.5f, iters);
  cudaDeviceSynchronize();
  cudaEventRecord(t0);
  for (int r = 0; r < reps; ++r)
    fma_kernel<CHAINS><<<blocks, threads>>>(sink, 1.0001f, 0.5f, iters);
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, t0, t1);
  const double flops = (double)reps * blocks * threads * iters * CHAINS * 2;
  printf("  %2d independent chains : %7.2f TFLOPS\n", CHAINS, flops / (ms / 1e3) / 1e12);
}

int main() {
  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, 0));
  int clock_khz = 0;
  cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0);
  printf("device          : %s  sm_%d%d  %d SMs @ %.0f MHz\n", prop.name, prop.major,
         prop.minor, prop.multiProcessorCount, clock_khz / 1000.0);
  printf("shared per SM   : %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
  printf("L2              : %d KB\n", prop.l2CacheSize / 1024);

  const size_t bytes = 2ull << 30;  // 2 GB, far past any L2
  const size_t n4 = bytes / sizeof(float4);
  float4 *a = nullptr, *b = nullptr;
  float* sink = nullptr;
  CK(cudaMalloc(&a, bytes));
  CK(cudaMalloc(&b, bytes));
  CK(cudaMalloc(&sink, sizeof(float)));
  CK(cudaMemset(a, 1, bytes));

  const int blocks = prop.multiProcessorCount * 32;
  cudaEvent_t t0, t1;
  CK(cudaEventCreate(&t0));
  CK(cudaEventCreate(&t1));
  float ms = 0.0f;

  for (int i = 0; i < 3; ++i) read_kernel<<<blocks, 256>>>(a, n4, sink);
  CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(t0));
  for (int i = 0; i < 10; ++i) read_kernel<<<blocks, 256>>>(a, n4, sink);
  CK(cudaEventRecord(t1));
  CK(cudaEventSynchronize(t1));
  CK(cudaEventElapsedTime(&ms, t0, t1));
  printf("\nread  (2GB)     : %7.1f GB/s\n", 10.0 * bytes / (ms / 1e3) / 1e9);

  for (int i = 0; i < 3; ++i) copy_kernel<<<blocks, 256>>>(a, b, n4);
  CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(t0));
  for (int i = 0; i < 10; ++i) copy_kernel<<<blocks, 256>>>(a, b, n4);
  CK(cudaEventRecord(t1));
  CK(cudaEventSynchronize(t1));
  CK(cudaEventElapsedTime(&ms, t0, t1));
  printf("copy  (r+w 2GB) : %7.1f GB/s\n", 10.0 * 2.0 * bytes / (ms / 1e3) / 1e9);
  CK(cudaFree(b));

  printf("\nFP32 FMA against instruction-level parallelism\n");
  // The bandwidth section above has already held the GPU busy for long enough
  // to reach its clock ceiling. Keep this ordering, or warm it explicitly:
  // measuring this sweep cold invents a dependency cliff that is not there.
  for (int i = 0; i < 40; ++i)
    fma_kernel<32><<<prop.multiProcessorCount * 6, 256>>>(sink, 1.0001f, 0.5f, 4096);
  CK(cudaDeviceSynchronize());
  fma_rate<4>(sink, prop.multiProcessorCount, t0, t1);
  fma_rate<8>(sink, prop.multiProcessorCount, t0, t1);
  fma_rate<16>(sink, prop.multiProcessorCount, t0, t1);
  fma_rate<32>(sink, prop.multiProcessorCount, t0, t1);

  cublasHandle_t h;
  cublasCreate(&h);
  cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH);
  printf("\nBF16 GEMM (cuBLAS, FP32 accumulate)\n");
  const int sizes[] = {2048, 4096, 8192};
  for (int si = 0; si < 3; ++si) {
    const int n = sizes[si];
    __nv_bfloat16 *A, *B, *C;
    const size_t sz = (size_t)n * n * sizeof(__nv_bfloat16);
    if (cudaMalloc(&A, sz) != cudaSuccess) continue;
    if (cudaMalloc(&B, sz) != cudaSuccess) {
      cudaFree(A);
      continue;
    }
    if (cudaMalloc(&C, sz) != cudaSuccess) {
      cudaFree(A);
      cudaFree(B);
      continue;
    }
    cudaMemset(A, 0x3c, sz);
    cudaMemset(B, 0x3c, sz);
    const float alpha = 1.0f, beta = 0.0f;
    const int reps = 10;
    for (int i = 0; i < 3; ++i)
      cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, CUDA_R_16BF, n, B,
                   CUDA_R_16BF, n, &beta, C, CUDA_R_16BF, n, CUBLAS_COMPUTE_32F,
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(t0));
    for (int i = 0; i < reps; ++i)
      cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, CUDA_R_16BF, n, B,
                   CUDA_R_16BF, n, &beta, C, CUDA_R_16BF, n, CUBLAS_COMPUTE_32F,
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    CK(cudaEventRecord(t1));
    CK(cudaEventSynchronize(t1));
    CK(cudaEventElapsedTime(&ms, t0, t1));
    printf("  %5d^3        : %7.2f TFLOPS   (%6.2f ms/GEMM)\n", n,
           reps * 2.0 * (double)n * n * n / (ms / 1e3) / 1e12, ms / reps);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
  }
  cublasDestroy(h);
  CK(cudaFree(a));
  CK(cudaFree(sink));
  return 0;
}
