// CUDA matrix-product diagnostic used by cuda_product_precision_reference.py.
// Host transfers and CPU model operations make this unsuitable for timing or
// runtime acceptance. The production runtime does not link this library.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <climits>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string last_error;
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("cuBLAS status " + std::to_string(int(status)));
}
struct Buffer {
  void* pointer = nullptr;
  size_t capacity = 0;
  ~Buffer() { if (pointer) cudaFree(pointer); }
  void reserve(size_t bytes) {
    if (bytes <= capacity) return;
    // Allocate first so a failed growth does not leave a dangling pointer.
    void* next = nullptr;
    check(cudaMalloc(&next, bytes));
    if (pointer) cudaFree(pointer);
    pointer = next;
    capacity = bytes;
  }
  template<class T> T* as() { return static_cast<T*>(pointer); }
};
struct Context {
  cudaStream_t stream = nullptr;
  cublasHandle_t blas = nullptr;
  Buffer a, b, c, a_parts, b_parts;
  Context() {
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    try {
      check(cublasCreate(&blas));
      check(cublasSetStream(blas, stream));
      check(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH));
    } catch (...) {
      if (blas) cublasDestroy(blas);
      cudaStreamDestroy(stream);
      throw;
    }
  }
  ~Context() {
    cudaStreamSynchronize(stream);
    cublasDestroy(blas);
    cudaStreamDestroy(stream);
  }
};
__global__ void split_scaled(const float* input, half* output, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    const half high = __float2half_rn(input[i]);
    output[i] = high;
    output[count + i] = __float2half_rn((input[i] - __half2float(high)) * 4096.f);
  }
}
int checked_count(int rows, int cols) {
  if (rows <= 0 || cols <= 0 || size_t(rows) * cols > INT_MAX / 2)
    throw std::runtime_error("matrix dimensions exceed diagnostic bounds");
  return rows * cols;
}
}  // namespace

extern "C" const char* qwen_product_error() { return last_error.c_str(); }
extern "C" void* qwen_product_create() {
  try { last_error.clear(); return new Context; }
  catch (const std::exception& error) { last_error = error.what(); return nullptr; }
}
extern "C" void qwen_product_destroy(void* context) {
  delete static_cast<Context*>(context);
}
// Row-major A[M,K], B[K,N], C[M,N]. mode=0: FP32 pedantic;
// mode=3/4: that many scaled-FP16 component products, FP32 accumulation.
extern "C" int qwen_product_run(void* opaque, const float* a, const float* b,
                                float* c, int m, int n, int k, int mode) {
  try {
    last_error.clear();
    if (!opaque || !a || !b || !c || (mode != 0 && mode != 3 && mode != 4))
      throw std::runtime_error("invalid CUDA product arguments");
    const int ac = checked_count(m, k), bc = checked_count(k, n);
    const int cc = checked_count(m, n);
    auto& ctx = *static_cast<Context*>(opaque);
    ctx.a.reserve(size_t(ac) * sizeof(float));
    ctx.b.reserve(size_t(bc) * sizeof(float));
    ctx.c.reserve(size_t(cc) * sizeof(float));
    check(cudaMemcpyAsync(ctx.a.pointer, a, size_t(ac) * sizeof(float), cudaMemcpyHostToDevice, ctx.stream));
    check(cudaMemcpyAsync(ctx.b.pointer, b, size_t(bc) * sizeof(float), cudaMemcpyHostToDevice, ctx.stream));
    if (mode == 0) {
      const float one = 1, zero = 0;
      check(cublasGemmEx(ctx.blas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                        &one, ctx.b.pointer, CUDA_R_32F, n,
                        ctx.a.pointer, CUDA_R_32F, k, &zero, ctx.c.pointer,
                        CUDA_R_32F, n, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
    } else {
      ctx.a_parts.reserve(size_t(ac) * 2 * sizeof(half));
      ctx.b_parts.reserve(size_t(bc) * 2 * sizeof(half));
      split_scaled<<<(ac + 255) / 256, 256, 0, ctx.stream>>>(ctx.a.as<float>(), ctx.a_parts.as<half>(), ac);
      split_scaled<<<(bc + 255) / 256, 256, 0, ctx.stream>>>(ctx.b.as<float>(), ctx.b_parts.as<half>(), bc);
      check(cudaGetLastError());
      auto product = [&](int ai, int bi, float alpha, float beta) {
        check(cublasGemmEx(ctx.blas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                          &alpha, ctx.b_parts.as<half>() + bi * bc, CUDA_R_16F, n,
                          ctx.a_parts.as<half>() + ai * ac, CUDA_R_16F, k,
                          &beta, ctx.c.pointer, CUDA_R_32F, n,
                          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      };
      product(0, 0, 1.f, 0.f);
      product(0, 1, 1.f / 4096.f, 1.f);
      product(1, 0, 1.f / 4096.f, 1.f);
      if (mode == 4) product(1, 1, 1.f / (4096.f * 4096.f), 1.f);
    }
    check(cudaMemcpyAsync(c, ctx.c.pointer, size_t(cc) * sizeof(float), cudaMemcpyDeviceToHost, ctx.stream));
    check(cudaStreamSynchronize(ctx.stream));
    return 0;
  } catch (const std::exception& error) {
    last_error = error.what();
    // Finish any already queued host transfers before Python releases arrays.
    if (opaque) cudaStreamSynchronize(static_cast<Context*>(opaque)->stream);
    return 1;
  }
}
