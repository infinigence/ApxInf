// Copyright 2026 apxinf contributors.
// Standalone CUDA regression: caller-owned scratch reuse, including max -> sum.
// Run from the repository root (use the target GPU's architecture):
// mkdir -p devlocal/block-sum-unsafe/bin
// nvcc -O3 -std=c++17 -arch=sm_110 crates/apxinf-cuda/tests/reduction_scratch.cu \
//   -o devlocal/block-sum-unsafe/bin/reduction_scratch
// compute-sanitizer --tool racecheck --error-exitcode 1 \
//   devlocal/block-sum-unsafe/bin/reduction_scratch
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../kernels/custom/math.cuh"
#include "../kernels/custom/reduction.cuh"
#include "../kernels/custom/attention.cuh"
#include "../kernels/custom/normalization.cuh"

#define CHECK(call) do { auto e = (call); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); \
  std::exit(1); } } while (0)

// Exercise sum -> sum, sum -> max, max -> max, max -> sum, and loop reuse.
// Exactly representable positive integers isolate synchronization from rounding.
__global__ void reuse_scratch(float* output) {
  __shared__ float scratch[32];
  const int tid = threadIdx.x;
  float total = 0.0f;
  for (int i = 0; i < 8; ++i) {
    const float x = static_cast<float>((tid % 7) + i + 1);
    const float a = block_sum_parallel_unsafe(x, scratch);
    __syncthreads();
    const float b = block_sum_parallel_unsafe(x + 1.0f, scratch);
    __syncthreads();
    const float c = block_max_parallel_unsafe(x, scratch);
    __syncthreads();
    const float d = block_max_parallel_unsafe(x + 2.0f, scratch);
    total += a + b + c + d;
    if (i != 7) __syncthreads();  // Uniform; no barrier after the last use.
  }
  output[blockIdx.x * blockDim.x + tid] = total;
}

// Softmax scores may all be negative; inactive reduction lanes must not
// replace their actual maximum with zero.
__global__ void negative_max(float* output) {
  __shared__ float scratch[32];
  const float value = -static_cast<float>(threadIdx.x + 1);
  output[blockIdx.x * blockDim.x + threadIdx.x] =
      block_max_parallel_unsafe(value, scratch);
}

static void check_close(float actual, double expected, double tolerance) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
    std::fprintf(stderr, "actual=%.9g expected=%.9g tolerance=%.9g\n",
                 actual, expected, tolerance);
    std::exit(1);
  }
}

template<class T> static T* upload(const std::vector<T>& values) {
  T* p;
  CHECK(cudaMalloc(&p, values.size() * sizeof(T)));
  CHECK(cudaMemcpy(p, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
  return p;
}

template<class T> static void download(std::vector<T>& values, const T* p) {
  CHECK(cudaMemcpy(values.data(), p, values.size() * sizeof(T), cudaMemcpyDeviceToHost));
}

static void test_reuse() {
  for (int threads : {32, 64, 128, 256, 512, 1024}) {
    std::vector<float> values(9 * threads);
    float* p = upload(values);
    double expected = 0;
    for (int i = 0; i < 8; ++i) {
      for (int t = 0; t < threads; ++t) expected += 2 * ((t % 7) + i + 1) + 1;
      expected += (i + 7) + (i + 9);
    }
    for (int repeat = 0; repeat < 10; ++repeat) {
      reuse_scratch<<<9, threads>>>(p);
      CHECK(cudaGetLastError());
      download(values, p);
      for (float value : values) check_close(value, expected, 0);
      negative_max<<<9, threads>>>(p);
      CHECK(cudaGetLastError());
      download(values, p);
      for (float value : values) check_close(value, -1.0, 0);
    }
    CHECK(cudaFree(p));
  }
}

static void test_softmax() {
  for (int cols : {1, 7, 31, 32, 33, 127, 128, 129, 513, 1024}) {
    std::vector<half> input(9 * cols), result(input.size());
    for (size_t i = 0; i < input.size(); ++i)
      input[i] = __float2half(std::sin(i * 0.17f) * 3.0f);
    half* p = upload(input);
    for (int repeat = 0; repeat < 10; ++repeat) {
      CHECK(cudaMemcpy(p, input.data(), input.size() * sizeof(half), cudaMemcpyHostToDevice));
      mqa_softmax_f16_block_kernel<<<9, 128>>>(p, 9, cols);
      CHECK(cudaGetLastError());
      download(result, p);
      for (int row = 0; row < 9; ++row) {
        double sum = 0;
        for (int col = 0; col < cols; ++col) sum += std::exp(__half2float(input[row * cols + col]));
        for (int col = 0; col < cols; ++col) {
          double expected = std::exp(__half2float(input[row * cols + col])) / sum;
          check_close(__half2float(result[row * cols + col]), expected, 0.001 * expected + 1e-6);
        }
      }
    }
    CHECK(cudaFree(p));
  }
}

static void test_sdpa() {
  constexpr int dim = 64;
  for (int tokens : {1, 7, 31, 32, 33, 65}) {
    std::vector<__nv_bfloat16> q(tokens * dim), k(q.size()), v(q.size()), result(q.size());
    for (size_t i = 0; i < q.size(); ++i) {
      q[i] = __float2bfloat16(std::sin(i * 0.17f));
      k[i] = __float2bfloat16(std::cos(i * 0.13f));
      v[i] = __float2bfloat16(std::sin(i * 0.07f));
    }
    auto dq = upload(q), dk = upload(k), dv = upload(v), dout = upload(result);
    for (int variant = 0; variant < 2; ++variant) {
      for (int repeat = 0; repeat < 3; ++repeat) {
        if (variant == 0)
          vision_sdpa_bf16_kernel<<<tokens, 32, (tokens + 1) * sizeof(float)>>>(
              dq, dk, dv, dout, tokens, 1, dim, 0.125f);
        else
          noncausal_sdpa_bf16_kernel<<<tokens, 32, (tokens + 1) * sizeof(float)>>>(
              dq, dk, dv, dout, tokens, tokens, 1, dim, 0.125f);
        CHECK(cudaGetLastError());
        download(result, dout);
        for (int row = 0; row < tokens; ++row) {
          std::vector<double> weights(tokens);
          double denominator = 0;
          for (int t = 0; t < tokens; ++t) {
            double dot = 0;
            for (int d = 0; d < dim; ++d)
              dot += double(__bfloat162float(q[row * dim + d])) * __bfloat162float(k[t * dim + d]);
            weights[t] = std::exp(dot * 0.125);
            denominator += weights[t];
          }
          for (int d = 0; d < dim; ++d) {
            double expected = 0;
            for (int t = 0; t < tokens; ++t)
              expected += weights[t] * __bfloat162float(v[t * dim + d]);
            check_close(__bfloat162float(result[row * dim + d]), expected / denominator, 0.004);
          }
        }
      }
    }
    CHECK(cudaFree(dq)); CHECK(cudaFree(dk)); CHECK(cudaFree(dv)); CHECK(cudaFree(dout));
  }
}

// Qwen-Drive's vision path uses the cached row kernel, which has its own
// mean -> variance scratch-reuse boundary. Poison every output before replay.
template<int PerThread> static void test_cached_layer_norm() {
  constexpr int rows = 17, cols = 256 * PerThread;
  std::vector<__nv_bfloat16> input(rows * cols), weight(cols), bias(cols), result(input.size());
  std::vector<__nv_bfloat16> first;
  std::vector<double> expected(input.size());
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = __float2bfloat16(std::sin(i * 0.013f) * 3.f + std::sin((i / cols) * .41f));
  for (int c = 0; c < cols; ++c) {
    weight[c] = __float2bfloat16(1.f);
    bias[c] = __float2bfloat16(0.f);
  }
  for (int r = 0; r < rows; ++r) {
    double mean = 0, variance = 0;
    for (int c = 0; c < cols; ++c) mean += __bfloat162float(input[r * cols + c]);
    mean /= cols;
    for (int c = 0; c < cols; ++c) {
      double value = __bfloat162float(input[r * cols + c]) - mean;
      variance += value * value;
    }
    for (int c = 0; c < cols; ++c)
      expected[r * cols + c] = (__bfloat162float(input[r * cols + c]) - mean) / std::sqrt(variance / cols + 1e-6);
  }
  auto x = upload(input), w = upload(weight), b = upload(bias), y = upload(result);
  for (int fill : {0x00, 0x5a, 0xff}) {
    for (int repeat = 0; repeat < 3; ++repeat) {
      CHECK(cudaMemset(y, fill, result.size() * sizeof(__nv_bfloat16)));
      layer_norm_bf16_cached_kernel<PerThread><<<rows, 256>>>(x, w, b, y, rows, cols, 1e-6f);
      CHECK(cudaGetLastError());
      download(result, y);
      if (first.empty()) first = result;
      for (size_t i = 0; i < result.size(); ++i) {
        check_close(__bfloat162float(result[i]), expected[i], 0.016);
        check_close(__bfloat162float(result[i]), __bfloat162float(first[i]), 0);
      }
    }
  }
  CHECK(cudaFree(x)); CHECK(cudaFree(w)); CHECK(cudaFree(b)); CHECK(cudaFree(y));
}

int main() {
  test_reuse();
  test_softmax();
  test_sdpa();
  test_cached_layer_norm<2>();
  test_cached_layer_norm<4>();
  test_cached_layer_norm<8>();
  std::puts("PASS: scratch reuse, softmax, SDPA, and cached LayerNorm CPU references");
}
