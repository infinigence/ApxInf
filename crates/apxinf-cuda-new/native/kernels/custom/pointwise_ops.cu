// Copyright 2026 ApxInf contributors.
//
// The pointwise family's BF16 SwiGLU and residual add.
//
// Moved verbatim out of kernels/custom/mlp_ops.cu, which was named after a call
// site rather than a mathematical family (doc/cuda-new-design.md section 1.3).
// `silu`, `swiglu_kernel`, `add_kernel`, `add_vector_kernel` and both host
// launches are that code unchanged apart from the enclosing namespace.
//
// This is a separate translation unit rather than part of
// kernels/custom/pointwise.cuh because those kernels are templates over the
// F16/BF16 storage type dispatched through a Spec, while these two are
// hard-coded to __nv_bfloat16 and take flat parameters. Folding them into the
// template family would mean giving them a Spec they have no variants for.
//
// Single implementation, no persisted selection, so per
// doc/adding-new-kernels.md section 6 they carry no recipe.

#include "pointwise_ops.h"

#include "vector_elementwise.h"

#include <cuda_bf16.h>

namespace apxinf::cuda_new::pointwise_ops {
namespace {

__device__ __forceinline__ float silu(float value) {
  return value / (1.0f + __expf(-value));
}

// `fused` holds gate and up side by side: [rows, 2*width] with gate first.
// Qwen's MLP is SwiGLU -- silu(gate) * up -- which is not what gemm_geglu
// computes, so it cannot be reused.
__global__ void swiglu_kernel(const __nv_bfloat16* __restrict__ fused,
                              __nv_bfloat16* __restrict__ output, int rows,
                              int width) {
  const long long index = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  const long long total = (long long)rows * width;
  if (index >= total) return;
  const int row = static_cast<int>(index / width);
  const int column = static_cast<int>(index % width);
  const long long base = (long long)row * 2 * width;
  const float gate = __bfloat162float(fused[base + column]);
  const float up = __bfloat162float(fused[base + width + column]);
  // Reference semantics are bf16(bf16(silu(gate)) * up): the SiLU output
  // materializes in BF16 before the multiply (vLLM parity defect D4,
  // report 64 / Codex round 63 -- rounding only the product is not enough).
  const float activated = __bfloat162float(__float2bfloat16(silu(gate)));
  output[index] = __float2bfloat16(activated * up);
}

// Eight elements per thread through 16-byte vectors. One bf16 per thread
// (round-05 nsys: 46.7 ms of a 874 ms prefill across 130 calls) leaves the
// kernel launch-and-index bound; the tail elements run scalar so any count
// works. Same per-element arithmetic, bit-identical.
__global__ void add_kernel(const __nv_bfloat16* __restrict__ addend,
                           __nv_bfloat16* __restrict__ accumulator,
                           long long count) {
  const long long vectors = count / 8;
  const long long index = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (index < vectors) {
    uint4 a = reinterpret_cast<const uint4*>(addend)[index];
    uint4 c = reinterpret_cast<uint4*>(accumulator)[index];
    const __nv_bfloat16* av = reinterpret_cast<const __nv_bfloat16*>(&a);
    __nv_bfloat16* cv = reinterpret_cast<__nv_bfloat16*>(&c);
#pragma unroll
    for (int element = 0; element < 8; ++element) {
      cv[element] = __float2bfloat16(__bfloat162float(cv[element]) +
                                     __bfloat162float(av[element]));
    }
    reinterpret_cast<uint4*>(accumulator)[index] = c;
    return;
  }
  const long long tail = vectors * 8 + (index - vectors);
  if (tail < count) {
    accumulator[tail] = __float2bfloat16(__bfloat162float(accumulator[tail]) +
                                         __bfloat162float(addend[tail]));
  }
}

__global__ void add_vector_kernel(const uint4* __restrict__ addend,
                                  uint4* __restrict__ accumulator,
                                  long long vectors) {
  const long long index = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (index >= vectors) return;
  const uint4 source = addend[index];
  uint4 destination = accumulator[index];
  const auto* source_pairs = reinterpret_cast<const __nv_bfloat162*>(&source);
  auto* destination_pairs = reinterpret_cast<__nv_bfloat162*>(&destination);
#pragma unroll
  for (int offset = 0; offset < 4; ++offset) {
    destination_pairs[offset] = __hadd2(destination_pairs[offset], source_pairs[offset]);
  }
  accumulator[index] = destination;
}

}  // namespace

int swiglu_bf16(const void* fused_gate_up, void* output, int rows, int width,
                cudaStream_t stream) {
  if (rows <= 0 || width <= 0) return -1;
  const long long total = (long long)rows * width;
  const int threads = 256;
  const long long blocks = (total + threads - 1) / threads;
  swiglu_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(fused_gate_up),
      static_cast<__nv_bfloat16*>(output), rows, width);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

int add_bf16(const void* addend, void* accumulator, long long count,
             cudaStream_t stream) {
  if (count <= 0) return -1;
  const int threads = 256;
  if (vector_elementwise::enabled(addend, accumulator, count)) {
    const long long vectors = count / 8;
    add_vector_kernel<<<static_cast<int>((vectors + threads - 1) / threads), threads, 0, stream>>>(
        static_cast<const uint4*>(addend), static_cast<uint4*>(accumulator), vectors);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
  }
  // One thread per 8-element vector plus one per tail element.
  const long long work = count / 8 + count % 8;
  const long long blocks = (work + threads - 1) / threads;
  add_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(addend),
      static_cast<__nv_bfloat16*>(accumulator), count);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

}  // namespace apxinf::cuda_new::pointwise_ops
