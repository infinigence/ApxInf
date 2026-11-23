// Copyright 2026 ApxInf contributors.

#include "gather_embedding.h"

#include <cuda_bf16.h>

namespace apxinf::cuda_new::gather {
namespace {
__global__ void embedding_gather_kernel(const __nv_bfloat16* __restrict__ table,
                                        const int32_t* __restrict__ ids,
                                        __nv_bfloat16* __restrict__ output,
                                        int tokens, int hidden, int vocab) {
  const long long index = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  const long long total = (long long)tokens * hidden;
  if (index >= total) return;
  const int token = static_cast<int>(index / hidden);
  const int element = static_cast<int>(index % hidden);
  int id = ids[token];
  // Clamp rather than fault: an out-of-range id is a caller bug, but reading
  // arbitrary device memory would turn it into a silent wrong answer.
  if (id < 0 || id >= vocab) id = 0;
  output[index] = table[(long long)id * hidden + element];
}

// A vector owns eight contiguous BF16 elements. Width/alignment guards in
// the launcher ensure every row and every output vector is 16-byte aligned.
__global__ void embedding_gather_vector_kernel(const uint4* __restrict__ table,
    const int32_t* __restrict__ ids, uint4* __restrict__ output,
    uint32_t count, uint32_t width, int vocab) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const uint32_t row = index / width;
  const uint32_t element = index % width;
  int id = ids[row];
  if (id < 0 || id >= vocab) id = 0;
  output[index] = table[int64_t(id) * width + element];
}

}  // namespace

int embedding_gather(const void* table, const int32_t* ids, void* output,
                     int tokens, int hidden, int vocab, cudaStream_t stream) {
  if (tokens <= 0 || hidden <= 0 || vocab <= 0) return -1;
  const long long total = (long long)tokens * hidden;
  if (hidden % 8 == 0 && total / 8 <= UINT32_MAX &&
      reinterpret_cast<uintptr_t>(table) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(output) % 16 == 0) {
    const uint32_t vectors = static_cast<uint32_t>(total / 8);
    const int blocks = static_cast<int>((uint64_t(vectors) + 255) / 256);
    embedding_gather_vector_kernel<<<blocks, 256, 0, stream>>>(
        static_cast<const uint4*>(table), ids, static_cast<uint4*>(output),
        vectors, static_cast<uint32_t>(hidden / 8), vocab);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
  }
  const int threads = 256;
  const long long blocks = (total + threads - 1) / threads;
  embedding_gather_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(table), ids,
      static_cast<__nv_bfloat16*>(output), tokens, hidden, vocab);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

}  // namespace apxinf::cuda_new::gather
