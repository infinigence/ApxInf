// Copyright 2026 ApxInf contributors.
//
// The gather family's plain BF16 embedding lookup.
//
// Moved verbatim out of kernels/custom/model_ops.cu, which was named after a
// call site rather than a mathematical family (doc/cuda-new-design.md section
// 1.3). Kernel and host launch are unchanged apart from the namespace.
//
// See gather_embedding.h for why this is not GatherSemantic::EmbeddingLookup.

#include "gather_embedding.h"

#include <cuda_bf16.h>

namespace apxinf::cuda_new::gather_embedding {
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

}  // namespace

int embedding_gather(const void* table, const int32_t* ids, void* output,
                     int tokens, int hidden, int vocab, cudaStream_t stream) {
  if (tokens <= 0 || hidden <= 0 || vocab <= 0) return -1;
  const long long total = (long long)tokens * hidden;
  const int threads = 256;
  const long long blocks = (total + threads - 1) / threads;
  embedding_gather_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(table), ids,
      static_cast<__nv_bfloat16*>(output), tokens, hidden, vocab);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

}  // namespace apxinf::cuda_new::gather_embedding
