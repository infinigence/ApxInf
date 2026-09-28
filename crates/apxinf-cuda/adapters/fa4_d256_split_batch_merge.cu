#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {
constexpr int kQWords = 50 * 16 * 256;
constexpr int kRows = 50 * 16;
constexpr int kVectorsPerRow = 256 / 8;
constexpr int kVectorsPerQ = kQWords / 8;

// cudaMalloc/Torch device buffers and all row/batch strides here are 16B aligned.
// Each thread copies eight BF16 values via one 128-bit load and two stores.
__global__ void duplicate_q_vec8(const uint4* __restrict__ src,
                                 uint4* __restrict__ dst) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < kVectorsPerQ) {
    uint4 v = src[i];
    dst[i] = v;
    dst[i + kVectorsPerQ] = v;
  }
}

union Bf16x8 {
  uint4 vector;
  uint16_t words[8];
};

// One warp handles one (query,head) row. Each lane computes eight contiguous
// BF16 outputs; lane zero computes the shared LSE weights once for the warp.
__global__ void merge_partial_vec8(const uint4* __restrict__ partial,
                                   const float* __restrict__ lse,
                                   uint4* __restrict__ out) {
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  if (warp >= kRows) return;
  int lane = threadIdx.x & 31;
  int q = warp / 16;
  int h = warp % 16;
  int li = h * 50 + q;
  float w0 = 0.0f, w1 = 0.0f, sum = 0.0f;
  if (lane == 0) {
    float l0 = lse[li];
    float l1 = lse[800 + li];
    float m = fmaxf(l0, l1);
    w0 = expf(l0 - m);
    w1 = expf(l1 - m);
    sum = w0 + w1;
  }
  w0 = __shfl_sync(0xffffffff, w0, 0);
  w1 = __shfl_sync(0xffffffff, w1, 0);
  sum = __shfl_sync(0xffffffff, sum, 0);

  int vec = warp * kVectorsPerRow + lane;
  Bf16x8 a{}, b{}, r{};
  a.vector = partial[vec];
  b.vector = partial[kVectorsPerQ + vec];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    float fa = __bfloat162float(__ushort_as_bfloat16(a.words[j]));
    float fb = __bfloat162float(__ushort_as_bfloat16(b.words[j]));
    r.words[j] = __bfloat16_as_ushort(__float2bfloat16_rn((w0 * fa + w1 * fb) / sum));
  }
  out[vec] = r.vector;
}
}  // namespace

extern "C" int32_t apxinf_split_batch_duplicate_q(const void* src, void* dst,
                                                     cudaStream_t stream) {
  if (!src || !dst || ((uintptr_t(src) | uintptr_t(dst)) & 15))
    return int32_t(cudaErrorInvalidValue);
  duplicate_q_vec8<<<100, 256, 0, stream>>>(
      static_cast<const uint4*>(src), static_cast<uint4*>(dst));
  return int32_t(cudaGetLastError());
}

extern "C" int32_t apxinf_split_batch_merge(const void* partial, const float* lse,
                                              void* out, cudaStream_t stream) {
  if (!partial || !lse || !out || ((uintptr_t(partial) | uintptr_t(out)) & 15))
    return int32_t(cudaErrorInvalidValue);
  merge_partial_vec8<<<100, 256, 0, stream>>>(
      static_cast<const uint4*>(partial), lse, static_cast<uint4*>(out));
  return int32_t(cudaGetLastError());
}
