#pragma once

__global__ void suppress_logits_bf16_kernel(
    __nv_bfloat16* logits, const uint32_t* ids, int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  logits[ids[idx]] = __float2bfloat16(-INFINITY);
}
