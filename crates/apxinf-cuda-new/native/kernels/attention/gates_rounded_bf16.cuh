#pragma once

__device__ __forceinline__ float attention_rounded_sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__global__ void sigmoid_gate_mul_bf16_kernel(
    __nv_bfloat16* attn, const __nv_bfloat16* fused, int heads, int head_dim,
    int64_t fused_width) {
  const int row = blockIdx.x;
  const int64_t base = static_cast<int64_t>(row) * heads * head_dim;
  const int64_t gate_row = static_cast<int64_t>(row) * fused_width;
  for (int idx = threadIdx.x; idx < heads * head_dim; idx += blockDim.x) {
    const int head = idx / head_dim;
    const int d = idx - head * head_dim;
    const float gate = __bfloat162float(
        fused[gate_row + head * 2 * head_dim + head_dim + d]);
    const float s = __bfloat162float(
        __float2bfloat16(attention_rounded_sigmoid(gate)));
    attn[base + idx] =
        __float2bfloat16(__bfloat162float(attn[base + idx]) * s);
  }
}

__global__ void expert_sigmoid_gate_mul_bf16_kernel(
    __nv_bfloat16* attn, const __nv_bfloat16* gate, int64_t count) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const float g = __bfloat162float(gate[index]);
    const float s =
        __bfloat162float(__float2bfloat16(attention_rounded_sigmoid(g)));
    attn[index] = __float2bfloat16(__bfloat162float(attn[index]) * s);
  }
}

struct alignas(16) SigmoidGatePack8 {
  __nv_bfloat16 v[8];
};

__global__ void sigmoid_gate_mul_bf16_vec8_kernel(
    __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ f,
    int rows) {
  const int vi = blockIdx.x * 256 + threadIdx.x;
  const int row = vi / 512;
  if (row >= rows) return;
  const int within = (vi % 512) * 8, head = within / 256, d = within % 256;
  SigmoidGatePack8 x = *reinterpret_cast<const SigmoidGatePack8*>(
      a + static_cast<int64_t>(row) * 4096 + within);
  SigmoidGatePack8 gate = *reinterpret_cast<const SigmoidGatePack8*>(
      f + static_cast<int64_t>(row) * 10240 + head * 512 + 256 + d);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float g = __bfloat162float(gate.v[j]);
    const float sg =
        __bfloat162float(__float2bfloat16(attention_rounded_sigmoid(g)));
    x.v[j] = __float2bfloat16(__bfloat162float(x.v[j]) * sg);
  }
  *reinterpret_cast<SigmoidGatePack8*>(
      a + static_cast<int64_t>(row) * 4096 + within) = x;
}
