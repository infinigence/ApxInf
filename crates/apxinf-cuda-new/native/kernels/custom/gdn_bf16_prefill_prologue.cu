// SPDX-License-Identifier: Apache-2.0
// L0 BF16 GDN prefill prologue and state-transpose kernels, shared by the AOT
// and in-tree provider arms.

#include "gdn_bf16_prefill.h"
#include "gdn_bf16_prefill.cuh"

namespace {
using namespace apxinf_gdn_bf16;

__global__ void write_offsets(int32_t* out, int32_t seq) {
  if (threadIdx.x == 0) { out[0] = 0; out[1] = seq; }
}

__global__ void state_vmajor_to_kmajor(const float* in, float* out) {
  int block = blockIdx.x, row = blockIdx.y * 32 + threadIdx.y,
      col = blockIdx.z * 32 + threadIdx.x;
  __shared__ float tile[32][33];
  tile[threadIdx.y][threadIdx.x] =
      in[size_t(block) * D * D + size_t(row) * D + col];
  __syncthreads();
  row = blockIdx.z * 32 + threadIdx.y;
  col = blockIdx.y * 32 + threadIdx.x;
  out[size_t(block) * D * D + size_t(row) * D + col] =
      tile[threadIdx.x][threadIdx.y];
}

__global__ void pack_ab_only_kernel(const __nv_bfloat16* zba,
                                    __nv_bfloat16* a, __nv_bfloat16* b,
                                    int seq, int seq_pad) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size_t(seq_pad) * 32) return;
  int token = int(i / 32), head = int(i % 32);
  if (token < seq) {
    a[i] = zba[size_t(token) * 12352 + 8192 + 4096 + 32 + head];
    b[i] = zba[size_t(token) * 12352 + 8192 + 4096 + head];
  } else {
    a[i] = __float2bfloat16(0.f);
    b[i] = __float2bfloat16(0.f);
  }
}

// The GDN kernel takes a LINEAR-space decay and applies `lg2` itself; see
// native/kernels/flashinfer_gdn/README.md for why consuming `g` directly is
// wrong.
// Padding tokens get the benign `alpha = 1, beta = 0`.
__global__ void flash_alpha_beta(const __nv_bfloat16* a,
                                 const __nv_bfloat16* b, const float* A_log,
                                 const __nv_bfloat16* dt_bias, float* alpha,
                                 float* beta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= FLASH_PAD * HV) return;
  int token = i / HV, head = i % HV;
  if (token >= FLASH_SEQ) { alpha[i] = 1.f; beta[i] = 0.f; return; }
  float x = __bfloat162float(a[i]) + __bfloat162float(dt_bias[head]);
  float softplus = x > 20.f ? x : logf(1.f + expf(x));
  float g_log = -expf(A_log[head]) * softplus;
  alpha[i] = expf(g_log);
  beta[i] = 1.f / (1.f + expf(-__bfloat162float(b[i])));
}
}  // namespace

extern "C" int32_t apxinf_gdn_bf16_prefill_prologue(
    const void* zba, const void* conv_weight, void* new_conv_state,
    const float* A_log, const void* dt_bias_bf16, void* raw_qk_scratch,
    void* q, void* k, void* v, void* a, void* b, int32_t* cu_seqlens,
    int32_t seq, int32_t seq_pad, float** alpha, float** beta,
    void** descriptor_workspace, cudaStream_t stream) {
  auto* scratch = static_cast<unsigned char*>(raw_qk_scratch);
  *alpha = reinterpret_cast<float*>(scratch);
  *beta = reinterpret_cast<float*>(scratch + FLASH_GATE_BYTES);
  *descriptor_workspace = scratch + 2 * FLASH_GATE_BYTES;
  constexpr int CONV_TOKENS = 32, CONV_KERNEL = 4;
  constexpr size_t FUSED_SHARED =
      (CONV_TOKENS + CONV_KERNEL - 1 + CONV_TOKENS) * 256 *
      sizeof(__nv_bfloat16);
  causal_conv_edge_fused_qkv_kernel<<<
      dim3((FLASH_SEQ + CONV_TOKENS - 1) / CONV_TOKENS, CONV / 256), 256,
      FUSED_SHARED, stream>>>(
      static_cast<const __nv_bfloat16*>(zba),
      static_cast<const __nv_bfloat16*>(conv_weight),
      static_cast<__nv_bfloat16*>(q), static_cast<__nv_bfloat16*>(k),
      static_cast<__nv_bfloat16*>(v),
      static_cast<__nv_bfloat16*>(new_conv_state), seq, seq_pad);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return status(error);
  write_offsets<<<1, 1, 0, stream>>>(cu_seqlens, seq);
  if ((error = cudaGetLastError()) != cudaSuccess) return status(error);
  pack_ab_only_kernel<<<(FLASH_PAD * HV + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(zba),
      static_cast<__nv_bfloat16*>(a), static_cast<__nv_bfloat16*>(b), seq,
      seq_pad);
  if ((error = cudaGetLastError()) != cudaSuccess) return status(error);
  flash_alpha_beta<<<(FLASH_PAD * HV + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(a),
      static_cast<const __nv_bfloat16*>(b), A_log,
      static_cast<const __nv_bfloat16*>(dt_bias_bf16), *alpha, *beta);
  return status(cudaGetLastError());
}

extern "C" int32_t apxinf_gdn_bf16_prefill_state_transpose(
    const float* h_output_vmajor, float* final_state_kmajor,
    cudaStream_t stream) {
  state_vmajor_to_kmajor<<<dim3(HV, 4, 4), dim3(32, 32), 0, stream>>>(
      h_output_vmajor, final_state_kmajor);
  return status(cudaGetLastError());
}
