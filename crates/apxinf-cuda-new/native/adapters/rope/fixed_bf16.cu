// Copyright 2026 ApxInf contributors.
// Fixed BF16 implementation for this operator family; ported without numerical changes.
#include "../../include/apxinf_cuda/transformer_bf16.h"
#include "../../kernels/custom/reduction.cuh"
#ifdef APXINF_GEMM_CUTLASS_SM87_W8A8
#include "../../kernels/cutlass/ops/gemm/gemm_i8_bf16_sm80.h"
#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kBlockSize = 256;

__device__ __forceinline__ float gelu_tanh(float value) {
  constexpr float kAlpha = 0.7978845608028654f;
  return 0.5f * value *
         (1.0f + tanhf(kAlpha * (value + 0.044715f * value * value * value)));
}

struct alignas(8) Bf16x4 {
  __nv_bfloat162 low;
  __nv_bfloat162 high;
};

struct alignas(8) W8Bf16x4 {
  __nv_bfloat162 low;
  __nv_bfloat162 high;
};

struct alignas(16) Bf16Pairx8 {
  Bf16x4 first;
  Bf16x4 second;
};

// ── RoPE kernels (from legacy rope.cuh) ──────────────────────────────────
__device__ __forceinline__ uint32_t mrope_axis_for_pair(
    uint32_t pair_idx, uint32_t sec_h, uint32_t sec_w)
{
    uint32_t rem = pair_idx % 3;
    if (rem == 1 && pair_idx < sec_h * 3) return 1;
    if (rem == 2 && pair_idx < sec_w * 3) return 2;
    return 0;
}

__global__ void build_vision_rotation_table_f32_kernel(
    const uint32_t* pos_ids, float2* rotation_table,
    uint32_t head_dim, uint32_t seq_len, float theta)
{
    const uint64_t linear = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t half = head_dim / 2;
    const uint64_t total = (uint64_t)seq_len * half;
    if (linear >= total) return;

    const uint32_t pair_idx = (uint32_t)(linear % half);
    const uint32_t seq_idx = (uint32_t)(linear / half);
    const uint32_t axis = pair_idx < half / 2 ? 0u : 1u;
    const uint32_t pair_in_axis =
        pair_idx < half / 2 ? pair_idx : pair_idx - half / 2;
    const uint32_t pos = pos_ids[seq_idx * 2 + axis];
    const float frequency =
        1.0f / powf(theta, 2.0f * static_cast<float>(pair_in_axis) /
                               static_cast<float>(half));
    float sine, cosine;
    sincosf(static_cast<float>(pos) * frequency, &sine, &cosine);
    rotation_table[linear] = make_float2(cosine, sine);
}

__global__ void qk_rms_norm_mrope_bf16_kernel(
    const __nv_bfloat16* query_input, const __nv_bfloat16* query_weight,
    __nv_bfloat16* query_output, const __nv_bfloat16* key_input,
    const __nv_bfloat16* key_weight, __nv_bfloat16* key_output,
    uint32_t head_dim, uint32_t query_heads, uint32_t key_heads,
    uint32_t seq_len, float eps, float theta, const uint32_t* pos_ids,
    uint32_t sec_h, uint32_t sec_w) {
  extern __shared__ __nv_bfloat16 rounded[];
  __shared__ float warp_sums[32];
  __shared__ float block_square_sum;
  const uint32_t combined_head = blockIdx.y;
  const bool is_query = combined_head < query_heads;
  const uint32_t heads = is_query ? query_heads : key_heads;
  const uint32_t head = is_query ? combined_head : combined_head - query_heads;
  const uint32_t seq = blockIdx.x;
  if (seq >= seq_len || head >= heads) return;
  const __nv_bfloat16* input = is_query ? query_input : key_input;
  const __nv_bfloat16* weight = is_query ? query_weight : key_weight;
  __nv_bfloat16* output = is_query ? query_output : key_output;
  const int64_t base =
      (static_cast<int64_t>(seq) * heads + head) * head_dim;

  float square_sum = 0.0f;
  for (uint32_t col = threadIdx.x; col < head_dim; col += blockDim.x) {
    const float value = __bfloat162float(input[base + col]);
    square_sum += value * value;
  }
  // Keep the reduction tree byte-for-byte equivalent to the public BF16
  // RMSNorm kernel. The legacy boundary is observable because its BF16
  // output is consumed by mRoPE.
  for (int offset = 16; offset > 0; offset >>= 1)
    square_sum += __shfl_xor_sync(0xffffffff, square_sum, offset);
  const uint32_t warp = threadIdx.x / 32;
  const uint32_t lane = threadIdx.x % 32;
  if (lane == 0) warp_sums[warp] = square_sum;
  __syncthreads();
  if (warp == 0) {
    float value =
        threadIdx.x < (blockDim.x + 31) / 32 ? warp_sums[threadIdx.x] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1)
      value += __shfl_xor_sync(0xffffffff, value, offset);
    if (lane == 0) block_square_sum = value;
  }
  __syncthreads();
  const float inverse_rms =
      rsqrtf(block_square_sum / static_cast<float>(head_dim) + eps);
  for (uint32_t col = threadIdx.x; col < head_dim; col += blockDim.x) {
    rounded[col] = __float2bfloat16(
        __bfloat162float(input[base + col]) * inverse_rms *
        __bfloat162float(weight[col]));
  }
  __syncthreads();

  const uint32_t pair = threadIdx.x;
  if (pair >= head_dim / 2) return;
  const uint32_t axis = mrope_axis_for_pair(pair, sec_h, sec_w);
  const uint32_t pos = pos_ids[seq * 3 + axis];
  const float freq =
      1.0f / powf(theta, 2.0f * static_cast<float>(pair) /
                            static_cast<float>(head_dim));
  const float angle = static_cast<float>(pos) * freq;
  const float cos_value = cosf(angle);
  const float sin_value = sinf(angle);
  const float first = __bfloat162float(rounded[pair]);
  const float second = __bfloat162float(rounded[head_dim / 2 + pair]);
  output[base + pair] =
      __float2bfloat16(first * cos_value - second * sin_value);
  output[base + head_dim / 2 + pair] =
      __float2bfloat16(first * sin_value + second * cos_value);
}

__global__ void qkv_split_bias_vision_rope_precomputed_bf16_vec2_kernel(
    const __nv_bfloat16* qkv, const __nv_bfloat16* bias,
    __nv_bfloat16* q_out, __nv_bfloat16* k_out, __nv_bfloat16* v_out,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    const float2* rotation_table) {
  const uint64_t group =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint32_t half = head_dim / 2;
  const uint32_t groups_per_head = half / 2;
  const uint64_t total =
      static_cast<uint64_t>(seq_len) * n_heads * groups_per_head;
  if (group >= total) return;

  const uint32_t group_in_head = static_cast<uint32_t>(group % groups_per_head);
  const uint64_t token_head = group / groups_per_head;
  const uint32_t head = static_cast<uint32_t>(token_head % n_heads);
  const uint32_t row = static_cast<uint32_t>(token_head / n_heads);
  const uint32_t width = n_heads * head_dim;
  const uint64_t qkv_row = static_cast<uint64_t>(row) * 3 * width;
  const uint32_t head_col = head * head_dim;
  const uint64_t out_base = token_head * head_dim;

#pragma unroll
  for (uint32_t lane = 0; lane < 2; ++lane) {
    const uint32_t pair = group_in_head * 2 + lane;
    const uint32_t second = pair + half;
    const __nv_bfloat16 q0_bf16 = __float2bfloat16(
        __bfloat162float(qkv[qkv_row + head_col + pair]) +
        __bfloat162float(bias[head_col + pair]));
    const __nv_bfloat16 q1_bf16 = __float2bfloat16(
        __bfloat162float(qkv[qkv_row + head_col + second]) +
        __bfloat162float(bias[head_col + second]));
    const __nv_bfloat16 k0_bf16 = __float2bfloat16(
        __bfloat162float(qkv[qkv_row + width + head_col + pair]) +
        __bfloat162float(bias[width + head_col + pair]));
    const __nv_bfloat16 k1_bf16 = __float2bfloat16(
        __bfloat162float(qkv[qkv_row + width + head_col + second]) +
        __bfloat162float(bias[width + head_col + second]));
    const float2 rotation =
        rotation_table[static_cast<size_t>(row) * half + pair];
    const float q0 = __bfloat162float(q0_bf16);
    const float q1 = __bfloat162float(q1_bf16);
    const float k0 = __bfloat162float(k0_bf16);
    const float k1 = __bfloat162float(k1_bf16);
    q_out[out_base + pair] =
        __float2bfloat16(q0 * rotation.x - q1 * rotation.y);
    q_out[out_base + second] =
        __float2bfloat16(q0 * rotation.y + q1 * rotation.x);
    k_out[out_base + pair] =
        __float2bfloat16(k0 * rotation.x - k1 * rotation.y);
    k_out[out_base + second] =
        __float2bfloat16(k0 * rotation.y + k1 * rotation.x);
    v_out[out_base + pair] = __float2bfloat16(
        __bfloat162float(qkv[qkv_row + 2 * width + head_col + pair]) +
        __bfloat162float(bias[2 * width + head_col + pair]));
    v_out[out_base + second] = __float2bfloat16(
        __bfloat162float(qkv[qkv_row + 2 * width + head_col + second]) +
        __bfloat162float(bias[2 * width + head_col + second]));
  }
}

__global__ void rope_mrope_bf16_kernel(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    float theta, const uint32_t* pos_ids,
    uint32_t sec_h, uint32_t sec_w)
{
    uint32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head_idx = blockIdx.y;
    uint32_t seq_idx  = blockIdx.z;
    if (pair_idx >= head_dim / 2) return;

    uint32_t axis = mrope_axis_for_pair(pair_idx, sec_h, sec_w);
    uint32_t pos  = pos_ids[seq_idx * 3 + axis];

    float freq    = 1.0f / powf(theta, 2.0f * (float)pair_idx / (float)head_dim);
    float angle   = (float)pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    uint32_t base = seq_idx * n_heads * head_dim + head_idx * head_dim;
    uint32_t half = head_dim / 2;
    uint32_t idx0 = base + pair_idx;
    uint32_t idx1 = base + half + pair_idx;
    float x0 = __bfloat162float(input[idx0]);
    float x1 = __bfloat162float(input[idx1]);
    output[idx0] = __float2bfloat16(x0 * cos_val - x1 * sin_val);
    output[idx1] = __float2bfloat16(x0 * sin_val + x1 * cos_val);
}

__global__ void rope_vision_2d_pair_bf16_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k,
    __nv_bfloat16* q_out, __nv_bfloat16* k_out,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    float theta, const uint32_t* pos_ids)
{
    uint64_t linear = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t half = head_dim / 2;
    uint64_t total = (uint64_t)seq_len * n_heads * half;
    if (linear >= total) return;

    uint32_t pair_idx = (uint32_t)(linear % half);
    uint64_t token_head = linear / half;
    uint32_t head_idx = (uint32_t)(token_head % n_heads);
    uint32_t seq_idx = (uint32_t)(token_head / n_heads);
    uint32_t axis = pair_idx < half / 2 ? 0u : 1u;
    uint32_t pair_in_axis = pair_idx < half / 2 ? pair_idx : pair_idx - half / 2;
    uint32_t pos = pos_ids[seq_idx * 2 + axis];

    float freq = 1.0f / powf(theta, 2.0f * (float)pair_in_axis / (float)half);
    float angle = (float)pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    uint64_t base = ((uint64_t)seq_idx * n_heads + head_idx) * head_dim;
    uint64_t idx0 = base + pair_idx;
    uint64_t idx1 = base + half + pair_idx;

    float q0 = __bfloat162float(q[idx0]);
    float q1 = __bfloat162float(q[idx1]);
    q_out[idx0] = __float2bfloat16(q0 * cos_val - q1 * sin_val);
    q_out[idx1] = __float2bfloat16(q0 * sin_val + q1 * cos_val);

    float k0 = __bfloat162float(k[idx0]);
    float k1 = __bfloat162float(k[idx1]);
    k_out[idx0] = __float2bfloat16(k0 * cos_val - k1 * sin_val);
    k_out[idx1] = __float2bfloat16(k0 * sin_val + k1 * cos_val);
}

template <bool kPrecomputed>
__global__ void qkv_split_bias_vision_rope_bf16_kernel(
    const __nv_bfloat16* qkv, const __nv_bfloat16* bias,
    __nv_bfloat16* q_out, __nv_bfloat16* k_out, __nv_bfloat16* v_out,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    float theta, const uint32_t* pos_ids, const float2* rotation_table)
{
    uint64_t linear = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t half = head_dim / 2;
    const uint64_t total = (uint64_t)seq_len * n_heads * half;
    if (linear >= total) return;

    const uint32_t pair_idx = (uint32_t)(linear % half);
    const uint64_t token_head = linear / half;
    const uint32_t head_idx = (uint32_t)(token_head % n_heads);
    const uint32_t seq_idx = (uint32_t)(token_head / n_heads);
    const uint32_t width = n_heads * head_dim;
    const uint64_t qkv_row = (uint64_t)seq_idx * 3 * width;
    const uint32_t head_col = head_idx * head_dim;
    const uint32_t second = pair_idx + half;
    const uint64_t out_base = ((uint64_t)seq_idx * n_heads + head_idx) * head_dim;

    // Match split_qkv_bias_bf16's intermediate output rounding exactly.
    const __nv_bfloat16 q0_bf16 = __float2bfloat16(
        __bfloat162float(qkv[qkv_row + head_col + pair_idx]) +
        __bfloat162float(bias[head_col + pair_idx]));
    const __nv_bfloat16 q1_bf16 = __float2bfloat16(
        __bfloat162float(qkv[qkv_row + head_col + second]) +
        __bfloat162float(bias[head_col + second]));
    const __nv_bfloat16 k0_bf16 = __float2bfloat16(
        __bfloat162float(qkv[qkv_row + width + head_col + pair_idx]) +
        __bfloat162float(bias[width + head_col + pair_idx]));
    const __nv_bfloat16 k1_bf16 = __float2bfloat16(
        __bfloat162float(qkv[qkv_row + width + head_col + second]) +
        __bfloat162float(bias[width + head_col + second]));

    float sine, cosine;
    if constexpr (kPrecomputed) {
        const float2 rotation =
            rotation_table[static_cast<size_t>(seq_idx) * half + pair_idx];
        cosine = rotation.x;
        sine = rotation.y;
    } else {
        const uint32_t axis = pair_idx < half / 2 ? 0u : 1u;
        const uint32_t pair_in_axis =
            pair_idx < half / 2 ? pair_idx : pair_idx - half / 2;
        const uint32_t pos = pos_ids[seq_idx * 2 + axis];
        const float frequency =
            1.0f / powf(theta, 2.0f * static_cast<float>(pair_in_axis) /
                                   static_cast<float>(half));
        sincosf(static_cast<float>(pos) * frequency, &sine, &cosine);
    }
    const float q0 = __bfloat162float(q0_bf16);
    const float q1 = __bfloat162float(q1_bf16);
    const float k0 = __bfloat162float(k0_bf16);
    const float k1 = __bfloat162float(k1_bf16);
    q_out[out_base + pair_idx] = __float2bfloat16(q0 * cosine - q1 * sine);
    q_out[out_base + second] = __float2bfloat16(q0 * sine + q1 * cosine);
    k_out[out_base + pair_idx] = __float2bfloat16(k0 * cosine - k1 * sine);
    k_out[out_base + second] = __float2bfloat16(k0 * sine + k1 * cosine);

    v_out[out_base + pair_idx] = __float2bfloat16(
        __bfloat162float(qkv[qkv_row + 2 * width + head_col + pair_idx]) +
        __bfloat162float(bias[2 * width + head_col + pair_idx]));
    v_out[out_base + second] = __float2bfloat16(
        __bfloat162float(qkv[qkv_row + 2 * width + head_col + second]) +
        __bfloat162float(bias[2 * width + head_col + second]));
}


// Split a rank-2 packed [tokens, 3*hidden] BF16 projection (Q then K then V,
// each hidden = heads * head_dim) into three contiguous [tokens, heads,
// head_dim] tensors. Equivalent to the legacy FA2 strided-QKV reader, done
// with an explicit copy so the generic attention op can consume it.
__global__ void split_strided_qkv_bf16_kernel(
    const __nv_bfloat16* qkv, __nv_bfloat16* q, __nv_bfloat16* k,
    __nv_bfloat16* v, int tokens, int hidden) {
  const int64_t total = static_cast<int64_t>(tokens) * hidden;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total; index += stride) {
    const int64_t token = index / hidden;
    const int col = static_cast<int>(index % hidden);
    const __nv_bfloat16* row = qkv + token * 3 * hidden;
    q[index] = row[col];
    k[index] = row[hidden + col];
    v[index] = row[2 * hidden + col];
  }
}


}  // namespace

// ── RoPE launchers ───────────────────────────────────────────────────────

extern "C" cudaError_t apxinf_transformer_bf16_rope_mrope_bf16(
    const void* input, void* output, uint32_t head_dim, uint32_t n_heads,
    uint32_t seq_len, float theta, const void* pos_ids, uint32_t sec_h,
    uint32_t sec_w, cudaStream_t stream) {
  dim3 grid((head_dim / 2 + kBlockSize - 1) / kBlockSize, n_heads,
            seq_len);
  dim3 block(kBlockSize, 1, 1);
  rope_mrope_bf16_kernel<<<grid, block, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_bfloat16*>(output), head_dim, n_heads, seq_len, theta,
      static_cast<const uint32_t*>(pos_ids), sec_h, sec_w);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_transformer_bf16_build_vision_rotation_table_f32(
    const void* pos_ids, void* rotation_table, uint32_t head_dim,
    uint32_t seq_len, float theta, cudaStream_t stream) {
  if (pos_ids == nullptr || rotation_table == nullptr || head_dim == 0 ||
      (head_dim % 2) != 0 || seq_len == 0 || !(theta > 0.0f))
    return cudaErrorInvalidValue;
  const uint64_t total = static_cast<uint64_t>(seq_len) * (head_dim / 2);
  dim3 grid(static_cast<uint32_t>((total + kBlockSize - 1) /
                                  kBlockSize),
            1, 1);
  dim3 block(kBlockSize, 1, 1);
  build_vision_rotation_table_f32_kernel<<<grid, block, 0, stream>>>(
      static_cast<const uint32_t*>(pos_ids),
      static_cast<float2*>(rotation_table), head_dim, seq_len, theta);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_transformer_bf16_rope_vision_2d_pair_bf16(
    const void* q, const void* k, void* q_out, void* k_out, uint32_t head_dim,
    uint32_t n_heads, uint32_t seq_len, float theta, const void* pos_ids,
    cudaStream_t stream) {
  uint64_t total = static_cast<uint64_t>(seq_len) * n_heads * (head_dim / 2);
  dim3 grid(static_cast<uint32_t>((total + kBlockSize - 1) /
                                  kBlockSize),
            1, 1);
  dim3 block(kBlockSize, 1, 1);
  rope_vision_2d_pair_bf16_kernel<<<grid, block, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(q),
      static_cast<const __nv_bfloat16*>(k),
      static_cast<__nv_bfloat16*>(q_out), static_cast<__nv_bfloat16*>(k_out),
      head_dim, n_heads, seq_len, theta,
      static_cast<const uint32_t*>(pos_ids));
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_transformer_bf16_qk_rms_norm_mrope_bf16_with_threads(
    const void* query_input, const void* query_weight, void* query_output,
    const void* key_input, const void* key_weight, void* key_output,
    uint32_t head_dim, uint32_t query_heads, uint32_t key_heads,
    uint32_t seq_len, float eps, float theta, const void* pos_ids,
    uint32_t sec_h, uint32_t sec_w, uint32_t block_threads,
    cudaStream_t stream) {
  if ((block_threads != 128 && block_threads != 256) ||
      query_input == nullptr || query_weight == nullptr ||
      query_output == nullptr || key_input == nullptr ||
      key_weight == nullptr || key_output == nullptr || pos_ids == nullptr ||
      head_dim == 0 || (head_dim % 2) != 0 || query_heads == 0 ||
      key_heads == 0 || seq_len == 0 || head_dim / 2 > block_threads ||
      !(eps > 0.0f) || !(theta > 0.0f))
    return cudaErrorInvalidValue;
  dim3 grid(seq_len, query_heads + key_heads, 1);
  dim3 block(block_threads, 1, 1);
  qk_rms_norm_mrope_bf16_kernel<<<grid, block,
                                  head_dim * sizeof(__nv_bfloat16), stream>>>(
      static_cast<const __nv_bfloat16*>(query_input),
      static_cast<const __nv_bfloat16*>(query_weight),
      static_cast<__nv_bfloat16*>(query_output),
      static_cast<const __nv_bfloat16*>(key_input),
      static_cast<const __nv_bfloat16*>(key_weight),
      static_cast<__nv_bfloat16*>(key_output), head_dim, query_heads,
      key_heads, seq_len, eps, theta, static_cast<const uint32_t*>(pos_ids),
      sec_h, sec_w);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_transformer_bf16_qkv_split_bias_vision_rope_bf16(
    const void* qkv, const void* bias, void* q_out, void* k_out, void* v_out,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len, float theta,
    const void* pos_ids, cudaStream_t stream) {
  uint64_t total = static_cast<uint64_t>(seq_len) * n_heads * (head_dim / 2);
  dim3 grid(static_cast<uint32_t>((total + kBlockSize - 1) /
                                  kBlockSize),
            1, 1);
  dim3 block(kBlockSize, 1, 1);
  qkv_split_bias_vision_rope_bf16_kernel<false><<<grid, block, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(qkv),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<__nv_bfloat16*>(q_out), static_cast<__nv_bfloat16*>(k_out),
      static_cast<__nv_bfloat16*>(v_out), head_dim, n_heads, seq_len, theta,
      static_cast<const uint32_t*>(pos_ids), nullptr);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_transformer_bf16_qkv_split_bias_vision_rope_precomputed_bf16(
    const void* qkv, const void* bias, void* q_out, void* k_out, void* v_out,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    const void* rotation_table, cudaStream_t stream) {
  if (qkv == nullptr || bias == nullptr || q_out == nullptr ||
      k_out == nullptr || v_out == nullptr || rotation_table == nullptr ||
      head_dim == 0 || (head_dim % 2) != 0 || n_heads == 0 || seq_len == 0)
    return cudaErrorInvalidValue;
  uint64_t total = static_cast<uint64_t>(seq_len) * n_heads * (head_dim / 2);
  dim3 grid(static_cast<uint32_t>((total + kBlockSize - 1) /
                                  kBlockSize),
            1, 1);
  dim3 block(kBlockSize, 1, 1);
  qkv_split_bias_vision_rope_bf16_kernel<true><<<grid, block, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(qkv),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<__nv_bfloat16*>(q_out), static_cast<__nv_bfloat16*>(k_out),
      static_cast<__nv_bfloat16*>(v_out), head_dim, n_heads, seq_len, 0.0f,
      nullptr, static_cast<const float2*>(rotation_table));
  return cudaGetLastError();
}

extern "C" cudaError_t
apxinf_transformer_bf16_qkv_split_bias_vision_rope_precomputed_vec2_bf16(
    const void* qkv, const void* bias, void* q_out, void* k_out, void* v_out,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    const void* rotation_table, cudaStream_t stream) {
  if (qkv == nullptr || bias == nullptr || q_out == nullptr ||
      k_out == nullptr || v_out == nullptr || rotation_table == nullptr ||
      head_dim == 0 || head_dim % 4 != 0 || n_heads == 0 || seq_len == 0)
    return cudaErrorInvalidValue;
  const uint64_t total =
      static_cast<uint64_t>(seq_len) * n_heads * (head_dim / 4);
  constexpr uint32_t threads = 128;
  const dim3 grid(static_cast<uint32_t>((total + threads - 1) / threads), 1, 1);
  qkv_split_bias_vision_rope_precomputed_bf16_vec2_kernel<<<grid, threads, 0,
                                                            stream>>>(
      static_cast<const __nv_bfloat16*>(qkv),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<__nv_bfloat16*>(q_out), static_cast<__nv_bfloat16*>(k_out),
      static_cast<__nv_bfloat16*>(v_out), head_dim, n_heads, seq_len,
      static_cast<const float2*>(rotation_table));
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_transformer_bf16_split_strided_qkv_bf16(
    const void* qkv, void* q, void* k, void* v, int tokens, int hidden,
    cudaStream_t stream) {
  if (qkv == nullptr || q == nullptr || k == nullptr || v == nullptr ||
      tokens <= 0 || hidden <= 0)
    return cudaErrorInvalidValue;
  const int64_t total = static_cast<int64_t>(tokens) * hidden;
  const int blocks =
      static_cast<int>((total + kBlockSize - 1) / kBlockSize);
  split_strided_qkv_bf16_kernel<<<blocks, kBlockSize, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(qkv), static_cast<__nv_bfloat16*>(q),
      static_cast<__nv_bfloat16*>(k), static_cast<__nv_bfloat16*>(v), tokens,
      hidden);
  return cudaGetLastError();
}
