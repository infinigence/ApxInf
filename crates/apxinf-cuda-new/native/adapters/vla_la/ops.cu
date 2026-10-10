// Copyright 2026 ApxInf contributors.
// Direct C-ABI launch adapters for the linear-attention / hybrid-recurrent
// operator family (qwen_drive), ported verbatim from the legacy
// adapters/custom_kernels.cu. Symbols carry the apxinf_cn_ prefix
// because the legacy crate links
// into the same binary. Per doc/adding-new-kernels.md section 6 these are
// single-implementation forwarders with nothing to tune.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace {
#include "../../kernels/custom/gdn_policy.h"
#include "../../kernels/custom/math.cuh"
#include "../../kernels/custom/reduction.cuh"
#include "../../kernels/custom/pillow_bicubic_u8.cuh"
#include "../../kernels/custom/linear_attention.cuh"
#include "../../kernels/custom/gdn_raw_inverse_f1.cuh"
#include "../../kernels/custom/gdn_chunk_state_wmma.cuh"
#include "../../kernels/custom/gdn_chunk_gemm_tri.cuh"

__global__ void adaln_gate_residual_rms_bf16_kernel(
    const __nv_bfloat16* proj, const __nv_bfloat16* residual,
    const __nv_bfloat16* gate, const __nv_bfloat16* weight,
    const __nv_bfloat16* scale, const __nv_bfloat16* shift,
    __nv_bfloat16* hidden, __nv_bfloat16* out, int cols, float eps) {
  const int row = blockIdx.x;
  extern __shared__ float la_adaln[];
  const int64_t base = static_cast<int64_t>(row) * cols;
  float partial = 0.0f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    const float multiplier = __bfloat162float(
        __float2bfloat16(1.0f + __bfloat162float(gate[i])));
    const float projected = __bfloat162float(
        __float2bfloat16(__bfloat162float(proj[base + i]) * multiplier));
    const __nv_bfloat16 rounded =
        __float2bfloat16(__bfloat162float(residual[base + i]) + projected);
    hidden[base + i] = rounded;
    const float v = __bfloat162float(rounded);
    la_adaln[i] = v;
    partial += v * v;
  }
  __shared__ float warp_sums[32];
  for (int offset = 16; offset > 0; offset >>= 1)
    partial += __shfl_xor_sync(0xffffffff, partial, offset);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) warp_sums[warp] = partial;
  __syncthreads();
  if (warp == 0) {
    float v = (lane < (blockDim.x + 31) / 32) ? warp_sums[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1)
      v += __shfl_xor_sync(0xffffffff, v, offset);
    if (lane == 0) warp_sums[0] = v;
  }
  __syncthreads();
  float mean = warp_sums[0] / cols;
  if (gridDim.x >= 16 && cols > 128 && cols % 4 == 0)
    mean = rms_vector_square_mean_bf16(hidden + base, cols);
  const float rms = rsqrtf(__fadd_rn(mean, eps));
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    const float normed = __bfloat162float(__float2bfloat16(
        la_adaln[i] * rms * __bfloat162float(weight[i])));
    const float multiplier = __bfloat162float(
        __float2bfloat16(1.0f + __bfloat162float(scale[i])));
    const float scaled = __bfloat162float(__float2bfloat16(normed * multiplier));
    out[base + i] = __float2bfloat16(scaled + __bfloat162float(shift[i]));
  }
}

__global__ void rgb_u8_to_temporal2_merge2_rect_bf16_kernel(
    const uint8_t* __restrict__ rgb, uint16_t* __restrict__ patches,
    const uint16_t* __restrict__ lut, int grid_h, int grid_w) {
  constexpr int kPatch = 16;
  constexpr int kArea = 256;
  constexpr int kRowWidth = 3 * 2 * kArea;
  const int row = blockIdx.x;
  int rem = row;
  const int merge_x = rem & 1;
  rem >>= 1;
  const int merge_y = rem & 1;
  rem >>= 1;
  const int groups_w = grid_w / 2;
  const int group_x = rem % groups_w;
  const int group_y = rem / groups_w;
  const int patch_y = group_y * 2 + merge_y;
  const int patch_x = group_x * 2 + merge_x;
  const int dy = threadIdx.x / kPatch;
  const int dx = threadIdx.x % kPatch;
  const int64_t pixel = ((static_cast<int64_t>(patch_y) * kPatch + dy) *
                         (static_cast<int64_t>(grid_w) * kPatch) +
                         static_cast<int64_t>(patch_x) * kPatch + dx) * 3;
  const int64_t base = static_cast<int64_t>(row) * kRowWidth + threadIdx.x;
#pragma unroll
  for (int channel = 0; channel < 3; ++channel) {
    const uint16_t normalized = lut[rgb[pixel + channel]];
    patches[base + channel * 2 * kArea] = normalized;
    patches[base + channel * 2 * kArea + kArea] = normalized;
  }
}

__global__ void sinusoidal_embedding_bf16_kernel(const float* positions,__nv_bfloat16* out,
    int dim,float scale,float frequency_step) {
  const int row=blockIdx.x,half=dim/2;
  for(int i=threadIdx.x;i<half;i+=blockDim.x) {
    const float frequency=expf(__fmul_rn(-frequency_step,static_cast<float>(i)));
    const float angle=__fmul_rn(__fmul_rn(scale,positions[row]),frequency);
    const int64_t base=static_cast<int64_t>(row)*dim;
    out[base+i]=__float2bfloat16(sinf(angle));
    out[base+half+i]=__float2bfloat16(cosf(angle));
  }
}


template <bool RoundSilu = false>
__global__ void swiglu_bf16_kernel(
    const __nv_bfloat16* gate_up, __nv_bfloat16* output,
    int rows, int inner) {
  const int64_t count = static_cast<int64_t>(rows) * inner;
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const int row = static_cast<int>(index / inner);
    const int col = static_cast<int>(index % inner);
    const float gate = __bfloat162float(gate_up[static_cast<int64_t>(row) * 2 * inner + col]);
    const float up = __bfloat162float(gate_up[static_cast<int64_t>(row) * 2 * inner + inner + col]);
    float silu = gate / (1.0f + expf(-gate));
    if (RoundSilu) silu = __bfloat162float(__float2bfloat16(silu));
    output[index] = __float2bfloat16(silu * up);
  }
}

// Eight output columns per thread through 16-byte accesses. The scalar loop
// also pays an integer divide and a modulo per element to recover (row, col);
// with inner a multiple of eight all eight land in one row, so that arithmetic
// happens once per eight. Per-element expressions are unchanged.
template <bool RoundSilu = false>
__global__ void swiglu_bf16_vec8_kernel(
    const __nv_bfloat16* __restrict__ gate_up, __nv_bfloat16* __restrict__ output,
    int rows, int inner) {
  const int64_t vec_per_row = inner / 8;
  const int64_t vec_count = static_cast<int64_t>(rows) * vec_per_row;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t v = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       v < vec_count; v += stride) {
    const int64_t row = v / vec_per_row;
    const int64_t col = (v - row * vec_per_row) * 8;
    const int64_t base = row * 2 * inner + col;
    const float4 g4 = *reinterpret_cast<const float4*>(gate_up + base);
    const float4 u4 = *reinterpret_cast<const float4*>(gate_up + base + inner);
    const __nv_bfloat16* gl = reinterpret_cast<const __nv_bfloat16*>(&g4);
    const __nv_bfloat16* ul = reinterpret_cast<const __nv_bfloat16*>(&u4);
    float4 out;
    __nv_bfloat16* ol = reinterpret_cast<__nv_bfloat16*>(&out);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float gate = __bfloat162float(gl[i]);
      const float up = __bfloat162float(ul[i]);
      float silu = gate / (1.0f + expf(-gate));
      if (RoundSilu) silu = __bfloat162float(__float2bfloat16(silu));
      ol[i] = __float2bfloat16(silu * up);
    }
    *reinterpret_cast<float4*>(output + row * inner + col) = out;
  }
}

// True when every 16-byte access the wide kernel makes is aligned: eight
// columns stay inside one row, and both halves of gate_up start on a multiple
// of eight elements.
__host__ __device__ __forceinline__ bool swiglu_vec8_ok(
    const void* gate_up, const void* output, int inner) {
  return (inner % 8) == 0 &&
         (reinterpret_cast<uintptr_t>(gate_up) % 16u) == 0 &&
         (reinterpret_cast<uintptr_t>(output) % 16u) == 0;
}



__global__ void row_softmax_f32_bf16_kernel(
    const float* input, __nv_bfloat16* output, uint32_t cols, uint32_t rows) {
  const uint32_t row = blockIdx.x;
  if (row >= rows) return;
  const float* x = input + static_cast<size_t>(row) * cols;
  __shared__ float scratch[32];
  float local = -INFINITY;
  for (uint32_t i = threadIdx.x; i < cols; i += blockDim.x) {
    local = fmaxf(local, x[i]);
  }
  const float max_val = block_max_parallel_unsafe(local, scratch);
  float partial = 0.0f;
  for (uint32_t i = threadIdx.x; i < cols; i += blockDim.x) {
    partial += expf(x[i] - max_val);
  }
  // Finish all maximum reads before the sum overwrites shared scratch.
  __syncthreads();
  const float sum = block_sum_parallel_unsafe(partial, scratch);
  __nv_bfloat16* y = output + static_cast<size_t>(row) * cols;
  for (uint32_t i = threadIdx.x; i < cols; i += blockDim.x) {
    y[i] = __float2bfloat16(expf(x[i] - max_val) / sum);
  }
}


// FIX (implement_final_r20): Option A budget repair for the composed hdim256 text
// path -- tiled block-per-row causal fp32 softmax replacing the thread-per-element
// attention_softmax_f32_kernel (measured ~3.9-4.1s per composed full-attention layer,
// ~31.6s of the ~43s prefill). Row contract identical to the stock kernel
// (kernels/custom/attention.cuh): row = blockIdx.x, seq_pos = row / n_heads,
// valid = min(seq_pos + kv_offset + 1u, cols); valid cells expf(x-max)/sum, masked
// cells exact 0.0f. In-place safe with input == output: every index is owned by
// exactly one thread and the block reductions (block_max/block_sum) synchronize
// between the read and write passes. Revert in the acceptance-bound revision.
__global__ void row_softmax_causal_f32_kernel(
    const float* input, float* output, uint32_t cols, uint32_t rows,
    uint32_t kv_offset, uint32_t n_heads) {
  const uint32_t row = blockIdx.x;
  if (row >= rows) return;
  const float* x = input + static_cast<size_t>(row) * cols;
  float* y = output + static_cast<size_t>(row) * cols;
  const uint32_t seq_pos = row / n_heads;
  const uint32_t valid = min(seq_pos + kv_offset + 1u, cols);
  __shared__ float scratch[32];
  float local = -INFINITY;
  for (uint32_t i = threadIdx.x; i < valid; i += blockDim.x) {
    local = fmaxf(local, x[i]);
  }
  const float max_val = block_max_parallel_unsafe(local, scratch);
  float partial = 0.0f;
  for (uint32_t i = threadIdx.x; i < valid; i += blockDim.x) {
    partial += expf(x[i] - max_val);
  }
  // Finish all maximum reads before the sum overwrites shared scratch.
  __syncthreads();
  const float sum = block_sum_parallel_unsafe(partial, scratch);
  for (uint32_t i = threadIdx.x; i < cols; i += blockDim.x) {
    y[i] = (i < valid) ? (expf(x[i] - max_val) / sum) : 0.0f;
  }
}

}  // namespace

extern "C" cudaError_t apxinf_cn_row_softmax_f32_bf16(
    const void* input, void* output, uint32_t cols, uint32_t rows,
    cudaStream_t stream) {
  if (input == nullptr || output == nullptr || cols == 0 || rows == 0) {
    return cudaErrorInvalidValue;
  }
  row_softmax_f32_bf16_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const float*>(input), static_cast<__nv_bfloat16*>(output),
      cols, rows);
  return cudaGetLastError();
}


extern "C" cudaError_t apxinf_cn_row_softmax_causal_f32(
    const void* input, void* output, uint32_t cols, uint32_t rows,
    uint32_t kv_offset, uint32_t n_heads, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || cols == 0 || rows == 0 ||
      n_heads == 0 || n_heads > rows) {
    return cudaErrorInvalidValue;
  }
  row_softmax_causal_f32_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const float*>(input), static_cast<float*>(output), cols, rows,
      kv_offset, n_heads);
  return cudaGetLastError();
}


// ── linear_attention.cuh launch adapters ───────────────────────────────────

extern "C" cudaError_t apxinf_cn_cast_f32_bf16(
    const void* input, void* output, int64_t count, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || count <= 0)
    return cudaErrorInvalidValue;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 4096 ? 4096 : blocks;
  cast_f32_to_bf16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const float*>(input),
      static_cast<__nv_bfloat16*>(output), count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_cn_cast_bf16_f32(
    const void* input, void* output, int64_t count, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || count <= 0)
    return cudaErrorInvalidValue;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 4096 ? 4096 : blocks;
  cast_bf16_to_f32_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<float*>(output), count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_cn_broadcast_bf16_f32_rows(
    const void* bias, void* output, int rows, int cols, cudaStream_t stream) {
  if (bias == nullptr || output == nullptr || rows <= 0 || cols <= 0)
    return cudaErrorInvalidValue;
  const int64_t count = static_cast<int64_t>(rows) * cols;
  const int64_t requested_blocks = (count + 255) / 256;
  const int blocks = static_cast<int>(requested_blocks > 4096 ? 4096 : requested_blocks);
  broadcast_bf16_f32_rows_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(bias), static_cast<float*>(output), count, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_cn_causal_conv1d_silu_bf16(
    const void* x, const void* weight, const void* state, void* out,
    void* new_state, int channels, int seq, int kernel_size,
    int64_t x_row_stride, cudaStream_t stream) {
  if (x == nullptr || weight == nullptr || out == nullptr ||
      new_state == nullptr || channels <= 0 || seq <= 0 || kernel_size <= 0 ||
      kernel_size > 8 || x_row_stride < channels) {
    return cudaErrorInvalidValue;
  }
  const int channel_blocks = (channels + 255) / 256;
  const int token_blocks = (seq + CONV_TOKENS - 1) / CONV_TOKENS;
  // The window a block stages: its own tokens plus the left context they share.
  const int staged_tokens = seq < CONV_TOKENS ? seq : CONV_TOKENS;
  const size_t conv_smem = static_cast<size_t>(
      (staged_tokens + kernel_size - 1) * 256) * sizeof(__nv_bfloat16);
  causal_conv1d_silu_bf16_kernel<<<dim3(token_blocks, channel_blocks), 256,
                                   conv_smem, stream>>>(
      static_cast<const __nv_bfloat16*>(x),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<const __nv_bfloat16*>(state),
      static_cast<__nv_bfloat16*>(out),
      static_cast<__nv_bfloat16*>(new_state),
      channels, seq, kernel_size, x_row_stride);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_cn_gdn_qk_prep_bf16(
    const void* conv_out, void* q_out, void* k_out,
    int seq, int seq_pad, int conv_dim, int key_dim,
    int num_v_heads, int head_k_dim, float scale, float eps, int recurrent,
    cudaStream_t stream) {
  if (conv_out == nullptr || q_out == nullptr || k_out == nullptr ||
      seq <= 0 || seq_pad < seq || conv_dim < 2 * key_dim || key_dim <= 0 ||
      num_v_heads <= 0 || head_k_dim <= 0 ||
      (head_k_dim & (head_k_dim - 1)) != 0 || !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  if (key_dim % head_k_dim != 0) return cudaErrorInvalidValue;
  const int num_k_heads = key_dim / head_k_dim;
  if (num_v_heads % num_k_heads != 0) return cudaErrorInvalidValue;
  const size_t smem = static_cast<size_t>(2 * head_k_dim) * sizeof(float);
  gdn_qk_prep_kernel_t<float><<<dim3(seq, num_k_heads), head_k_dim, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(conv_out),
      static_cast<float*>(q_out), static_cast<float*>(k_out),
      seq, seq_pad, conv_dim, key_dim, num_v_heads, head_k_dim, scale, eps, recurrent != 0);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_cn_gdn_vb_prep_bf16(
    const void* conv_out, const void* b_proj, const void* a_proj,
    const void* dt_bias, const void* a_log,
    void* v_out, void* beta_out, void* g_out,
    int seq, int seq_pad, int conv_dim, int v_offset,
    int num_v_heads, int ba_row_stride, int head_v_dim,
    cudaStream_t stream) {
  if (conv_out == nullptr || b_proj == nullptr || a_proj == nullptr ||
      dt_bias == nullptr || a_log == nullptr || v_out == nullptr ||
      beta_out == nullptr || g_out == nullptr || seq <= 0 || seq_pad < seq ||
      num_v_heads <= 0 || head_v_dim <= 0 || ba_row_stride < num_v_heads) {
    return cudaErrorInvalidValue;
  }
  // The block width has to cover both the value channels and the VB_TOKENS
  // threads that fold the per-token gate scalars.
  const int vb_threads = head_v_dim > VB_TOKENS ? head_v_dim : VB_TOKENS;
  gdn_vb_prep_kernel<<<dim3((seq + VB_TOKENS - 1) / VB_TOKENS, num_v_heads),
                       vb_threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(conv_out),
      static_cast<const __nv_bfloat16*>(b_proj),
      static_cast<const __nv_bfloat16*>(a_proj),
      static_cast<const float*>(dt_bias), static_cast<const float*>(a_log),
      static_cast<__nv_bfloat16*>(v_out), static_cast<float*>(beta_out),
      static_cast<float*>(g_out),
      seq, seq_pad, conv_dim, v_offset, ba_row_stride, head_v_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_cn_gdn_cumsum_f32(
    const void* g, void* g_cum, int seq_pad, int num_v_heads, int chunk_size,
    cudaStream_t stream) {
  if (g == nullptr || g_cum == nullptr || seq_pad <= 0 ||
      num_v_heads <= 0 || chunk_size <= 0 || seq_pad % chunk_size != 0) {
    return cudaErrorInvalidValue;
  }
  const int chunks = seq_pad / chunk_size;
  gdn_cumsum_kernel<<<dim3(chunks, num_v_heads), 32, 0, stream>>>(
      static_cast<const float*>(g), static_cast<float*>(g_cum),
      seq_pad, chunk_size);
  return cudaGetLastError();
}


extern "C" cudaError_t apxinf_cn_gdn_attn_raw_f32(
    const void* q, const void* k, const void* beta, const void* g_cum,
    void* a_out, void* t_out, int seq_pad, int num_v_heads, int head_k_dim,
    int chunk_size, const ApxinfGdnPolicy* policy, cudaStream_t stream) {
  if (policy == nullptr) return cudaErrorInvalidValue;
  if (q == nullptr || k == nullptr || beta == nullptr || g_cum == nullptr ||
      a_out == nullptr || t_out == nullptr || seq_pad <= 0 ||
      num_v_heads <= 0 || head_k_dim <= 0 || chunk_size <= 0 ||
      seq_pad % chunk_size != 0) {
    return cudaErrorInvalidValue;
  }
  const int chunks = seq_pad / chunk_size;
  // One K tile and one Q tile for the chunk, rows padded by one float to keep
  // the warp off a single shared-memory bank (see the kernel comment).
  const size_t attn_smem =
      2u * static_cast<size_t>(chunk_size) * (head_k_dim + 1) * sizeof(float);
  if (attn_smem > 48u * 1024u) {
    static bool attn_opted_in = false;
    if (!attn_opted_in) {
      const cudaError_t attr = cudaFuncSetAttribute(
          reinterpret_cast<const void*>(gdn_attn_raw_kernel),
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(attn_smem));
      if (attr != cudaSuccess) {
        return attr;
      }
      attn_opted_in = true;
    }
  }
  gdn_attn_raw_kernel<<<dim3(chunks, num_v_heads), 256, attn_smem, stream>>>(
      static_cast<const float*>(q), static_cast<const float*>(k),
      static_cast<const float*>(beta), static_cast<const float*>(g_cum),
      static_cast<float*>(a_out), static_cast<__nv_bfloat16*>(t_out),
      seq_pad, head_k_dim, chunk_size);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_cn_gdn_tri_solve_f32(
    void* a, int matrices, int chunk_size, cudaStream_t stream) {
  if (a == nullptr || matrices <= 0 || chunk_size <= 0 || chunk_size > 128) {
    return cudaErrorInvalidValue;
  }
  if (chunk_size == 64) {
    gdn_block_inverse64_kernel<<<matrices, 256, 0, stream>>>(static_cast<float*>(a));
    return cudaGetLastError();
  }
  const size_t smem =
      (static_cast<size_t>(chunk_size) * chunk_size + chunk_size) * sizeof(float);
  gdn_tri_solve_kernel<<<matrices, 64, smem, stream>>>(
      static_cast<float*>(a), chunk_size);
  return cudaGetLastError();
}



#define CHUNK_GEMM_ARGS                                                    \
  chunks, num_v_heads, gemm_smem, stream, a, v, k, beta, g_cum, vt_out,    \
      kcd_out, seq_pad, head_k_dim, head_v_dim, chunk_size

template <int TILE>
static cudaError_t launch_chunk_gemm(
    int chunks, int num_v_heads, size_t gemm_smem, cudaStream_t stream,
    const void* a, const void* v, const void* k, const void* beta,
    const void* g_cum, void* vt_out, void* kcd_out, int seq_pad,
    int head_k_dim, int head_v_dim, int chunk_size) {
  if (gemm_smem > 48u * 1024u) {
    static bool gemm_opted_in = false;
    if (!gemm_opted_in) {
      const cudaError_t attr = cudaFuncSetAttribute(
          reinterpret_cast<const void*>(gdn_chunk_gemm_kernel<TILE>),
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(gemm_smem));
      if (attr != cudaSuccess) {
        return attr;
      }
      gemm_opted_in = true;
    }
  }
  gdn_chunk_gemm_kernel<TILE><<<dim3(chunks, num_v_heads), 256, gemm_smem, stream>>>(
      static_cast<const float*>(a), static_cast<const __nv_bfloat16*>(v),
      static_cast<const float*>(k), static_cast<const float*>(beta),
      static_cast<const float*>(g_cum),
      static_cast<__nv_bfloat16*>(vt_out), static_cast<__nv_bfloat16*>(kcd_out),
      seq_pad, head_k_dim, head_v_dim, chunk_size);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_cn_gdn_chunk_gemm_f32(
    const void* a, const void* v, const void* k, const void* beta,
    const void* g_cum, void* vt_out, void* kcd_out,
    int seq_pad, int num_v_heads, int head_k_dim, int head_v_dim,
    int chunk_size, const ApxinfGdnPolicy* policy, cudaStream_t stream) {
  if (policy == nullptr) return cudaErrorInvalidValue;
  if (a == nullptr || v == nullptr || k == nullptr || beta == nullptr ||
      g_cum == nullptr || vt_out == nullptr || kcd_out == nullptr ||
      seq_pad <= 0 || num_v_heads <= 0 || head_k_dim <= 0 ||
      head_v_dim <= 0 || chunk_size <= 0 || seq_pad % chunk_size != 0) {
    return cudaErrorInvalidValue;
  }
  const int chunks = seq_pad / chunk_size;
  // vb and kb tiles, precomputed once per chunk and held in BF16 because every
  // value in them is already on the BF16 grid (see the kernel comment). 32KB at
  // the shipped shape rather than 64KB.
  const size_t gemm_smem = static_cast<size_t>(chunk_size) *
                           (head_v_dim + head_k_dim) * sizeof(__nv_bfloat16);
  if (head_k_dim == 128 && head_v_dim == 128 && chunk_size == 64 &&
      policy->chunk_gemm_tile == 4) {
    constexpr size_t tri_smem = 64 * 256 * sizeof(__nv_bfloat16) +
                                64 * 64 * sizeof(float);
    static thread_local int tri_opted_device = -1;
    int device = -1;
    const cudaError_t device_status = cudaGetDevice(&device);
    if (device_status != cudaSuccess) return device_status;
    if (tri_opted_device != device) {
      const cudaError_t attr = cudaFuncSetAttribute(
          reinterpret_cast<const void*>(gdn_chunk_gemm_tri_kernel<true, true, true>),
          cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(tri_smem));
      if (attr != cudaSuccess) return attr;
      tri_opted_device = device;
    }
    gdn_chunk_gemm_tri_kernel<true, true, true>
        <<<dim3(chunks, num_v_heads), 256, tri_smem, stream>>>(
            static_cast<const float*>(a), static_cast<const __nv_bfloat16*>(v),
            static_cast<const float*>(k), static_cast<const float*>(beta),
            static_cast<const float*>(g_cum),
            static_cast<__nv_bfloat16*>(vt_out), static_cast<__nv_bfloat16*>(kcd_out),
            seq_pad);
    return cudaGetLastError();
  }
  switch (policy->chunk_gemm_tile) {
    case 1: return launch_chunk_gemm<1>(CHUNK_GEMM_ARGS);
    case 2: return launch_chunk_gemm<2>(CHUNK_GEMM_ARGS);
    case 4: return launch_chunk_gemm<4>(CHUNK_GEMM_ARGS);
    case 8: return launch_chunk_gemm<8>(CHUNK_GEMM_ARGS);
    case 32: return launch_chunk_gemm<32>(CHUNK_GEMM_ARGS);
    default: return launch_chunk_gemm<16>(CHUNK_GEMM_ARGS);
  }
}



#define CHUNK_STATE_ARGS                                                      \
  num_v_heads, v_split, block_threads, smem, stream, q, k, g_cum, t_in,       \
      vt_in, kcd_in, state, out, seq, seq_pad, head_k_dim, head_v_dim,        \
      chunk_size, total_chunks, out_row_width

template <int TILE, bool V_SPLIT>
static cudaError_t launch_chunk_state(
    int num_v_heads, int v_split, int block_threads, size_t smem,
    cudaStream_t stream,
    const void* q, const void* k, const void* g_cum, const void* t_in,
    const void* vt_in, const void* kcd_in, void* state, void* out,
    int seq, int seq_pad, int head_k_dim, int head_v_dim, int chunk_size,
    int total_chunks, int out_row_width) {
  if (smem > 48u * 1024u) {
    // Per instantiation, and each instantiation is its own function.
    static bool opted_in = false;
    if (!opted_in) {
      const cudaError_t attr = cudaFuncSetAttribute(
          reinterpret_cast<const void*>(gdn_chunk_state_kernel<TILE, V_SPLIT>),
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(smem));
      if (attr != cudaSuccess) {
        return attr;
      }
      opted_in = true;
    }
  }
  gdn_chunk_state_kernel<TILE, V_SPLIT>
      <<<dim3(num_v_heads, v_split), block_threads, smem, stream>>>(
          static_cast<const float*>(q), static_cast<const float*>(k),
          static_cast<const float*>(g_cum),
          static_cast<const __nv_bfloat16*>(t_in),
          static_cast<const __nv_bfloat16*>(vt_in),
          static_cast<const __nv_bfloat16*>(kcd_in),
          static_cast<float*>(state), static_cast<__nv_bfloat16*>(out),
          seq, seq_pad, head_k_dim, head_v_dim, chunk_size, total_chunks,
          out_row_width,
          static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_k_dim))),
          v_split);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_cn_gdn_chunk_state_f32(
    const void* q, const void* k, const void* g_cum, const void* t_in,
    const void* vt_in, const void* kcd_in, void* state, void* out,
    int seq, int seq_pad, int num_v_heads, int head_k_dim, int head_v_dim,
    int chunk_size, int total_chunks, int out_row_width,
    const ApxinfGdnPolicy* policy, cudaStream_t stream) {
  if (policy == nullptr) return cudaErrorInvalidValue;
  if (q == nullptr || k == nullptr || g_cum == nullptr || t_in == nullptr ||
      vt_in == nullptr || kcd_in == nullptr || state == nullptr ||
      out == nullptr || seq <= 0 || seq_pad < seq || num_v_heads <= 0 ||
      head_k_dim <= 0 || head_v_dim <= 0 || chunk_size <= 0 ||
      total_chunks <= 0 || seq_pad != total_chunks * chunk_size ||
      out_row_width != num_v_heads * head_v_dim) {
    return cudaErrorInvalidValue;
  }
  // The chunk loop is sequential because the state is recurrent, so the only
  // parallelism is inside a chunk: chunk_size*head_v_dim cells against one
  // block per head. At the shipped shape that is 32 blocks of 256 threads on
  // 16 SMs, a third of the threads an SM can hold. A wider block raises
  // occupancy without touching the arithmetic -- each cell is independent and
  // keeps its own accumulation order. Must stay a multiple of head_v_dim so a
  // thread's column index remains fixed (see the kernel comment), and no
  // larger than the per-thread attn_inter budget allows.
  //
  // A head's scan may also be split across v_split blocks along the value
  // dimension, which changes no arithmetic (see the kernel) but multiplies the
  // grid. The policy asks for a width from the device's multiprocessor count;
  // the shape decides what it can have, and a slice that would leave partial
  // warps of columns falls back towards one block per head.
  int v_split = policy->chunk_state_v_split > 0 ? policy->chunk_state_v_split : 1;
  while (v_split > 1 &&
         (head_v_dim % v_split != 0 || (head_v_dim / v_split) % 32 != 0)) {
    v_split /= 2;
  }
  const int v_cols = head_v_dim / v_split;
  const int cells_per_chunk = chunk_size * v_cols;
  auto block_ok = [&](int threads) {
    return threads >= 32 && threads <= 1024 && threads % 32 == 0 &&
           v_cols % 32 == 0 &&
           cells_per_chunk % threads == 0 && cells_per_chunk / threads <= 32 &&
           // Keep the tiled path reachable: with a trip count the accumulator
           // array is indexed dynamically and nvcc spills it, which costs more
           // than the wider block wins. The tiles run along the columns, so
           // the slice has to hold a whole number of them.
           v_cols % policy->chunk_state_tile == 0 &&
           cells_per_chunk % (threads * policy->chunk_state_tile) == 0;
  };
  // 1024. That is not what an earlier sweep found -- on the CUDA 12.6 board,
  // before this kernel was tiled, 512 won at 33.27ms per layer against 36.35ms
  // at 1024, and the reasoning was that two 1024-thread blocks exceed the 1536
  // threads an SM holds. The tiling changed which side of that trade wins: with
  // eight cells carried per thread there is enough work in flight for one block
  // per SM to keep the pipes busy, and the wider block cuts the per-block
  // prologue. Re-measured on orin2, four interleaved pairs, every 1024 sample
  // faster than every 512 sample: 5.9319 s/scene mean against 5.9564, 0.41%.
  //
  // Re-sweep after changing this kernel; the optimum has moved once already
  // and will again. The width comes from the policy table, and anything the
  // shape cannot take falls back rather than launching something invalid --
  // 1024 exceeds the register budget for this kernel on some boards.
  int block_threads = policy->chunk_state_threads;
  if (!block_ok(block_threads)) {
    block_threads = 256;
    for (const int candidate : {1024, 512, 256, 128, 64, 32}) {
      if (block_ok(candidate)) {
        block_threads = candidate;
        break;
      }
    }
  }
  // v_new, a BF16 copy of the carried state, and a BF16 tile holding v_new's
  // round trip for the chunk (see the kernel comment). 32KB + 32KB + 16KB at
  // the shipped shape, so two blocks still fit in an SM's 164KB.
  const size_t smem =
      static_cast<size_t>(chunk_size) * v_cols * sizeof(float) +
      static_cast<size_t>(head_k_dim) * v_cols * sizeof(__nv_bfloat16) +
      static_cast<size_t>(chunk_size) * v_cols * sizeof(__nv_bfloat16);
  // At the shipped shape this lands at 80KB, past the 48KB a kernel receives
  // without asking. Opt in once per instantiation; a device that refuses keeps
  // the error rather than launching with too little shared memory.
  // Tensor-core form, for the shipped shape only. Every operand reaching this
  // scan on the prefill path was already rounded to BF16 by its producer, so
  // the one pass is exact here; on Thor it takes the fixed cost from 1.4623 to
  // 1.1909 s.
  if (policy->chunk_state_wmma != APXINF_GDN_WMMA_OFF && head_k_dim == 128 &&
      head_v_dim == 128 && chunk_size == 64) {
    // Leading dimensions are padded by 8 elements so the sixteen rows of a wmma
    // tile do not all start on the same shared-memory bank; see the kernel for
    // the bank arithmetic. Keep these strides identical to the kernel's.
    constexpr int GDN_KP = 128 + 8;
    constexpr int GDN_VP = 128 + 8;
    const size_t wmma_smem =
        static_cast<size_t>(128 * GDN_VP) * sizeof(__nv_bfloat16) +  // state
        // Keep this in step with LHS_ELEMS in the kernel.
        static_cast<size_t>(2 * 64 * GDN_KP) * sizeof(__nv_bfloat16) +
        static_cast<size_t>(64 * GDN_VP) * sizeof(__nv_bfloat16) +   // v_round
        static_cast<size_t>(64 * GDN_VP) * sizeof(float) * 2;        // v_new, inter/out
    const void* entry =
        reinterpret_cast<const void*>(gdn_chunk_state_wmma_kernel<float>);
    static const void* opted_in = nullptr;
    if (opted_in != entry) {
      const cudaError_t attr = cudaFuncSetAttribute(
          entry, cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(wmma_smem));
      if (attr != cudaSuccess) {
        return attr;
      }
      opted_in = entry;
    }
    const float chunk_scale =
        static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_k_dim)));
    gdn_chunk_state_wmma_kernel<float>
        <<<num_v_heads, 1024, wmma_smem, stream>>>(
            static_cast<const float*>(q), static_cast<const float*>(k),
            static_cast<const float*>(g_cum),
            static_cast<const __nv_bfloat16*>(t_in),
            static_cast<const __nv_bfloat16*>(vt_in),
            static_cast<const __nv_bfloat16*>(kcd_in),
            static_cast<float*>(state), static_cast<__nv_bfloat16*>(out), seq,
            seq_pad, total_chunks, out_row_width, chunk_scale);
    return cudaGetLastError();
  }

  const int tile = policy->chunk_state_tile;
  cudaError_t launched = cudaErrorInvalidValue;
  // One block per head compiles to the kernel as it was before the split
  // existed; see the V_SPLIT comment in the kernel for why that matters.
  if (v_split == 1) {
    switch (tile) {
      case 1: launched = launch_chunk_state<1, false>(CHUNK_STATE_ARGS); break;
      case 2: launched = launch_chunk_state<2, false>(CHUNK_STATE_ARGS); break;
      case 4: launched = launch_chunk_state<4, false>(CHUNK_STATE_ARGS); break;
      case 8: launched = launch_chunk_state<8, false>(CHUNK_STATE_ARGS); break;
      default: launched = launch_chunk_state<16, false>(CHUNK_STATE_ARGS); break;
    }
  } else {
    switch (tile) {
      case 1: launched = launch_chunk_state<1, true>(CHUNK_STATE_ARGS); break;
      case 2: launched = launch_chunk_state<2, true>(CHUNK_STATE_ARGS); break;
      case 4: launched = launch_chunk_state<4, true>(CHUNK_STATE_ARGS); break;
      case 8: launched = launch_chunk_state<8, true>(CHUNK_STATE_ARGS); break;
      default: launched = launch_chunk_state<16, true>(CHUNK_STATE_ARGS); break;
    }
  }
  return launched;
}


extern "C" cudaError_t apxinf_cn_gdn_recurrent_f32(
    const void* q, const void* k, const void* v, const void* beta,
    const void* g, void* state, void* out,
    int num_v_heads, int head_k_dim, int head_v_dim,
    const ApxinfGdnPolicy* policy, cudaStream_t stream) {
  if (policy == nullptr) return cudaErrorInvalidValue;
  if (q == nullptr || k == nullptr || v == nullptr || beta == nullptr ||
      g == nullptr || state == nullptr || out == nullptr ||
      num_v_heads <= 0 || head_k_dim <= 0 || head_v_dim <= 0) {
    return cudaErrorInvalidValue;
  }
  // Split width for the decode recurrence. The scalar kernel is SPLIT=1 and
  // stays available for any shape the split form is not instantiated for, and
  // for A/B on one binary: APXINF_GDN_RECURRENT_SPLIT=1 selects it.
  // 8 would ask for 1024 threads each holding 16 registers of state and the
  // launcher rejects it with "too many resources requested", so the policy
  // does not offer it.
  const int split = policy->recurrent_split;
  if (split > 1 && head_k_dim == 128) {
    const size_t smem =
        (static_cast<size_t>(2 * head_k_dim) +
         static_cast<size_t>(head_v_dim) * split) * sizeof(float);
    const dim3 block(static_cast<unsigned>(head_v_dim * split));
    if (block.x > 1024u) return cudaErrorInvalidValue;
#define GDN_SPLIT_ARGS                                                        \
  static_cast<const float*>(q), static_cast<const float*>(k),                 \
      static_cast<const __nv_bfloat16*>(v), static_cast<const float*>(beta),          \
      static_cast<const float*>(g), static_cast<float*>(state),               \
      static_cast<__nv_bfloat16*>(out), head_k_dim, head_v_dim
    switch (split) {
      case 2:
        gdn_recurrent_split_kernel<2, 128><<<num_v_heads, block, smem, stream>>>(GDN_SPLIT_ARGS);
        return cudaGetLastError();
      case 4:
        gdn_recurrent_split_kernel<4, 128><<<num_v_heads, block, smem, stream>>>(GDN_SPLIT_ARGS);
        return cudaGetLastError();
      default:
        break;
    }
#undef GDN_SPLIT_ARGS
  }
  const size_t smem = static_cast<size_t>(2 * head_k_dim) * sizeof(float);
  gdn_recurrent_kernel<<<num_v_heads, head_v_dim, smem, stream>>>(
      static_cast<const float*>(q), static_cast<const float*>(k),
      static_cast<const __nv_bfloat16*>(v), static_cast<const float*>(beta),
      static_cast<const float*>(g), static_cast<float*>(state),
      static_cast<__nv_bfloat16*>(out), head_k_dim, head_v_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_cn_gated_rms_silu_bf16(
    const void* x, const void* z, const void* weight, void* out,
    int rows, int cols, int z_heads, int64_t z_row_stride,
    int64_t z_col_offset, float eps, cudaStream_t stream) {
  if (x == nullptr || z == nullptr || weight == nullptr || out == nullptr ||
      rows <= 0 || cols <= 0 || z_heads <= 0 || rows % z_heads != 0 ||
      !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  if (cols == 128) {
    gated_rms_silu_bf16_warp4_kernel<<<(rows + GRS_ROWS - 1) / GRS_ROWS, 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<const __nv_bfloat16*>(z),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_bfloat16*>(out),
        rows, cols, z_heads, z_row_stride, z_col_offset, eps);
    return cudaGetLastError();
  }
  const size_t smem = static_cast<size_t>(cols) * sizeof(float);
  gated_rms_silu_bf16_kernel<<<rows, 128, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(x),
      static_cast<const __nv_bfloat16*>(z),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(out),
      cols, z_heads, z_row_stride, z_col_offset, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_cn_rms_norm_plus1_bf16(
    const void* input, const void* weight, void* output,
    int rows, int cols, float eps, cudaStream_t stream) {
  if (input == nullptr || weight == nullptr || output == nullptr ||
      rows <= 0 || cols <= 0 || !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  const size_t smem = static_cast<size_t>(cols) * sizeof(float);
  rms_norm_plus1_bf16_kernel<<<rows, 256, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(output), cols, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_cn_add_rms_norm_plus1_bf16(
    const void* a, const void* b, const void* weight, void* sum_out,
    void* output, int rows, int cols, float eps, cudaStream_t stream) {
  if (a == nullptr || b == nullptr || weight == nullptr || sum_out == nullptr ||
      output == nullptr || rows <= 0 || cols <= 0 || !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  const size_t smem = static_cast<size_t>(cols) * sizeof(float);
  // The tested prefill shape uses only the exact vector mean branch.
  if (rows >= 16 && cols == 2560) {
    add_rms_norm_plus1_bf16_kernel<true><<<rows, 256, smem, stream>>>(
        static_cast<const __nv_bfloat16*>(a),
        static_cast<const __nv_bfloat16*>(b),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_bfloat16*>(sum_out),
        static_cast<__nv_bfloat16*>(output), cols, eps);
    return cudaGetLastError();
  }
  add_rms_norm_plus1_bf16_kernel<false><<<rows, 256, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(a),
      static_cast<const __nv_bfloat16*>(b),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(sum_out),
      static_cast<__nv_bfloat16*>(output), cols, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_cn_full_attn_prepare_bf16(
    const void* fused, const void* q_norm_w, const void* k_norm_w,
    const void* cos, const void* sin, void* q_out, void* k_cache, void* v_cache,
    int seq, int cache_offset, int q_heads, int kv_heads, int head_dim,
    int rotary_dim, int64_t fused_width, int64_t cache_width, float eps,
    cudaStream_t stream) {
  if (fused == nullptr || q_norm_w == nullptr || k_norm_w == nullptr ||
      cos == nullptr || sin == nullptr || q_out == nullptr ||
      k_cache == nullptr || v_cache == nullptr || seq <= 0 ||
      cache_offset < 0 || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
      rotary_dim <= 0 || (rotary_dim & 1) != 0 || rotary_dim > head_dim ||
      !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  // Opt-in only for the measured D256 BF16 prefill geometry. Keep the
  // original C ABI and generic route for all other inputs and decode.
  if (seq > 1 && q_heads == 16 && kv_heads == 4 && head_dim == 256 &&
      rotary_dim == 64 && fused_width == 10240 && cache_width == 1024) {
    full_attn_prepare_bf16_warp_kernel
        <<<dim3(seq, 6), 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(fused),
        static_cast<const __nv_bfloat16*>(q_norm_w),
        static_cast<const __nv_bfloat16*>(k_norm_w),
        static_cast<const __nv_bfloat16*>(cos),
        static_cast<const __nv_bfloat16*>(sin),
        static_cast<__nv_bfloat16*>(q_out),
        static_cast<__nv_bfloat16*>(k_cache),
        static_cast<__nv_bfloat16*>(v_cache),
        cache_offset, q_heads, kv_heads, fused_width, cache_width, eps);
    return cudaGetLastError();
  }
  const size_t smem = static_cast<size_t>(head_dim) * sizeof(float);
  full_attn_prepare_bf16_kernel
      <<<dim3(seq, q_heads + 2 * kv_heads), 128, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(fused),
      static_cast<const __nv_bfloat16*>(q_norm_w),
      static_cast<const __nv_bfloat16*>(k_norm_w),
      static_cast<const __nv_bfloat16*>(cos),
      static_cast<const __nv_bfloat16*>(sin),
      static_cast<__nv_bfloat16*>(q_out),
      static_cast<__nv_bfloat16*>(k_cache),
      static_cast<__nv_bfloat16*>(v_cache),
      cache_offset, q_heads, kv_heads, head_dim, rotary_dim, fused_width,
      cache_width, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_cn_sigmoid_gate_mul_bf16(
    void* attn, const void* fused, int rows, int heads, int head_dim,
    int64_t fused_width, cudaStream_t stream) {
  if (attn == nullptr || fused == nullptr || rows <= 0 || heads <= 0 ||
      head_dim <= 0) {
    return cudaErrorInvalidValue;
  }
  // Vector BF16 route for D256 prefill. Inputs must be aligned and disjoint
  // because the vector kernel promises restrict; retain the generic fallback.
  if (rows > 1 && rows <= 65536 && heads == 16 && head_dim == 256 &&
      fused_width == 10240) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(attn);
    const uintptr_t f = reinterpret_cast<uintptr_t>(fused);
    const uint64_t a_bytes = static_cast<uint64_t>(rows) * 4096 * 2;
    const uint64_t f_bytes = static_cast<uint64_t>(rows) * 10240 * 2;
    const bool aligned = ((a | f) & 15U) == 0;
    const bool no_overflow = a <= UINTPTR_MAX - a_bytes && f <= UINTPTR_MAX - f_bytes;
    const bool disjoint = no_overflow && (a + a_bytes <= f || f + f_bytes <= a);
    if (aligned && disjoint) {
      sigmoid_gate_mul_bf16_vec8_kernel<<<rows * 2, 256, 0, stream>>>(
          static_cast<__nv_bfloat16*>(attn),
          static_cast<const __nv_bfloat16*>(fused), rows);
      return cudaGetLastError();
    }
  }
  sigmoid_gate_mul_bf16_kernel<<<rows, 256, 0, stream>>>(
      static_cast<__nv_bfloat16*>(attn),
      static_cast<const __nv_bfloat16*>(fused), heads, head_dim, fused_width);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_cn_adaln_rms_norm_bf16(
    const void* x, const void* weight, const void* scale, const void* shift,
    void* out, int rows, int cols, float eps, cudaStream_t stream) {
  if (x == nullptr || weight == nullptr || scale == nullptr ||
      shift == nullptr || out == nullptr || rows <= 0 || cols <= 0 ||
      !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  const size_t smem = static_cast<size_t>(cols) * sizeof(float);
  adaln_rms_norm_bf16_kernel<<<rows, 256, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(x),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<const __nv_bfloat16*>(scale),
      static_cast<const __nv_bfloat16*>(shift),
      static_cast<__nv_bfloat16*>(out), cols, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_cn_adaln_gate_residual_bf16(
    const void* proj, const void* residual, const void* gate, void* out,
    int64_t count, int cols, cudaStream_t stream) {
  if (proj == nullptr || residual == nullptr || gate == nullptr ||
      out == nullptr || count <= 0 || cols <= 0) {
    return cudaErrorInvalidValue;
  }
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  adaln_gate_residual_bf16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(proj),
      static_cast<const __nv_bfloat16*>(residual),
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<__nv_bfloat16*>(out), count, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_cn_expert_qkv_prepare_bf16(
    const void* fused, const void* q_norm_w, const void* k_norm_w,
    const void* cos, const void* sin, void* q_out, void* gate_out, void* k_out,
    void* v_out, int seq, int q_heads, int kv_heads, int head_dim,
    int rotary_dim, int64_t fused_width, float eps, cudaStream_t stream) {
  if (fused == nullptr || q_norm_w == nullptr || k_norm_w == nullptr ||
      cos == nullptr || sin == nullptr || q_out == nullptr ||
      gate_out == nullptr || k_out == nullptr || v_out == nullptr ||
      seq <= 0 || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
      rotary_dim <= 0 || (rotary_dim & 1) != 0 || rotary_dim > head_dim ||
      q_heads % kv_heads != 0 || !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  const size_t smem = static_cast<size_t>(head_dim) * sizeof(float);
  expert_qkv_prepare_bf16_kernel
      <<<dim3(seq, q_heads + 2 * kv_heads), 128, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(fused),
      static_cast<const __nv_bfloat16*>(q_norm_w),
      static_cast<const __nv_bfloat16*>(k_norm_w),
      static_cast<const __nv_bfloat16*>(cos),
      static_cast<const __nv_bfloat16*>(sin),
      static_cast<__nv_bfloat16*>(q_out),
      static_cast<__nv_bfloat16*>(gate_out),
      static_cast<__nv_bfloat16*>(k_out),
      static_cast<__nv_bfloat16*>(v_out),
      q_heads, kv_heads, head_dim, rotary_dim, fused_width, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_cn_expert_sigmoid_gate_mul_bf16(
    void* attn, const void* gate, int64_t count, cudaStream_t stream) {
  if (attn == nullptr || gate == nullptr || count <= 0) {
    return cudaErrorInvalidValue;
  }
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  expert_sigmoid_gate_mul_bf16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<__nv_bfloat16*>(attn),
      static_cast<const __nv_bfloat16*>(gate), count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_cn_fourier_features_bf16(
    const void* waypoints, const void* freqs, void* out,
    int rows, int point_dim, int num_features, cudaStream_t stream) {
  if (waypoints == nullptr || freqs == nullptr || out == nullptr ||
      rows <= 0 || point_dim <= 0 || num_features <= 0) {
    return cudaErrorInvalidValue;
  }
  fourier_features_bf16_kernel<<<rows, 64, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(waypoints),
      static_cast<const __nv_bfloat16*>(freqs),
      static_cast<__nv_bfloat16*>(out), point_dim, num_features);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_cn_concat7_cols_bf16(
    const void* s0, const void* s1, const void* s2, const void* s3,
    const void* s4, const void* s5, const void* s6, void* dst,
    int rows, int cols, int broadcast_mask, cudaStream_t stream) {
  if (s0 == nullptr || s1 == nullptr || s2 == nullptr || s3 == nullptr ||
      s4 == nullptr || s5 == nullptr || s6 == nullptr || dst == nullptr ||
      rows <= 0 || cols <= 0) {
    return cudaErrorInvalidValue;
  }
  concat7_cols_bf16_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(s0),
      static_cast<const __nv_bfloat16*>(s1),
      static_cast<const __nv_bfloat16*>(s2),
      static_cast<const __nv_bfloat16*>(s3),
      static_cast<const __nv_bfloat16*>(s4),
      static_cast<const __nv_bfloat16*>(s5),
      static_cast<const __nv_bfloat16*>(s6),
      static_cast<__nv_bfloat16*>(dst), rows, cols, broadcast_mask);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_cn_flow_update_f32(
    void* w, const void* endpoint, float remaining, float step, int64_t count,
    cudaStream_t stream) {
  if (w == nullptr || endpoint == nullptr || count <= 0 ||
      !(remaining > 0.0f) || !std::isfinite(step)) {
    return cudaErrorInvalidValue;
  }
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  flow_update_f32_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<float*>(w), static_cast<const float*>(endpoint),
      remaining, step, count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_cn_suppress_logits_bf16(
    void* logits, const uint32_t* ids, int count, cudaStream_t stream) {
  if (logits == nullptr || ids == nullptr || count <= 0) {
    return cudaErrorInvalidValue;
  }
  const int blocks = (count + 63) / 64;
  suppress_logits_bf16_kernel<<<blocks, 64, 0, stream>>>(
      static_cast<__nv_bfloat16*>(logits), ids, count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_cn_gelu_exact_bf16(
    const void* input, void* output, int64_t count, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || count <= 0) {
    return cudaErrorInvalidValue;
  }
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  gelu_exact_bf16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_bfloat16*>(output), count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_cn_adaln_gate_residual_rms_bf16(
    const void* proj, const void* residual, const void* gate, const void* weight,
    const void* scale, const void* shift, void* hidden, void* normalized,
    int rows, int cols, float eps, cudaStream_t stream) {
  if (!proj || !residual || !gate || !weight || !scale || !shift || !hidden || !normalized ||
      rows <= 0 || cols <= 0 || cols > 8192 || !(eps > 0.0f)) return cudaErrorInvalidValue;
  adaln_gate_residual_rms_bf16_kernel<<<rows, 256, cols * sizeof(float), stream>>>(
      static_cast<const __nv_bfloat16*>(proj), static_cast<const __nv_bfloat16*>(residual),
      static_cast<const __nv_bfloat16*>(gate), static_cast<const __nv_bfloat16*>(weight),
      static_cast<const __nv_bfloat16*>(scale), static_cast<const __nv_bfloat16*>(shift),
      static_cast<__nv_bfloat16*>(hidden), static_cast<__nv_bfloat16*>(normalized), cols, eps);
  return cudaGetLastError();
}

// Isolated BF16-physical Q/K prefill route. The four entry points reject any
// policy/shape that would need a legacy FP32 consumer.
extern "C" cudaError_t apxinf_cn_gdn_qk_prep_qk_bf16(
    const void* conv_out, void* q_out, void* k_out,
    int seq, int seq_pad, int conv_dim, int key_dim,
    int num_v_heads, int head_k_dim, float scale, float eps, int recurrent,
    cudaStream_t stream) {
  if (!conv_out || !q_out || !k_out || seq <= 0 || seq_pad < seq ||
      conv_dim < 2 * key_dim || key_dim <= 0 || num_v_heads <= 0 ||
      head_k_dim != 128 || (head_k_dim & (head_k_dim - 1)) || !(eps > 0.0f) ||
      recurrent || key_dim % head_k_dim) return cudaErrorInvalidValue;
  const int num_k_heads = key_dim / head_k_dim;
  if (num_v_heads % num_k_heads) return cudaErrorInvalidValue;
  const size_t smem = static_cast<size_t>(2 * head_k_dim) * sizeof(float);
  gdn_qk_prep_kernel_t<__nv_bfloat16>
      <<<dim3(seq, num_k_heads), head_k_dim, smem, stream>>>(
          static_cast<const __nv_bfloat16*>(conv_out),
          static_cast<__nv_bfloat16*>(q_out), static_cast<__nv_bfloat16*>(k_out),
          seq, seq_pad, conv_dim, key_dim, num_v_heads, head_k_dim, scale, eps,
          false);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_cn_gdn_attn_raw_solve_f1_qk_bf16(
    const void* q, const void* k, const void* beta, const void* g_cum,
    void* a_out, void* t_out, int seq_pad, int num_v_heads, int head_k_dim,
    int chunk_size, cudaStream_t stream) {
  if (!q || !k || !beta || !g_cum || !a_out || !t_out ||
      seq_pad <= 0 || num_v_heads <= 0 || head_k_dim != 128 ||
      chunk_size != 64 || seq_pad % 64) return cudaErrorInvalidValue;
  // BF16 Q/K staging fits below the inverse's FP32 raw/inv/scratch stage.
  constexpr int smem = (2 * 64 * 64 + 3 * 256) * sizeof(float);
  static thread_local int opted_device = -1;
  int device = -1;
  const cudaError_t device_status = cudaGetDevice(&device);
  if (device_status != cudaSuccess) return device_status;
  if (opted_device != device) {
    const cudaError_t attr = cudaFuncSetAttribute(
        reinterpret_cast<const void*>(gdn_raw_inverse_f1_t<__nv_bfloat16>),
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (attr != cudaSuccess) return attr;
    opted_device = device;
  }
  gdn_raw_inverse_f1_t<__nv_bfloat16>
      <<<dim3(seq_pad / 64, num_v_heads), 256, smem, stream>>>(
          static_cast<const __nv_bfloat16*>(q),
          static_cast<const __nv_bfloat16*>(k),
          static_cast<const float*>(beta), static_cast<const float*>(g_cum),
          static_cast<float*>(a_out), static_cast<__nv_bfloat16*>(t_out), seq_pad);
  return cudaGetLastError();
}

// Experimental BF16 Q/K dot reassociation; inverse arithmetic is unchanged.
extern "C" cudaError_t apxinf_cn_gdn_gate_prep_bf16(
    const void* b_proj, const void* a_proj, const void* dt_bias,
    const void* a_log, void* beta_out, void* g_out,
    int seq, int seq_pad, int num_v_heads, int ba_row_stride,
    cudaStream_t stream) {
  if (!b_proj || !a_proj || !dt_bias || !a_log || !beta_out || !g_out ||
      seq <= 0 || seq_pad < seq || num_v_heads <= 0 ||
      ba_row_stride < num_v_heads) return cudaErrorInvalidValue;
  gdn_gate_only_kernel<<<dim3((seq - 1) / VB_TOKENS + 1, num_v_heads),
                         128, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(b_proj),
      static_cast<const __nv_bfloat16*>(a_proj),
      static_cast<const float*>(dt_bias), static_cast<const float*>(a_log),
      static_cast<float*>(beta_out), static_cast<float*>(g_out),
      seq, seq_pad, ba_row_stride);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_cn_gdn_chunk_gemm_tri_k_bf16_direct_v(
    const void* a, const void* conv_out, const void* k, const void* beta,
    const void* g_cum, void* vt_out, void* kcd_out,
    int seq, int seq_pad, int conv_dim, int v_offset, int num_v_heads,
    int head_k_dim, int head_v_dim, int chunk_size,
    cudaStream_t stream) {
  if (!a || !conv_out || !k || !beta || !g_cum || !vt_out ||
      !kcd_out || seq <= 0 || seq_pad < seq || seq_pad % 64 ||
      num_v_heads <= 0 || head_k_dim != 128 || head_v_dim != 128 ||
      chunk_size != 64 || v_offset < 0 ||
      static_cast<int64_t>(v_offset) + static_cast<int64_t>(num_v_heads) * 128 > conv_dim) return cudaErrorInvalidValue;
  constexpr size_t smem = 64 * 256 * sizeof(__nv_bfloat16) +
                          64 * 64 * sizeof(float);
  static thread_local int opted_device = -1;
  int device = -1;
  const cudaError_t device_status = cudaGetDevice(&device);
  if (device_status != cudaSuccess) return device_status;
  if (opted_device != device) {
    const cudaError_t attr = cudaFuncSetAttribute(
        reinterpret_cast<const void*>(
            gdn_chunk_gemm_tri_kernel<true, true, true, __nv_bfloat16, 8, true>),
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
    if (attr != cudaSuccess) return attr;
    opted_device = device;
  }
  gdn_chunk_gemm_tri_kernel<true, true, true, __nv_bfloat16, 8, true>
      <<<dim3(seq_pad / 64, num_v_heads), 256, smem, stream>>>(
          static_cast<const float*>(a),
          static_cast<const __nv_bfloat16*>(conv_out),
          static_cast<const __nv_bfloat16*>(k),
          static_cast<const float*>(beta), static_cast<const float*>(g_cum),
          static_cast<__nv_bfloat16*>(vt_out),
          static_cast<__nv_bfloat16*>(kcd_out),
          seq_pad, seq, conv_dim, v_offset);
  return cudaGetLastError();
}

// Typed BF16 state scan with fixed 1024-thread launch geometry. It does not
// consume the legacy float-input kernel policy.
extern "C" cudaError_t apxinf_cn_gdn_chunk_state_qk_bf16(
    const void* q, const void* k, const void* g_cum, const void* t_in,
    const void* vt_in, const void* kcd_in, void* state, void* out,
    int seq, int seq_pad, int num_v_heads, int head_k_dim, int head_v_dim,
    int chunk_size, int total_chunks, int out_row_width,
    cudaStream_t stream) {
  if (!q || !k || !g_cum || !t_in || !vt_in || !kcd_in || !state ||
      !out || seq <= 0 || seq_pad < seq || seq_pad != total_chunks * 64 ||
      num_v_heads <= 0 || head_k_dim != 128 || head_v_dim != 128 ||
      chunk_size != 64 || out_row_width != num_v_heads * 128)
    return cudaErrorInvalidValue;
  constexpr int KP = 136, VP = 136;
  constexpr size_t smem =
      static_cast<size_t>(128 * VP + 2 * 64 * KP + 64 * VP) *
          sizeof(__nv_bfloat16) +
      static_cast<size_t>(64 * VP) * sizeof(float);
  static thread_local int opted_device = -1;
  int device = -1;
  const cudaError_t device_status = cudaGetDevice(&device);
  if (device_status != cudaSuccess) return device_status;
  if (opted_device != device) {
    const cudaError_t attr = cudaFuncSetAttribute(
        reinterpret_cast<const void*>(
            gdn_chunk_state_wmma_kernel<__nv_bfloat16, true, true>),
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
    if (attr != cudaSuccess) return attr;
    opted_device = device;
  }
  const float scale =
      static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_k_dim)));
  gdn_chunk_state_wmma_kernel<__nv_bfloat16, true, true>
      <<<num_v_heads, 1024, smem, stream>>>(
          static_cast<const __nv_bfloat16*>(q),
          static_cast<const __nv_bfloat16*>(k),
          static_cast<const float*>(g_cum),
          static_cast<const __nv_bfloat16*>(t_in),
          static_cast<const __nv_bfloat16*>(vt_in),
          static_cast<const __nv_bfloat16*>(kcd_in),
          static_cast<float*>(state), static_cast<__nv_bfloat16*>(out),
          seq, seq_pad, total_chunks, out_row_width, scale);
  return cudaGetLastError();
}


extern "C" cudaError_t apxinf_cn_sinusoidal_embedding_bf16(const void* positions,void* output,
    int rows,int dim,float scale,float frequency_step,cudaStream_t stream) {
  if(!positions||!output||rows<=0||dim<=0||dim%2||!std::isfinite(scale)||!std::isfinite(frequency_step))
    return cudaErrorInvalidValue;
  sinusoidal_embedding_bf16_kernel<<<rows,128,0,stream>>>(
      (const float*)positions,(__nv_bfloat16*)output,dim,scale,frequency_step);
  return cudaGetLastError();
}


extern "C" cudaError_t apxinf_cn_rgb_u8_to_temporal2_merge2_rect_bf16(
    const void* rgb, void* patches, const void* lut, int grid_h, int grid_w,
    cudaStream_t stream) {
  if (!rgb || !patches || !lut || grid_h <= 0 || grid_w <= 0 ||
      (grid_h & 1) || (grid_w & 1) ||
      static_cast<int64_t>(grid_h) * grid_w > std::numeric_limits<int>::max())
    return cudaErrorInvalidValue;
  rgb_u8_to_temporal2_merge2_rect_bf16_kernel<<<grid_h * grid_w, 256, 0, stream>>>(
      static_cast<const uint8_t*>(rgb), static_cast<uint16_t*>(patches),
      static_cast<const uint16_t*>(lut), grid_h, grid_w);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_cn_pillow_bicubic_u8_axis(
    const void* input, void* output, const void* input_offsets,
    const void* output_offsets, const void* bounds, const void* weights,
    int ksize, int in_w, int in_h, int out_w, int out_h,
    int batch, bool horizontal, cudaStream_t stream) {
  if (!input || !output || !input_offsets || !output_offsets || !bounds || !weights ||
      ksize <= 0 || in_w <= 0 || in_h <= 0 || out_w <= 0 || out_h <= 0 ||
      (horizontal ? out_h != in_h : out_w != in_w) ||
      batch <= 0 || batch > 65535 ||
      static_cast<int64_t>(out_w) * out_h > std::numeric_limits<int>::max())
    return cudaErrorInvalidValue;
  const int64_t output_pixels = static_cast<int64_t>(out_w) * out_h;
  const int64_t blocks = (output_pixels + 255) / 256;
  if (blocks > std::numeric_limits<int>::max()) return cudaErrorInvalidValue;
  pillow_bicubic_u8_axis_kernel<<<dim3(static_cast<unsigned int>(blocks), batch), 256, 0, stream>>>(
      static_cast<const uint8_t*>(input), static_cast<uint8_t*>(output),
      static_cast<const int64_t*>(input_offsets),
      static_cast<const int64_t*>(output_offsets),
      static_cast<const int32_t*>(bounds), static_cast<const int32_t*>(weights),
      ksize, in_w, in_h, out_w, out_h, horizontal);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_cn_swiglu_bf16_rounded(
    const void* gate_up, void* output, int rows, int inner, cudaStream_t stream) {
  if (!gate_up || !output || rows <= 0 || inner <= 0) return cudaErrorInvalidValue;
  const int64_t count = static_cast<int64_t>(rows) * inner;
  const int blocks = static_cast<int>((count + 255) / 256 > 65535 ? 65535 : (count + 255) / 256);
  if (swiglu_vec8_ok(gate_up, output, inner)) {
    const int vblocks = static_cast<int>((count / 8 + 255) / 256 > 65535
                                             ? 65535
                                             : (count / 8 + 255) / 256);
    swiglu_bf16_vec8_kernel<true><<<vblocks, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gate_up),
        static_cast<__nv_bfloat16*>(output), rows, inner);
    return cudaGetLastError();
  }
  swiglu_bf16_kernel<true><<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(gate_up), static_cast<__nv_bfloat16*>(output), rows, inner);
  return cudaGetLastError();
}

