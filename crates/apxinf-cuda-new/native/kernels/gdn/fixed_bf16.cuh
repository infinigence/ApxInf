#pragma once

#include "../custom/gdn_block_inverse.cuh"
#include "../custom/rms_reduction.cuh"

// Copyright 2026 apxinf contributors.
// Pure CUDA operators grouped by physical operation; launch policy lives under adapters/.
//
// GDN operators: gated delta rule (chunked
// prefill + rank-1 recurrent decode), depthwise causal conv1d with SiLU,
//
// Semantics follow the published equations (Qwen3.5 gated delta net, DiT-style
// adaLN diffusion experts); every kernel is model-neutral and parameterized by
// raw geometry (heads, head dims, chunk size).

// small device helpers

__device__ __forceinline__ float gdn_exp2_approx(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float la_sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// Triton's sigmoid lowering used by FLA gated RMSNorm: exp(x) becomes
// ex2.approx(x*log2(e)) and the reciprocal uses div.full.f32.
__device__ __forceinline__ float la_triton_sigmoid(float x) {
  const float exponent = __fmul_rn(-x, 1.4426950408889634f);
  float power;
  asm("ex2.approx.f32 %0, %1;" : "=f"(power) : "f"(exponent));
  const float denominator = __fadd_rn(1.0f, power);
  float result;
  asm("div.full.f32 %0, %1, %2;" : "=f"(result) : "f"(1.0f), "f"(denominator));
  return result;
}

// softplus with the PyTorch threshold=20 convention.
__device__ __forceinline__ float la_softplus(float x) {
  return x > 20.0f ? x : log1pf(expf(x));
}

__device__ __forceinline__ float la_silu(float x) {
  return x / (1.0f + expf(-x));
}

// depthwise causal conv1d + SiLU
//
// x is a strided token-major stream [seq, x_row_stride] whose first `channels`
// columns carry the mixed projection; out is dense [seq, channels].
// weight/state/new_state are [channels, kernel_size]. The effective left
// context of a cached call is state[:, 1..kernel_size-1] (the trailing
// kernel_size-1 pre-conv activations), zero left-pad when state == nullptr.
// new_state receives the last kernel_size entries of cat(effective_state, x),
// matching the reference cache contract for prefill and cached continuation.
//
// A block owns CONV_TOKENS consecutive tokens for one 256-channel slab. One
// thread per channel walks those tokens in sequence.
//
// The obvious shape -- one block per (token, slab) -- reads every x element
// kernel_size times, once for each output whose window covers it, and rereads
// the channel's filter taps for every token. At the shipped prefill that is
// 222 MB of loads where the data is 55 MB, and 108k blocks each doing four
// fused multiply-adds per thread. Staging the window in shared memory instead
// brings the redundancy from kernel_size to (CONV_TOKENS + kernel_size - 1) /
// CONV_TOKENS, which is 35/32, and cuts the block count by CONV_TOKENS.
//
// The per-output accumulation is still the same kernel_size products summed in
// the same order over the same values, so the result is bit-identical.
#define CONV_TOKENS 32

__global__ void causal_conv1d_silu_bf16_kernel(
    const __nv_bfloat16* x, const __nv_bfloat16* weight,
    const __nv_bfloat16* state, __nv_bfloat16* out, __nv_bfloat16* new_state,
    int channels, int seq, int kernel_size, int64_t x_row_stride) {
  extern __shared__ __nv_bfloat16 conv_smem[];  // [tokens + kernel_size - 1][blockDim.x]
  const int lane = static_cast<int>(threadIdx.x);
  const int channel = blockIdx.y * blockDim.x + lane;
  const int width = static_cast<int>(blockDim.x);
  const int token_base = blockIdx.x * CONV_TOKENS;
  if (token_base >= seq) return;
  const int tokens = min(CONV_TOKENS, seq - token_base);
  const int active = channel < channels;

  // Stage this block's window once: the kernel_size - 1 tokens of left context
  // followed by the tokens it owns. Out-of-range reads take the cached state,
  // or zero, exactly as the per-token form did.
  for (int r = 0; r < tokens + kernel_size - 1; ++r) {
    const int src = token_base - (kernel_size - 1) + r;
    float value = 0.0f;
    if (active) {
      if (src >= 0) {
        value = __bfloat162float(x[static_cast<int64_t>(src) * x_row_stride + channel]);
      } else if (state != nullptr) {
        value = __bfloat162float(state[channel * kernel_size + kernel_size + src]);
      }
    }
    conv_smem[r * width + lane] = __float2bfloat16(value);
  }
  __syncthreads();
  if (!active) return;

  float taps[8];
  for (int i = 0; i < kernel_size; ++i) {
    taps[i] = __bfloat162float(weight[channel * kernel_size + i]);
  }

  for (int t = 0; t < tokens; ++t) {
    float acc = 0.0f;
    for (int i = 0; i < kernel_size; ++i) {
      acc += __bfloat162float(conv_smem[(t + i) * width + lane]) * taps[i];
    }
    // Preserve the BF16 convolution output before the separate SiLU operation.
    const float conv = __bfloat162float(__float2bfloat16(acc));
    const int token = token_base + t;
    out[static_cast<int64_t>(token) * channels + channel] =
        __float2bfloat16(la_silu(conv));
  }

  // The last token's threads emit the new state, so state writes never race
  // the reads of other tokens.
  if (token_base + tokens == seq) {
    for (int i = 0; i < kernel_size; ++i) {
      float value = 0.0f;
      if (seq + i < kernel_size) {
        if (state != nullptr) {
          value = __bfloat162float(state[channel * kernel_size + seq + i]);
        }
      } else {
        value = __bfloat162float(
            x[static_cast<int64_t>(seq - kernel_size + i) * x_row_stride + channel]);
      }
      new_state[channel * kernel_size + i] = __float2bfloat16(value);
    }
  }
}

// gated delta rule: prefill preparation
//
// Prefill stores normalized BF16 Q/K before chunk products. Recurrent decode
// retains FP32 normalized Q/K and scales Q before the recurrent dot products.
// Values are scattered to the value-head layout
// (each key head repeats to num_v_heads/num_k_heads consecutive value heads).
// Outputs are fp32 head-major [num_v_heads, seq_pad, head_k_dim]; the padded
// tail keeps the caller's zero fill. One block per (token, key head) with
// head_k_dim threads.

__device__ __forceinline__ float gdn_qk_widen(float value) { return value; }
__device__ __forceinline__ float gdn_qk_widen(__nv_bfloat16 value) {
  return __bfloat162float(value);
}
__device__ __forceinline__ void gdn_qk_store(float* out, int64_t idx, float value) {
  out[idx] = value;
}
__device__ __forceinline__ void gdn_qk_store(__nv_bfloat16* out, int64_t idx, float value) {
  out[idx] = __float2bfloat16(value);
}

template <typename Output>
__global__ void gdn_qk_prep_kernel_t(
    const __nv_bfloat16* conv_out, Output* q_out, Output* k_out,
    int seq, int seq_pad, int conv_dim, int key_dim,
    int num_v_heads, int head_k_dim, float scale, float eps, bool recurrent) {
  const int token = blockIdx.x;
  const int k_head = blockIdx.y;
  const int reps = num_v_heads / gridDim.y;
  const int d = threadIdx.x;
  if (d >= head_k_dim) return;
  extern __shared__ float la_reduce[];
  const int64_t conv_base = static_cast<int64_t>(token) * conv_dim;
  const float q_value = __bfloat162float(conv_out[conv_base + k_head * head_k_dim + d]);
  const float k_value = __bfloat162float(conv_out[conv_base + key_dim + k_head * head_k_dim + d]);
  la_reduce[threadIdx.x] = q_value * q_value;
  la_reduce[blockDim.x + threadIdx.x] = k_value * k_value;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      la_reduce[threadIdx.x] += la_reduce[threadIdx.x + offset];
      la_reduce[blockDim.x + threadIdx.x] += la_reduce[blockDim.x + threadIdx.x + offset];
    }
    __syncthreads();
  }
  const float q_inv = rsqrtf(la_reduce[0] + eps);
  const float k_inv = rsqrtf(la_reduce[blockDim.x] + eps);
  const float q_normed = recurrent ? q_value * q_inv * scale
      : __bfloat162float(__float2bfloat16(q_value * q_inv));
  const float k_normed = recurrent ? k_value * k_inv
      : __bfloat162float(__float2bfloat16(k_value * k_inv));
  for (int r = 0; r < reps; ++r) {
    const int head = k_head * reps + r;
    const int64_t dst = (static_cast<int64_t>(head) * seq_pad + token) * head_k_dim + d;
    gdn_qk_store(q_out, dst, q_normed);
    gdn_qk_store(k_out, dst, k_normed);
  }
}

// beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias); v copied to the
// head-major layout. b_proj/a_proj point at the first row of their column
// slice inside a fused projection output (row stride given in elements).
//
// v carries the convolution output unchanged, so it is stored at the width it
// already has. Widening it to fp32 on the way out doubled the buffer and every
// read of it downstream without adding a bit: at the shipped prefill that is
// 55 MB a layer written and another 55 read to carry BF16 values.
//
// A block covers VB_TOKENS tokens of one head. One token per block gave every
// thread a single load and a single store -- 108k blocks of 128 threads to move
// the buffer, which reaches about half the board's bandwidth. The per-token
// scalars stay on one thread each, as before.
#define VB_TOKENS 16

__global__ void gdn_vb_prep_kernel(
    const __nv_bfloat16* conv_out,
    const __nv_bfloat16* b_proj, const __nv_bfloat16* a_proj,
    const float* dt_bias, const float* a_log,
    __nv_bfloat16* v_out, float* beta_out, float* g_out,
    int seq, int seq_pad, int conv_dim, int v_offset,
    int ba_row_stride, int head_v_dim) {
  const int head = blockIdx.y;
  const int token_base = blockIdx.x * VB_TOKENS;
  if (token_base >= seq) return;
  const int tokens = min(VB_TOKENS, seq - token_base);
  const int d = threadIdx.x;

  // One thread per token folds the gate scalars, which depend on the token and
  // the head but not on the value channel.
  if (d < tokens) {
    const int token = token_base + d;
    const float b_value =
        __bfloat162float(b_proj[static_cast<int64_t>(token) * ba_row_stride + head]);
    const float a_value =
        __bfloat162float(a_proj[static_cast<int64_t>(token) * ba_row_stride + head]);
    beta_out[static_cast<int64_t>(head) * seq_pad + token] =
        __bfloat162float(__float2bfloat16(la_sigmoid(b_value)));
    g_out[static_cast<int64_t>(head) * seq_pad + token] =
        -expf(a_log[head]) * la_softplus(a_value + dt_bias[head]);
  }
  if (d >= head_v_dim) return;
  for (int t = 0; t < tokens; ++t) {
    const int token = token_base + t;
    v_out[(static_cast<int64_t>(head) * seq_pad + token) * head_v_dim + d] =
        conv_out[static_cast<int64_t>(token) * conv_dim + v_offset +
                 head * head_v_dim + d];
  }
}

// Same gate arithmetic as vb-prep, without materializing V.
__global__ void gdn_gate_only_kernel(
    const __nv_bfloat16* b_proj, const __nv_bfloat16* a_proj,
    const float* dt_bias, const float* a_log,
    float* beta_out, float* g_out,
    int seq, int seq_pad, int ba_row_stride) {
  const int head = blockIdx.y;
  const int token_base = blockIdx.x * VB_TOKENS;
  if (token_base >= seq) return;
  const int tokens = min(VB_TOKENS, seq - token_base);
  const int d = threadIdx.x;
  if (d < tokens) {
    const int token = token_base + d;
    const float b_value =
        __bfloat162float(b_proj[static_cast<int64_t>(token) * ba_row_stride + head]);
    const float a_value =
        __bfloat162float(a_proj[static_cast<int64_t>(token) * ba_row_stride + head]);
    beta_out[static_cast<int64_t>(head) * seq_pad + token] =
        __bfloat162float(__float2bfloat16(la_sigmoid(b_value)));
    g_out[static_cast<int64_t>(head) * seq_pad + token] =
        -expf(a_log[head]) * la_softplus(a_value + dt_bias[head]);
  }
}

// Chunk-local inclusive sum in log2 units, matching FLA's exp2 gate arithmetic.
__global__ void gdn_cumsum_kernel(
    const float* g, float* g_cum, int seq_pad, int chunk_size) {
  const int head = blockIdx.y;
  const int chunk = blockIdx.x;
  const int64_t base = static_cast<int64_t>(head) * seq_pad + static_cast<int64_t>(chunk) * chunk_size;
  const int lane = threadIdx.x;
  float carry = 0.0f;
  // Match the warp-local inclusive scan and then add the preceding warp sum.
  // A scalar running sum has different FP32 rounding at almost every token.
  for (int start = 0; start < chunk_size; start += 32) {
    const int i = start + lane;
    float value = i < chunk_size ? g[base + i] : 0.0f;
    for (int offset = 1; offset < 32; offset <<= 1) {
      const float previous = __shfl_up_sync(0xffffffff, value, offset);
      if (lane >= offset) value = __fadd_rn(previous, value);
    }
    if (start != 0) value = __fadd_rn(carry, value);
    if (i < chunk_size) g_cum[base + i] = value * 1.4426950408889634f;
    carry = __shfl_sync(0xffffffff, value, 31);
  }
}

// A1[i,j] = -beta_i * (k_i . k_j) * exp2(g_i-g_j), strictly lower triangular.
// T[i,j] = bf16((q_i . k_j) * exp2(g_i-g_j)), including the diagonal.
// Q remains unscaled until the final attention output. One block per chunk/head.
__global__ __launch_bounds__(256, 4) void gdn_attn_raw_kernel(
    const float* q, const float* k, const float* beta, const float* g_cum,
    float* a_out, __nv_bfloat16* t_out,
    int seq_pad, int head_k_dim, int chunk_size) {
  const int head = blockIdx.y;
  const int chunk = blockIdx.x;
  const int64_t token_base = static_cast<int64_t>(head) * seq_pad + static_cast<int64_t>(chunk) * chunk_size;
  const int64_t matrix_base = (static_cast<int64_t>(head) * gridDim.x + chunk) * chunk_size * chunk_size;
  // This block forms K.K^T and Q.K^T over one chunk, and every cell walked the
  // same head_k_dim rows straight out of global memory: three rows per cell,
  // chunk_size*chunk_size cells, against only two chunk_size*head_k_dim tiles
  // of distinct data -- 96x redundancy at the shipped shape. Staging the tiles
  // in shared memory leaves the arithmetic untouched (same d order, same FP32
  // accumulation) and only changes where the operands are read from.
  extern __shared__ float gdn_attn_smem[];
  // Rows are padded by one float. Adjacent threads in a warp hold adjacent j
  // and read k_tile[j * stride + d]; with an unpadded stride of head_k_dim the
  // whole warp lands on one bank (head_k_dim is a multiple of 32), which costs
  // a 32-way conflict and gives back what the staging saves. A stride of
  // head_k_dim + 1 moves each j to its own bank.
  const int tile_stride = head_k_dim + 1;
  float* k_tile = gdn_attn_smem;
  float* q_tile = gdn_attn_smem + chunk_size * tile_stride;
  for (int idx = threadIdx.x; idx < chunk_size * head_k_dim; idx += blockDim.x) {
    const int t = idx / head_k_dim;
    const int d = idx - t * head_k_dim;
    const int64_t src = (token_base + t) * head_k_dim + d;
    k_tile[t * tile_stride + d] = k[src];
    q_tile[t * tile_stride + d] = q[src];
  }
  __syncthreads();
  // One cell per thread issues three shared reads for two FMAs: k_tile[i][d],
  // q_tile[i][d] and k_tile[j][d]. A rectangular register tile of cells shares
  // all three across the block it covers -- 2*ATI+ATJ reads for 2*ATI*ATJ
  // FMAs, four times fewer per FMA at 4x4 -- which is the read side of the
  // same argument that moved the chunk-state loops. At the shipped shape a
  // block's 4096 cells over 256 threads is exactly one 4x4 tile per thread, so
  // the tile costs no extra trips. Each cell still sums over d in increasing
  // order against the same operands, so the values do not change.
  constexpr int ATI = 4;
  constexpr int ATJ = 4;
  if (chunk_size % ATI == 0 && chunk_size % ATJ == 0) {
    const int tiles_j = chunk_size / ATJ;
    const int total_tiles = (chunk_size / ATI) * tiles_j;
    for (int t = threadIdx.x; t < total_tiles; t += blockDim.x) {
      const int i0 = (t / tiles_j) * ATI;
      const int j0 = (t - (t / tiles_j) * tiles_j) * ATJ;
      float a1[ATI][ATJ];
      float a2[ATI][ATJ];
#pragma unroll
      for (int ii = 0; ii < ATI; ++ii) {
#pragma unroll
        for (int jj = 0; jj < ATJ; ++jj) {
          a1[ii][jj] = 0.0f;
          a2[ii][jj] = 0.0f;
        }
      }
      for (int d = 0; d < head_k_dim; ++d) {
        float kv[ATI];
        float qv[ATI];
        float kj[ATJ];
#pragma unroll
        for (int ii = 0; ii < ATI; ++ii) {
          kv[ii] = k_tile[(i0 + ii) * tile_stride + d];
          qv[ii] = q_tile[(i0 + ii) * tile_stride + d];
        }
#pragma unroll
        for (int jj = 0; jj < ATJ; ++jj) {
          kj[jj] = k_tile[(j0 + jj) * tile_stride + d];
        }
#pragma unroll
        for (int ii = 0; ii < ATI; ++ii) {
#pragma unroll
          for (int jj = 0; jj < ATJ; ++jj) {
            a1[ii][jj] += kv[ii] * kj[jj];
            a2[ii][jj] += qv[ii] * kj[jj];
          }
        }
      }
#pragma unroll
      for (int ii = 0; ii < ATI; ++ii) {
        const int i = i0 + ii;
        const float beta_i = beta[token_base + i];
        const float g_i = g_cum[token_base + i];
#pragma unroll
        for (int jj = 0; jj < ATJ; ++jj) {
          const int j = j0 + jj;
          const int cell = i * chunk_size + j;
          const float decay = gdn_exp2_approx(g_i - g_cum[token_base + j]);
          a_out[matrix_base + cell] =
              (j < i) ? -__fmul_rn(__fmul_rn(a1[ii][jj], decay), beta_i) : 0.0f;
          t_out[matrix_base + cell] =
              __float2bfloat16((j <= i) ? a2[ii][jj] * decay : 0.0f);
        }
      }
    }
  } else {
    for (int cell = threadIdx.x; cell < chunk_size * chunk_size; cell += blockDim.x) {
      const int i = cell / chunk_size;
      const int j = cell - i * chunk_size;
      const int row_i = i * tile_stride;
      const int row_j = j * tile_stride;
      float a1 = 0.0f;
      float a2 = 0.0f;
      const float beta_i = beta[token_base + i];
      for (int d = 0; d < head_k_dim; ++d) {
        const float k_j = k_tile[row_j + d];
        a1 += k_tile[row_i + d] * k_j;
        a2 += q_tile[row_i + d] * k_j;
      }
      const float decay = gdn_exp2_approx(g_cum[token_base + i] - g_cum[token_base + j]);
      a_out[matrix_base + cell] = (j < i) ? -__fmul_rn(__fmul_rn(a1, decay), beta_i) : 0.0f;
      t_out[matrix_base + cell] = __float2bfloat16((j <= i) ? a2 * decay : 0.0f);
    }
  }
}

// In-place forward substitution over strictly-lower A, then A += I:
// solves (I - A)^{-1}, then materializes BF16 coefficients for the WY products.
__global__ void gdn_tri_solve_kernel(float* a, int chunk_size) {
  extern __shared__ float la_solve[];
  float* matrix = la_solve;
  float* row_copy = la_solve + chunk_size * chunk_size;
  const int64_t matrix_base = static_cast<int64_t>(blockIdx.x) * chunk_size * chunk_size;
  for (int cell = threadIdx.x; cell < chunk_size * chunk_size; cell += blockDim.x) {
    matrix[cell] = a[matrix_base + cell];
  }
  __syncthreads();
  for (int i = 1; i < chunk_size; ++i) {
    if (threadIdx.x < i) row_copy[threadIdx.x] = matrix[i * chunk_size + threadIdx.x];
    __syncthreads();
    if (threadIdx.x < i) {
      const int j = threadIdx.x;
      float sum = 0.0f;
      for (int m = 0; m < i; ++m) sum += row_copy[m] * matrix[m * chunk_size + j];
      matrix[i * chunk_size + j] = row_copy[j] + sum;
    }
    __syncthreads();
  }
  for (int cell = threadIdx.x; cell < chunk_size * chunk_size; cell += blockDim.x) {
    const int i = cell / chunk_size;
    const int j = cell - i * chunk_size;
    a[matrix_base + cell] = __bfloat162float(__float2bfloat16(matrix[cell] + (i == j ? 1.0f : 0.0f)));
  }
}

// VT/U = bf16(A @ bf16(V * beta)); W = bf16(A @ bf16(bf16(K * beta) * exp2(g))).
// Same shape of redundancy as the chunk-state kernel, and the same fix: a
// thread's column is fixed, so the tile read in the inner loop does not depend
// on which cell is being accumulated, and a compile-time tile of accumulators
// lets it move out. See chunk_state_tile() for why the width is measured
// rather than chosen.
template <int GEMM_TILE>
__global__ __launch_bounds__(256, 4) void gdn_chunk_gemm_kernel(
    const float* a, const __nv_bfloat16* v, const float* k, const float* beta,
    const float* g_cum, __nv_bfloat16* vt_out, __nv_bfloat16* kcd_out,
    int seq_pad, int head_k_dim, int head_v_dim, int chunk_size) {
  const int head = blockIdx.y;
  const int chunk = blockIdx.x;
  const int64_t token_base = static_cast<int64_t>(head) * seq_pad + static_cast<int64_t>(chunk) * chunk_size;
  const int64_t a_base = (static_cast<int64_t>(head) * gridDim.x + chunk) * chunk_size * chunk_size;
  const int64_t vt_base = (static_cast<int64_t>(head) * gridDim.x + chunk) * chunk_size * head_v_dim;
  const int64_t kcd_base = (static_cast<int64_t>(head) * gridDim.x + chunk) * chunk_size * head_k_dim;
  // vb and kb depend only on (m, j), never on i, but the inner loops rebuilt
  // them for every one of the chunk_size values of i: chunk_size-fold redundant
  // multiplies, BF16 round trips, and -- for kb -- exp2 calls. Precompute each
  // tile once. Threads in a warp hold consecutive j and read tile[m * dim + j],
  // so these stay conflict-free without padding.
  // Both tiles hold values that have already been through a BF16 round trip,
  // so holding them as BF16 is bit-identical and halves the block's shared
  // footprint. At float width this kernel asks for 64KB, an SM fits two blocks,
  // and ncu measures 33% occupancy against an L1 pipeline at 44%.
  extern __shared__ float gdn_gemm_smem[];
  __nv_bfloat16* vb_tile = reinterpret_cast<__nv_bfloat16*>(gdn_gemm_smem);
  __nv_bfloat16* kb_tile = vb_tile + chunk_size * head_v_dim;
  for (int idx = threadIdx.x; idx < chunk_size * head_v_dim; idx += blockDim.x) {
    const int m = idx / head_v_dim;
    const int j = idx - m * head_v_dim;
    vb_tile[idx] = __float2bfloat16(
        __bfloat162float(v[(token_base + m) * head_v_dim + j]) *
        beta[token_base + m]);
  }
  for (int idx = threadIdx.x; idx < chunk_size * head_k_dim; idx += blockDim.x) {
    const int m = idx / head_k_dim;
    const int j = idx - m * head_k_dim;
    const float kb0 = __bfloat162float(__float2bfloat16(
        k[(token_base + m) * head_k_dim + j] * beta[token_base + m]));
    kb_tile[idx] = __float2bfloat16(kb0 * gdn_exp2_approx(g_cum[token_base + m]));
  }
  __syncthreads();
  const int v_cells = chunk_size * head_v_dim;
  const int k_cells = chunk_size * head_k_dim;
  const int v_span = static_cast<int>(blockDim.x) * GEMM_TILE;
  const int k_span = v_span;
  // The two products share a: vt_out = a @ vb_tile and kcd_out = a @ kb_tile
  // walk the same a[i][m] over the same i and m. Run as two loops each cell
  // fetches its row of a twice, which at the shipped shape is 4096 global
  // loads per thread where 2048 carry all the data. Fusing them halves that
  // and costs one more accumulator per tile element, so the tile width has to
  // come down far enough that the pair still fits in registers under the
  // block's launch bounds -- see chunk_state_tile() for how the width is
  // chosen. The two output cells at a given (i, j) are independent, and each
  // still accumulates over m in increasing order, so the values are unchanged.
  //
  // Requires the two tiles to have the same row length; they do at every
  // shipped shape, and the pair of loops below covers the case where they do
  // not.
  if (head_v_dim == head_k_dim && v_cells == k_cells &&
      v_cells % v_span == 0 && blockDim.x % head_v_dim == 0) {
    const int row_step = static_cast<int>(blockDim.x) / head_v_dim;
    for (int base = threadIdx.x; base < v_cells; base += v_span) {
      const int row0 = base / head_v_dim;
      const int j = base - row0 * head_v_dim;
      float vt[GEMM_TILE];
      float kcd[GEMM_TILE];
      #pragma unroll
      for (int s = 0; s < GEMM_TILE; ++s) {
        vt[s] = 0.0f;
        kcd[s] = 0.0f;
      }
      for (int m = 0; m < chunk_size; ++m) {
        const float vv = __bfloat162float(vb_tile[m * head_v_dim + j]);
        const float kv = __bfloat162float(kb_tile[m * head_k_dim + j]);
        #pragma unroll
        for (int s = 0; s < GEMM_TILE; ++s) {
          const int i = row0 + s * row_step;
          const float av = a[a_base + i * chunk_size + m];
          vt[s] += av * vv;
          kcd[s] += av * kv;
        }
      }
      #pragma unroll
      for (int s = 0; s < GEMM_TILE; ++s) {
        const int off = base + s * static_cast<int>(blockDim.x);
        vt_out[vt_base + off] = __float2bfloat16(vt[s]);
        kcd_out[kcd_base + off] = __float2bfloat16(kcd[s]);
      }
    }
  } else if (v_cells % v_span == 0 && blockDim.x % head_v_dim == 0) {
    const int row_step = static_cast<int>(blockDim.x) / head_v_dim;
    for (int base = threadIdx.x; base < v_cells; base += v_span) {
      const int row0 = base / head_v_dim;
      const int j = base - row0 * head_v_dim;
      float vt[GEMM_TILE];
      #pragma unroll
      for (int s = 0; s < GEMM_TILE; ++s) vt[s] = 0.0f;
      for (int m = 0; m < chunk_size; ++m) {
        const float vv = __bfloat162float(vb_tile[m * head_v_dim + j]);
        #pragma unroll
        for (int s = 0; s < GEMM_TILE; ++s) {
          const int i = row0 + s * row_step;
          vt[s] += a[a_base + i * chunk_size + m] * vv;
        }
      }
      #pragma unroll
      for (int s = 0; s < GEMM_TILE; ++s) {
        vt_out[vt_base + base + s * static_cast<int>(blockDim.x)] =
            __float2bfloat16(vt[s]);
      }
    }
  } else {
    for (int cell = threadIdx.x; cell < v_cells; cell += blockDim.x) {
      const int i = cell / head_v_dim;
      const int j = cell - i * head_v_dim;
      float vt = 0.0f;
      for (int m = 0; m < chunk_size; ++m) {
        vt += a[a_base + i * chunk_size + m] *
              __bfloat162float(vb_tile[m * head_v_dim + j]);
      }
      vt_out[vt_base + cell] = __float2bfloat16(vt);
    }
  }
  if (head_v_dim == head_k_dim && v_cells == k_cells &&
      v_cells % v_span == 0 && blockDim.x % head_v_dim == 0) {
    // kcd_out was produced by the fused loop above.
  } else if (k_cells % k_span == 0 && blockDim.x % head_k_dim == 0) {
    const int row_step = static_cast<int>(blockDim.x) / head_k_dim;
    for (int base = threadIdx.x; base < k_cells; base += k_span) {
      const int row0 = base / head_k_dim;
      const int j = base - row0 * head_k_dim;
      float kcd[GEMM_TILE];
      #pragma unroll
      for (int s = 0; s < GEMM_TILE; ++s) kcd[s] = 0.0f;
      for (int m = 0; m < chunk_size; ++m) {
        const float kv = __bfloat162float(kb_tile[m * head_k_dim + j]);
        #pragma unroll
        for (int s = 0; s < GEMM_TILE; ++s) {
          const int i = row0 + s * row_step;
          kcd[s] += a[a_base + i * chunk_size + m] * kv;
        }
      }
      #pragma unroll
      for (int s = 0; s < GEMM_TILE; ++s) {
        kcd_out[kcd_base + base + s * static_cast<int>(blockDim.x)] =
            __float2bfloat16(kcd[s]);
      }
    }
  } else {
    for (int cell = threadIdx.x; cell < k_cells; cell += blockDim.x) {
      const int i = cell / head_k_dim;
      const int j = cell - i * head_k_dim;
      float kcd = 0.0f;
      for (int m = 0; m < chunk_size; ++m) {
        kcd += a[a_base + i * chunk_size + m] *
               __bfloat162float(kb_tile[m * head_k_dim + j]);
      }
      kcd_out[kcd_base + cell] = __float2bfloat16(kcd);
    }
  }
}

// Sequential chunk recurrence per head (state lives in global scratch):
//   v_new = VT - KCD @ bf16(S)
//   out = bf16(((q @ bf16(S)) * exp2(g_cum))*scale + (T @ bf16(v_new))*scale)
//   S = S * exp2(g_last) + k^T @ bf16(v_new * exp2(g_last-g_cum))
// state is [head_k_dim, head_v_dim] fp32 in global memory, read at the first
// chunk and rewritten per chunk, so the buffer carries the initial state in
// and the final state out. Grid is (num_v_heads); each block loops chunks.
// Cells accumulated per thread at once in the chunk-state kernel. Trades
// registers for shared-memory traffic: the operand hoisted out of the inner
// loop serves GDN_TILE accumulators instead of one. It has to be a template
// parameter rather than a runtime value -- with a runtime trip count the
// accumulator array is indexed dynamically, nvcc spills it to local memory and
// the kernel roughly halves in speed -- and the right value is not obvious
// enough to guess, so the launcher instantiates several and picks one.
// Load a tile of contiguous bf16 out of shared memory.
//
// A column tile reads GDN_TILE adjacent bf16, which at every shipped width is
// sixteen bytes on a sixteen-byte boundary: both shared buffers start on one
// (the extern block does, and both offsets are whole numbers of kilobytes),
// a row is v_cols bf16 and v_cols is a multiple of eight, and the tile's
// column index is a multiple of the tile width. That is enough for one wide load instead of eight narrow ones,
// but it takes the reinterpret to say so -- left to itself nvcc emits the
// eight, which is the whole cost the column tile was meant to move. The
// values are the same either way.
template <int N>
__device__ __forceinline__ void gdn_load_bf16_tile(const __nv_bfloat16* src,
                                                   float* dst) {
  if constexpr (N % 8 == 0) {
#pragma unroll
    for (int b = 0; b < N / 8; ++b) {
      const float4 packed = *reinterpret_cast<const float4*>(src + b * 8);
      const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&packed);
#pragma unroll
      for (int t = 0; t < 8; ++t) dst[b * 8 + t] = __bfloat162float(e[t]);
    }
  } else {
#pragma unroll
    for (int t = 0; t < N; ++t) dst[t] = __bfloat162float(src[t]);
  }
}

template <int GDN_TILE, bool V_SPLIT>
__global__ __launch_bounds__(1024) void gdn_chunk_state_kernel(
    const float* q, const float* k,
    const float* g_cum, const __nv_bfloat16* t_in,
    const __nv_bfloat16* vt_in, const __nv_bfloat16* kcd_in,
    float* state, __nv_bfloat16* out,
    int seq, int seq_pad, int head_k_dim, int head_v_dim, int chunk_size,
    int total_chunks, int out_row_width, float scale, int v_split) {
  const int head = blockIdx.x;
  // One head's scan can be shared by v_split blocks, each owning a slice of
  // the value dimension. Nothing here crosses that dimension: the decay scales
  // rows of the state, every accumulation runs over the key dimension, and an
  // output column reads only its own column of the state. A slice therefore
  // computes exactly the values it would have computed inside the whole block,
  // summed in the same order, so the result is bit-identical and not merely
  // close. What it buys is blocks: the scan is sequential over chunks and its
  // only other parallelism is the head count, which is 32 at the shipped shape
  // and leaves three quarters of a 128-SM device idle.
  //
  // V_SPLIT is a template parameter and not just the runtime `v_split`
  // because a block that is not split must compile to exactly what it
  // compiled to before this existed. Carrying the offset as a runtime value
  // cost enough registers to push the 1024-thread width past the per-block
  // budget: the launch came back with CUDA 701 on a board that had been
  // running it for weeks. With V_SPLIT false the compiler folds j0 to zero
  // and every index below collapses to its original form.
  const int v_cols = V_SPLIT ? head_v_dim / v_split : head_v_dim;
  const int j0 = V_SPLIT ? static_cast<int>(blockIdx.y) * v_cols : 0;
  extern __shared__ float gdn_chunk_smem[];
  float* v_new = gdn_chunk_smem;
  const int64_t head_token_base = static_cast<int64_t>(head) * seq_pad;
  float* state_head = state + static_cast<int64_t>(head) * head_k_dim * head_v_dim;
  const int cells = chunk_size * v_cols;
  const int state_cells_total = head_k_dim * v_cols;
  // The inter-chunk term reads the carried state only through a BF16 round
  // trip, so a BF16 copy is bit-identical to re-reading the FP32 state from
  // global memory. Caching it pays here: each thread walks a full head_k_dim
  // column per cell, and because blockDim is a multiple of head_v_dim a
  // thread's column index is fixed, so the same values were fetched once per
  // cell -- chunk_size*head_v_dim reads of a head_k_dim*head_v_dim matrix,
  // 64x redundancy at the shipped shape.
  __nv_bfloat16* state_cache =
      reinterpret_cast<__nv_bfloat16*>(gdn_chunk_smem + cells);
  // Both loops below consume v_new only through a BF16 round trip, and each
  // element is rounded once per row of the matrix it is multiplied against:
  // chunk_size times in the intra term, head_k_dim times in the state update,
  // which at the shipped shape is 64x and 128x. The state update also calls
  // exp2 on a value that depends only on the chunk row, once per cell. Rounding
  // each element once into this tile, twice per chunk, is the same arithmetic
  // on the same values -- the expressions are unchanged, only their count is.
  __nv_bfloat16* v_round = state_cache + state_cells_total;
  for (int c = 0; c < total_chunks; ++c) {
    if constexpr (V_SPLIT) {
      for (int cell = threadIdx.x; cell < state_cells_total; cell += blockDim.x) {
        const int m = cell / v_cols;
        state_cache[cell] =
            __float2bfloat16(state_head[m * head_v_dim + j0 + (cell - m * v_cols)]);
      }
    } else {
      for (int cell = threadIdx.x; cell < state_cells_total; cell += blockDim.x) {
        state_cache[cell] = __float2bfloat16(state_head[cell]);
      }
    }
    __syncthreads();
    const int64_t token_base = head_token_base + static_cast<int64_t>(c) * chunk_size;
    const int64_t vt_base = (static_cast<int64_t>(head) * total_chunks + c) * chunk_size * head_v_dim;
    const int64_t kcd_base = (static_cast<int64_t>(head) * total_chunks + c) * chunk_size * head_k_dim;
    const int64_t a_base = (static_cast<int64_t>(head) * total_chunks + c) * chunk_size * chunk_size;
    float attn_inter[32];
    // All three accumulation loops in this kernel tile over a thread's *columns*
    // rather than its rows. The distinction is what the tile amortises. A row
    // tile fixes j and walks i, so the one shared read per step is reused and
    // the GDN_TILE global reads are not: the inter term then issues eight
    // kcd_in loads and eight q loads for sixteen FMAs, and ncu charges 51.5% of
    // this kernel's stalls to long_scoreboard, which is the global path. A
    // column tile fixes i and walks j, so the global reads are the ones reused
    // -- one kcd_in and one q per sixteen FMAs -- and the shared reads become
    // GDN_TILE contiguous bf16, sixteen bytes, on a pipeline ncu puts under 1%.
    // Occupancy is not the lever here: raising it to 100% by dropping the state
    // cache was measured slower because it added global traffic, which is the
    // same finding read from the other side.
    //
    // Each cell still sums over m in increasing order against the same
    // operands; only which thread owns it changes. The values are unchanged.
    const int inter_span = static_cast<int>(blockDim.x) * GDN_TILE;
    if (cells % inter_span == 0 && v_cols % GDN_TILE == 0 &&
        head_k_dim % 4 == 0) {
      for (int base = static_cast<int>(threadIdx.x) * GDN_TILE, it = 0;
           base < cells; base += inter_span) {
        const int i = base / v_cols;
        const int j = base - i * v_cols;
        float vp[GDN_TILE];
        float ai[GDN_TILE];
#pragma unroll
        for (int t = 0; t < GDN_TILE; ++t) {
          vp[t] = 0.0f;
          ai[t] = 0.0f;
        }
        const int64_t krow = kcd_base + static_cast<int64_t>(i) * head_k_dim;
        const int64_t qrow = static_cast<int64_t>(token_base + i) * head_k_dim;
        // kcd_in and q are walked along m, which is their contiguous axis, and
        // a row of either starts on a sixteen-byte boundary (head_k_dim is a
        // multiple of four floats and the chunk bases are whole rows). Four
        // floats per instruction instead of one: with the shared side already
        // one wide load per m, this is what is left of the inner loop's
        // instruction count. Same values, same order over m.
        for (int m = 0; m < head_k_dim; m += 4) {
          // kcd is stored at BF16 width, so four of them are one eight-byte
          // load rather than one sixteen-byte load. q is still fp32.
          const __nv_bfloat162* kp =
              reinterpret_cast<const __nv_bfloat162*>(kcd_in + krow + m);
          const float2 k01 = __bfloat1622float2(kp[0]);
          const float2 k23 = __bfloat1622float2(kp[1]);
          const float4 qv4 = *reinterpret_cast<const float4*>(q + qrow + m);
          const float kv[4] = {k01.x, k01.y, k23.x, k23.y};
          const float qv[4] = {qv4.x, qv4.y, qv4.z, qv4.w};
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            float sv[GDN_TILE];
            gdn_load_bf16_tile<GDN_TILE>(state_cache + (m + u) * v_cols + j, sv);
#pragma unroll
            for (int t = 0; t < GDN_TILE; ++t) {
              vp[t] += kv[u] * sv[t];
              ai[t] += qv[u] * sv[t];
            }
          }
        }
        const float qg = gdn_exp2_approx(g_cum[token_base + i]);
#pragma unroll
        for (int t = 0; t < GDN_TILE; ++t, ++it) {
          const int cell = base + t;
          v_new[cell] =
              __bfloat162float(
                  vt_in[vt_base + (V_SPLIT ? i * head_v_dim + j0 + j + t : cell)]) -
              vp[t];
          attn_inter[it] = ai[t] * qg;
        }
      }
    } else {
      for (int cell = threadIdx.x, it = 0; cell < cells; cell += blockDim.x, ++it) {
        const int i = cell / v_cols;
        const int j = cell - i * v_cols;
        float vp = 0.0f;
        float ai = 0.0f;
        const float qg = gdn_exp2_approx(g_cum[token_base + i]);
        for (int m = 0; m < head_k_dim; ++m) {
          const float s = __bfloat162float(state_cache[m * v_cols + j]);
          vp += __bfloat162float(kcd_in[kcd_base + i * head_k_dim + m]) * s;
          ai += q[(token_base + i) * head_k_dim + m] * s;
        }
        v_new[cell] =
            __bfloat162float(
                vt_in[vt_base + (V_SPLIT ? i * head_v_dim + j0 + j : cell)]) -
            vp;
        attn_inter[it] = ai * qg;
      }
    }
    __syncthreads();
    for (int cell = threadIdx.x; cell < cells; cell += blockDim.x) {
      v_round[cell] = __float2bfloat16(v_new[cell]);
    }
    __syncthreads();
    // Column-tiled for the reason given above the inter term: t_in[i][m] is
    // loaded once and multiplied against GDN_TILE contiguous v_round entries,
    // instead of one v_round read against GDN_TILE t_in loads.
    // The tile count must be a compile-time constant: with a runtime trip count
    // the accumulators are indexed dynamically, nvcc puts them in local memory
    // instead of registers, and the kernel gets about twice as slow. Hence the
    // divisibility guard and the plain path beside it.
    const int tile_span = static_cast<int>(blockDim.x) * GDN_TILE;
    if (cells % tile_span == 0 && v_cols % GDN_TILE == 0 &&
        chunk_size % 4 == 0) {
      for (int base = static_cast<int>(threadIdx.x) * GDN_TILE, it = 0;
           base < cells; base += tile_span) {
        const int i = base / v_cols;
        const int j = base - i * v_cols;
        float intra[GDN_TILE];
        #pragma unroll
        for (int s = 0; s < GDN_TILE; ++s) intra[s] = 0.0f;
        const int64_t arow = a_base + static_cast<int64_t>(i) * chunk_size;
        for (int m = 0; m < chunk_size; m += 4) {
          // t is stored at BF16 width; four entries are one eight-byte load.
          const __nv_bfloat162* tp =
              reinterpret_cast<const __nv_bfloat162*>(t_in + arow + m);
          const float2 t01 = __bfloat1622float2(tp[0]);
          const float2 t23 = __bfloat1622float2(tp[1]);
          const float tv[4] = {t01.x, t01.y, t23.x, t23.y};
          #pragma unroll
          for (int u = 0; u < 4; ++u) {
            float vv[GDN_TILE];
            gdn_load_bf16_tile<GDN_TILE>(v_round + (m + u) * v_cols + j, vv);
            #pragma unroll
            for (int s = 0; s < GDN_TILE; ++s) {
              intra[s] += tv[u] * vv[s];
            }
          }
        }
        const int token = c * chunk_size + i;
        #pragma unroll
        for (int s = 0; s < GDN_TILE; ++s, ++it) {
          const float acc = attn_inter[it] * scale + intra[s] * scale;
          if (token < seq) {
            out[static_cast<int64_t>(token) * out_row_width + head * head_v_dim +
                j0 + j + s] = __float2bfloat16(acc);
          }
        }
      }
    } else {
      for (int cell = threadIdx.x, it = 0; cell < cells; cell += blockDim.x, ++it) {
        const int i = cell / v_cols;
        const int j = cell - i * v_cols;
        float intra = 0.0f;
        for (int m = 0; m < chunk_size; ++m) {
          intra += __bfloat162float(t_in[a_base + i * chunk_size + m]) *
                   __bfloat162float(v_round[m * v_cols + j]);
        }
        const float acc = attn_inter[it] * scale + intra * scale;
        const int token = c * chunk_size + i;
        if (token < seq) {
          out[static_cast<int64_t>(token) * out_row_width + head * head_v_dim +
              j0 + j] = __float2bfloat16(acc);
        }
      }
    }
    __syncthreads();
    const float g_last = g_cum[token_base + chunk_size - 1];
    const float decay = gdn_exp2_approx(g_last);
    const int state_cells = head_k_dim * v_cols;
    for (int cell = threadIdx.x; cell < cells; cell += blockDim.x) {
      const int i = cell / v_cols;
      v_round[cell] = __float2bfloat16(
          v_new[cell] * gdn_exp2_approx(g_last - g_cum[token_base + i]));
    }
    __syncthreads();
    if (state_cells % tile_span == 0 && v_cols % GDN_TILE == 0) {
      for (int base = static_cast<int>(threadIdx.x) * GDN_TILE;
           base < state_cells; base += tile_span) {
        const int m = base / v_cols;
        const int j = base - m * v_cols;
        float acc[GDN_TILE];
        #pragma unroll
        for (int s = 0; s < GDN_TILE; ++s) acc[s] = 0.0f;
        for (int i = 0; i < chunk_size; ++i) {
          const float kv = k[static_cast<int64_t>(token_base + i) * head_k_dim + m];
          float vv[GDN_TILE];
          gdn_load_bf16_tile<GDN_TILE>(v_round + i * v_cols + j, vv);
          #pragma unroll
          for (int s = 0; s < GDN_TILE; ++s) {
            acc[s] += kv * vv[s];
          }
        }
        #pragma unroll
        for (int s = 0; s < GDN_TILE; ++s) {
          const int cell = V_SPLIT ? m * head_v_dim + j0 + j + s : base + s;
          state_head[cell] = state_head[cell] * decay + acc[s];
        }
      }
    } else {
      for (int cell = threadIdx.x; cell < state_cells; cell += blockDim.x) {
        const int m = cell / v_cols;
        const int j = cell - m * v_cols;
        float acc = 0.0f;
        for (int i = 0; i < chunk_size; ++i) {
          acc += k[(token_base + i) * head_k_dim + m] *
                 __bfloat162float(v_round[i * v_cols + j]);
        }
        const int gcell = V_SPLIT ? m * head_v_dim + j0 + j : cell;
        state_head[gcell] = state_head[gcell] * decay + acc;
      }
    }
    __syncthreads();
  }
}

// Rank-1 recurrent update for single-token decode:
//   S *= exp(g); kv = S^T k; delta = (v - kv) * beta; S += k x delta; out = S^T q
// Decode-step GDN recurrence, one block per value head, SPLIT threads per
// output column.
//
// The one-thread-per-column form this replaces launches num_v_heads blocks of
// head_v_dim threads -- 32 x 128 = 4096 threads on a device with 30720 thread
// slots, 13% occupancy -- and each of those threads walks all head_k_dim rows
// of the state with a dependent global load per row. It also touches the state
// four times: read and write in the decay pass, read and write again in the
// update pass, 8 MB per layer where 4 MB is the work.
//
// Here each column is owned by SPLIT threads, each holding head_k_dim/SPLIT
// decayed rows in registers across the two passes. The state is read once and
// written once, and the thread count rises by SPLIT.
//
// The k-term sums change association: each thread sums its own contiguous
// stripe and the stripes are then combined in index order through shared
// memory, instead of one sequential sum over all head_k_dim terms. Same terms,
// same fp32 arithmetic, different grouping.
// HEAD_K is a template parameter only so the register stripe has a
// compile-time size; the launcher dispatches it from head_k_dim and falls
// back to the scalar kernel for any shape it does not instantiate.
template <int SPLIT, int HEAD_K>
__global__ void gdn_recurrent_split_kernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const float* __restrict__ beta,
    const float* __restrict__ g, float* state, __nv_bfloat16* out,
    int head_k_dim, int head_v_dim) {
  const int head = blockIdx.x;
  const int j = threadIdx.x % head_v_dim;       // output column
  const int slot = threadIdx.x / head_v_dim;    // which stripe of rows
  constexpr int rows = HEAD_K / SPLIT;          // rows this thread owns

  extern __shared__ float la_split[];
  float* k_row = la_split;                      // head_k_dim
  float* q_row = k_row + head_k_dim;            // head_k_dim
  float* partial = q_row + head_k_dim;          // head_v_dim * SPLIT
  for (int d = threadIdx.x; d < head_k_dim; d += blockDim.x) {
    const int64_t base = static_cast<int64_t>(head) * head_k_dim + d;
    k_row[d] = k[base];
    q_row[d] = q[base];
  }
  __syncthreads();

  const float g_exp = expf(g[head]);
  float* state_head = state + static_cast<int64_t>(head) * head_k_dim * head_v_dim;

  // Pass 1: decay this thread's stripe, keep it, and sum k . decayed over it.
  float decayed[rows];
  float kv_mem = 0.0f;
#pragma unroll
  for (int r = 0; r < rows; ++r) {
    const int m = slot * rows + r;
    const float d = state_head[static_cast<int64_t>(m) * head_v_dim + j] * g_exp;
    decayed[r] = d;
    kv_mem += d * k_row[m];
  }
  partial[slot * head_v_dim + j] = kv_mem;
  __syncthreads();

  // Combine stripes in index order, so the association is deterministic.
  float kv_total = 0.0f;
#pragma unroll
  for (int sp = 0; sp < SPLIT; ++sp) {
    kv_total += partial[sp * head_v_dim + j];
  }
  const float delta =
      (__bfloat162float(v[static_cast<int64_t>(head) * head_v_dim + j]) -
       kv_total) * beta[head];

  // Pass 2: update this thread's stripe from registers, write once, and sum
  // q . updated over it.
  float acc = 0.0f;
#pragma unroll
  for (int r = 0; r < rows; ++r) {
    const int m = slot * rows + r;
    const float updated = decayed[r] + k_row[m] * delta;
    state_head[static_cast<int64_t>(m) * head_v_dim + j] = updated;
    acc += updated * q_row[m];
  }
  __syncthreads();
  partial[slot * head_v_dim + j] = acc;
  __syncthreads();
  if (slot == 0) {
    float total = 0.0f;
#pragma unroll
    for (int sp = 0; sp < SPLIT; ++sp) {
      total += partial[sp * head_v_dim + j];
    }
    out[static_cast<int64_t>(head) * head_v_dim + j] = __float2bfloat16(total);
  }
}

__global__ void gdn_recurrent_kernel(
    const float* q, const float* k, const __nv_bfloat16* v, const float* beta,
    const float* g, float* state, __nv_bfloat16* out,
    int head_k_dim, int head_v_dim) {
  const int head = blockIdx.x;
  const int j = threadIdx.x;
  if (j >= head_v_dim) return;
  extern __shared__ float la_recurrent[];
  float* q_row = la_recurrent;
  float* k_row = la_recurrent + head_k_dim;
  for (int d = threadIdx.x; d < head_k_dim; d += blockDim.x) {
    q_row[d] = q[static_cast<int64_t>(head) * head_k_dim + d];
    k_row[d] = k[static_cast<int64_t>(head) * head_k_dim + d];
  }
  __syncthreads();
  const float beta_h = beta[head];
  const float g_exp = expf(g[head]);
  float* state_head = state + static_cast<int64_t>(head) * head_k_dim * head_v_dim;
  // The decayed state used to be written back here and read again below. It
  // does not need to be: the second pass can apply the same `* g_exp` to the
  // value it reads, which is the identical float operation, so the stored
  // result is unchanged. That removes a full write and a full read of the
  // state per step -- the kernel moved four passes over 128x128 floats per
  // head and now moves three -- and leaves the first pass's lines clean, so
  // the second pass can be served by L2 (the whole state is 2MB against a 4MB
  // cache) instead of waiting on writebacks.
  // __fmul_rn, not `*`. The store this loop used to do forced the decay to be
  // materialised as a rounded float; with the store gone and --use_fast_math
  // on, the optimiser is free to reassociate it into the surrounding
  // arithmetic, and the 130-token probe moves. The intrinsic pins the same
  // round-to-nearest product the stored value used to be.
  float kv_mem = 0.0f;
  for (int m = 0; m < head_k_dim; ++m) {
    const float decayed = __fmul_rn(state_head[m * head_v_dim + j], g_exp);
    kv_mem += decayed * k_row[m];
  }
  const float delta =
      (__bfloat162float(v[static_cast<int64_t>(head) * head_v_dim + j]) - kv_mem) *
      beta_h;
  float acc = 0.0f;
  for (int m = 0; m < head_k_dim; ++m) {
    // Keep `decayed` a separate product rather than folding it into the sum.
    // Written as one expression the compiler contracts it into an FMA, which
    // rounds differently from the multiply-then-add the stored value used to
    // go through, and the 130-token probe moves. This shape matches what the
    // old second pass computed: a rounded product, then added to k*delta.
    const float decayed = __fmul_rn(state_head[m * head_v_dim + j], g_exp);
    const float updated = decayed + k_row[m] * delta;
    state_head[m * head_v_dim + j] = updated;
    acc += updated * q_row[m];
  }
  out[static_cast<int64_t>(head) * head_v_dim + j] = __float2bfloat16(acc);
}
