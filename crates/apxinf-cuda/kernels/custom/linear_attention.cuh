#pragma once

// Copyright 2026 apxinf contributors.
// Pure CUDA operators grouped by physical operation; launch policy lives under adapters/.
//
// Linear-attention / hybrid-recurrent operators: gated delta rule (chunked
// prefill + rank-1 recurrent decode), depthwise causal conv1d with SiLU,
// gated RMSNorm, partial-rotary application from precomputed tables, adaLN
// modulation, and small dtype/layout utilities shared by hybrid models.
//
// Semantics follow the published equations (Qwen3.5 gated delta net, DiT-style
// adaLN diffusion experts); every kernel is model-neutral and parameterized by
// raw geometry (heads, head dims, chunk size).

// small device helpers

__device__ __forceinline__ float la_sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// softplus with the PyTorch threshold=20 convention.
__device__ __forceinline__ float la_softplus(float x) {
  return x > 20.0f ? x : log1pf(expf(x));
}

__device__ __forceinline__ float la_silu(float x) {
  return x / (1.0f + expf(-x));
}

// dtype casts

__global__ void cast_f32_to_bf16_kernel(
    const float* input, __nv_bfloat16* output, int64_t count) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = index; i < count; i += stride) {
    output[i] = __float2bfloat16(input[i]);
  }
}

__global__ void cast_bf16_to_f32_kernel(
    const __nv_bfloat16* input, float* output, int64_t count) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = index; i < count; i += stride) {
    output[i] = __bfloat162float(input[i]);
  }
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
// One thread per (token, channel); the last token's threads also emit the new
// state so state writes never race the reads of other tokens.

__global__ void causal_conv1d_silu_bf16_kernel(
    const __nv_bfloat16* x, const __nv_bfloat16* weight,
    const __nv_bfloat16* state, __nv_bfloat16* out, __nv_bfloat16* new_state,
    int channels, int seq, int kernel_size, int64_t x_row_stride) {
  const int channel = blockIdx.y * blockDim.x + threadIdx.x;
  if (channel >= channels) return;
  const int token = blockIdx.x;
  float acc = 0.0f;
  for (int i = 0; i < kernel_size; ++i) {
    const int src = token - (kernel_size - 1) + i;
    float value = 0.0f;
    if (src >= 0) {
      value = __bfloat162float(x[static_cast<int64_t>(src) * x_row_stride + channel]);
    } else if (state != nullptr) {
      value = __bfloat162float(state[channel * kernel_size + kernel_size + src]);
    }
    acc += value * __bfloat162float(weight[channel * kernel_size + i]);
  }
  out[static_cast<int64_t>(token) * channels + channel] = __float2bfloat16(la_silu(acc));
  if (token == seq - 1) {
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
// q/k rows are L2-normalized (fp32, eps) from the post-conv bf16 stream,
// scaled by 1/sqrt(head_k_dim) for q, and scattered to the value-head layout
// (each key head repeats to num_v_heads/num_k_heads consecutive value heads).
// Outputs are fp32 head-major [num_v_heads, seq_pad, head_k_dim]; the padded
// tail keeps the caller's zero fill. One block per (token, key head) with
// head_k_dim threads.

__global__ void gdn_qk_prep_kernel(
    const __nv_bfloat16* conv_out, float* q_out, float* k_out,
    int seq, int seq_pad, int conv_dim, int key_dim,
    int num_v_heads, int head_k_dim, float scale, float eps) {
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
  const float q_normed = q_value * q_inv * scale;
  const float k_normed = k_value * k_inv;
  for (int r = 0; r < reps; ++r) {
    const int head = k_head * reps + r;
    const int64_t dst = (static_cast<int64_t>(head) * seq_pad + token) * head_k_dim + d;
    q_out[dst] = q_normed;
    k_out[dst] = k_normed;
  }
}

// beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias); v copied to the
// fp32 head-major layout. b_proj/a_proj point at the first row of their column
// slice inside a fused projection output (row stride given in elements).
__global__ void gdn_vb_prep_kernel(
    const __nv_bfloat16* conv_out,
    const __nv_bfloat16* b_proj, const __nv_bfloat16* a_proj,
    const float* dt_bias, const float* a_log,
    float* v_out, float* beta_out, float* g_out,
    int seq, int seq_pad, int conv_dim, int v_offset,
    int ba_row_stride, int head_v_dim) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const int d = threadIdx.x;
  if (d >= head_v_dim) return;
  const int64_t conv_base = static_cast<int64_t>(token) * conv_dim;
  const float b_value = __bfloat162float(b_proj[static_cast<int64_t>(token) * ba_row_stride + head]);
  const float a_value = __bfloat162float(a_proj[static_cast<int64_t>(token) * ba_row_stride + head]);
  const float beta = la_sigmoid(b_value);
  const float g = -expf(a_log[head]) * la_softplus(a_value + dt_bias[head]);
  v_out[(static_cast<int64_t>(head) * seq_pad + token) * head_v_dim + d] =
      __bfloat162float(conv_out[conv_base + v_offset + head * head_v_dim + d]);
  if (d == 0) {
    beta_out[static_cast<int64_t>(head) * seq_pad + token] = beta;
    g_out[static_cast<int64_t>(head) * seq_pad + token] = g;
  }
}

// Chunk-local inclusive cumsum of g: g_cum[h, c*chunk + i] = sum_{j<=i} g.
__global__ void gdn_cumsum_kernel(
    const float* g, float* g_cum, int seq_pad, int chunk_size) {
  const int head = blockIdx.y;
  const int chunk = blockIdx.x;
  if (threadIdx.x != 0) return;
  const int64_t base = static_cast<int64_t>(head) * seq_pad + static_cast<int64_t>(chunk) * chunk_size;
  float running = 0.0f;
  for (int i = 0; i < chunk_size; ++i) {
    running += g[base + i];
    g_cum[base + i] = running;
  }
}

// A1[i,j] = -(k_beta_i . k_j) * exp(g_i - g_j) for j < i else 0
// T[i,j]  =  (q_i . k_j)       * exp(g_i - g_j) for j <= i else 0
// with k_beta_i = k_i * beta_i folded on the fly. One block per (head, chunk).
__global__ void gdn_attn_raw_kernel(
    const float* q, const float* k, const float* beta, const float* g_cum,
    float* a_out, float* t_out,
    int seq_pad, int head_k_dim, int chunk_size) {
  const int head = blockIdx.y;
  const int chunk = blockIdx.x;
  const int64_t token_base = static_cast<int64_t>(head) * seq_pad + static_cast<int64_t>(chunk) * chunk_size;
  const int64_t matrix_base = (static_cast<int64_t>(head) * gridDim.x + chunk) * chunk_size * chunk_size;
  for (int cell = threadIdx.x; cell < chunk_size * chunk_size; cell += blockDim.x) {
    const int i = cell / chunk_size;
    const int j = cell - i * chunk_size;
    const int64_t row_i = (token_base + i) * head_k_dim;
    const int64_t row_j = (token_base + j) * head_k_dim;
    float a1 = 0.0f;
    float a2 = 0.0f;
    const float beta_i = beta[token_base + i];
    for (int d = 0; d < head_k_dim; ++d) {
      const float k_j = k[row_j + d];
      a1 += k[row_i + d] * beta_i * k_j;
      a2 += q[row_i + d] * k_j;
    }
    const float decay = expf(g_cum[token_base + i] - g_cum[token_base + j]);
    a_out[matrix_base + cell] = (j < i) ? -a1 * decay : 0.0f;
    t_out[matrix_base + cell] = (j <= i) ? a2 * decay : 0.0f;
  }
}

// In-place forward substitution over strictly-lower A, then A += I:
// computes (I - A)^{-1} exactly like the reference's sequential loop.
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
    a[matrix_base + cell] = matrix[cell] + (i == j ? 1.0f : 0.0f);
  }
}

// VT[i,j]  = sum_m A[i,m] * (v[m,j] * beta[m])          over head_v_dim columns
// KCD[i,j] = sum_m A[i,m] * (k[m,j] * beta[m] * exp(g_cum[m])) over head_k_dim
__global__ void gdn_chunk_gemm_kernel(
    const float* a, const float* v, const float* k, const float* beta,
    const float* g_cum, float* vt_out, float* kcd_out,
    int seq_pad, int head_k_dim, int head_v_dim, int chunk_size) {
  const int head = blockIdx.y;
  const int chunk = blockIdx.x;
  const int64_t token_base = static_cast<int64_t>(head) * seq_pad + static_cast<int64_t>(chunk) * chunk_size;
  const int64_t a_base = (static_cast<int64_t>(head) * gridDim.x + chunk) * chunk_size * chunk_size;
  const int64_t vt_base = (static_cast<int64_t>(head) * gridDim.x + chunk) * chunk_size * head_v_dim;
  const int64_t kcd_base = (static_cast<int64_t>(head) * gridDim.x + chunk) * chunk_size * head_k_dim;
  for (int cell = threadIdx.x; cell < chunk_size * head_v_dim; cell += blockDim.x) {
    const int i = cell / head_v_dim;
    const int j = cell - i * head_v_dim;
    float vt = 0.0f;
    for (int m = 0; m < chunk_size; ++m) {
      vt += a[a_base + i * chunk_size + m] * (v[(token_base + m) * head_v_dim + j] * beta[token_base + m]);
    }
    vt_out[vt_base + cell] = vt;
  }
  for (int cell = threadIdx.x; cell < chunk_size * head_k_dim; cell += blockDim.x) {
    const int i = cell / head_k_dim;
    const int j = cell - i * head_k_dim;
    float kcd = 0.0f;
    for (int m = 0; m < chunk_size; ++m) {
      kcd += a[a_base + i * chunk_size + m] *
             (k[(token_base + m) * head_k_dim + j] * beta[token_base + m] * expf(g_cum[token_base + m]));
    }
    kcd_out[kcd_base + cell] = kcd;
  }
}

// Sequential chunk recurrence per head (state lives in global scratch):
//   v_prime = KCD @ S;  v_new = VT - v_prime
//   out = (q * exp(g_cum)) @ S + T @ v_new           (bf16, token-major)
//   S = S * exp(g_last) + (k * exp(g_last - g_cum))^T @ v_new
// state is [head_k_dim, head_v_dim] fp32 in global memory, read at the first
// chunk and rewritten per chunk, so the buffer carries the initial state in
// and the final state out. Grid is (num_v_heads); each block loops chunks.
__global__ void gdn_chunk_state_kernel(
    const float* q, const float* k,
    const float* g_cum, const float* t_in, const float* vt_in, const float* kcd_in,
    float* state, __nv_bfloat16* out,
    int seq, int seq_pad, int head_k_dim, int head_v_dim, int chunk_size,
    int total_chunks, int out_row_width) {
  const int head = blockIdx.x;
  extern __shared__ float v_new[];
  const int64_t head_token_base = static_cast<int64_t>(head) * seq_pad;
  float* state_head = state + static_cast<int64_t>(head) * head_k_dim * head_v_dim;
  const int cells = chunk_size * head_v_dim;
  for (int c = 0; c < total_chunks; ++c) {
    const int64_t token_base = head_token_base + static_cast<int64_t>(c) * chunk_size;
    const int64_t vt_base = (static_cast<int64_t>(head) * total_chunks + c) * chunk_size * head_v_dim;
    const int64_t kcd_base = (static_cast<int64_t>(head) * total_chunks + c) * chunk_size * head_k_dim;
    const int64_t a_base = (static_cast<int64_t>(head) * total_chunks + c) * chunk_size * chunk_size;
    float attn_inter[32];
    for (int cell = threadIdx.x, it = 0; cell < cells; cell += blockDim.x, ++it) {
      const int i = cell / head_v_dim;
      const int j = cell - i * head_v_dim;
      float vp = 0.0f;
      float ai = 0.0f;
      const float qg = expf(g_cum[token_base + i]);
      for (int m = 0; m < head_k_dim; ++m) {
        const float s = state_head[m * head_v_dim + j];
        vp += kcd_in[kcd_base + i * head_k_dim + m] * s;
        ai += q[(token_base + i) * head_k_dim + m] * qg * s;
      }
      v_new[cell] = vt_in[vt_base + cell] - vp;
      attn_inter[it] = ai;
    }
    __syncthreads();
    for (int cell = threadIdx.x, it = 0; cell < cells; cell += blockDim.x, ++it) {
      const int i = cell / head_v_dim;
      const int j = cell - i * head_v_dim;
      float acc = attn_inter[it];
      for (int m = 0; m < chunk_size; ++m) {
        acc += t_in[a_base + i * chunk_size + m] * v_new[m * head_v_dim + j];
      }
      const int token = c * chunk_size + i;
      if (token < seq) {
        out[static_cast<int64_t>(token) * out_row_width + head * head_v_dim + j] =
            __float2bfloat16(acc);
      }
    }
    __syncthreads();
    const float g_last = g_cum[token_base + chunk_size - 1];
    const float decay = expf(g_last);
    const int state_cells = head_k_dim * head_v_dim;
    for (int cell = threadIdx.x; cell < state_cells; cell += blockDim.x) {
      const int m = cell / head_v_dim;
      const int j = cell - m * head_v_dim;
      float acc = 0.0f;
      for (int i = 0; i < chunk_size; ++i) {
        const float ksd = k[(token_base + i) * head_k_dim + m] * expf(g_last - g_cum[token_base + i]);
        acc += ksd * v_new[i * head_v_dim + j];
      }
      state_head[cell] = state_head[cell] * decay + acc;
    }
    __syncthreads();
  }
}

// Rank-1 recurrent update for single-token decode:
//   S *= exp(g); kv = S^T k; delta = (v - kv) * beta; S += k x delta; out = S^T q
__global__ void gdn_recurrent_kernel(
    const float* q, const float* k, const float* v, const float* beta,
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
  float kv_mem = 0.0f;
  for (int m = 0; m < head_k_dim; ++m) {
    const float decayed = state_head[m * head_v_dim + j] * g_exp;
    state_head[m * head_v_dim + j] = decayed;
    kv_mem += decayed * k_row[m];
  }
  const float delta = (v[static_cast<int64_t>(head) * head_v_dim + j] - kv_mem) * beta_h;
  float acc = 0.0f;
  for (int m = 0; m < head_k_dim; ++m) {
    const float updated = state_head[m * head_v_dim + j] + k_row[m] * delta;
    state_head[m * head_v_dim + j] = updated;
    acc += updated * q_row[m];
  }
  out[static_cast<int64_t>(head) * head_v_dim + j] = __float2bfloat16(acc);
}

// Gated RMSNorm: y = bf16(rms(x)), y2 = bf16(w * y), out = bf16(y2 * silu(z)).
// x rows are [rows, cols]; z rows are strided slices z[(row/z_heads) *
// z_row_stride + z_col_offset + (row%z_heads)*cols].
__global__ void gated_rms_silu_bf16_kernel(
    const __nv_bfloat16* x, const __nv_bfloat16* z, const __nv_bfloat16* weight,
    __nv_bfloat16* out, int cols, int z_heads, int64_t z_row_stride,
    int64_t z_col_offset, float eps) {
  const int row = blockIdx.x;
  extern __shared__ float la_gated[];
  const int64_t base = static_cast<int64_t>(row) * cols;
  const int64_t z_base = static_cast<int64_t>(row / z_heads) * z_row_stride + z_col_offset
      + static_cast<int64_t>(row % z_heads) * cols;
  float partial = 0.0f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    const float v = __bfloat162float(x[base + i]);
    la_gated[i] = v;
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
  const float rms = rsqrtf(warp_sums[0] / cols + eps);
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    const float y0 = __bfloat162float(__float2bfloat16(la_gated[i] * rms));
    const float y1 = __bfloat162float(__float2bfloat16(__bfloat162float(weight[i]) * y0));
    const float zf = __bfloat162float(z[z_base + i]);
    out[base + i] = __float2bfloat16(y1 * la_silu(zf));
  }
}

// RMSNorm with zero-init (1 + w) weights: out = rms(x) * (1 + w), fp32 compute.
__global__ void rms_norm_plus1_bf16_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* weight, __nv_bfloat16* output,
    int cols, float eps) {
  const int row = blockIdx.x;
  extern __shared__ float la_plus1[];
  const int64_t base = static_cast<int64_t>(row) * cols;
  float partial = 0.0f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    const float v = __bfloat162float(input[base + i]);
    la_plus1[i] = v;
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
  const float rms = rsqrtf(warp_sums[0] / cols + eps);
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    output[base + i] = __float2bfloat16(
        la_plus1[i] * rms * (1.0f + __bfloat162float(weight[i])));
  }
}

// Partial rotary from precomputed bf16 cos/sin tables [rows, rotary_dim]:
// rotate the leading rotary_dim channels (pairs (p, p + rotary/2)) with the
// torch elementwise rounding chain (mul -> round, mul -> round, add -> round),
// pass the remaining channels through. Works in place (out may alias x).
__global__ void partial_rope_table_bf16_kernel(
    const __nv_bfloat16* x, const __nv_bfloat16* cos, const __nv_bfloat16* sin,
    __nv_bfloat16* out, int heads, int head_dim, int rotary_dim) {
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int half = rotary_dim / 2;
  const int64_t table_base = static_cast<int64_t>(row) * rotary_dim;
  const int64_t base = (static_cast<int64_t>(row) * heads + head) * head_dim;
  for (int p = threadIdx.x; p < half; p += blockDim.x) {
    const float a = __bfloat162float(x[base + p]);
    const float b = __bfloat162float(x[base + half + p]);
    const float c = __bfloat162float(cos[table_base + p]);
    const float s = __bfloat162float(sin[table_base + p]);
    const float t1 = __bfloat162float(__float2bfloat16(a * c));
    const float t2 = __bfloat162float(__float2bfloat16(-b * s));
    const float t3 = __bfloat162float(__float2bfloat16(b * c));
    const float t4 = __bfloat162float(__float2bfloat16(a * s));
    out[base + p] = __float2bfloat16(t1 + t2);
    out[base + half + p] = __float2bfloat16(t3 + t4);
  }
  for (int d = rotary_dim + threadIdx.x; d < head_dim; d += blockDim.x) {
    out[base + d] = x[base + d];
  }
}

// Fused full-attention input preparation for the (q|gate)-per-head layout:
// per (token, head): per-head RMSNorm (1 + w semantics) on q/k, partial rotary
// from tables, K/V cache append. The gate half of each q head stays in the
// fused buffer for the post-attention sigmoid gate.
// Grid y: [0, q_heads) -> q, [q_heads, q_heads+kv_heads) -> k, rest -> v.
__global__ void full_attn_prepare_bf16_kernel(
    const __nv_bfloat16* fused, const __nv_bfloat16* q_norm_w, const __nv_bfloat16* k_norm_w,
    const __nv_bfloat16* cos, const __nv_bfloat16* sin,
    __nv_bfloat16* q_out, __nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
    int cache_offset, int q_heads, int kv_heads, int head_dim,
    int rotary_dim, int64_t fused_width, int64_t cache_width, float eps) {
  const int token = blockIdx.x;
  const int slot = blockIdx.y;
  const int half = rotary_dim / 2;
  extern __shared__ float la_attn[];
  const int64_t row = static_cast<int64_t>(token) * fused_width;
  const int64_t table_base = static_cast<int64_t>(token) * rotary_dim;
  if (slot < q_heads + kv_heads) {
    const bool is_q = slot < q_heads;
    const int head = is_q ? slot : slot - q_heads;
    const __nv_bfloat16* norm_w = is_q ? q_norm_w : k_norm_w;
    const int64_t src = row + (is_q
        ? static_cast<int64_t>(head) * 2 * head_dim
        : static_cast<int64_t>(q_heads) * 2 * head_dim + head * head_dim);
    float partial = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
      const float v = __bfloat162float(fused[src + i]);
      la_attn[i] = v;
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
    const float rms = rsqrtf(warp_sums[0] / head_dim + eps);
    __syncthreads();
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
      la_attn[i] = __bfloat162float(__float2bfloat16(
          la_attn[i] * rms * (1.0f + __bfloat162float(norm_w[i]))));
    }
    __syncthreads();
    __nv_bfloat16* dst = is_q
        ? q_out + (static_cast<int64_t>(token) * q_heads + head) * head_dim
        : k_cache + (static_cast<int64_t>(cache_offset + token)) * cache_width + head * head_dim;
    for (int p = threadIdx.x; p < half; p += blockDim.x) {
      const float a = la_attn[p];
      const float b = la_attn[half + p];
      const float c = __bfloat162float(cos[table_base + p]);
      const float s = __bfloat162float(sin[table_base + p]);
      const float t1 = __bfloat162float(__float2bfloat16(a * c));
      const float t2 = __bfloat162float(__float2bfloat16(-b * s));
      const float t3 = __bfloat162float(__float2bfloat16(b * c));
      const float t4 = __bfloat162float(__float2bfloat16(a * s));
      dst[p] = __float2bfloat16(t1 + t2);
      dst[half + p] = __float2bfloat16(t3 + t4);
    }
    for (int d = rotary_dim + threadIdx.x; d < head_dim; d += blockDim.x) {
      dst[d] = __float2bfloat16(la_attn[d]);
    }
  } else {
    const int head = slot - q_heads - kv_heads;
    const int64_t src = row + static_cast<int64_t>(q_heads) * 2 * head_dim
        + static_cast<int64_t>(kv_heads) * head_dim + head * head_dim;
    __nv_bfloat16* dst = v_cache + (static_cast<int64_t>(cache_offset + token)) * cache_width
        + head * head_dim;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
      dst[i] = fused[src + i];
    }
  }
}

// Post-attention output gate: attn *= sigmoid(gate), with the gate read from
// the (q|gate) interleaved fused projection (column h*2*head_dim + head_dim).
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
    const float s = __bfloat162float(__float2bfloat16(la_sigmoid(gate)));
    attn[base + idx] = __float2bfloat16(__bfloat162float(attn[base + idx]) * s);
  }
}

// adaLN normalization: out = bf16(bf16(rms(x)*w) * bf16(1 + scale) + shift).
__global__ void adaln_rms_norm_bf16_kernel(
    const __nv_bfloat16* x, const __nv_bfloat16* weight,
    const __nv_bfloat16* scale, const __nv_bfloat16* shift,
    __nv_bfloat16* out, int cols, float eps) {
  const int row = blockIdx.x;
  extern __shared__ float la_adaln[];
  const int64_t base = static_cast<int64_t>(row) * cols;
  float partial = 0.0f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    const float v = __bfloat162float(x[base + i]);
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
  const float rms = rsqrtf(warp_sums[0] / cols + eps);
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    const float normed = __bfloat162float(__float2bfloat16(
        la_adaln[i] * rms * __bfloat162float(weight[i])));
    const float multiplier = __bfloat162float(
        __float2bfloat16(1.0f + __bfloat162float(scale[i])));
    const float scaled = __bfloat162float(__float2bfloat16(normed * multiplier));
    out[base + i] = __float2bfloat16(scaled + __bfloat162float(shift[i]));
  }
}

// adaLN residual: out = bf16(residual + bf16(proj * bf16(1 + gate))).
__global__ void adaln_gate_residual_bf16_kernel(
    const __nv_bfloat16* proj, const __nv_bfloat16* residual,
    const __nv_bfloat16* gate, __nv_bfloat16* out, int64_t count, int cols) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const int col = static_cast<int>(index % cols);
    const float multiplier = __bfloat162float(
        __float2bfloat16(1.0f + __bfloat162float(gate[col])));
    const float projected = __bfloat162float(
        __float2bfloat16(__bfloat162float(proj[index]) * multiplier));
    out[index] = __float2bfloat16(__bfloat162float(residual[index]) + projected);
  }
}

// Expert fused-QKV preparation: per (token, group-major slot) split the fused
// projection into query/gate/key/value, apply per-head RMSNorm (plain weight)
// and partial rotary to q/k. Grid y covers q_heads + kv_heads + kv_heads slots.
__global__ void expert_qkv_prepare_bf16_kernel(
    const __nv_bfloat16* fused, const __nv_bfloat16* q_norm_w, const __nv_bfloat16* k_norm_w,
    const __nv_bfloat16* cos, const __nv_bfloat16* sin,
    __nv_bfloat16* q_out, __nv_bfloat16* gate_out, __nv_bfloat16* k_out, __nv_bfloat16* v_out,
    int q_heads, int kv_heads, int head_dim, int rotary_dim,
    int64_t fused_width, float eps) {
  const int token = blockIdx.x;
  const int slot = blockIdx.y;
  const int half = rotary_dim / 2;
  const int heads_per_group = q_heads / kv_heads;
  const int group_width = (2 * heads_per_group + 2) * head_dim;
  const int64_t row = static_cast<int64_t>(token) * fused_width;
  const int64_t table_base = static_cast<int64_t>(token) * rotary_dim;
  extern __shared__ float la_expert[];
  if (slot < q_heads + kv_heads) {
    const bool is_q = slot < q_heads;
    const int head = is_q ? slot : slot - q_heads;
    const int group = is_q ? head / heads_per_group : head;
    const int64_t src = row + static_cast<int64_t>(group) * group_width
        + (is_q ? static_cast<int64_t>(head - group * heads_per_group) * head_dim
                : static_cast<int64_t>(2 * heads_per_group) * head_dim);
    const __nv_bfloat16* norm_w = is_q ? q_norm_w : k_norm_w;
    float partial = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
      const float v = __bfloat162float(fused[src + i]);
      la_expert[i] = v;
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
    const float rms = rsqrtf(warp_sums[0] / head_dim + eps);
    __syncthreads();
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
      la_expert[i] = __bfloat162float(__float2bfloat16(
          la_expert[i] * rms * __bfloat162float(norm_w[i])));
    }
    __syncthreads();
    __nv_bfloat16* dst = is_q
        ? q_out + (static_cast<int64_t>(token) * q_heads + head) * head_dim
        : k_out + (static_cast<int64_t>(token) * kv_heads + head) * head_dim;
    for (int p = threadIdx.x; p < half; p += blockDim.x) {
      const float a = la_expert[p];
      const float b = la_expert[half + p];
      const float c = __bfloat162float(cos[table_base + p]);
      const float s = __bfloat162float(sin[table_base + p]);
      const float t1 = __bfloat162float(__float2bfloat16(a * c));
      const float t2 = __bfloat162float(__float2bfloat16(-b * s));
      const float t3 = __bfloat162float(__float2bfloat16(b * c));
      const float t4 = __bfloat162float(__float2bfloat16(a * s));
      dst[p] = __float2bfloat16(t1 + t2);
      dst[half + p] = __float2bfloat16(t3 + t4);
    }
    for (int d = rotary_dim + threadIdx.x; d < head_dim; d += blockDim.x) {
      dst[d] = __float2bfloat16(la_expert[d]);
    }
    if (is_q) {
      __nv_bfloat16* gate_dst = gate_out + (static_cast<int64_t>(token) * q_heads + head) * head_dim;
      const int64_t gate_src = row + static_cast<int64_t>(group) * group_width
          + static_cast<int64_t>(heads_per_group + head - group * heads_per_group) * head_dim;
      for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        gate_dst[i] = fused[gate_src + i];
      }
    }
  } else {
    const int head = slot - q_heads - kv_heads;
    const int64_t src = row + static_cast<int64_t>(head) * group_width
        + static_cast<int64_t>(2 * heads_per_group + 1) * head_dim;
    __nv_bfloat16* dst = v_out + (static_cast<int64_t>(token) * kv_heads + head) * head_dim;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
      dst[i] = fused[src + i];
    }
  }
}

// Expert post-attention gate from a standalone gate tensor [rows, heads*dim].
__global__ void expert_sigmoid_gate_mul_bf16_kernel(
    __nv_bfloat16* attn, const __nv_bfloat16* gate, int64_t count) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const float g = __bfloat162float(gate[index]);
    const float s = __bfloat162float(__float2bfloat16(la_sigmoid(g)));
    attn[index] = __float2bfloat16(__bfloat162float(attn[index]) * s);
  }
}

// Fourier waypoint features: per channel, cat(sin(w*f*2pi), cos(w*f*2pi)).
__global__ void fourier_features_bf16_kernel(
    const __nv_bfloat16* waypoints, const __nv_bfloat16* freqs, __nv_bfloat16* out,
    int point_dim, int num_features) {
  const int token = blockIdx.x;
  const int width = point_dim * num_features * 2;
  const int64_t base = static_cast<int64_t>(token) * width;
  for (int idx = threadIdx.x; idx < point_dim * num_features; idx += blockDim.x) {
    const int c = idx / num_features;
    const int f = idx - c * num_features;
    const float w = __bfloat162float(waypoints[static_cast<int64_t>(token) * point_dim + c]);
    const float freq = __bfloat162float(freqs[f]);
    const float angle = (w * freq) * 6.2831855f;
    out[base + c * 2 * num_features + f] = __float2bfloat16(sinf(angle));
    out[base + c * 2 * num_features + num_features + f] = __float2bfloat16(cosf(angle));
  }
}

// Column concat of seven same-width sources into one [rows, 7*cols] buffer;
// a set broadcast bit replicates source row 0 for every output row.
__global__ void concat7_cols_bf16_kernel(
    const __nv_bfloat16* s0, const __nv_bfloat16* s1, const __nv_bfloat16* s2,
    const __nv_bfloat16* s3, const __nv_bfloat16* s4, const __nv_bfloat16* s5,
    const __nv_bfloat16* s6, __nv_bfloat16* dst, int rows, int cols,
    int broadcast_mask) {
  const int token = blockIdx.x;
  const int64_t dst_base = static_cast<int64_t>(token) * 7 * cols;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    const int64_t src_idx = static_cast<int64_t>(token) * cols + i;
    const int64_t bcast_idx = i;
    dst[dst_base + i] = (broadcast_mask & 1) ? s0[bcast_idx] : s0[src_idx];
    dst[dst_base + cols + i] = (broadcast_mask & 2) ? s1[bcast_idx] : s1[src_idx];
    dst[dst_base + 2 * cols + i] = (broadcast_mask & 4) ? s2[bcast_idx] : s2[src_idx];
    dst[dst_base + 3 * cols + i] = (broadcast_mask & 8) ? s3[bcast_idx] : s3[src_idx];
    dst[dst_base + 4 * cols + i] = (broadcast_mask & 16) ? s4[bcast_idx] : s4[src_idx];
    dst[dst_base + 5 * cols + i] = (broadcast_mask & 32) ? s5[bcast_idx] : s5[src_idx];
    dst[dst_base + 6 * cols + i] = (broadcast_mask & 64) ? s6[bcast_idx] : s6[src_idx];
  }
}

// Flow-matching Euler update: w += (endpoint - w) / remaining * step (fp32).
__global__ void flow_update_f32_kernel(
    float* w, const float* endpoint, float remaining, float step, int64_t count) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const float delta = endpoint[index] - w[index];
    w[index] = w[index] + (delta / remaining) * step;
  }
}

// Set the listed logits columns to -inf (min-new-tokens EOS suppression).
__global__ void suppress_logits_bf16_kernel(
    __nv_bfloat16* logits, const uint32_t* ids, int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  logits[ids[idx]] = __float2bfloat16(-INFINITY);
}

// Exact (erf) GELU, fp32 compute, bf16 storage.
__global__ void gelu_exact_bf16_kernel(
    const __nv_bfloat16* input, __nv_bfloat16* output, int64_t count) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const float x = __bfloat162float(input[index]);
    output[index] = __float2bfloat16(x * 0.5f * (1.0f + erff(x * 0.70710678f)));
  }
}

// Merge `factor` consecutive rows into one: out[r, s*cols + c] = in[r*factor+s, c].
__global__ void merge_rows_bf16_kernel(
    const __nv_bfloat16* input, __nv_bfloat16* output, int64_t out_count, int cols, int factor) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < out_count; index += stride) {
    const int64_t r = index / (static_cast<int64_t>(factor) * cols);
    const int64_t rem = index - r * factor * cols;
    const int64_t s = rem / cols;
    const int64_t c = rem - s * cols;
    output[index] = input[(r * factor + s) * cols + c];
  }
}
