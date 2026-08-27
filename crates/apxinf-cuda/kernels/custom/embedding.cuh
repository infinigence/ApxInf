#pragma once

// Copyright 2026 apxinf contributors.
// Pure CUDA operators grouped by physical operation; launch policy lives under adapters/.

// ── Embedding ─────────────────────────────────────────────────────────────

__global__ void embedding_f32_kernel(
    const float* table, const uint32_t* ids, float* output,
    uint32_t embed_dim, uint32_t seq_len)
{
    uint32_t dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t seq_idx = blockIdx.y;
    if (dim_idx >= embed_dim || seq_idx >= seq_len) return;

    uint32_t token_id = ids[seq_idx];
    output[seq_idx * embed_dim + dim_idx] = table[token_id * embed_dim + dim_idx];
}



// ── Embedding (bf16) — table and output are bf16, ids stay u32 ────────────

__global__ void embedding_bf16_kernel(
    const __nv_bfloat16* table, const uint32_t* ids, __nv_bfloat16* output,
    uint32_t embed_dim, uint32_t seq_len)
{
    uint32_t dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t seq_idx = blockIdx.y;
    if (dim_idx >= embed_dim || seq_idx >= seq_len) return;

    uint32_t token_id = ids[seq_idx];
    output[seq_idx * embed_dim + dim_idx] = table[token_id * embed_dim + dim_idx];
}




__global__ void embedding_f16_kernel(
    const half* table, const uint32_t* ids, half* output,
    int tokens, int width, int vocab_size) {
  int64_t count = static_cast<int64_t>(tokens) * width;
  float normalizer = sqrtf(static_cast<float>(width));
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    int token = static_cast<int>(index / width);
    int col = static_cast<int>(index % width);
    uint32_t id = ids[token];
    output[index] = id < static_cast<uint32_t>(vocab_size)
        ? __float2half(__half2float(
              table[static_cast<int64_t>(id) * width + col]) * normalizer)
        : __float2half(0.0f);
  }
}


__global__ void embedding_bf16_kernel(
    const __nv_bfloat16* table, const uint32_t* ids, __nv_bfloat16* output,
    int tokens, int width, int vocab_size) {
  const int64_t count = static_cast<int64_t>(tokens) * width;
  const float normalizer = sqrtf(static_cast<float>(width));
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const int token = static_cast<int>(index / width);
    const int col = static_cast<int>(index % width);
    const uint32_t id = ids[token];
    output[index] = id < static_cast<uint32_t>(vocab_size)
        ? __float2bfloat16(__bfloat162float(
              table[static_cast<int64_t>(id) * width + col]) * normalizer)
        : __float2bfloat16(0.0f);
  }
}


// Explicitly scaled BF16 embedding. The caller owns the scale's dtype
// semantics; for example, Gemma-style runtimes pass sqrt(width) after rounding
// that scalar to the checkpoint activation dtype. Keeping the scalar explicit
// avoids silently imposing one model family's scaling convention here.
__global__ void embedding_scaled_bf16_kernel(
    const __nv_bfloat16* table, const uint32_t* ids, __nv_bfloat16* output,
    int tokens, int width, int vocab_size, float scale) {
  const int64_t count = static_cast<int64_t>(tokens) * width;
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const int token = static_cast<int>(index / width);
    const int col = static_cast<int>(index % width);
    const uint32_t id = ids[token];
    output[index] = id < static_cast<uint32_t>(vocab_size)
        ? __float2bfloat16_rn(
              __bfloat162float(table[static_cast<int64_t>(id) * width + col]) *
              scale)
        : __float2bfloat16_rn(0.0f);
  }
}


