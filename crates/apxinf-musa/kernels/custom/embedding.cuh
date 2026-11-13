#pragma once

// Derived in part from `crates/apxinf-cuda/kernels/custom/` (Apache-2.0,
// Copyright 2026 apxinf contributors); the differences are explained below.

// Embedding kernels, a port of the parts of
// `crates/apxinf-cuda/kernels/custom/embedding.cuh` this port reaches.
//
// CUDA's file has two `embedding_bf16_kernel` overloads with different
// behaviour: a four-argument row gather that copies, and a six-argument one that
// additionally scales by `sqrtf(width)`. Only the second serves PI0.5
// (`_lookup_bf16` in the eager bridge says why: the reference multiplies the
// language lookup by `math.sqrt(width)`). A port that copied the name would pick
// between them by overload resolution, and the wrong pick is a silent factor of
// 45.25 -- so the name states which contract it implements and there is no
// four-argument sibling to select by accident.

// Row gather, scaled by `bf16(sqrt(width))`.
//
// The multiplier is the **bf16 value `sqrtf(width)` rounds to**, applied in f32,
// which is what the bridge does:
//
//     scale = float(torch.tensor(math.sqrt(width), dtype=torch.bfloat16))
//     out   = f32(table.index_select(0, ids)) * scale
//
// For width 2048 that is 45.25 rather than `sqrtf`'s 45.254833. The two agree on
// 0 of 20480 recorded language rows for the bf16-rounded multiplier and disagree
// on 542 for the f32 one, so this is a measured choice, not a rounding taste, and it is
// deliberately *not* what the CUDA kernel's `sqrtf(width)` computes. A native
// MUSA kernel has to do what the recorded reference did, and the CUDA kernel was
// never the reference here.
//
// Out-of-range ids are rejected before the launch rather than written as zeros
// the way CUDA's kernel does; see `kernels/embedding.rs`.
__global__ void embedding_bf16_scaled_kernel(const __mt_bfloat16* table,
                                             const uint32_t* ids,
                                             __mt_bfloat16* output, int tokens,
                                             int width, float normalizer) {
  // Tokens on `gridDim.y`, columns on `gridDim.x`; the table row the token
  // selects is resolved once per token instead of once per element. See
  // `activation.cuh` for what the per-element `index / width` cost.
  const int stride = blockDim.x * gridDim.x;
  for (int token = blockIdx.y; token < tokens; token += gridDim.y) {
    const int64_t table_base = static_cast<int64_t>(ids[token]) * width;
    const int64_t output_base = static_cast<int64_t>(token) * width;
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < width;
         col += stride) {
      output[output_base + col] = __float2bfloat16(
          __bfloat162float(table[table_base + col]) * normalizer);
    }
  }
}
