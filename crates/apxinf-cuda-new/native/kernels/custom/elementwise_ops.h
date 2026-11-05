// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>

// Out-of-place elementwise and activation operators.
//
// These back the portable `apxinf_core::Backend` trait so a model written
// against `dyn Backend` (llama, qwen3-vl) can run on the cuda-new runtime
// without a separate legacy backend. Each has one fixed implementation and no
// tunable choice, so per `doc/adding-new-kernels.md` section 6 they carry no
// candidate registry, tuning key, or autotuner.
//
// The arithmetic matches the legacy `apxinf-cuda` kernels bit for bit: every
// input is widened to f32, the operation runs in f32, and the result rounds
// once to the storage dtype. A model migrated off the legacy backend must
// produce identical bytes, so this is a contract, not a preference.
namespace apxinf::cuda_new::elementwise_ops {

// output[i] = activation(input[i]); `activation` uses the shared pointwise
// encoding (0 none, 1 gelu_tanh, 2 silu). Used for the standalone `silu`
// and `gelu_tanh` trait methods; count is the total element count.
int activation_bf16(const void* input, void* output, long long count,
                    int activation, cudaStream_t stream);

// output[i] = a[i] * b[i], broadcast over `count` elements.
int mul_bf16(const void* a, const void* b, void* output, long long count,
             cudaStream_t stream);

// output[i] = a[i] + b[i].
int add_bf16(const void* a, const void* b, void* output, long long count,
             cudaStream_t stream);

// output[i] = input[i] * factor.
int scale_bf16(const void* input, void* output, long long count, float factor,
               cudaStream_t stream);

// output[r, c] = input[r, c] + bias[c], broadcasting a length-`cols` vector
// over `rows` rows. `rows * cols` must fit the launch grid.
int add_bias_bf16(const void* input, const void* bias, void* output,
                  long long rows, long long cols, cudaStream_t stream);

// output[r, :] = input[indices[r], :]; gathers `rows` whole rows by u32 index.
int gather_rows_bf16(const void* input, const void* indices, void* output,
                     long long rows, long long cols, cudaStream_t stream);

// output[r, :] = row_map[r] == 0xffffffff ? base[r, :]
//                                         : replacement[row_map[r], :].
int replace_rows_bf16(const void* base, const void* replacement,
                      const void* row_map, void* output, long long rows,
                      long long cols, cudaStream_t stream);

// output[i] = bf16(projection[i] + position[token(i), col(i)] + bias?[col(i)])
// with F32 projection/position/bias and `tokens_per_view` tokens per view.
// The legacy vision patch-embedding epilogue.
int bias_position_f32_bf16(const void* projection, const void* bias,
                           const void* position, void* output, long long count,
                           int cols, int tokens_per_view, cudaStream_t stream);

// *out = remap[argmax(logits)], single-block device argmax over BF16 logits
// with a u32 remap table. Tie-break favors the higher index, matching the
// legacy kernel bit for bit.
int argmax_remap_bf16(const void* logits, unsigned int n, const void* remap,
                      void* out, cudaStream_t stream);

}  // namespace apxinf::cuda_new::elementwise_ops
