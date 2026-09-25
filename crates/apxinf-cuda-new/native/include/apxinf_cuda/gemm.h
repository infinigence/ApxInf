#pragma once

#include "gemm_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct apxinf_gemm_execution* apxinf_gemm_execution_t;

apxinf_status_t apxinf_gemm_prepare(
    apxinf_runtime_t runtime, const apxinf_gemm_spec_t* spec,
    const apxinf_gemm_policy_t* policy,
    const apxinf_gemm_bindings_t* bindings,
    apxinf_gemm_execution_t* execution);
apxinf_status_t apxinf_gemm_enqueue(apxinf_gemm_execution_t execution);
void apxinf_gemm_destroy(apxinf_gemm_execution_t execution);
const char* apxinf_gemm_summary(apxinf_gemm_execution_t execution);

/* Bytes one operand's NVFP4 block-scale buffer occupies in the layout the
   kernel reads. `rows` is M for an activation and N for a weight. Returns 0
   when the block size is unsupported. */
uint64_t apxinf_gemm_nvfp4_scale_buffer_bytes(int64_t rows, int64_t k,
                                              uint32_t sf_vec_size);

/* Rewrite row-major [rows, k/sf_vec_size] E4M3 block scales -- the layout a
   checkpoint stores -- into the layout the kernel reads. The result does not
   depend on the tuned tactic, so it is computed once at load time and stays
   valid if the autotuner later picks a different configuration. */
apxinf_status_t apxinf_gemm_nvfp4_pack_block_scales(
    const void* source_row_major, void* destination, int64_t rows, int64_t k,
    uint32_t sf_vec_size, apxinf_cuda_stream_t stream);

/* Quantize a BF16 activation into the packed-FP4 operand and block scales the
   NVFP4 GEMM consumes. `input_scale` is the checkpoint's per-tensor activation
   scale; block scales are stored relative to it, and the GEMM recovers
   absolute magnitudes by folding it into alpha. */
apxinf_status_t apxinf_gemm_nvfp4_quantize_activation(
    const void* source_bf16, void* destination_packed,
    void* destination_scales, int64_t rows, int64_t k, uint32_t sf_vec_size,
    float input_scale, int32_t row_major_scales,
    apxinf_cuda_stream_t stream);

/* RMSNorm fused with NVFP4 quantization: avoids writing and re-reading the
   normalized BF16 tensor. */
apxinf_status_t apxinf_gemm_nvfp4_quantize_rms_norm(
    const void* source_bf16, const void* norm_weight,
    void* destination_packed, void* destination_scales, int64_t rows,
    int64_t k, uint32_t sf_vec_size, float epsilon, float input_scale,
    int32_t row_major_scales, apxinf_cuda_stream_t stream);

/* SwiGLU over a fused [rows, 2*k] gate/up projection, fused with NVFP4
   quantization. */
apxinf_status_t apxinf_gemm_nvfp4_quantize_swiglu(
    const void* source_bf16, void* destination_packed,
    void* destination_scales, int64_t rows, int64_t k, uint32_t sf_vec_size,
    float input_scale, int32_t row_major_scales,
    apxinf_cuda_stream_t stream);

uint64_t apxinf_gemm_execution_weight_prepack_count(
    apxinf_gemm_execution_t execution);
apxinf_status_t apxinf_gemm_test_validate_candidates(
    apxinf_runtime_t runtime, const apxinf_gemm_spec_t* spec,
    const apxinf_gemm_policy_t* policy,
    const apxinf_gemm_bindings_t* bindings, const float* expected_output,
    uint64_t expected_output_len);

#ifdef __cplusplus
}
#endif
