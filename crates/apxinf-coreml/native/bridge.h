#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
const char* apx_coreml_error(void);
/* Compiles `package` (.mlpackage) into a new directory `compiled` (.mlmodelc), which must not exist. */
int apx_coreml_compile(const char* package, const char* compiled);
/* compute_units: 0 all, 1 CPU only, 2 CPU and GPU, 3 CPU and Neural Engine. */
int apx_coreml_load(const char* compiled, int32_t compute_units, void** out);
void apx_coreml_free(void* model);
/* FP16 inputs and outputs, row-major and contiguous; output strides reported by Core ML are honoured. */
int apx_coreml_predict(void* model, const char* const* input_names, const uint16_t* const* inputs,
                       const int64_t* const* input_shapes, const size_t* input_ranks, size_t input_count,
                       const char* const* output_names, uint16_t* const* outputs, const int64_t* const* output_shapes,
                       const size_t* output_ranks, size_t output_count);
#ifdef __cplusplus
}
#endif
