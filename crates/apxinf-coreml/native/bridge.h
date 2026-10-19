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
typedef struct { const uint8_t* data; size_t len; } ApxPlanString;
typedef struct { int32_t kind; ApxPlanString class_name; } ApxPlanDevice;
typedef struct { int32_t name_present; ApxPlanString name; int32_t literal_present; } ApxPlanBinding;
typedef struct { ApxPlanString name; const ApxPlanBinding* bindings; size_t count; } ApxPlanInput;
typedef struct { ApxPlanString name; size_t root_operations; } ApxPlanFunction;
typedef struct {
    size_t function_index;
    const size_t* path; size_t path_len;
    ApxPlanString operator_name;
    const ApxPlanInput* inputs; size_t input_count;
    const ApxPlanString* outputs; size_t output_count;
    const size_t* child_operations; size_t child_count;
    int32_t usage_present;
    const ApxPlanDevice* supported; size_t supported_count;
    int32_t preferred_present; ApxPlanDevice preferred;
} ApxPlanRow;
typedef int32_t (*ApxPlanVisitor)(void*, const ApxPlanFunction*, const ApxPlanRow*);
int apx_coreml_inspect_plan(const char* compiled, int32_t units, uint64_t timeout_ms,
                          ApxPlanVisitor visitor, void* context);
/* FP16 inputs and outputs, row-major and contiguous; output strides reported by Core ML are honoured. */
int apx_coreml_predict(void* model, const char* const* input_names, const uint16_t* const* inputs,
                       const int64_t* const* input_shapes, const size_t* input_ranks, size_t input_count,
                       const char* const* output_names, uint16_t* const* outputs, const int64_t* const* output_shapes,
                       const size_t* output_ranks, size_t output_count);
/* One Core ML batch prediction (predictionsFromBatch) of `batch` items with the same names and shapes:
   item b's inputs are inputs[b * input_count + i], its outputs outputs[b * output_count + o]. */
int apx_coreml_predict_batch(void* model, size_t batch, const char* const* input_names, const uint16_t* const* inputs,
                             const int64_t* const* input_shapes, const size_t* input_ranks, size_t input_count,
                             const char* const* output_names, uint16_t* const* outputs,
                             const int64_t* const* output_shapes, const size_t* output_ranks, size_t output_count);
#ifdef __cplusplus
}
#endif
