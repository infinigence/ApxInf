#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
const char* apx_mlx_error(void);
int apx_mlx_stream_new(int32_t gpu, int32_t index, void** out);
void apx_mlx_stream_free(void* stream);
int apx_mlx_sync(void* stream);
int apx_mlx_memory_stats(size_t* active, size_t* cache, size_t* peak);
int apx_mlx_reset_peak_memory(void);
int apx_mlx_clear_cache(void);
int apx_mlx_array_new(void* stream, const int32_t* shape, size_t rank, int32_t dtype, const uint8_t* bytes, size_t len, void** out);
int apx_mlx_array_clone(const void* array, void** out);
void apx_mlx_array_free(void* array);
int apx_mlx_array_info(const void* array, int32_t* shape, size_t capacity, size_t* rank, int32_t* dtype);
int apx_mlx_array_read(void* stream, const void* array, uint8_t* bytes, size_t len);
int apx_mlx_eval(void* stream, const void* const* arrays, size_t count);
int apx_mlx_op(void* stream, int32_t op, const void* const* inputs, size_t count, const int32_t* ints, size_t nints, const float* floats, size_t nfloats, void** out);
typedef int (*apx_mlx_callback)(void*, const void* const*, size_t, void**, size_t);
int apx_mlx_compile_new(apx_mlx_callback callback, void* context, size_t outputs, int32_t shapeless, void** out);
void apx_mlx_compile_free(void* compiled);
int apx_mlx_compile_call(void* compiled, const void* const* inputs, size_t count, void** outputs, size_t output_count);
int apx_mlx_quantize(void* stream, const void* input, int32_t group, int32_t bits, void** outputs);
int apx_mlx_metal_new(const char* name, const char* source, const char* header, const char* const* inputs, size_t input_count, const char* const* outputs, size_t output_count, void** out);
void apx_mlx_metal_free(void* kernel);
int apx_mlx_metal_call(void* stream, void* kernel, const void* const* inputs, size_t input_count, const int32_t* shapes, const size_t* ranks, const int32_t* dtypes, size_t output_count, const int32_t* grid, const int32_t* group, int32_t template_dtype, void** outputs);
#ifdef __cplusplus
}
#endif
