#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace apxinf::gather::kernels {

template <class T>
__device__ float to_float(T value);

template <>
__device__ inline float to_float(__half value) {
  return __half2float(value);
}

template <>
__device__ inline float to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <class T>
__device__ T from_float(float value);

template <>
__device__ inline __half from_float(float value) {
  return __float2half(value);
}

template <>
__device__ inline __nv_bfloat16 from_float(float value) {
  return __float2bfloat16(value);
}

// Token-embedding gather, scaled by sqrt(width).
//
// The scale is not optional: the Gemma backbone folds it into the embedding
// lookup.  The legacy backend also carries an unscaled same-named overload
// that differs only by `uint32_t` versus `int` parameters; dropping the scale
// silently destroys the model output, so this port keeps one signature and
// always scales.  Out-of-range ids produce zeros rather than reading past the
// table.
template <class T>
__global__ void embedding_lookup(const T* table, const uint32_t* ids, T* output,
                                 int tokens, int width, int vocab_size) {
  const int64_t count = static_cast<int64_t>(tokens) * width;
  const float normalizer = sqrtf(static_cast<float>(width));
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const int token = static_cast<int>(index / width);
    const int col = static_cast<int>(index % width);
    const uint32_t id = ids[token];
    output[index] =
        id < static_cast<uint32_t>(vocab_size)
            ? from_float<T>(
                  to_float(table[static_cast<int64_t>(id) * width + col]) *
                  normalizer)
            : from_float<T>(0.0f);
  }
}

// Vision patch-embedding epilogue: projection + learned position embedding,
// plus an optional per-column bias.  `position` is [tokens_per_view, cols] and
// repeats across views.
template <class T>
__global__ void bias_position(const T* projection, const T* bias,
                              const T* position, T* output, int64_t count,
                              int cols, int tokens_per_view) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const int col = static_cast<int>(index % cols);
    const int token = static_cast<int>((index / cols) % tokens_per_view);
    float value = to_float(projection[index]) +
                  to_float(position[static_cast<int64_t>(token) * cols + col]);
    if (bias != nullptr) value += to_float(bias[col]);
    output[index] = from_float<T>(value);
  }
}

// Cuts RGB images into flattened patches and maps [0, 255] to [-1, 1].
// Output is [views * patches_per_view, 3 * patch_size * patch_size].
template <class T, bool kNhwc>
__global__ void rgb_u8_to_patches(const uint8_t* images, T* patches, int views,
                                  int image_size, int patch_size) {
  const int patches_per_side = image_size / patch_size;
  const int patches_per_view = patches_per_side * patches_per_side;
  const int patch_area = patch_size * patch_size;
  const int patch_width = 3 * patch_area;
  const int64_t count =
      static_cast<int64_t>(views) * patches_per_view * patch_width;
  int64_t output_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; output_index < count; output_index += stride) {
    const int patch_element = static_cast<int>(output_index % patch_width);
    const int patch_index = static_cast<int>(output_index / patch_width);
    const int view = patch_index / patches_per_view;
    const int patch_in_view = patch_index - view * patches_per_view;
    const int patch_y = patch_in_view / patches_per_side;
    const int patch_x = patch_in_view - patch_y * patches_per_side;
    const int channel = patch_element / patch_area;
    const int pixel_in_patch = patch_element - channel * patch_area;
    const int dy = pixel_in_patch / patch_size;
    const int dx = pixel_in_patch - dy * patch_size;
    const int y = patch_y * patch_size + dy;
    const int x = patch_x * patch_size + dx;
    const int64_t input_index =
        kNhwc ? ((static_cast<int64_t>(view) * image_size + y) * image_size +
                 x) * 3 + channel
              : ((static_cast<int64_t>(view) * 3 + channel) * image_size + y) *
                        image_size + x;
    const float normalized =
        (static_cast<float>(images[input_index]) / 255.0f) * 2.0f - 1.0f;
    patches[output_index] = from_float<T>(normalized);
  }
}

}  // namespace apxinf::gather::kernels
