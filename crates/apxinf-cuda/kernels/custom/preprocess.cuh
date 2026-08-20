#pragma once

// Copyright 2026 apxinf contributors.
// Pure CUDA operators grouped by physical operation; launch policy lives under adapters/.

template <bool kNhwc>
__global__ void rgb_u8_to_patches_e4m3_kernel(
    const uint8_t* images, __nv_fp8_e4m3* patches, int views,
    int image_size, int patch_size, float inverse_scale) {
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

    int64_t input_index;
    if constexpr (kNhwc) {
      input_index =
          ((static_cast<int64_t>(view) * image_size + y) * image_size + x) *
              3 +
          channel;
    } else {
      input_index =
          ((static_cast<int64_t>(view) * 3 + channel) * image_size + y) *
              image_size +
          x;
    }

    // Preserve the established preprocessing contract exactly: normalize in
    // FP32, round the patch value to FP16, then apply the static FP8 scale.
    // This lets the fused raw-image path replace the former FP16 patch tensor
    // without changing the patch projection's numerical input.
    const float normalized = __fsub_rn(
        __fmul_rn(__fdiv_rn(static_cast<float>(images[input_index]), 255.0f),
                  2.0f),
        1.0f);
    const half rounded = __float2half_rn(normalized);
    const float quantized = fminf(
        448.0f,
        fmaxf(-448.0f, __half2float(rounded) * inverse_scale));
    patches[output_index] = static_cast<__nv_fp8_e4m3>(quantized);
  }
}


template <bool kNhwc>
__global__ void rgb_u8_to_patches_bf16_kernel(
    const uint8_t* images, __nv_bfloat16* patches, int views,
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
    const int64_t input_index = kNhwc
        ? ((static_cast<int64_t>(view) * image_size + y) * image_size + x) * 3 + channel
        : ((static_cast<int64_t>(view) * 3 + channel) * image_size + y) * image_size + x;
    const float normalized =
        (static_cast<float>(images[input_index]) / 255.0f) * 2.0f - 1.0f;
    patches[output_index] = __float2bfloat16(normalized);
  }
}



