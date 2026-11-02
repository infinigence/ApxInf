#pragma once

// Derived in part from `crates/apxinf-cuda/kernels/custom/` (Apache-2.0,
// Copyright 2026 apxinf contributors); the differences are explained below.

// Raw RGB to BF16 patch layout, a port of
// `crates/apxinf-cuda/kernels/custom/preprocess.cuh`'s
// `rgb_u8_to_patches_bf16_kernel`. PI0.5 BF16 reaches it from an `RgbU8`
// observation, inside the block of work the engine treats as one inference.
//
// The contract is one gather and one affine map, and nothing else. There is no
// resize and no interpolation -- the frames arrive already resized to the
// square `image_size` -- no patch embedding (that is a GEMM downstream), and
// **no per-channel mean/std**. PaliGemma's SigLIP patch embedding was trained
// on `[-1, 1]`, so "helpfully" adding a statistic here would produce a wrong
// number, not a missing feature.
//
// Layout: `patches[output_index]` for a fixed `output_index` is
// view -> (patch_y, patch_x) row-major -> channel -> (dy, dx), each patch row
// being `3 * patch_size^2` wide. `images` is the dense NHWC or NCHW batch, so
// `count == views * 3 * image_size^2` is simultaneously the input byte count,
// the output element count, and a bijection between them -- which is what makes
// the flat index below sufficient rather than approximate.
//
// Arithmetic: the affine map is spelled with explicit rounding intrinsics.
// CUDA's bf16 kernel writes it bare (`x / 255.0f * 2.0f - 1.0f`) and happens to
// be correct, but only because `* 2` is exact in binary floating point, so an
// FMA contraction of `x * 2 - 1` cannot steal a rounding. That is a property of
// the constant, not of the source: it does not survive the compiler rewriting
// `x / 255 * 2` as `x * (2 / 255)`, nor a reciprocal-multiply division. The f32
// and e4m3 siblings in the CUDA file already spell the intrinsics out; doing
// the same here makes the contract explicit instead of accidental.
// `__float2bfloat16` is round-to-nearest-even (`musa_bf16.h:127-136`); what
// pins that is this module's device test, which asserts bit equality against a
// host reference, not the spelling.
//
// The launch stays a flat grid-stride over `count` rather than a `row_grid`
// two-dimensional decomposition. The map is elementwise and runs once per
// inference next to the vision tower, so the only reason to split it would be a
// profile showing it on the critical path. `elementwise_grid` caps the grid, so
// the loop is required regardless of the shape.
//
// **This file must be `#include`d by `adapters/custom_kernels.cu`**, inside its
// anonymous namespace: `build.rs` globs `*.cu` only, so a `.cuh` nothing
// includes is neither compiled nor part of the kernel build id.

// Template parameter: `kNhwc` says whether `images` is `[views, H, W, 3]`
// (true) or `[views, 3, H, W]` (false). The adapter picks the instantiation
// from the ABI's layout code and rejects anything else, so an unknown code
// cannot reach a kernel that would silently misread the buffer.
template <bool kNhwc>
__global__ void rgb_u8_to_patches_bf16_kernel(const uint8_t* images,
                                              __mt_bfloat16* patches, int views,
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
        kNhwc ? ((static_cast<int64_t>(view) * image_size + y) * image_size + x) * 3 +
                    channel
              : ((static_cast<int64_t>(view) * 3 + channel) * image_size + y) *
                        image_size +
                    x;
    const float normalized = __fsub_rn(
        __fmul_rn(
            __fdiv_rn(static_cast<float>(images[input_index]), 255.0f), 2.0f),
        1.0f);
    patches[output_index] = __float2bfloat16(normalized);
  }
}
