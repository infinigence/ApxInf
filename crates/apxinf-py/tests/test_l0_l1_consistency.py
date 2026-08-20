"""L0/L1 layering consistency: external patchify → L0 matches raw RGB → L1.

L1 runs vision→patches inside the CUDA graph; L0 takes pre-computed patches.
Feeding the same image both ways must agree within tolerance. This is only
exact for the BF16 path, whose kernel is a plain ``(u8/255)*2 - 1`` normalize
(``static_bf16.cu``); the FP8 path additionally quantizes with a vision scale,
so this test skips unless precision is bf16.
"""

import os

import numpy as np
import pytest

from conftest import make_noise, make_tokens


def patchify_bf16(rgb_nhwc: np.ndarray, patch_size: int) -> np.ndarray:
    """Reference for ``rgb_u8_to_patches_bf16_kernel`` (NHWC input).

    Normalizes to [-1, 1] and lays out each patch as [channel, dy, dx], with
    patches ordered row-major (patch_y, patch_x) per view — matching the kernel.
    """
    views, height, width, channels = rgb_nhwc.shape
    per_side = height // patch_size
    norm = (rgb_nhwc.astype(np.float32) / 255.0) * 2.0 - 1.0
    # [views, patch_y, dy, patch_x, dx, channel]
    grid = norm.reshape(views, per_side, patch_size, per_side, patch_size, channels)
    # -> [views, patch_y, patch_x, channel, dy, dx]
    grid = grid.transpose(0, 1, 3, 5, 2, 4)
    return grid.reshape(views * per_side * per_side, channels * patch_size * patch_size)


def test_l0_l1_consistency(model, precision):
    if precision != "bf16":
        pytest.skip("L0/L1 exact consistency reference is defined for bf16 only")

    rng = np.random.default_rng(1234)
    rgb = rng.integers(
        0, 256, size=(model.num_views, model.image_size, model.image_size, 3), dtype=np.uint8
    )
    tokens = make_tokens(model)
    noise = make_noise(model, seed=7)

    patches = np.ascontiguousarray(patchify_bf16(rgb, model.patch_size), dtype=np.float32)
    assert patches.shape == (
        model.num_views * model.patches_per_view,
        3 * model.patch_size * model.patch_size,
    )

    action_l0 = model._infer_patches(patches, tokens, noise)
    action_l1 = model.infer_rgb(np.ascontiguousarray(rgb), "nhwc", tokens, noise)

    atol = float(os.environ.get("APXINF_PI05_CONSISTENCY_ATOL", "2e-3"))
    np.testing.assert_allclose(action_l0, action_l1, atol=atol, rtol=0.0)
