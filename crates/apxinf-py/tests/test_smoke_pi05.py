"""PyO3 binding smoke test: load + L0 + L1, shapes/dtypes, error handling.

Covers Phase 1 acceptance: load succeeds, L0 `_infer_patches` (internal, private)
and L1 `infer_rgb` each run, return numpy float32 of shape
[action_horizon, action_dim], shape contract properties are readable, and bad
inputs raise clear exceptions.
"""

import numpy as np
import pytest

from conftest import make_noise, make_tokens


def test_shape_contract(model):
    # All contract properties are positive ints readable without inference.
    assert model.action_dim > 0
    assert model.action_horizon > 0
    assert model.num_views > 0
    assert model.image_size > 0
    assert model.patch_size > 0
    assert model.patches_per_view == (model.image_size // model.patch_size) ** 2
    assert model.max_token_len > 0


def test_l0_infer_patches(model):
    patch_rows = model.num_views * model.patches_per_view
    patch_width = 3 * model.patch_size * model.patch_size
    patches = np.zeros((patch_rows, patch_width), dtype=np.float32)
    tokens = make_tokens(model)
    noise = make_noise(model)

    action = model._infer_patches(patches, tokens, noise)

    assert isinstance(action, np.ndarray)
    assert action.dtype == np.float32
    assert action.shape == (model.action_horizon, model.action_dim)
    assert np.all(np.isfinite(action))


def test_l1_infer_rgb(model):
    rgb = np.zeros(
        (model.num_views, model.image_size, model.image_size, 3), dtype=np.uint8
    )
    tokens = make_tokens(model)
    noise = make_noise(model)

    action = model.infer_rgb(rgb, "nhwc", tokens, noise)

    assert action.dtype == np.float32
    assert action.shape == (model.action_horizon, model.action_dim)
    assert np.all(np.isfinite(action))


def test_bad_patch_shape_raises(model):
    bad = np.zeros((3, 4), dtype=np.float32)
    tokens = make_tokens(model)
    noise = make_noise(model)
    with pytest.raises(ValueError, match="patches expected shape"):
        model._infer_patches(bad, tokens, noise)


def test_empty_tokens_raise(model):
    patch_rows = model.num_views * model.patches_per_view
    patch_width = 3 * model.patch_size * model.patch_size
    patches = np.zeros((patch_rows, patch_width), dtype=np.float32)
    noise = make_noise(model)
    with pytest.raises(ValueError, match="token_ids must be non-empty"):
        model._infer_patches(patches, np.zeros(0, dtype=np.uint32), noise)


def test_bad_rgb_bytes_raise(model):
    rgb = np.zeros((1, 8, 8, 3), dtype=np.uint8)
    tokens = make_tokens(model)
    noise = make_noise(model)
    with pytest.raises(ValueError, match="rgb_u8 expected"):
        model.infer_rgb(rgb, "nhwc", tokens, noise)
