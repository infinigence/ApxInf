"""Golden test: ParseImage + ResizeWithPad vs the OpenPI reference numerics.

The reference functions below are transcribed from the old websocket server
(``_parse_image`` / ``_resize_with_pad``, OpenPI-derived). Reproducing them here
independently anchors the ``apxinf`` steps so they cannot silently drift.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from apxinf.processors import ParseImage, ResizeWithPad

IMAGE_SIZE = 224


def ref_parse_image(value, name="image") -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"{name} must have rank 3, got shape {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    if image.shape[-1] != 3:
        raise ValueError(f"{name} must have three RGB channels, got shape {image.shape}")
    return np.ascontiguousarray(image)


def ref_resize_with_pad(image: np.ndarray) -> np.ndarray:
    if image.shape[:2] == (IMAGE_SIZE, IMAGE_SIZE):
        return image
    current_height, current_width = image.shape[:2]
    ratio = max(current_width / IMAGE_SIZE, current_height / IMAGE_SIZE)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized = Image.fromarray(image).resize(
        (resized_width, resized_height), resample=Image.Resampling.BILINEAR
    )
    canvas = Image.new(resized.mode, (IMAGE_SIZE, IMAGE_SIZE), 0)
    pad_height = max(0, int((IMAGE_SIZE - resized_height) / 2))
    pad_width = max(0, int((IMAGE_SIZE - resized_width) / 2))
    canvas.paste(resized, (pad_width, pad_height))
    return np.asarray(canvas)


@pytest.mark.parametrize("shape", [(224, 224, 3), (256, 320, 3), (480, 640, 3), (100, 50, 3)])
def test_resize_matches_reference(shape):
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=shape, dtype=np.uint8)
    got = ResizeWithPad(IMAGE_SIZE)(ParseImage()(image))
    want = ref_resize_with_pad(ref_parse_image(image))
    assert got.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    np.testing.assert_array_equal(got, want)


def test_parse_float_chw_matches_reference():
    rng = np.random.default_rng(1)
    image = rng.random((3, 128, 96), dtype=np.float32)  # CHW float in [0, 1]
    got = ParseImage()(image)
    want = ref_parse_image(image)
    assert got.dtype == np.uint8 and got.shape == (128, 96, 3)
    np.testing.assert_array_equal(got, want)


def test_resize_passthrough_is_untouched():
    rng = np.random.default_rng(2)
    image = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    np.testing.assert_array_equal(ResizeWithPad(224)(image), image)


def test_parse_rejects_bad_rank():
    with pytest.raises(ValueError):
        ParseImage()(np.zeros((224, 224), dtype=np.uint8))
