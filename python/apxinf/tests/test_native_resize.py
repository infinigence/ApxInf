"""Optional CPU primitive: reference pixels, bounds and storage ownership."""
import sys

import numpy as np
import pytest

native = pytest.importorskip("apxinf.processors._resize_native")


def resize(image, height=224, width=224):
    return np.frombuffer(native.bilinear_upsample_rgb(image, height, width), np.uint8).reshape(height, width, 3)


def test_native_resize_matches_numpy_and_torch_reference():
    from apxinf.policies.impls.cosmos_policy import _resize_crop_numpy
    torch = pytest.importorskip("torch")
    functional = pytest.importorskip("torchvision.transforms.functional")
    rng = np.random.default_rng(20261101)
    y, x = np.indices((212, 212))
    cases = [rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)[6:218, 6:218] for _ in range(12)]
    cases += [np.stack([x, y, (x + y) // 2], axis=-1).astype(np.uint8),
              np.stack([x // 4, y // 3, (x * y) % 256], axis=-1).astype(np.uint8)]
    cases += [np.full((212, 212, 3), v, np.uint8) for v in [0, 1, 127, 128, 254, 255]]
    for image in cases:
        before = image.copy()
        got = resize(image)
        reference = functional.resize(torch.from_numpy(image.copy()).permute(2, 0, 1), [224, 224], antialias=True)
        np.testing.assert_array_equal(got, reference.permute(1, 2, 0).numpy())
        np.testing.assert_array_equal(got, _resize_crop_numpy(image))
        np.testing.assert_array_equal(image, before)


def test_identity_single_pixel_and_independent_writable_storage():
    rng = np.random.default_rng(9)
    image = rng.integers(0, 256, (5, 7, 3), dtype=np.uint8)
    got = resize(image, 5, 7)
    np.testing.assert_array_equal(got, image)
    image[:] = 0
    assert got.any()
    got[:] = 17
    assert not image.any()
    pixel = np.array([[[0, 128, 255]]], np.uint8)
    pixel.setflags(write=False)
    np.testing.assert_array_equal(resize(pixel, 4, 6), np.broadcast_to(pixel, (4, 6, 3)))


@pytest.mark.parametrize("image,height,width", [
    (np.zeros((212, 212, 3), np.float32), 224, 224),
    (np.zeros((212, 212), np.uint8), 224, 224),
    (np.zeros((212, 212, 4), np.uint8), 224, 224),
    (np.zeros((0, 212, 3), np.uint8), 224, 224),
    (np.zeros((212, 212, 3), np.uint8)[::-1], 224, 224),
    (np.zeros((212, 212, 3), np.uint8)[:, ::2], 224, 224),
    (np.zeros((212, 212, 3), np.uint8), 211, 224),
    (np.zeros((212, 212, 3), np.uint8), 224, -1),
])
def test_rejects_invalid_input(image, height, width):
    with pytest.raises(ValueError):
        native.bilinear_upsample_rgb(image, height, width)


def test_rejects_overflow_before_allocation():
    with pytest.raises(OverflowError):
        native.bilinear_upsample_rgb(np.zeros((1, 1, 3), np.uint8), sys.maxsize, sys.maxsize)


def test_policy_numpy_fallback_matches_native(monkeypatch):
    import apxinf.policies.impls.cosmos_policy as implementation
    image = np.random.default_rng(1).integers(0, 256, (212, 212, 3), dtype=np.uint8)
    expected = resize(image)
    monkeypatch.setattr(implementation, "_native_resize", None)
    np.testing.assert_array_equal(implementation._resize_crop(image), expected)
