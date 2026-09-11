"""Image pre-processing steps: :class:`ParseImage` and the two letterbox resizes.

These reproduce, byte for byte, the OpenPI-derived reference used by the old
websocket server (``_parse_image`` / ``_resize_with_pad``): float images are
scaled by 255 and cast to ``uint8``, CHW is transposed to HWC, and the image is
letterbox-resized (BILINEAR, aspect-preserving) onto a zero-padded square
canvas. Splitting parse and resize into two steps lets a caller swap either one
independently while chaining them in a :class:`~apxinf.processors.Pipeline`.

There are two letterbox resizes because the reference model families do not
share one. :class:`ResizeWithPad` is the PIL/BILINEAR one above, which is what
PI0.5's OpenPI-derived pipeline uses. :class:`TorchResizeWithPad` reproduces
``F.interpolate(mode="bilinear", align_corners=False)`` — the numerics of
LeRobot's ``resize_with_pad_torch``, which is what π0-FAST's checkpoint was
trained and evaluated with. They are *not* interchangeable: the two
interpolators disagree by up to ~21/255 on a 256→224 downscale, and that is
enough to change which FAST action tokens the model emits.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .base import ProcessorStep

__all__ = ["ParseImage", "ResizeWithPad", "TorchResizeWithPad"]


class ParseImage(ProcessorStep):
    """Coerce an arbitrary image-like value into a contiguous ``uint8`` HWC RGB array.

    Accepts rank-3 arrays in HWC or CHW order. Floating-point images are assumed
    to be in ``[0, 1]`` and scaled by 255. Raises on anything that is not a
    3-channel RGB image, with a message naming the offending value.
    """

    def __init__(self, name: str = "image"):
        self.name = name

    def __call__(self, value) -> np.ndarray:
        image = np.asarray(value)
        if image.ndim != 3:
            raise ValueError(f"{self.name} must have rank 3, got shape {image.shape}")
        if np.issubdtype(image.dtype, np.floating):
            image = (255 * image).astype(np.uint8)
        if image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        if image.shape[-1] != 3:
            raise ValueError(f"{self.name} must have three RGB channels, got shape {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"{self.name} must be uint8 or floating point, got {image.dtype}")
        return np.ascontiguousarray(image)


class ResizeWithPad(ProcessorStep):
    """Letterbox-resize a ``uint8`` HWC image onto a ``size``x``size`` padded canvas.

    Aspect ratio is preserved: the image is scaled so its longer side fits
    ``size``, then centered on a zero (black) canvas. An already-``size``-square
    image is returned untouched, matching the reference fast path.
    """

    PARAMS = ("size",)

    def __init__(self, size: int = 224):
        self.size = int(size)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        size = self.size
        if image.shape[:2] == (size, size):
            return image
        current_height, current_width = image.shape[:2]
        ratio = max(current_width / size, current_height / size)
        resized_height = int(current_height / ratio)
        resized_width = int(current_width / ratio)
        resized = Image.fromarray(image).resize(
            (resized_width, resized_height), resample=Image.Resampling.BILINEAR
        )
        canvas = Image.new(resized.mode, (size, size), 0)
        pad_height = max(0, int((size - resized_height) / 2))
        pad_width = max(0, int((size - resized_width) / 2))
        canvas.paste(resized, (pad_width, pad_height))
        return np.asarray(canvas)


def _torch_interpolate_taps(in_length: int, out_length: int):
    """Source taps and weights for ``F.interpolate(..., align_corners=False)``.

    PyTorch maps output index ``dst`` to ``scale*(dst + 0.5) - 0.5`` with
    ``scale = in/out``, clamps negatives to zero, and linearly blends the two
    neighbouring source samples. This transcribes that mapping so the resize
    below can be reproduced without a torch dependency.
    """
    scale = np.float32(in_length / out_length)
    destination = np.arange(out_length, dtype=np.float32)
    source = np.maximum(
        scale * (destination + np.float32(0.5)) - np.float32(0.5), np.float32(0.0)
    )
    low = np.minimum(np.floor(source).astype(np.int64), in_length - 1)
    high = np.minimum(low + 1, in_length - 1)
    return low, high, (source - low.astype(np.float32)).astype(np.float32)


def _torch_bilinear_resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bilinear resize of a ``float32`` HWC image, matching ``F.interpolate``.

    The four-corner blend is written the way PyTorch accumulates it
    (`h0 * (w0*v00 + w1*v01) + h1 * (w0*v10 + w1*v11)`) rather than as a
    separable two-pass filter, so the float rounding matches too.
    """
    image = np.asarray(image, dtype=np.float32)
    rows_low, rows_high, rows_frac = _torch_interpolate_taps(image.shape[0], height)
    cols_low, cols_high, cols_frac = _torch_interpolate_taps(image.shape[1], width)
    horizontal = cols_frac[None, :, None]
    vertical = rows_frac[:, None, None]
    top = (
        image[np.ix_(rows_low, cols_low)] * (np.float32(1.0) - horizontal)
        + image[np.ix_(rows_low, cols_high)] * horizontal
    )
    bottom = (
        image[np.ix_(rows_high, cols_low)] * (np.float32(1.0) - horizontal)
        + image[np.ix_(rows_high, cols_high)] * horizontal
    )
    return top * (np.float32(1.0) - vertical) + bottom * vertical


class TorchResizeWithPad(ProcessorStep):
    """Letterbox-resize a ``uint8`` HWC image the way LeRobot does.

    Same geometry as :class:`ResizeWithPad` — scale the longer side to ``size``
    and center the result on a zero canvas — but the interpolation is
    ``F.interpolate(mode="bilinear", align_corners=False)``, which is what
    ``lerobot.policies.pi0_fast.modeling_pi0_fast.resize_with_pad_torch`` calls
    (and what the π0-FAST checkpoint saw during training). The result is rounded
    back to ``uint8`` because the runtime's image entry point takes ``uint8``;
    that round-trip reproduces the reference's tokens exactly, while PIL's
    BILINEAR does not.
    """

    PARAMS = ("size",)

    def __init__(self, size: int = 224):
        self.size = int(size)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        size = self.size
        if image.shape[:2] == (size, size):
            return image
        current_height, current_width = image.shape[:2]
        ratio = max(current_width / size, current_height / size)
        resized_height = int(current_height / ratio)
        resized_width = int(current_width / ratio)
        resized = _torch_bilinear_resize(
            image.astype(np.float32) / np.float32(255.0), resized_height, resized_width
        )
        resized = np.clip(np.round(resized * np.float32(255.0)), 0.0, 255.0).astype(np.uint8)
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        pad_height, _ = divmod(size - resized_height, 2)
        pad_width, _ = divmod(size - resized_width, 2)
        canvas[pad_height : pad_height + resized_height, pad_width : pad_width + resized_width] = resized
        return canvas
