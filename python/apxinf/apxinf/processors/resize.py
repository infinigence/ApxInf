"""Image pre-processing steps: :class:`ParseImage` and :class:`ResizeWithPad`.

These reproduce, byte for byte, the OpenPI-derived reference used by the old
websocket server (``_parse_image`` / ``_resize_with_pad``): float images are
scaled by 255 and cast to ``uint8``, CHW is transposed to HWC, and the image is
letterbox-resized (BILINEAR, aspect-preserving) onto a zero-padded square
canvas. Splitting parse and resize into two steps lets a caller swap either one
independently while chaining them in a :class:`~apxinf.processors.Pipeline`.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .base import ProcessorStep

__all__ = ["ParseImage", "ResizeWithPad"]


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
