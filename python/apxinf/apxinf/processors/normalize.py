"""Normalize / unnormalize steps for state and action vectors.

Two conventions are supported, mirroring OpenPI ``norm_stats.json`` and
lerobot's ``Normalizer``:

* ``quantile`` — map ``[q01, q99]`` to ``[-1, 1]``. Unnormalize is
  ``(x + 1) * (q99 - q01 + eps) / 2 + q01`` (the exact reference the old
  websocket server used); normalize is its inverse.
* ``mean_std`` — standardize with ``(x - mean) / std``; unnormalize is
  ``x * std + mean``.

Both steps broadcast over the trailing axis, so a ``[horizon, dim]`` action
chunk unnormalizes in one call. :meth:`from_norm_stats` reads the quantiles /
moments straight out of a checkpoint's ``norm_stats.json``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .base import ProcessorStep

__all__ = ["Normalizer", "Unnormalizer", "load_norm_stats"]

_QUANTILE = "quantile"
_MEAN_STD = "mean_std"


def load_norm_stats(model_dir, key: str = "actions") -> dict:
    """Return the raw stats dict for ``key`` (e.g. ``"actions"``/``"state"``).

    Handles both the nested ``{"norm_stats": {...}}`` and flat top-level layouts.
    """
    document = json.loads((Path(model_dir) / "norm_stats.json").read_text())
    stats = document.get("norm_stats", document)
    if key not in stats:
        raise KeyError(f"norm_stats.json has no entry {key!r}; keys: {sorted(stats)}")
    return stats[key]


def _as_vector(values: Sequence[float], name: str, dims: Optional[int]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be rank 1, got shape {vector.shape}")
    if dims is not None:
        if vector.size < dims:
            raise ValueError(f"{name} has {vector.size} entries, need at least {dims}")
        vector = vector[:dims]
    return np.ascontiguousarray(vector)


class _AffineStats:
    """Shared statistics + the two-mode affine math for (un)normalization."""

    def __init__(self, *, q01=None, q99=None, mean=None, std=None, mode=_QUANTILE, dims=None, eps=1e-6):
        if mode not in (_QUANTILE, _MEAN_STD):
            raise ValueError(f"mode must be {_QUANTILE!r} or {_MEAN_STD!r}, got {mode!r}")
        self.mode = mode
        self.dims = None if dims is None else int(dims)
        self.eps = float(eps)
        if mode == _QUANTILE:
            if q01 is None or q99 is None:
                raise ValueError("quantile mode requires q01 and q99")
            self.q01 = _as_vector(q01, "q01", self.dims)
            self.q99 = _as_vector(q99, "q99", self.dims)
            if self.q01.shape != self.q99.shape:
                raise ValueError(f"q01 {self.q01.shape} and q99 {self.q99.shape} must match")
            self.mean = None
            self.std = None
        else:
            if mean is None or std is None:
                raise ValueError("mean_std mode requires mean and std")
            self.mean = _as_vector(mean, "mean", self.dims)
            self.std = _as_vector(std, "std", self.dims)
            if self.mean.shape != self.std.shape:
                raise ValueError(f"mean {self.mean.shape} and std {self.std.shape} must match")
            self.q01 = None
            self.q99 = None

    @property
    def width(self) -> int:
        return (self.q01 if self.mode == _QUANTILE else self.mean).size

    def _check(self, array: np.ndarray, who: str) -> np.ndarray:
        array = np.asarray(array, dtype=np.float32)
        if array.shape[-1] != self.width:
            raise ValueError(
                f"{who}: last dim must be {self.width}, got array shape {array.shape}"
            )
        return array

    def unnormalize(self, array: np.ndarray) -> np.ndarray:
        array = self._check(array, "Unnormalizer")
        if self.mode == _QUANTILE:
            span = (self.q99 - self.q01 + np.float32(self.eps)) / np.float32(2.0)
            out = (array + np.float32(1.0)) * span + self.q01
        else:
            out = array * self.std + self.mean
        return _finite(out.astype(np.float32, copy=False), "Unnormalizer")

    def normalize(self, array: np.ndarray) -> np.ndarray:
        array = self._check(array, "Normalizer")
        if self.mode == _QUANTILE:
            span = self.q99 - self.q01 + np.float32(self.eps)
            out = np.float32(2.0) * (array - self.q01) / span - np.float32(1.0)
        else:
            out = (array - self.mean) / self.std
        return _finite(out.astype(np.float32, copy=False), "Normalizer")


def _finite(array: np.ndarray, who: str) -> np.ndarray:
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{who} produced non-finite values")
    return array


def _from_norm_stats(cls, model_dir, key, mode, dims, eps):
    stats = load_norm_stats(model_dir, key)
    if mode == _QUANTILE:
        return cls(q01=stats["q01"], q99=stats["q99"], mode=mode, dims=dims, eps=eps)
    return cls(mean=stats["mean"], std=stats["std"], mode=mode, dims=dims, eps=eps)


class Unnormalizer(ProcessorStep):
    """Map a normalized-domain array back to physical units (last axis)."""

    PARAMS = ("eps",)

    def __init__(self, *, q01=None, q99=None, mean=None, std=None, mode=_QUANTILE, dims=None, eps=1e-6):
        self._stats = _AffineStats(q01=q01, q99=q99, mean=mean, std=std, mode=mode, dims=dims, eps=eps)
        self.eps = self._stats.eps

    @classmethod
    def from_norm_stats(cls, model_dir, key: str = "actions", mode: str = _QUANTILE, dims=None, eps=1e-6):
        return _from_norm_stats(cls, model_dir, key, mode, dims, eps)

    @property
    def width(self) -> int:
        return self._stats.width

    def __call__(self, array: np.ndarray) -> np.ndarray:
        return self._stats.unnormalize(array)

    def _apply_overrides(self, overrides: dict) -> None:
        super()._apply_overrides(overrides)
        # ``with_overrides`` shallow-copied us, so ``_stats`` is still shared with
        # the original; copy it before tweaking eps to avoid mutating the source.
        self._stats = copy.copy(self._stats)
        self._stats.eps = self.eps


class Normalizer(ProcessorStep):
    """Map a physical-units array to the normalized domain (last axis)."""

    PARAMS = ("eps",)

    def __init__(self, *, q01=None, q99=None, mean=None, std=None, mode=_QUANTILE, dims=None, eps=1e-6):
        self._stats = _AffineStats(q01=q01, q99=q99, mean=mean, std=std, mode=mode, dims=dims, eps=eps)
        self.eps = self._stats.eps

    @classmethod
    def from_norm_stats(cls, model_dir, key: str = "actions", mode: str = _QUANTILE, dims=None, eps=1e-6):
        return _from_norm_stats(cls, model_dir, key, mode, dims, eps)

    @property
    def width(self) -> int:
        return self._stats.width

    def __call__(self, array: np.ndarray) -> np.ndarray:
        return self._stats.normalize(array)

    def _apply_overrides(self, overrides: dict) -> None:
        super()._apply_overrides(overrides)
        self._stats = copy.copy(self._stats)
        self._stats.eps = self.eps
