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

Widths follow OpenPI's asymmetry: :class:`Unnormalizer` accepts an array *wider*
than its stats and passes the extra tail through unchanged (a checkpoint's stats
are the robot's width, the model emits its padded width), while
:class:`Normalizer` requires an exact match.
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


def load_norm_stats(model_dir=None, key: str = "actions", *, path=None) -> dict:
    """Return the raw stats dict for ``key`` (e.g. ``"actions"``/``"state"``).

    Handles both the nested ``{"norm_stats": {...}}`` and flat top-level layouts.

    ``path`` names the file directly and is what callers should pass: an openpi
    PyTorch export keeps its statistics at ``assets/<asset_id>/norm_stats.json``,
    not in the checkpoint root, so :func:`apxinf.checkpoints.detect_checkpoint`
    resolves the path and hands it over here. ``model_dir`` remains for the flat
    layout — it just means ``<model_dir>/norm_stats.json``.
    """
    if path is None:
        if model_dir is None:
            raise TypeError("load_norm_stats needs either model_dir or path")
        path = Path(model_dir) / "norm_stats.json"
    path = Path(path)
    document = json.loads(path.read_text())
    stats = document.get("norm_stats", document)
    if key not in stats:
        raise KeyError(f"{path} has no entry {key!r}; keys: {sorted(stats)}")
    return stats[key]


def _as_vector(values: Sequence[float], name: str, dims: Optional[int]) -> np.ndarray:
    # Kept at float64 (the dtype norm_stats.json parses to). The compute dtype is
    # decided per call in :meth:`_AffineStats._check`, which downcasts these to
    # float32 for float32 inputs -- bit-identical to storing them as float32.
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be rank 1, got shape {vector.shape}")
    if dims is not None:
        if vector.size < dims:
            raise ValueError(f"{name} has {vector.size} entries, need at least {dims}")
        vector = vector[:dims]
    return np.ascontiguousarray(vector)


class _AffineStats:
    """Shared statistics + the two-mode affine math for (un)normalization."""

    def __init__(self, *, q01=None, q99=None, mean=None, std=None, mode=_QUANTILE, dims=None, eps=1e-6, dtype=None):
        if mode not in (_QUANTILE, _MEAN_STD):
            raise ValueError(f"mode must be {_QUANTILE!r} or {_MEAN_STD!r}, got {mode!r}")
        self.mode = mode
        self.dims = None if dims is None else int(dims)
        self.eps = float(eps)
        self.dtype = None if dtype is None else np.dtype(dtype)
        # dtype -> (q01, q99, mean, std) downcast to that dtype; see _stats_as.
        self._cast_cache: dict = {}
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

    def _check(self, array: np.ndarray, who: str, *, allow_wider: bool = False) -> tuple:
        """Coerce ``array`` and the stats to a common compute dtype, checking width.

        The default dtype is ``result_type(array, float32)``: float32 in stays
        float32 (bit-identical to the previous float32-only implementation,
        because the stats are downcast *before* the arithmetic), while float64 in
        stays float64. That matters for the pi05 prompt path -- openpi's numpy
        input chain runs in float64, so a float32 state would discretize to a
        different bin whenever a normalized value lands within ~1e-7 of a bin
        edge. An explicit ``dtype=`` pins the compute dtype regardless of the
        input, which is how a caller reproduces openpi's *output* chain: there
        the stats stay float64 (they are parsed from JSON and never downcast), so
        a float32 action array is promoted rather than the stats demoted.

        ``allow_wider`` mirrors openpi's asymmetry: its ``Unnormalize`` widens
        narrow stats by passing the tail through, its ``Normalize`` does not (a
        wider array there is a broadcast error). See :meth:`unnormalize`.
        """
        array = np.asarray(array)
        dtype = self.dtype if self.dtype is not None else np.result_type(array.dtype, np.float32)
        array = array.astype(dtype, copy=False)
        got = array.shape[-1]
        if got < self.width or (got != self.width and not allow_wider):
            raise ValueError(
                f"{who}: last dim must be {self.width}, got array shape {array.shape}"
            )
        return array, dtype

    def _stats_as(self, dtype):
        cached = self._cast_cache.get(dtype)
        if cached is None:
            cached = tuple(
                None if v is None else v.astype(dtype, copy=False)
                for v in (self.q01, self.q99, self.mean, self.std)
            )
            self._cast_cache[dtype] = cached
        return cached

    def unnormalize(self, array: np.ndarray) -> np.ndarray:
        """Map back to physical units, **passing a wider array's tail through**.

        A checkpoint's ``norm_stats`` is computed from the dataset, so it is the
        *robot's* width (16 for a Unitree G1) while the model emits its padded
        width (32). openpi's ``Unnormalize._unnormalize_quantile`` handles that by
        unnormalizing the head and concatenating ``x[..., dim:]`` verbatim; the
        padded tail is unused downstream, so the passthrough exists to keep the
        chain running rather than to produce meaningful numbers. Without it the
        G1 path cannot serve a real 16-wide ``norm_stats.json`` at all.
        """
        array, dtype = self._check(array, "Unnormalizer", allow_wider=True)
        head, tail = array[..., : self.width], array[..., self.width :]
        q01, q99, mean, std = self._stats_as(dtype)
        scalar = dtype.type
        if self.mode == _QUANTILE:
            span = (q99 - q01 + scalar(self.eps)) / scalar(2.0)
            out = (head + scalar(1.0)) * span + q01
        else:
            out = head * std + mean
        if tail.shape[-1]:
            out = np.concatenate([out, tail], axis=-1)
        return _finite(out.astype(dtype, copy=False), "Unnormalizer")

    def normalize(self, array: np.ndarray) -> np.ndarray:
        """Map to the normalized domain. Width must match exactly.

        No tail passthrough here, matching openpi: its ``Normalize`` slices the
        stats to the array width, which broadcast-fails on an array *wider* than
        the stats. The input chain never hits that case anyway --
        ``PadStatesAndActions`` runs after ``Normalize``, so state is still the
        robot's width when it is normalized.
        """
        array, dtype = self._check(array, "Normalizer")
        q01, q99, mean, std = self._stats_as(dtype)
        scalar = dtype.type
        if self.mode == _QUANTILE:
            span = q99 - q01 + scalar(self.eps)
            out = scalar(2.0) * (array - q01) / span - scalar(1.0)
        else:
            out = (array - mean) / std
        return _finite(out.astype(dtype, copy=False), "Normalizer")


def _finite(array: np.ndarray, who: str) -> np.ndarray:
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{who} produced non-finite values")
    return array


def _from_norm_stats(cls, model_dir, key, mode, dims, eps, dtype, path=None):
    stats = load_norm_stats(model_dir, key, path=path)
    if mode == _QUANTILE:
        return cls(q01=stats["q01"], q99=stats["q99"], mode=mode, dims=dims, eps=eps, dtype=dtype)
    return cls(mean=stats["mean"], std=stats["std"], mode=mode, dims=dims, eps=eps, dtype=dtype)


class Unnormalizer(ProcessorStep):
    """Map a normalized-domain array back to physical units (last axis).

    An array wider than :attr:`width` keeps its tail unchanged, mirroring
    OpenPI — see :meth:`_AffineStats.unnormalize`. A narrower one is an error.

    ``dtype`` pins the compute dtype; the default follows the input (see
    :meth:`_AffineStats._check`).
    """

    PARAMS = ("eps", "dtype")

    def __init__(self, *, q01=None, q99=None, mean=None, std=None, mode=_QUANTILE, dims=None, eps=1e-6, dtype=None):
        self._stats = _AffineStats(
            q01=q01, q99=q99, mean=mean, std=std, mode=mode, dims=dims, eps=eps, dtype=dtype
        )
        self.eps = self._stats.eps
        self.dtype = self._stats.dtype

    @classmethod
    def from_norm_stats(cls, model_dir=None, key: str = "actions", mode: str = _QUANTILE, dims=None, eps=1e-6, dtype=None, *, path=None):
        """Build from a ``norm_stats.json``; ``path`` names the file directly."""
        return _from_norm_stats(cls, model_dir, key, mode, dims, eps, dtype, path)

    @property
    def width(self) -> int:
        return self._stats.width

    def __call__(self, array: np.ndarray) -> np.ndarray:
        return self._stats.unnormalize(array)

    def _apply_overrides(self, overrides: dict) -> None:
        super()._apply_overrides(overrides)
        # ``with_overrides`` shallow-copied us, so ``_stats`` is still shared with
        # the original; copy it before tweaking the knobs to avoid mutating the
        # source. ``_cast_cache`` is keyed by dtype, so sharing it stays correct.
        self._stats = copy.copy(self._stats)
        self._stats.eps = self.eps
        self.dtype = None if self.dtype is None else np.dtype(self.dtype)
        self._stats.dtype = self.dtype


class Normalizer(ProcessorStep):
    """Map a physical-units array to the normalized domain (last axis).

    ``dtype`` pins the compute dtype; the default follows the input (see
    :meth:`_AffineStats._check`).
    """

    PARAMS = ("eps", "dtype")

    def __init__(self, *, q01=None, q99=None, mean=None, std=None, mode=_QUANTILE, dims=None, eps=1e-6, dtype=None):
        self._stats = _AffineStats(
            q01=q01, q99=q99, mean=mean, std=std, mode=mode, dims=dims, eps=eps, dtype=dtype
        )
        self.eps = self._stats.eps
        self.dtype = self._stats.dtype

    @classmethod
    def from_norm_stats(cls, model_dir=None, key: str = "actions", mode: str = _QUANTILE, dims=None, eps=1e-6, dtype=None, *, path=None):
        """Build from a ``norm_stats.json``; ``path`` names the file directly."""
        return _from_norm_stats(cls, model_dir, key, mode, dims, eps, dtype, path)

    @property
    def width(self) -> int:
        return self._stats.width

    def __call__(self, array: np.ndarray) -> np.ndarray:
        return self._stats.normalize(array)

    def _apply_overrides(self, overrides: dict) -> None:
        super()._apply_overrides(overrides)
        self._stats = copy.copy(self._stats)
        self._stats.eps = self.eps
        self.dtype = None if self.dtype is None else np.dtype(self.dtype)
        self._stats.dtype = self.dtype
