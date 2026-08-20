"""Noise sampling step for the flow-matching prior.

Draws ``standard_normal`` noise of shape ``[action_horizon, action_dim]`` from a
seeded generator, so a fixed seed gives a reproducible action. ``dtype`` selects
the output precision: ``float32`` (default, what the L0/L1 bindings accept) or
``float16`` to reproduce the old websocket path, which cast noise to fp16 before
handing it to the engine (kept for Phase 3 parity).
"""

from __future__ import annotations

import numpy as np

from .base import ProcessorStep

__all__ = ["GaussianNoise"]


class GaussianNoise(ProcessorStep):
    """Seeded ``standard_normal`` sampler producing a ``[horizon, dim]`` array."""

    PARAMS = ("seed", "dtype")

    def __init__(self, action_horizon: int, action_dim: int, seed: int = 0, dtype=np.float32):
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.seed = int(seed)
        self.dtype = np.dtype(dtype)
        self._rng = np.random.default_rng(self.seed)

    def __call__(self) -> np.ndarray:
        sample = self._rng.standard_normal((self.action_horizon, self.action_dim), dtype=np.float32)
        if self.dtype != np.float32:
            sample = sample.astype(self.dtype)
        return sample

    def reset(self, seed: int | None = None) -> None:
        """Re-seed the generator (defaults to the configured seed)."""
        if seed is not None:
            self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

    def _apply_overrides(self, overrides: dict) -> None:
        super()._apply_overrides(overrides)
        self.dtype = np.dtype(self.dtype)
        # A fresh, independently-seeded RNG so the override does not alias the
        # original step's stream.
        self._rng = np.random.default_rng(self.seed)
