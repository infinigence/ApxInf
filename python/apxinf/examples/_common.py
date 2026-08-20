"""Shared helpers for the ``apxinf`` examples.

Keeps each example short: a ``sys.path`` shim so the examples run straight from a
source checkout without installing ``apxinf``, plus a synthetic observation
builder so no dataset / simulator is needed to see the API work.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any, Dict, Sequence

import numpy as np

# Run from a source checkout without ``pip install``: put python/apxinf on the
# path. (When ``apxinf`` is installed this is a harmless no-op.) The ``apxinf_py``
# CUDA binding must still be installed separately — ``maturin develop`` of
# crates/apxinf-py — for the real-model examples.
_APXINF_PKG = pathlib.Path(__file__).resolve().parents[1]
if _APXINF_PKG.is_dir() and str(_APXINF_PKG) not in sys.path:
    sys.path.insert(0, str(_APXINF_PKG))


def synthetic_observation(
    *,
    image_keys: Sequence[str] = ("observation/image", "observation/wrist_image"),
    height: int = 256,
    width: int = 256,
    state_dim: int = 8,
    prompt: str = "pick up the block",
    seed: int = 0,
) -> Dict[str, Any]:
    """Build one raw observation dict shaped like the policy expects.

    Random ``uint8`` camera frames (raw ``HWC``; the policy's own resize step
    handles them) plus a float32 state vector and a text prompt. This is only to
    exercise the interface end-to-end — the actions it yields are meaningless.
    """
    rng = np.random.default_rng(seed)
    observation: Dict[str, Any] = {
        key: rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        for key in image_keys
    }
    observation["observation/state"] = rng.standard_normal(state_dim).astype(np.float32)
    observation["prompt"] = prompt
    return observation
