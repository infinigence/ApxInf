"""Shared helpers for the ``apxinf`` examples.

Keeps each example short: a ``sys.path`` shim so the examples run straight from a
source checkout without installing ``apxinf``, plus a synthetic observation
builder so no dataset / simulator is needed to see the API work.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

# Run from a source checkout without ``pip install``: put python/apxinf on the
# path. (When ``apxinf`` is installed this is a harmless no-op.) The ``apxinf_py``
# CUDA binding must still be installed separately — ``maturin develop`` of
# crates/apxinf-py — for the real-model examples.
_APXINF_PKG = pathlib.Path(__file__).resolve().parents[1]
if _APXINF_PKG.is_dir() and str(_APXINF_PKG) not in sys.path:
    sys.path.insert(0, str(_APXINF_PKG))


def json_object(value: str) -> Dict[str, Any]:
    """Parse a JSON object for model-specific ``AutoPolicy`` options."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"invalid JSON: {error.msg}") from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return parsed


def policy_kwargs(
    options: Mapping[str, Any],
    *,
    device: str,
    precision: Optional[str] = None,
    compute_variant: Optional[str] = None,
    action_dim: int = 0,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge generic CLI flags with concrete-policy options.

    Dedicated flags win over duplicate JSON keys. Server-owned metadata wins
    over caller metadata while preserving unrelated caller fields.
    """
    kwargs = dict(options)
    if metadata is not None:
        caller_metadata = kwargs.pop("metadata", {})
        if not isinstance(caller_metadata, dict):
            raise ValueError("policy-options metadata must be a JSON object")
        kwargs["metadata"] = {**caller_metadata, **metadata}
    if precision is not None and compute_variant is not None:
        raise ValueError("choose compute_variant for PI0.5 or precision for other models")
    kwargs["device"] = device
    if compute_variant is not None:
        kwargs.pop("precision", None)
        kwargs["compute_variant"] = compute_variant
    elif precision is not None:
        kwargs.pop("compute_variant", None)
        kwargs["precision"] = precision
    if action_dim:
        kwargs["action_dim"] = action_dim
    return kwargs


def synthetic_observation(
    *,
    image_keys: Sequence[str],
    state_key: Optional[str] = None,
    prompt_key: str = "prompt",
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

    ``image_keys`` has no default on purpose: camera wire keys are a dataset's
    convention, and every caller here can read the real ones off the policy
    (``policy.image_keys`` / ``policy.state_key``). A helper that guessed them
    would drift from whatever the policy is actually serving.

    ``state_key`` is ``None`` by default for the same reason, and ``None`` is
    also what a policy that *drops* state publishes — so no state is put in the
    dict, which is exactly what such a policy reads.
    """
    rng = np.random.default_rng(seed)
    observation: Dict[str, Any] = {
        key: rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        for key in image_keys
    }
    if state_key is not None:
        observation[state_key] = rng.standard_normal(state_dim).astype(np.float32)
    observation[prompt_key] = prompt
    return observation
