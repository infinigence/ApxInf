"""``AutoPolicy``: build the right concrete policy from a checkpoint.

Mirrors the Rust ``AutoModel`` frontend at the Python policy layer. Read the
checkpoint's ``config.json`` model type, look up the registered policy class, and
defer to its ``from_pretrained``:

    policy = AutoPolicy.from_pretrained("pi05_libero_base", precision="bf16")

Use this when you don't want to hard-code which model you're serving (generic
code — the websocket server, batch eval). Use the concrete class (e.g.
``Pi05Policy``) directly when you need model-specific constructor knobs. Both
return an object satisfying the :class:`~apxinf.policies.base.Policy` contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .base import Policy
from .registry import available_policies, get_policy

__all__ = ["AutoPolicy"]

# ``config.json`` discriminator keys, in priority order. pi05's LeRobot-style
# config uses ``type``; the fallbacks cover other layouts.
_MODEL_TYPE_KEYS = ("type", "model_type", "model")


class AutoPolicy:
    """Checkpoint -> concrete policy, dispatched by ``config.json`` model type."""

    def __new__(cls, *args, **kwargs):  # noqa: D401 - guard, not a constructor
        raise TypeError("AutoPolicy is not instantiable; use AutoPolicy.from_pretrained(...)")

    @staticmethod
    def from_pretrained(model_dir, *, model_type: Optional[str] = None, **kwargs) -> Policy:
        """Construct the registered policy for ``model_dir``.

        ``model_type`` overrides the value read from ``config.json`` (use it when
        the config lacks a type field). Extra ``kwargs`` pass through to the
        concrete policy's ``from_pretrained``.

        Built-in policies register themselves when :mod:`apxinf.policies` is
        imported (which always happens before this method is reachable), so the
        registry is already populated here.
        """
        model_dir = Path(model_dir)
        resolved = model_type or _read_model_type(model_dir)
        policy_cls = get_policy(resolved)
        return policy_cls.from_pretrained(model_dir, **kwargs)


def _read_model_type(model_dir: Path) -> str:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"AutoPolicy: no config.json in {model_dir}; pass model_type= explicitly "
            f"(known: {available_policies()})"
        )
    document = json.loads(config_path.read_text())
    for key in _MODEL_TYPE_KEYS:
        value = document.get(key)
        if isinstance(value, str) and value:
            return value
    raise KeyError(
        f"AutoPolicy: config.json in {model_dir} has none of {_MODEL_TYPE_KEYS}; "
        f"pass model_type= explicitly (known: {available_policies()})"
    )
