"""Build the registered concrete policy identified by a checkpoint layout.

The model type comes from ``config.json`` or supported layout metadata:

    policy = AutoPolicy.from_pretrained("pi05_libero_base", compute_variant="bf16")

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
from ..checkpoints import OPENPI_PYTORCH, CheckpointError, detect_checkpoint

__all__ = ["AutoPolicy"]

# ``config.json`` discriminator keys, in priority order. pi05's LeRobot-style
# config uses ``type``; the fallbacks cover other layouts.
_MODEL_TYPE_KEYS = ("type", "model_type", "model")


class AutoPolicy:
    """Checkpoint-to-policy dispatch through the registered model type."""

    def __new__(cls, *args, **kwargs):  # noqa: D401 - guard, not a constructor
        raise TypeError("AutoPolicy is not instantiable; use AutoPolicy.from_pretrained(...)")

    @staticmethod
    def from_pretrained(model_dir, *, model_type: Optional[str] = None, **kwargs) -> Policy:
        """Construct the registered policy for ``model_dir``.

        ``model_type`` overrides the value detected from the checkpoint. Extra
        ``kwargs`` pass through to the concrete policy's ``from_pretrained``.

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
        # OpenPI PyTorch exports declare their model type in metadata.pt.
        try:
            layout = detect_checkpoint(model_dir)
        except CheckpointError as exc:
            raise FileNotFoundError(
                f"AutoPolicy: no config.json in {model_dir} and its layout could not "
                f"be identified ({exc}); pass model_type= explicitly "
                f"(known: {available_policies()})"
            ) from exc
        if layout.format == OPENPI_PYTORCH:
            # train_config_facts already refused anything that is not pi05
            # (pi05=False, or a backbone this runtime is not built for).
            return "pi05"
        raise FileNotFoundError(
            f"AutoPolicy: no config.json in {model_dir}; pass model_type= explicitly "
            f"(known: {available_policies()})"
        )
    document = json.loads(config_path.read_text())
    # The published WallOSS checkpoint retains Qwen2.5-VL's generic
    # ``model_type``. Its two execution experts plus action head are the stable
    # architecture discriminator; do not register every qwen2_5_vl as WallOSS.
    if (
        document.get("model_type") == "qwen2_5_vl"
        and isinstance(document.get("experts"), list)
        and len(document["experts"]) == 2
        and "action_hidden_size" in document
        and "noise_scheduler" in document
    ):
        return "walloss"
    for key in _MODEL_TYPE_KEYS:
        value = document.get(key)
        if isinstance(value, str) and value:
            return value
    raise KeyError(
        f"AutoPolicy: config.json in {model_dir} has none of {_MODEL_TYPE_KEYS}; "
        f"pass model_type= explicitly (known: {available_policies()})"
    )
