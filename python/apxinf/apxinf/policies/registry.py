"""Registry mapping a checkpoint ``model_type`` to its L2 policy class.

Kept dependency-free (imports no policy) so it can be shared by both the policy
modules that *register* and :class:`~apxinf.policies.auto.AutoPolicy` that *reads*
without an import cycle. A policy module registers itself at import time via the
:func:`register_policy` decorator; importing :mod:`apxinf.policies` triggers every
built-in registration.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Type

__all__ = ["register_policy", "get_policy", "available_policies"]

_REGISTRY: Dict[str, type] = {}


def register_policy(model_type: str) -> Callable[[Type], Type]:
    """Class decorator: register a policy class under a ``model_type`` string.

    The key matches the ``type`` field of a checkpoint's ``config.json`` (case-
    insensitive). Re-registering the same class is a no-op; a *different* class
    under a taken key is an error, to catch accidental clashes.
    """
    key = model_type.lower()

    def decorator(cls: Type) -> Type:
        existing = _REGISTRY.get(key)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"policy model_type {key!r} already registered to "
                f"{existing.__name__}, cannot re-register to {cls.__name__}"
            )
        _REGISTRY[key] = cls
        return cls

    return decorator


def get_policy(model_type: str) -> type:
    """Return the policy class registered for ``model_type`` (case-insensitive)."""
    key = model_type.lower()
    try:
        return _REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"no policy registered for model_type {model_type!r}; "
            f"known: {available_policies()}"
        ) from None


def available_policies() -> List[str]:
    """Sorted list of registered ``model_type`` keys."""
    return sorted(_REGISTRY)
