"""Concrete per-model L2 policies — the volatile part of the policy layer.

One module per model family (:mod:`~apxinf.policies.impls.pi05`, and future
``groot``, ...). The stable machinery — contracts, registry, dispatch — lives one
level up in :mod:`apxinf.policies`; only this package grows as models are added.

Importing this package imports every model module for its side effect: each
registers its class under a ``model_type`` via
:func:`~apxinf.policies.registry.register_policy`, so
:class:`~apxinf.policies.auto.AutoPolicy` can dispatch. **To add a new model:**
create ``apxinf/policies/impls/<name>.py`` following ``pi05.py`` (decorate the
class with ``@register_policy("<name>")``), then add its import + re-export here.

Nothing here imports ``apxinf_py``; policy classes load the CUDA binding lazily
inside ``from_pretrained``, so importing the package stays offline-friendly.
"""

from __future__ import annotations

from .pi05 import Pi05Policy
from .groot_n17 import GrootN17Policy

__all__ = ["Pi05Policy", "GrootN17Policy"]
