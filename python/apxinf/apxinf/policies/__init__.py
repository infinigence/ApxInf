"""The L2 policy layer: contracts, registry, dispatch, and model policies.

This package owns everything policy-related, mirroring how :mod:`apxinf.processors`
owns its own steps and :class:`~apxinf.processors.base.ProcessorStep`. It is split
into a **stable** outer layer and a **volatile** inner one:

* :mod:`~apxinf.policies.base` — the :class:`Policy` / :class:`BareModel` contracts.
* :mod:`~apxinf.policies.registry` — the ``model_type -> policy class`` registry.
* :mod:`~apxinf.policies.auto` — :class:`AutoPolicy`, dispatch by ``config.json`` type.
* :mod:`~apxinf.policies.impls` — the concrete per-model policies (``pi05``, ...),
  the only part that grows as models are added.

Importing this package imports :mod:`~apxinf.policies.impls`, whose modules
register themselves under a ``model_type`` via :func:`register_policy` so
:class:`AutoPolicy` can dispatch. **To add a new model:** create
``apxinf/policies/impls/<name>.py`` following ``pi05.py`` (decorate the class with
``@register_policy("<name>")``), then re-export it from
:mod:`apxinf.policies.impls`.

None of this imports ``apxinf_py`` — policy classes load the CUDA binding lazily,
inside ``from_pretrained`` — so importing the package stays offline-friendly.
"""

from __future__ import annotations

from .auto import AutoPolicy
from .base import BareModel, Policy
from .registry import available_policies, get_policy, register_policy

# Concrete model policies (importing registers them under their model_type).
from .impls import GrootN17Policy, Pi05Policy

__all__ = [
    "Policy",
    "BareModel",
    "AutoPolicy",
    "register_policy",
    "get_policy",
    "available_policies",
    "Pi05Policy",
    "GrootN17Policy",
]
