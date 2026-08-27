"""The contracts shared across the frontend: :class:`Policy` and :class:`BareModel`.

Both are structural :class:`typing.Protocol` types — nothing has to inherit
them; any object of the right shape satisfies them. They are the stable anchor
points other layers code against:

* :class:`BareModel` — the model-side contract (L1 bare inference) a policy
  consumes, i.e. the subset of a ``apxinf_py.Model`` handle it relies on.
* :class:`Policy` — the L2 contract (``obs dict -> result dict``) every
  model-specific policy satisfies (``Pi05Policy`` today; e.g. a future
  ``GrootPolicy``). Downstream consumers (the websocket server, the
  :class:`~apxinf.policies.auto.AutoPolicy` registry, a lerobot adaptor) code
  against this, never a concrete class.

It exists to pin these contracts *now*, before a second model lands, so the
layout is set for whoever adds one. They intentionally stay tiny: extract richer
shared structure when a second model actually teaches us what is common, not by
guessing from one example.

The ``Policy.infer`` result is a plain dict. Two keys are guaranteed across all
policies: ``actions`` (deployable, unnormalized-domain ``float32``
``[action_horizon, action_dim]``) and ``timing`` (at least ``model_ms`` and
``total_ms``). Policies may add model-specific keys (``Pi05Policy`` also returns
``normalized_actions`` / ``token_ids`` / ``noise``); consumers must not rely on
anything beyond the two guaranteed ones.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Protocol, runtime_checkable

import numpy as np

__all__ = ["Policy", "BareModel"]


@runtime_checkable
class BareModel(Protocol):
    """The subset of a ``apxinf_py.Model`` handle an L2 policy relies on (L1).

    Only L1 ``infer_rgb`` is part of the contract. The L0 patches path exists on
    the binding but is internal (exposed privately as ``_infer_patches``) and is
    deliberately not surfaced here.
    """

    action_horizon: int
    action_dim: int
    num_views: int
    image_size: int

    def infer_rgb(
        self,
        rgb_u8: np.ndarray,
        layout: str,
        token_ids: np.ndarray,
        noise: np.ndarray | None = None,
    ) -> np.ndarray: ...


@runtime_checkable
class Policy(Protocol):
    """Structural contract for an L2 policy: ``obs dict -> result dict``."""

    #: Static description of the policy (model_type, shapes, keys, ...).
    metadata: Mapping[str, Any]

    @property
    def action_dim(self) -> int:
        """Width of one deployable action vector (post-processing output)."""
        ...

    @property
    def action_horizon(self) -> int:
        """Number of actions in one predicted chunk."""
        ...

    def infer(
        self, observation: Mapping[str, Any], *, noise: np.ndarray | None = None
    ) -> Dict[str, Any]:
        """Map a raw observation dict (+ prompt) to a result dict.

        ``noise`` optionally supplies an exact continuous initial latent. A
        policy whose model samples such a latent internally uses that path when
        the argument is absent.

        Guarantees the ``actions`` and ``timing`` keys (see module docstring).
        """
        ...

    def close(self) -> None:
        """Release any underlying model resources."""
        ...
