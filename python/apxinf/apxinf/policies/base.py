"""The contracts shared across the frontend: :class:`Policy` and :class:`BareModel`.

Both are structural :class:`typing.Protocol` types — nothing has to inherit
them; any object of the right shape satisfies them. They are the stable anchor
points other layers code against:

* :class:`BareModel` — the model-side contract (L1 bare inference) a policy
  consumes, i.e. the subset of a ``apxinf_py.Model`` handle it relies on.
* :class:`Policy` — the L2 contract (``obs dict -> result dict``) every
  model-specific policy satisfies. Downstream consumers (the websocket server, the
  :class:`~apxinf.policies.auto.AutoPolicy` registry, a lerobot adaptor) code
  against this, never a concrete class.
* :class:`ComposablePolicy` — the narrow extra capability a policy needs before
  an *outer* layer (a robot adapter) can wrap steps around its chain. Optional:
  a policy is a perfectly good :class:`Policy` without it.
* :data:`VIEW_SLOTS` — the camera slot names a checkpoint's weights consume, in
  order. Model vocabulary, kept here so a policy and an outer adaptation layer
  can both name a slot without importing each other.
* :data:`CANONICAL_IMAGE_KEYS` / :data:`CANONICAL_STATE_KEY` /
  :data:`CANONICAL_PROMPT_KEY` — the observation keys a policy reads when the
  caller names none. Not a dataset's dialect: the *absence* of one.

The ``Policy.infer`` result is a plain dict. Two keys are guaranteed across all
policies: ``actions`` (deployable, unnormalized-domain ``float32``
``[action_horizon, action_dim]``) and ``timing`` (at least ``model_ms`` and
``total_ms``). Policies may add model-specific keys (``Pi05Policy`` also returns
``normalized_actions`` / ``token_ids`` / ``noise``); consumers must not rely on
anything beyond the two guaranteed ones.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

__all__ = [
    "Policy",
    "BareModel",
    "ComposablePolicy",
    "VIEW_SLOTS",
    "CANONICAL_IMAGE_KEYS",
    "CANONICAL_STATE_KEY",
    "CANONICAL_PROMPT_KEY",
]

#: Camera view slots, in the order a checkpoint's weights consume them, as named
#: by openpi's ``model.IMAGE_KEYS``.
#:
#: These are **model** vocabulary, not wire keys and not a dataset's convention.
#: A checkpoint's ``num_views`` says how many of these it was trained on; the
#: *order* is baked into the weights, and nothing here ever crosses the network.
#: They live in this module rather than in a policy implementation because both
#: sides need to agree on them without either importing the other: a policy uses
#: them to name the cameras it wants when the caller does not, and an outer
#: adaptation layer uses them to say which slot each of its wire keys fills.
#:
#: The pi05 family shares this vocabulary. A model with a different camera layout
#: declares its own rather than stretching this tuple.
VIEW_SLOTS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

#: The observation keys a policy reads when the caller names none.
#:
#: This engine holds **no dataset's wire contract**. What a client calls its
#: cameras is a property of the recording, not of the weights, and it varies
#: across datasets that share one checkpoint architecture — so naming keys is the
#: caller's job, through ``image_keys=`` / ``state_key=`` / ``prompt_key=``.
#:
#: These three constants are what is left when nobody names anything: the cameras
#: fall back to the model's own slot names, and the prompt to the one spelling
#: openpi's wire has always used. They are published so that a caller *can*
#: address the fallback contract by name instead of restating string literals,
#: and so an adaptation layer can assert against them.
#:
#: ``CANONICAL_STATE_KEY`` is deliberately **not** a default anywhere. State has
#: no safe fallback: a wrong camera key raises on the first inference, a wrong
#: state key is silent, and a policy that reads state refuses to be built without
#: an explicit ``state_key``. It is here to name the neutral spelling, not to
#: supply one.
CANONICAL_IMAGE_KEYS = VIEW_SLOTS
CANONICAL_STATE_KEY = "state"
CANONICAL_PROMPT_KEY = "prompt"


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


@runtime_checkable
class ComposablePolicy(Protocol):
    """A :class:`Policy` that accepts outer input and output processing steps."""

    def with_adapter(
        self,
        *,
        before: Sequence[Any] = (),
        after: Sequence[Any] = (),
        action_dim: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "Policy":
        """Return a policy running ``before`` ahead of, and ``after`` behind, its own chain.

        Each entry is a ``(name, step)`` pair (a
        :data:`~apxinf.processors.base.StepSpec`). ``action_dim`` declares the
        deployable width the wrapped chain now emits, since an appended step may
        change it; ``metadata`` is merged over the inherited description. The
        returned policy shares the underlying model handle — do not close both.
        """
        ...
