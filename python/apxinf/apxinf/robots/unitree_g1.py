"""Unitree G1 adapter: wire the G1 steps onto a pi05 policy.

Assembles the robot-specific steps in
:mod:`apxinf.processors.robots.unitree_g1` with a generic
:class:`~apxinf.policies.impls.pi05.Pi05Policy`. Only the primary serving path
(3 cameras, ``adapt_to_pi=True``, no fixed-hand override) is wired — that is what
the shipped ``pi05_UnitreeG1_groundwire`` config uses; the integrator's
``_NoLeftCam`` / ``fixed_hand`` variants are training-data cleaning knobs, not
serving paths, so they are intentionally omitted.

.. note::
   With a stand-in pi05 checkpoint (no real G1 weights / norm_stats) this adapter
   validates *plumbing and shape* (G1 obs -> ``[action_horizon, 16]``), not G1
   action values. Numeric parity needs the integrator's real gripper limits
   **and** the G1 checkpoint/norm_stats; the same adapter code then runs
   unchanged.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from ..policies.impls.pi05 import Pi05Policy
from ..processors import Pipeline
from ..processors.robots.unitree_g1 import (
    G1_CAMERAS,
    G1_ROBOT_DIM,
    UnitreeG1AbsoluteActions,
    UnitreeG1DecodeState,
    UnitreeG1EncodeActions,
)
from ..processors.transforms import Unnormalize

__all__ = ["build_unitree_g1_policy"]


def build_unitree_g1_policy(
    model_dir,
    *,
    use_delta_joint_actions: bool = True,
    adapt_to_pi: bool = True,
    state_key: str = "observation/state",
    image_keys: Sequence[str] = G1_CAMERAS,
    discrete_state: bool = False,
    unnormalizer: Optional[Unnormalize] = None,
    metadata: Optional[dict] = None,
    **from_pretrained_kwargs: Any,
) -> Pi05Policy:
    """Build a :class:`Pi05Policy` wired with the Unitree G1 pre/post adapter.

    Loads the checkpoint through the generic :meth:`Pi05Policy.from_pretrained`
    (full model-width unnormalizer, so the delta->absolute step sees the whole
    action before the 32->16 robot truncation), then swaps in the G1 pipelines:

    * **input**  — ``[image_stack, g1_decode_state?, tokenize]``; PI0.5 samples
      its latent internally unless the caller supplies ``noise``
    * **output** — ``[unnormalize, g1_absolute?, g1_encode]``

    ``adapt_to_pi=False`` drops the decode/encode conventions (raw robot space);
    ``use_delta_joint_actions=False`` drops the delta->absolute step (the model
    already emits absolute actions). By default the unnormalizer comes from the
    checkpoint's ``norm_stats["actions"]``, so a real deployment needs G1
    norm_stats at least ``16`` wide; pass ``unnormalizer=`` to inject one (e.g. a
    full-width identity for a shape/plumbing test on a stand-in checkpoint).
    """
    base = Pi05Policy.from_pretrained(
        model_dir,
        image_keys=tuple(image_keys),
        action_dim=None,  # keep full model width; the g1_encode step truncates to 16
        state_key=state_key,
        discrete_state=discrete_state,
        **from_pretrained_kwargs,
    )
    if unnormalizer is None:
        unnormalizer = base.output_pipeline["unnormalize"]

    input_pipeline = base.input_pipeline
    if adapt_to_pi:
        input_pipeline = input_pipeline.insert_before(
            "tokenize", ("g1_decode_state", UnitreeG1DecodeState(state_key))
        )

    output_steps = [("unnormalize", unnormalizer)]
    if use_delta_joint_actions:
        output_steps.append(("g1_absolute", UnitreeG1AbsoluteActions(state_key)))
    if adapt_to_pi:
        output_steps.append(("g1_encode", UnitreeG1EncodeActions()))
    output_pipeline = Pipeline(output_steps)

    return Pi05Policy(
        base.model,
        input_pipeline=input_pipeline,
        output_pipeline=output_pipeline,
        image_keys=tuple(image_keys),
        state_key=state_key,
        action_dim=G1_ROBOT_DIM,
        metadata={"robot": "unitree_g1", **(metadata or {})},
    )
