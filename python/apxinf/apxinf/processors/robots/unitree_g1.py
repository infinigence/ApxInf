"""Unitree G1 (humanoid, dual-arm + 2 dexterous hands) processing steps.

Ports the robot-specific numpy transforms of an integrator's Unitree G1
``DataTransformFn`` pipeline (a ``pi05_UnitreeG1`` OpenPI finetune: 3 cameras,
16-DoF state ``[L-arm 7, L-gripper 1, R-arm 7, R-gripper 1]``, ``action_dim=32``,
``action_horizon=50``, delta joint actions) into our
:class:`~apxinf.processors.base.ProcessorStep` contract **without importing their
framework**:

===========================  ==============================================
integrator transform          apxinf equivalent
===========================  ==============================================
camera rename + CHW->HWC       ``image_keys`` config + ``ParseImage`` (already
  + float->uint8                handles CHW->HWC and float->uint8)
``_decode_state``              :class:`UnitreeG1DecodeState` (input step)
32-wide unnormalize            existing ``Unnormalize`` (full model width)
``AbsoluteActions`` (delta)    :class:`UnitreeG1AbsoluteActions` (output step)
``unitreeG1Outputs`` 32->16    :class:`UnitreeG1EncodeActions` (output step)
  + joint flip + gripper
===========================  ==============================================

These steps are model-agnostic — they reference no policy symbols and vary only
with the G1 body. They are also **dataset-agnostic**: the G1 client's wire keys
(``images/cam_high``, a flat ``state``) are a recording convention, so they live
in :mod:`apxinf.conventions` and reach these steps as arguments. What is left
here is arithmetic that changes only if the hardware does — the 16-DoF layout,
the delta mask, the gripper and joint-sign conventions.

The :func:`~apxinf.robots.unitree_g1.build_unitree_g1_policy`
adapter wraps them around whichever policy the checkpoint turns out to be, using
:meth:`~apxinf.policies.base.ComposablePolicy.with_adapter`; it names no model
class either.

.. note::
   The joint-flip mask and gripper transforms in the integrator's file are
   near-identity placeholders ("需要根据实际参数更新"): the flip mask is all
   ``1``, the gripper maps are ``clip(0,1)`` round-trips. They are ported
   faithfully (as the hooks where real G1 calibration would live) but currently
   pass their input through unchanged.
"""

from __future__ import annotations

from typing import Any, MutableMapping

import numpy as np

from ..base import ProcessorStep
from ..transforms import ACTIONS, OBSERVATION, lookup_key, set_key

__all__ = [
    "UnitreeG1DecodeState",
    "UnitreeG1AbsoluteActions",
    "UnitreeG1EncodeActions",
    "G1_ROBOT_DIM",
    "G1_DELTA_MASK",
]

#: G1 state/action layout: [L-arm 7, L-gripper 1, R-arm 7, R-gripper 1].
G1_ROBOT_DIM = 16
_GRIPPER_INDICES = (7, 15)


def _joint_flip_mask() -> np.ndarray:
    """Per-joint ``+1/-1`` sign convention between G1 and pi runtime.

    All ``+1`` in the integrator's file (a documented placeholder). Kept as the
    hook for real left/right mirror calibration.

    Integer dtype on purpose, matching openpi's ``np.array([1, 1, ...])``: these
    are exact signs, and the ``int64 * float32 -> float64`` promotion is what
    puts openpi's whole G1 input chain in float64. Reproducing it is what makes
    our discretized prompt bit-identical to theirs (see
    :func:`~apxinf.processors.tokenize.discretize_state`).
    """
    return np.ones(G1_ROBOT_DIM, dtype=np.int64)


def _gripper_to_angular(value: np.ndarray) -> np.ndarray:
    """G1 gripper -> pi angular space. Currently ``clip(value, 0, 1)`` (identity
    on the ``[0, 1]`` on/off command range the integrator uses)."""
    return np.clip(value, 0.0, 1.0)


def _gripper_from_angular(value: np.ndarray) -> np.ndarray:
    """pi angular -> G1 gripper space. Inverse of :func:`_gripper_to_angular`;
    currently ``clip(value, 0, 1)``."""
    return np.clip(value, 0.0, 1.0)


def _make_bool_mask(*dims: int) -> np.ndarray:
    """openpi ``make_bool_mask``: ``+n`` -> ``n`` True, ``-n`` -> ``n`` False.

    ``make_bool_mask(7, -1, 7, -1)`` -> the G1 delta mask: arm joints delta
    (True), grippers absolute (False).
    """
    parts = [np.full(abs(n), n > 0, dtype=bool) for n in dims]
    return np.concatenate(parts)


#: Delta-action mask: arm joints are trained as deltas, grippers as absolute.
G1_DELTA_MASK = _make_bool_mask(7, -1, 7, -1)


class UnitreeG1DecodeState(ProcessorStep):
    """Decode raw G1 state into pi runtime space before it enters the model.

    Ports ``_decode_state``: joint-flip then gripper-to-angular on the two
    gripper dims. Runs *before* ``tokenize`` so discretized state (when enabled)
    and the delta->absolute output step both see the decoded state. Operates on a
    shallow copy of the observation so the caller's dict is left untouched. A
    no-op when no state is present (state-off serving).

    ``state_key`` is required and has no default: which key carries state is a
    property of the *convention* the data was recorded under
    (:mod:`apxinf.conventions`), not of this body, and a default here would be
    one dataset's dialect built into a robot's arithmetic.
    """

    def __init__(self, state_key: str, *, observation_key: str = OBSERVATION):
        self.state_key = state_key
        self.observation_key = observation_key

    def __call__(self, data: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        observation = data.get(self.observation_key)
        if observation is None:
            return data
        raw = lookup_key(observation, self.state_key, None)
        if raw is None:
            return data
        raw = np.asarray(raw)
        # Promote to at least float32 before the flip so an integer state cannot
        # be truncated by the gripper clip (openpi has no such guard). The int64
        # flip mask then promotes the product to float64, matching openpi.
        state = raw.astype(np.result_type(raw.dtype, np.float32), copy=False) * _joint_flip_mask()
        idx = list(_GRIPPER_INDICES)
        state[idx] = _gripper_to_angular(state[idx])
        data[self.observation_key] = set_key(observation, self.state_key, state)
        return data


class UnitreeG1AbsoluteActions(ProcessorStep):
    """Convert the model's delta arm-joint actions back to absolute using state.

    Ports openpi ``AbsoluteActions``: on masked dims (arm joints), add the current
    decoded state; gripper dims are already absolute and pass through. Reads the
    unnormalized ``actions`` and the decoded ``observation`` state threaded in by
    ``Pi05Policy.infer``. A no-op when state is absent (delta cannot be resolved).
    ``state_key`` is required for the same reason as in
    :class:`UnitreeG1DecodeState`.
    """

    def __init__(self, state_key: str, *, observation_key: str = OBSERVATION):
        self.state_key = state_key
        self.observation_key = observation_key

    def __call__(self, data: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        observation = data.get(self.observation_key) or {}
        state = lookup_key(observation, self.state_key, None)
        if state is None:
            return data
        # Both operands keep their own dtype: openpi's output chain is float64
        # end to end (its stats are float64 and its flip mask is int64), so
        # coercing either side to float32 here would reintroduce the ~2e-7
        # divergence the rest of the G1 path was aligned to remove (A15).
        actions = np.array(data[ACTIONS], copy=True)
        state = np.asarray(state)[:G1_ROBOT_DIM]
        offset = np.where(G1_DELTA_MASK, state, state.dtype.type(0))
        actions[:, :G1_ROBOT_DIM] = actions[:, :G1_ROBOT_DIM] + offset
        data[ACTIONS] = actions
        return data


class UnitreeG1EncodeActions(ProcessorStep):
    """Map absolute pi actions back to G1 robot space: 32->16, flip, gripper.

    Ports ``unitreeG1Outputs`` + ``_encode_actions``: truncate to the 16 robot
    dims, apply the joint flip, and gripper-from-angular on the two gripper dims.
    """

    def __call__(self, data: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        # No dtype coercion, in either direction. The +-1 flip is exact in any
        # float dtype, but the int64 mask promotes a float32 input to float64 --
        # which is precisely what openpi's ``_encode_actions`` does, so its G1
        # server puts float64 actions on the wire. Matching that keeps the two
        # servers' outputs comparable bit for bit (A15); an openpi G1 client
        # reads the dtype off the msgpack payload either way.
        actions = np.asarray(data[ACTIONS])[:, :G1_ROBOT_DIM] * _joint_flip_mask()
        idx = list(_GRIPPER_INDICES)
        actions[:, idx] = _gripper_from_angular(actions[:, idx])
        data[ACTIONS] = np.ascontiguousarray(actions)
        return data
