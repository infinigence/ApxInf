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
with the G1 body. The :func:`~apxinf.robots.unitree_g1.build_unitree_g1_policy`
adapter assembles them with a :class:`~apxinf.policies.impls.pi05.Pi05Policy`.

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
    "G1_CAMERAS",
    "G1_STATE_KEY",
    "G1_ROBOT_DIM",
    "G1_DELTA_MASK",
]

#: G1 camera **wire keys**, ordered base / left-wrist / right-wrist to match the
#: model's view slots (openpi ``base_0_rgb`` / ``left_wrist_0_rgb`` /
#: ``right_wrist_0_rgb``). These are the keys an unmodified openpi G1 client
#: sends, i.e. the nested ``obs["images"]["cam_high"]`` layout, spelled as a path
#: for :func:`~apxinf.processors.transforms.lookup_key`.
G1_CAMERAS = ("images/cam_high", "images/cam_left_wrist", "images/cam_right_wrist")

#: G1 state wire key. openpi's G1 client sends a flat top-level ``"state"``, not
#: LIBERO's ``"observation/state"``.
G1_STATE_KEY = "state"

#: G1 state/action layout: [L-arm 7, L-gripper 1, R-arm 7, R-gripper 1].
G1_ROBOT_DIM = 16
_GRIPPER_INDICES = (7, 15)


def _joint_flip_mask() -> np.ndarray:
    """Per-joint ``+1/-1`` sign convention between G1 and pi runtime.

    All ``+1`` in the integrator's file (a documented placeholder). Kept as the
    hook for real left/right mirror calibration.
    """
    return np.ones(G1_ROBOT_DIM, dtype=np.float32)


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
    """

    def __init__(self, state_key: str = G1_STATE_KEY, *, observation_key: str = OBSERVATION):
        self.state_key = state_key
        self.observation_key = observation_key

    def __call__(self, data: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        observation = data.get(self.observation_key)
        if observation is None:
            return data
        raw = lookup_key(observation, self.state_key, None)
        if raw is None:
            return data
        state = np.asarray(raw, dtype=np.float32) * _joint_flip_mask()
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
    """

    def __init__(self, state_key: str = G1_STATE_KEY, *, observation_key: str = OBSERVATION):
        self.state_key = state_key
        self.observation_key = observation_key

    def __call__(self, data: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        observation = data.get(self.observation_key) or {}
        state = lookup_key(observation, self.state_key, None)
        if state is None:
            return data
        actions = np.asarray(data[ACTIONS], dtype=np.float32).copy()
        state = np.asarray(state, dtype=np.float32)[:G1_ROBOT_DIM]
        offset = np.where(G1_DELTA_MASK, state, 0.0)
        actions[:, :G1_ROBOT_DIM] = actions[:, :G1_ROBOT_DIM] + offset
        data[ACTIONS] = actions
        return data


class UnitreeG1EncodeActions(ProcessorStep):
    """Map absolute pi actions back to G1 robot space: 32->16, flip, gripper.

    Ports ``unitreeG1Outputs`` + ``_encode_actions``: truncate to the 16 robot
    dims, apply the joint flip, and gripper-from-angular on the two gripper dims.
    """

    def __call__(self, data: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        actions = np.asarray(data[ACTIONS], dtype=np.float32)[:, :G1_ROBOT_DIM] * _joint_flip_mask()
        idx = list(_GRIPPER_INDICES)
        actions[:, idx] = _gripper_from_angular(actions[:, idx])
        data[ACTIONS] = np.ascontiguousarray(actions)
        return data
