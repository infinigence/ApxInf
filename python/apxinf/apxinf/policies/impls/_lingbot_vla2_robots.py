"""Bundled LingBot-VLA 2.0 robot configs, transcribed from ``configs/robot_configs/*.yaml``.

Source: github.com/robbyant/lingbot-vla-v2 (Apache-2.0). Kept as plain mappings so no data
files need packaging; :func:`~apxinf.policies.impls.lingbot_vla2.load_robot_config` also takes
any YAML path with the same schema. The ``norm_stats`` entry of the upstream files is dropped:
statistics are passed to the policy explicitly.
"""

from __future__ import annotations

from typing import Any, Dict

__all__ = ["BUNDLED_ROBOTS"]


def _slices(key: str, *spans):
    return {"origin_keys": [{key: {"start": a, "end": b}} for a, b in spans]}


BUNDLED_ROBOTS: Dict[str, Dict[str, Any]] = {
    # RoboTwin 2.0 dual-arm: a 14-vector [left arm 6, left gripper, right arm 6, right gripper].
    "robotwin": {
        "states": [
            {"observation.state.arm.position": _slices("observation.state", (0, 6), (7, 13))},
            {"observation.state.effector.position": _slices("observation.state", (6, 7), (13, 14))},
        ],
        "actions": [
            {"action.arm.position": {**_slices("action", (0, 6), (7, 13)), "subtract_state": False}},
            {"action.effector.position": {**_slices("action", (6, 7), (13, 14)), "subtract_state": False}},
        ],
        "images": [
            {"observation.images.camera_top": {"origin_keys": "observation.images.cam_high"}},
            {"observation.images.camera_wrist_left": {"origin_keys": "observation.images.cam_left_wrist"}},
            {"observation.images.camera_wrist_right": {"origin_keys": "observation.images.cam_right_wrist"}},
        ],
    },
    # AgileX Cobot Magic: state and action already split per joint group, arm actions are deltas.
    # DEVIATION from upstream configs/robot_configs/agilex_cobot_magic.yaml: it omits ``relative_type`` on
    # ``action.arm.position``, which upstream defaults to ``quaternion_local``; FeatureTransform then asserts
    # ``'end.position' in action_feature`` in both apply() and unapply(), so the file cannot run unchanged.
    # ``joint`` (any value outside the quaternion set) selects upstream's plain ``action += state`` branch.
    "agilex_cobot_magic": {
        "states": [
            {"observation.state.arm.position": {"origin_keys": "observation.state.arm.position"}},
            {"observation.state.effector.position": {"origin_keys": "observation.state.effector.position"}},
        ],
        "actions": [
            {"action.arm.position": {"origin_keys": "action.arm.position", "subtract_state": True,
                                      "relative_type": "joint"}},
            {"action.effector.position": {"origin_keys": "action.effector.position", "subtract_state": False}},
        ],
        "images": [
            "observation.images.camera_top",
            "observation.images.camera_wrist_left",
            "observation.images.camera_wrist_right",
        ],
    },
}
