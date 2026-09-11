"""Translate native LIBERO simulator observations into ApxInf Observations.

**This module is a deliberate mirror of ``apxinf_robo.envs.libero``.** Both
repositories evaluate on LIBERO — ApxInf so it can regress its own engine
end-to-end without a downstream checkout, apxinf-robo so a robot deployment can
be scored — and both need the same two conversions. The duplication is accepted;
the *divergence* is not, because a changed rotation or a changed state layout
produces wrong success rates on both sides with no error anywhere.

If you change ``libero_images`` or ``libero_state`` here, change the mirror. The
golden values in ``tests/test_libero_observation.py`` (and the matching file in
apxinf-robo) are stated as literals for exactly that reason: they cannot be
kept passing by editing the derivation.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(quat[3]) / denominator).astype(np.float32)


def libero_images(base: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    """Orient raw LIBERO frames; the selected policy owns model-specific resize."""
    return np.stack(
        [np.ascontiguousarray(base[::-1, ::-1]), np.ascontiguousarray(wrist[::-1, ::-1])]
    )


def libero_state(observation, *, finger_joints: int = 1) -> np.ndarray:
    """Convert LIBERO proprioception into the vector a checkpoint consumes.

    LIBERO reports two mirrored finger joints. ``finger_joints=1`` collapses
    them into a single gripper coordinate (7 values), the openpi-derived
    convention this harness has used for PI0.5. ``finger_joints=2`` keeps both
    (8 values), which is what LeRobot's own LIBERO environment produces and what
    a checkpoint trained on ``HuggingFaceVLA/libero`` was trained on — π0-FAST's
    ``observation.state`` statistics are 8 wide, so its policy declares
    ``state_dim=8`` and the harness builds that vector.
    """
    gripper = np.asarray(observation["robot0_gripper_qpos"]).reshape(-1)
    if gripper.size != 2:
        raise ValueError(f"robot0_gripper_qpos must have 2 values, got {gripper.size}")
    if finger_joints not in (1, 2):
        raise ValueError(f"finger_joints must be 1 or 2, got {finger_joints}")
    return np.concatenate(
        (
            observation["robot0_eef_pos"],
            quat_to_axis_angle(observation["robot0_eef_quat"]),
            gripper[:finger_joints],
        )
    ).astype(np.float32, copy=False)


def make_env(task, seed: int):
    """Build the same off-screen LIBERO environment used by evaluation."""
    try:
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as error:
        raise ImportError(
            "native LIBERO observations require the LIBERO and MuJoCo evaluation "
            "dependencies; install LIBERO as described in README.md"
        ) from error

    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(seed)
    return env


def to_apxinf_observation(
    observation,
    *,
    prompt: str,
    image_keys: tuple[str, str],
    prompt_key: str,
    state_key: str,
    finger_joints: int = 1,
) -> dict:
    """Convert one raw simulator frame using the evaluation-time convention."""
    images = libero_images(
        observation["agentview_image"],
        observation["robot0_eye_in_hand_image"],
    )
    state = libero_state(observation, finger_joints=finger_joints)
    return {
        image_keys[0]: images[0],
        image_keys[1]: images[1],
        state_key: state,
        prompt_key: prompt,
    }
