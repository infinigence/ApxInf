"""Translate native LIBERO simulator observations into ApxInf Observations.

**The OpenPI conversion in this module deliberately mirrors
``apxinf_robo.envs.libero``.** Both repositories evaluate on LIBERO — ApxInf so
it can regress its own engine end-to-end without a downstream checkout,
apxinf-robo so a robot deployment can be scored — and both need the same camera
and seven-value state conversion. GR00T's named eight-value state and decoded
gripper conversion mirror NVIDIA's official ``LiberoEnv`` instead. The
duplication is accepted; the *divergence* is not, because a changed rotation,
state layout, or action convention produces wrong success rates with no error.

If you change ``libero_images`` or ``libero_state`` here, change the mirror. The
golden values in ``tests/test_libero_observation.py`` (and the matching file in
apxinf-robo) are stated as literals for exactly that reason: they cannot be
kept passing by editing the derivation.
"""

from __future__ import annotations

import math
import pathlib
import pickle

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


def libero_state(observation) -> np.ndarray:
    """Convert LIBERO's two mirrored finger joints to one gripper coordinate."""
    gripper = np.asarray(observation["robot0_gripper_qpos"]).reshape(-1)
    if gripper.size != 2:
        raise ValueError(f"robot0_gripper_qpos must have 2 values, got {gripper.size}")
    return np.concatenate(
        (
            observation["robot0_eef_pos"],
            quat_to_axis_angle(observation["robot0_eef_quat"]),
            gripper[:1],
        )
    ).astype(np.float32, copy=False)


def libero_gr00t_state(observation) -> dict[str, np.ndarray]:
    """Preserve NVIDIA GR00T's named 8-DoF LIBERO state contract.

    Unlike the OpenPI wire convention used by :func:`libero_state`, the
    official GR00T processor consumes both mirrored finger joint positions.
    Keeping this conversion beside the shared camera/orientation conversion
    prevents evaluation from silently dropping the second joint.
    """
    position = np.asarray(observation["robot0_eef_pos"], dtype=np.float32).reshape(-1)
    gripper = np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    if position.size != 3 or gripper.size != 2:
        raise ValueError(
            "GR00T LIBERO state requires eef_pos with 3 values and "
            f"gripper_qpos with 2 values, got {position.size} and {gripper.size}"
        )
    rotation = quat_to_axis_angle(observation["robot0_eef_quat"])
    return {
        "x": np.ascontiguousarray(position[0:1]),
        "y": np.ascontiguousarray(position[1:2]),
        "z": np.ascontiguousarray(position[2:3]),
        "roll": np.ascontiguousarray(rotation[0:1]),
        "pitch": np.ascontiguousarray(rotation[1:2]),
        "yaw": np.ascontiguousarray(rotation[2:3]),
        "gripper": np.ascontiguousarray(gripper),
    }


def libero_gr00t_action(action: np.ndarray) -> np.ndarray:
    """Map decoded GR00T LIBERO actions to robosuite's gripper convention.

    NVIDIA's official ``LiberoEnv`` performs this conversion after policy
    decode: the dataset uses ``0=closed, 1=open`` while robosuite expects
    ``+1=closed, -1=open``. The six Cartesian components pass through.
    """
    action = np.asarray(action, dtype=np.float32)
    if action.ndim not in (1, 2) or action.shape[-1] != 7:
        raise ValueError(f"GR00T LIBERO action must end in 7 values, got {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError("GR00T LIBERO action must contain only finite values")
    result = np.ascontiguousarray(action.copy())
    result[..., -1] = -np.sign(2.0 * result[..., -1] - 1.0)
    return result


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


def load_libero_init_states(suite, task_id: int):
    """Load LIBERO's trusted bundled init states across PyTorch versions.

    LIBERO releases before PyTorch 2.6 call ``torch.load(path)`` and therefore
    inherit the new ``weights_only=True`` default, which rejects their NumPy
    arrays before an episode starts. Keep the vendor call on its normal path;
    only that exact compatibility failure is retried against LIBERO's bundled,
    read-only init-state file with the legacy behavior made explicit.
    """
    try:
        return suite.get_task_init_states(task_id)
    except pickle.UnpicklingError as error:
        if "Weights only load failed" not in str(error):
            raise

    import torch
    from libero.libero import get_libero_path

    task = suite.get_task(task_id)
    init_states_path = (
        pathlib.Path(get_libero_path("init_states"))
        / task.problem_folder
        / task.init_states_file
    )
    return torch.load(init_states_path, weights_only=False)


def to_apxinf_observation(
    observation,
    *,
    prompt: str,
    image_keys: tuple[str, str],
    prompt_key: str,
    state_key: str,
) -> dict:
    """Convert one raw simulator frame using the evaluation-time convention."""
    images = libero_images(
        observation["agentview_image"],
        observation["robot0_eye_in_hand_image"],
    )
    state = libero_state(observation)
    return {
        image_keys[0]: images[0],
        image_keys[1]: images[1],
        state_key: state,
        prompt_key: prompt,
    }
