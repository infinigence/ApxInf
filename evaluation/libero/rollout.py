"""LIBERO environment adaptation and one-episode rollout."""

from __future__ import annotations

import collections
import math
import pathlib
import time
from typing import Protocol, Tuple

import numpy as np
from PIL import Image

from .contract import LIBERO_ACTION_DIM


MAX_STEPS = 520
WAIT_STEPS = 10
REPLAN_STEPS = 5
IMAGE_SIZE = 224


class ActionClient(Protocol):
    def infer(self, base, wrist, state, prompt) -> Tuple[np.ndarray, dict]: ...


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(quat[3]) / denominator).astype(np.float32)


def resize_images(base: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    """Rotate LIBERO cameras and letterbox them to the policy input size."""

    def resize_with_pad(image: np.ndarray) -> np.ndarray:
        image = np.ascontiguousarray(image[::-1, ::-1])
        height, width = image.shape[:2]
        ratio = max(width / IMAGE_SIZE, height / IMAGE_SIZE)
        resized_height = int(height / ratio)
        resized_width = int(width / ratio)
        resized = np.asarray(
            Image.fromarray(image).resize(
                (resized_width, resized_height), resample=Image.Resampling.BILINEAR
            )
        )
        canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        offset_y = (IMAGE_SIZE - resized_height) // 2
        offset_x = (IMAGE_SIZE - resized_width) // 2
        canvas[
            offset_y : offset_y + resized_height,
            offset_x : offset_x + resized_width,
        ] = resized
        return canvas

    return np.stack([resize_with_pad(base), resize_with_pad(wrist)], axis=0)


def make_env(task, seed: int):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=256, camera_widths=256
    )
    env.seed(seed)
    return env


def run_episode(
    env,
    initial_state: np.ndarray,
    suite: str,
    task_id: int,
    trial_id: int,
    prompt: str,
    client: ActionClient,
    transport: str,
    seed: int,
) -> dict:
    """Run until success or the action budget, rejecting unsafe server output."""
    episode_started = time.perf_counter()
    env.reset()
    observation = env.set_init_state(initial_state)
    for _ in range(WAIT_STEPS):
        observation, _, _, _ = env.step([0.0] * 6 + [-1.0])

    action_plan: collections.deque[np.ndarray] = collections.deque()
    success = False
    action_steps = 0
    replans = 0
    preprocess_seconds = 0.0
    inference_seconds = 0.0
    model_seconds = 0.0
    server_processor_seconds = 0.0
    transport_seconds = 0.0
    first_action_checksum = None

    while action_steps < MAX_STEPS:
        if not action_plan:
            preprocess_started = time.perf_counter()
            images = resize_images(
                observation["agentview_image"],
                observation["robot0_eye_in_hand_image"],
            )
            state = np.concatenate(
                (
                    observation["robot0_eef_pos"],
                    quat_to_axis_angle(observation["robot0_eef_quat"]),
                    observation["robot0_gripper_qpos"],
                )
            ).astype(np.float32, copy=False)
            preprocess_seconds += time.perf_counter() - preprocess_started

            request_started = time.perf_counter()
            actions, timing = client.infer(images[0], images[1], state, prompt)
            round_trip_seconds = time.perf_counter() - request_started
            inference_seconds += round_trip_seconds
            if (
                actions.ndim != 2
                or actions.shape[1] != LIBERO_ACTION_DIM
                or actions.shape[0] < REPLAN_STEPS
            ):
                raise ValueError(
                    f"expected actions (>= {REPLAN_STEPS}, {LIBERO_ACTION_DIM}), "
                    f"got {actions.shape}"
                )
            if not np.isfinite(actions).all():
                raise FloatingPointError("server returned non-finite actions")

            segment_model = float(timing.get("model_seconds", 0.0))
            segment_processor = float(timing.get("server_processor_seconds", 0.0))
            model_seconds += segment_model
            server_processor_seconds += segment_processor
            transport_seconds += max(
                0.0, round_trip_seconds - segment_model - segment_processor
            )
            if first_action_checksum is None:
                first_action_checksum = float(np.abs(actions).sum())
            action_plan.extend(actions[:REPLAN_STEPS])
            replans += 1

        observation, _, done, _ = env.step(action_plan.popleft().tolist())
        action_steps += 1
        if done:
            success = True
            break

    return {
        "status": "completed",
        "suite": suite,
        "task_id": task_id,
        "trial_id": trial_id,
        "prompt": prompt,
        "success": success,
        "action_steps": action_steps,
        "replans": replans,
        "preprocess_seconds": preprocess_seconds,
        "inference_seconds": inference_seconds,
        "model_seconds": model_seconds,
        "server_processor_seconds": server_processor_seconds,
        "websocket_transport_seconds": transport_seconds,
        "elapsed_seconds": time.perf_counter() - episode_started,
        "first_action_abs_checksum": first_action_checksum,
        "transport": transport,
        "image_input": "openpi_uint8_hwc",
        "seed": seed,
    }
