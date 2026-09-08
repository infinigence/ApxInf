#!/usr/bin/env python3
"""Connect a caller-selected LIBERO ControlEnv to an ApxInf WebSocket policy.

Copy this file alone; requires numpy, Pillow, openpi-client and your LIBERO
installation. The Python interface is ApxInfPolicy.get_action(env, observation).
For CLI use, --env-factory libero.libero.envs:OffScreenRenderEnv accepts
--env-kwargs with the caller's bddl_file_name and camera settings.
No suite/task layout is hardcoded. The caller supplies the prompt and any
benchmark-specific initial state or settling protocol when using the interface.

Supports the PI0.5-LIBERO Panda OSC_POSE contract only: raw agentview and wrist
RGB, world-frame EEF position/XYZW quaternion, two raw finger joint positions,
and 7D normalized OSC delta + gripper (-1 open, +1 close). Images are rotated
180 degrees and resized/padded to 224. This is not an accuracy evaluator."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from collections import deque
from typing import Any, Mapping

import numpy as np

ACTION_DIM = 7
IMAGE_KEYS = ("observation/image", "observation/wrist_image")
STATE_KEY = "observation/state"
PROMPT_KEY = "prompt"


def quat_xyzw_to_axis_angle(value: Any) -> np.ndarray:
    """Convert robosuite's ``(x, y, z, w)`` quaternion to a rotation vector."""
    quat = np.asarray(value, dtype=np.float64)
    if quat.shape != (4,):
        raise ValueError(f"expected quaternion (4,), got {quat.shape}")
    norm = float(np.linalg.norm(quat))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("quaternion must be finite and non-zero")
    quat = quat / norm
    if quat[3] < 0.0:
        quat = -quat
    vector = quat[:3]
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm < 1e-8:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * math.atan2(vector_norm, float(np.clip(quat[3], -1.0, 1.0)))
    return np.asarray(vector * (angle / vector_norm), dtype=np.float32)


def _camera(value: Any) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(f"expected HWC RGB image, got {image.shape}")
    image = image[..., :3]
    if not image.size or not np.isfinite(image).all():
        raise ValueError("camera image must be non-empty and finite")
    if image.dtype != np.uint8:
        if not np.issubdtype(image.dtype, np.floating):
            raise ValueError(f"camera image must be uint8 or float, got {image.dtype}")
        image = np.clip(image, 0.0, 1.0) * 255.0
        image = image.astype(np.uint8)
    # PI0.5-LIBERO training orientation: rotate both raw MuJoCo images by 180 degrees.
    from PIL import Image

    image = np.ascontiguousarray(image[::-1, ::-1])
    height, width = image.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("camera image must not be empty")
    ratio = max(width / 224, height / 224)
    resized_width, resized_height = max(1, int(width / ratio)), max(1, int(height / ratio))
    resized = np.asarray(Image.fromarray(image).resize(
        (resized_width, resized_height), resample=Image.Resampling.BILINEAR,
    ))
    canvas = np.zeros((224, 224, 3), dtype=np.uint8)
    y, x = (224 - resized_height) // 2, (224 - resized_width) // 2
    canvas[y:y + resized_height, x:x + resized_width] = resized
    return canvas


def build_policy_observation(observation: Mapping[str, Any], prompt: str) -> dict:
    """Translate a Panda robosuite observation to the franka_libero wire contract."""
    required = (
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    )
    missing = [key for key in required if key not in observation]
    if missing:
        raise KeyError(f"robosuite observation is missing {missing}")
    position = np.asarray(observation["robot0_eef_pos"], dtype=np.float32)
    gripper = np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32)
    if position.shape != (3,) or gripper.shape != (2,):
        raise ValueError(
            f"expected eef_pos (3,) and gripper_qpos (2,), got "
            f"{position.shape} and {gripper.shape}"
        )
    state = np.concatenate(
        (
            position,
            quat_xyzw_to_axis_angle(observation["robot0_eef_quat"]),
            gripper,
        )
    )
    if not np.isfinite(state).all():
        raise ValueError("robot state must be finite")
    return {
        IMAGE_KEYS[0]: _camera(observation["agentview_image"]),
        IMAGE_KEYS[1]: _camera(observation["robot0_eye_in_hand_image"]),
        STATE_KEY: np.ascontiguousarray(state, dtype=np.float32),
        PROMPT_KEY: str(prompt),
    }


def validate_action_chunk(value: Any, replan_steps: int) -> np.ndarray:
    actions = np.asarray(value, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"expected actions [H,{ACTION_DIM}], got {actions.shape}")
    if actions.shape[0] < replan_steps:
        raise ValueError(
            f"server returned horizon {actions.shape[0]}, shorter than "
            f"--replan-steps {replan_steps}"
        )
    if not np.isfinite(actions).all():
        raise FloatingPointError("policy returned non-finite actions")
    return actions


def validate_server_metadata(metadata: Mapping[str, Any]) -> None:
    action_dim = metadata.get("action_dim")
    if action_dim is not None and int(action_dim) != ACTION_DIM:
        raise RuntimeError(
            f"server action_dim is {action_dim}, expected {ACTION_DIM}; "
            "serve the checkpoint with --robot franka_libero"
        )
    image_keys = metadata.get("image_keys")
    if image_keys is not None and tuple(image_keys) != IMAGE_KEYS:
        raise RuntimeError(f"server image_keys are {image_keys}, expected {list(IMAGE_KEYS)}")
    state_key = metadata.get("state_key")
    if state_key is not None and state_key != STATE_KEY:
        raise RuntimeError(f"server state_key is {state_key!r}, expected {STATE_KEY!r}")



def connect(host: str, port: int):
    """Create an OpenPI-compatible client; the simulator needs no ApxInf install."""
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    host = host.strip()
    if not host:
        raise ValueError("host must not be empty")
    for key in ("NO_PROXY", "no_proxy"):
        entries = [entry for entry in os.environ.get(key, "").split(",") if entry]
        if host not in entries:
            entries.append(host)
        os.environ[key] = ",".join(entries)
    return WebsocketClientPolicy(host, port)


def close_client(client) -> None:
    connection = getattr(client, "_ws", None)
    if connection is not None:
        connection.close()


def validate_environment(env) -> None:
    base = getattr(env, "env", env)
    if getattr(base, "action_dim", None) != ACTION_DIM or len(base.robots) != 1:
        raise ValueError("requires one Panda with 7D OSC_POSE + gripper actions")
    robot = base.robots[0]
    parts = getattr(robot, "part_controllers", {})
    controller = parts.get("right") if parts else getattr(robot, "controller", None)
    if getattr(controller, "name", None) != "OSC_POSE" or not controller.use_delta:
        raise ValueError("requires a relative OSC_POSE controller, not joint/absolute control")
    for key, expected in (
        ("input_max", np.ones(6)), ("input_min", -np.ones(6)),
        ("output_max", np.array([0.05] * 3 + [0.5] * 3)),
        ("output_min", -np.array([0.05] * 3 + [0.5] * 3)),
    ):
        if not np.allclose(getattr(controller, key), expected):
            raise ValueError(f"OSC controller {key} does not match PI0.5-LIBERO")


class ApxInfPolicy:
    """Adapt one compatible environment to PI0.5-LIBERO actions.

    The caller owns reset/step/render and must call reset() at every episode
    boundary. Injected clients remain caller-owned; close() only closes clients
    created by this adapter. No environment or background input thread is created.
    """

    def __init__(self, *, prompt: str, host: str = "127.0.0.1", port: int = 8000,
                 replan_steps: int = 5, client=None):
        if replan_steps <= 0:
            raise ValueError("replan_steps must be positive")
        self.prompt = prompt
        self.replan_steps = replan_steps
        self._plan = deque()
        self._owns_client = client is None
        self._client = connect(host, port) if client is None else client
        try:
            validate_server_metadata(self._client.get_server_metadata())
        except Exception:
            self.close()
            raise

    def set_prompt(self, prompt: str) -> None:
        self.prompt = prompt
        self.reset()

    def reset(self) -> None:
        self._plan.clear()

    def get_action(self, env, observation: Mapping[str, Any]) -> np.ndarray:
        validate_environment(env)
        if not self._plan:
            request = build_policy_observation(observation, self.prompt)
            response = self._client.infer(request)
            actions = validate_action_chunk(response["actions"], self.replan_steps)
            self._plan.extend(row.copy() for row in actions[:self.replan_steps])
        return self._plan.popleft()

    def close(self) -> None:
        if self._owns_client:
            close_client(self._client)


def run(env, policy: ApxInfPolicy, *, max_steps: int, render: bool = False) -> dict:
    """Reset and run one episode; stop at done or the explicit step limit.

    Does not close caller-owned env/policy. No grasp heuristic, hold steps,
    prompt rewriting, default scene, or benchmark success-rate calculation.
    """
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    observation = env.reset()
    policy.reset()
    done = False
    for step in range(1, max_steps + 1):
        action = policy.get_action(env, observation)
        observation, _, done, _ = env.step(action)
        if render:
            getattr(env, "env", env).render()
        if bool(done):
            policy.reset()
            break
    return {"steps": step, "done": bool(done)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-factory", required=True,
                        help="installed module:callable returning the configured environment")
    parser.add_argument("--env-kwargs", default="{}", help="JSON keyword arguments for the factory")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=520)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.max_steps <= 0 or args.replan_steps <= 0:
        parser.error("max-steps and replan-steps must be positive")
    module_name, separator, callable_name = args.env_factory.partition(":")
    if not separator or not module_name or not callable_name:
        parser.error("--env-factory must have the form module:callable")
    kwargs = json.loads(args.env_kwargs)
    if not isinstance(kwargs, dict):
        parser.error("--env-kwargs must be a JSON object")
    factory = getattr(importlib.import_module(module_name), callable_name)
    env = factory(**kwargs)
    policy = None
    try:
        policy = ApxInfPolicy(prompt=args.prompt, host=args.host, port=args.port,
                             replan_steps=args.replan_steps)
        print(run(env, policy, max_steps=args.max_steps, render=args.render), flush=True)
    finally:
        if policy is not None:
            policy.close()
        env.close()


if __name__ == "__main__":
    main()
