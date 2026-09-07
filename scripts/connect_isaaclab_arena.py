#!/usr/bin/env python3
"""Standalone Isaac Lab-Arena -> ApxInf policy adapter (PI0.5-LIBERO contract).

Copy this file alone. Implements the PolicyBase public protocol consumed by
Arena's native policy_runner without importing Arena/torch before AppLauncher.
Targets Arena aad4f25 (feature/arena_v0.2_on_lab_2.3) and Isaac Lab 2.3.2.
Requires numpy, openpi-client, and a working caller-installed Arena stack;
this file does not install assets or patch third-party packages.

Invoke this script with Arena's environment arguments plus --apxinf-primary-camera
and --apxinf-wrist-camera. Arena owns scene/reset/step/metrics. No default task,
LW-BenchHub placement, camera pose override, or interactive prompt UI is bundled.
ApxInfPolicy can also be imported and used with a caller-created Arena environment.
Policy state requires eef_pos/eef_quat (WXYZ)/gripper_pos; images come from
camera_obs. Camera keys are explicit: no silent one-camera-to-two-view fallback.

Only relative-pose IK arm_action followed by binary gripper_action is supported.
Physical frame/frequency compatibility remains the caller's responsibility.
Extra action channels are zero for a stationary base, not a general mobile policy."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from collections import deque
from typing import Any, Mapping

import numpy as np

POLICY_ACTION_DIM = 7
POLICY_IMAGE_KEYS = ("observation/image", "observation/wrist_image")
POLICY_STATE_KEY = "observation/state"
POLICY_PROMPT_KEY = "prompt"


def _scale_vector(value: Any) -> np.ndarray:
    scale = np.asarray(value, dtype=np.float32)
    if scale.shape == ():
        scale = np.full(6, scale, dtype=np.float32)
    if scale.shape != (6,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("relative IK scale must be a positive finite scalar or six-vector")
    return scale


def relative_ik_scale(env: Any) -> np.ndarray:
    """Read the target environment contract; never guess from the robot name."""
    arm = env.unwrapped.cfg.actions.arm_action
    controller = getattr(arm, "controller", None)
    if (getattr(controller, "command_type", None) != "pose"
            or not getattr(controller, "use_relative_mode", False)):
        raise ValueError("LIBERO adapter requires a relative pose IK arm_action controller")
    return _scale_vector(arm.scale)


def libero_action_to_relative_ik(
    value: Any, *, controller_scale: Any, action_dim: int = 7,
) -> np.ndarray:
    """Match LIBERO's +/-5 cm, +/-0.5 rad OSC limits and gripper intent.

    Input is the server's dataset-domain action, not a network-normalized latent.
    Isaac applies its configured scale after this conversion. Extra mobile-base
    channels stay zero. Coordinate frames and control frequency must still be
    compatible with the chosen environment; this is not benchmark equivalence.
    """
    action = np.asarray(value, dtype=np.float32)
    if action.shape != (7,):
        raise ValueError(f"expected policy action (7,), got {action.shape}")
    if not np.isfinite(action).all():
        raise FloatingPointError("policy returned a non-finite action")
    if action_dim < 7:
        raise ValueError("target action dimension cannot be smaller than 7")
    result = np.zeros(action_dim, dtype=np.float32)
    physical_limits = np.array([0.05] * 3 + [0.5] * 3, dtype=np.float32)
    result[:6] = np.clip(action[:6], -1, 1) * physical_limits / _scale_vector(controller_scale)
    # LIBERO: positive closes. Isaac BinaryJointPositionAction: negative closes.
    result[6] = -1.0 if action[6] >= 0.0 else 1.0
    return result

def _as_numpy(value: Any) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    return np.asarray(value)


def _env_value(value: Any, env_index: int) -> np.ndarray:
    array = _as_numpy(value)
    if array.ndim == 0:
        raise ValueError("expected an observation with an environment dimension")
    if env_index >= array.shape[0]:
        raise IndexError(f"environment {env_index} is outside observation shape {array.shape}")
    return array[env_index]


def _rgb_uint8(value: Any) -> np.ndarray:
    image = _as_numpy(value)
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(f"expected HWC RGB/RGBA image, got {image.shape}")
    image = image[..., :3]
    if not image.size or not np.isfinite(image).all():
        raise ValueError("camera image must be non-empty and finite")
    if image.dtype != np.uint8:
        image = np.asarray(image, dtype=np.float32)
        if image.size and float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.rint(np.clip(image, 0.0, 255.0)).astype(np.uint8)
    return np.ascontiguousarray(image)


def _quat_wxyz_to_axis_angle(value: Any) -> np.ndarray:
    quat = np.asarray(value, dtype=np.float64).copy()
    if quat.shape != (4,):
        raise ValueError(f"expected quaternion (4,), got {quat.shape}")
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("quaternion must be finite and non-zero")
    quat /= norm
    if quat[0] < 0.0:
        quat = -quat
    vector = quat[1:]
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm < 1e-8:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(quat[0], -1.0, 1.0))
    return np.asarray(vector * (angle / vector_norm), dtype=np.float32)


def build_policy_observation(
    observation: Mapping[str, Any],
    *,
    env_index: int,
    prompt: str,
    primary_camera: str,
    wrist_camera: str,
) -> dict[str, Any]:
    """Convert one Arena Franka environment observation to PI0.5 keys."""
    policy = observation.get("policy", observation)
    cameras = observation.get("camera_obs")
    if not isinstance(policy, Mapping):
        raise TypeError("Arena observation['policy'] must be a mapping")
    if not isinstance(cameras, Mapping):
        raise KeyError("Arena observation has no 'camera_obs' group; pass --enable_cameras")

    missing_state = [key for key in ("eef_pos", "eef_quat", "gripper_pos") if key not in policy]
    missing_cameras = [key for key in (primary_camera, wrist_camera) if key not in cameras]
    if missing_state:
        raise KeyError(f"Arena policy observation is missing {missing_state}")
    if missing_cameras:
        raise KeyError(
            f"Arena camera observation is missing {missing_cameras}; available={list(cameras)}"
        )

    eef_pos = np.asarray(_env_value(policy["eef_pos"], env_index), dtype=np.float32)
    eef_quat = _env_value(policy["eef_quat"], env_index)
    gripper = np.asarray(_env_value(policy["gripper_pos"], env_index), dtype=np.float32).reshape(-1)
    if eef_pos.shape != (3,) or gripper.shape != (2,):
        raise ValueError(
            f"expected eef_pos (3,) and gripper_pos (2,), got {eef_pos.shape} and {gripper.shape}"
        )
    state = np.concatenate((eef_pos, _quat_wxyz_to_axis_angle(eef_quat), gripper))
    if not np.isfinite(state).all():
        raise ValueError("robot state must be finite")
    return {
        POLICY_IMAGE_KEYS[0]: _rgb_uint8(_env_value(cameras[primary_camera], env_index)),
        POLICY_IMAGE_KEYS[1]: _rgb_uint8(_env_value(cameras[wrist_camera], env_index)),
        POLICY_STATE_KEY: np.ascontiguousarray(state, dtype=np.float32),
        POLICY_PROMPT_KEY: str(prompt),
    }


def validate_server_metadata(metadata: Mapping[str, Any]) -> None:
    action_dim = metadata.get("action_dim")
    if action_dim is not None and int(action_dim) != POLICY_ACTION_DIM:
        raise RuntimeError(f"server action_dim is {action_dim}, expected {POLICY_ACTION_DIM}")
    image_keys = metadata.get("image_keys")
    if image_keys is not None and tuple(image_keys) != POLICY_IMAGE_KEYS:
        raise RuntimeError(
            f"server image_keys are {image_keys}, expected {list(POLICY_IMAGE_KEYS)}"
        )
    state_key = metadata.get("state_key")
    if state_key is not None and state_key != POLICY_STATE_KEY:
        raise RuntimeError(f"server state_key is {state_key!r}, expected {POLICY_STATE_KEY!r}")



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

class ApxInfPolicy:
    """Batch-aware adapter; the environment owner remains responsible for step/reset.

    Requires relative pose IK as arm_action followed by a BinaryJointPosition
    gripper_action. Set camera names explicitly to match the environment.
    Call reset(terminated_env_ids) after environment auto-reset, or reset() after
    resetting every environment. Inference is sequential across environments.
    """

    def __init__(self, *, primary_camera: str, wrist_camera: str, prompt: str | None = None,
                 host: str = "127.0.0.1", port: int = 8000, replan_steps: int = 5,
                 action_dim: int = 7, client=None):
        if replan_steps <= 0 or action_dim < 7:
            raise ValueError("replan_steps must be positive and action_dim must be at least 7")
        self.prompt = prompt
        self.task_description = None
        self.primary_camera = primary_camera
        self.wrist_camera = wrist_camera
        self.replan_steps = replan_steps
        self.action_dim = action_dim
        self._plans = []
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

    def set_task_description(self, task_description: str | None) -> str:
        self.task_description = task_description
        self.reset()
        return self._prompt()

    def _prompt(self) -> str:
        prompt = self.prompt if self.prompt is not None else self.task_description
        if not prompt:
            raise ValueError("supply a policy prompt or an environment task description")
        return prompt

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            for plan in self._plans:
                plan.clear()
            return
        for index in np.asarray(_as_numpy(env_ids), dtype=np.int64).reshape(-1):
            if self._plans:
                self._plans[int(index)].clear()

    def get_action(self, env, observation: Mapping[str, Any]):
        import torch

        target = env.unwrapped
        scale = relative_ik_scale(env)
        if target.action_manager.total_action_dim != self.action_dim:
            raise ValueError("environment action dimension differs from configured action_dim")
        if list(target.action_manager.active_terms[:2]) != ["arm_action", "gripper_action"]:
            raise ValueError("requires arm_action then gripper_action in the target action layout")
        gripper_type = target.cfg.actions.gripper_action.class_type
        if gripper_type.__name__ != "BinaryJointPositionAction":
            raise ValueError("requires BinaryJointPositionAction for the gripper")
        policy_obs = observation.get("policy", observation)
        positions = _as_numpy(policy_obs["eef_pos"])
        if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 1:
            raise ValueError("eef_pos must have shape [num_envs, 3]")
        num_envs = positions.shape[0]
        if not self._plans:
            self._plans = [deque() for _ in range(num_envs)]
        if len(self._plans) != num_envs:
            raise ValueError("num_envs changed; create a new policy for the new environment")
        for index, plan in enumerate(self._plans):
            if plan:
                continue
            request = build_policy_observation(
                observation, env_index=index, prompt=self._prompt(),
                primary_camera=self.primary_camera, wrist_camera=self.wrist_camera,
            )
            response = self._client.infer(request)
            actions = np.asarray(response["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != 7:
                raise ValueError(f"expected actions [H,7], got {actions.shape}")
            if len(actions) < self.replan_steps:
                raise ValueError("server horizon is shorter than replan_steps")
            if not np.isfinite(actions).all():
                raise FloatingPointError("policy returned non-finite actions")
            plan.extend(libero_action_to_relative_ik(
                row, controller_scale=scale, action_dim=self.action_dim,
            ) for row in actions[:self.replan_steps])
        batch = np.stack([plan.popleft() for plan in self._plans])
        return torch.as_tensor(batch, device=target.device, dtype=torch.float32)

    def close(self) -> None:
        if self._owns_client:
            close_client(self._client)

    # PolicyBase's public protocol, without importing Arena before AppLauncher.
    @classmethod
    def from_dict(cls, config: dict):
        return cls(**config)

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group("ApxInf remote policy")
        group.add_argument("--apxinf-host", default="127.0.0.1")
        group.add_argument("--apxinf-port", type=int, default=8000)
        group.add_argument("--apxinf-prompt", default=None)
        group.add_argument("--apxinf-replan-steps", type=int, default=5)
        group.add_argument("--apxinf-primary-camera", required=True)
        group.add_argument("--apxinf-wrist-camera", required=True)
        group.add_argument("--apxinf-action-dim", type=int, default=7)
        group.add_argument("--remote_kill_on_exit", action="store_true",
                           help="Arena compatibility only; never kills the shared server")
        return parser

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        return cls(
            host=args.apxinf_host, port=args.apxinf_port, prompt=args.apxinf_prompt,
            replan_steps=args.apxinf_replan_steps, action_dim=args.apxinf_action_dim,
            primary_camera=args.apxinf_primary_camera, wrist_camera=args.apxinf_wrist_camera,
        )

    def has_length(self) -> bool:
        return False

    def length(self):
        return None

    @property
    def is_remote(self) -> bool:
        return True

    def shutdown_remote(self, kill_server: bool = False) -> None:
        # Arena owns the rollout, not the shared inference server's lifecycle.
        self.close()


def main() -> None:
    # Native Arena imports this module by dotted path only after AppLauncher.
    script = Path(__file__).resolve()
    sys.path.insert(0, str(script.parent))
    if "--policy_type" not in sys.argv:
        sys.argv[1:1] = ["--policy_type", f"{script.stem}.ApxInfPolicy"]
    from isaaclab_arena.evaluation.policy_runner import main as arena_main

    arena_main()


if __name__ == "__main__":
    main()
