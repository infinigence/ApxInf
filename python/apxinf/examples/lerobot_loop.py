#!/usr/bin/env python3
"""Drop an apxinf policy into a lerobot-shaped control loop.

Mirrors lerobot's own ``examples/tutorial/pi0/using_pi0_example.py`` with the
policy swapped for :class:`apxinf.adapters.lerobot.ApxInfPolicy`. Everything on
the robot side of the loop — ``robots`` / ``cameras`` / ``hw_to_dataset_features``
/ ``make_robot_action`` / ``send_action`` — is lerobot's and stays untouched;
only policy construction and the observation-frame call change.

By default this runs **without a robot**: ``--mock-robot`` synthesizes raw camera
frames and joint readings in the shape ``robot.get_observation()`` returns, so the
whole translation path is exercisable on any machine. Pass ``--robot-port`` to
drive a real SO-100 follower instead.

Requires ``apxinf[lerobot]`` (adds torch), the ``apxinf_py`` CUDA binding, a
checkpoint directory, and lerobot importable for its feature/action helpers.

    python examples/lerobot_loop.py --model-dir /path/to/checkpoint --mock-robot
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

from _common import synthetic_observation  # noqa: F401,E402 (path shim in _common)

from apxinf.adapters.lerobot import ApxInfPolicy

# lerobot's side of the loop — unchanged from the upstream example. ``lerobot`` is
# the user's own install and moves its helper modules between versions
# (``datasets.utils`` in 0.4.x, ``utils.feature_utils`` on later main), so resolve
# both and say so plainly rather than dumping an ImportError.
try:
    from lerobot.policies.utils import make_robot_action

    try:
        from lerobot.utils.feature_utils import hw_to_dataset_features
    except ImportError:
        from lerobot.datasets.utils import hw_to_dataset_features
except ImportError as exc:  # pragma: no cover - depends on the environment
    raise SystemExit(
        f"this example drives lerobot's own helpers ({exc.name} is missing).\n"
        f"Install lerobot in this environment, or see examples/pi05policy_infer.py "
        f"for the native apxinf API, which needs neither lerobot nor torch."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=pathlib.Path)
    parser.add_argument("--precision", choices=("auto", "fp8", "bf16", "int8"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--action-dim", type=int, default=7, help="deployable width (LIBERO=7)")
    parser.add_argument("--task", default="pick up the block")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--mock-robot",
        action="store_true",
        help="synthesize observations instead of connecting to hardware",
    )
    parser.add_argument("--robot-port", help="e.g. /dev/tty.usbmodem58760431631")
    parser.add_argument("--robot-id", default="follower_so100", help="selects the calibration file")
    return parser.parse_args()


class MockRobot:
    """Stand-in with the two surfaces this loop uses: features + observations.

    Returns what a real ``robot.get_observation()`` returns: a flat dict of raw
    ``HWC`` ``uint8`` camera frames keyed by camera name plus scalar joint
    readings — the form ``build_dataset_frame`` regroups.
    """

    def __init__(self, cameras, motors, height=480, width=640, seed=0):
        self._cameras = tuple(cameras)
        self._motors = tuple(motors)
        self._shape = (height, width, 3)
        self._rng = np.random.default_rng(seed)

    @property
    def observation_features(self):
        return {
            **{motor: float for motor in self._motors},
            **{camera: self._shape for camera in self._cameras},
        }

    @property
    def action_features(self):
        return {motor: float for motor in self._motors}

    def get_observation(self):
        return {
            **{motor: float(self._rng.standard_normal()) for motor in self._motors},
            **{
                camera: self._rng.integers(0, 256, size=self._shape, dtype=np.uint8)
                for camera in self._cameras
            },
        }

    def send_action(self, action):
        return action

    def connect(self):
        return None

    def disconnect(self):
        return None


def build_robot(args, cameras, motors):
    if args.mock_robot or not args.robot_port:
        return MockRobot(cameras, motors)

    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

    camera_config = {
        name: OpenCVCameraConfig(index_or_path=index, width=640, height=480, fps=30)
        for index, name in enumerate(cameras)
    }
    robot = SO100Follower(
        SO100FollowerConfig(port=args.robot_port, id=args.robot_id, cameras=camera_config)
    )
    robot.connect()
    return robot


def main() -> None:
    args = parse_args()

    # ── the only apxinf-specific lines ────────────────────────────────────
    model = ApxInfPolicy.from_pretrained(
        args.model_dir,
        device=args.device,
        precision=args.precision,
        action_dim=(args.action_dim or None),
    )
    preprocess, postprocess = model.make_pre_post_processors()
    print("policy:", model)
    print("metadata:", model.metadata)

    # Camera names must line up, in order, with the checkpoint's cameras. The
    # policy declares how many it wants; naming them is the integrator's job.
    cameras = ("base_0_rgb", "left_wrist_0_rgb")[: len(model.image_keys)]
    motors = tuple(f"joint_{index}.pos" for index in range(model.action_dim))

    robot = build_robot(args, cameras, motors)
    try:
        # ── lerobot's feature plumbing, unchanged ─────────────────────────
        dataset_features = {
            **hw_to_dataset_features(robot.action_features, "action"),
            **hw_to_dataset_features(robot.observation_features, "observation"),
        }

        for step in range(args.steps):
            obs = robot.get_observation()  # lerobot's

            # apxinf's frame builder: lerobot's build_dataset_frame + key rename,
            # stopping before the tensor/device conversion.
            frame = model.build_inference_frame(
                obs, ds_features=dataset_features, task=args.task
            )

            obs_in = preprocess(frame)  # unchanged
            action = model.select_action(obs_in)  # unchanged
            action = postprocess(action)  # unchanged
            action_dict = make_robot_action(action, dataset_features)  # lerobot's
            robot.send_action(action_dict)  # lerobot's

            print(f"step {step}: action={action.shape} first={float(action[0, 0]):+.4f}")

        # The chunk API skips per-step pre-processing when the loop can take one.
        chunk = model.predict_action_chunk(
            model.build_inference_frame(
                robot.get_observation(), ds_features=dataset_features, task=args.task
            )
        )
        print(f"chunk: {tuple(chunk.shape)}  (batch, horizon, action_dim)")
    finally:
        model.close()
        robot.disconnect()


if __name__ == "__main__":
    main()
