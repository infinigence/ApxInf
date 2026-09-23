import json
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np

from apxinf.calibration import CalibrationPlan
from scripts import calibrate_pi0fast, pi0fast_calibration_data


def fake_policy(*, state_width=8, sites=("multimodal_projector.input",)):
    """A policy stub exposing only the seam the calibration job consumes."""

    class Policy:
        image_keys = ("observation/image", "observation/wrist_image")
        prompt_key = "prompt"
        state_key = "observation/state"
        state_mean = np.zeros(state_width, np.float32)

        class _Model:
            image_size = 4

        model = _Model()

        def calibration_plan(self):
            return CalibrationPlan.runtime_validated_sites(
                model_family="pi0fast",
                sites=sites,
                schema=calibrate_pi0fast.SCHEMA,
                seed_algorithm="greedy-argmax-deterministic-v1",
            )

        def collect_calibration(self, observation, context):
            del observation, context
            return {site: 4.0 for site in sites}

        def close(self):
            pass

    return Policy()


class CalibratePi0FastTest(unittest.TestCase):
    def test_checkpoint_identity_matches_shared_cross_language_fixture(self):
        fixture = pathlib.Path(__file__).parent / "fixtures" / "checkpoint_identity"
        expected = (fixture / "expected.sha256").read_text().strip()

        self.assertEqual(calibrate_pi0fast.checkpoint_identity(fixture), expected)
        self.assertEqual(
            calibrate_pi0fast.checkpoint_identity(
                fixture / "model.safetensors.index.json"
            ),
            expected,
        )

    def test_state_width_is_a_checkpoint_property(self):
        # LIBERO reports two mirrored finger joints. π0-FAST's state statistics
        # are 8 wide, so the capture has to keep both; guessing 7 produced a
        # 7-value prompt and the policy refused it.
        self.assertEqual(
            calibrate_pi0fast.state_finger_joints(fake_policy(state_width=7)), 1
        )
        self.assertEqual(
            calibrate_pi0fast.state_finger_joints(fake_policy(state_width=8)), 2
        )
        with self.assertRaises(ValueError):
            calibrate_pi0fast.state_finger_joints(fake_policy(state_width=9))

    def test_libero_capture_keeps_the_checkpoint_state_width(self):
        class Suite:
            n_tasks = 1

            def get_task_init_states(self, _task_id):
                return np.zeros((1, 3), np.float32)

            def get_task(self, _task_id):
                return type("Task", (), {"language": "task 0"})()

        class Env:
            def reset(self):
                return self._observation()

            def set_init_state(self, _state):
                return self._observation()

            def step(self, _action):
                return self._observation(), 0.0, False, {}

            def _observation(self):
                return {
                    "agentview_image": np.zeros((3, 4, 3), np.uint8),
                    "robot0_eye_in_hand_image": np.zeros((3, 4, 3), np.uint8),
                    "robot0_eef_pos": np.arange(3, dtype=np.float32),
                    "robot0_eef_quat": np.asarray([0, 0, 0, 1], np.float32),
                    "robot0_gripper_qpos": np.asarray([0.1, 0.2], np.float32),
                }

            def close(self):
                pass

        with mock.patch.object(
            pi0fast_calibration_data, "_load_libero_suite", return_value=Suite()
        ), mock.patch.object(
            pi0fast_calibration_data, "make_env", side_effect=lambda *_: Env()
        ):
            observations = pi0fast_calibration_data.load_libero_observations(
                "libero_10",
                image_keys=("observation/image", "observation/wrist_image"),
                sample_count=None,
                seed=7,
                prompt_key="prompt",
                state_key="observation/state",
                finger_joints=2,
            )

        # 3 position + 3 axis-angle + 2 mirrored finger joints.
        self.assertEqual(observations[0]["observation/state"].shape, (8,))

    def test_requires_exactly_one_input_mode(self):
        args = calibrate_pi0fast.parse_args(["--model-dir", "."])
        with self.assertRaises(ValueError):
            calibrate_pi0fast.validate_args(args)

    def test_samples_is_rejected_without_a_libero_suite(self):
        args = calibrate_pi0fast.parse_args(
            ["--model-dir", ".", "--zero-fixture", "--samples", "4"]
        )
        with self.assertRaises(ValueError):
            calibrate_pi0fast.validate_args(args)

    def test_zero_fixture_builds_a_checkpoint_shaped_observation(self):
        args = calibrate_pi0fast.parse_args(
            ["--model-dir", ".", "--zero-fixture", "--source-revision", "test"]
        )
        observations, identity = calibrate_pi0fast.resolve_observations(
            args, fake_policy(state_width=8)
        )

        self.assertEqual(identity, "synthetic:zero-observation-v1")
        observation = observations[0]
        self.assertEqual(observation["observation/image"].shape, (4, 4, 3))
        self.assertEqual(observation["observation/state"].shape, (8,))

    def test_calibration_job_writes_a_validated_pi0fast_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "model.safetensors").write_bytes(b"weights")
            captured = root / "captured"
            captured.mkdir()
            for index in range(2):
                np.savez(
                    captured / f"sample-{index:03d}.npz",
                    **{
                        "observation/image": np.zeros((4, 4, 3), np.uint8),
                        "observation/wrist_image": np.zeros((4, 4, 3), np.uint8),
                        "observation/state": np.zeros(8, np.float32),
                        "prompt": np.asarray(f"task {index}"),
                    },
                )
            output = root / "profile.json"
            args = calibrate_pi0fast.parse_args(
                [
                    "--model-dir",
                    str(root),
                    "--input-dir",
                    str(captured),
                    "--output",
                    str(output),
                    "--source-revision",
                    "test-revision",
                ]
            )
            with mock.patch.object(calibrate_pi0fast, "_progress"):
                result = calibrate_pi0fast.run_from_args(
                    args, policy_factory=lambda *_a, **_k: fake_policy()
                )

            document = json.loads(output.read_text())

        self.assertEqual(result.output, output)
        self.assertEqual(document["schema"], "apxinf.pi0fast.fp8-calibration.v1")
        self.assertEqual(document["model"]["family"], "pi0fast")
        self.assertEqual(document["calibration_data"]["sample_count"], 2)
        self.assertEqual(document["calibration_data"]["kind"], "representative")
        self.assertEqual(document["seed_policy"]["algorithm"], "greedy-argmax-deterministic-v1")
        self.assertEqual(document["plan"]["sites"], ["multimodal_projector.input"])
        self.assertEqual(document["observed_sites"], ["multimodal_projector.input"])
        self.assertAlmostEqual(
            document["scales"]["multimodal_projector.input"]["scale"],
            4.0 * 1.1 / 448.0,
        )

    def test_rejects_overwrite_without_force(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "model.safetensors").write_bytes(b"weights")
            (root / "calibration.json").write_text("{}")
            args = calibrate_pi0fast.parse_args(
                ["--model-dir", str(root), "--zero-fixture"]
            )
            with self.assertRaises(ValueError):
                calibrate_pi0fast.validate_args(args)


if __name__ == "__main__":
    unittest.main()
