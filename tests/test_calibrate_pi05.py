import json
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from apxinf.calibration import CalibrationPlan
from scripts import calibrate_pi05, pi05_calibration_data


class CalibratePi05Test(unittest.TestCase):
    def test_checkpoint_identity_matches_shared_cross_language_fixture(self):
        fixture = pathlib.Path(__file__).parent / "fixtures" / "checkpoint_identity"
        expected = (fixture / "expected.sha256").read_text().strip()

        self.assertEqual(calibrate_pi05.checkpoint_identity(fixture), expected)
        self.assertEqual(
            calibrate_pi05.checkpoint_identity(
                fixture / "model.safetensors.index.json"
            ),
            expected,
        )

    def test_task_stratified_indices_cover_tasks_before_repeating(self):
        indices = calibrate_pi05.task_stratified_indices(
            [0, 0, 0, 1, 1, 2], sample_count=5, seed=7
        )

        self.assertEqual(len(indices), 5)
        selected_tasks = [
            [0, 0, 0, 1, 1, 2][index]
            for index in indices
        ]
        self.assertEqual(set(selected_tasks[:3]), {0, 1, 2})
        self.assertLessEqual(max(selected_tasks.count(task) for task in set(selected_tasks)), 2)

    def test_libero_mode_captures_task_balanced_native_observations(self):
        class Task:
            def __init__(self, task_id):
                self.language = f"task {task_id}"

        class Suite:
            n_tasks = 2

            def get_task(self, task_id):
                return Task(task_id)

            def get_task_init_states(self, task_id):
                return np.asarray([[task_id, 0], [task_id, 1]], dtype=np.float32)

        class Env:
            def reset(self):
                pass

            def set_init_state(self, initial_state):
                self.value = int(initial_state[0] * 10 + initial_state[1])
                return self._observation()

            def step(self, _action):
                return self._observation(), 0.0, False, {}

            def _observation(self):
                return {
                    "agentview_image": np.full((3, 4, 3), self.value, np.uint8),
                    "robot0_eye_in_hand_image": np.full(
                        (3, 4, 3), self.value + 20, np.uint8
                    ),
                    "robot0_eef_pos": np.arange(3, dtype=np.float32),
                    "robot0_eef_quat": np.asarray([0, 0, 0, 1], np.float32),
                    "robot0_gripper_qpos": np.asarray([0.1, 0.2], np.float32),
                }

            def close(self):
                pass

        with mock.patch.object(
            pi05_calibration_data, "_load_libero_suite", return_value=Suite()
        ), mock.patch.object(pi05_calibration_data, "make_env", side_effect=lambda *_: Env()):
            observations = pi05_calibration_data.load_libero_observations(
                "libero_10",
                image_keys=("observation/image", "observation/wrist_image"),
                sample_count=2,
                seed=7,
                prompt_key="prompt",
                state_key="observation/state",
            )

        self.assertEqual(len(observations), 2)
        self.assertEqual(
            {observation["prompt"] for observation in observations},
            {"task 0", "task 1"},
        )
        # Frames come out at the simulator's own resolution: `libero_images` only
        # reorients, and the selected policy owns the resize to the model's edge.
        # State is 3 position + 3 axis-angle + 1 gripper. Both of these asserted
        # (224, 224, 3) and (8,) until aa82c96 moved the resize into the policy and
        # collapsed the mirrored gripper joints, and left the expectations behind;
        # tests/test_libero_observation.py pins the same widths as literals.
        self.assertEqual(observations[0]["observation/image"].shape, (3, 4, 3))
        self.assertEqual(observations[0]["observation/wrist_image"].shape, (3, 4, 3))
        self.assertEqual(observations[0]["observation/state"].shape, (7,))
        self.assertEqual(observations[0]["observation/state"].dtype, np.float32)

    def test_input_directory_expands_npz_files_in_stable_order(self):
        class Policy:
            image_keys = ("observation/image",)
            prompt_key = "prompt"
            state_key = "observation/state"
            discrete_state = False

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for name, prompt in (("sample-010.npz", "ten"), ("sample-002.npz", "two")):
                np.savez(
                    root / name,
                    **{
                        "observation/image": np.zeros((2, 2, 3), np.uint8),
                        "prompt": np.asarray(prompt),
                    },
                )
            args = calibrate_pi05.parse_args(
                ["--model-dir", directory, "--input-dir", directory]
            )
            observations, identity = calibrate_pi05.resolve_observations(args, Policy())

        self.assertEqual([item["prompt"] for item in observations], ["two", "ten"])
        self.assertTrue(identity.startswith("sha256:"))

    def test_manifest_loads_model_native_observations_without_external_dataset(self):
        class Policy:
            image_keys = ("observation/image", "observation/wrist_image")
            prompt_key = "prompt"
            state_key = "observation/state"

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            Image.fromarray(np.full((3, 4, 3), 11, np.uint8)).save(root / "base.png")
            Image.fromarray(np.full((3, 4, 3), 22, np.uint8)).save(root / "wrist.png")
            manifest = root / "observations.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "observation/image": "base.png",
                        "observation/wrist_image": "wrist.png",
                        "observation/state": [1, 2, 3],
                        "prompt": "pick up the block",
                    }
                )
                + "\n"
            )
            args = calibrate_pi05.parse_args(
                ["--model-dir", directory, "--manifest", str(manifest)]
            )
            observations, identity = calibrate_pi05.resolve_observations(args, Policy())

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]["prompt"], "pick up the block")
        self.assertEqual(observations[0]["observation/image"].dtype, np.uint8)
        self.assertEqual(observations[0]["observation/state"].dtype, np.float32)
        self.assertTrue(identity.startswith("sha256:"))

    def test_calibration_job_consumes_a_captured_npz_directory_end_to_end(self):
        # The NPZ directory is the seam between whoever owns the environment
        # and this engine-side calibrator: the observations are an artifact on
        # disk, not a live simulator.
        class Model:
            image_size = 2
            action_horizon = 2
            action_dim = 3

        class Policy:
            model = Model()
            image_keys = ("observation/image", "observation/wrist_image")
            prompt_key = "prompt"
            state_key = "observation/state"
            discrete_state = False

            def calibration_plan(self):
                return CalibrationPlan.runtime_validated_sites(
                    model_family="pi05",
                    sites=("vision.patch_input",),
                    schema=calibrate_pi05.SCHEMA,
                    seed_algorithm="numpy-pcg64-seed-sequence-v1",
                )

            def collect_calibration(self, observation, context):
                return {"vision.patch_input": float(context.sample_index + 1)}

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "model.safetensors").write_bytes(b"weights")
            captured = root / "captured"
            captured.mkdir()
            for index in range(2):
                np.savez(
                    captured / f"sample-{index:03d}.npz",
                    **{
                        "observation/image": np.zeros((2, 2, 3), np.uint8),
                        "observation/wrist_image": np.zeros((2, 2, 3), np.uint8),
                        "observation/state": np.zeros(8, np.float32),
                        "prompt": np.asarray(f"task {index}"),
                    },
                )
            output = root / "profile.json"
            args = calibrate_pi05.parse_args(
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
            with mock.patch.object(calibrate_pi05, "_progress") as progress:
                result = calibrate_pi05.run_from_args(
                    args, policy_factory=lambda *_args, **_kwargs: Policy()
                )

            document = json.loads(output.read_text())
            progress_messages = [call.args[0] for call in progress.call_args_list]

        self.assertEqual(result.output, output)
        self.assertEqual(document["calibration_data"]["sample_count"], 2)
        self.assertTrue(document["calibration_data"]["identity"].startswith("sha256:"))
        self.assertEqual(document["scales"]["vision.patch_input"]["amax"], 2.0)
        self.assertIn(
            "Hashing the checkpoint for profile identity (this reads all weight files)...",
            progress_messages,
        )
        self.assertIn(
            "Running eager BF16 calibration over 2 observation(s)...",
            progress_messages,
        )
        self.assertEqual(progress_messages[-1], "Calibration profile written.")

    def test_calibration_job_consumes_native_libero_source_end_to_end(self):
        # The other half of the pair above: --libero-suite drives the simulator in
        # this process, so ApxInf can recalibrate against its own published LIBERO
        # protocol without a downstream checkout. Both must stay wired up.
        class Model:
            image_size = 2
            action_horizon = 2
            action_dim = 3

        class Policy:
            model = Model()
            image_keys = ("observation/image", "observation/wrist_image")
            prompt_key = "prompt"
            state_key = "observation/state"
            discrete_state = False

            def calibration_plan(self):
                return CalibrationPlan.runtime_validated_sites(
                    model_family="pi05",
                    sites=("vision.patch_input",),
                    schema=calibrate_pi05.SCHEMA,
                    seed_algorithm="numpy-pcg64-seed-sequence-v1",
                )

            def collect_calibration(self, observation, context):
                return {"vision.patch_input": float(context.sample_index + 1)}

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "model.safetensors").write_bytes(b"weights")
            output = root / "profile.json"
            args = calibrate_pi05.parse_args(
                [
                    "--model-dir",
                    str(root),
                    "--libero-suite",
                    "libero_10",
                    "--output",
                    str(output),
                    "--source-revision",
                    "test-revision",
                ]
            )
            observations = tuple(
                {
                    "observation/image": np.zeros((2, 2, 3), np.uint8),
                    "observation/wrist_image": np.zeros((2, 2, 3), np.uint8),
                    "observation/state": np.zeros(8, np.float32),
                    "prompt": f"task {index}",
                }
                for index in range(2)
            )
            with mock.patch.object(
                calibrate_pi05,
                "load_libero_observations",
                return_value=observations,
            ) as capture, mock.patch.object(calibrate_pi05, "_progress"):
                result = calibrate_pi05.run_from_args(
                    args, policy_factory=lambda *_args, **_kwargs: Policy()
                )

            document = json.loads(output.read_text())

        self.assertEqual(result.output, output)
        self.assertEqual(document["calibration_data"]["sample_count"], 2)
        self.assertTrue(document["calibration_data"]["identity"].startswith("sha256:"))
        # The capture is asked for the *policy's* keys, not a restated dialect, so
        # calibration observations arrive under the names inference will read.
        self.assertEqual(capture.call_args.args, ("libero_10",))
        self.assertEqual(
            capture.call_args.kwargs["image_keys"],
            ("observation/image", "observation/wrist_image"),
        )
        self.assertEqual(capture.call_args.kwargs["state_key"], "observation/state")

    def test_samples_is_rejected_without_a_libero_suite(self):
        # --samples only means something for the sampler that reads it; accepting
        # it against an NPZ directory would silently ignore the operator's intent.
        args = calibrate_pi05.parse_args(
            ["--model-dir", ".", "--zero-fixture", "--samples", "4"]
        )
        with self.assertRaises(ValueError) as raised:
            calibrate_pi05.validate_args(args)
        self.assertIn("--samples applies only to --libero-suite", str(raised.exception))

    def test_calibration_job_accepts_observation_iterable_without_source_adapter(self):
        class Policy:
            def calibration_plan(self):
                return CalibrationPlan.runtime_validated_sites(
                    model_family="pi05",
                    sites=("vision.patch_input",),
                    schema=calibrate_pi05.SCHEMA,
                    seed_algorithm="numpy-pcg64-seed-sequence-v1",
                )

            def collect_calibration(self, observation, context):
                return {"vision.patch_input": float(observation["amax"])}

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            checkpoint = root / "model.safetensors"
            checkpoint.write_bytes(b"weights")
            output = root / "profile.json"
            args = calibrate_pi05.parse_args(
                [
                    "--model-dir",
                    str(root),
                    "--output",
                    str(output),
                    "--zero-fixture",
                    "--source-revision",
                    "test-revision",
                ]
            )
            observations = ({"amax": value} for value in (2.0, 7.0))
            result = calibrate_pi05.Pi05CalibrationJob(
                args,
                policy=Policy(),
                output=output,
                checkpoint=checkpoint,
            ).run(
                observations,
                data_identity="dataset:custom-observation-source",
            )

        self.assertEqual(result.document["calibration_data"]["sample_count"], 2)
        self.assertEqual(result.document["scales"]["vision.patch_input"]["amax"], 7.0)

    def test_calibration_job_maps_observation_to_manifest(self):
        import apxinf

        class Model:
            image_size = 4
            action_horizon = 2
            action_dim = 3

            @staticmethod
            def _calibration_plan():
                return ["vision.patch_input"]

        class Policy:
            model = Model()
            image_keys = ("observation/image",)
            prompt_key = "prompt"
            state_key = "observation/state"
            discrete_state = False

            def calibration_plan(self):
                return CalibrationPlan.runtime_validated_sites(
                    model_family="pi05",
                    sites=("vision.patch_input",),
                    schema=calibrate_pi05.SCHEMA,
                    seed_algorithm="numpy-pcg64-seed-sequence-v1",
                )

            def collect_calibration(self, observation, context):
                self.last_observation = observation
                self.last_noise = calibrate_pi05.deterministic_noise(
                    self, context.seed, context.sample_index
                )
                return {"vision.patch_input": 4.0}

            def close(self):
                pass

        policy = Policy()
        captured_options = {}

        class Pi05Policy:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                captured_options.update(kwargs)
                return policy

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            checkpoint = root / "model.safetensors"
            checkpoint.write_bytes(b"weights")
            sample = root / "sample.npz"
            np.savez(
                sample,
                **{
                    "observation/image": np.zeros((4, 4, 3), np.uint8),
                    "prompt": np.asarray("move"),
                },
            )
            output = root / "profile.json"
            with mock.patch.object(apxinf, "Pi05Policy", Pi05Policy):
                calibrate_pi05.main(
                    [
                        "--model-dir",
                        str(root),
                        "--input",
                        str(sample),
                        "--output",
                        str(output),
                        "--data-id",
                        "dataset:test-v1",
                        "--image-key",
                        "observation/image",
                        "--num-views",
                        "1",
                        "--prompt-key",
                        "prompt",
                        "--action-horizon",
                        "2",
                        "--seed",
                        "9",
                    ]
                )

            document = json.loads(output.read_text())
            self.assertEqual(document["calibration_data"]["identity"], "dataset:test-v1")
            self.assertEqual(document["observed_sites"], ["vision.patch_input"])
            self.assertEqual(document["plan"], {"sites": ["vision.patch_input"]})
            self.assertEqual(
                document["seed_policy"]["algorithm"],
                "numpy-pcg64-seed-sequence-v1",
            )
            self.assertEqual(policy.last_observation["prompt"], "move")
            self.assertEqual(policy.last_noise.shape, (2, 3))
            self.assertEqual(captured_options["image_keys"], ("observation/image",))
            self.assertEqual(captured_options["num_views"], 1)
            self.assertEqual(captured_options["action_horizon"], 2)

    def test_requires_exactly_one_input_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "model.safetensors").write_bytes(b"weights")
            for extra in ([], ["--zero-fixture", "--input", str(root / "x.npz")]):
                args = calibrate_pi05.parse_args(["--model-dir", str(root), *extra])
                with self.assertRaisesRegex(ValueError, "--input"):
                    calibrate_pi05.validate_args(args)

    def test_representative_input_rejects_synthetic_data_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "model.safetensors").write_bytes(b"weights")
            sample = root / "sample.npz"
            np.savez(sample, prompt=np.asarray("move"))
            args = calibrate_pi05.parse_args(
                [
                    "--model-dir",
                    str(root),
                    "--input",
                    str(sample),
                    "--data-id",
                    "synthetic:zero-observation-v1",
                ]
            )
            with self.assertRaisesRegex(ValueError, "cannot use a synthetic"):
                calibrate_pi05.validate_args(args)

    def test_rejects_overwrite_without_force(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / "model.safetensors").write_bytes(b"weights")
            (root / "calibration.json").write_text("existing")
            args = calibrate_pi05.parse_args(
                ["--model-dir", str(root), "--zero-fixture"]
            )
            with self.assertRaisesRegex(ValueError, "--force"):
                calibrate_pi05.validate_args(args)

    def test_write_profile_exclusively_protects_against_overwrite_races(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "calibration.json"
            calibrate_pi05.write_profile(output, {"version": 1}, force=False)
            with self.assertRaisesRegex(ValueError, "--force"):
                calibrate_pi05.write_profile(output, {"version": 2}, force=False)
            self.assertEqual(output.read_text(), '{\n  "version": 1\n}\n')

    def test_loads_business_observation_without_preprocessed_tensors(self):
        class Policy:
            image_keys = ("observation/image", "observation/wrist_image")
            prompt_key = "prompt"
            state_key = "observation/state"
            discrete_state = False

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "sample.npz"
            np.savez(
                path,
                **{
                    "observation/image": np.zeros((4, 5, 3), np.uint8),
                    "observation/wrist_image": np.ones((3, 2, 3), np.uint8),
                    "prompt": np.asarray("move the block"),
                },
            )
            args = calibrate_pi05.parse_args(
                ["--model-dir", directory, "--input", str(path)]
            )
            observation = next(iter(calibrate_pi05.load_observations(args, Policy())))
            self.assertEqual(observation["prompt"], "move the block")
            self.assertNotIn("rgb", observation)
            self.assertNotIn("token_ids", observation)

    def test_deterministic_noise_is_stable_per_sample(self):
        class Model:
            action_horizon = 2
            action_dim = 3

        class Policy:
            model = Model()

        first = calibrate_pi05.deterministic_noise(Policy(), 17, 2)
        second = calibrate_pi05.deterministic_noise(Policy(), 17, 2)
        other = calibrate_pi05.deterministic_noise(Policy(), 17, 3)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, other))

    def test_source_revision_fails_closed_outside_git(self):
        with mock.patch.object(
            calibrate_pi05.subprocess,
            "check_output",
            side_effect=OSError("git unavailable"),
        ):
            with self.assertRaisesRegex(ValueError, "--source-revision"):
                calibrate_pi05.source_revision()
        with self.assertRaisesRegex(ValueError, "real commit"):
            calibrate_pi05.source_revision("unknown")
        self.assertEqual(calibrate_pi05.source_revision("release-1.2.3"), "release-1.2.3")

    def test_bootstrap_is_unambiguously_non_production(self):
        class Policy:
            def calibration_plan(self):
                return CalibrationPlan.runtime_validated_sites(
                    model_family="pi05",
                    sites=("vision.patch_input", "action.input"),
                    schema=calibrate_pi05.SCHEMA,
                    seed_algorithm="numpy-pcg64-seed-sequence-v1",
                )

            def collect_calibration(self, observation, context):
                return {"vision.patch_input": 0.0, "action.input": 0.0}

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            checkpoint = root / "model.safetensors"
            checkpoint.write_bytes(b"weights")
            output = root / "calibration.json"
            args = calibrate_pi05.parse_args(
                [
                    "--model-dir",
                    str(root),
                    "--zero-fixture",
                    "--margin",
                    "2.35",
                    "--source-revision",
                    "test-revision",
                ]
            )
            document = calibrate_pi05.Pi05CalibrationJob(
                args,
                policy=Policy(),
                output=output,
                checkpoint=checkpoint,
            ).run(
                ({"fixture": True},),
                data_identity="synthetic:zero-observation-v1",
                bootstrap=True,
            ).document

        self.assertFalse(document["calibration_data"]["production"])
        self.assertEqual(document["calibration_data"]["kind"], "synthetic-zero-fixture")
        self.assertAlmostEqual(document["scales"]["vision.patch_input"]["amax"], 1 / 2.35)
        self.assertAlmostEqual(document["scales"]["action.input"]["amax"], 5 / 2.35)
        self.assertAlmostEqual(document["scales"]["vision.patch_input"]["scale"], 1 / 448)
        self.assertAlmostEqual(document["scales"]["action.input"]["scale"], 5 / 448)


if __name__ == "__main__":
    unittest.main()
