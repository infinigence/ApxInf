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
        # (`apxinf-robo capture-libero`) and this engine-side calibrator: the
        # observations are an artifact on disk, not a live simulator.
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
