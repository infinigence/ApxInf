from __future__ import annotations

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import types
import unittest
from typing import Optional
from unittest import mock

import numpy as np


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.libero.client import parse_args  # noqa: E402
from evaluation.libero.contract import (  # noqa: E402
    LiberoWebsocketClient,
    validate_libero_server_metadata,
)
from evaluation.libero.ledger import (  # noqa: E402
    append_record,
    completed_runs,
    write_summary,
)
from evaluation.libero.rollout import run_episode  # noqa: E402


class ClientOnlyCliTest(unittest.TestCase):
    def test_cli_cannot_select_a_local_backend_or_model(self) -> None:
        """The evaluator must remain a remote client, not regain model ownership."""
        args = parse_args(
            [
                "--precision",
                "bf16",
                "--host",
                "inference.example",
                "--port",
                "9000",
                "--results-jsonl",
                "results.jsonl",
                "--summary-json",
                "summary.json",
            ]
        )
        self.assertEqual((args.host, args.port), ("inference.example", 9000))
        self.assertFalse(hasattr(args, "backend"))
        self.assertFalse(hasattr(args, "model_dir"))
        self.assertFalse(hasattr(args, "device"))

    def test_cli_rejects_checkpoint_loading_options(self) -> None:
        """A user cannot accidentally make the LIBERO process load a checkpoint."""
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--precision",
                        "bf16",
                        "--model-dir",
                        "/checkpoint",
                        "--results-jsonl",
                        "results.jsonl",
                        "--summary-json",
                        "summary.json",
                    ]
                )


class LiberoServerContractTest(unittest.TestCase):
    class FakeSocket:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    def _client_package(self, metadata: dict, response: Optional[dict] = None):
        test = self
        instances = []

        class FakePolicy:
            def __init__(self, host: str, port: int) -> None:
                self.endpoint = (host, port)
                self._ws = test.FakeSocket()
                self.last_observation = None
                instances.append(self)

            def get_server_metadata(self) -> dict:
                return metadata

            def infer(self, observation: dict) -> dict:
                self.last_observation = observation
                return response or {"actions": np.zeros((5, 7), dtype=np.float32)}

        package = types.ModuleType("openpi_client")
        package.websocket_client_policy = types.SimpleNamespace(
            WebsocketClientPolicy=FakePolicy
        )
        package.instances = instances
        return package

    def test_client_sends_only_the_openpi_libero_observation_contract(self) -> None:
        """Host selection and observations must reach the official client adapter."""
        metadata = {
            "precision": "bf16",
            "robot": "franka_libero",
            "action_dim": 7,
        }
        response = {
            "actions": np.zeros((5, 7), dtype=np.float32),
            "policy_timing": {"infer_ms": 2.0},
            "server_timing": {"infer_ms": 5.0},
        }
        with mock.patch.dict(
            sys.modules, {"openpi_client": self._client_package(metadata, response)}
        ):
            client = LiberoWebsocketClient("inference.example", 9000, "bf16")

        actions, timing = client.infer(
            np.zeros((2, 2, 3)), np.ones((2, 2, 3)), np.zeros(7), "pick"
        )
        self.assertEqual(client._client.endpoint, ("inference.example", 9000))
        self.assertEqual(
            set(client._client.last_observation),
            {
                "observation/image",
                "observation/wrist_image",
                "observation/state",
                "prompt",
            },
        )
        self.assertEqual(actions.shape, (5, 7))
        self.assertEqual(timing["model_seconds"], 0.002)
        self.assertEqual(timing["server_processor_seconds"], 0.003)
        client.close()
        self.assertTrue(client._client._ws.closed)

    def test_client_closes_connection_when_precision_is_wrong(self) -> None:
        """A mismatched server cannot remain connected and start an evaluation."""
        package = self._client_package({"precision": "fp8", "robot": "franka_libero"})
        with mock.patch.dict(sys.modules, {"openpi_client": package}):
            with self.assertRaisesRegex(RuntimeError, "expected 'bf16'"):
                LiberoWebsocketClient("localhost", 8000, "bf16")
        self.assertTrue(package.instances[0]._ws.closed)

    def test_non_libero_metadata_is_rejected_before_rollout(self) -> None:
        """Actions for another robot or width must never reach the LIBERO env."""
        for metadata, message in (
            ({"robot": "unitree_g1", "action_dim": 16}, "unitree_g1"),
            ({"robot": "franka_libero", "action_dim": 6}, "action_dim=6"),
        ):
            with self.subTest(metadata=metadata):
                with self.assertRaisesRegex(RuntimeError, message):
                    validate_libero_server_metadata(metadata)


class AccuracyRolloutTest(unittest.TestCase):
    class FakeEnv:
        def __init__(self, done_after: int = 11) -> None:
            self.steps = 0
            self.actions = []
            self.done_after = done_after
            self.observation = {
                "agentview_image": np.zeros((8, 8, 3), dtype=np.uint8),
                "robot0_eye_in_hand_image": np.zeros((8, 8, 3), dtype=np.uint8),
                "robot0_eef_pos": np.zeros(3, dtype=np.float32),
                "robot0_eef_quat": np.asarray([0, 0, 0, 1], dtype=np.float32),
                "robot0_gripper_qpos": np.zeros(1, dtype=np.float32),
            }

        def reset(self) -> None:
            self.steps = 0
            self.actions = []

        def set_init_state(self, _state):
            return self.observation

        def step(self, action):
            self.steps += 1
            self.actions.append(action)
            return self.observation, 0.0, self.steps >= self.done_after, {}

    class FakeClient:
        def __init__(self, actions: Optional[np.ndarray] = None) -> None:
            self.actions = (
                actions
                if actions is not None
                else np.zeros((5, 7), dtype=np.float32)
            )
            self.calls = 0

        def infer(self, _base, _wrist, _state, _prompt):
            self.calls += 1
            return self.actions, {}

    @staticmethod
    def _run(env, client):
        return run_episode(
            env,
            np.zeros(1),
            "libero_10",
            0,
            0,
            "pick up the object",
            client,
            "openpi_websocket",
            7,
        )

    def test_rollout_stops_immediately_after_success(self) -> None:
        """The remaining action chunk must not execute after LIBERO reports success."""
        env = self.FakeEnv(done_after=11)
        client = self.FakeClient()
        record = self._run(env, client)
        self.assertTrue(record["success"])
        self.assertEqual(record["action_steps"], 1)
        self.assertEqual(record["replans"], 1)
        self.assertEqual(client.calls, 1)
        self.assertEqual(env.steps, 11)  # ten settling steps plus one policy action

    def test_wrong_action_width_never_reaches_the_environment(self) -> None:
        """Malformed server output must fail before the first policy action."""
        env = self.FakeEnv(done_after=100)
        with self.assertRaisesRegex(ValueError, r"\(>= 5, 7\)"):
            self._run(env, self.FakeClient(np.zeros((5, 6), dtype=np.float32)))
        self.assertEqual(env.steps, 10)

    def test_non_finite_action_never_reaches_the_environment(self) -> None:
        """NaN/Inf actions must not be applied to the simulator."""
        env = self.FakeEnv(done_after=100)
        actions = np.zeros((5, 7), dtype=np.float32)
        actions[0, 0] = np.nan
        with self.assertRaises(FloatingPointError):
            self._run(env, self.FakeClient(actions))
        self.assertEqual(env.steps, 10)


class ResumableLedgerTest(unittest.TestCase):
    def test_technical_errors_do_not_mark_an_episode_complete(self) -> None:
        """A failed attempt must be retried instead of silently skipped on resume."""
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "results.jsonl"
            append_record(
                path,
                {
                    "status": "technical_error",
                    "suite": "libero_10",
                    "task_id": 0,
                    "trial_id": 0,
                    "precision": "bf16",
                },
            )
            append_record(
                path,
                {
                    "status": "completed",
                    "suite": "libero_10",
                    "task_id": 0,
                    "trial_id": 1,
                    "precision": "bf16",
                    "success": True,
                },
            )
            self.assertEqual(set(completed_runs(path, "bf16")), {("libero_10", 0, 1)})

    def test_resume_rejects_results_from_another_precision(self) -> None:
        """Accuracy from fp8 and bf16 must never be merged into one campaign."""
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "results.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "suite": "libero_10",
                        "task_id": 0,
                        "trial_id": 0,
                        "precision": "fp8",
                    }
                )
                + "\n"
            )
            with self.assertRaisesRegex(ValueError, "requested 'bf16'"):
                completed_runs(path, "bf16")

    def test_summary_excludes_missing_episodes_from_the_success_rate(self) -> None:
        """Partial runs must be reported missing, not counted as failures."""
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "summary.json"
            expected = {("libero_10", 0, trial) for trial in range(3)}
            ledger = {
                ("libero_10", 0, 0): {
                    "status": "completed",
                    "success": True,
                    "replans": 1,
                },
                ("libero_10", 0, 1): {
                    "status": "completed",
                    "success": False,
                    "replans": 1,
                },
            }
            write_summary(path, ledger, expected, "bf16", "openpi_websocket")
            summary = json.loads(path.read_text())
            self.assertEqual(summary["completed_runs"], 2)
            self.assertEqual(summary["success_rate"], 0.5)
            self.assertEqual(
                summary["missing_runs"],
                [{"suite": "libero_10", "task_id": 0, "trial_id": 2}],
            )
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
