"""Exercise the evaluator CLI-to-checkpoint path without CUDA or a simulator."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from apxinf import AutoPolicy
from apxinf.checkpoints import detect_checkpoint
from scripts import eval_libero


def parse(monkeypatch, tmp_path, *extra, backend="in-process"):
    monkeypatch.setattr(sys, "argv", [
        "eval_libero.py", "--backend", backend, "--precision", "bf16",
        "--model-dir", str(tmp_path),
        "--results-jsonl", str(tmp_path / "results.jsonl"),
        "--summary-json", str(tmp_path / "summary.json"), *extra,
    ])
    return eval_libero.parse_args()


def test_explicit_norm_stats_reaches_checkpoint_loader(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"type": "pi05"}))
    stats = tmp_path / "external-norms.json"
    stats.write_text(json.dumps({"norm_stats": {
        "state": {"q01": [0.0] * 7, "q99": [1.0] * 7},
        "actions": {"q01": [2.0] * 7, "q99": [4.0] * 7},
    }}))
    loaded = []

    def load(model_dir, **options):
        loaded.append(detect_checkpoint(model_dir, norm_stats=options["norm_stats"]))
        return SimpleNamespace(metadata={})

    monkeypatch.setattr(AutoPolicy, "from_pretrained", load)
    args = parse(monkeypatch, tmp_path, "--norm-stats", str(stats))
    eval_libero.InProcessBackend(args, eval_libero.resolve_wire_keys(args))
    assert loaded[0].norm_stats == stats
    assert loaded[0].normalization.action.values["q01"] == (2.0,) * 7


def test_omitted_norm_stats_preserves_checkpoint_defaults(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"type": "pi05"}))
    options = {}

    def load(model_dir, **kwargs):
        options.update(kwargs)
        return SimpleNamespace(metadata={})

    monkeypatch.setattr(AutoPolicy, "from_pretrained", load)
    args = parse(monkeypatch, tmp_path)
    eval_libero.InProcessBackend(args, eval_libero.resolve_wire_keys(args))
    assert "norm_stats" not in options
    assert options["compute_variant"] == "bf16"
    assert "precision" not in options


def test_gr00t_backbone_and_two_joint_state_reach_policy(monkeypatch, tmp_path):
    options = {}

    def load(model_dir, **kwargs):
        options.update(kwargs)
        return SimpleNamespace(metadata={"model_type": "gr00t"})

    monkeypatch.setattr(AutoPolicy, "from_pretrained", load)
    backbone = tmp_path / "backbone"
    args = parse(monkeypatch, tmp_path, "--model-type", "gr00t", "--backbone", str(backbone))
    backend = eval_libero.InProcessBackend(args, eval_libero.resolve_wire_keys(args))
    state = backend.state_from_observation(
        {
            "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.array([0.04, -0.04]),
        }
    )

    assert options["backbone"] == backbone
    np.testing.assert_array_equal(
        state["gripper"], np.array([0.04, -0.04], dtype=np.float32)
    )


def test_gr00t_decoded_gripper_is_adapted_before_libero_step(monkeypatch, tmp_path):
    class FakePolicy:
        metadata = {"model_type": "gr00t"}

        def infer(self, observation, *, noise=None):
            actions = np.zeros((2, 7), dtype=np.float32)
            actions[:, -1] = [0.0, 1.0]
            return {
                "actions": actions,
                "normalized_actions": np.zeros((2, 132), dtype=np.float32),
                "timing": {},
            }

    monkeypatch.setattr(AutoPolicy, "from_pretrained", lambda *args, **kwargs: FakePolicy())
    args = parse(monkeypatch, tmp_path, "--model-type", "gr00t")
    backend = eval_libero.InProcessBackend(args, eval_libero.resolve_wire_keys(args))

    actions, _, _ = backend.infer(None, None, None, "prompt")

    np.testing.assert_array_equal(actions[:, -1], np.array([1.0, -1.0], dtype=np.float32))


@pytest.mark.parametrize("model_type", ["pi05", "walloss"])
def test_existing_in_process_models_keep_openpi_state_actions_and_seed(
    monkeypatch, tmp_path, model_type
):
    loaded = {}

    class FakePolicy:
        metadata = {"model_type": model_type}

        def infer(self, observation, *, noise=None):
            loaded["observation"] = observation
            actions = np.zeros((5, 7), dtype=np.float32)
            actions[:, -1] = np.linspace(-1.0, 1.0, 5, dtype=np.float32)
            return {
                "actions": actions,
                "normalized_actions": np.zeros((5, 7), dtype=np.float32),
                "timing": {},
            }

    def load(model_dir, **kwargs):
        loaded["model_dir"] = model_dir
        loaded["options"] = kwargs
        return FakePolicy()

    monkeypatch.setattr(AutoPolicy, "from_pretrained", load)
    checkpoint = tmp_path / "weights.safetensors"
    args = parse(
        monkeypatch,
        tmp_path,
        "--model-type",
        model_type,
        "--checkpoint",
        str(checkpoint),
        "--model-seed",
        "23",
    )
    keys = eval_libero.resolve_wire_keys(args)
    backend = eval_libero.InProcessBackend(args, keys)
    observation = {
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.array([0.04, -0.04]),
    }

    state = backend.state_from_observation(observation)
    actions, _, _ = backend.infer("base", "wrist", state, "task")

    np.testing.assert_array_equal(
        state,
        np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.04], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        actions[:, -1], np.linspace(-1.0, 1.0, 5, dtype=np.float32)
    )
    assert loaded["model_dir"] == tmp_path
    assert loaded["options"]["checkpoint"] == checkpoint
    assert loaded["options"]["seed"] == 23
    assert "backbone" not in loaded["options"]
    assert loaded["observation"] == {
        "observation/image": "base",
        "observation/wrist_image": "wrist",
        "observation/state": state,
        "prompt": "task",
    }


def test_websocket_keeps_openpi_state_and_action_contract(monkeypatch, tmp_path):
    requests = []

    class FakeClient:
        def __init__(self, host, port):
            assert (host, port) == ("127.0.0.1", 8000)

        def get_server_metadata(self):
            return {
                "precision": "bf16",
                "image_keys": ["observation/image", "observation/wrist_image"],
                "state_key": "observation/state",
            }

        def infer(self, observation):
            requests.append(observation)
            return {"actions": np.array([[0, 0, 0, 0, 0, 0, 0.25]], np.float32)}

    client_module = SimpleNamespace(WebsocketClientPolicy=FakeClient)
    monkeypatch.setitem(
        sys.modules,
        "openpi_client",
        SimpleNamespace(websocket_client_policy=client_module),
    )
    args = parse(monkeypatch, tmp_path, backend="websocket")
    backend = eval_libero.WebsocketBackend(
        args.host,
        args.port,
        args.precision,
        eval_libero.resolve_wire_keys(args),
    )
    observation = {
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.array([0.04, -0.04]),
    }

    state = backend.state_from_observation(observation)
    actions, normalized, _ = backend.infer("base", "wrist", state, "task")

    assert state.shape == (7,)
    assert actions[0, -1] == np.float32(0.25)
    assert normalized is None
    assert requests[0]["observation/state"] is state


def test_websocket_norm_stats_is_rejected(monkeypatch, tmp_path, capsys):
    stats = tmp_path / "norm_stats.json"
    stats.write_text("{}")
    with pytest.raises(SystemExit) as exc:
        parse(monkeypatch, tmp_path, "--norm-stats", str(stats), backend="websocket")
    assert exc.value.code == 2
    assert "pass it to pi05_openpi_websocket_server.py" in capsys.readouterr().err


def test_missing_norm_stats_is_rejected_before_rollout(monkeypatch, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        parse(monkeypatch, tmp_path, "--norm-stats", str(tmp_path / "missing.json"))
    assert exc.value.code == 2
    assert "--norm-stats must name an existing file" in capsys.readouterr().err


def test_rollout_protocol_defaults_remain_openpi_compatible(monkeypatch, tmp_path):
    args = parse(monkeypatch, tmp_path)

    assert args.max_steps == 520
    assert args.replan_steps == 5


def test_rollout_protocol_can_select_official_gr00t_values(monkeypatch, tmp_path):
    args = parse(
        monkeypatch,
        tmp_path,
        "--max-steps",
        "720",
        "--replan-steps",
        "8",
    )

    assert args.max_steps == 720
    assert args.replan_steps == 8


def test_completed_runs_rejects_mixed_rollout_protocol(tmp_path):
    ledger = tmp_path / "results.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "status": "completed",
                "precision": "bf16",
                "suite": "libero_10",
                "task_id": 0,
                "trial_id": 0,
                "max_steps": 520,
                "replan_steps": 5,
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="ledger max_steps is 520, requested 720"):
        eval_libero.completed_runs(
            ledger,
            "bf16",
            max_steps=720,
            replan_steps=8,
        )


def test_completed_runs_accepts_legacy_ledger_for_openpi_defaults(tmp_path):
    ledger = tmp_path / "results.jsonl"
    legacy_record = {
        "status": "completed",
        "precision": "bf16",
        "suite": "libero_10",
        "task_id": 0,
        "trial_id": 0,
    }
    ledger.write_text(json.dumps(legacy_record) + "\n")

    actual = eval_libero.completed_runs(
        ledger,
        "bf16",
        max_steps=520,
        replan_steps=5,
    )

    assert actual == {("libero_10", 0, 0): legacy_record}


def test_completed_runs_rejects_legacy_ledger_for_gr00t_protocol(tmp_path):
    ledger = tmp_path / "results.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "status": "completed",
                "precision": "bf16",
                "suite": "libero_10",
                "task_id": 0,
                "trial_id": 0,
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="ledger max_steps is 520, requested 720"):
        eval_libero.completed_runs(
            ledger,
            "bf16",
            max_steps=720,
            replan_steps=8,
        )


def test_summary_records_rollout_protocol(tmp_path):
    summary = tmp_path / "summary.json"
    expected = {("libero_10", 0, 0)}

    eval_libero.write_summary(
        summary,
        {},
        expected,
        "bf16",
        "in_process_api",
        max_steps=720,
        replan_steps=8,
    )

    assert json.loads(summary.read_text())["rollout_protocol"] == {
        "max_steps": 720,
        "replan_steps": 8,
        "wait_steps": 10,
    }


def test_run_episode_obeys_explicit_max_steps(monkeypatch):
    observation = {
        "agentview_image": np.zeros((2, 2, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((2, 2, 3), dtype=np.uint8),
    }

    class FakeEnv:
        def reset(self):
            return None

        def set_init_state(self, initial_state):
            return observation

        def step(self, action):
            return observation, 0.0, False, {}

    class FakeBackend:
        def state_from_observation(self, value):
            return np.zeros(8, dtype=np.float32)

        def infer(self, base, wrist, state, prompt, noise=None):
            return (
                np.zeros((8, 7), dtype=np.float32),
                np.zeros((8, 7), dtype=np.float32),
                {},
            )

    monkeypatch.setattr(
        eval_libero,
        "libero_images",
        lambda base, wrist: (base, wrist),
    )
    record = eval_libero.run_episode(
        FakeEnv(),
        np.zeros(1),
        "libero_10",
        0,
        0,
        "task",
        FakeBackend(),
        "in_process_api",
        7,
        False,
        0.5,
        replan_steps=8,
        max_steps=3,
    )

    assert record["action_steps"] == 3
    assert record["replans"] == 1
    assert record["max_steps"] == 3
    assert record["replan_steps"] == 8
