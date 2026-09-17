"""Offline tests for the user-facing GR00T policy layer."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from apxinf import AutoPolicy, Gr00tPolicy, Policy
from apxinf.calibration import CalibrationContext, CalibrationRunner, ConsumerContract
from apxinf.policies import available_policies, get_policy


class _FakeModel:
    action_horizon = 4
    action_dim = 6

    def _infer_preprocessed(self, *inputs):
        self.last_noise = np.asarray(inputs[-1]).copy()
        return np.arange(24, dtype=np.float32).reshape(4, 6) / 10

    def _calibration_plan(self):
        return ["backbone.text.layers.0.query.input", "action_head.output_projection.input"]

    def _calibrate_preprocessed(self, *inputs):
        self.last_calibration_noise = np.asarray(inputs[-1]).copy()
        return {
            "backbone.text.layers.0.query.input": 2.0,
            "action_head.output_projection.input": 3.0,
        }


class _FakeProcessor:
    image_keys = ("observation/image", "observation/wrist_image")
    state_key = "observation/state"
    prompt_key = "prompt"
    state_dim = 6
    action_horizon = 4
    action_dim = 3

    def encode(self, observation):
        return {
            "pixel_values": np.zeros((8, 3), np.float32),
            "image_grid_thw": np.asarray([[1, 2, 2], [1, 2, 2]], np.uint32),
            "token_ids": np.arange(5, dtype=np.uint32),
            "attention_mask": np.ones(5, np.uint8),
            "state": np.zeros((1, 1, 6), np.float32),
            "embodiment_id": 24,
            "raw_states": {"state": np.zeros((1, 6), np.float32)},
        }

    def decode(self, normalized, encoded):
        assert encoded["embodiment_id"] == 24
        return normalized[:, :3] + 1


def test_registry_exposes_gr00t_aliases():
    assert "gr00t" in available_policies()
    assert "gr00tn1d7" in available_policies()
    assert get_policy("Gr00tN1d7") is Gr00tPolicy


def test_policy_contract_and_raw_observation_call():
    policy = Gr00tPolicy(
        _FakeModel(), processor=_FakeProcessor(), seed=7, noise_mode="fixed", action_dim=3
    )
    result = policy.infer(
        {
            "observation/image": np.zeros((32, 32, 3), np.uint8),
            "observation/wrist_image": np.zeros((32, 32, 3), np.uint8),
            "observation/state": np.zeros(6, np.float32),
            "prompt": "pick up the object",
        }
    )
    assert isinstance(policy, Policy)
    assert result["actions"].shape == (4, 3)
    assert result["normalized_actions"].shape == (4, 6)
    assert result["noise"].shape == (1, 4, 6)
    assert result["actions"].dtype == np.float32
    assert result["metadata"]["model_type"] == "gr00t"
    assert result["metadata"]["num_views"] == 2
    assert result["metadata"]["image_keys"] == [
        "observation/image",
        "observation/wrist_image",
    ]
    assert result["metadata"]["state_key"] == "observation/state"
    assert result["metadata"]["prompt_key"] == "prompt"
    assert result["metadata"]["state_dim"] == 6
    assert policy.action_horizon == 4
    assert result["metadata"]["model_action_horizon"] == 4


def test_policy_metadata_tracks_custom_processor_input_contract():
    processor = _FakeProcessor()
    processor.image_keys = ("front", "wrist")
    processor.state_key = "robot/state"
    processor.prompt_key = "instruction"
    processor.state_dim = 8

    policy = Gr00tPolicy(_FakeModel(), processor=processor, action_dim=3)

    assert policy.metadata["num_views"] == 2
    assert policy.metadata["image_keys"] == ["front", "wrist"]
    assert policy.metadata["state_key"] == "robot/state"
    assert policy.metadata["prompt_key"] == "instruction"
    assert policy.metadata["state_dim"] == 8


def test_fixed_noise_repeats_and_stream_noise_advances():
    observation = {"prompt": "test"}
    fixed = Gr00tPolicy(
        _FakeModel(), processor=_FakeProcessor(), seed=3, noise_mode="fixed", action_dim=3
    )
    assert np.array_equal(fixed(observation)["noise"], fixed(observation)["noise"])
    stream = Gr00tPolicy(
        _FakeModel(), processor=_FakeProcessor(), seed=3, noise_mode="stream", action_dim=3
    )
    assert not np.array_equal(stream(observation)["noise"], stream(observation)["noise"])


def test_explicit_noise_matches_shared_policy_contract():
    policy = Gr00tPolicy(
        _FakeModel(), processor=_FakeProcessor(), seed=3, noise_mode="stream", action_dim=3
    )
    noise = np.full((1, 4, 6), 0.25, np.float32)
    assert np.array_equal(policy.infer({"prompt": "test"}, noise=noise)["noise"], noise)
    unbatched = noise[0]
    assert np.array_equal(
        policy.infer({"prompt": "test"}, noise=unbatched)["noise"], noise
    )

    with pytest.raises(ValueError, match="noise has shape"):
        policy.infer({"prompt": "test"}, noise=np.zeros((2, 4, 6), np.float32))
    bad = noise.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        policy.infer({"prompt": "test"}, noise=bad)


def test_invalid_noise_mode_fails_early():
    with pytest.raises(ValueError, match="noise_mode"):
        Gr00tPolicy(_FakeModel(), processor=_FakeProcessor(), noise_mode="bad")


def test_int8_precision_name_is_accepted(tmp_path, monkeypatch):
    from apxinf.policies.impls.gr00t import _NvidiaProcessorAdapter

    monkeypatch.setattr(_NvidiaProcessorAdapter, "load", lambda *args, **kwargs: _FakeProcessor())

    class Native:
        @staticmethod
        def load(model_name, checkpoint, **kwargs):
            assert model_name == "gr00t"
            assert kwargs["precision"] == "int8"
            assert kwargs["assets"] == {"backbone": str(tmp_path)}
            assert "calibration" not in kwargs
            assert "tactics" not in kwargs
            return _FakeModel()

    monkeypatch.setitem(
        __import__("sys").modules,
        "apxinf_py",
        SimpleNamespace(Model=Native),
    )
    policy = Gr00tPolicy.from_pretrained(
        tmp_path, backbone=tmp_path, precision="int8", action_dim=3
    )
    assert policy.metadata["precision"] == "int8"


def test_fp8_requires_calibration(tmp_path):
    with pytest.raises(ValueError, match="requires calibration"):
        Gr00tPolicy.from_pretrained(tmp_path, backbone=tmp_path, precision="fp8")


def test_calibration_uses_common_manifest_consumer_contract():
    policy = Gr00tPolicy(
        _FakeModel(), processor=_FakeProcessor(), seed=3, noise_mode="stream", action_dim=3
    )
    plan = policy.calibration_plan()
    assert plan.model_family == "gr00t"
    assert plan.consumer_contract is ConsumerContract.MANIFEST
    assert plan.consumers == {
        "backbone.text.layers.0.query": "backbone.text.layers.0.query.input",
        "action_head.output_projection": "action_head.output_projection.input",
    }
    assert set(plan.sites) == set(plan.consumers.values())


def test_calibration_collects_processor_tensors_with_context_seed():
    model = _FakeModel()
    policy = Gr00tPolicy(model, processor=_FakeProcessor(), action_dim=3)
    observation = {"prompt": "test"}
    first = policy.collect_calibration(observation, CalibrationContext(seed=7, sample_index=2))
    first_noise = model.last_calibration_noise.copy()
    second = policy.collect_calibration(observation, CalibrationContext(seed=7, sample_index=2))
    assert first == second
    np.testing.assert_array_equal(first_noise, model.last_calibration_noise)
    assert model.last_calibration_noise.shape == (1, 4, 6)


def test_public_observation_builds_common_calibration_manifest():
    policy = Gr00tPolicy(_FakeModel(), processor=_FakeProcessor(), action_dim=3)
    runner = CalibrationRunner(
        policy,
        policy.calibration_plan(),
        checkpoint="sha256:test-checkpoint",
        data_identity="sha256:test-data",
        source_revision="test-revision",
        device={"requested": "cuda:0", "host": "test-host"},
        margin=1.1,
        seed=11,
    )
    document = runner.run([{"prompt": "test"}])
    assert document is not None
    assert document["schema"] == "apxinf.fp8-calibration.v1"
    assert document["model"] == {
        "family": "gr00t",
        "checkpoint": "sha256:test-checkpoint",
    }
    assert document["plan"]["consumers"] == {
        "backbone.text.layers.0.query": "backbone.text.layers.0.query.input",
        "action_head.output_projection": "action_head.output_projection.input",
    }
    assert set(document["scales"]) == set(document["plan"]["sites"])


def test_checkpoint_identity_covers_primary_and_backbone(tmp_path):
    primary = tmp_path / "primary"
    backbone = tmp_path / "backbone"
    primary.mkdir()
    backbone.mkdir()
    (primary / "model.safetensors").write_bytes(b"primary-v1")
    (backbone / "model.safetensors").write_bytes(b"backbone-v1")

    identity = Gr00tPolicy.checkpoint_identity(primary, backbone)
    assert identity.startswith("sha256:")
    assert len(identity) == len("sha256:") + 64
    assert identity == Gr00tPolicy.checkpoint_identity(primary, backbone)

    (backbone / "model.safetensors").write_bytes(b"backbone-v2")
    assert identity != Gr00tPolicy.checkpoint_identity(primary, backbone)


def test_w8a8_is_not_a_public_precision_name(tmp_path):
    with pytest.raises(ValueError, match="precision must be"):
        Gr00tPolicy.from_pretrained(tmp_path, backbone=tmp_path, precision="w8a8")


def test_user_noise_is_reported_exactly_as_consumed():
    model = _FakeModel()
    policy = Gr00tPolicy(
        model, processor=_FakeProcessor(), noise_mode="fixed", action_dim=3
    )
    noise = np.linspace(-1.0, 1.0, model.action_horizon * model.action_dim, dtype=np.float32)
    noise = noise.reshape(model.action_horizon, model.action_dim)

    result = policy.infer({"prompt": "test"}, noise=noise)

    assert result["noise"].shape == (1, model.action_horizon, model.action_dim)
    np.testing.assert_array_equal(result["noise"], model.last_noise)


def test_processor_adapter_splits_flat_state_by_checkpoint_dimensions():
    from apxinf.policies.impls.gr00t import _NvidiaProcessorAdapter

    adapter = object.__new__(_NvidiaProcessorAdapter)
    adapter.state_keys = ["x", "gripper"]
    adapter.state_dims = {"x": 1, "gripper": 2}
    states = adapter._states(np.asarray([1.0, 2.0, 3.0], np.float32))
    assert states["x"].shape == (1, 1)
    assert states["gripper"].shape == (1, 2)
    assert np.array_equal(states["gripper"], [[2.0, 3.0]])

    with pytest.raises(ValueError, match="vector of length 3"):
        adapter._states(np.zeros(2, np.float32))


def test_autopolicy_dispatches_gr00t_config(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text('{"model_type":"Gr00tN1d7"}')
    sentinel = object()

    def fake_from_pretrained(cls, model_dir, **kwargs):
        assert model_dir == tmp_path
        assert kwargs == {"backbone": "/models/backbone", "precision": "bf16"}
        return sentinel

    monkeypatch.setattr(Gr00tPolicy, "from_pretrained", classmethod(fake_from_pretrained))
    assert AutoPolicy.from_pretrained(
        tmp_path, backbone="/models/backbone", precision="bf16"
    ) is sentinel
