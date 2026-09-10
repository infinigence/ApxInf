"""Offline tests for the user-facing GR00T policy layer."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apxinf import AutoPolicy, Gr00tPolicy, Policy
from apxinf.policies import available_policies, get_policy


class _FakeModel:
    action_horizon = 4
    action_dim = 6

    def infer(self, *inputs):
        self.last_noise = np.asarray(inputs[-1]).copy()
        return np.arange(24, dtype=np.float32).reshape(4, 6) / 10


class _FakeProcessor:
    image_keys = ("observation/image", "observation/wrist_image")
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
    assert policy.action_horizon == 4
    assert result["metadata"]["model_action_horizon"] == 4


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
        def load(checkpoint, backbone, device, precision, calibration, tactics):
            assert precision == "int8"
            assert calibration is None
            assert tactics is None
            return _FakeModel()

    monkeypatch.setitem(
        __import__("sys").modules,
        "apxinf_py",
        SimpleNamespace(Gr00tModel=Native),
    )
    policy = Gr00tPolicy.from_pretrained(
        tmp_path, backbone=tmp_path, precision="int8", action_dim=3
    )
    assert policy.metadata["precision"] == "int8"


def test_fp8_requires_calibration(tmp_path):
    with pytest.raises(ValueError, match="requires calibration"):
        Gr00tPolicy.from_pretrained(tmp_path, backbone=tmp_path, precision="fp8")


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


@pytest.mark.skipif(
    not os.environ.get("APXINF_GR00T_CHECKPOINT")
    or not os.environ.get("APXINF_GR00T_BACKBONE"),
    reason="real GR00T processor paths are not configured",
)
def test_real_libero_processor_accepts_friendly_observation():
    from apxinf.policies.impls.gr00t import _NvidiaProcessorAdapter

    adapter = _NvidiaProcessorAdapter.load(
        Path(os.environ["APXINF_GR00T_CHECKPOINT"]),
        backbone=Path(os.environ["APXINF_GR00T_BACKBONE"]),
        embodiment="libero_sim",
        image_keys=("observation/image", "observation/wrist_image"),
        state_key="observation/state",
        prompt_key="prompt",
        action_key=None,
    )
    encoded = adapter.encode(
        {
            "observation/image": np.zeros((256, 256, 3), np.uint8),
            "observation/wrist_image": np.zeros((256, 256, 3), np.uint8),
            "observation/state": np.zeros(8, np.float32),
            "prompt": "put the moka pot on the stove",
        }
    )
    assert adapter.video_keys == ["image", "wrist_image"]
    assert adapter.action_keys == ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    assert adapter.state_dims == {
        "x": 1,
        "y": 1,
        "z": 1,
        "roll": 1,
        "pitch": 1,
        "yaw": 1,
        "gripper": 2,
    }
    assert adapter.action_dim == 7
    assert adapter.action_horizon == 16
    assert encoded["pixel_values"].ndim == 2
    assert encoded["image_grid_thw"].shape == (2, 3)
    assert encoded["token_ids"].ndim == 1
    assert encoded["attention_mask"].shape == encoded["token_ids"].shape
    assert encoded["state"].shape[-1] == 132


@pytest.mark.skipif(
    os.environ.get("APXINF_GR00T_NATIVE_SMOKE") != "1"
    or not os.environ.get("APXINF_GR00T_CHECKPOINT")
    or not os.environ.get("APXINF_GR00T_BACKBONE"),
    reason="real GR00T native smoke is not configured",
)
def test_real_libero_policy_runs_native_model_core_and_decode():
    precision = os.environ.get("APXINF_GR00T_PRECISION", "bf16")
    calibration = os.environ.get("APXINF_GR00T_CALIBRATION")
    if precision == "fp8" and not calibration:
        pytest.skip("APXINF_GR00T_CALIBRATION is required for FP8 native smoke")
    policy = Gr00tPolicy.from_pretrained(
        Path(os.environ["APXINF_GR00T_CHECKPOINT"]),
        backbone=Path(os.environ["APXINF_GR00T_BACKBONE"]),
        precision=precision,
        calibration=Path(calibration) if calibration else None,
        noise_mode="fixed",
    )
    try:
        observation = {
            "observation/image": np.zeros((256, 256, 3), np.uint8),
            "observation/wrist_image": np.zeros((256, 256, 3), np.uint8),
            "observation/state": np.zeros(8, np.float32),
            "prompt": "put the moka pot on the stove",
        }
        result = policy.infer(observation)
        assert result["normalized_actions"].shape == (40, 132)
        assert result["actions"].shape == (16, 7)
        assert np.isfinite(result["actions"]).all()

        replay = policy.infer(observation)
        np.testing.assert_array_equal(
            replay["normalized_actions"], result["normalized_actions"]
        )
        np.testing.assert_array_equal(replay["actions"], result["actions"])
    finally:
        policy.close()
