from __future__ import annotations

import json

import numpy as np
import pytest


class _FakeModel:
    action_horizon = 10
    action_dim = 26
    num_views = 2

    def __init__(self):
        self.call = None

    def _infer_patches(self, patches, token_ids, noise=None, action_mask=None):
        self.call = (patches, token_ids, noise, action_mask)
        return np.zeros((10, 26), dtype=np.float32)


class _FakeProcessor:
    image_keys = ("observation/image", "observation/wrist_image")
    state_key = "observation/state"
    prompt_key = "prompt"
    state_bins = 256

    def __call__(self, observation):
        return (
            np.zeros((648, 1176), dtype=np.float32),
            np.arange(313, dtype=np.uint32),
            np.ones((10, 26), dtype=np.float32),
        )


class _ArrayLike:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def numpy(self):
        return self.value


class _FakeImageProcessor:
    patch_size = 14
    merge_size = 2
    min_pixels = 56 * 56
    max_pixels = 14 * 14 * 4 * 1280

    def __call__(self, *, images, return_tensors):
        assert return_tensors == "pt"
        assert [image.shape for image in images] == [(252, 252, 3)] * 2
        return {
            "image_grid_thw": _ArrayLike([[1, 18, 18], [1, 18, 18]]),
            "pixel_values": _ArrayLike(np.zeros((648, 1176), np.float32)),
        }


class _FakeTokenizer:
    image_pad_token_id = 99

    def __init__(self):
        self.prompt = None

    def __call__(self, prompt, **kwargs):
        self.prompt = prompt
        # 141 text + 2 image placeholders + 10 action tokens. Expansion adds
        # 80 tokens per image, yielding the real 313-token fixture shape.
        return {"input_ids": [1] * 141 + [99, 99] + [2] * 10}


def test_registry_exports_walloss():
    from apxinf import WallossPolicy
    from apxinf.policies import available_policies, get_policy

    assert "walloss" in available_policies()
    assert "wall_oss_05" in available_policies()
    assert get_policy("walloss") is WallossPolicy


def test_autopolicy_detects_walloss_checkpoint_signature(tmp_path, monkeypatch):
    from apxinf import AutoPolicy, WallossPolicy

    document = {
        "model_type": "qwen2_5_vl",
        "architectures": ["Qwen2_5_VLForConditionalGeneration"],
        "experts": [{}, {}],
        "action_hidden_size": 1024,
        "noise_scheduler": {},
    }
    (tmp_path / "config.json").write_text(json.dumps(document))
    monkeypatch.setattr(
        WallossPolicy, "from_pretrained", classmethod(lambda cls, model_dir, **kwargs: (model_dir, kwargs))
    )
    model_dir, kwargs = AutoPolicy.from_pretrained(tmp_path, precision="bf16")
    assert model_dir == tmp_path
    assert kwargs == {"precision": "bf16"}


def test_caller_supplied_wire_keys_reach_the_published_metadata(tmp_path, monkeypatch):
    # The engine names no dataset's keys, so a deployment states its own. What is
    # worth checking here is that they survive the trip: the policy publishes the
    # keys it will actually read, and a narrowed action_dim narrows the map too.
    from apxinf import WallossPolicy
    from apxinf.policies.impls import walloss

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2_5_vl",
                "experts": [{}, {}],
                "action_hidden_size": 1024,
                "noise_scheduler": {},
            }
        )
    )
    processor = _FakeProcessor()
    processor.state_key = "observation/state"
    processor.prompt_key = "prompt"
    monkeypatch.setattr(
        walloss,
        "_load_normalizer",
        lambda path, norm_key: (
            np.full(26, -1.0, np.float32),
            np.full(26, 2.0, np.float32),
        ),
    )
    monkeypatch.setattr(walloss, "_WallossProcessor", lambda *args, **kwargs: processor)

    policy = WallossPolicy.from_pretrained(
        tmp_path,
        model=_FakeModel(),
        image_keys=("observation/image", "observation/wrist_image"),
        state_key="observation/state",
        action_dim=7,
    )

    assert policy.metadata["image_keys"] == [
        "observation/image",
        "observation/wrist_image",
    ]
    assert policy.metadata["state_key"] == "observation/state"
    assert policy.metadata["action_dim"] == 7


def test_walloss_rejects_disabling_checkpoint_state_encoding(tmp_path):
    from apxinf import WallossPolicy

    with pytest.raises(ValueError, match="require discretized state"):
        WallossPolicy.from_pretrained(
            tmp_path,
            model=_FakeModel(),
            discrete_state=False,
        )


def test_walloss_rejects_static_fp8_calibration(tmp_path):
    from apxinf import WallossPolicy

    with pytest.raises(TypeError, match="unsupported WallossPolicy options.*calibration"):
        WallossPolicy.from_pretrained(
            tmp_path,
            model=_FakeModel(),
            precision="fp8",
            calibration=tmp_path / "calibration.json",
        )


def test_walloss_action_width_uses_checkpoint_unless_user_overrides(tmp_path, monkeypatch):
    from apxinf import WallossPolicy
    from apxinf.policies.impls import walloss

    monkeypatch.setattr(walloss, "_WallossProcessor", lambda *args, **kwargs: _FakeProcessor())
    monkeypatch.setattr(
        walloss,
        "_load_normalizer",
        lambda path, norm_key: (
            np.full(26, -1.0, np.float32),
            np.full(26, 2.0, np.float32),
        ),
    )

    native = WallossPolicy.from_pretrained(tmp_path, model=_FakeModel())
    overridden = WallossPolicy.from_pretrained(tmp_path, model=_FakeModel(), action_dim=7)

    assert native.action_dim == 26
    assert overridden.action_dim == 7


def test_policy_calls_patch_contract_and_unnormalizes():
    from apxinf import WallossPolicy

    model = _FakeModel()
    minimum = np.arange(26, dtype=np.float32)
    delta = np.full(26, 2.0, dtype=np.float32)
    policy = WallossPolicy(
        model,
        _FakeProcessor(),
        action_min=minimum,
        action_delta=delta,
        action_dim=7,
    )
    noise = np.zeros((10, 26), dtype=np.float32)
    result = policy.infer({}, noise=noise)

    patches, token_ids, passed_noise, action_mask = model.call
    assert patches.shape == (648, 1176)
    assert token_ids.shape == (313,)
    np.testing.assert_array_equal(passed_noise, noise)
    assert action_mask.shape == (10, 26)
    np.testing.assert_array_equal(result["actions"][0], minimum[:7] + 1.0)
    assert result["actions"].shape == (10, 7)
    assert result["timing"]["total_ms"] >= result["timing"]["model_ms"]


@pytest.mark.parametrize(("state_bins", "midpoint"), [(256, 128), (512, 256)])
def test_processor_builds_fixed_walloss_contract(state_bins, midpoint):
    from apxinf.policies.impls.walloss import _WallossProcessor

    processor = _WallossProcessor.__new__(_WallossProcessor)
    processor.image_keys = ("observation/image", "observation/wrist_image")
    processor.camera_names = ("face_view", "right_wrist_view")
    processor.state_key = "observation/state"
    processor.prompt_key = "prompt"
    processor.action_horizon = 10
    processor.action_dim = 26
    processor.state_bins = state_bins
    processor.tokenizer = _FakeTokenizer()
    processor.image_processor = _FakeImageProcessor()
    processor.image_pad_token_id = 99
    processor.propri_min = np.full(26, -1.0, np.float32)
    processor.propri_delta = np.full(26, 2.0, np.float32)
    processor.factor = 28
    processor.min_pixels = 56 * 56
    processor.max_pixels = 14 * 14 * 4 * 1280

    image = np.zeros((256, 256, 3), dtype=np.uint8)
    patches, tokens, mask = processor(
        {
            "observation/image": image,
            "observation/wrist_image": image,
            "observation/state": np.zeros(7, np.float32),
            "prompt": "move",
        }
    )
    assert patches.shape == (648, 1176)
    assert tokens.shape == (313,)
    assert mask.shape == (10, 26)
    np.testing.assert_array_equal(mask[:, :7], 1.0)
    np.testing.assert_array_equal(mask[:, 7:], 0.0)
    assert "front view" in processor.tokenizer.prompt
    assert "right wrist view" in processor.tokenizer.prompt
    assert "Proprioception: " + " ".join([str(midpoint)] * 7) in processor.tokenizer.prompt


@pytest.mark.parametrize(("override", "expected"), [(None, 512), (1024, 1024)])
def test_walloss_state_bins_precedence(tmp_path, monkeypatch, override, expected):
    from apxinf import WallossPolicy
    from apxinf.policies.impls import walloss

    (tmp_path / "config.yml").write_text("data:\n  state_bins: 512\n")
    captured = {}

    def fake_processor(*args, **kwargs):
        captured.update(kwargs)
        processor = _FakeProcessor()
        processor.state_bins = kwargs["state_bins"]
        return processor

    monkeypatch.setattr(walloss, "_WallossProcessor", fake_processor)
    monkeypatch.setattr(
        walloss,
        "_load_normalizer",
        lambda path, norm_key: (
            np.full(26, -1.0, np.float32),
            np.full(26, 2.0, np.float32),
        ),
    )

    policy = WallossPolicy.from_pretrained(
        tmp_path,
        model=_FakeModel(),
        action_dim=7,
        state_bins=override,
    )

    assert captured["state_bins"] == expected
    assert policy.metadata["state_bins"] == expected
