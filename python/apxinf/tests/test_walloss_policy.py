from __future__ import annotations

import hashlib
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


class _FakeNativeModel(_FakeModel):
    accepts_rgb_u8 = True

    def infer_rgb(self, rgb_u8, layout, token_ids, noise=None, action_mask=None):
        self.call = (rgb_u8, layout, token_ids, noise, action_mask)
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


class _FakeNativeProcessor(_FakeProcessor):
    native_rgb = True

    def __call__(self, observation):
        return (
            np.zeros((2, 252, 252, 3), dtype=np.uint8),
            np.arange(313, dtype=np.uint32),
            np.ones((10, 26), dtype=np.float32),
        )


class _FakeImageProcessor:
    patch_size = 14
    merge_size = 2
    min_pixels = 56 * 56
    max_pixels = 14 * 14 * 4 * 1280

    def __call__(self, images):
        assert [image.shape for image in images] == [(256, 256, 3)] * 2
        return {
            "image_grid_thw": np.asarray([[1, 18, 18], [1, 18, 18]]),
            "pixel_values": np.zeros((648, 1176), np.float32),
        }


class _FakeTokenizer:
    image_pad_token_id = 99

    def __init__(self):
        self.prompt = None

    def encode(self, prompt):
        self.prompt = prompt
        # 141 text + 2 image placeholders + 10 action tokens. Expansion adds
        # 80 tokens per image, yielding the real 313-token fixture shape.
        return [1] * 141 + [99, 99] + [2] * 10


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
        WallossPolicy,
        "from_pretrained",
        classmethod(lambda cls, model_dir, **kwargs: (model_dir, kwargs)),
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

    with pytest.raises(
        TypeError, match="unsupported WallossPolicy options.*calibration"
    ):
        WallossPolicy.from_pretrained(
            tmp_path,
            model=_FakeModel(),
            precision="fp8",
            calibration=tmp_path / "calibration.json",
        )


def test_walloss_action_width_uses_checkpoint_unless_user_overrides(
    tmp_path, monkeypatch
):
    from apxinf import WallossPolicy
    from apxinf.policies.impls import walloss

    monkeypatch.setattr(
        walloss, "_WallossProcessor", lambda *args, **kwargs: _FakeProcessor()
    )
    monkeypatch.setattr(
        walloss,
        "_load_normalizer",
        lambda path, norm_key: (
            np.full(26, -1.0, np.float32),
            np.full(26, 2.0, np.float32),
        ),
    )

    native = WallossPolicy.from_pretrained(tmp_path, model=_FakeModel())
    overridden = WallossPolicy.from_pretrained(
        tmp_path, model=_FakeModel(), action_dim=7
    )

    assert native.action_dim == 26
    assert overridden.action_dim == 7


def test_walloss_from_pretrained_accepts_custom_python_processor(tmp_path, monkeypatch):
    from apxinf import WallossPolicy
    from apxinf.policies.impls import walloss

    processor = _FakeProcessor()
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
        processor=processor,
    )

    assert policy.processor is processor
    assert policy.metadata["state_bins"] == processor.state_bins


def test_builtin_processor_selects_native_rgb_from_model_capability(
    tmp_path, monkeypatch
):
    from apxinf import WallossPolicy
    from apxinf.policies.impls import walloss

    captured = {}

    def fake_processor(*args, **kwargs):
        captured.update(kwargs)
        processor = _FakeNativeProcessor()
        processor.native_rgb = kwargs["native_rgb"]
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

    policy = WallossPolicy.from_pretrained(tmp_path, model=_FakeNativeModel())

    assert captured["native_rgb"] is True
    assert policy.processor.native_rgb is True


def test_walloss_custom_processor_only_needs_to_be_callable(tmp_path, monkeypatch):
    from apxinf import WallossPolicy
    from apxinf.policies.impls import walloss

    def processor(observation):
        return _FakeProcessor()(observation)

    monkeypatch.setattr(walloss, "_checkpoint_state_bins", lambda model_dir: 512)
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
        processor=processor,
        image_keys=(),
        camera_names=(),
    )

    assert policy.processor is processor
    assert policy.metadata["image_keys"] == []
    assert policy.metadata["state_bins"] == 512


def test_custom_processor_cannot_implicitly_select_native_rgb(tmp_path, monkeypatch):
    from apxinf import WallossPolicy
    from apxinf.policies.impls import walloss

    monkeypatch.setattr(
        walloss,
        "_load_normalizer",
        lambda path, norm_key: (
            np.full(26, -1.0, np.float32),
            np.full(26, 2.0, np.float32),
        ),
    )
    model = _FakeNativeModel()
    policy = WallossPolicy.from_pretrained(
        tmp_path,
        model=model,
        processor=_FakeNativeProcessor(),
    )

    policy.infer({})

    assert len(model.call) == 4  # _infer_patches, not infer_rgb


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


def test_policy_calls_native_rgb_contract_when_processor_opts_in():
    from apxinf import WallossPolicy

    model = _FakeNativeModel()
    policy = WallossPolicy(
        model,
        _FakeNativeProcessor(),
        action_min=np.zeros(26, dtype=np.float32),
        action_delta=np.ones(26, dtype=np.float32),
        action_dim=7,
        native_rgb=True,
    )
    noise = np.zeros((10, 26), dtype=np.float32)
    policy.infer({}, noise=noise)

    rgb_u8, layout, token_ids, passed_noise, action_mask = model.call
    assert rgb_u8.shape == (2, 252, 252, 3)
    assert rgb_u8.dtype == np.uint8
    assert layout == "nhwc"
    assert token_ids.shape == (313,)
    np.testing.assert_array_equal(passed_noise, noise)
    assert action_mask.shape == (10, 26)


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
    assert (
        "Proprioception: " + " ".join([str(midpoint)] * 7) in processor.tokenizer.prompt
    )


def test_native_image_processor_matches_transformers_449_golden(tmp_path):
    from apxinf.policies.impls.walloss import _WallossImageProcessor

    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "image_processor_type": "Qwen2VLImageProcessor",
                "image_mean": [0.48145466, 0.4578275, 0.40821073],
                "image_std": [0.26862954, 0.26130258, 0.27577711],
                "min_pixels": 3136,
                "max_pixels": 12845056,
                "patch_size": 14,
                "temporal_patch_size": 2,
                "merge_size": 2,
            }
        )
    )
    processor = _WallossImageProcessor(tmp_path)
    image = (np.arange(252 * 252 * 3, dtype=np.uint32) % 256).astype(np.uint8)
    image = image.reshape(252, 252, 3)

    output = processor([image])
    patches = output["pixel_values"]

    assert patches.shape == (324, 1176)
    assert patches.dtype == np.float32
    np.testing.assert_array_equal(output["image_grid_thw"], [[1, 18, 18]])
    assert hashlib.sha256(patches.tobytes()).hexdigest() == (
        "c49b6d88be7086491c81cd45f80c362d4ec930e16768767d299154b250418b35"
    )

    resized = processor.resize([image, image])
    assert resized["rgb_u8"].shape == (2, 252, 252, 3)
    assert resized["rgb_u8"].dtype == np.uint8
    np.testing.assert_array_equal(resized["rgb_u8"][0], image)
    np.testing.assert_array_equal(resized["image_grid_thw"], [[1, 18, 18]] * 2)


def test_walloss_tokenizer_uses_tokenizer_json_and_adds_model_tokens(tmp_path):
    pytest.importorskip("apxinf_py")
    from apxinf.policies.impls.walloss import _WallossTokenizer

    (tmp_path / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "added_tokens": [],
                "normalizer": None,
                "pre_tokenizer": None,
                "post_processor": None,
                "decoder": None,
                "model": {
                    "type": "WordLevel",
                    "vocab": {"[UNK]": 0, "<|image_pad|>": 1},
                    "unk_token": "[UNK]",
                },
            }
        )
    )

    wrapped = _WallossTokenizer(tmp_path)

    assert wrapped.token_to_id("<|image_pad|>") == 1
    assert wrapped.token_to_id("<|propri|>") == 2
    assert wrapped.token_to_id("<|action|>") == 3
    assert wrapped.encode("<|action|>") == [3]


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
