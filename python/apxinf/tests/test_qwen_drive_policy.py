"""Planning policy contract, exercised without weights or a GPU."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

import apxinf
from apxinf.policies import AutoPolicy, QwenDrivePolicy
from apxinf.policies.impls import qwen_drive


class Tokenizer:
    def __init__(self, *args):
        pass

    def token_id(self, token):
        return {"<|im_start|>": 11, "<|im_end|>": 12}[token]

    def encode(self, text):
        return [8] if text == "\n" else [7]

    def decode(self, ids, **kwargs):
        return "slow down"


@pytest.fixture
def config():
    return {
        "model_type": "qwen_drive",
        "vlm_config": {"image_token_id": 20, "vision_start_token_id": 21,
                       "vision_end_token_id": 22, "text_config": {"eos_token_id": 12}},
        "image_patch_size": 2, "image_spatial_merge_size": 2, "image_temporal_patch_size": 2,
        "history_image_pixels": 64, "current_image_pixels": 64,
        "trajectory_scale": [2, 3, 1.5703125], "num_future_points": 3,
        "trajectory_point_dim": 3, "num_inference_steps": 10,
    }


class Runner:
    def __init__(self):
        self.calls = []

    def _infer_preprocessed(self, pixels, grids, tokens, attention_mask, state, embodiment_id, noise, **options):
        real = int(attention_mask.sum())
        assert real > 0 and len(attention_mask) == len(tokens)
        np.testing.assert_array_equal(attention_mask[:real], 1)
        np.testing.assert_array_equal(attention_mask[real:], 0)
        np.testing.assert_array_equal(tokens[real:], 0)
        assert embodiment_id is None
        self.calls.append((pixels, grids, tokens, state, noise, options))
        return np.ones((3, 3), np.float32)


def policy(config, mode="direct_planning", steps=4):
    return QwenDrivePolicy(Runner(), config=config, tokenizer=Tokenizer(), mode=mode,
                           eos_token_ids=[12, 13], seed=42, max_new_tokens=8,
                           min_new_tokens=2, num_steps=steps)


def scene():
    return {
        "views": {"<FRONT VIEW>": [np.zeros((8, 8, 3), np.uint8)]},
        "history": np.zeros((16, 3), np.float32),
        "history_velocity": np.ones((16, 2), np.float32),
        "history_acceleration": np.full((16, 2), 2, np.float32),
        "ego_velocity": [3, 4], "ego_acceleration": [5, 6], "driving_command": [1, 0, 0],
        "nav_command": 1,
    }


@pytest.mark.parametrize("mode", ["direct_planning", "reasoning_planning"])
def test_planning_preserves_noise_steps_and_conditioning(config, mode):
    p = policy(config, mode)
    noise = np.arange(9, dtype=np.float32).reshape(3, 3)
    result = p.infer(scene(), noise=noise)
    pixels, grids, tokens, state, sent_noise, options = p.model_runner.calls[0]
    assert pixels.shape == (16, 24)
    np.testing.assert_array_equal(grids, [[1, 4, 4]])
    assert tokens.dtype == np.uint32
    assert state.shape == (45 + 32 + 32 + 7 + 1,)
    np.testing.assert_array_equal(state[45:77], 1)
    np.testing.assert_array_equal(state[77:109], 2)
    np.testing.assert_array_equal(state[-8:], [3, 4, 5, 6, 1, 0, 0, 1])
    np.testing.assert_array_equal(sent_noise, noise)
    assert options["num_steps"] == 4
    assert result["metadata"]["num_inference_steps"] == 4
    assert result["actions"].shape == (config["num_future_points"], config["trajectory_point_dim"])
    np.testing.assert_allclose(result["actions"], np.tile(config["trajectory_scale"], (3, 1)), atol=3e-7)
    if mode == "reasoning_planning":
        assert options["closing_ids"] == [12, 8]
        assert options["terminator_ids"] == [12, 13]
        assert options["max_new_tokens"] == 8
        assert options["min_new_tokens"] == 2
        assert "reasoning" not in result
        assert "token_ids" not in result
    else:
        assert options == {"num_steps": 4}
        assert "reasoning" not in result


def test_autopolicy_loads_shared_runner_with_required_planner_asset(config, tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps(config))
    calls = []
    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return Runner()
    monkeypatch.setitem(apxinf.__dict__, "ModelRunner", SimpleNamespace(load=load))
    monkeypatch.setattr(qwen_drive, "_Tokenizer", Tokenizer)
    p = AutoPolicy.from_pretrained(tmp_path, planner=tmp_path / "rl", num_steps=6)
    assert isinstance(p, QwenDrivePolicy)
    assert calls[0][0] == ("qwen_drive", tmp_path)
    assert calls[0][1]["assets"] == {"planner": tmp_path / "rl"}
    assert calls[0][1]["model_variant"] == "bf16"
    p.infer(scene(), noise=np.zeros((3, 3), np.float32))
    assert p.model_runner.calls[0][-1]["num_steps"] == 6


def test_omitted_steps_fall_back_to_the_checkpoint_default(config, tmp_path, monkeypatch):
    # Without num_steps the config value has to reach the native call, not a
    # hardcoded default on either side of the binding.
    (tmp_path / "config.json").write_text(json.dumps(config))
    monkeypatch.setitem(apxinf.__dict__, "ModelRunner", SimpleNamespace(load=lambda *a, **k: Runner()))
    monkeypatch.setattr(qwen_drive, "_Tokenizer", Tokenizer)
    p = AutoPolicy.from_pretrained(tmp_path, planner=tmp_path / "rl")
    assert p.num_steps == config["num_inference_steps"]
    result = p.infer(scene(), noise=np.zeros((3, 3), np.float32))
    assert p.model_runner.calls[0][-1]["num_steps"] == config["num_inference_steps"]
    assert result["metadata"]["num_inference_steps"] == config["num_inference_steps"]


@pytest.mark.parametrize("mode,message", [
    ("vqa", "not exposed by the planning runtime"),
    ("perception", "declared pending gap"),
    ("text", "only direct/reasoning planning"),
])
def test_nonplanning_modes_fail_before_loading(mode, message):
    # vqa/perception carry their own reason so a caller can tell "deferred behind an
    # interface/kernel decision" from "this mode never existed".
    with pytest.raises(ValueError, match=message):
        QwenDrivePolicy.from_pretrained("missing-checkpoint", mode=mode)


def test_invalid_steps_and_noise_fail_before_execution(config):
    with pytest.raises(ValueError, match="num_steps"):
        policy(config, steps=0)
    p = policy(config)
    with pytest.raises(ValueError, match="noise"):
        p.infer(scene(), noise=np.zeros((9,), np.float32))
    with pytest.raises(ValueError, match="noise"):
        p.infer(scene(), noise=np.full((3, 3), np.nan, np.float32))
    assert not p.model_runner.calls


def test_native_planning_uses_shared_runner():
    native = pytest.importorskip("apxinf_py")
    assert hasattr(native.ModelRunner, "_infer_preprocessed")
    assert not hasattr(native.ModelRunner, "_infer_planning")
    assert not hasattr(native, "QwenDriveModel")


@pytest.mark.parametrize("pillow_version,owned", [("12.3.0", False), ("12.3.0", True), ("13.0.0", False)])
def test_rgb_bridge_and_pillow_gate(config, monkeypatch, pillow_version, owned):
    import PIL
    monkeypatch.setattr(PIL, "__version__", pillow_version)
    config = dict(config, image_patch_size=16, history_image_pixels=1024,
                  current_image_pixels=1024)
    p = policy(config)
    runner = p.model_runner
    runner._infer_resized_rgb = runner._infer_preprocessed
    if owned:
        runner._pack_rgb_u8_frames = lambda frames: ("owned", np.concatenate(frames).copy())
        runner._infer_resized_rgb_packed = runner._infer_preprocessed
    result = p.infer(scene(), noise=np.zeros((3, 3), np.float32))
    pixels, grids, _, _, _, options = runner.calls[0]
    assert result["actions"].shape == (3, 3)
    if pillow_version == "12.3.0":
        assert options["raw_resize_frames"].shape == (1, 6)
        np.testing.assert_array_equal(options["raw_resize_frames"][0, :2], [8, 8])
        if owned:
            assert pixels[0] == "owned"
            assert pixels[1].dtype == np.uint8
        else:
            assert pixels.dtype == np.uint8 and pixels.size == 8 * 8 * 3
    else:
        assert "raw_resize_frames" not in options
        assert pixels.dtype == np.uint8
        assert pixels.size == int(np.prod(grids[0, 1:])) * 16 * 16 * 3


@pytest.mark.parametrize("dimension", range(6))
def test_raw_resize_dimension_limits_use_pillow(config, monkeypatch, dimension):
    _check_raw_resize_fallback(config, monkeypatch, dimension=dimension)


@pytest.mark.parametrize("limit", ["frames", "raw_bytes", "final_bytes"])
def test_raw_resize_resource_limits_use_pillow(config, monkeypatch, limit):
    _check_raw_resize_fallback(config, monkeypatch, limit=limit)


def _check_raw_resize_fallback(config, monkeypatch, dimension=None, limit=None):
    import PIL
    monkeypatch.setattr(PIL, "__version__", "12.3.0")
    p = policy(dict(config, image_patch_size=16))
    p.model_runner._infer_resized_rgb = p.model_runner._infer_preprocessed
    sizes = [32] * 6  # source, stage and final width/height
    if dimension is not None:
        sizes[dimension] = 8193
    count = 1
    if limit == "frames":
        count = 65536
    elif limit == "raw_bytes":
        sizes[:2], count = [8192, 8192], 3
    elif limit == "final_bytes":
        sizes[4:], count = [8192, 8192], 2
    # Model only geometry, so oversized-resource cases allocate no large arrays.
    image = SimpleNamespace(ndim=3, shape=(sizes[1], sizes[0], 3),
                            reshape=lambda *args: np.zeros(1, np.uint8))
    contiguous = np.ascontiguousarray
    monkeypatch.setattr(np, "ascontiguousarray", lambda x, *a, **k:
                        image if x is image else contiguous(x, *a, **k))
    monkeypatch.setattr(p, "_scene_frames", lambda views:
                        [(image, tuple(sizes[2:4]), True)] * count)
    monkeypatch.setattr(qwen_drive, "smart_resize", lambda *a: (sizes[5], sizes[4]))
    fallback = []
    def resize(jobs):
        fallback.append(len(jobs))
        return [(np.zeros(32 * 32 * 3, np.uint8), (2, 2))]
    monkeypatch.setattr(p, "_resized_rgb_batch", resize)
    p.infer(scene(), noise=np.zeros((3, 3), np.float32))
    assert fallback == [count]
    assert "raw_resize_frames" not in p.model_runner.calls[0][-1]


def test_native_rgb_pack_rejects_invalid_inputs():
    native = pytest.importorskip("apxinf_py")
    pack = native.ModelRunner._pack_rgb_u8_frames
    for frames in ([], [np.zeros(3, np.float32)], [np.zeros((1, 3), np.uint8)],
                   [np.arange(6, dtype=np.uint8)[::2]], [np.zeros(0, np.uint8)],
                   [np.zeros(1, np.uint8)] * 65536):
        with pytest.raises(ValueError):
            pack(frames)


def test_production_action_shape(config, monkeypatch):
    p = policy(dict(config, num_future_points=50))
    monkeypatch.setattr(p.model_runner, "_infer_preprocessed",
                        lambda *args, **kwargs: np.ones((50, 3), np.float32))
    result = p.infer(scene(), noise=np.zeros((50, 3), np.float32))
    assert result["actions"].shape == (50, 3)
