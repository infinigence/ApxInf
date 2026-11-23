import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from apxinf.policies import AutoPolicy, Cosmos3EdgePolicy
from apxinf.policies.impls.cosmos3_edge import _FIXED_CONFIG
from apxinf.checkpoints.cosmos3_edge import _build_policy_prompt, _check_transformer_geometry


class Runner:
    action_horizon = 16
    action_dim = 10
    num_views = 1
    image_size = 544
    initial_latent_shape = [1, 376384]

    def __init__(self):
        self.calls = []
        self.output = np.zeros((16, 10), np.float32)
        self.closed = 0

    def infer_rgb(self, rgb, layout, tokens, noise, *, image_grid_thw):
        np.testing.assert_array_equal(image_grid_thw, [[1, 544, 736]])
        assert image_grid_thw.dtype == np.uint32
        self.calls.append((rgb.copy(), layout, tokens.copy(), noise))
        return self.output

    def close(self):
        self.closed += 1


@pytest.fixture
def observation():
    return {"camera": {"image": np.full((544, 736, 3), 127, np.uint8)},
            "prompt": "Pick up the banana and place it in the bowl."}


def policy(runner, observation):
    return Cosmos3EdgePolicy(runner, prompts=[observation["prompt"]],
                             image_key="camera/image")


@pytest.fixture
def prepared(tmp_path, observation):
    (tmp_path / "config.json").write_text(json.dumps(_FIXED_CONFIG))
    (tmp_path / "prompts.json").write_text(json.dumps({"prompts": [observation["prompt"]]}))
    for name in ("model.safetensors", "vae_encoder.safetensors", "text.safetensors"):
        (tmp_path / name).write_bytes(b"mock runner owns the tensor contract")
    hashes = {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
              for name in ("model.safetensors", "vae_encoder.safetensors",
                           "text.safetensors", "prompts.json")}
    (tmp_path / "conversion.json").write_text(json.dumps({"output_sha256": hashes}))
    return tmp_path


def test_public_rgb_and_complete_joint_noise(observation):
    runner = Runner()
    model = policy(runner, observation)
    supplied = np.random.RandomState(1).standard_normal((1, 376384)).astype(np.float32)
    result = model(observation, noise=supplied)
    rgb, layout, tokens, noise = runner.calls[0]
    assert rgb.shape == (1, 544, 736, 3) and rgb.dtype == np.uint8
    np.testing.assert_array_equal(rgb[0], observation["camera"]["image"])
    assert layout == "nhwc" and tokens.dtype == np.uint32
    np.testing.assert_array_equal(tokens, [0])
    np.testing.assert_array_equal(noise, supplied)
    np.testing.assert_array_equal(result["actions"], runner.output)
    assert result["actions"].dtype == np.float32
    assert set(result["timing"]) == {"model_ms", "total_ms"}
    # The native gripper inversion must not be repeated by the Python policy.
    assert result["actions"][0, -1] == 0


def test_generated_and_explicit_noise_precedence(observation):
    runner = Runner()
    model = policy(runner, observation)
    model(observation)
    assert runner.calls[0][3] is None
    observation["noise"] = np.zeros((1, 376384), np.float32)
    model(observation)
    explicit = np.ones((1, 376384), np.float32)
    model(observation, noise=explicit)
    np.testing.assert_array_equal(runner.calls[1][3], observation["noise"])
    np.testing.assert_array_equal(runner.calls[2][3], explicit)


@pytest.mark.parametrize("noise", [np.zeros((16, 10), np.float32),
                                    np.zeros((1, 375360), np.float32),
                                    np.full((1, 376384), np.inf, np.float32)])
def test_incomplete_or_nonfinite_noise_rejected(observation, noise):
    runner = Runner()
    with pytest.raises(ValueError, match="noise"):
        policy(runner, observation)(observation, noise=noise)
    assert not runner.calls


@pytest.mark.parametrize("image", [np.zeros((544, 736, 3), np.float32),
                                    np.zeros((736, 544, 3), np.uint8),
                                    np.zeros((17, 544, 736, 3), np.uint8)])
def test_noncanonical_rgb_rejected(observation, image):
    runner = Runner()
    observation["camera"]["image"] = image
    with pytest.raises(ValueError, match="canonical"):
        policy(runner, observation)(observation)
    assert not runner.calls


def test_uncached_prompt_rejected_before_native_execution(observation):
    runner = Runner()
    model = policy(runner, observation)
    observation["prompt"] = "move something else"
    with pytest.raises(ValueError, match="cached"):
        model(observation)
    assert not runner.calls


def test_retained_actions_and_idempotent_close(observation):
    runner = Runner()
    model = policy(runner, observation)
    first = model(observation)
    runner.output.fill(1)
    second = model(observation)
    np.testing.assert_array_equal(first["actions"], 0)
    np.testing.assert_array_equal(second["actions"], 1)
    model.close()
    model.close()
    assert runner.closed == 1
    with pytest.raises(RuntimeError, match="closed"):
        model(observation)


@pytest.mark.parametrize("output", [np.zeros((16, 8)), np.full((16, 10), np.nan)])
def test_invalid_native_actions_rejected(observation, output):
    runner = Runner()
    runner.output = output
    with pytest.raises(ValueError, match="invalid actions"):
        policy(runner, observation)(observation)


def test_auto_policy_and_lazy_native_loading(prepared, monkeypatch):
    runner = Runner()
    injected = AutoPolicy.from_pretrained(prepared, model_runner=runner)
    assert isinstance(injected, Cosmos3EdgePolicy) and injected.model_runner is runner
    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return runner

    monkeypatch.setitem(sys.modules, "apxinf_py", SimpleNamespace(
        ModelRunner=SimpleNamespace(load=load)))
    loaded = AutoPolicy.from_pretrained(prepared, image_key="camera", autotune=True)
    assert loaded.image_key == "camera"
    assert calls == [(("cosmos3-edge", str(prepared)), {"device": "cuda:0", "autotune": True})]
    with pytest.raises(ValueError, match="load options"):
        AutoPolicy.from_pretrained(prepared, model_runner=runner, autotune=True)


@pytest.mark.parametrize("key,value", [("num_tasks", 2), ("max_text_tokens", 127),
                                        ("pixel_frames", 17), ("num_inference_steps", 30),
                                        ("raw_action_dim", 7),
                                        ("source_revision", "0" * 40),
                                        ("checkpoint_revision", "0" * 40)])
def test_unsupported_prepared_geometry_rejected(prepared, key, value):
    config = json.loads((prepared / "config.json").read_text())
    config[key] = value
    (prepared / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match=key):
        AutoPolicy.from_pretrained(prepared, model_runner=Runner())


@pytest.mark.parametrize("name", ["prompts.json", "text.safetensors",
                                    "model.safetensors", "vae_encoder.safetensors"])
def test_corrupt_prepared_assets_rejected(prepared, name):
    if name == "prompts.json":
        (prepared / name).write_text(json.dumps({"prompts": ["another task"]}))
    else:
        (prepared / name).write_bytes(b"replaced")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        AutoPolicy.from_pretrained(prepared, model_runner=Runner())


def test_missing_or_unsafe_manifest_entries_rejected(prepared):
    path = prepared / "conversion.json"
    manifest = json.loads(path.read_text())
    manifest["output_sha256"].pop("text.safetensors")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="requires a SHA256 for text"):
        AutoPolicy.from_pretrained(prepared, model_runner=Runner())
    manifest["output_sha256"]["text.safetensors"] = "0" * 64
    manifest["output_sha256"]["../outside"] = "0" * 64
    # Remove the earlier mismatch so the path traversal entry is reached.
    manifest["output_sha256"]["text.safetensors"] = hashlib.sha256(
        (prepared / "text.safetensors").read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="invalid"):
        AutoPolicy.from_pretrained(prepared, model_runner=Runner())


def test_converter_prompt_framing_matches_droid_profile():
    framed = json.loads(_build_policy_prompt("Pick up the banana and place it in the bowl."))
    assert framed == {
        "cinematography": {"framing": "This video is captured from a first-person perspective looking at the scene."},
        "actions": [{"time": "0:00-0:03", "description": "Pick up the banana and place it in the bowl."}],
        "duration": "3s", "fps": 5.0, "resolution": {"H": 544, "W": 736},
        "aspect_ratio": "4,3"}


def test_converter_rejects_wrong_layer_geometry_without_loading_device():
    with pytest.raises(ValueError, match="28 transformer layers"):
        _check_transformer_geometry({"layers.0.self_attn.to_q.weight": SimpleNamespace(shape=(2048, 2048))})
    weights = {f"layers.{i}.self_attn.to_q.weight": SimpleNamespace(shape=(2048, 2048))
               for i in range(28)}
    weights["embed_tokens.weight"] = SimpleNamespace(shape=(131072, 1024))
    with pytest.raises(ValueError, match="embed_tokens.weight"):
        _check_transformer_geometry(weights)
