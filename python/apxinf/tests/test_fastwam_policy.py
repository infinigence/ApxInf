import json
import os
from pathlib import Path

import numpy as np
import pytest
from apxinf import FastWamPolicy
from apxinf.policies.registry import get_policy


def test_native_umt5_tokenizer_matches_official_whitespace_and_special_tokens():
    """Golden IDs from the pinned reference T5 tokenizer, without Transformers."""
    from apxinf.policies.impls.fastwam import _Tokenizer

    path = os.environ.get("APXINF_FASTWAM_TOKENIZER")
    if path is None:
        pytest.skip("set APXINF_FASTWAM_TOKENIZER to the UMT5 tokenizer.json")
    tokenizer = _Tokenizer(Path(path))
    cases = [
        ("", [1]),
        (" \t\n", [1]),
        ("pick\tbowl", [21998, 73255, 1]),
        ("pick\nbowl", [21998, 73255, 1]),
        ("pick\u00a0bowl", [21998, 73255, 1]),
        ("pick\u001cbowl", [21998, 162840, 70748, 405, 1]),
        ("pick\u001fbowl", [21998, 166324, 70748, 405, 1]),
        ("pick\t</s>bowl", [21998, 1, 73255, 1]),
        ("pick\t<extra_id_0>bowl", [21998, 256299, 73255, 1]),
        ("<pad>\t<pad>", [0, 0, 1]),
        ("\t<extra_id_0>\t", [256299, 1]),
        ("café\t\n  bowl", [12831, 73255, 1]),
    ]
    for text, expected in cases:
        assert tokenizer.encode(text) == expected, repr(text)


class Runner:
    action_horizon, action_dim, num_views, image_size, max_token_len = (
        32,
        7,
        2,
        224,
        128,
    )

    def infer_rgb(self, rgb, layout, ids, noise, *, state):
        self.call = (rgb.copy(), layout, ids.copy(), noise, state.copy())
        result = np.zeros((32, 7), np.float32)
        result[:3, -1] = [-1, 0, 1]
        return result


class Tokenizer:
    def encode(self, text):
        self.text = text
        return [9, 8, 1]


def make_policy(**kwargs):
    return FastWamPolicy(
        Runner(),
        tokenizer=Tokenizer(),
        state_min=np.zeros(8),
        state_max=np.ones(8),
        action_min=np.zeros(7),
        action_max=np.ones(7),
        image_keys=["front", "wrist"],
        **kwargs,
    )


def observation():
    return {
        "front": np.zeros((224, 224, 3), np.uint8),
        "wrist": np.full((3, 224, 224), 255, np.uint8),
        "observation.state": np.linspace(0, 1, 8, dtype=np.float32),
        "task": "pick up the block",
    }


def test_camera_order_normalization_prompt_and_toggle():
    policy = make_policy()
    result = policy.infer(observation(), noise=np.zeros((32, 7), np.float32))
    rgb, layout, ids, noise, state = policy.model_runner.call
    assert layout == "nhwc"
    assert rgb.shape == (2, 224, 224, 3)
    assert np.all(rgb[0] == 0) and np.all(rgb[1] == 255)
    np.testing.assert_array_equal(
        state, np.float32(2) * observation()["observation.state"] - np.float32(1)
    )
    assert ids.tolist() == [9, 8, 1] + [0] * 125
    assert (
        policy.tokenizer.text
        == "A video recorded from a robot's point of view executing the following instruction: pick up the block"
    )
    np.testing.assert_array_equal(result["actions"][:3, -1], [1, 0, -1])
    np.testing.assert_array_equal(
        result["actions"][:, :6], np.full((32, 6), 0.5, np.float32)
    )
    assert get_policy("fastwam") is FastWamPolicy


@pytest.mark.parametrize(
    "change", ["float_image", "wrong_size", "bad_state", "bad_noise"]
)
def test_invalid_inputs_fail_before_native_execution(change):
    policy = make_policy()
    obs = observation()
    noise = None
    if change == "float_image":
        obs["front"] = obs["front"].astype(np.float32)
    if change == "wrong_size":
        obs["front"] = np.zeros((225, 224, 3), np.uint8)
    if change == "bad_state":
        obs["observation.state"][0] = np.nan
    if change == "bad_noise":
        noise = np.zeros((7, 32), np.float32)
    with pytest.raises(ValueError):
        policy.infer(obs, noise=noise)
    assert not hasattr(policy.model_runner, "call")


def test_constant_state_dimension_and_raw_prompt():
    policy = FastWamPolicy(
        Runner(),
        tokenizer=Tokenizer(),
        state_min=np.ones(8),
        state_max=np.ones(8),
        action_min=np.zeros(7),
        action_max=np.ones(7),
        image_keys=["front", "wrist"],
        prompt_template="{task}",
    )
    obs = observation()
    obs["observation.state"] = np.ones(8, np.float32)
    policy.infer(obs)
    np.testing.assert_array_equal(policy.model_runner.call[-1], -np.ones(8, np.float32))
    assert policy.tokenizer.text == obs["task"]


@pytest.mark.parametrize("toggle", [None, [], [-1]], ids=["missing", "empty", "libero"])
def test_checkpoint_toggle_matches_official_optional_default(tmp_path, monkeypatch, toggle):
    from apxinf.policies.impls import fastwam

    config = {
        "type": "fastwam",
        "input_features": {key: {"type": "VISUAL"} for key in ("front", "wrist")},
    }
    if toggle is not None:
        config["toggle_action_dimensions"] = toggle
    (tmp_path / "config.json").write_text(json.dumps(config))
    stats = {
        "observation.state.min": np.zeros(8),
        "observation.state.max": np.ones(8),
        "action.min": np.zeros(7),
        "action.max": np.ones(7),
    }
    monkeypatch.setattr(fastwam, "_stats", lambda *args: (stats, np.float32(1e-8)))
    policy = FastWamPolicy.from_pretrained(
        tmp_path, model_runner=Runner(), tokenizer=Tokenizer()
    )
    result = policy.infer(observation(), noise=np.zeros((32, 7), np.float32))
    expected = [1, 0, -1] if toggle else [0, 0.5, 1]
    np.testing.assert_array_equal(result["actions"][:3, -1], expected)
    np.testing.assert_array_equal(result["actions"][:, :6], np.full((32, 6), 0.5))
