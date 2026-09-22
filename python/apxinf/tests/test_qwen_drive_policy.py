"""Public planning options must reach the native Qwen-Drive executor."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from apxinf.policies.impls import qwen_drive


class _Tokenizer:
    def __init__(self, model_dir):
        pass

    def token_id(self, token):
        return {"<|im_start|>": 1, "<|im_end|>": 2}[token]

    def encode(self, text):
        return [3]

    def decode(self, ids, **kwargs):
        return "reasoning"


class _Native:
    def plan_direct(self, *args):
        self.steps = args[-1]
        return np.zeros((50, 3), dtype=np.float32)

    def plan_reasoning(self, *args):
        self.steps = args[-1]
        return [3, 2], np.zeros((50, 3), dtype=np.float32)


@pytest.mark.parametrize("mode", ["direct_planning", "reasoning_planning"])
@pytest.mark.parametrize("override", [None, 7])
def test_planning_step_count_matches_metadata_and_native_call(tmp_path, monkeypatch, mode, override):
    config = {
        "vlm_config": {
            "image_token_id": 4, "vision_start_token_id": 5,
            "vision_end_token_id": 6, "text_config": {"eos_token_id": 2},
        },
        "image_patch_size": 2, "image_spatial_merge_size": 1,
        "image_temporal_patch_size": 1, "history_image_pixels": 64,
        "current_image_pixels": 64, "trajectory_scale": [1, 1, 1],
        "num_future_points": 50, "trajectory_point_dim": 3,
        "num_inference_steps": 4,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    native = _Native()
    monkeypatch.setattr(qwen_drive, "_Tokenizer", _Tokenizer)
    monkeypatch.setitem(__import__("sys").modules, "apxinf_py", SimpleNamespace(
        QwenDriveModel=SimpleNamespace(load=lambda *args, **kwargs: native)))
    policy = qwen_drive.QwenDrivePolicy.from_pretrained(
        tmp_path, mode=mode, num_steps=override)
    observation = {
        "views": {"front": [{"image": np.zeros((4, 4, 3), dtype=np.uint8)}]},
        "nav_command": 0, "history": [[0, 0, 0], [0, 0, 0]],
        "history_velocity": [[0, 0]], "history_acceleration": [[0, 0]],
        "ego_velocity": [0, 0], "ego_acceleration": [0, 0],
        "driving_command": [0],
    }
    result = policy.infer(observation, noise=np.zeros((1, 50, 3), dtype=np.float32))
    expected = 4 if override is None else override
    assert result["metadata"]["num_inference_steps"] == expected
    assert native.steps == expected
    assert result["actions"].shape == (1, 50, 3)
    policy.close()
