from __future__ import annotations

import numpy as np

from apxinf.policies import SmolVlaPolicy, available_policies
from apxinf.policies.impls.smolvla import SmolVlaPolicy as ConcretePolicy


class FakeTokenizer:
    pad_id = 99

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]


class FakeRunner:
    action_horizon = 50
    action_dim = 7
    num_views = 2
    image_size = 512
    max_token_len = 48

    def __init__(self):
        self.calls = []

    def infer_rgb(self, rgb, layout, token_ids, noise=None, state=None):
        self.calls.append(
            {
                "rgb": rgb,
                "layout": layout,
                "token_ids": token_ids,
                "noise": noise,
                "state": state,
            }
        )
        return np.zeros((self.action_horizon, self.action_dim), dtype=np.float32)


def test_smolvla_policy_registered():
    assert "smolvla" in available_policies()
    assert "smolvla_libero" in available_policies()
    assert SmolVlaPolicy is ConcretePolicy


def test_smolvla_policy_preprocesses_observation():
    runner = FakeRunner()
    policy = SmolVlaPolicy(
        runner,
        tokenizer=FakeTokenizer(),
        state_mean=np.full(8, 2.0, dtype=np.float32),
        state_std=np.full(8, 4.0, dtype=np.float32),
        action_mean=np.full(7, 1.0, dtype=np.float32),
        action_std=np.full(7, 0.5, dtype=np.float32),
        image_keys=("camera_0", "camera_1"),
    )

    images = [
        np.full((300, 400, 3), 200, dtype=np.uint8),
        np.full((400, 300, 3), 100, dtype=np.uint8),
    ]
    result = policy.infer(
        {
            "camera_0": images[0],
            "camera_1": images[1],
            "observation.state": np.arange(8, dtype=np.float32),
            "task": "put the mug on the plate",
        }
    )

    assert result["actions"].shape == (50, 7)
    assert np.allclose(result["actions"], 1.0)
    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert call["rgb"].shape == (2, 512, 512, 3)
    assert call["rgb"].dtype == np.uint8
    assert call["token_ids"].shape == (3,)
    assert call["token_ids"][:3].tolist() == [1, 2, 3]
    assert np.allclose(call["state"], (np.arange(8) - 2.0) / 4.0)
    assert call["rgb"][0, 0, 0].tolist() == [0, 0, 0]
    assert call["rgb"][0, -1, -1].tolist() == [200, 200, 200]
