import json

import numpy as np
import pytest

from apxinf.policies import AutoPolicy, CosmosPolicy
from apxinf.policies.impls.cosmos_policy import _resize_crop


class Runner:
    action_horizon = 16
    action_dim = 7
    num_views = 2
    image_size = 224
    initial_latent_shape = [1, 16, 9, 28, 28]

    def __init__(self):
        self.calls = []
        self.output = np.zeros((16, 7), dtype=np.float32)

    def infer_rgb(self, rgb, layout, tokens, noise, *, state):
        self.calls.append((rgb.copy(), layout, tokens.copy(), noise, state.copy()))
        return self.output


@pytest.fixture
def stats():
    return {
        "actions_min": [-2.] * 7, "actions_max": [4.] * 7,
        "proprio_min": [-1.] * 9, "proprio_max": [3.] * 9,
    }


@pytest.fixture
def observation():
    return {"wrist_image": np.full((256, 256, 3), 127, np.uint8),
            "primary_image": np.full((256, 256, 3), 64, np.uint8),
            "robot": {"state": np.ones(9)}, "prompt": "move the cup"}


def policy(runner, stats):
    return CosmosPolicy(runner, tasks=["move the cup", "open the drawer"],
                        dataset_stats=stats, state_key="robot/state")


def test_observation_to_action_and_joint_noise(stats, observation):
    runner = Runner()
    model = policy(runner, stats)
    noise = np.random.RandomState(1).standard_normal(runner.initial_latent_shape).astype(np.float32)
    result = model.infer(observation, noise=noise)
    rgb, layout, tokens, supplied, state = runner.calls[0]
    assert rgb.shape == (2, 224, 224, 3) and rgb.dtype == np.uint8
    assert layout == "nhwc"
    np.testing.assert_array_equal(rgb[0], 127)
    np.testing.assert_array_equal(rgb[1], 64)
    np.testing.assert_array_equal(tokens, [0])
    np.testing.assert_array_equal(supplied, noise)
    np.testing.assert_array_equal(state, np.zeros(9, dtype=np.float32))
    assert state.dtype == np.float32
    np.testing.assert_array_equal(result["actions"], np.ones((16, 7), dtype=np.float32))
    assert result["actions"].dtype == np.float32
    assert set(result["timing"]) == {"model_ms", "total_ms"}


def test_generated_noise_and_task_rebinding(stats, observation):
    runner = Runner()
    model = policy(runner, stats)
    model.infer(observation)
    observation["prompt"] = "open the drawer"
    model.infer(observation)
    assert runner.calls[0][3] is None and runner.calls[1][3] is None
    np.testing.assert_array_equal(runner.calls[1][2], [1])
    observation["prompt"] = "uncached task"
    with pytest.raises(ValueError, match="cached text"):
        model.infer(observation)
    assert len(runner.calls) == 2


@pytest.mark.parametrize("shape", [(16, 7), (1, 16, 7), (16, 9, 28, 28), (1, 16, 8, 28, 28)])
def test_action_only_or_incomplete_noise_rejected(stats, observation, shape):
    runner = Runner()
    with pytest.raises(ValueError, match="noise"):
        policy(runner, stats).infer(observation, noise=np.zeros(shape, np.float32))
    assert not runner.calls


def test_nonfinite_inputs_and_outputs_rejected(stats, observation):
    runner = Runner()
    model = policy(runner, stats)
    noise = np.zeros(runner.initial_latent_shape, np.float32)
    noise.flat[-1] = np.inf
    with pytest.raises(ValueError, match="noise"):
        model.infer(observation, noise=noise)
    observation["robot"]["state"][0] = np.nan
    with pytest.raises(ValueError, match="proprioception"):
        model.infer(observation)
    assert not runner.calls
    observation["robot"]["state"][0] = 1
    runner.output[0, 0] = np.inf
    with pytest.raises(ValueError, match="invalid actions"):
        model.infer(observation)


def test_retained_results_and_close(stats, observation):
    runner = Runner()
    model = policy(runner, stats)
    first = model.infer(observation)
    runner.output.fill(1)
    second = model.infer(observation)
    np.testing.assert_array_equal(first["normalized_actions"], 0)
    np.testing.assert_array_equal(first["actions"], 1)
    np.testing.assert_array_equal(second["actions"], 4)
    model.close()
    model.close()
    with pytest.raises(RuntimeError, match="closed"):
        model.infer(observation)


def test_auto_policy_loads_prepared_assets_with_injected_runner(tmp_path, stats):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "cosmos-policy", "profile": "libero-2b-direct-v1"}))
    (tmp_path / "tasks.json").write_text(json.dumps({"tasks": ["move the cup"]}))
    (tmp_path / "libero_dataset_statistics.json").write_text(json.dumps(stats))
    runner = Runner()
    model = AutoPolicy.from_pretrained(tmp_path, model_runner=runner, state_key="state")
    assert isinstance(model, CosmosPolicy) and model.model_runner is runner
    with pytest.raises(ValueError, match="load options"):
        AutoPolicy.from_pretrained(tmp_path, model_runner=runner, state_key="state", precision="fp8")


def test_wrong_runtime_profile_and_stats_rejected(stats):
    runner = Runner()
    runner.initial_latent_shape = [16, 7]
    with pytest.raises(ValueError, match="joint latent"):
        policy(runner, stats)
    stats["proprio_max"][0] = stats["proprio_min"][0]
    with pytest.raises(ValueError, match="increasing bounds"):
        policy(Runner(), stats)


def test_crop_resize_matches_torch_tensor_reference():
    torch = pytest.importorskip("torch")
    functional = pytest.importorskip("torchvision.transforms.functional")
    rng = np.random.RandomState(42)
    for _ in range(5):
        image = rng.randint(0, 256, (212, 212, 3), dtype=np.uint8)
        reference = functional.resize(torch.from_numpy(image).permute(2, 0, 1), [224, 224], antialias=True)
        np.testing.assert_array_equal(_resize_crop(image), reference.permute(1, 2, 0).numpy())

    # Smooth regions exercise half-integer ties which random images rarely hit.
    y, x = np.indices((212, 212))
    for image in [np.stack([x, y, (x + y) // 2], axis=-1).astype(np.uint8),
                  np.stack([x // 4, y // 3, (x * y) % 256], axis=-1).astype(np.uint8)]:
        reference = functional.resize(torch.from_numpy(image).permute(2, 0, 1), [224, 224], antialias=True)
        np.testing.assert_array_equal(_resize_crop(image), reference.permute(1, 2, 0).numpy())
