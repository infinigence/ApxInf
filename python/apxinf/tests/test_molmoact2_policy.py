"""MolmoAct2 policy tests: prompt / state / image / action processing and the runner binding.

Offline: a fake runner and tokenizer stand in for the native binding. Comparisons with the official
processor's frames and tokenizer ids, and with the official model's actions, need the checkpoint and a
reference run; they are not part of this suite (see ``doc/molmoact2.md``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# torch/torchvision are the host-resize dependency of the policy and are declared test dependencies, so
# this module needs them (the tests compare against the official torch operations).
torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")

from apxinf.policies import AutoPolicy, MolmoAct2Policy, available_policies, get_policy
from apxinf.policies.impls.molmoact2 import (
    ACTION_DIM,
    STATE_DIM,
    _Q01Q99,
    build_prompt,
    check_checkpoint_config,
    discretize_state,
    expand_image_tokens,
    insert_bos,
    normalize_task_text,
    resize_bilinear_u8,
    round_to_bf16,
)

BOS = 151645
KEYS = ("observation.images.image", "observation.images.wrist_image")


def _stats(width, *, mask_last=True):
    q01 = np.linspace(-1.0, -0.2, width)
    q99 = np.linspace(0.4, 1.2, width)
    stats = {"q01": q01.tolist(), "q99": q99.tolist(), "min": (q01 - 1).tolist(), "max": (q99 + 1).tolist()}
    stats["mask"] = [True] * (width - 1) + [not mask_last]
    return stats


class FakeTokenizer:
    bos_id = BOS

    def __init__(self):
        self.texts = []

    def encode(self, text):
        self.texts.append(text)
        # deterministic, text dependent ids (never BOS)
        return [10 + (sum(map(ord, text[i : i + 7])) % 1000) for i in range(0, len(text), 7)]


class FakeRunner:
    action_horizon = 10
    action_dim = 32
    num_views = 2
    image_size = 378
    patch_size = 14
    initial_latent_shape = (1, 10, 32)
    model_variant = "bf16"

    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def infer_rgb(self, rgb_u8, layout, token_ids, noise=None):
        self.calls.append({"rgb": rgb_u8.copy(), "layout": layout, "ids": token_ids.copy(), "noise": noise})
        if self.result is not None:
            return self.result
        out = np.zeros((10, 32), dtype=np.float32)
        out[:, :7] = np.linspace(-2.0, 2.0, 70, dtype=np.float32).reshape(10, 7)
        return out


def _policy(runner=None, **kw):
    return MolmoAct2Policy(
        runner or FakeRunner(),
        tokenizer=FakeTokenizer(),
        state_normalizer=_Q01Q99(_stats(STATE_DIM), STATE_DIM, "state"),
        action_normalizer=_Q01Q99(_stats(ACTION_DIM), ACTION_DIM, "action"),
        setup_type="single franka robotic arm in libero",
        control_mode="delta end-effector pose",
        add_setup_tokens=True,
        add_control_tokens=True,
        image_keys=KEYS,
        **kw,
    )


def _observation(task="put the white mug on the plate", seed=0):
    rng = np.random.default_rng(seed)
    return {
        KEYS[0]: rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
        KEYS[1]: rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
        "observation.state": rng.uniform(-0.3, 0.3, STATE_DIM).astype(np.float32),
        "task": task,
    }


# ----------------------------------------------------------------------------- registration


def test_policy_is_registered_under_the_checkpoint_model_type():
    assert "molmoact2" in available_policies()
    assert get_policy("molmoact2") is MolmoAct2Policy
    assert get_policy("MolmoAct2") is MolmoAct2Policy


def test_auto_policy_dispatches_by_model_type_and_rejects_other_profiles(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "molmoact2", "add_action_expert": False}))
    with pytest.raises(ValueError, match="plain continuous-action"):
        AutoPolicy.from_pretrained(tmp_path)


@pytest.mark.parametrize(
    "key, value",
    [
        ("add_action_expert", False),
        ("action_mode", "discrete"),
        ("state_format", "continuous"),
        ("enable_depth_reasoning", True),
        ("mask_action_dim_padding", False),
        ("n_obs_steps", 2),
        ("num_state_tokens", 128),
        ("model_type", "other"),
    ],
)
def test_capability_check_rejects_every_deviation(key, value):
    good = {
        "model_type": "molmoact2",
        "add_action_expert": True,
        "action_mode": "both",
        "state_format": "discrete",
        "enable_depth_reasoning": False,
        "mask_action_dim_padding": True,
        "n_obs_steps": 1,
        "num_state_tokens": 256,
    }
    check_checkpoint_config(good)
    with pytest.raises(ValueError, match=key):
        check_checkpoint_config({**good, key: value})


# ----------------------------------------------------------------------------- prompt pieces


def test_task_text_normalization_follows_the_official_rules():
    assert normalize_task_text("  Put the Mug.  ") == "put the mug"
    assert normalize_task_text("Task: open the drawer!") == "open the drawer"
    assert normalize_task_text("The task is to pick it up") == "pick it up"
    assert normalize_task_text('"Pick it. Place it."') == "pick it; place it"
    assert normalize_task_text("a   b\n c") == "a b c"
    assert normalize_task_text("   ") == ""


def test_state_discretization_is_clipped_float32_round_half_even():
    bins = discretize_state(np.array([-1.0, 1.0, 0.0, 3.0, -9.0, np.nan, np.inf, -np.inf], dtype=np.float32))
    assert bins.tolist() == [0, 255, 128, 255, 0, 128, 255, 0]  # 127.5 rounds to the even 128
    assert discretize_state(np.array([-0.5], np.float32)).tolist() == [64]  # 63.75 -> 64
    assert discretize_state(np.zeros(8, np.float32)).shape == (8,)


def test_prompt_layout_and_image_token_counts():
    text = build_prompt(
        task="put the mug",
        state_bins=np.arange(8),
        setup_type="the setup",
        control_mode="the control",
        add_setup_tokens=True,
        add_control_tokens=True,
        num_images=2,
    )
    assert text.startswith("Image 1<|image|>Image 2<|image|><|im_start|>user\nThe task is to put the mug. ")
    assert "The setup is <setup_start>the setup<setup_end>." in text
    assert " The current state of the robot is <state_start><state_0><state_1><state_2><state_3><state_4><state_5><state_6><state_7><state_end>." in text
    assert "The expected control mode is <control_start>the control<control_end>. Given these, what action" in text
    assert text.endswith("<|im_end|>\n<|im_start|>assistant\n<action_output>")
    plain = build_prompt(
        task="t", state_bins=np.arange(2), setup_type="s", control_mode="c",
        add_setup_tokens=False, add_control_tokens=False, num_images=1,
    )
    assert "The setup is s." in plain and "<setup_start>" not in plain and "<control_start>" not in plain
    assert plain.startswith("Image 1<|image|><|im_start|>")

    expanded = expand_image_tokens(text, 14)
    assert "<|image|>" not in expanded
    assert expanded.count("<im_patch>") == 2 * 14 * 14
    assert expanded.count("<im_start>") == 2 and expanded.count("<im_end>") == 2
    assert expanded.count("<im_col>") == 0


def test_bos_is_inserted_once():
    assert insert_bos([5, 6], BOS) == [BOS, 5, 6]
    assert insert_bos([BOS, 5], BOS) == [BOS, 5]
    assert insert_bos([], BOS) == [BOS]


# ----------------------------------------------------------------------------- statistics


def test_normalizers_follow_the_masked_q01_q99_rule():
    action = _Q01Q99(_stats(ACTION_DIM), ACTION_DIM, "action")
    x = np.full((10, ACTION_DIM), 0.25, dtype=np.float32)
    x[0, 0], x[0, 6] = 5.0, 5.0  # outside [-1, 1] for the first dimension and the gripper
    out = action.unnormalize(x)
    q_low, q_high = action.q_low, action.q_high
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[1, :6], (1.25 * (q_high - q_low) / 2.0 + q_low)[:6], rtol=1e-6)
    assert out[1, 6] == np.float32(0.25)  # gripper: not mapped
    np.testing.assert_allclose(out[0, 0], q_high[0], rtol=1e-6)  # clipped to 1 first
    assert out[0, 6] == np.float32(1.0)  # gripper is clipped but not mapped

    state = _Q01Q99(_stats(STATE_DIM), STATE_DIM, "state")
    raw = np.linspace(-0.5, 0.5, STATE_DIM).astype(np.float32)
    normalized = state.normalize(raw)
    assert normalized.dtype == np.float32 and np.abs(normalized[:7]).max() <= 1.0
    assert normalized[7] == raw[7]  # the 8th entry is not normalized
    with pytest.raises(ValueError):
        _Q01Q99({"q01": [0.0] * 3, "q99": [1.0] * 3}, 4, "state")


# ----------------------------------------------------------------------------- image resize
#
# The default resize IS the checkpoint's CPU resize (torchvision ``Resize`` on ``uint8``), so the
# differential tests compare the product function with a verbatim copy of the official ``resize_image`` on
# a range of shapes. Every comparison is exact (no tolerance, no count of "rare" differences).


def _official_resize(image):
    """``resize_image`` of ``image_processing_molmoact2.py`` for a ``uint8`` HWC array, size 378 (verbatim ops)."""
    t = torch.permute(torch.from_numpy(image), [2, 0, 1])
    resized = torchvision.transforms.Resize([378, 378], torchvision.transforms.InterpolationMode.BILINEAR, antialias=False)(t)
    resized = torch.clip(resized, 0, 255).to(torch.uint8)
    return torch.permute(resized, [1, 2, 0]).numpy()


@pytest.mark.parametrize(
    "shape",
    [
        (3, 5),  # tiny, extreme upsample (an emulation of the resize differs here)
        (379, 379),  # one pixel above the target
        (720, 1280),  # large non-square downsample
        (256, 256),  # LIBERO camera size, upsample
        (378, 378),  # matching size
        (377, 377),
        (480, 640),  # non-square downsample
        (100, 200),  # non-square upsample
        (640, 480),  # portrait downsample
        (1000, 1000),  # large square downsample
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_resize_equals_the_official_resize_on_other_shapes(shape, seed):
    image = np.random.default_rng(1000 * seed + shape[0]).integers(0, 256, (*shape, 3), dtype=np.uint8)
    assert np.array_equal(resize_bilinear_u8(image), _official_resize(image))


def test_resize_edge_values_and_input_organisation():
    const = np.full((256, 256, 3), 77, dtype=np.uint8)
    assert (resize_bilinear_u8(const) == 77).all()
    ramp = np.zeros((64, 64, 3), dtype=np.uint8)
    ramp[:, ::2] = 255  # saturated checkerboard columns: 0 and 255 only
    out = resize_bilinear_u8(ramp)
    assert out.dtype == np.uint8 and out.shape == (378, 378, 3) and np.array_equal(out, _official_resize(ramp))
    base = np.random.default_rng(5).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    want = _official_resize(base)
    readonly = base.copy()
    readonly.setflags(write=False)
    sliced = np.ascontiguousarray(np.zeros((300, 300, 3), np.uint8))
    sliced[20:276, 10:266] = base
    for variant in (readonly, sliced[20:276, 10:266], np.asfortranarray(base)):
        assert np.array_equal(resize_bilinear_u8(variant), want)
    with pytest.raises(ValueError):
        resize_bilinear_u8(base[..., :2])
    with pytest.raises(ValueError):
        resize_bilinear_u8(base.astype(np.float32))


def test_policy_resize_takes_hwc_and_chw_images_to_the_same_official_frame():
    runner = FakeRunner()
    policy = _policy(runner)
    observation = _observation()
    chw = {**observation, **{k: np.ascontiguousarray(observation[k].transpose(2, 0, 1)) for k in KEYS}}
    policy.infer(observation)
    policy.infer(chw)
    for view, key in enumerate(KEYS):
        want = _official_resize(observation[key])
        assert np.array_equal(runner.calls[0]["rgb"][view], want)
        assert np.array_equal(runner.calls[1]["rgb"][view], want)


def test_missing_torch_is_an_explicit_error_and_never_an_approximation(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torchvision", None)  # makes ``import torchvision`` raise ImportError
    with pytest.raises(ImportError, match="torchvision"):
        resize_bilinear_u8(np.zeros((8, 8, 3), np.uint8))
    runner = FakeRunner()
    with pytest.raises(ImportError, match="No approximate fallback"):
        _policy(runner).infer(_observation())
    assert runner.calls == []
    with pytest.raises(ImportError, match="torchvision"):  # also at load time, before any checkpoint file is read
        MolmoAct2Policy.from_pretrained("/nonexistent-checkpoint-dir", model_runner=object())


# ----------------------------------------------------------------------------- BF16 output boundary


def _official_unnormalize_bf16(normalized_bf16, q_low, q_high, mask):
    """The official ``_FeatureNormalizer.unnormalize`` (q01_q99) applied to a BF16 torch tensor (verbatim ops)."""
    arr = normalized_bf16.float().cpu().numpy()
    arr = np.clip(arr, -1.0, 1.0)
    out = (arr + 1.0) * (q_high - q_low) / 2.0 + q_low
    out = np.where(mask, out, arr)
    return torch.as_tensor(out, dtype=normalized_bf16.dtype)


def test_round_to_bf16_is_round_to_nearest_even_like_torch():
    rng = np.random.default_rng(0)
    values = np.concatenate([
        rng.standard_normal(100000).astype(np.float32) * np.float32(3.0),
        np.float32([0.0, -0.0, 1.0, 1.00390625, 1.01171875, 1.0078125, 0.99609375, 65504.0, 1e-30, -1e-30]),
        # exact ties between two BF16 neighbours (low 16 bits == 0x8000), odd and even upper halves
        (np.arange(0x3F80, 0x3F90, dtype=np.uint32) << 16 | np.uint32(0x8000)).view(np.float32),
    ])
    want = torch.from_numpy(values).to(torch.bfloat16).float().numpy()
    got = round_to_bf16(values)
    assert got.dtype == np.float32 and np.array_equal(got.view(np.uint32), want.view(np.uint32))
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(FloatingPointError):
            round_to_bf16(np.float32([0.0, bad]))


def test_postprocess_equals_the_official_bf16_output_on_the_same_operands():
    policy = _policy()
    q_low, q_high = policy.action_normalizer.q_low, policy.action_normalizer.q_high
    rng = np.random.default_rng(7)
    for scale in (0.5, 1.0, 2.5):  # inside the clip range, around it, and far outside
        trajectory = torch.from_numpy(rng.standard_normal((10, 32)).astype(np.float32) * scale).to(torch.bfloat16)
        widened = trajectory.float().numpy()  # what the native runner returns: BF16 values in float32
        got = policy.postprocess(widened)
        want = _official_unnormalize_bf16(trajectory[:, :ACTION_DIM], q_low, q_high, policy.action_normalizer.mask)
        want = want.float().numpy()
        assert got.shape == (10, 7) and got.dtype == np.float32 and got.flags["C_CONTIGUOUS"]
        assert np.array_equal(got.view(np.uint32), want.view(np.uint32))
        # the rounding is not a no-op: the float32 unnormalize alone differs from the BF16 result
        plain = policy.action_normalizer.unnormalize(np.ascontiguousarray(widened[:, :ACTION_DIM]))
        assert not np.array_equal(plain, got)


def test_float32_reference_path_keeps_its_own_precision():
    """The FP32 official reference does not round: the normalizer alone is the float32 path."""
    policy = _policy()
    x = np.random.default_rng(8).standard_normal((10, ACTION_DIM)).astype(np.float32)
    plain = policy.action_normalizer.unnormalize(x)
    assert plain.dtype == np.float32 and not np.array_equal(plain, round_to_bf16(plain))


def test_only_the_bf16_runner_profile_is_accepted():
    class Fp16Runner(FakeRunner):
        model_variant = "fp16"

    with pytest.raises(ValueError, match="bf16 profile"):
        _policy(Fp16Runner())


# ----------------------------------------------------------------------------- policy


def test_infer_binds_views_tokens_noise_and_returns_physical_actions():
    runner = FakeRunner()
    policy = _policy(runner)
    observation = _observation()
    noise = np.random.default_rng(3).standard_normal((1, 10, 32)).astype(np.float32)
    result = policy.infer(observation, noise=noise)

    (call,) = runner.calls
    assert call["layout"] == "nhwc" and call["rgb"].shape == (2, 378, 378, 3) and call["rgb"].dtype == np.uint8
    assert np.array_equal(call["rgb"][0], resize_bilinear_u8(observation[KEYS[0]]))  # image first, wrist second
    assert np.array_equal(call["rgb"][1], resize_bilinear_u8(observation[KEYS[1]]))
    assert call["ids"].dtype == np.uint32 and call["ids"][0] == BOS
    assert np.array_equal(call["noise"], noise)

    actions = result["actions"]
    assert actions.shape == (10, 7) and actions.dtype == np.float32 and np.isfinite(actions).all()
    assert policy.action_dim == 7 and policy.action_horizon == 10
    normalized = result["normalized_actions"]
    assert normalized.shape == (10, 32)
    np.testing.assert_array_equal(actions, policy.postprocess(normalized))
    np.testing.assert_array_equal(actions[:, 6], round_to_bf16(np.clip(normalized[:, 6], -1.0, 1.0)))  # gripper
    assert set(result["timing"]) >= {"model_ms", "total_ms"}
    assert result["metadata"]["action_dim"] == 7


def test_execution_mode_is_reported_by_the_runner_and_in_every_result():
    class Graphed(FakeRunner):
        def execution_mode(self):
            return "graph"

    policy = _policy(Graphed())
    assert policy.execution_mode == "graph"
    assert policy.infer(_observation())["execution_mode"] == "graph"
    # a runner that cannot report its mode is shown as unknown, never guessed
    assert _policy(FakeRunner()).execution_mode == "unknown"


@pytest.mark.parametrize("execution", [None, "eager", "require-graph"])
def test_from_pretrained_forwards_the_execution_policy_to_the_native_loader(monkeypatch, tmp_path, execution):
    import sys
    import types

    import apxinf.policies.impls.molmoact2 as module

    seen = {}

    class Loaded(FakeRunner):
        pass

    class ModelRunner:
        @staticmethod
        def load(name, directory, **kwargs):
            seen.update(kwargs, name=name, directory=Path(directory))
            return Loaded()

    monkeypatch.setitem(sys.modules, "apxinf_py", types.SimpleNamespace(ModelRunner=ModelRunner))
    monkeypatch.setattr(module, "check_checkpoint_config", lambda config: None)
    monkeypatch.setattr(module, "_Tokenizer", lambda directory: FakeTokenizer())
    stats = {
        "norm_mode": "q01_q99",
        "metadata_by_tag": {
            module.NORM_TAG: {
                "state_stats": _stats(STATE_DIM),
                "action_stats": _stats(ACTION_DIM),
                "action_horizon": 10,
                "n_action_steps": 10,
                "camera_keys": list(KEYS),
            }
        },
    }
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "molmoact2"}))
    (tmp_path / "norm_stats.json").write_text(json.dumps(stats))

    policy = MolmoAct2Policy.from_pretrained(tmp_path, image_transform=lambda image: image, execution=execution)
    assert seen["name"] == "molmoact2" and seen["directory"] == tmp_path
    assert seen["execution"] == execution  # forwarded verbatim; None leaves the native default (prefer-graph)
    assert isinstance(policy, MolmoAct2Policy)


def test_prompt_depends_on_the_task_state_and_nothing_else():
    policy = _policy()
    base = _observation()
    a = policy.infer(base)["token_ids"]
    again = policy.infer(base)["token_ids"]
    assert np.array_equal(a, again)
    other_task = policy.infer({**base, "task": "open the drawer"})["token_ids"]
    assert not np.array_equal(a, other_task)
    moved = {**base, "observation.state": base["observation.state"] + np.float32(0.2)}
    assert not np.array_equal(a, policy.infer(moved)["token_ids"])
    # a state change below the bin width keeps the prompt, a new image never changes it
    nudged = {**base, "observation.state": base["observation.state"] + np.float32(1e-7)}
    assert np.array_equal(a, policy.infer(nudged)["token_ids"])
    new_image = {**base, KEYS[0]: np.zeros((256, 256, 3), dtype=np.uint8)}
    assert np.array_equal(a, policy.infer(new_image)["token_ids"])


def test_new_images_reach_the_runner_and_chw_input_is_accepted():
    runner = FakeRunner()
    policy = _policy(runner)
    base = _observation()
    policy.infer(base)
    chw = {**base, KEYS[0]: np.ascontiguousarray(base[KEYS[0]].transpose(2, 0, 1))}
    policy.infer(chw)
    changed = {**base, KEYS[1]: 255 - base[KEYS[1]]}
    policy.infer(changed)
    assert np.array_equal(runner.calls[0]["rgb"], runner.calls[1]["rgb"])
    assert not np.array_equal(runner.calls[0]["rgb"][1], runner.calls[2]["rgb"][1])
    assert np.array_equal(runner.calls[0]["rgb"][0], runner.calls[2]["rgb"][0])


def test_image_transform_replaces_the_resize_and_is_validated():
    seen = []

    def exact(image):
        seen.append(image.shape)
        return np.full((378, 378, 3), 9, dtype=np.uint8)

    runner = FakeRunner()
    _policy(runner, image_transform=exact).infer(_observation())
    assert seen == [(256, 256, 3)] * 2 and (runner.calls[0]["rgb"] == 9).all()
    with pytest.raises(ValueError, match="image_transform"):
        _policy(image_transform=lambda image: image).infer(_observation())


@pytest.mark.parametrize(
    "mutate, error",
    [
        (lambda o: o.pop(KEYS[1]), KeyError),
        (lambda o: o.pop("observation.state"), KeyError),
        (lambda o: o.pop("task"), KeyError),
        (lambda o: o.update({KEYS[0]: o[KEYS[0]].astype(np.float32) / 255}), TypeError),
        (lambda o: o.update({KEYS[0]: np.zeros((256, 256), np.uint8)}), ValueError),
        (lambda o: o.update({"observation.state": np.zeros(7, np.float32)}), ValueError),
        (lambda o: o.update({"observation.state": np.full(8, np.nan, np.float32)}), FloatingPointError),
        (lambda o: o.update({"task": 3}), TypeError),
    ],
)
def test_bad_observations_fail_before_the_runner_is_called(mutate, error):
    runner = FakeRunner()
    observation = _observation()
    mutate(observation)
    with pytest.raises(error):
        _policy(runner).infer(observation)
    assert runner.calls == []


def test_bad_noise_is_rejected_before_the_runner_is_called():
    runner = FakeRunner()
    policy = _policy(runner)
    for bad in (np.zeros((10, 32), np.float32), np.zeros((1, 10, 7), np.float32)):
        with pytest.raises(ValueError):
            policy.infer(_observation(), noise=bad)
    nan = np.zeros((1, 10, 32), np.float32)
    nan[0, 0, 0] = np.nan
    with pytest.raises(FloatingPointError):
        policy.infer(_observation(), noise=nan)
    assert runner.calls == []


def test_runner_failures_and_bad_outputs_propagate():
    class Failing(FakeRunner):
        def infer_rgb(self, *args, **kwargs):
            raise RuntimeError("device failure")

    with pytest.raises(RuntimeError, match="device failure"):
        _policy(Failing()).infer(_observation())
    with pytest.raises(FloatingPointError):
        _policy(FakeRunner(np.full((10, 32), np.nan, np.float32))).infer(_observation())
    with pytest.raises(ValueError, match="runner returned"):
        _policy(FakeRunner(np.zeros((10, 7), np.float32))).infer(_observation())
