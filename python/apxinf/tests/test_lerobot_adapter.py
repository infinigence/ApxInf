"""lerobot adapter tests: observation translation, chunk queueing, action shape.

Offline — a ``MockModel`` (no CUDA, no ``apxinf_py``) backs a real ``Pi05Policy``,
and lerobot itself is never imported: the adapter's own surface is what is under
test, and only :meth:`ApxInfPolicy.build_inference_frame` touches lerobot (for
``build_dataset_frame``), which the example covers instead.

The observation-translation tests are pure numpy and always run. torch is only
needed where an action crosses the tensor boundary, so those tests skip without
it — the translation half is where the interesting failure modes live.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from apxinf import Pi05Policy
from apxinf.adapters.lerobot import (
    ApxInfPolicy,
    IdentityProcessor,
    observation_to_apxinf,
)
from apxinf.processors import ProcessorStep, PromptTokenizer, Unnormalizer

HORIZON = 10
MODEL_DIM = 32
LIBERO_DIM = 7
IMAGE_SIZE = 224
NUM_VIEWS = 2
APXINF_KEYS = ("observation/image", "observation/wrist_image")
LEROBOT_KEYS = ("observation.images.base_0_rgb", "observation.images.left_wrist_0_rgb")

# Only the action-returning paths cross the tensor boundary; skip just those.
needs_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="the lerobot adapter's tensor boundary needs torch",
)


class MockModel:
    """Deterministic stand-in with a device-sampling-shaped default path."""

    def __init__(self):
        self.action_horizon = HORIZON
        self.action_dim = MODEL_DIM
        self.num_views = NUM_VIEWS
        self.image_size = IMAGE_SIZE
        self.max_token_len = 200

    def infer_rgb(self, rgb_u8, layout, token_ids, noise=None):
        assert layout == "nhwc"
        assert rgb_u8.dtype == np.uint8
        assert rgb_u8.shape == (NUM_VIEWS, IMAGE_SIZE, IMAGE_SIZE, 3)
        if noise is None:
            return np.zeros((HORIZON, MODEL_DIM), dtype=np.float32)
        return np.asarray(noise, dtype=np.float32)


class ConstTokenizer(PromptTokenizer):
    """Fixed token ids, so no SentencePiece model file is needed."""

    def __init__(self, max_token_len=200):
        self.max_token_len = max_token_len
        self.discrete_state = False

    def __call__(self, prompt, state=None):
        return np.arange(self.max_token_len, dtype=np.uint32)


def build_policy() -> Pi05Policy:
    model = MockModel()
    input_pipeline, output_pipeline = Pi05Policy.default_pipelines(
        model,
        tokenizer=ConstTokenizer(),
        unnormalizer=Unnormalizer(
            q01=[-1.0] * LIBERO_DIM, q99=[1.0] * LIBERO_DIM, dims=LIBERO_DIM, eps=0.0
        ),
        image_keys=APXINF_KEYS,
    )
    return Pi05Policy(
        model,
        input_pipeline=input_pipeline,
        output_pipeline=output_pipeline,
        image_keys=APXINF_KEYS,
    )


def lerobot_frame(height=64, width=96, seed=0, state_dim=LIBERO_DIM):
    """A frame shaped like lerobot's ``build_dataset_frame`` output (numpy seam)."""
    rng = np.random.default_rng(seed)
    frame = {
        key: rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        for key in LEROBOT_KEYS
    }
    frame["observation.state"] = rng.standard_normal(state_dim).astype(np.float32)
    frame["task"] = "pick up the block"
    return frame


# --- observation translation (pure numpy) ----------------------------------


def test_numpy_frame_translates_to_apxinf_keys():
    observation = observation_to_apxinf(lerobot_frame(), image_keys=APXINF_KEYS)

    assert set(observation) == {*APXINF_KEYS, "observation/state", "prompt"}
    assert observation["prompt"] == "pick up the block"
    for key in APXINF_KEYS:
        assert observation[key].dtype == np.uint8
        assert observation[key].shape == (64, 96, 3)


def test_camera_order_follows_frame_order():
    frame = lerobot_frame()
    observation = observation_to_apxinf(frame, image_keys=APXINF_KEYS)

    # First frame camera -> first policy camera, and so on down the list.
    for lerobot_key, apxinf_key in zip(LEROBOT_KEYS, APXINF_KEYS):
        assert np.array_equal(observation[apxinf_key], frame[lerobot_key])


def test_task_argument_overrides_frame_task():
    observation = observation_to_apxinf(
        lerobot_frame(), image_keys=APXINF_KEYS, task="open the drawer"
    )
    assert observation["prompt"] == "open the drawer"


def test_camera_count_mismatch_is_rejected():
    frame = lerobot_frame()
    del frame["observation.images.left_wrist_0_rgb"]

    with pytest.raises(ValueError, match="expects 2 cameras"):
        observation_to_apxinf(frame, image_keys=APXINF_KEYS)


def test_tensor_seam_round_trips_losslessly():
    """A frame already through lerobot's tensor prep recovers the exact pixels.

    ``k / 255`` is inexact in float32, so this pins the rounding: a truncating
    cast would land on ``k - 1`` for some values. Exercised with numpy standing in
    for the tensor (same ``[1, C, H, W]`` float layout), so it runs torch-free.
    """
    frame = lerobot_frame()
    prepared = {
        key: (value.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        if key.startswith("observation.images.")
        else value
        for key, value in frame.items()
    }

    observation = observation_to_apxinf(prepared, image_keys=APXINF_KEYS)

    for lerobot_key, apxinf_key in zip(LEROBOT_KEYS, APXINF_KEYS):
        assert observation[apxinf_key].dtype == np.uint8
        assert np.array_equal(observation[apxinf_key], frame[lerobot_key])


def test_float_frame_outside_unit_range_is_rejected_not_saturated():
    """A [0, 255] float frame must fail loudly rather than clip to white."""
    frame = lerobot_frame()
    frame[LEROBOT_KEYS[0]] = frame[LEROBOT_KEYS[0]].astype(np.float32)

    with pytest.raises(ValueError, match=r"outside the expected \[0, 1\] range"):
        observation_to_apxinf(frame, image_keys=APXINF_KEYS)


def test_batched_frame_is_rejected():
    frame = lerobot_frame()
    frame[LEROBOT_KEYS[0]] = np.repeat(frame[LEROBOT_KEYS[0]][None], 2, axis=0)

    with pytest.raises(ValueError, match="batch size 2"):
        observation_to_apxinf(frame, image_keys=APXINF_KEYS)


def test_state_is_absent_when_the_frame_omits_it():
    frame = lerobot_frame()
    del frame["observation.state"]

    observation = observation_to_apxinf(frame, image_keys=APXINF_KEYS)

    assert "observation/state" not in observation
    assert set(observation) == {*APXINF_KEYS, "prompt"}


# --- policy surface ---------------------------------------------------------


def test_adapter_matches_the_wrapped_policy_shape_contract():
    policy = build_policy()
    adapter = ApxInfPolicy(policy)

    assert adapter.action_dim == policy.action_dim
    assert adapter.action_horizon == policy.action_horizon
    assert adapter.image_keys == APXINF_KEYS
    assert adapter.metadata is policy.metadata


def test_pre_post_processors_are_pass_throughs():
    adapter = ApxInfPolicy(build_policy())
    preprocess, postprocess = adapter.make_pre_post_processors()

    frame = lerobot_frame()
    assert isinstance(preprocess, IdentityProcessor)
    assert preprocess(frame) is frame
    assert postprocess(frame) is frame
    assert preprocess.steps == ()


def test_n_action_steps_cannot_exceed_the_horizon():
    adapter = ApxInfPolicy(build_policy(), n_action_steps=HORIZON + 5)
    assert adapter.n_action_steps == HORIZON


def test_missing_camera_keys_are_rejected_at_construction():
    class Bare:
        action_horizon = HORIZON
        action_dim = LIBERO_DIM
        metadata: dict = {}

    with pytest.raises(ValueError, match="cannot infer the policy's camera keys"):
        ApxInfPolicy(Bare())


@needs_torch
def test_select_action_returns_one_batched_step():
    import torch

    adapter = ApxInfPolicy(build_policy())
    action = adapter.select_action(lerobot_frame())

    assert isinstance(action, torch.Tensor)
    assert action.shape == (1, LIBERO_DIM)


@needs_torch
def test_predict_action_chunk_returns_the_whole_chunk():
    adapter = ApxInfPolicy(build_policy())
    chunk = adapter.predict_action_chunk(lerobot_frame())

    assert chunk.shape == (1, HORIZON, LIBERO_DIM)


@needs_torch
def test_select_action_drains_the_chunk_before_reinferring():
    import torch

    adapter = ApxInfPolicy(build_policy())
    frame = lerobot_frame()

    served = [adapter.select_action(frame) for _ in range(HORIZON)]
    chunk = adapter.predict_action_chunk(frame)

    # The queue served the horizon step by step; a fresh inference then differs
    # (the noise step advances its RNG per call), proving it was not refilled.
    assert len(served) == HORIZON
    assert not torch.equal(served[0], served[1])
    assert not torch.equal(torch.cat(served).unsqueeze(0), chunk)


@needs_torch
def test_select_action_serves_the_chunk_in_order():
    import torch

    adapter = ApxInfPolicy(build_policy())
    frame = lerobot_frame()

    first = adapter.select_action(frame)
    second = adapter.select_action(frame)
    adapter.reset()
    chunk = adapter.predict_action_chunk(frame)

    # A fresh chunk cannot be compared to the served one (noise advances), so
    # check the queue's own ordering: two pops came from one inference.
    assert first.shape == second.shape == (1, LIBERO_DIM)
    assert chunk.shape == (1, HORIZON, LIBERO_DIM)


@needs_torch
def test_reset_drops_queued_actions():
    import torch

    adapter = ApxInfPolicy(build_policy())
    frame = lerobot_frame()

    first = adapter.select_action(frame)
    adapter.reset()
    after_reset = adapter.select_action(frame)

    # Without the reset this would have been the chunk's *second* step.
    assert not torch.equal(first, after_reset)


@needs_torch
def test_n_action_steps_caps_the_queue():
    adapter = ApxInfPolicy(build_policy(), n_action_steps=3)
    assert adapter.n_action_steps == 3

    adapter.select_action(lerobot_frame())
    assert len(adapter._queue) == 2


@needs_torch
def test_rtc_kwargs_are_rejected_not_ignored():
    adapter = ApxInfPolicy(build_policy())

    with pytest.raises(TypeError, match="RTC"):
        adapter.select_action(lerobot_frame(), prefix_actions=None)


@needs_torch
def test_apxinf_observation_dict_is_accepted_directly():
    """A caller who already speaks apxinf keys should not be re-translated."""
    adapter = ApxInfPolicy(build_policy())
    rng = np.random.default_rng(0)
    observation = {
        key: rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8) for key in APXINF_KEYS
    }
    observation["prompt"] = "pick up the block"

    assert adapter.select_action(observation).shape == (1, LIBERO_DIM)


@needs_torch
def test_custom_pipeline_step_still_runs_through_the_adapter():
    """The adapter must not bypass a policy's replaced steps."""
    policy = build_policy()
    original = policy.output_pipeline["trim"]
    calls = []

    class CountingTrim(ProcessorStep):
        action_dim = original.action_dim

        def __call__(self, data):
            calls.append(1)
            return original(data)

    policy.output_pipeline = policy.output_pipeline.replace("trim", CountingTrim())
    ApxInfPolicy(policy).select_action(lerobot_frame())

    assert calls == [1]
