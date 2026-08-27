"""Pi05Policy composition tests.

Most run offline against a ``MockModel`` implementing the ``BareModel`` protocol
(no CUDA, no apxinf_py). One gated test cross-checks the real ``apxinf_py`` model
when ``APXINF_PI05_MODEL_DIR`` + a CUDA build are available.

The policy is ``input_pipeline -> model -> output_pipeline`` (openpi-shaped), so
these tests build the default pre/post chains with ``default_pipelines`` and also
exercise reordering / insertion / replacement of pipeline steps.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from apxinf import Pi05Policy
from apxinf.processors import (
    GaussianNoise,
    ParseImage,
    Pipeline,
    ProcessorStep,
    PromptTokenizer,
    ResizeWithPad,
    Unnormalizer,
)

HORIZON = 10
MODEL_DIM = 32
LIBERO_DIM = 7
IMAGE_SIZE = 224
NUM_VIEWS = 2
DEFAULT_KEYS = ("observation/image", "observation/wrist_image")


class MockModel:
    """Stand-in: provided noise is exact; absent noise is sampled internally."""

    def __init__(self):
        self.action_horizon = HORIZON
        self.action_dim = MODEL_DIM
        self.num_views = NUM_VIEWS
        self.image_size = IMAGE_SIZE
        self.max_token_len = 200
        self.last_rgb = None
        self.last_tokens = None
        self.last_noise = None
        self.sampling_seed = 0
        self.sampling_draw = 0

    def reset_sampling(self, seed=None):
        if seed is not None:
            self.sampling_seed = int(seed)
        self.sampling_draw = 0

    def infer_rgb(self, rgb_u8, layout, token_ids, noise=None):
        assert layout == "nhwc"
        assert rgb_u8.shape == (NUM_VIEWS, IMAGE_SIZE, IMAGE_SIZE, 3)
        assert rgb_u8.dtype == np.uint8
        self.last_rgb = rgb_u8
        self.last_tokens = np.asarray(token_ids)
        if noise is None:
            noise = np.full(
                (self.action_horizon, self.action_dim),
                self.sampling_seed + self.sampling_draw,
                dtype=np.float32,
            )
            self.sampling_draw += 1
        self.last_noise = np.asarray(noise, dtype=np.float32).copy()
        return self.last_noise


class ConstTokenizer(PromptTokenizer):
    """A tokenizer that needs no SentencePiece model (bypasses __init__)."""

    def __init__(self, tokens=(1, 2, 3, 4)):
        self.max_token_len = 200
        self.discrete_state = False
        self._tokens = np.asarray(tokens, dtype=np.uint32)

    def __call__(self, prompt, state=None):
        return self._tokens


def make_quantile_unnormalizer():
    rng = np.random.default_rng(0)
    q01 = rng.uniform(-2.0, -0.5, size=LIBERO_DIM).astype(np.float32)
    q99 = rng.uniform(0.5, 2.0, size=LIBERO_DIM).astype(np.float32)
    return Unnormalizer(q01=q01, q99=q99)


def make_obs():
    rng = np.random.default_rng(1)
    return {
        "observation/image": rng.integers(0, 256, size=(256, 320, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 256, size=(240, 240, 3), dtype=np.uint8),
        "observation/state": rng.uniform(-1, 1, size=8).astype(np.float32),
        "prompt": "pick up the block",
    }


def make_parts(*, image_pipeline=None, image_keys=DEFAULT_KEYS, unnormalizer=None):
    """Build (model, input_pipeline, output_pipeline) via the default factory."""
    model = MockModel()
    in_pipe, out_pipe = Pi05Policy.default_pipelines(
        model,
        tokenizer=ConstTokenizer(),
        unnormalizer=unnormalizer or make_quantile_unnormalizer(),
        image_pipeline=image_pipeline,
        image_keys=image_keys,
    )
    return model, in_pipe, out_pipe


def build_policy(*, image_pipeline=None, image_keys=DEFAULT_KEYS, unnormalizer=None,
                 input_pipeline=None, output_pipeline=None):
    model, in_pipe, out_pipe = make_parts(
        image_pipeline=image_pipeline, image_keys=image_keys, unnormalizer=unnormalizer
    )
    return Pi05Policy(
        model,
        input_pipeline=input_pipeline or in_pipe,
        output_pipeline=output_pipeline or out_pipe,
        image_keys=image_keys,
    )


def test_infer_returns_unnormalized_numpy():
    policy = build_policy()
    result = policy.infer(make_obs())
    actions = result["actions"]
    assert isinstance(actions, np.ndarray)
    assert actions.dtype == np.float32
    assert actions.shape == (HORIZON, LIBERO_DIM)


def test_missing_external_noise_uses_internal_sampling():
    policy = build_policy()
    first = policy.infer(make_obs())
    second = policy.infer(make_obs())
    assert first["noise"] is None
    assert second["noise"] is None
    assert not np.array_equal(first["normalized_actions"], second["normalized_actions"])


def test_explicit_noise_is_forwarded_exactly():
    policy = build_policy()
    noise = np.arange(HORIZON * MODEL_DIM, dtype=np.float32).reshape(HORIZON, MODEL_DIM)
    result = policy.infer(make_obs(), noise=noise)
    np.testing.assert_array_equal(result["noise"], noise)
    np.testing.assert_array_equal(result["normalized_actions"], noise)
    np.testing.assert_array_equal(policy.model.last_noise, noise)


def test_explicit_noise_does_not_advance_internal_stream():
    policy = build_policy()
    first = policy.infer(make_obs())["normalized_actions"]
    policy.infer(make_obs(), noise=np.full((HORIZON, MODEL_DIM), 99.0, dtype=np.float32))
    second = policy.infer(make_obs())["normalized_actions"]
    np.testing.assert_array_equal(first, np.zeros((HORIZON, MODEL_DIM), dtype=np.float32))
    np.testing.assert_array_equal(second, np.ones((HORIZON, MODEL_DIM), dtype=np.float32))


def test_observation_noise_is_supported_and_keyword_wins():
    policy = build_policy()
    observation_noise = np.ones((HORIZON, MODEL_DIM), dtype=np.float32)
    keyword_noise = np.full((HORIZON, MODEL_DIM), 2.0, dtype=np.float32)
    obs = make_obs()
    obs["noise"] = observation_noise
    np.testing.assert_array_equal(policy.infer(obs)["normalized_actions"], observation_noise)
    np.testing.assert_array_equal(
        policy.infer(obs, noise=keyword_noise)["normalized_actions"], keyword_noise
    )


def test_invalid_external_noise_shape_is_rejected():
    with pytest.raises(ValueError, match="noise shape"):
        build_policy().infer(make_obs(), noise=np.zeros((HORIZON, MODEL_DIM - 1)))


def test_layering_invariant_l2_minus_unnormalize_equals_l1():
    """L2's action == unnormalize(model output[:, :dim]); the model output is L1."""
    policy = build_policy()
    result = policy.infer(make_obs())
    normalized = result["normalized_actions"]  # == L1 (bare-model normalized output)
    unnormalizer = policy.output_pipeline["unnormalize"].unnormalizer
    recomputed = unnormalizer(np.ascontiguousarray(normalized[:, :LIBERO_DIM]))
    np.testing.assert_array_equal(result["actions"], recomputed)


def test_identity_unnormalizer_passes_normalized_through():
    zeros = np.zeros(LIBERO_DIM, dtype=np.float32)
    ones = np.ones(LIBERO_DIM, dtype=np.float32)
    policy = build_policy(unnormalizer=Unnormalizer(mean=zeros, std=ones, mode="mean_std"))
    result = policy.infer(make_obs())
    np.testing.assert_array_equal(result["actions"], result["normalized_actions"][:, :LIBERO_DIM])


def test_timing_distinguishes_model_and_end_to_end():
    result = build_policy().infer(make_obs())
    timing = result["timing"]
    assert "model_ms" in timing and "total_ms" in timing
    assert timing["total_ms"] >= timing["model_ms"] >= 0.0


def test_pluggable_step_replacement_is_isolated():
    """A custom resize step replaces the default without disturbing tokenize/noise."""
    calls = {"n": 0}

    class CountingResize(ResizeWithPad):
        def __call__(self, image):
            calls["n"] += 1
            return super().__call__(image)

    pipeline = Pipeline([("parse", ParseImage()), ("resize", CountingResize(IMAGE_SIZE))])
    policy = build_policy(image_pipeline=pipeline)
    policy.infer(make_obs())
    assert calls["n"] == NUM_VIEWS  # one per view, default steps otherwise intact


def test_too_many_views_raises():
    with pytest.raises(ValueError):
        # 3 keys for a 2-view model: the key set must match the checkpoint's cameras.
        build_policy(image_keys=("observation/image", "observation/wrist_image", "observation/extra"))


def test_fewer_views_raises():
    # A 2-view model driven with a single camera is a contract error: real views
    # only, no padding — the caller must supply exactly the checkpoint's cameras.
    with pytest.raises(ValueError):
        build_policy(image_keys=("observation/image",))


def test_missing_key_raises():
    policy = build_policy()
    obs = make_obs()
    del obs["prompt"]
    with pytest.raises(KeyError):
        policy.infer(obs)


def test_satisfies_policy_protocol():
    """Pi05Policy structurally satisfies the shared Policy contract."""
    from apxinf import Policy

    policy = build_policy()
    assert isinstance(policy, Policy)
    assert policy.action_dim == LIBERO_DIM
    assert policy.action_horizon == HORIZON
    result = policy.infer(make_obs())
    # The two guaranteed keys of the contract.
    assert "actions" in result and "timing" in result


# --- pipeline reorder / insert / replace ----------------------------------


def test_metadata_reports_pipeline_step_names():
    policy = build_policy()
    assert policy.metadata["input_pipeline"] == ["image_stack", "tokenize"]
    assert policy.metadata["output_pipeline"] == ["trim", "unnormalize"]


def test_reorder_independent_pre_steps_is_identical():
    """Image stacking and tokenization are independent pre-processing steps."""
    obs = make_obs()
    baseline = build_policy().infer(obs)["actions"]

    model, in_pipe, out_pipe = make_parts()
    reordered = in_pipe.reorder(["tokenize", "image_stack"])
    policy = Pi05Policy(model, input_pipeline=reordered, output_pipeline=out_pipe)
    np.testing.assert_array_equal(policy.infer(obs)["actions"], baseline)


def test_insert_passthrough_step_does_not_change_result():
    """A no-op dict->dict step inserted into the pre chain leaves actions unchanged."""

    class Passthrough(ProcessorStep):
        def __call__(self, data):
            return data

    obs = make_obs()
    baseline = build_policy().infer(obs)["actions"]

    model, in_pipe, out_pipe = make_parts()
    injected = in_pipe.insert_after("image_stack", ("noop", Passthrough()))
    policy = Pi05Policy(model, input_pipeline=injected, output_pipeline=out_pipe)
    assert policy.metadata["input_pipeline"] == ["image_stack", "noop", "tokenize"]
    np.testing.assert_array_equal(policy.infer(obs)["actions"], baseline)


def test_explicit_host_sampler_remains_pluggable():
    model = MockModel()
    in_pipe, out_pipe = Pi05Policy.default_pipelines(
        model,
        tokenizer=ConstTokenizer(),
        unnormalizer=make_quantile_unnormalizer(),
        noise=GaussianNoise(HORIZON, MODEL_DIM, seed=7),
    )
    policy = Pi05Policy(model, input_pipeline=in_pipe, output_pipeline=out_pipe)
    assert policy.metadata["input_pipeline"] == ["image_stack", "tokenize", "sample_noise"]
    result = policy.infer(make_obs())
    assert result["noise"] is not None
    np.testing.assert_array_equal(result["normalized_actions"], result["noise"])


def test_default_vs_explicitly_built_pipeline_are_bit_identical():
    """The default path and a hand-built equivalent produce identical numerics."""
    obs = make_obs()
    default = build_policy().infer(obs)

    model, in_pipe, out_pipe = make_parts()
    explicit = Pi05Policy(model, input_pipeline=in_pipe, output_pipeline=out_pipe).infer(obs)
    np.testing.assert_array_equal(default["actions"], explicit["actions"])
    np.testing.assert_array_equal(default["normalized_actions"], explicit["normalized_actions"])
    np.testing.assert_array_equal(default["token_ids"], explicit["token_ids"])


# --- gated real-model cross-check -----------------------------------------


def test_real_model_layering(model_dir):
    apxinf_py = pytest.importorskip("apxinf_py")
    from apxinf import AutoPolicy

    precision = os.environ.get("APXINF_PI05_PRECISION", "bf16")
    try:
        # AutoPolicy reads config.json (type="pi05") and dispatches to Pi05Policy,
        # exercising the real registry path end to end.
        policy = AutoPolicy.from_pretrained(
            model_dir,
            device=os.environ.get("APXINF_PI05_DEVICE", "cuda:0"),
            precision=precision,
            action_dim=LIBERO_DIM,
            seed=0,
        )
    except Exception as error:  # noqa: BLE001 - load needs CUDA build + checkpoint
        pytest.skip(f"pi05 load failed: {error}")

    assert isinstance(policy, Pi05Policy)
    obs = make_obs()
    noise = np.random.default_rng(0).standard_normal((HORIZON, MODEL_DIM), dtype=np.float32)
    result = policy.infer(obs, noise=noise)
    normalized = result["normalized_actions"]

    # L1: feed identical preprocessed inputs straight to the binding. The pre-chain's
    # ImageStack step holds the same image_pipeline / real-view key set the policy used
    # (real views only — no padding).
    image_stack = policy.input_pipeline["image_stack"]
    views = [image_stack.image_pipeline(obs[key]) for key in image_stack.image_keys]
    rgb = np.ascontiguousarray(np.stack(views), dtype=np.uint8)
    l1 = policy.model.infer_rgb(rgb, "nhwc", result["token_ids"], result["noise"])
    np.testing.assert_allclose(normalized, l1, rtol=0.0, atol=2e-3)

    # L2 minus unnormalize reproduces L1's first action_dim columns.
    unnormalizer = policy.output_pipeline["unnormalize"].unnormalizer
    recomputed = unnormalizer(np.ascontiguousarray(l1[:, :LIBERO_DIM]))
    np.testing.assert_allclose(result["actions"], recomputed, rtol=0.0, atol=2e-3)
