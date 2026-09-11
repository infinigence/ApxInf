"""π0-FAST L2 policy: prompt assembly, FAST detokenization, and checkpoint load.

The 5.8 GB LIBERO checkpoint and its two hub tokenizers are not test fixtures, so
the plumbing is exercised with a fabricated checkpoint directory and a fabricated
local hub cache: a miniature `tokenizer.json` for each side, hand-written
`safetensors` state files, and a stub model that plays back known action tokens.
Real-tokenizer numerics were validated against a captured LeRobot trace.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from apxinf.processors import ImageStack, ParseImage, Pipeline, ResizeWithPad
from apxinf.processors.tokenize import discretize_state
from apxinf.policies.impls.pi0fast import (
    Pi0FastPolicy,
    _hf_cache_matches,
    detokenize_action_tokens,
)

_STATE_MEAN = np.arange(8, dtype=np.float32) * 0.1
_STATE_STD = np.full(8, 0.5, dtype=np.float32)
_ACTION_MEAN = np.arange(7, dtype=np.float32) * 0.01
_ACTION_STD = np.full(7, 0.25, dtype=np.float32)

#: Stub vocabulary wide enough for the action-id offset to stay positive.
_VOCAB_SIZE = 1024
_FAST_SKIP_TOKENS = 128


def _action_token(coefficient: int) -> int:
    """PaliGemma token a FAST coefficient is encoded as, on the reference's math."""
    return _VOCAB_SIZE - 1 - _FAST_SKIP_TOKENS - (coefficient + 203)


class _StubPaligemma:
    """Minimal id map: `<bos>`, `|`, the `Action: ` preamble, and dummy pieces."""

    def __init__(self, *, max_length=200):
        self.max_length = int(max_length)
        self.vocab_size = _VOCAB_SIZE
        self.bos_token_id = 1
        self.pipe_token_id = 2
        self.action_prefix_ids = [3, 4]
        self.texts = []

    def encode(self, text):
        self.texts.append(text)
        return [5, 6]

    def prompt_ids(self, text):
        prompt = ([self.bos_token_id] + self.encode(text))[: self.max_length - 1]
        return prompt + [self.bos_token_id]


class _StubFast:
    """``decode`` returns one character per action id, offset by ``min_token``."""

    min_token = -203
    scale = 10.0

    def __init__(self):
        self.calls = []

    def decode(self, token_ids):
        self.calls.append(list(token_ids))
        # The real FAST BPE decodes action id ``t`` to ``chr(t)`` for t < 256.
        return "".join(chr(int(token)) for token in token_ids)


class _StubModel:
    image_size = 224
    num_views = 2

    def __init__(self, tokens):
        self.tokens = np.asarray(tokens, dtype=np.uint32)
        self.calls = []

    def infer_action_tokens_rgb(self, rgb, layout, token_ids, stop_token=None):
        self.calls.append((rgb, layout, np.asarray(token_ids), stop_token))
        return self.tokens


def _image_pipeline():
    return Pipeline(
        [
            (
                "image_stack",
                ImageStack(
                    Pipeline([("parse", ParseImage()), ("resize", ResizeWithPad(224))]),
                    ("observation/image", "observation/wrist_image"),
                    224,
                ),
            )
        ]
    )


def _build_policy(tokens, *, horizon=10, dim=7, tokenizer=None, max_action_tokens=256):
    return Pi0FastPolicy(
        _StubModel(tokens),
        tokenizer=tokenizer or _StubPaligemma(),
        fast_tokenizer=_StubFast(),
        state_mean=_STATE_MEAN,
        state_std=_STATE_STD,
        action_mean=_ACTION_MEAN,
        action_std=_ACTION_STD,
        image_pipeline=_image_pipeline(),
        image_keys=("observation/image", "observation/wrist_image"),
        state_key="observation/state",
        prompt_key="prompt",
        action_dim=dim,
        action_horizon=horizon,
        fast_skip_tokens=_FAST_SKIP_TOKENS,
        max_state_dim=32,
        max_action_tokens=max_action_tokens,
    )


def _observation(state=None):
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    return {
        "observation/image": image,
        "observation/wrist_image": image + 1,
        "observation/state": (
            np.asarray(state, dtype=np.float32) if state is not None else np.zeros(8, np.float32)
        ),
        "prompt": "put_the pot on the stove",
    }


def _expected_bins(state):
    normalized = (np.asarray(state, np.float32) - _STATE_MEAN) / (_STATE_STD + 1e-8)
    padded = np.zeros(32, dtype=np.float32)
    padded[: normalized.size] = normalized
    return " ".join(str(int(value)) for value in discretize_state(padded))


def test_prompt_uses_the_reference_template_and_discretized_padded_state():
    tokenizer = _StubPaligemma()
    policy = _build_policy([], tokenizer=tokenizer)

    token_ids = policy._prompt_ids(_observation(), "put_the pot on the stove")

    assert tokenizer.texts == [
        f"Task: put the pot on the stove, State: {_expected_bins(np.zeros(8, np.float32))};\n"
    ]
    np.testing.assert_array_equal(token_ids, np.array([1, 5, 6, 1], dtype=np.uint32))


def test_prompt_ends_with_the_bos_that_triggers_decoding():
    """The runtime embeds the ids it is handed and appends nothing itself.

    LeRobot's ``sample_actions_fast_kv_cache`` concatenates a trailing BOS to the
    (right-padded) prompt before prefill. Without it the model continues the
    prompt text instead of emitting ``Action: …|`` — which is exactly what a real
    LIBERO frame did before this was fixed, so pin the id here.
    """
    tokenizer = _StubPaligemma()
    policy = _build_policy([], tokenizer=tokenizer)

    token_ids = policy._prompt_ids(_observation(), "task")

    assert token_ids[0] == tokenizer.bos_token_id
    assert token_ids[-1] == tokenizer.bos_token_id
    assert list(token_ids).count(tokenizer.bos_token_id) == 2


def test_prompt_keeps_the_trailing_bos_inside_the_token_budget():
    tokenizer = _StubPaligemma(max_length=4)
    policy = _build_policy([], tokenizer=tokenizer)

    token_ids = policy._prompt_ids(_observation(), "task")

    assert token_ids.size == 4
    assert token_ids[-1] == tokenizer.bos_token_id


def test_prompt_rejects_a_state_of_the_wrong_width():
    policy = _build_policy([])

    with pytest.raises(ValueError, match="was trained on 8"):
        policy._prompt_ids(_observation(np.zeros(7, np.float32)), "task")


def test_detokenize_strips_preamble_and_everything_after_the_terminator():
    coefficients = (np.arange(70) - 30).astype(np.float64)
    stream = (
        [3, 4]
        + [_action_token(int(value)) for value in coefficients]
        + [2, 7, 8]
    )

    decoded = detokenize_action_tokens(
        stream,
        tokenizer=_StubPaligemma(),
        fast_tokenizer=_StubFast(),
        fast_skip_tokens=_FAST_SKIP_TOKENS,
        action_horizon=10,
        action_dim=7,
    )

    np.testing.assert_allclose(
        decoded, _idct(coefficients.reshape(10, 7)), rtol=0, atol=1e-9
    )


def test_detokenize_pads_a_short_stream_to_a_full_chunk():
    coefficients = np.arange(20).astype(np.float64)

    decoded = detokenize_action_tokens(
        [_action_token(int(value)) for value in coefficients],
        tokenizer=_StubPaligemma(),
        fast_tokenizer=_StubFast(),
        fast_skip_tokens=_FAST_SKIP_TOKENS,
        action_horizon=10,
        action_dim=2,
    )

    padded = np.zeros(20, dtype=np.float64)
    padded[:20] = coefficients
    np.testing.assert_allclose(decoded, _idct(padded.reshape(10, 2)), atol=1e-9)


def test_infer_returns_detokenized_actions_and_the_raw_tokens():
    coefficients = (np.arange(70) - 30).astype(np.float64)
    stream = [3, 4] + [_action_token(int(value)) for value in coefficients]
    policy = _build_policy(stream, max_action_tokens=len(stream))

    result = policy.infer(_observation())

    expected = _idct(coefficients.reshape(10, 7))
    np.testing.assert_allclose(result["normalized_actions"], expected, atol=1e-6)
    np.testing.assert_allclose(result["actions"], expected * _ACTION_STD + _ACTION_MEAN, atol=1e-6)
    np.testing.assert_array_equal(result["action_tokens"], np.asarray(stream))
    np.testing.assert_array_equal(result["prompt_token_ids"], np.array([1, 5, 6, 1], np.uint32))
    assert result["actions"].shape == (10, 7)
    assert set(result["timing"]) == {"model_ms", "total_ms"}
    assert policy.action_dim == 7 and policy.action_horizon == 10
    assert policy.metadata["state_dim"] == 8
    assert policy.metadata["discrete_state"] is True
    rgb = policy.model.calls[0][0]
    assert policy.model.calls[0][1] == "nhwc"
    assert rgb.shape == (2, 224, 224, 3) and rgb.dtype == np.uint8


def test_infer_refuses_a_continuous_latent():
    policy = _build_policy([2], max_action_tokens=1)

    with pytest.raises(ValueError, match="no continuous latent"):
        policy.infer(_observation(), noise=np.zeros((10, 7), np.float32))


def test_infer_reports_the_keys_it_consumes():
    policy = _build_policy([2], max_action_tokens=1)

    with pytest.raises(KeyError, match="prompt"):
        policy.infer({"observation/image": np.zeros((4, 4, 3), np.uint8)})


def test_from_pretrained_rejects_options_that_cannot_mean_anything_here(tmp_path):
    for kwargs, error, match in (
        ({"norm_stats": tmp_path / "norm_stats.json"}, NotImplementedError, "norm_stats"),
        ({"calibration": tmp_path / "cal.json"}, NotImplementedError, "calibration"),
        ({"num_flow_steps": 10}, NotImplementedError, "flow-matching"),
        ({"discrete_state": False}, ValueError, "always discretizes"),
        ({"something_else": 1}, TypeError, "unsupported options"),
    ):
        with pytest.raises(error, match=match):
            Pi0FastPolicy.from_pretrained(tmp_path, **kwargs)


def test_pi0fast_from_pretrained_needs_a_checkpoint_config(tmp_path):
    with pytest.raises(FileNotFoundError, match="no config.json"):
        Pi0FastPolicy.from_pretrained(tmp_path, model=_StubModel([2]))


def test_hf_cache_lookup_is_newest_first(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    snapshots = tmp_path / "hub" / "models--org--name" / "snapshots"
    for revision in ("aaa", "bbb"):
        (snapshots / revision).mkdir(parents=True)
        (snapshots / revision / "tokenizer.json").write_text("{}")

    assert len(_hf_cache_matches("org/name", "tokenizer.json")) == 2
    assert _hf_cache_matches("org/missing", "tokenizer.json") == []


def test_from_pretrained_loads_a_checkpoint_directory(tmp_path, monkeypatch):
    pytest.importorskip("tokenizers")
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    model_dir = _write_checkpoint(tmp_path)
    coefficients = (np.arange(70) - 30).astype(np.float64)
    # The runtime always spends the full token budget: the terminator lands at 72
    # and the rest is decoded-but-unread tail, exactly as the real decoder emits it.
    stream = [3] + [_action_token(int(value)) for value in coefficients] + [2]
    model = _StubModel(stream + [1] * (256 - len(stream)))

    policy = Pi0FastPolicy.from_pretrained(
        model_dir,
        model=model,
        state_key="observation/state",
        image_keys=("observation/image", "observation/wrist_image"),
    )

    assert policy.metadata["action_dim"] == 7
    assert policy.metadata["action_horizon"] == 10
    assert policy.metadata["state_dim"] == 8
    assert policy.metadata["max_action_tokens"] == 256
    assert policy.metadata["image_keys"] == ["observation/image", "observation/wrist_image"]
    assert policy.tokenizer.vocab_size == _VOCAB_SIZE
    result = policy.infer(_observation())
    np.testing.assert_allclose(
        result["normalized_actions"], _idct(coefficients.reshape(10, 7)), atol=1e-6
    )


def _idct(coefficients):
    from scipy.fft import idct

    return idct(coefficients / 10.0, axis=0, norm="ortho")


def _paligemma_vocab():
    """`<pad> <bos> | "Action: "` then filler, so the id space is `_VOCAB_SIZE` wide."""
    vocab = {"<pad>": 0, "<bos>": 1, "|": 2, "Action: ": 3}
    vocab.update({f"c{index}": index for index in range(4, _VOCAB_SIZE)})
    return vocab


def _write_checkpoint(root):
    """A miniature LeRobot π0-FAST checkpoint, stats and tokenizer assets included."""
    model_dir = root / "ckpt"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "type": "pi0_fast",
                "chunk_size": 10,
                "n_action_steps": 10,
                "max_action_tokens": 256,
                "max_state_dim": 32,
                "tokenizer_max_length": 200,
                "empty_cameras": 1,
                "action_tokenizer_name": "test/fast-action",
                "text_tokenizer_name": "test/paligemma",
                "output_features": {"action": {"type": "ACTION", "shape": [7]}},
            }
        )
    )
    _write_safetensors(
        model_dir / "policy_preprocessor_step_2_normalizer_processor.safetensors",
        {"observation.state.mean": _STATE_MEAN, "observation.state.std": _STATE_STD},
    )
    _write_safetensors(
        model_dir / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        {"action.mean": _ACTION_MEAN, "action.std": _ACTION_STD},
    )
    assets = {
        "test/paligemma": {
            "tokenizer.json": {
                "version": "1.0",
                "added_tokens": [],
                "normalizer": None,
                "pre_tokenizer": None,
                "post_processor": None,
                "decoder": None,
                "model": {
                    "type": "WordLevel",
                    "vocab": _paligemma_vocab(),
                    "unk_token": "<pad>",
                },
            }
        },
        "test/fast-action": {
            "tokenizer.json": {
                "version": "1.0",
                "added_tokens": [],
                "normalizer": None,
                "pre_tokenizer": None,
                "post_processor": None,
                # ``Fuse`` keeps pieces adjacent, so decoding a full stream
                # yields one character per id — the same contract the real
                # ByteLevel decoder satisfies for this id range.
                "decoder": {"type": "Fuse"},
                "model": {
                    "type": "WordLevel",
                    "vocab": {chr(index): index for index in range(256)},
                    "unk_token": "\x00",
                },
            },
            "processor_config.json": {"min_token": -203, "scale": 10.0},
        },
    }
    for repo, files in assets.items():
        directory = root / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / "rev"
        directory.mkdir(parents=True)
        for name, payload in files.items():
            (directory / name).write_text(json.dumps(payload))
    return model_dir


def _write_safetensors(path, tensors):
    header = {}
    offset = 0
    payload = b""
    for name, values in tensors.items():
        array = np.asarray(values, dtype="<f4")
        raw = array.tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        offset += len(raw)
        payload += raw
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + payload)
