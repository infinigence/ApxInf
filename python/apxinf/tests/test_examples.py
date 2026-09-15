from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load_example(name: str):
    sys.path.insert(0, str(EXAMPLES))
    try:
        spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXAMPLES))


class _PolicyContractOnly:
    metadata = {
        "image_keys": ["front", "wrist"],
        "state_key": "observation/state",
        "state_dim": 8,
    }
    action_dim = 7


def test_autopolicy_example_uses_public_metadata_contract():
    example = _load_example("autopolicy_infer")
    observation = example.synthetic_observation_for(_PolicyContractOnly())

    assert set(observation) == {"front", "wrist", "observation/state", "prompt"}
    assert observation["front"].shape == (256, 256, 3)
    assert observation["front"].dtype == np.uint8
    assert observation["observation/state"].shape == (8,)


@pytest.mark.parametrize("image_keys", [None, [], ["front", ""]])
def test_autopolicy_example_rejects_missing_or_invalid_image_contract(image_keys):
    example = _load_example("autopolicy_infer")

    class BadPolicy:
        metadata = {} if image_keys is None else {"image_keys": image_keys}
        action_dim = 7

    with pytest.raises(ValueError, match="image_keys"):
        example.synthetic_observation_for(BadPolicy())


def test_autopolicy_example_requires_state_width_when_state_is_served():
    example = _load_example("autopolicy_infer")

    class BadPolicy:
        metadata = {"image_keys": ["front"], "state_key": "observation/state"}

    with pytest.raises(ValueError, match="state_dim"):
        example.synthetic_observation_for(BadPolicy())


def test_common_policy_options_accept_a_json_object():
    common = _load_example("_common")

    assert common.json_object('{"norm_key":"x2_normal","image_keys":["front","wrist"]}') == {
        "norm_key": "x2_normal",
        "image_keys": ["front", "wrist"],
    }


def test_common_policy_options_keep_model_knobs_and_protect_generic_flags():
    common = _load_example("_common")

    options = common.policy_kwargs(
        {
            "norm_key": "x2_normal",
            "device": "wrong",
            "metadata": {"deployment": "lab", "protocol": "wrong"},
        },
        device="cuda:1",
        precision="fp8",
        action_dim=7,
        metadata={"protocol": "openpi.websocket_policy"},
    )

    assert options == {
        "norm_key": "x2_normal",
        "device": "cuda:1",
        "precision": "fp8",
        "action_dim": 7,
        "metadata": {"deployment": "lab", "protocol": "openpi.websocket_policy"},
    }


@pytest.mark.parametrize("value", ["[]", '"walloss"', "not-json"])
def test_common_policy_options_reject_non_objects(value):
    common = _load_example("_common")

    with pytest.raises(argparse.ArgumentTypeError):
        common.json_object(value)


def test_websocket_example_passes_caller_named_wire_keys_through(monkeypatch, tmp_path):
    pytest.importorskip("websockets")
    pytest.importorskip("msgpack")
    example = _load_example("openpi_server")
    captured = {}

    class FakePolicy:
        def close(self):
            captured["closed"] = True

    class FakeServer:
        def __init__(self, policy, host, port):
            captured.update(policy=policy, host=host, port=port)

        def serve_forever(self):
            captured["served"] = True

    def fake_load(model_dir, **kwargs):
        captured.update(model_dir=model_dir, kwargs=kwargs)
        return FakePolicy()

    monkeypatch.setattr(
        example,
        "parse_args",
        lambda: argparse.Namespace(
            model_dir=tmp_path,
            precision="bf16",
            device="cuda:0",
            action_dim=0,
            policy_options={"norm_key": "libero_all"},
            image_keys="observation/image, observation/wrist_image",
            state_key="observation/state",
            host="127.0.0.1",
            port=8017,
        ),
    )
    monkeypatch.setattr(example.AutoPolicy, "from_pretrained", fake_load)
    monkeypatch.setattr(example, "WebsocketPolicyServer", FakeServer)

    example.main()

    assert captured["model_dir"] == tmp_path
    assert captured["kwargs"] == {
        "norm_key": "libero_all",
        "device": "cuda:0",
        "precision": "bf16",
        "image_keys": ("observation/image", "observation/wrist_image"),
        "state_key": "observation/state",
        "metadata": {
            "protocol": "openpi.websocket_policy",
            "precision": "bf16",
        },
    }
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8017
    assert captured["served"] is True
    assert captured["closed"] is True


def test_websocket_example_leaves_unnamed_keys_to_the_policy(monkeypatch, tmp_path):
    pytest.importorskip("websockets")
    pytest.importorskip("msgpack")
    example = _load_example("openpi_server")
    captured = {}

    class FakePolicy:
        def close(self):
            captured["closed"] = True

    class FakeServer:
        def __init__(self, policy, host, port):
            captured["policy"] = policy

        def serve_forever(self):
            captured["served"] = True

    def fake_load(model_dir, **kwargs):
        captured.update(model_dir=model_dir, kwargs=kwargs)
        return FakePolicy()

    monkeypatch.setattr(
        example,
        "parse_args",
        lambda: argparse.Namespace(
            model_dir=tmp_path,
            precision="bf16",
            device="cuda:0",
            action_dim=0,
            policy_options={},
            image_keys=None,
            state_key=None,
            host="127.0.0.1",
            port=8000,
        ),
    )
    monkeypatch.setattr(example.AutoPolicy, "from_pretrained", fake_load)
    monkeypatch.setattr(example, "WebsocketPolicyServer", FakeServer)

    example.main()

    assert captured["model_dir"] == tmp_path
    # Nothing invented on the caller's behalf: the policy's own fallback applies.
    assert "action_dim" not in captured["kwargs"]
    assert "image_keys" not in captured["kwargs"]
    assert "state_key" not in captured["kwargs"]
    assert captured["served"] is True
    assert captured["closed"] is True
