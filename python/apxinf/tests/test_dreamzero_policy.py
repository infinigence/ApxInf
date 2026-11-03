"""Public adapter semantics; device numerical acceptance is separate."""
import json
import numpy as np
import pytest

from apxinf.policies.impls.dreamzero import DreamZeroPolicy, _prompt, _round_bf16


class Tokenizer:
    def encode(self, text):
        return [len(text), 1] + [0] * 510, [1, 1] + [0] * 510


class Runner:
    noise_shape = (113408,)

    def __init__(self):
        self.calls = []
        self.resets = self.evictions = 0

    def _infer_preprocessed(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return np.zeros((24, 32), np.float32)

    def reset_history(self):
        self.resets += 1

    def clear_prepared(self):
        self.evictions += 1


def policy(runner=None):
    stats = {
        modality: {
            "joint_position": {"q01": [-2.] * 7, "q99": [2.] * 7},
            "gripper_position": {"q01": [0.], "q99": [1.]},
        } for modality in ("state", "action")
    }
    return DreamZeroPolicy(runner or Runner(), tokenizer=Tokenizer(), statistics=stats,
                           image_keys=("left", "right", "wrist"), images_preprocessed=True)


def observation(frames=1):
    wrist = np.zeros((frames, 176, 320, 3), np.uint8)
    wrist[:, :, :, 0] = np.arange(320, dtype=np.uint16) % 256
    return dict(left=np.full_like(wrist, 17), right=np.full_like(wrist, 251), wrist=wrist,
                **{"state.joint_position": np.ones(7, np.float32),
                   "state.gripper_position": np.array([.25], np.float32), "task": "Pick Red Cup"})


def test_joint_input_montage_state_and_relative_decode():
    p = policy()
    noise = np.linspace(-1, 1, 113408, dtype=np.float32)
    output = p.infer(observation(), noise=noise)
    args, identity = p.model_runner.calls[-1]
    pixels, grid, ids, mask, state, embodiment, actual_noise = args
    assert pixels.shape == (352 * 640, 3)
    assert grid.tolist() == [[1, 352, 640]]
    assert ids.shape == mask.shape == (1024,)
    assert embodiment == 17
    np.testing.assert_array_equal(actual_noise, noise)
    np.testing.assert_array_equal(state[0, :8], [.5] * 7 + [-.5])
    assert np.count_nonzero(state[0, 8:]) == 0
    image = pixels.reshape(352, 640, 3)
    np.testing.assert_array_equal(image[0, ::2], image[0, 1::2])
    expected_left = _round_bf16((_round_bf16(np.array([17 / 255], np.float32)) - .5)) * 2
    np.testing.assert_array_equal(image[176:, :320], np.full((176, 320, 3), expected_left[0]))
    np.testing.assert_array_equal(output["actions"], np.tile([1.] * 7 + [.5], (24, 1)))
    assert identity["history_session"] and identity["history_episode"]


def test_history_identity_reset_eviction_and_generated_noise_are_distinct():
    p, other = policy(), policy()
    p.infer(observation())
    first = p.model_runner.calls[-1][1].copy()
    p.clear_prepared()
    p.infer(observation(4))
    assert p.model_runner.calls[-1][1] == first
    assert p.model_runner.calls[-1][0][-1] is None
    p.reset_history("next")
    p.infer(observation(9))
    assert p.model_runner.calls[-1][1]["history_session"] == first["history_session"]
    assert p.model_runner.calls[-1][1]["history_episode"] == "next"
    p.infer(observation(4), episode="third")
    assert p.model_runner.calls[-1][1]["history_episode"] == "third"
    other.infer(observation())
    assert other.model_runner.calls[-1][1]["history_session"] != first["history_session"]
    assert p.model_runner.resets == p.model_runner.evictions == 1


@pytest.mark.parametrize("noise", [np.zeros((24, 32)), np.zeros((1, 113408)), np.full(113408, np.nan)])
def test_bad_joint_noise_fails_before_native_call(noise):
    p = policy()
    with pytest.raises(ValueError, match="noise"):
        p.infer(observation(), noise=noise)
    assert not p.model_runner.calls


@pytest.mark.parametrize("frames", [0, 2, 3, 8, 10])
def test_unsupported_history_shapes_fail_before_native(frames):
    p = policy()
    with pytest.raises(ValueError):
        p.infer(observation(frames))
    assert not p.model_runner.calls


def test_prompt_literal_and_case_match_reference_contract():
    assert _prompt("['Pick Red Cup', 'unused']") == _prompt("Pick Red Cup")
    assert _prompt("Pick Red Cup").endswith("The robot pick red cup")


def test_checkpoint_dispatch_uses_official_head_not_generic_vla(tmp_path):
    from apxinf.policies.auto import _read_model_type
    document = {"model_type": "vla", "action_head_cfg": {"_target_":
        "groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf.WANPolicyHead"}}
    (tmp_path / "config.json").write_text(json.dumps(document))
    assert _read_model_type(tmp_path) == "dreamzero"
    document["action_head_cfg"]["_target_"] = "different_family.Head"
    (tmp_path / "config.json").write_text(json.dumps(document))
    assert _read_model_type(tmp_path) == "vla"


@pytest.mark.parametrize("head", [None, [], "different_family.Head", 7])
def test_non_mapping_action_head_preserves_other_model_dispatch(tmp_path, head):
    from apxinf.policies.auto import _read_model_type

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "pi05", "action_head_cfg": head})
    )
    assert _read_model_type(tmp_path) == "pi05"


def test_bf16_rounding_matches_ties_to_even():
    x = np.array([1.00390625, 1.01171875, -1.00390625, -1.01171875], np.float32)
    np.testing.assert_array_equal(_round_bf16(x), [1., 1.015625, -1., -1.015625])


def test_public_policy_protocol_and_close_release_owned_runner():
    import weakref
    from apxinf.policies.base import Policy

    p = policy()
    assert isinstance(p, Policy)
    assert (p.action_horizon, p.action_dim) == (24, 8)
    runner = weakref.ref(p.model_runner)
    p.close()
    p.close()
    assert runner() is None
    with pytest.raises(RuntimeError, match="closed"):
        p.infer(observation())
