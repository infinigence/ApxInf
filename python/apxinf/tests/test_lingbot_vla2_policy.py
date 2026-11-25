"""LingBot-VLA 2.0 policy contract and tokenisation. Offline: no GPU, no ``apxinf_py``.

Always-run contract tests use a mock native runner and a synthetic tokenizer. The golden-token tests
need the real ``tokenizer.json`` (``LINGBOT_MODEL_DIR``), the reference tensors
(``LINGBOT_GOLDEN_DIR``) and the Qwen3-VL chat template (``LINGBOT_QWEN3VL_CFG``) and skip without them. Parity against the upstream package lives in
``test_lingbot_vla2_parity.py``.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from apxinf.policies import AutoPolicy, get_policy
from apxinf.policies.impls import lingbot_vla2 as lv

from _lingbot_vla2_support import (  # noqa: F401
    DIM, GOLDEN, HORIZON, MODEL_DIR, QWEN3VL_CFG, RAW_KEYS, UPSTREAM, Runner, Tokenizer, make_policy,
    robotwin_observation, stats_dict,
)


# ------------------------------------------------------------------------------------------
# contract (always runs)
# ------------------------------------------------------------------------------------------


def test_registered_and_not_auto_detected():
    assert get_policy("lingbot_vla2") is lv.LingbotVla2Policy
    assert get_policy("LingBot_VLA2") is lv.LingbotVla2Policy


def test_autopolicy_dispatch_needs_explicit_model_type(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text('{"vlm_family": "qwen3_vl"}')
    with pytest.raises(KeyError):
        AutoPolicy.from_pretrained(tmp_path)
    seen = {}
    monkeypatch.setattr(lv.LingbotVla2Policy, "from_pretrained",
                        classmethod(lambda cls, d, **kw: seen.update(dir=d, kw=kw) or "ok"))
    assert AutoPolicy.from_pretrained(tmp_path, model_type="lingbot_vla2", robot="robotwin") == "ok"
    assert seen["kw"] == {"robot": "robotwin"}


def test_runner_call_contract():
    rng = np.random.default_rng(0)
    runner = Runner()
    policy = make_policy(runner=runner)
    result = policy.infer(robotwin_observation(rng), noise=np.zeros((HORIZON, DIM), np.float32))
    call = runner.calls[0]
    assert call.pixels.shape == (768, 1536) and call.pixels.dtype == np.float32 and call.pixels.flags.c_contiguous
    assert call.grids.dtype == np.uint32 and call.grids.tolist() == [[1, 16, 16]] * 3
    assert call.tokens.dtype == np.uint32 and call.mask.dtype == np.uint8
    assert call.tokens.ndim == 1 and (call.mask == 1).all() and call.mask.shape == call.tokens.shape
    assert call.state.shape == (1, 55) and call.state.dtype == np.float32
    assert call.embodiment is None and call.noise.shape == (HORIZON, DIM) and call.noise.dtype == np.float32
    assert result["actions"].shape == (HORIZON, 14) and result["actions"].dtype == np.float32
    assert set(result["timing"]) >= {"model_ms", "total_ms"}
    assert policy.action_dim == 14 and policy.action_horizon == HORIZON


def test_noise_is_rounded_to_bf16_and_validated():
    rng = np.random.default_rng(1)
    runner = Runner()
    policy = make_policy(runner=runner)
    noise = rng.normal(size=(HORIZON, DIM)).astype(np.float32)
    policy.infer(robotwin_observation(rng), noise=noise)
    sent = runner.calls[0].noise
    assert (sent.view(np.uint32) & 0xFFFF == 0).all()
    np.testing.assert_allclose(sent, noise, rtol=2 ** -8)
    with pytest.raises(ValueError, match="noise has shape"):
        policy.infer(robotwin_observation(rng), noise=np.zeros((HORIZON, DIM + 1), np.float32))
    bad = noise.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        policy.infer(robotwin_observation(rng), noise=bad)
    policy.infer(robotwin_observation(rng))  # sampled internally, still BF16
    assert (runner.calls[-1].noise.view(np.uint32) & 0xFFFF == 0).all()


def test_seeded_noise_is_reproducible():
    rng = np.random.default_rng(2)
    a, b = Runner(), Runner()
    make_policy(runner=a, seed=7).infer(robotwin_observation(rng))
    make_policy(runner=b, seed=7).infer(robotwin_observation(np.random.default_rng(2)))
    np.testing.assert_array_equal(a.calls[0].noise, b.calls[0].noise)


def test_missing_inputs_fail_loudly():
    rng = np.random.default_rng(3)
    policy = make_policy()
    obs = robotwin_observation(rng)
    del obs[RAW_KEYS["cam_left"]]
    with pytest.raises(KeyError, match="cam_left_wrist"):
        policy.infer(obs)
    obs = robotwin_observation(rng)
    del obs["observation.state"]
    with pytest.raises(KeyError, match="observation.state"):
        policy.infer(obs)
    obs = robotwin_observation(rng)
    del obs["prompt"]
    with pytest.raises(KeyError, match="prompt"):
        policy.infer(obs)
    with pytest.raises(ValueError, match="missing views are not supported"):
        make_policy(cameras=("camera_top", "camera_wrist_left", "camera_head"))


def test_norm_type_is_required_and_validated():
    with pytest.raises(ValueError, match="needs norm_type"):
        lv.LingbotVla2Policy(Runner(), robot="robotwin", tokenizer=Tokenizer(), norm_stats=stats_dict())
    with pytest.raises(ValueError, match="unknown norm type"):
        make_policy(norm="bogus")
    with pytest.raises(NotImplementedError, match="sincos"):
        make_policy(norm="sincos")
    with pytest.raises(ValueError, match="needs norm_stats"):
        lv.LingbotVla2Policy(Runner(), robot="robotwin", tokenizer=Tokenizer(), norm_type="meanstd")
    # identity needs no statistics, but then the real width of a whole raw key must be given
    with pytest.raises(ValueError, match="joint_dims"):
        lv.LingbotVla2Policy(Runner(), robot="agilex_cobot_magic", tokenizer=Tokenizer(), norm_type="identity")
    policy = lv.LingbotVla2Policy(Runner(), robot="agilex_cobot_magic", tokenizer=Tokenizer(), norm_type="identity",
                                  joints=[("arm.position", 14), ("effector.position", 2)],
                                  joint_dims={"arm.position": 7, "effector.position": 1})
    assert policy.action_dim == 8


def test_prompt_is_truncated_and_chat_wrapped():
    rng = np.random.default_rng(4)
    runner = Runner()
    policy = make_policy(runner=runner, tokenizer_max_length=20)
    policy.infer(robotwin_observation(rng, "x" * 100))
    assert runner.calls[0].tokens.shape == (20,)
    policy.infer(robotwin_observation(rng, "hi"))
    expected = Tokenizer().encode("<|im_start|>user\nhi<|im_end|>\n")[:20]
    assert runner.calls[1].tokens.tolist() == expected
    short = make_policy(runner=runner)
    short.infer(robotwin_observation(rng, "hi"))
    assert runner.calls[2].tokens.tolist() == Tokenizer().encode("<|im_start|>user\nhi<|im_end|>\n")


def test_from_pretrained_wires_the_native_runner(tmp_path, monkeypatch):
    (tmp_path / "tokenizer.json").write_text("{}")
    captured = {}

    class Loaded(Runner):
        pass

    def load(model, path, **kwargs):
        captured.update(model=model, path=path, **kwargs)
        return Loaded()

    monkeypatch.setitem(sys.modules, "apxinf_py", SimpleNamespace(ModelRunner=SimpleNamespace(load=load)))
    monkeypatch.setattr(lv, "_Tokenizer", lambda d: Tokenizer())
    cli = tmp_path / "lingbotvla_cli.yaml"
    cli.write_text(
        "data:\n  cameras: [camera_top, camera_wrist_left, camera_wrist_right]\n  img_size: 256\n"
        "  joints: [\"{'arm.position': 14}\", \"{'end.position': 14}\", \"{'effector.position': 2}\"]\n"
        "  norm_type: [\"{'arm.position': 'bounds_99_woclip'}\", \"{'end.position': 'bounds_99_woclip'}\","
        " \"{'effector.position': 'bounds_99_woclip'}\"]\n"
        "train:\n  max_state_dim: 55\n  tokenizer_max_length: 72\n")
    policy = AutoPolicy.from_pretrained(tmp_path, model_type="lingbot_vla2", robot="robotwin",
                                        norm_stats=stats_dict(), model_variant="int8_convrot")
    assert captured["model"] == "lingbot_vla2" and captured["model_variant"] == "int8_convrot"
    assert "precision" not in captured
    assert policy.joints == [("arm.position", 14), ("end.position", 14), ("effector.position", 2)]
    assert policy._norm["arm.position"] == "bounds_99_woclip"
    with pytest.raises(TypeError, match="precision"):
        lv.LingbotVla2Policy.from_pretrained(tmp_path, robot="robotwin", precision="bf16")


def test_state_layout_places_groups_in_joint_order():
    rng = np.random.default_rng(5)
    policy = make_policy(norm="identity", stats={"norm_stats": {}}, runner=Runner())
    obs = robotwin_observation(rng)
    state = obs["observation.state"]
    encoded = policy.encode(obs)["state"][0]
    assert encoded.shape == (55,)
    np.testing.assert_allclose(encoded[:6], state[0:6], rtol=1e-6)       # left arm joints
    np.testing.assert_allclose(encoded[6:12], state[7:13], rtol=1e-6)    # right arm joints
    assert (encoded[12:14] == 0).all()                                   # arm slot padding (14 wide)
    assert (encoded[14:28] == 0).all()                                   # end.position absent -> zeros
    np.testing.assert_allclose(encoded[28:30], state[[6, 13]], rtol=1e-6)  # grippers
    assert (encoded[30:] == 0).all()


def test_default_quaternion_relative_type_on_a_joint_group_is_rejected_up_front():
    robot = {"states": [{"observation.state.arm.position": {"origin_keys": "s"}}],
             "actions": [{"action.arm.position": {"origin_keys": "a", "subtract_state": True}}],
             "images": ["observation.images.camera_top", "observation.images.camera_wrist_left",
                        "observation.images.camera_wrist_right"]}
    with pytest.raises(ValueError, match="relative_type"):
        lv.load_robot_config(robot)
    robot["actions"][0]["action.arm.position"]["relative_type"] = "joint"
    assert lv.load_robot_config(robot).actions["action.arm.position"].subtract_state


def test_subtract_state_adds_state_back():
    rng = np.random.default_rng(6)
    stats = {"norm_stats": {}}
    runner = Runner(lambda noise: np.full((HORIZON, DIM), 0.25, np.float32))
    policy = lv.LingbotVla2Policy(
        runner, robot="agilex_cobot_magic", tokenizer=Tokenizer(), norm_stats=stats, norm_type="identity",
        joints=[("arm.position", 14), ("effector.position", 2)],
        joint_dims={"arm.position": 7, "effector.position": 1})
    arm = rng.normal(size=7).astype(np.float32)
    eff = rng.normal(size=1).astype(np.float32)
    obs = {k: rng.integers(0, 256, (256, 256, 3), dtype=np.uint8) for k in
           ("observation.images.camera_top", "observation.images.camera_wrist_left",
            "observation.images.camera_wrist_right")}
    obs.update({"observation.state.arm.position": arm, "observation.state.effector.position": eff, "prompt": "go"})
    out = policy.infer(obs)["action_parts"]
    np.testing.assert_allclose(out["action.arm.position"], 0.25 + arm[None, :].repeat(HORIZON, 0), rtol=1e-6)
    np.testing.assert_allclose(out["action.effector.position"], np.full((HORIZON, 1), 0.25), rtol=1e-6)




# ------------------------------------------------------------------------------------------
# tokenisation against the golden tensors and the real tokenizer
# ------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_tokenizer():
    if not (MODEL_DIR / "tokenizer.json").is_file():
        pytest.skip(f"{MODEL_DIR}/tokenizer.json not found")
    return lv._Tokenizer(MODEL_DIR)


@pytest.mark.parametrize("tag", ["a", "b"])
def test_tokens_match_reference_golden(real_tokenizer, tag):
    safetensors = pytest.importorskip("safetensors")
    path = GOLDEN / f"golden_{tag}.safetensors"
    if not path.is_file():
        pytest.skip(f"{path} not found")
    with safetensors.safe_open(str(path), "np") as handle:
        tokens, mask = handle.get_tensor("lang_tokens"), handle.get_tensor("lang_masks")
    meta = json.loads(path.with_suffix(".json").read_text())
    policy = make_policy()
    policy.tokenizer = real_tokenizer
    got = policy._tokens({"prompt": meta["prompt"]})
    valid = int(mask.sum())
    assert got.tolist() == tokens[:valid].tolist() and got.shape == (meta["prompt_tokens"],)


def test_chat_template_matches_hf_template(real_tokenizer):
    tf = pytest.importorskip("transformers")
    template = QWEN3VL_CFG / "chat_template.json"
    if not template.is_file():
        pytest.skip("chat_template.json not found")
    hf = tf.AutoTokenizer.from_pretrained(str(MODEL_DIR))
    hf.chat_template = json.loads(template.read_text())["chat_template"]
    policy = make_policy()
    policy.tokenizer = real_tokenizer
    for prompt in ("open the drawer", "Put the cup in the microwave, then close the door!  (carefully)",
                   "把杯子放到盘子上", "héllo wörld — 12345\n\nnew line", "a " * 80):
        text = hf.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                      add_generation_prompt=False)
        assert text == "<|im_start|>user\n" + prompt + "<|im_end|>\n"
        reference = hf(text, padding="max_length", padding_side="right", max_length=72, truncation=True,
                       return_tensors="np")
        want = reference["input_ids"][0, : int(reference["attention_mask"].sum())]
        assert policy._tokens({"prompt": prompt}).tolist() == want.tolist()
