"""LingBot-VLA 2.0 host-side processing against the upstream ``lingbotvla`` package and HF processor.

Needs ``torch``, ``torchvision``, ``transformers``, ``pydantic`` and the upstream checkout
(``LINGBOT_UPSTREAM``) and the Qwen3-VL processor config (``LINGBOT_QWEN3VL_CFG``);
every test skips cleanly without them.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="parity tests need torch")
pytest.importorskip("torchvision", reason="parity tests need torchvision")
pytest.importorskip("transformers", reason="parity tests need transformers")
pytest.importorskip("pydantic", reason="upstream FeatureTransform needs pydantic")

from apxinf.policies.impls import lingbot_vla2 as lv  # noqa: E402

from _lingbot_vla2_support import (  # noqa: E402,F401
    HORIZON, QWEN3VL_CFG, RAW_KEYS, UPSTREAM, Runner, Tokenizer, make_policy, robotwin_observation, stats_dict,
)



class _Stub(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """The upstream package imports ``lerobot`` and friends at import time, never at inference."""

    NAMES = ("lerobot", "av", "decord", "torchcodec", "cv2", "wandb", "qwen_vl_utils", "datasets", "pyarrow",
             "jsonlines", "blobfile", "h5py", "imageio", "peft", "dinov3", "pandas", "zstandard", "torchdata")

    def find_spec(self, name, path, target=None):
        if any(name == n or name.startswith(n + ".") for n in self.NAMES):
            return importlib.machinery.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        module = mock.MagicMock(name=spec.name)
        module.__path__, module.__spec__, module.__name__ = [], spec, spec.name
        return module

    def exec_module(self, module):
        pass


@pytest.fixture(scope="module")
def upstream():
    if not (UPSTREAM / "lingbotvla").is_dir():
        pytest.skip(f"upstream checkout not found at {UPSTREAM}")
    import transformers.models.qwen2_vl.image_processing_qwen2_vl_fast  # noqa: F401  (real import first)

    stub = _Stub()
    sys.meta_path.append(stub)
    sys.path.insert(0, str(UPSTREAM))
    try:
        from lingbotvla.data.vla_data.utils import FeatureTransform
        from lingbotvla.data.vla_data import transform as tf
    except Exception as exc:  # pragma: no cover - environment specific
        pytest.skip(f"upstream package not importable: {exc!r}")
    finally:
        sys.path.remove(str(UPSTREAM))
        sys.meta_path.remove(stub)
    return SimpleNamespace(FeatureTransform=FeatureTransform, transform=tf)


def _data_config(joints, norm):
    return SimpleNamespace(
        joints=[str({k: v}) for k, v in joints],
        cameras=["camera_top", "camera_wrist_left", "camera_wrist_right"],
        norm_type=[str({k: v}) for k, v in norm.items()])


@pytest.mark.parametrize("norm", ["meanstd", "std", "bounds_98", "bounds_99", "bounds_98_woclip",
                                  "bounds_99_woclip", "minmax", "minmax_woclip", "identity"])
def test_normalizer_matches_upstream(upstream, norm):
    stats = stats_dict(3)["norm_stats"]
    rng = np.random.default_rng(7)
    types = {"action.arm.position": norm, "action.effector.position": norm,
             "observation.state.arm.position": norm, "observation.state.effector.position": norm}
    up = upstream.transform.Normalizer(norm_stats=stats, norm_type=types)
    for key, dim in (("observation.state.arm.position", 12), ("observation.state.effector.position", 2)):
        value = rng.normal(scale=3, size=dim)
        want = up.normalize({key: torch.from_numpy(value)})[key].numpy()
        got = lv._normalize(value, norm, {k: np.asarray(v) for k, v in stats[key].items()}, key)
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)
    for key, dim in (("action.arm.position", 12), ("action.effector.position", 2)):
        value = rng.normal(scale=3, size=(HORIZON, dim))
        want = up.unnormalize({key: value})[key]
        got = lv._unnormalize(value, norm, {k: np.asarray(v) for k, v in stats[key].items()}, key)
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


def _upstream_transform(upstream, robot_yaml, joints, norm, stats_path):
    return upstream.FeatureTransform(
        str(robot_yaml), _data_config(joints, norm),
        SimpleNamespace(max_state_dim=55, max_action_dim=55, return_image_grid_thw=True,
                        qwen3vl_use_vision_boundaries=True, use_qwen3_chat_template=True, tokenizer_max_length=72),
        None, disabled_image_features=True, chunk_size=HORIZON, norm_stats_path=str(stats_path))


@pytest.mark.parametrize("norm", ["bounds_99_woclip", "meanstd", "minmax"])
def test_state_and_action_roundtrip_matches_featuretransform(upstream, tmp_path, norm):
    """Pad+normalise+concat of the state, and select+unnormalise+reassemble of a model output."""
    yaml_path = UPSTREAM / "configs/robot_configs/robotwin.yaml"
    if not yaml_path.is_file():
        pytest.skip("robotwin robot config missing")
    joints = [("arm.position", 14), ("end.position", 14), ("effector.position", 2)]
    stats = stats_dict(11)
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps({"norm_stats": {k: {s: np.asarray(v).tolist() for s, v in e.items()}
                                                     for k, e in stats["norm_stats"].items()}, "count": 1}))
    rng = np.random.default_rng(8)
    ft = _upstream_transform(upstream, yaml_path, joints, {k: norm for k, _ in joints}, stats_path)
    policy = make_policy(norm=norm, stats=str(stats_path))

    raw_state = rng.normal(size=14).astype(np.float32)
    item = {"observation.state": torch.from_numpy(raw_state), "task": "t", "action_is_pad": torch.zeros(HORIZON)}
    # Upstream apply (policy_eval) with images disabled; read the padded state it builds.
    converted = ft.convert_features(dict(item), w_action=False)
    converted = ft.normalizer.normalize(converted)
    batch = ft.pad_and_concat(converted, w_action=False)
    want_state = upstream.transform.prepare_state(batch, 55).numpy()
    obs = {"observation.state": raw_state, "prompt": "t",
           **{v: np.zeros((8, 8, 3), np.uint8) for v in RAW_KEYS.values()}}
    got_state, groups = policy._state(obs)
    np.testing.assert_allclose(got_state, want_state, rtol=1e-5, atol=1e-6)

    # Reverse path: a random normalised chunk, padded and masked like the model output.
    chunk = rng.normal(size=(HORIZON, 55)).astype(np.float32)
    mask = batch["action_joint_mask"]
    pad = lambda m: torch.nn.functional.pad(m, (0, 55 - m.shape[-1])).to(torch.bool)  # noqa: E731  (as in apply())
    reverse = {"actions": torch.from_numpy(chunk), "state": torch.from_numpy(want_state),
               "state_joint_mask": pad(batch["state_joint_mask"]), "action_joint_mask": pad(mask)}
    want = ft.unapply(reverse)
    got = policy.decode(chunk, groups)
    np.testing.assert_allclose(got["action"], want["action"].numpy(), rtol=1e-5, atol=1e-5)


def _agilex_yaml_with_joint_delta(tmp_path):
    """The upstream file as shipped cannot run (see the bundled-robot comment); add the one missing key."""
    source = UPSTREAM / "configs/robot_configs/agilex_cobot_magic.yaml"
    if not source.is_file():
        pytest.skip("agilex robot config missing")
    text = source.read_text()
    patched = text.replace("      origin_keys: action.arm.position\n      subtract_state: true",
                           "      origin_keys: action.arm.position\n      subtract_state: true\n      relative_type: joint")
    assert patched != text
    path = tmp_path / "agilex_joint.yaml"
    path.write_text(patched)
    return source, path


def test_upstream_agilex_yaml_as_shipped_is_rejected_with_a_reason(tmp_path):
    source, _ = _agilex_yaml_with_joint_delta(tmp_path)
    with pytest.raises(ValueError, match="relative_type"):
        lv.load_robot_config(str(source))


def test_agilex_subtract_state_matches_featuretransform(upstream, tmp_path):
    _, yaml_path = _agilex_yaml_with_joint_delta(tmp_path)
    joints = [("arm.position", 14), ("effector.position", 2)]
    stats = stats_dict(12)
    stats["norm_stats"]["action.arm.position"] = {k: v[:7] for k, v in stats["norm_stats"]["action.arm.position"].items()}
    stats["norm_stats"]["observation.state.arm.position"] = {
        k: v[:7] for k, v in stats["norm_stats"]["observation.state.arm.position"].items()}
    stats["norm_stats"]["action.effector.position"] = {k: v[:1] for k, v in stats["norm_stats"]["action.effector.position"].items()}
    stats["norm_stats"]["observation.state.effector.position"] = {
        k: v[:1] for k, v in stats["norm_stats"]["observation.state.effector.position"].items()}
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps({"norm_stats": {k: {s: np.asarray(v).tolist() for s, v in e.items()}
                                                     for k, e in stats["norm_stats"].items()}}))
    rng = np.random.default_rng(9)
    norm = "meanstd"
    ft = _upstream_transform(upstream, yaml_path, joints, {k: norm for k, _ in joints}, stats_path)
    policy = lv.LingbotVla2Policy(Runner(), robot=str(yaml_path), tokenizer=Tokenizer(), norm_stats=str(stats_path),
                                  norm_type=norm, joints=joints)
    arm, eff = rng.normal(size=7).astype(np.float32), rng.normal(size=1).astype(np.float32)
    item = {"observation.state.arm.position": torch.from_numpy(arm),
            "observation.state.effector.position": torch.from_numpy(eff), "task": "t",
            "action_is_pad": torch.zeros(HORIZON)}
    converted = ft.normalizer.normalize(ft.convert_features(dict(item), w_action=False))
    batch = ft.pad_and_concat(converted, w_action=False)
    want_state = upstream.transform.prepare_state(batch, 55)
    obs = {"observation.state.arm.position": arm, "observation.state.effector.position": eff,
           **{k: np.zeros((8, 8, 3), np.uint8) for k in ("observation.images.camera_top",
                                                          "observation.images.camera_wrist_left",
                                                          "observation.images.camera_wrist_right")}}
    got_state, groups = policy._state(obs)
    np.testing.assert_allclose(got_state, want_state.numpy(), rtol=1e-5, atol=1e-6)
    chunk = rng.normal(size=(HORIZON, 55)).astype(np.float32)
    pad = lambda m: torch.nn.functional.pad(m, (0, 55 - m.shape[-1])).to(torch.bool)  # noqa: E731  (as in apply())
    reverse = {"actions": torch.from_numpy(chunk), "state": want_state,
               "state_joint_mask": pad(batch["state_joint_mask"]), "action_joint_mask": pad(batch["action_joint_mask"])}
    want = ft.unapply(reverse)
    got = policy.decode(chunk, groups)
    for key in ("action.arm.position", "action.effector.position"):
        np.testing.assert_allclose(got[key], want[key].numpy(), rtol=1e-4, atol=1e-5)


@pytest.fixture(scope="module")
def image_processor():
    config = QWEN3VL_CFG / "preprocessor_config.json"
    if not config.is_file():
        pytest.skip("Qwen3-VL preprocessor_config.json not found")
    from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast

    return Qwen2VLImageProcessorFast.from_pretrained(str(config.parent))


@pytest.mark.parametrize("shape", [(240, 320), (480, 640), (256, 256), (180, 320), (300, 200), (720, 1280)])
def test_pixels_match_upstream_resize_and_hf_processor(image_processor, shape):
    from torchvision.transforms.v2 import Resize

    rng = np.random.default_rng(10)
    image = rng.integers(0, 256, (*shape, 3), dtype=np.uint8)
    # deploy/lingbot_vla_v2_policy.py::resize_image: float32 CHW then v2.Resize
    chw = torch.as_tensor(image).permute(2, 0, 1).contiguous().to(torch.float32)
    resized = Resize((256, 256))(chw)
    mine_resized = lv._prepare_view(image, "img", 256)
    assert np.abs(mine_resized - resized.numpy()).max() < 1e-3
    out = image_processor(resized)  # what FeatureTransform.prepare_images passes to the HF processor
    want = out["pixel_values"].numpy()
    got = lv._patchify(resized.numpy())
    assert out["image_grid_thw"].tolist() == [[1, 16, 16]]
    assert got.shape == want.shape == (256, 1536)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-6)
    # end to end: uint8 in -> same pixels as the reference up to the resize arithmetic
    np.testing.assert_allclose(lv._patchify(mine_resized), want, rtol=0, atol=1e-3 / 127.5 + 1e-6)


def test_hf_processor_does_not_resize_256_inputs(image_processor):
    """256x256 is already a multiple of 32 and inside [65536, 16777216] pixels: no second resize."""
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    assert smart_resize(256, 256, factor=32, min_pixels=65536, max_pixels=16777216) == (256, 256)


def test_whole_pipeline_pixels_for_three_views(image_processor):
    rng = np.random.default_rng(13)
    policy = make_policy()
    obs = robotwin_observation(rng)
    pixels, grids = policy._vision(obs)
    assert pixels.shape == (768, 1536) and grids.tolist() == [[1, 16, 16]] * 3
    from torchvision.transforms.v2 import Resize

    expected = []
    for key in (RAW_KEYS["cam_high"], RAW_KEYS["cam_left"], RAW_KEYS["cam_right"]):
        chw = torch.as_tensor(obs[key]).permute(2, 0, 1).contiguous().to(torch.float32)
        expected.append(image_processor(Resize((256, 256))(chw))["pixel_values"].numpy())
    np.testing.assert_allclose(pixels, np.concatenate(expected), rtol=0, atol=1e-3 / 127.5 + 1e-6)


