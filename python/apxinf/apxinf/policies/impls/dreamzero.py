"""DROID observation adapter for the native joint DreamZero model.

Torchvision is used only for the checkpoint's public CPU camera crop/resize.
All learned vision, language, video/action sampling and history run natively.
"""
from __future__ import annotations

import ast
import html
import json
import re
import time
import uuid
from pathlib import Path

import numpy as np

from ...processors.transforms import lookup_key
from ..registry import register_policy

_HEAD = "groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf.WANPolicyHead"
_VIEWS = ("video.exterior_image_1_left", "video.exterior_image_2_left", "video.wrist_image_left")
_NEGATIVE = (
    "Vibrant colors, overexposed, static, blurry details, text, subtitles, style, artwork, "
    "painting, image, still, grayscale, dull, worst quality, low quality, JPEG artifacts, "
    "ugly, mutilated, extra fingers, bad hands, bad face, deformed, disfigured, mutated limbs, "
    "fused fingers, stagnant image, cluttered background, three legs, many people in the "
    "background, walking backwards."
)


def _round_bf16(values):
    """Round finite policy values to BF16, returned as lossless NumPy F32."""
    values = np.ascontiguousarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)


def _prompt(task):
    if not isinstance(task, str) or not task.strip():
        raise ValueError("DreamZero task must be a nonempty string")
    try:
        parsed = ast.literal_eval(task)
        task = str(parsed[0] if isinstance(parsed, (list, tuple)) else parsed)
    except (ValueError, SyntaxError, TypeError, IndexError):
        pass
    task = task.lower()
    return (
        "A multi-view video shows that a robot " + task
        + " The video is split into three views: The top view shows the camera view from "
        "the robot's wrist, the bottom-left view shows the camera view from the left exterior "
        "camera, and the bottom-right view shows the camera view from the right exterior "
        "camera. During training, one of the two bottom exterior views may be a black screen "
        "(dropped view). The robot " + task
    )


class _Tokenizer:
    def __init__(self, path):
        import apxinf_py
        path = Path(path)
        if path.is_dir():
            path = path / "tokenizer.json"
        self.native = apxinf_py.HfTokenizer.from_file(str(path))
        if self.native.token_to_id("<pad>") != 0 or self.native.token_to_id("</s>") != 1:
            raise ValueError("DreamZero requires the official UMT5 tokenizer with pad=0 and EOS=1")

    def encode(self, text):
        import ftfy
        text = re.sub(r"\s+", " ", html.unescape(html.unescape(ftfy.fix_text(text)))).strip()
        ids = list(self.native.encode_with_special_tokens(text, True))
        if len(ids) > 512:
            ids = ids[:511] + [1]
        mask = [1] * len(ids) + [0] * (512 - len(ids))
        return ids + [0] * (512 - len(ids)), mask


@register_policy("dreamzero")
@register_policy("dreamzero_droid")
class DreamZeroPolicy:
    def __init__(self, model_runner, *, tokenizer, statistics, image_keys=_VIEWS,
                 joint_key="state.joint_position", gripper_key="state.gripper_position",
                 prompt_key="task", images_preprocessed=False):
        self.model_runner = model_runner
        self.tokenizer = tokenizer
        self.image_keys = tuple(image_keys)
        if len(self.image_keys) != 3 or tuple(model_runner.noise_shape) != (113408,):
            raise ValueError("DreamZero requires three cameras and the complete joint latent")
        self.joint_key, self.gripper_key, self.prompt_key = joint_key, gripper_key, prompt_key
        self.images_preprocessed = bool(images_preprocessed)
        self._stats = {}
        for modality in ("state", "action"):
            for bound in ("q01", "q99"):
                value = np.concatenate([
                    np.asarray(statistics[modality][key][bound], dtype=np.float32)
                    for key in ("joint_position", "gripper_position")
                ])
                if value.shape != (8,) or not np.isfinite(value).all():
                    raise ValueError(f"invalid DreamZero {modality} {bound} statistics")
                self._stats[modality, bound] = value
            if np.any(self._stats[modality, "q99"] < self._stats[modality, "q01"]):
                raise ValueError(f"reversed DreamZero {modality} quantiles")
        self._session = uuid.uuid4().hex
        self._episode = uuid.uuid4().hex
        self.metadata = dict(model_type="dreamzero", model_variant="bf16",
                             sampling_profile="standard", action_horizon=24, action_dim=8,
                             native_action_dim=32, noise_shape=[113408], num_views=3,
                             image_keys=list(self.image_keys), history=True)

    @property
    def action_dim(self):
        return 8

    @property
    def action_horizon(self):
        return 24

    def close(self):
        # Drop this policy's reference; another injected owner remains valid.
        self.model_runner = None

    @classmethod
    def from_pretrained(cls, model_dir, *, device="cuda:0", seed=0, model_runner=None,
                        tokenizer_path=None, model_variant="bf16", sampling_profile="standard",
                        autotune=False, tactics=None, **kwargs):
        if model_variant not in ("auto", "bf16") or sampling_profile != "standard":
            raise ValueError("DreamZero currently admits the released DROID BF16 standard checkpoint; Flash weights are not released/verified")
        root = Path(model_dir)
        config = json.loads((root / "config.json").read_text())
        if (config.get("action_head_cfg", {}).get("_target_") != _HEAD
                or config.get("action_dim") != 32):
            raise ValueError("DreamZeroPolicy requires the official joint DROID checkpoint")
        import yaml
        training = yaml.safe_load((root / "experiment_cfg" / "conf.yaml").read_text())
        if (training.get("relative_action") is not True
                or training.get("relative_action_keys") != ["joint_position"]
                or training.get("relative_action_per_horizon", False)):
            raise ValueError("unsupported DreamZero robot action normalization profile")
        metadata = json.loads((root / "experiment_cfg" / "metadata.json").read_text())["oxe_droid"]
        tokenizer = _Tokenizer(tokenizer_path or root / "tokenizer")
        if model_runner is None:
            import apxinf_py
            model_runner = apxinf_py.ModelRunner.load(
                "dreamzero", root, device=device, sampling_seed=int(seed),
                model_variant=model_variant, autotune=bool(autotune),
                tactics=str(tactics) if tactics is not None else None,
            )
        return cls(model_runner, tokenizer=tokenizer, statistics=metadata["statistics"], **kwargs)

    def reset_history(self, episode=None):
        if episode is not None and (not isinstance(episode, str) or not episode):
            raise ValueError("episode must be a nonempty string")
        self.model_runner.reset_history()
        self._episode = episode if episode is not None else uuid.uuid4().hex

    def clear_prepared(self):
        self.model_runner.clear_prepared()

    def _images(self, observation):
        cameras = []
        for key in self.image_keys:
            image = np.asarray(lookup_key(observation, key))
            if image.ndim == 3:
                image = image[None]
            if (image.ndim != 4 or image.shape[-1] != 3 or image.dtype != np.uint8
                    or image.shape[0] not in (1, 4, 9) or min(image.shape[1:3]) < 2):
                raise ValueError(f"{key} must be uint8 RGB [1|4|9,H,W,3]")
            if self.images_preprocessed:
                if image.shape[1:3] != (176, 320):
                    raise ValueError("preprocessed DreamZero camera frames must be 176x320")
            else:
                import torch
                from torchvision.transforms.v2 import functional as F
                # Public CPU preprocessing matches the official VideoToTensor,
                # eval CenterCrop, antialiased Tensor Resize, VideoToNumpy chain.
                tensor = torch.from_numpy(np.ascontiguousarray(image)).float().permute(0, 3, 1, 2) / 255.
                tensor = F.center_crop(tensor, [int(image.shape[1] * .95), int(image.shape[2] * .95)])
                tensor = F.resize(tensor, [176, 320], interpolation=F.InterpolationMode.BILINEAR, antialias=True)
                image = (tensor.permute(0, 2, 3, 1) * 255).to(torch.uint8).numpy()
            cameras.append(image)
        if len({c.shape for c in cameras}) != 1:
            raise ValueError("DreamZero camera histories must have identical shapes")
        left, right, wrist = cameras
        montage = np.concatenate([np.repeat(wrist, 2, axis=2), np.concatenate([left, right], axis=2)], axis=1)
        pixels = _round_bf16(montage.astype(np.float32) / np.float32(255.))
        pixels = _round_bf16(_round_bf16(pixels - np.float32(.5)) / np.float32(.5))
        return np.ascontiguousarray(pixels.reshape(-1, 3)), cameras[0].shape[0]

    def _state(self, observation):
        components = []
        for key, size in ((self.joint_key, 7), (self.gripper_key, 1)):
            x = np.asarray(lookup_key(observation, key), dtype=np.float32)
            if x.shape == (1, size):
                x = x[0]
            if x.shape != (size,) or not np.isfinite(x).all():
                raise ValueError(f"{key} must have shape [{size}] or [1,{size}] and finite values")
            components.append(x)
        state = np.concatenate(components)
        low, high = self._stats["state", "q01"], self._stats["state", "q99"]
        normalized = state.copy()
        mask = low != high
        normalized[mask] = (state[mask] - low[mask]) / (high[mask] - low[mask])
        normalized[mask] = 2 * normalized[mask] - 1
        result = np.zeros((1, 64), dtype=np.float32)
        result[0, :8] = np.clip(normalized, -1, 1)
        return result, state

    def infer(self, observation, *, noise=None, episode=None, profile=False):
        if self.model_runner is None:
            raise RuntimeError("DreamZeroPolicy is closed")
        if profile:
            raise ValueError("DreamZero phase profiling is available through the native validation harness")
        started = time.perf_counter()
        identity = self._episode if episode is None else episode
        if not isinstance(identity, str) or not identity:
            raise ValueError("episode must be a nonempty string")
        pixels, frames = self._images(observation)
        state, raw_state = self._state(observation)
        positive, pmask = self.tokenizer.encode(_prompt(lookup_key(observation, self.prompt_key)))
        negative, nmask = self.tokenizer.encode(_NEGATIVE)
        if noise is not None:
            noise = np.ascontiguousarray(noise, dtype=np.float32)
            if noise.shape != (113408,) or not np.isfinite(noise).all():
                raise ValueError("DreamZero noise must contain video then action: float32 [113408]")
        model_started = time.perf_counter()
        output = np.asarray(self.model_runner._infer_preprocessed(
            pixels, np.array([[frames, 352, 640]], dtype=np.uint32),
            np.array(positive + negative, dtype=np.uint32), np.array(pmask + nmask, dtype=np.uint8),
            state, 17, noise, history_session=self._session, history_episode=identity,
        ), dtype=np.float32)
        model_ms = (time.perf_counter() - model_started) * 1000
        if output.shape != (24, 32) or not np.isfinite(output).all():
            raise ValueError("DreamZero native output must be finite [24,32]")
        low, high = self._stats["action", "q01"], self._stats["action", "q99"]
        actions = (output[:, :8] + 1) / 2 * (high - low) + low
        actions[:, :7] += raw_state[:7]
        self._episode = identity
        return dict(actions=np.ascontiguousarray(actions), normalized_actions=output,
                    timing=dict(model_ms=model_ms, total_ms=(time.perf_counter() - started) * 1000))
