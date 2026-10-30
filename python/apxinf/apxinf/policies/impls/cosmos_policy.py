"""Direct Cosmos Policy inference with joint video/action latent sampling.

The native family owns VAE encoding, the complete joint denoiser and sampling.
This policy owns RGB preprocessing, cached-task selection and physical units.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image

from ...processors.transforms import lookup_key
from ..base import StateConditionedModelRunnerProtocol
from ..registry import register_policy

__all__ = ["CosmosPolicy"]

_PROFILE = "libero-2b-direct-v1"
_LATENT_SHAPE = (1, 16, 9, 28, 28)


def _resize_crop(image: np.ndarray) -> np.ndarray:
    """212 -> 224 bilinear AA in FP32, followed by Torch's uint8 rounding.

    Pillow rounds each separable pass to uint8. The upstream tensor transform
    retains the horizontal FP32 result and rounds only after the vertical pass.
    Its scalar interpolation uses fused multiply-add for the second tap. A
    float64 intermediate emulates that rounding for these bounded RGB values;
    rounding both products separately changes pixels exactly at half integers.
    """
    scale = np.float32(212 / 224)
    center = (np.arange(224, dtype=np.float32) + np.float32(0.5)) * scale
    low = np.clip(np.floor(center - np.float32(0.5)).astype(np.int64), 0, 211)
    high = np.minimum(low + 1, 211)
    left = np.maximum(np.float32(0), np.float32(1) - np.abs(low.astype(np.float32) + np.float32(0.5) - center))
    right = np.maximum(np.float32(0), np.float32(1) - np.abs(high.astype(np.float32) + np.float32(0.5) - center))
    right = np.where(high == low, np.float32(0), right)
    total = left + right
    left, right = left / total, right / total
    image = image.astype(np.float32)
    first = image[:, low] * left[None, :, None]
    horizontal = (first.astype(np.float64)
                  + image[:, high].astype(np.float64) * right.astype(np.float64)[None, :, None]).astype(np.float32)
    first = horizontal[low] * left[:, None, None]
    output = (first.astype(np.float64)
              + horizontal[high].astype(np.float64) * right.astype(np.float64)[:, None, None]).astype(np.float32)
    return np.rint(output).clip(0, 255).astype(np.uint8)


def _prepare_image(value: Any) -> np.ndarray:
    image = np.asarray(value)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) == 0:
        raise ValueError("Cosmos Policy images must be nonempty HWC uint8 RGB")
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        # Upstream uses Pillow's default RGB resize (bicubic) at this stage.
        resized = np.asarray(decoded if decoded.size == (224, 224) else decoded.resize((224, 224)))
    return _resize_crop(resized[6:218, 6:218])


def _range(stats: Mapping[str, Any], name: str, width: int) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(stats[f"{name}_min"], dtype=np.float64)
    upper = np.asarray(stats[f"{name}_max"], dtype=np.float64)
    if lower.shape != (width,) or upper.shape != (width,):
        raise ValueError(f"Cosmos Policy {name} statistics must have width {width}")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(upper <= lower):
        raise ValueError(f"Cosmos Policy {name} statistics must have finite increasing bounds")
    return lower, upper


@register_policy("cosmos-policy")
class CosmosPolicy:
    """LIBERO direct policy, using the checkpoint's 40 cached text conditions.

    Camera order is wrist, primary. The caller names the proprioception key
    explicitly. Simulator image flipping and robot-specific gripper commands
    belong to the caller's observation/action adapter.

    ``image_transform`` replaces the complete per-image reference preprocessing
    recipe and must return HWC ``[224,224,3]`` uint8 RGB. An adapter that already
    supplies those canonical pixels can pass an identity transform. The default
    retains the official profile's JPEG/crop/resize behavior for parity.
    """

    def __init__(
        self,
        model_runner: StateConditionedModelRunnerProtocol,
        *,
        tasks: Sequence[str],
        dataset_stats: Mapping[str, Any],
        state_key: str,
        image_keys: Sequence[str] = ("wrist_image", "primary_image"),
        prompt_key: str = "prompt",
        image_transform: Callable[[Any], np.ndarray] | None = None,
    ):
        if not isinstance(state_key, str) or not state_key:
            raise ValueError("Cosmos Policy requires an explicit nonempty state_key")
        if len(image_keys) != 2 or any(not isinstance(key, str) or not key for key in image_keys):
            raise ValueError("Cosmos Policy requires wrist and primary image keys in that order")
        if not tasks or any(not isinstance(task, str) or not task for task in tasks) or len(set(tasks)) != len(tasks):
            raise ValueError("Cosmos Policy task table must contain unique nonempty strings")
        if (model_runner.action_horizon, model_runner.action_dim, model_runner.num_views, model_runner.image_size) != (16, 7, 2, 224):
            raise ValueError("Cosmos Policy runner does not match the LIBERO direct profile")
        if tuple(model_runner.initial_latent_shape) != _LATENT_SHAPE:
            raise ValueError("Cosmos Policy requires the complete joint latent noise shape")
        self.model_runner: StateConditionedModelRunnerProtocol | None = model_runner
        self._tasks = {task: index for index, task in enumerate(tasks)}
        self._proprio_min, self._proprio_max = _range(dataset_stats, "proprio", 9)
        self._action_min, self._action_max = _range(dataset_stats, "actions", 7)
        self.state_key = state_key
        self.image_keys = tuple(image_keys)
        self.prompt_key = prompt_key
        # The default reproduces the reference policy's image recipe. An outer
        # adapter may instead supply canonical RGB, without JPEG/cropping twice.
        if image_transform is not None and not callable(image_transform):
            raise TypeError("Cosmos Policy image_transform must be callable")
        self.image_transform = _prepare_image if image_transform is None else image_transform
        self.metadata = {
            "model_type": "cosmos-policy", "profile": _PROFILE,
            "action_horizon": 16, "action_dim": 7,
            "initial_latent_shape": list(_LATENT_SHAPE),
            "image_keys": self.image_keys, "state_key": state_key,
            "prompt_key": prompt_key, "supported_tasks": tuple(tasks),
            "image_preprocessing": "libero-reference-v1" if image_transform is None else "custom",
        }

    @classmethod
    def from_pretrained(
        cls,
        model_dir,
        *,
        state_key: str,
        model_runner: StateConditionedModelRunnerProtocol | None = None,
        image_keys: Sequence[str] = ("wrist_image", "primary_image"),
        prompt_key: str = "prompt",
        image_transform: Callable[[Any], np.ndarray] | None = None,
        device: str = "cuda:0",
        **load_options,
    ) -> "CosmosPolicy":
        root = Path(model_dir)
        config = json.loads((root / "config.json").read_text())
        if config.get("model_type") != "cosmos-policy" or config.get("profile") != _PROFILE:
            raise ValueError("Cosmos Policy requires prepared libero-2b-direct-v1 assets")
        tasks = json.loads((root / "tasks.json").read_text())["tasks"]
        stats = json.loads((root / "libero_dataset_statistics.json").read_text())
        if model_runner is None:
            import apxinf_py
            model_runner = apxinf_py.ModelRunner.load("cosmos-policy", str(root), device=device, **load_options)
        elif load_options:
            raise ValueError("load options cannot be applied to an injected model_runner")
        return cls(model_runner, tasks=tasks, dataset_stats=stats, state_key=state_key,
                   image_keys=image_keys, prompt_key=prompt_key,
                   image_transform=image_transform)

    @property
    def action_horizon(self) -> int:
        return 16

    @property
    def action_dim(self) -> int:
        return 7

    def infer(self, observation: Mapping[str, Any], *, noise: np.ndarray | None = None) -> dict:
        if self.model_runner is None:
            raise RuntimeError("Cosmos Policy is closed")
        started = time.perf_counter()
        task = lookup_key(observation, self.prompt_key)
        if not isinstance(task, str) or task not in self._tasks:
            raise ValueError("Cosmos Policy task is absent from this checkpoint's cached text conditions")
        token_ids = np.asarray([self._tasks[task]], dtype=np.uint32)
        images = []
        for key in self.image_keys:
            image = np.asarray(self.image_transform(lookup_key(observation, key)))
            if image.dtype != np.uint8 or image.shape != (224, 224, 3):
                raise ValueError("Cosmos Policy image_transform must return [224,224,3] uint8 RGB")
            images.append(image)
        rgb = np.ascontiguousarray(np.stack(images))
        proprio = np.asarray(lookup_key(observation, self.state_key), dtype=np.float64)
        if proprio.shape != (9,) or not np.isfinite(proprio).all():
            raise ValueError("Cosmos Policy proprioception must be finite with shape [9]")
        state = np.ascontiguousarray(2 * ((proprio - self._proprio_min) / (self._proprio_max - self._proprio_min)) - 1, dtype=np.float32)
        selected_noise = noise if noise is not None else observation.get("noise")
        if selected_noise is not None:
            selected_noise = np.ascontiguousarray(selected_noise, dtype=np.float32)
            if selected_noise.shape != _LATENT_SHAPE or not np.isfinite(selected_noise).all():
                raise ValueError(f"Cosmos Policy noise must be finite with shape {_LATENT_SHAPE}")
        model_started = time.perf_counter()
        normalized = np.array(self.model_runner.infer_rgb(rgb, "nhwc", token_ids, selected_noise, state=state), dtype=np.float32, copy=True)
        model_ms = (time.perf_counter() - model_started) * 1000
        if normalized.shape != (16, 7) or not np.isfinite(normalized).all():
            raise ValueError("Cosmos Policy runtime returned invalid actions")
        actions = np.ascontiguousarray(0.5 * (normalized + 1) * (self._action_max - self._action_min) + self._action_min, dtype=np.float32)
        return {
            "actions": actions, "normalized_actions": normalized,
            "timing": {"model_ms": model_ms, "total_ms": (time.perf_counter() - started) * 1000},
        }

    __call__ = infer

    def close(self) -> None:
        runner = self.model_runner
        self.model_runner = None
        close = getattr(runner, "close", None)
        if callable(close):
            close()
