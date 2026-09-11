"""WallOSS policy: raw robot observations to deployable action chunks.

WallOSS uses Qwen2.5-VL preprocessing, which is intentionally owned by this
model module. The built-in adapter sends resized RGB to a capable Rust runtime;
custom processors retain the canonical patch/token/mask contract.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ...checkpoints import load_torch_state_file
from ..registry import register_policy

__all__ = ["WallossPolicy"]

_ROLE_START = "<|im_start|>"
_ROLE_END = "<|im_end|>"
_VISION_START = "<|vision_start|>"
_VISION_END = "<|vision_end|>"
_IMAGE_PAD = "<|image_pad|>"
_PROPRI = "<|propri|>"
_ACTION = "<|action|>"
_CAMERA_LABELS = {"face_view": "front view", "right_wrist_view": "right wrist view"}
_DEFAULT_IMAGE_KEYS = ("observation/image", "observation/wrist_image")
_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC


def _lookup(data: Mapping[str, Any], path: str) -> Any:
    if path in data:
        return data[path]
    value: Any = data
    for part in path.split("/"):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(path)
        value = value[part]
    return value


def _round_by_factor(value: int, factor: int) -> int:
    return round(value / factor) * factor


def _smart_resize(
    height: int, width: int, factor: int, min_pixels: int, max_pixels: int
):
    if height < factor or width < factor:
        raise ValueError(
            f"height:{height} or width:{width} must be larger than factor:{factor}"
        )
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            "absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width)}"
        )
    new_h = _round_by_factor(height, factor)
    new_w = _round_by_factor(width, factor)
    if new_h * new_w > max_pixels:
        beta = (height * width / max_pixels) ** 0.5
        new_h = max(factor, int(height / beta // factor) * factor)
        new_w = max(factor, int(width / beta // factor) * factor)
    elif new_h * new_w < min_pixels:
        beta = (min_pixels / (height * width)) ** 0.5
        new_h = (int(height * beta) + factor - 1) // factor * factor
        new_w = (int(width * beta) + factor - 1) // factor * factor
    return new_h, new_w


def _load_normalizer(path: Path, norm_key: str) -> tuple[np.ndarray, np.ndarray]:
    state = load_torch_state_file(path)
    available = sorted(key[4:] for key in state if key.startswith("min."))
    resolved = norm_key
    if resolved not in available:
        matches = [key for key in available if key.startswith(f"{norm_key}_")]
        if len(matches) == 1:
            resolved = matches[0]
        else:
            raise KeyError(
                f"norm_key={norm_key!r} not in {path.name}; available={available}"
            )
    minimum = state[f"min.{resolved}"]
    delta = state[f"delta.{resolved}"]
    return np.asarray(minimum, np.float32), np.asarray(delta, np.float32)


class _WallossTokenizer:
    """Qwen2.5-VL tokenizer backed by ApxInf's native Rust tokenizer."""

    def __init__(self, model_dir: Path):
        try:
            import apxinf_py
        except ImportError as error:  # pragma: no cover - dependency error is user-facing
            raise ImportError(
                "WallossPolicy requires the native apxinf-py package"
            ) from error

        path = model_dir / "tokenizer.json"
        if not path.is_file():
            raise FileNotFoundError(f"WallOSS tokenizer file does not exist: {path}")
        self._tokenizer = apxinf_py.HfTokenizer.from_file(str(path))
        self._tokenizer.add_tokens([_PROPRI, _ACTION])
        config_path = model_dir / "config.json"
        if config_path.is_file():
            document = json.loads(config_path.read_text())
            if not isinstance(document, Mapping):
                raise ValueError("WallOSS config.json must contain an object")
            image_token_id = document.get("image_token_id")
            if image_token_id is not None and self.token_to_id(_IMAGE_PAD) != int(
                image_token_id
            ):
                raise ValueError(
                    "WallOSS tokenizer image token does not match config.json: "
                    f"{self.token_to_id(_IMAGE_PAD)} != {image_token_id}"
                )
            vocab_size = document.get("vocab_size")
            if vocab_size is not None:
                for token in (_PROPRI, _ACTION):
                    if self.token_to_id(token) >= int(vocab_size):
                        raise ValueError(
                            f"WallOSS token {token!r} is outside vocab_size={vocab_size}"
                        )

    def token_to_id(self, token: str) -> int:
        value = self._tokenizer.token_to_id(token)
        if value is None:
            raise ValueError(f"WallOSS tokenizer has no token {token!r}")
        return int(value)

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text))


class _WallossImageProcessor:
    """NumPy/Pillow implementation of Qwen2VLImageProcessor 4.49 semantics."""

    def __init__(self, model_dir: Path):
        path = model_dir / "preprocessor_config.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"WallOSS image processor config does not exist: {path}"
            )
        document = json.loads(path.read_text())
        if not isinstance(document, Mapping):
            raise TypeError(f"{path.name} must contain an object")
        processor_type = document.get("image_processor_type")
        if processor_type not in (None, "Qwen2VLImageProcessor"):
            raise ValueError(
                f"unsupported WallOSS image_processor_type {processor_type!r}"
            )
        for flag in ("do_resize", "do_rescale", "do_normalize"):
            if document.get(flag, True) is False:
                raise ValueError(f"WallOSS requires {flag}=true")
        if document.get("resample") not in (None, int(_BICUBIC)):
            raise ValueError("WallOSS requires bicubic image resampling")

        self.image_mean = np.asarray(
            document.get("image_mean", [0.48145466, 0.4578275, 0.40821073]),
            dtype=np.float32,
        )
        self.image_std = np.asarray(
            document.get("image_std", [0.26862954, 0.26130258, 0.27577711]),
            dtype=np.float32,
        )
        self.rescale_factor = float(document.get("rescale_factor", 1 / 255))
        self.min_pixels = int(document.get("min_pixels", 56 * 56))
        self.max_pixels = int(document.get("max_pixels", 14 * 14 * 4 * 1280))
        self.patch_size = int(document.get("patch_size", 14))
        self.temporal_patch_size = int(document.get("temporal_patch_size", 2))
        self.merge_size = int(document.get("merge_size", 2))
        if self.image_mean.shape != (3,) or self.image_std.shape != (3,):
            raise ValueError(
                "WallOSS image_mean and image_std must contain three channels"
            )
        if not np.all(np.isfinite(self.image_mean)) or not np.all(
            np.isfinite(self.image_std)
        ):
            raise ValueError("WallOSS image normalization contains non-finite values")
        if not np.isfinite(self.rescale_factor) or self.rescale_factor <= 0:
            raise ValueError("WallOSS rescale_factor must be finite and positive")
        if np.any(self.image_std == 0):
            raise ValueError("WallOSS image_std must be non-zero")
        if (
            min(
                self.min_pixels,
                self.max_pixels,
                self.patch_size,
                self.temporal_patch_size,
                self.merge_size,
            )
            <= 0
        ):
            raise ValueError("WallOSS image processor dimensions must be positive")
        if self.min_pixels > self.max_pixels:
            raise ValueError("WallOSS min_pixels exceeds max_pixels")

    def _resize_one(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
        factor = self.patch_size * self.merge_size
        new_h, new_w = _smart_resize(
            image.shape[0], image.shape[1], factor, self.min_pixels, self.max_pixels
        )
        resized = np.asarray(
            Image.fromarray(image).resize((new_w, new_h), resample=_BICUBIC)
        )
        return np.ascontiguousarray(resized, dtype=np.uint8), (
            1,
            new_h // self.patch_size,
            new_w // self.patch_size,
        )

    def _patchify_one(self, resized: np.ndarray) -> np.ndarray:
        # Match Transformers 4.49 exactly: rescale through float64, downcast to
        # float32, then normalize using float32 channel constants.
        normalized = (resized.astype(np.float64) * self.rescale_factor).astype(
            np.float32
        )
        normalized = (normalized - self.image_mean) / self.image_std
        patches = normalized.transpose(2, 0, 1)[None, ...]
        remainder = patches.shape[0] % self.temporal_patch_size
        if remainder:
            repeats = np.repeat(
                patches[-1][None], self.temporal_patch_size - remainder, axis=0
            )
            patches = np.concatenate([patches, repeats], axis=0)

        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h = resized.shape[0] // self.patch_size
        grid_w = resized.shape[1] // self.patch_size
        patches = patches.reshape(
            grid_t,
            self.temporal_patch_size,
            3,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flattened = patches.reshape(
            grid_t * grid_h * grid_w,
            3 * self.temporal_patch_size * self.patch_size * self.patch_size,
        )
        return flattened

    def resize(self, images: Sequence[np.ndarray]):
        processed = [self._resize_one(image) for image in images]
        shapes = {image.shape for image, _ in processed}
        if len(shapes) != 1:
            raise ValueError(
                f"WallOSS resized camera shapes must match; got {sorted(shapes)}"
            )
        return {
            "rgb_u8": np.ascontiguousarray(
                np.stack([image for image, _ in processed]), dtype=np.uint8
            ),
            "image_grid_thw": np.asarray(
                [grid for _, grid in processed], dtype=np.int64
            ),
        }

    def __call__(self, images: Sequence[np.ndarray]):
        resized = self.resize(images)
        return {
            "pixel_values": np.ascontiguousarray(
                np.concatenate(
                    [self._patchify_one(image) for image in resized["rgb_u8"]]
                ),
                dtype=np.float32,
            ),
            "image_grid_thw": resized["image_grid_thw"],
        }


def _checkpoint_state_bins(model_dir: Path) -> int:
    """Resolve state-token bins from checkpoint metadata, with legacy fallback."""
    documents = []
    config_json = model_dir / "config.json"
    if config_json.is_file():
        documents.append((config_json, json.loads(config_json.read_text())))
    config_yaml = None
    for name in ("config.yml", "config.yaml"):
        candidate = model_dir / name
        if candidate.is_file():
            config_yaml = candidate
            break
    if config_yaml is not None:
        try:
            import yaml
        except ImportError as error:  # pragma: no cover - declared project dependency
            raise ImportError(
                "WallossPolicy requires PyYAML to read config.yml"
            ) from error
        documents.append((config_yaml, yaml.safe_load(config_yaml.read_text()) or {}))

    for path, document in documents:
        if not isinstance(document, Mapping):
            raise TypeError(f"{path.name} must contain a mapping")
        value = document.get("state_bins")
        data = document.get("data")
        if value is None and isinstance(data, Mapping):
            value = data.get("state_bins")
        if value is not None:
            bins = int(value)
            if bins < 2:
                raise ValueError(f"{path.name} state_bins must be >= 2, got {bins}")
            return bins
    return 256


class _WallossProcessor:
    def __init__(
        self,
        model_dir: Path,
        *,
        image_keys: Sequence[str],
        camera_names: Sequence[str],
        state_key: str,
        prompt_key: str,
        action_horizon: int,
        action_dim: int,
        norm_key: str,
        state_bins: int,
        native_rgb: bool = False,
    ):
        self.image_keys = tuple(image_keys)
        self.camera_names = tuple(camera_names)
        self.state_key = str(state_key)
        self.prompt_key = str(prompt_key)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.state_bins = int(state_bins)
        self.native_rgb = bool(native_rgb)
        self.tokenizer = _WallossTokenizer(model_dir)
        self.image_processor = _WallossImageProcessor(model_dir)
        self.action_token_id = self.tokenizer.token_to_id(_ACTION)
        self.image_pad_token_id = self.tokenizer.token_to_id(_IMAGE_PAD)
        self.propri_min, self.propri_delta = _load_normalizer(
            model_dir / "normalizer_propri.pth", norm_key
        )

    def _prompt(self, instruction: str, state: np.ndarray, active: np.ndarray) -> str:
        normalized = np.clip(
            (state - self.propri_min) / self.propri_delta * 2 - 1, -1, 1
        )
        # Keep numpy's float64 linspace: this mirrors the training/reference
        # processor exactly at bin boundaries.
        edges = np.linspace(-1.0, 1.0, self.state_bins + 1)[:-1]
        bins = np.clip(np.digitize(normalized, edges) - 1, 0, self.state_bins - 1)
        state_text = " ".join(str(int(value)) for value in bins[active])
        cameras = "".join(
            f" {_CAMERA_LABELS.get(name, name.replace('_', ' '))}: "
            f"{_VISION_START}{_IMAGE_PAD}{_VISION_END}"
            for name in self.camera_names
        )
        return (
            f"{_ROLE_START}system\nYou are a helpful assistant.{_ROLE_END}\n"
            f"{_ROLE_START}user\nObservation:{cameras}\nInstruction: {instruction}"
            f"\nPredict the next action in robot action.\nProprioception: {state_text}\n"
            f"{_ROLE_END}\n{_ROLE_START}assistant\n" + _ACTION * self.action_horizon
        )

    def __call__(self, observation: Mapping[str, Any]):
        raw_state = np.asarray(
            _lookup(observation, self.state_key), dtype=np.float32
        ).reshape(-1)
        if raw_state.size > self.action_dim:
            raise ValueError(
                f"state has {raw_state.size} values, maximum is {self.action_dim}"
            )
        state = np.zeros(self.action_dim, dtype=np.float32)
        state[: raw_state.size] = raw_state
        agent_pos_mask = observation.get("agent_pos_mask")
        if agent_pos_mask is None:
            active = np.zeros(self.action_dim, dtype=bool)
            active[: raw_state.size] = True
        else:
            active = (
                np.asarray(agent_pos_mask, dtype=np.float32).reshape(-1).astype(bool)
            )
            if active.size != self.action_dim:
                raise ValueError(
                    f"agent_pos_mask has {active.size} values, expected {self.action_dim}"
                )
        prompt = _lookup(observation, self.prompt_key)
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")

        images = []
        for key in self.image_keys:
            image = np.asarray(_lookup(observation, key))
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError(
                    f"{key} must be HxWx3 uint8 RGB, got {image.shape} {image.dtype}"
                )
            images.append(image)

        processed = (
            self.image_processor.resize(images)
            if getattr(self, "native_rgb", False)
            else self.image_processor(images)
        )
        grids = np.asarray(processed["image_grid_thw"])
        if grids.shape != (2, 3) or not np.array_equal(
            grids, np.array([[1, 18, 18]] * 2)
        ):
            raise ValueError(
                f"WallOSS runtime currently requires two 18x18 image grids; got {grids.tolist()} "
                "(256x256 input images smart-resize to the required 252x252)"
            )
        if getattr(self, "native_rgb", False):
            vision = np.ascontiguousarray(processed["rgb_u8"], dtype=np.uint8)
        else:
            vision = np.ascontiguousarray(processed["pixel_values"], dtype=np.float32)

        ids = self.tokenizer.encode(self._prompt(prompt, state, active))
        expanded = []
        image_index = 0
        merge_sq = int(self.image_processor.merge_size) ** 2
        for token in ids:
            if token == self.image_pad_token_id:
                count = int(np.prod(grids[image_index]) // merge_sq)
                expanded.extend([token] * count)
                image_index += 1
            else:
                expanded.append(token)
        if image_index != len(images):
            raise ValueError(
                f"prompt contained {image_index} image placeholders for {len(images)} images"
            )

        dof = observation.get("dof_mask")
        if dof is None:
            dof = active.astype(np.float32)
        dof = np.asarray(dof, dtype=np.float32)
        if dof.shape == (self.action_dim,):
            dof = np.broadcast_to(dof, (self.action_horizon, self.action_dim))
        if dof.shape != (self.action_horizon, self.action_dim):
            raise ValueError(
                f"dof_mask shape {dof.shape}, expected [{self.action_dim}] or "
                f"[{self.action_horizon}, {self.action_dim}]"
            )
        action_mask = np.ascontiguousarray(dof, dtype=np.float32)
        return vision, np.ascontiguousarray(expanded, dtype=np.uint32), action_mask


@register_policy("walloss")
@register_policy("wall_oss_05")
class WallossPolicy:
    """Public WallOSS L2 policy. The websocket server consumes it unchanged."""

    def __init__(
        self,
        model,
        processor,
        *,
        action_min,
        action_delta,
        action_dim: int,
        native_rgb: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ):
        self.model = model
        self.processor = processor
        self._action_min = np.asarray(action_min, np.float32)[:action_dim]
        self._action_delta = np.asarray(action_delta, np.float32)[:action_dim]
        self._action_dim = int(action_dim)
        self._native_rgb = bool(native_rgb)
        self.metadata = {
            "model_type": "walloss",
            "action_horizon": model.action_horizon,
            "action_dim": self._action_dim,
            "model_action_dim": model.action_dim,
            "num_views": model.num_views,
            "image_keys": list(getattr(processor, "image_keys", ())),
            "state_key": getattr(processor, "state_key", None),
            "prompt_key": getattr(processor, "prompt_key", None),
            **(
                {"state_dim": int(processor.propri_min.size)}
                if hasattr(processor, "propri_min")
                else {}
            ),
            "dof_mask_key": "dof_mask",
            "discrete_state": True,
            "state_bins": getattr(processor, "state_bins", None),
            **(dict(metadata) if metadata else {}),
        }

    @classmethod
    def from_pretrained(
        cls,
        model_dir,
        *,
        model=None,
        checkpoint=None,
        device="cuda:0",
        precision="auto",
        tactics=None,
        norm_key="x2_normal",
        action_dim=None,
        image_keys=_DEFAULT_IMAGE_KEYS,
        camera_names=("face_view", "right_wrist_view"),
        state_key="observation/state",
        prompt_key="prompt",
        discrete_state=None,
        state_bins=None,
        processor=None,
        seed=0,
        metadata=None,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"unsupported WallossPolicy options: {sorted(kwargs)}")
        if discrete_state is False:
            raise ValueError(
                "WallOSS checkpoints require discretized state in the prompt; "
                "discrete_state=False is unsupported"
            )
        model_dir = Path(model_dir)
        custom_state_bins = getattr(processor, "state_bins", None)
        if state_bins is not None:
            resolved_state_bins = int(state_bins)
        elif custom_state_bins is not None:
            resolved_state_bins = int(custom_state_bins)
        else:
            resolved_state_bins = _checkpoint_state_bins(model_dir)
        if resolved_state_bins < 2:
            raise ValueError(f"state_bins must be >= 2, got {resolved_state_bins}")
        if model is None:
            import apxinf_py

            ckpt = (
                str(checkpoint)
                if checkpoint is not None
                else str(model_dir / "model.safetensors")
            )
            model = apxinf_py.Model.load(
                "walloss",
                ckpt,
                device=device,
                precision=precision,
                **({"tactics": str(tactics)} if tactics else {}),
                sampling_seed=int(seed),
            )
        width = int(action_dim) if action_dim is not None else int(model.action_dim)
        if width < 1 or width > int(model.action_dim):
            raise ValueError(
                f"action_dim must be in 1..={model.action_dim}, got {width}"
            )
        action_min, action_delta = _load_normalizer(
            model_dir / "normalizer_action.pth", norm_key
        )
        if action_min.size < width or action_delta.size < width:
            raise ValueError(
                f"normalizer {norm_key!r} has width {action_min.size}, requested {width}"
            )
        native_rgb = False
        if processor is None:
            if len(image_keys) != 2 or len(camera_names) != 2:
                raise ValueError(
                    "WallOSS runtime currently requires exactly two camera views"
                )
            native_rgb = bool(getattr(model, "accepts_rgb_u8", False))
            processor = _WallossProcessor(
                model_dir,
                image_keys=image_keys,
                camera_names=camera_names,
                state_key=state_key,
                prompt_key=prompt_key,
                action_horizon=model.action_horizon,
                action_dim=model.action_dim,
                norm_key=norm_key,
                state_bins=resolved_state_bins,
                native_rgb=native_rgb,
            )
        elif not callable(processor):
            raise TypeError("processor must be callable")
        processor_metadata = {
            "image_keys": list(getattr(processor, "image_keys", image_keys)),
            "state_key": getattr(processor, "state_key", state_key),
            "prompt_key": getattr(processor, "prompt_key", prompt_key),
            "state_bins": resolved_state_bins,
        }
        if metadata:
            processor_metadata.update(metadata)
        reset = getattr(model, "reset_sampling", None)
        if callable(reset):
            reset(int(seed))
        return cls(
            model,
            processor,
            action_min=action_min,
            action_delta=action_delta,
            action_dim=width,
            native_rgb=native_rgb,
            metadata=processor_metadata,
        )

    def infer(self, observation: Mapping[str, Any], *, noise=None) -> dict:
        started = time.perf_counter()
        vision, token_ids, action_mask = self.processor(observation)
        model_started = time.perf_counter()
        inference_kwargs = {
            **(
                {"noise": np.ascontiguousarray(noise, dtype=np.float32)}
                if noise is not None
                else {}
            ),
            "action_mask": action_mask,
        }
        if self._native_rgb:
            normalized = self.model.infer_rgb(
                vision, "nhwc", token_ids, **inference_kwargs
            )
        else:
            normalized = self.model._infer_patches(
                vision, token_ids, **inference_kwargs
            )
        normalized = np.asarray(normalized, dtype=np.float32)
        model_ms = (time.perf_counter() - model_started) * 1000
        actions = (normalized[:, : self._action_dim] + 1) * 0.5
        actions = np.ascontiguousarray(
            actions * self._action_delta + self._action_min, np.float32
        )
        return {
            "actions": actions,
            "normalized_actions": normalized,
            "token_ids": token_ids,
            "noise": noise,
            "timing": {
                "model_ms": model_ms,
                "total_ms": (time.perf_counter() - started) * 1000,
            },
        }

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def action_horizon(self):
        return int(self.model.action_horizon)

    def close(self):
        close = getattr(self.model, "close", None)
        if callable(close):
            close()
