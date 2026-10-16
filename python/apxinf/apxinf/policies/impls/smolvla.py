"""Native SmolVLA policy for LIBERO-style observations."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from ...processors.transforms import has_key, lookup_key
from ..registry import register_policy

__all__ = ["SmolVlaPolicy"]


class _SmolVlaTokenizer:
    """Hugging Face tokenizer.json backed by ApxInf's native Rust binding."""

    def __init__(self, tokenizer_path: Path):
        try:
            import apxinf_py
        except ImportError as error:
            raise ImportError(
                "SmolVlaPolicy requires the native apxinf-py package"
            ) from error

        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"SmolVLA tokenizer not found: {tokenizer_path}")
        self._tokenizer = apxinf_py.HfTokenizer.from_file(str(tokenizer_path))
        self._config = self._load_config(tokenizer_path)
        self.pad_id = self._resolve_pad_id()

    @staticmethod
    def _load_config(tokenizer_path: Path) -> Mapping[str, Any]:
        config_path = tokenizer_path.parent / "tokenizer_config.json"
        if not config_path.is_file():
            return {}
        document = json.loads(config_path.read_text())
        if not isinstance(document, Mapping):
            raise ValueError(f"{config_path.name} must contain an object")
        return document

    def _resolve_pad_id(self) -> int:
        pad_id = self._config.get("pad_token_id")
        if pad_id is not None:
            return int(pad_id)
        token = self._config.get("pad_token") or self._config.get("eos_token")
        if token is None:
            raise ValueError("SmolVLA tokenizer config has no pad or EOS token")
        resolved = self._tokenizer.token_to_id(token)
        if resolved is None:
            raise ValueError(f"SmolVLA tokenizer has no pad token {token!r}")
        return int(resolved)

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode_with_special_tokens(text, True))


def _read_safetensors_f32(path: Path, name: str) -> np.ndarray:
    """Read one F32 tensor from a safetensors sidecar without ML dependencies."""
    with path.open("rb") as file:
        header_length = int.from_bytes(file.read(8), "little")
        header = json.loads(file.read(header_length))
        if not isinstance(header, Mapping):
            raise ValueError(f"{path.name} safetensors header must be an object")
        entry = header.get(name)
        if not isinstance(entry, Mapping) or entry.get("dtype") != "F32":
            raise ValueError(f"{path.name} has no F32 tensor {name!r}")
        shape = [int(size) for size in entry.get("shape", [])]
        offsets = entry.get("data_offsets")
        if not isinstance(offsets, Sequence) or len(offsets) != 2:
            raise ValueError(f"{path.name} tensor {name!r} has invalid offsets")
        start, end = map(int, offsets)
        count = math.prod(shape) if shape else 1
        if count * 4 != end - start:
            raise ValueError(f"{path.name} tensor {name!r} has invalid length")
        file.seek(8 + header_length + start)
        values = np.frombuffer(file.read(end - start), dtype="<f4")
        return np.ascontiguousarray(values.reshape(shape), dtype=np.float32)


def _sidecar_root(model_dir: Path) -> Path:
    sidecar = model_dir / "smolvla_libero"
    if (sidecar / "tokenizer.json").is_file() or (sidecar / "config.json").is_file():
        return sidecar
    return model_dir


def _resolve_file(model_dir: Path, filename: str) -> Path:
    for root in (model_dir, _sidecar_root(model_dir)):
        candidate = root / filename
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"SmolVLA {filename} not found under {model_dir}")


def _default_image_keys(model_dir: Path, num_views: int) -> Tuple[str, ...]:
    preprocessor_path = _sidecar_root(model_dir) / "policy_preprocessor.json"
    keys = []
    if preprocessor_path.is_file():
        document = json.loads(preprocessor_path.read_text())
        if not isinstance(document, Mapping):
            raise ValueError(f"{preprocessor_path.name} must contain an object")
        rename_map = document.get("rename_map")
        if isinstance(rename_map, Mapping):
            keys = [str(value) for value in rename_map.values()]
    if not keys:
        keys = [f"observation.images.camera{index + 1}" for index in range(num_views)]
    if len(keys) < num_views:
        raise ValueError(
            f"SmolVLA checkpoint names {len(keys)} cameras but model uses {num_views}"
        )
    return tuple(keys[:num_views])


@register_policy("smolvla")
@register_policy("smolvla_libero")
class SmolVlaPolicy:
    """Observation-to-action policy for native BF16 SmolVLA."""

    def __init__(
        self,
        model_runner,
        *,
        tokenizer,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        action_mean: np.ndarray,
        action_std: np.ndarray,
        image_keys: Sequence[str],
        state_key: str = "observation.state",
        prompt_key: str = "task",
        metadata: Optional[Mapping[str, Any]] = None,
    ):
        self.model_runner = model_runner
        self.tokenizer = tokenizer
        self.state_mean = np.asarray(state_mean, dtype=np.float32)
        self.state_std = np.asarray(state_std, dtype=np.float32)
        self.action_mean = np.asarray(action_mean, dtype=np.float32)
        self.action_std = np.asarray(action_std, dtype=np.float32)
        self.image_keys = tuple(image_keys)
        self.state_key = state_key
        self.prompt_key = prompt_key

        if not self.image_keys or len(self.image_keys) != model_runner.num_views:
            raise ValueError(
                f"SmolVlaPolicy requires {model_runner.num_views} image keys, "
                f"got {len(self.image_keys)}"
            )
        if self.state_mean.shape != self.state_std.shape:
            raise ValueError("SmolVLA state mean/std shapes differ")
        if self.action_mean.shape != self.action_std.shape:
            raise ValueError("SmolVLA action mean/std shapes differ")
        if self.action_mean.size != model_runner.action_dim:
            raise ValueError(
                f"SmolVLA action stats have {self.action_mean.size} values, "
                f"expected {model_runner.action_dim}"
            )

        self.metadata = {
            "model_type": "smolvla",
            "model_variant": getattr(model_runner, "model_variant", "bf16"),
            "image_keys": list(self.image_keys),
            "state_key": self.state_key,
            "prompt_key": self.prompt_key,
            "action_horizon": int(model_runner.action_horizon),
            "action_dim": int(model_runner.action_dim),
            "state_dim": int(self.state_mean.size),
            "num_views": int(model_runner.num_views),
            "image_size": int(model_runner.image_size),
            "max_token_len": int(model_runner.max_token_len),
            **(dict(metadata) if metadata else {}),
        }

    @classmethod
    def from_pretrained(
        cls,
        model_dir,
        *,
        device: str = "cuda:0",
        seed: int = 0,
        model_runner=None,
        action_dim: Optional[int] = None,
        image_keys: Optional[Sequence[str]] = None,
        num_views: Optional[int] = None,
        tactics: Optional[Path | str] = None,
        autotune: bool = False,
        model_variant: str = "bf16",
        state_key: str = "observation.state",
        prompt_key: str = "task",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "SmolVlaPolicy":
        if model_variant not in ("bf16", "fp16"):
            raise ValueError("SmolVlaPolicy model_variant must be bf16 or fp16")
        if action_dim is not None and model_runner is not None and action_dim != model_runner.action_dim:
            raise ValueError(
                f"SmolVlaPolicy action_dim={action_dim} conflicts with model "
                f"action_dim={model_runner.action_dim}"
            )
        model_dir = Path(model_dir)
        if model_runner is None:
            try:
                import apxinf_py
            except ImportError as error:
                raise ImportError(
                    "SmolVlaPolicy requires the native apxinf-py package"
                ) from error
            checkpoint = (
                model_dir / "smolvla_libero_model.safetensors"
                if (model_dir / "smolvla_libero_model.safetensors").is_file()
                else model_dir / "model.safetensors"
            )
            if not checkpoint.is_file():
                checkpoint = model_dir
            model_runner = apxinf_py.ModelRunner.load(
                "smolvla_libero",
                checkpoint,
                device=device,
                precision="auto",
                sampling_seed=int(seed),
                num_views=num_views,
                tactics=str(tactics) if tactics is not None else None,
                autotune=bool(autotune),
                model_variant=model_variant,
            )

        tokenizer = _SmolVlaTokenizer(_resolve_file(model_dir, "tokenizer.json"))
        state_file = _resolve_file(
            model_dir, "policy_preprocessor_step_5_normalizer_processor.safetensors"
        )
        action_file = _resolve_file(
            model_dir, "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
        )
        resolved_image_keys = tuple(
            image_keys
            if image_keys is not None
            else _default_image_keys(model_dir, model_runner.num_views)
        )
        return cls(
            model_runner,
            tokenizer=tokenizer,
            state_mean=_read_safetensors_f32(state_file, "observation.state.mean"),
            state_std=_read_safetensors_f32(state_file, "observation.state.std"),
            action_mean=_read_safetensors_f32(action_file, "action.mean"),
            action_std=_read_safetensors_f32(action_file, "action.std"),
            image_keys=resolved_image_keys,
            state_key=state_key,
            prompt_key=prompt_key,
            metadata=metadata,
        )

    def infer(
        self,
        observation: Mapping[str, Any],
        *,
        noise: Optional[np.ndarray] = None,
        profile: bool = False,
    ) -> dict:
        started = time.perf_counter()
        required = list(self.image_keys) + [self.state_key]
        if not has_key(observation, self.prompt_key):
            required.append(self.prompt_key)
        missing = [key for key in required if not has_key(observation, key)]
        if missing:
            raise KeyError(
                f"SmolVlaPolicy.infer: missing observation keys: {missing}; "
                f"observation has {sorted(observation)}"
            )

        token_ids = self._token_ids(lookup_key(observation, self.prompt_key))
        rgb = self._rgb(observation)
        state = self._state(lookup_key(observation, self.state_key))
        selected_noise = None
        if noise is not None:
            selected_noise = np.ascontiguousarray(noise, dtype=np.float32)
            expected = (self.model_runner.action_horizon, self.model_runner.action_dim)
            if selected_noise.shape != expected:
                raise ValueError(
                    f"SmolVlaPolicy noise shape {selected_noise.shape} != {expected}"
                )

        model_started = time.perf_counter()
        runner_method = (
            self.model_runner.infer_rgb_profiled
            if profile
            else self.model_runner.infer_rgb
        )
        runner_result = runner_method(
            rgb,
            "nhwc",
            token_ids,
            selected_noise,
            state=state,
        )
        if profile:
            normalized, phase_profile = runner_result
            normalized = np.asarray(normalized, dtype=np.float32)
        else:
            normalized = np.asarray(runner_result, dtype=np.float32)
            phase_profile = None
        model_ms = (time.perf_counter() - model_started) * 1000.0
        expected = (self.model_runner.action_horizon, self.model_runner.action_dim)
        if normalized.shape != expected:
            raise ValueError(
                f"SmolVLA returned action shape {normalized.shape}, expected {expected}"
            )
        if not np.isfinite(normalized).all():
            raise FloatingPointError("SmolVLA returned non-finite normalized actions")
        actions = np.ascontiguousarray(
            normalized * self.action_std + self.action_mean, dtype=np.float32
        )
        total_ms = (time.perf_counter() - started) * 1000.0
        return {
            "actions": actions,
            "normalized_actions": normalized,
            "token_ids": token_ids,
            "noise": selected_noise,
            "timing": {
                "model_ms": model_ms,
                "total_ms": total_ms,
                **({"phases": phase_profile} if phase_profile is not None else {}),
            },
            "metadata": self.metadata,
        }

    __call__ = infer

    @property
    def action_dim(self) -> int:
        return int(self.model_runner.action_dim)

    @property
    def action_horizon(self) -> int:
        return int(self.model_runner.action_horizon)

    def close(self) -> None:
        close = getattr(self.model_runner, "close", None)
        if callable(close):
            close()

    def _token_ids(self, prompt) -> np.ndarray:
        if not isinstance(prompt, str):
            raise TypeError(f"SmolVLA prompt must be a string, got {type(prompt)!r}")
        text = prompt if prompt.endswith("\n") else f"{prompt}\n"
        tokens = self.tokenizer.encode(text)[: self.model_runner.max_token_len]
        if not tokens:
            raise ValueError("SmolVLA tokenized prompt is empty")
        return np.asarray(tokens, dtype=np.uint32)

    def _rgb(self, observation: Mapping[str, Any]) -> np.ndarray:
        from PIL import Image

        frames = []
        target = self.model_runner.image_size
        for key in self.image_keys:
            image = np.asarray(lookup_key(observation, key))
            if image.ndim != 3 or 3 not in image.shape:
                raise ValueError(f"SmolVLA image {key!r} must be HWC or CHW uint8")
            if image.shape[0] == 3 and image.shape[-1] != 3:
                image = np.ascontiguousarray(image.transpose(1, 2, 0))
            if image.shape[-1] != 3:
                raise ValueError(f"SmolVLA image {key!r} must have 3 channels")
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)

            height, width = image.shape[:2]
            ratio = max(width / target, height / target)
            resized_width = max(1, int(width / ratio))
            resized_height = max(1, int(height / ratio))
            resized = Image.fromarray(image).resize(
                (resized_width, resized_height), resample=Image.BILINEAR
            )
            canvas = Image.new("RGB", (target, target), (0, 0, 0))
            canvas.paste(resized, (target - resized_width, target - resized_height))
            frames.append(np.asarray(canvas, dtype=np.uint8))
        return np.ascontiguousarray(np.stack(frames))

    def _state(self, raw_state) -> np.ndarray:
        state = np.asarray(raw_state, dtype=np.float32)
        if state.ndim != 1:
            raise ValueError(f"SmolVLA state must be 1-D, got shape {state.shape}")
        if state.size != self.state_mean.size:
            raise ValueError(
                f"SmolVLA state has {state.size} values, expected "
                f"{self.state_mean.size}"
            )
        if not np.isfinite(state).all():
            raise FloatingPointError("SmolVLA state contains non-finite values")
        return np.ascontiguousarray(
            (state - self.state_mean) / self.state_std, dtype=np.float32
        )
