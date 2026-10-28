"""X-VLA policy: raw robot observations to deployable action chunks.

X-VLA owns a FLORENCE-2 vision-language stack and a soft-prompted action
transformer. The native runtime accepts resized RGB frames, a proprioceptive
vector, an instruction token sequence and a domain id; this module only turns a
caller's observation dict into those tensors (Pillow resize + BART tokenize) and
returns the model's action chunk unchanged.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..registry import register_policy

__all__ = ["XvlaPolicy"]

_DEFAULT_IMAGE_KEYS = ("image0", "image1")
_DEFAULT_STATE_KEY = "proprio"
_DEFAULT_PROMPT_KEY = "language_instruction"
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


class _XvlaTokenizer:
    """BART tokenizer backed by the native Rust tokenizer.

    The reference processor tokenizes with ``add_special_tokens=True`` and pads
    to a fixed length; the native tokenizer deliberately encodes text only, so
    this wrapper adds BOS/EOS and the pad token explicitly.
    """

    def __init__(self, model_dir: Path, max_token_len: int):
        try:
            import apxinf_py
        except ImportError as error:  # pragma: no cover - dependency error is user-facing
            raise ImportError("XvlaPolicy requires the native apxinf-py package") from error

        path = model_dir / "tokenizer.json"
        if not path.is_file():
            raise FileNotFoundError(f"X-VLA tokenizer file does not exist: {path}")
        self._tokenizer = apxinf_py.HfTokenizer.from_file(str(path))
        self.max_token_len = int(max_token_len)
        self.bos_id = self._require_token("<s>")
        self.eos_id = self._require_token("</s>")
        self.pad_id = self._require_token("<pad>")
        if self.max_token_len < 3:
            raise ValueError(
                f"X-VLA max_token_len must be at least 3 (BOS + token + EOS), "
                f"got {self.max_token_len}"
            )

    def _require_token(self, token: str) -> int:
        value = self._tokenizer.token_to_id(token)
        if value is None:
            raise ValueError(f"X-VLA tokenizer has no token {token!r}")
        return int(value)

    def encode(self, prompt: str) -> np.ndarray:
        body = list(self._tokenizer.encode(prompt))
        # HF truncation keeps both special tokens and cuts the body.
        body = body[: self.max_token_len - 2]
        ids = [self.bos_id, *body, self.eos_id]
        ids.extend([self.pad_id] * (self.max_token_len - len(ids)))
        return np.ascontiguousarray(ids, dtype=np.uint32)


class _XvlaProcessor:
    """Reference observation encoding: two-view RGB, BART ids, ee6d proprio."""

    def __init__(
        self,
        model_dir: Path,
        *,
        image_keys: Sequence[str],
        state_key: str,
        prompt_key: str,
        max_token_len: int,
        proprio_dim: int,
    ):
        self.image_keys = tuple(image_keys)
        self.state_key = str(state_key)
        self.prompt_key = str(prompt_key)
        self.proprio_dim = int(proprio_dim)
        if not self.image_keys:
            raise ValueError("X-VLA requires at least one image key")
        if len(self.image_keys) > 3:
            raise ValueError(
                f"X-VLA accepts at most three camera views, got {len(self.image_keys)}"
            )
        self.tokenizer = _XvlaTokenizer(model_dir, max_token_len)

        config_path = model_dir / "preprocessor_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"X-VLA preprocessor config does not exist: {config_path}"
            )
        document = json.loads(config_path.read_text())
        if not isinstance(document, Mapping):
            raise TypeError(f"{config_path.name} must contain an object")
        if document.get("image_processor_type") not in (None, "CLIPImageProcessor"):
            raise ValueError(
                "unsupported X-VLA image_processor_type "
                f"{document.get('image_processor_type')!r}"
            )
        for flag in ("do_resize", "do_rescale", "do_normalize"):
            if document.get(flag, True) is False:
                raise ValueError(f"X-VLA requires {flag}=true")
        if int(document.get("resample", int(_BICUBIC))) != int(_BICUBIC):
            raise ValueError("X-VLA requires bicubic (resample=3) image resizing")
        size = document.get("size") or {}
        self.image_size = int(size.get("height", 224))
        if int(size.get("width", self.image_size)) != self.image_size:
            raise ValueError("X-VLA requires square image resizing")

    def __call__(self, observation: Mapping[str, Any]):
        prompt = _lookup(observation, self.prompt_key)
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        raw_state = np.asarray(
            _lookup(observation, self.state_key), dtype=np.float32
        ).reshape(-1)
        if raw_state.size > self.proprio_dim:
            raise ValueError(
                f"state has {raw_state.size} values, maximum is {self.proprio_dim}"
            )
        state = np.zeros(self.proprio_dim, dtype=np.float32)
        state[: raw_state.size] = raw_state

        frames = []
        for key in self.image_keys:
            image = np.asarray(_lookup(observation, key))
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError(
                    f"{key} must be HxWx3 uint8 RGB, got {image.shape} {image.dtype}"
                )
            if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
                image = np.asarray(
                    Image.fromarray(image).resize(
                        (self.image_size, self.image_size), _BICUBIC
                    ),
                    dtype=np.uint8,
                )
            frames.append(np.ascontiguousarray(image, dtype=np.uint8).reshape(-1))

        rgb = np.ascontiguousarray(np.concatenate(frames))
        ids = self.tokenizer.encode(prompt)
        return rgb, ids, state


@register_policy("xvla")
class XvlaPolicy:
    """Public X-VLA L2 policy. The websocket server consumes it unchanged."""

    def __init__(
        self,
        model_runner,
        processor,
        *,
        domain_id: int,
        metadata: Mapping[str, Any] | None = None,
    ):
        self.model_runner = model_runner
        self.processor = processor
        self.domain_id = int(domain_id)
        if self.domain_id < 0:
            raise ValueError(f"domain_id must be non-negative, got {self.domain_id}")
        self.metadata = {
            "model_type": "xvla",
            "action_horizon": int(model_runner.action_horizon),
            "action_dim": int(model_runner.action_dim),
            "num_views": int(model_runner.num_views),
            "image_size": int(model_runner.image_size),
            "domain_id": self.domain_id,
            "image_keys": list(processor.image_keys),
            "state_key": processor.state_key,
            "prompt_key": processor.prompt_key,
            "proprio_dim": int(processor.proprio_dim),
            **(dict(metadata) if metadata else {}),
        }

    @classmethod
    def from_pretrained(
        cls,
        model_dir,
        *,
        model_runner=None,
        checkpoint=None,
        device="cuda:0",
        image_keys=_DEFAULT_IMAGE_KEYS,
        state_key=_DEFAULT_STATE_KEY,
        prompt_key=_DEFAULT_PROMPT_KEY,
        domain_id=3,
        processor=None,
        seed=0,
        metadata=None,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"unsupported XvlaPolicy options: {sorted(kwargs)}")
        model_dir = Path(model_dir)
        if model_runner is None:
            import apxinf_py

            model_runner = apxinf_py.ModelRunner.load(
                "xvla",
                str(checkpoint) if checkpoint is not None else str(model_dir),
                device=device,
                sampling_seed=int(seed),
            )
        if processor is None:
            processor = _XvlaProcessor(
                model_dir,
                image_keys=image_keys,
                state_key=state_key,
                prompt_key=prompt_key,
                max_token_len=int(model_runner.max_token_len),
                proprio_dim=int(model_runner.action_dim),
            )
        elif not callable(processor):
            raise TypeError("processor must be callable")
        reset = getattr(model_runner, "reset_sampling", None)
        if callable(reset):
            reset(int(seed))
        return cls(model_runner, processor, domain_id=domain_id, metadata=metadata)

    def infer(self, observation: Mapping[str, Any], *, noise=None) -> dict:
        started = time.perf_counter()
        rgb, token_ids, state = self.processor(observation)
        model_started = time.perf_counter()
        inference_kwargs = (
            {"noise": np.ascontiguousarray(noise, dtype=np.float32)}
            if noise is not None
            else {}
        )
        actions = self.model_runner.infer_rgb_state(
            rgb,
            "nhwc",
            token_ids,
            state,
            self.domain_id,
            **inference_kwargs,
        )
        model_ms = (time.perf_counter() - model_started) * 1000
        actions = np.ascontiguousarray(actions, dtype=np.float32)
        return {
            "actions": actions,
            "token_ids": token_ids,
            "noise": noise,
            "timing": {
                "model_ms": model_ms,
                "total_ms": (time.perf_counter() - started) * 1000,
            },
        }

    @property
    def action_dim(self):
        return int(self.model_runner.action_dim)

    @property
    def action_horizon(self):
        return int(self.model_runner.action_horizon)

    def close(self):
        close = getattr(self.model_runner, "close", None)
        if callable(close):
            close()
