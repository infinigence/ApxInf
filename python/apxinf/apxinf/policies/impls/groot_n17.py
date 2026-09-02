"""Native ApxInf policy for the frozen NVIDIA GR00T N1.7 LIBERO profile."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..registry import register_policy
from ...processors.transforms import lookup_key

__all__ = ["GrootN17Policy"]

_IMAGE_TOKEN = 151655
_STATE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
_ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


@register_policy("Gr00tN1d7")
class GrootN17Policy:
    """One- or two-view LIBERO policy backed only by ApxInf CUDA execution."""

    def __init__(self, model, tokenizer, statistics: Mapping, *,
                 image_keys: Sequence[str] = ("observation/image", "observation/wrist_image"),
                 state_prefix: str = "observation/state", prompt_key: str = "prompt"):
        self.model = model
        self.tokenizer = tokenizer
        self.statistics = statistics
        self.image_keys = tuple(image_keys)
        if len(self.image_keys) not in (1, 2):
            raise ValueError("GrootN17Policy requires one or two image keys")
        self.state_prefix = state_prefix.rstrip("/")
        self.prompt_key = prompt_key
        self.metadata = {"model_type": "Gr00tN1d7", "action_horizon": 16,
            "action_dim": 7, "model_action_horizon": 40, "model_action_dim": 132,
            "num_views": len(self.image_keys), "image_size": [256, 256],
            "image_keys": list(self.image_keys),
            "prompt_key": prompt_key, "state_keys": list(_STATE_KEYS),
            "precision": "bf16", "cuda_graph": "whole-model"}

    @classmethod
    def from_pretrained(cls, model_dir, *, backbone_dir=None, model=None,
                        device="cuda:0", precision="bf16", image_keys=None,
                        state_prefix="observation/state", prompt_key="prompt", seed=0, **kwargs):
        if kwargs:
            raise TypeError(f"unsupported GR00T policy options: {sorted(kwargs)}")
        model_dir = Path(model_dir)
        if model is None:
            import apxinf_py
            selected_keys = tuple(image_keys or ("observation/image", "observation/wrist_image"))
            model = apxinf_py.Model.load("Gr00tN1d7", model_dir, device=device,
                                         precision=precision, num_views=len(selected_keys),
                                         sampling_seed=int(seed))
        tokenizer_root = Path(backbone_dir) if backbone_dir is not None else model_dir
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ImportError("GrootN17Policy requires `pip install apxinf[groot]`") from exc
        tokenizer_file = tokenizer_root / "tokenizer.json"
        if not tokenizer_file.is_file():
            raise FileNotFoundError(
                f"GR00T tokenizer not found at {tokenizer_file}; pass backbone_dir=Cosmos-Reason2-2B")
        tokenizer = Tokenizer.from_file(str(tokenizer_file))
        stats_path = model_dir / "statistics.json"
        statistics = json.loads(stats_path.read_text())["libero_sim"]
        return cls(model, tokenizer, statistics,
                   image_keys=image_keys or ("observation/image", "observation/wrist_image"),
                   state_prefix=state_prefix, prompt_key=prompt_key)

    def infer(self, observation: Mapping, *, noise: np.ndarray | None = None):
        started = time.perf_counter()
        missing = [key for key in (*self.image_keys, self.prompt_key)
                   if not _has_key(observation, key)]
        if missing:
            raise KeyError(f"GrootN17Policy missing observation keys: {missing}")
        images = np.ascontiguousarray(
            np.stack([_prepare_image(lookup_key(observation, key)) for key in self.image_keys]),
            dtype=np.uint8,
        )
        prompt = str(lookup_key(observation, self.prompt_key))
        token_ids = self._token_ids(prompt)
        state = self._state(observation)
        model_started = time.perf_counter()
        normalized = self.model.infer_rgb(
            images,
            "nhwc",
            np.asarray(token_ids, dtype=np.uint32),
            **({"noise": np.ascontiguousarray(noise, dtype=np.float32)} if noise is not None else {}),
            state=np.ascontiguousarray(state[None], dtype=np.float32),
        )
        model_ms = (time.perf_counter() - model_started) * 1000.0
        normalized = np.asarray(normalized, dtype=np.float32)[:16, :7]
        actions = self._decode(normalized)
        return {"actions": actions, "normalized_actions": normalized,
                "token_ids": np.asarray(token_ids, dtype=np.uint32),
                "timing": {"model_ms": model_ms,
                           "total_ms": (time.perf_counter() - started) * 1000.0},
                "metadata": self.metadata}

    __call__ = infer

    def _token_ids(self, prompt: str) -> list[int]:
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        if len(prompt_ids) != 5:
            raise ValueError(
                f"GR00T validated CUDA graph accepts a five-token instruction, got {len(prompt_ids)}")
        ids = [151644, 872, 198, 151652] + [_IMAGE_TOKEN] * 64 + [151653]
        if len(self.image_keys) == 2:
            ids += [151652] + [_IMAGE_TOKEN] * 64 + [151653]
        ids += prompt_ids + [151645, 198]
        assert len(ids) == (76 if len(self.image_keys) == 1 else 142)
        return ids

    def _state(self, observation: Mapping) -> np.ndarray:
        values = []
        stats = self.statistics["state"]
        for key in _STATE_KEYS:
            path = f"{self.state_prefix}/{key}"
            if not _has_key(observation, path):
                raise KeyError(f"GrootN17Policy missing state key {path!r}")
            raw = np.asarray(lookup_key(observation, path), dtype=np.float32).reshape(-1)
            low = np.asarray(stats[key]["q01"], dtype=np.float32)
            high = np.asarray(stats[key]["q99"], dtype=np.float32)
            values.extend(np.clip(2.0 * (raw - low) / (high - low + 1e-6) - 1.0, -1.0, 1.0))
        output = np.zeros(132, dtype=np.float32)
        output[:len(values)] = values
        return output

    def _decode(self, normalized: np.ndarray) -> np.ndarray:
        output = np.empty_like(normalized)
        stats = self.statistics["action"]
        for index, key in enumerate(_ACTION_KEYS):
            low = float(stats[key]["q01"][0]); high = float(stats[key]["q99"][0])
            output[:, index] = (normalized[:, index] + 1.0) * 0.5 * (high - low) + low
        return output

    @property
    def action_dim(self): return 7

    @property
    def action_horizon(self): return 16

    def close(self):
        close = getattr(self.model, "close", None)
        if callable(close): close()


def _prepare_image(value) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("GrootN17Policy requires `pip install apxinf[groot]`") from exc
    image = np.asarray(value)
    while image.ndim > 3: image = image[0]
    image = cv2.resize(image.astype(np.uint8), (256, 256), interpolation=cv2.INTER_AREA)
    # The frozen checkpoint uses the newer shortest-edge/crop-fraction path:
    # int(256 * .95) = 243, centered at offset 6, then resized to 256.
    image = cv2.resize(image[6:249, 6:249], (256, 256), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(image, dtype=np.uint8)


def _has_key(mapping: Mapping, path: str) -> bool:
    try: lookup_key(mapping, path); return True
    except (KeyError, TypeError): return False
