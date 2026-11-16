"""Cosmos3-Edge DROID inference from canonical RGB and one cached prompt.

The native model owns VAE tensor canonicalization, UND/GEN and joint sampling.
Images must already match the declared 544x736 profile. Camera orientation,
source-image resizing and conversion to a robot's action convention belong to
the caller's adapter.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from ...processors.transforms import lookup_key
from ...checkpoints.cosmos3_edge import CHECKPOINT_REVISION, SOURCE_REVISION
from ..base import StateConditionedModelRunnerProtocol
from ..registry import register_policy

class _EdgeRunner(StateConditionedModelRunnerProtocol, Protocol):
    """Existing native RGB binding with the fixed rectangular grid capability."""

    def infer_rgb(self, rgb_u8: np.ndarray, layout: str, token_ids: np.ndarray,
                  noise: np.ndarray | None = None, *, state: np.ndarray | None = None,
                  image_grid_thw: np.ndarray | None = None) -> np.ndarray: ...


__all__ = ["Cosmos3EdgePolicy"]

_PROFILE = "droid-policy-v1"
_LATENT_SHAPE = (1, 48 * 5 * 34 * 46 + 16 * 64)
_IMAGE_SHAPE = (544, 736, 3)
_FIXED_CONFIG = {
    "model_type": "cosmos3-edge", "profile": _PROFILE,
    "format_version": 1, "precision": "bf16", "num_tasks": 1,
    "hidden_size": 2048, "num_layers": 28, "num_heads": 16,
    "num_kv_heads": 8, "head_dim": 128, "intermediate_size": 9216,
    "rms_norm_eps": 1e-5, "rope_theta": 1e8, "mrope_section": [24, 20, 20],
    "latent_channel": 48, "latent_patch_size": 2,
    "latent_shape": [48, 5, 34, 46], "action_chunk": 16,
    "raw_action_dim": 10, "max_action_dim": 64,
    "num_inference_steps": 4, "flow_shift": 5.0,
    "num_train_timesteps": 1000, "num_embodiment_domains": 32,
    "domain_id": 8, "vocab_size": 131072, "max_text_tokens": 126,
    "image_size": [544, 736], "pixel_frames": 1,
    "source_revision": SOURCE_REVISION, "checkpoint_revision": CHECKPOINT_REVISION,
}


def _verify_assets(root: Path, config: Mapping[str, Any]) -> list[str]:
    for key, value in _FIXED_CONFIG.items():
        if config.get(key) != value:
            raise ValueError(f"Cosmos3-Edge requires prepared {_PROFILE} assets: {key}")
    prompts = json.loads((root / "prompts.json").read_text())["prompts"]
    if (not isinstance(prompts, list) or len(prompts) != 1
            or not isinstance(prompts[0], str) or not prompts[0].strip()):
        raise ValueError("Cosmos3-Edge requires one nonempty cached prompt")
    manifest_path = root / "conversion.json"
    if not manifest_path.is_file():
        raise ValueError("Cosmos3-Edge prepared assets require conversion.json")
    manifest = json.loads(manifest_path.read_text())
    hashes = manifest.get("output_sha256") if isinstance(manifest, Mapping) else None
    if not isinstance(hashes, Mapping):
        raise ValueError("Cosmos3-Edge conversion.json requires output_sha256")
    for name in ("model.safetensors", "vae_encoder.safetensors", "text.safetensors", "prompts.json"):
        if name not in hashes:
            raise ValueError(f"Cosmos3-Edge conversion.json requires a SHA256 for {name}")
    for name, digest in hashes.items():
        if (not isinstance(name, str) or Path(name).name != name
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None):
            raise ValueError(f"Cosmos3-Edge prepared asset SHA256 is invalid: {name}")
        path = root / name
        if not path.is_file():
            raise ValueError(f"Cosmos3-Edge prepared asset is missing: {name}")
        actual = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                actual.update(block)
        if actual.hexdigest() != digest.lower():
            raise ValueError(f"Cosmos3-Edge prepared asset SHA256 mismatch: {name}")
    return prompts


@register_policy("cosmos3-edge")
class Cosmos3EdgePolicy:
    """Single-view, four-step BF16 DROID policy with one cached instruction.

    ``actions`` contains the native 16x10 DROID output, including the reference
    gripper inversion. A robot adapter must select/convert the dimensions it
    consumes; the policy does not infer an embodiment from observation keys.
    """

    def __init__(self, model_runner: _EdgeRunner, *,
                 prompts: Sequence[str], image_key: str = "image",
                 prompt_key: str = "prompt"):
        if (len(prompts) != 1 or not isinstance(prompts[0], str)
                or not prompts[0].strip()):
            raise ValueError("Cosmos3-Edge requires one nonempty cached prompt")
        if any(not isinstance(key, str) or not key for key in (image_key, prompt_key)):
            raise ValueError("Cosmos3-Edge requires nonempty image_key and prompt_key")
        if (model_runner.action_horizon, model_runner.action_dim,
                model_runner.num_views, model_runner.image_size) != (16, 10, 1, 544):
            raise ValueError("Cosmos3-Edge runner does not match the DROID direct profile")
        if tuple(model_runner.initial_latent_shape) != _LATENT_SHAPE:
            raise ValueError("Cosmos3-Edge requires the complete joint latent noise shape")
        self.model_runner: _EdgeRunner | None = model_runner
        self.prompt = prompts[0]
        self.image_key = image_key
        self.prompt_key = prompt_key
        self.metadata = {
            "model_type": "cosmos3-edge", "profile": _PROFILE,
            "action_horizon": 16, "action_dim": 10, "num_views": 1,
            "initial_latent_shape": list(_LATENT_SHAPE),
            "image_shape": list(_IMAGE_SHAPE), "image_key": image_key,
            "prompt_key": prompt_key, "supported_prompts": tuple(prompts),
            "image_preprocessing": "canonical-rgb-544x736",
        }

    @classmethod
    def from_pretrained(cls, model_dir, *,
                        model_runner: _EdgeRunner | None = None,
                        image_key: str = "image", prompt_key: str = "prompt",
                        device: str = "cuda:0", **load_options) -> "Cosmos3EdgePolicy":
        root = Path(model_dir)
        config = json.loads((root / "config.json").read_text())
        prompts = _verify_assets(root, config)
        if model_runner is None:
            import apxinf_py
            model_runner = apxinf_py.ModelRunner.load(
                "cosmos3-edge", str(root), device=device, **load_options)
        elif load_options:
            raise ValueError("load options cannot be applied to an injected model_runner")
        return cls(model_runner, prompts=prompts, image_key=image_key, prompt_key=prompt_key)

    @property
    def action_horizon(self) -> int:
        return 16

    @property
    def action_dim(self) -> int:
        return 10

    def infer(self, observation: Mapping[str, Any], *, noise: np.ndarray | None = None) -> dict:
        if self.model_runner is None:
            raise RuntimeError("Cosmos3-Edge Policy is closed")
        started = time.perf_counter()
        if lookup_key(observation, self.prompt_key) != self.prompt:
            raise ValueError("Cosmos3-Edge prompt is absent from this checkpoint's cached conditions")
        image = np.asarray(lookup_key(observation, self.image_key))
        if image.dtype != np.uint8 or image.shape != _IMAGE_SHAPE:
            raise ValueError("Cosmos3-Edge requires canonical [544,736,3] uint8 RGB")
        rgb = np.ascontiguousarray(image[None])
        selected_noise = noise if noise is not None else observation.get("noise")
        if selected_noise is not None:
            selected_noise = np.ascontiguousarray(selected_noise, dtype=np.float32)
            if selected_noise.shape != _LATENT_SHAPE or not np.isfinite(selected_noise).all():
                raise ValueError(f"Cosmos3-Edge noise must be finite with shape {_LATENT_SHAPE}")
        tokens = np.asarray([0], dtype=np.uint32)
        model_started = time.perf_counter()
        # The existing binding uses this explicit grid for rectangular RGB;
        # without it the default RGB contract would interpret image_size as a
        # square. patch_size=1 makes these grid extents the pixel dimensions.
        grid = np.asarray([[1, 544, 736]], dtype=np.uint32)
        actions = np.array(self.model_runner.infer_rgb(
            rgb, "nhwc", tokens, selected_noise, image_grid_thw=grid),
                           dtype=np.float32, copy=True)
        model_ms = (time.perf_counter() - model_started) * 1000
        if actions.shape != (16, 10) or not np.isfinite(actions).all():
            raise ValueError("Cosmos3-Edge runtime returned invalid actions")
        return {"actions": actions, "timing": {
            "model_ms": model_ms, "total_ms": (time.perf_counter() - started) * 1000}}

    __call__ = infer

    def close(self) -> None:
        runner = self.model_runner
        self.model_runner = None
        close = getattr(runner, "close", None)
        if close is not None:
            close()
