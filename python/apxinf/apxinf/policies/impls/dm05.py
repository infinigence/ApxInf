"""Native ApxInf policy for the pinned DM05-libero checkpoint.

Python owns the public observation contract, exact image padding/resize,
Gemma3 tokenization and action unnormalization. Model structure, weights,
diffusion schedule and device execution are owned by Rust ``Dm05VlaRuntime``
loaded through :class:`apxinf.Model`; OpenDM is a private reference only and is
not imported or accepted as a runtime fallback.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError

from ..registry import register_policy

__all__ = ["Dm05Policy"]

MODEL_REVISION = "25a8e0d38a8eaeaae41a44d7b4a2378fd8ce1088"
MODEL_SHA256 = "575d0d8e0f75822e95f7adf3a5e62a7c331da0b82e6fc3efeea19ef1b927353f"
MODEL_SIZE = 11_658_431_136
TOKENIZER_SHA256 = "daab2354f8a74e70d70b4d1f804939b68a8c9624dd06cb7858e52dd8970e9726"
TOKENIZER_SIZE = 33_384_567
TRANSFORMERS_VERSION = "5.3.0"

ACTION_HORIZON = 10
ACTION_DIM = 7
MODEL_ACTION_DIM = 32
STATE_DIM = 8
IMAGE_SIZE = 448
IMAGE_PROMPTS = ("Head", "Left wrist")
DIFFUSION_STEPS = 10
MAX_PREFIX_LEN = 768

_TORCH_SEED_MIN = -(1 << 63)
_TORCH_SEED_MAX = (1 << 64) - 1

_CHECKPOINT_MANIFEST = {
    "chat_template.jinja": (
        1_532,
        "7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4",
    ),
    "config.json": (
        6_795,
        "43b2a56ed9c79c3068849caa0a140458515e654d34ae0b06bb0bbb3ef4dd0f80",
    ),
    "generation_config.json": (
        204,
        "640dbc106facaf0fb90980b5e182ce0c1fcfad6e88da14737578b5b65cb42f7a",
    ),
    "model.safetensors": (MODEL_SIZE, MODEL_SHA256),
    "norm_stats.json": (
        1_900,
        "06382f26d9f9fdba10ee2dba77783ec8c31e6a6dcb348806583cb6217e18303b",
    ),
    "processor_config.json": (
        560,
        "9eb2e8baf401c81b1517343d1dfc799a4c1b2238acaece111fe68f5fbe3a8d57",
    ),
    "tokenizer.json": (TOKENIZER_SIZE, TOKENIZER_SHA256),
    "tokenizer_config.json": (
        715,
        "eb28e3a9807f77cd74dce1b8aed91884621c0302941794470c5a46f884462615",
    ),
}

_CANONICAL_METADATA = {
    "schema",
    "model_type",
    "model_revision",
    "backend",
    "device",
    "precision",
    "action_horizon",
    "action_dim",
    "model_action_dim",
    "state_dim",
    "state_conditioned",
    "image_size",
    "image_prompts",
    "robot_type",
    "diffusion_steps",
    "max_prefix_len",
    "sampling_rng",
    "concurrency",
}


class NativeBackend(Protocol):
    metadata: Mapping[str, Any]

    def infer(
        self,
        rgb_u8: np.ndarray,
        token_ids: np.ndarray,
        *,
        seed: int,
        noise: np.ndarray | None,
    ) -> tuple[np.ndarray, float]: ...

    def close(self) -> None: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_file(path: Path, *, size: int, sha256: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"DM05 required file is missing or not regular: {path}")
    if path.stat().st_size != size:
        raise ValueError(
            f"DM05 file size mismatch for {path.name}: "
            f"{path.stat().st_size} != {size}"
        )
    observed = _sha256(path)
    if observed != sha256:
        raise ValueError(f"DM05 SHA-256 mismatch for {path.name}: {observed} != {sha256}")


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def _validate_checkpoint(model_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    for filename, (size, sha256) in _CHECKPOINT_MANIFEST.items():
        _verified_file(model_dir / filename, size=size, sha256=sha256)

    config = _read_json_object(model_dir / "config.json", label="DM05 config")
    if config.get("model_type") != "dm05":
        raise ValueError("DM05 config.model_type must be 'dm05'")
    if config.get("architectures") != ["DM05ForConditionalGeneration"]:
        raise ValueError("DM05 architecture identity mismatch")
    if config.get("dtype") != "bfloat16" or config.get("action_dim") != MODEL_ACTION_DIM:
        raise ValueError("DM05 deployment supports the pinned BF16/32D checkpoint only")

    document = _read_json_object(model_dir / "norm_stats.json", label="norm stats")
    profiles = document.get("norm_stats")
    if not isinstance(profiles, dict) or not isinstance(profiles.get("action"), dict):
        raise ValueError("norm_stats.json must contain norm_stats.action")
    action = profiles["action"]

    def vector(name: str) -> np.ndarray:
        try:
            value = np.asarray(action[name], dtype=np.float32)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"norm_stats.action.{name} is invalid") from exc
        if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
            raise ValueError(f"norm_stats.action.{name} must be {ACTION_DIM} finite values")
        return value

    low, high = vector("q01"), vector("q99")
    if not np.all(high > low):
        raise ValueError("norm_stats action q99 must exceed q01")
    return low, high


def _validate_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    values = dict(metadata)
    conflicts = sorted(set(values) & _CANONICAL_METADATA)
    if conflicts:
        raise ValueError(f"DM05 metadata cannot override canonical fields: {conflicts}")
    return values


def _number_list(value: Any, *, width: int, label: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != width:
        raise ValueError(f"{label} must contain exactly {width} numbers")
    try:
        result = np.asarray(value, dtype=np.float32)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain exactly {width} finite numbers") from exc
    if result.shape != (width,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain exactly {width} finite numbers")
    return result


def _decode_image(value: Any, *, label: str) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.ndim != 3 or array.shape[2] not in (3, 4) or array.dtype != np.uint8:
            raise ValueError(f"{label} numpy image must be uint8 HWC RGB/RGBA")
        return Image.fromarray(array[..., :3], mode="RGB")
    if not isinstance(value, str):
        raise ValueError(f"{label} must be base64, PIL, or uint8 HWC")
    payload = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    try:
        raw = base64.b64decode(payload, validate=True)
        return Image.open(BytesIO(raw)).convert("RGB")
    except (binascii.Error, OSError, UnidentifiedImageError, ValueError) as exc:
        raise ValueError(f"{label} is not a valid base64 image") from exc


def _seed_u64(seed: int) -> int:
    if seed < _TORCH_SEED_MIN or seed > _TORCH_SEED_MAX:
        raise ValueError(
            "sampling.seed must be in the inclusive range "
            f"[{_TORCH_SEED_MIN}, {_TORCH_SEED_MAX}]"
        )
    return seed if seed >= 0 else ((1 << 64) - 1 + seed)


class ApxInfNativeBackend:
    """Thin owner of the native PyO3 handle; no preprocessing lives here."""

    def __init__(self, model_dir: Path, *, device: str, default_seed: int) -> None:
        from apxinf import Model

        model = Model.load(
            "dm05",
            str(model_dir / "model.safetensors"),
            device=device,
            precision="bf16",
            action_horizon=ACTION_HORIZON,
            sampling_seed=_seed_u64(default_seed),
        )
        expected = {
            "action_horizon": ACTION_HORIZON,
            "action_dim": MODEL_ACTION_DIM,
            "num_views": 2,
            "image_size": IMAGE_SIZE,
            "patch_size": 14,
        }
        for field, value in expected.items():
            if int(getattr(model, field)) != value:
                raise RuntimeError(
                    f"native DM05 {field} mismatch: {getattr(model, field)!r} != {value}"
                )
        self.model = model
        self.metadata = {
            "backend": "apxinf-native",
            "model_revision": MODEL_REVISION,
            "device": str(model.device),
            "precision": "bf16",
        }

    def infer(
        self,
        rgb_u8: np.ndarray,
        token_ids: np.ndarray,
        *,
        seed: int,
        noise: np.ndarray | None,
    ) -> tuple[np.ndarray, float]:
        started = time.perf_counter()
        if noise is None:
            output = self.model.infer_rgb_seeded(
                rgb_u8, "nhwc", token_ids, _seed_u64(seed), 0, 0
            )
        else:
            output = self.model.infer_rgb(rgb_u8, "nhwc", token_ids, noise)
        return np.asarray(output, dtype=np.float32), (time.perf_counter() - started) * 1000.0

    def close(self) -> None:
        close = getattr(self.model, "close", None)
        if callable(close):
            close()
        self.model = None


@register_policy("dm05")
class Dm05Policy:
    """Pinned DM05-libero L2 policy over native Rust ``VlaRuntime``."""

    def __init__(
        self,
        backend: NativeBackend,
        *,
        processor: Any,
        cv2_module: Any,
        action_low: np.ndarray,
        action_high: np.ndarray,
        default_seed: int = 7,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.backend = backend
        self.processor = processor
        self.cv2 = cv2_module
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        if self.action_low.shape != (ACTION_DIM,) or self.action_high.shape != (ACTION_DIM,):
            raise ValueError("DM05 action quantiles must have shape [7]")
        self.default_seed = int(default_seed)
        _seed_u64(self.default_seed)
        backend_metadata = dict(getattr(backend, "metadata", {}))
        required_backend_metadata = {"backend", "model_revision", "device", "precision"}
        missing_backend_metadata = sorted(required_backend_metadata - set(backend_metadata))
        if missing_backend_metadata:
            raise RuntimeError(
                f"native DM05 backend metadata omitted: {missing_backend_metadata}"
            )
        if backend_metadata["backend"] != "apxinf-native":
            raise RuntimeError("DM05 policy accepts only the apxinf-native backend")
        policy_metadata = {
            "schema": "apxinf.dm05.libero.policy.v2",
            "model_type": "dm05",
            "action_horizon": ACTION_HORIZON,
            "action_dim": ACTION_DIM,
            "model_action_dim": MODEL_ACTION_DIM,
            "state_dim": STATE_DIM,
            "state_conditioned": False,
            "image_size": [IMAGE_SIZE, IMAGE_SIZE],
            "image_prompts": list(IMAGE_PROMPTS),
            "robot_type": "Franka",
            "diffusion_steps": DIFFUSION_STEPS,
            "max_prefix_len": MAX_PREFIX_LEN,
            "sampling_rng": "apxinf-philox-box-muller-v1",
            "concurrency": 1,
        }
        conflicts = sorted(
            field
            for field in set(policy_metadata) & set(backend_metadata)
            if backend_metadata[field] != policy_metadata[field]
        )
        if conflicts:
            raise RuntimeError(
                f"native DM05 backend conflicts with policy metadata: {conflicts}"
            )
        if backend_metadata["model_revision"] != MODEL_REVISION:
            raise RuntimeError("native DM05 backend model revision mismatch")
        if backend_metadata["precision"] != "bf16":
            raise RuntimeError("native DM05 backend precision mismatch")
        self.metadata = {
            **policy_metadata,
            **backend_metadata,
            **_validate_metadata(metadata),
        }

    @classmethod
    def from_pretrained(
        cls,
        model_dir,
        *,
        device: str = "cuda:0",
        precision: str = "bf16",
        default_seed: int = 7,
        metadata: Mapping[str, Any] | None = None,
        **unsupported,
    ) -> "Dm05Policy":
        if unsupported:
            raise TypeError(f"unsupported DM05 options: {sorted(unsupported)}")
        _validate_metadata(metadata)
        if precision.lower() != "bf16" or not device.startswith("cuda:"):
            raise ValueError("native DM05 deployment requires BF16 on cuda:N")
        _seed_u64(default_seed)
        model_dir = Path(model_dir).expanduser().resolve()
        action_low, action_high = _validate_checkpoint(model_dir)

        try:
            import cv2
            import transformers
            from transformers import AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "native DM05 policy requires transformers==5.3.0 and OpenCV"
            ) from exc
        if transformers.__version__ != TRANSFORMERS_VERSION:
            raise RuntimeError(
                f"DM05 requires transformers=={TRANSFORMERS_VERSION}, "
                f"got {transformers.__version__}"
            )
        processor = AutoProcessor.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            local_files_only=True,
            use_fast=False,
        )
        backend = ApxInfNativeBackend(
            model_dir, device=device, default_seed=default_seed
        )
        return cls(
            backend,
            processor=processor,
            cv2_module=cv2,
            action_low=action_low,
            action_high=action_high,
            default_seed=default_seed,
            metadata=metadata,
        )

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    @property
    def action_horizon(self) -> int:
        return ACTION_HORIZON

    def _resize(self, image: Image.Image) -> Image.Image:
        array = np.asarray(image.convert("RGB"))
        height, width = array.shape[:2]
        side = max(height, width)
        top = (side - height) // 2
        bottom = side - height - top
        left = (side - width) // 2
        right = side - width - left
        array = self.cv2.copyMakeBorder(
            array, top, bottom, left, right, self.cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        array = self.cv2.resize(
            array, (IMAGE_SIZE, IMAGE_SIZE), interpolation=self.cv2.INTER_LINEAR
        )
        return Image.fromarray(array, mode="RGB")

    def _tokens(self, prompt: str, images: Sequence[Image.Image]) -> np.ndarray:
        content = [
            {
                "type": "text",
                "text": (
                    "Robot: Franka\n"
                    "Overall speed: 0.5\n"
                    f"Task: {prompt}.\n"
                    "Head image: "
                ),
            },
            {"type": "image", "image": images[0]},
            {"type": "text", "text": "Left wrist image: "},
            {"type": "image", "image": images[1]},
        ]
        values = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="np",
        )

        ids = np.asarray(values["input_ids"])
        if ids.ndim == 2 and ids.shape[0] == 1 and ids.shape[1] > MAX_PREFIX_LEN:
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer is None:
                raise RuntimeError("DM05 processor omitted tokenizer for prompt shortening")
            prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            overflow = ids.shape[1] - MAX_PREFIX_LEN
            keep_tokens = max(0, len(prompt_token_ids) - overflow - 16)
            shortened_prompt = tokenizer.decode(
                prompt_token_ids[:keep_tokens], skip_special_tokens=False
            ).strip()
            content[0]["text"] = content[0]["text"].replace(
                prompt, shortened_prompt, 1
            )
            values = self.processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=True,
                return_dict=True,
                return_tensors="np",
            )
            ids = np.asarray(values["input_ids"])
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise RuntimeError(f"DM05 processor returned invalid input_ids shape {ids.shape}")
        if ids.shape[1] > MAX_PREFIX_LEN:
            raise RuntimeError(
                f"DM05 shortened prefix still exceeds {MAX_PREFIX_LEN} tokens: {ids.shape[1]}"
            )
        ids = np.ascontiguousarray(ids[0], dtype=np.uint32)
        if ids.size == 0 or np.count_nonzero(ids == 262_144) != 512:
            raise RuntimeError("DM05 processor did not produce two 256-token image blocks")
        return ids

    def infer(
        self,
        observation: Mapping[str, Any],
        *,
        noise: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if not isinstance(observation, Mapping):
            raise ValueError("DM05 observation must be an object")
        started = time.perf_counter()
        allowed = {"prompt", "state", "images", "robot_type", "sampling"}
        extra = set(observation) - allowed
        if extra:
            raise ValueError(f"unsupported DM05 observation fields: {sorted(extra)}")
        prompt = observation.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("DM05 prompt must be a non-empty string")
        _number_list(observation.get("state"), width=STATE_DIM, label="state")
        if observation.get("robot_type") != "Franka":
            raise ValueError("DM05-libero robot_type must be 'Franka'")
        image_values = observation.get("images")
        if not isinstance(image_values, Mapping) or set(image_values) != {"1", "2"}:
            raise ValueError("DM05-libero images must have exactly slots '1' and '2'")
        images = [
            self._resize(_decode_image(image_values[str(index)], label=f"images.{index}"))
            for index in (1, 2)
        ]
        sampling = observation.get("sampling", {})
        if not isinstance(sampling, Mapping):
            raise ValueError("sampling must be an object")
        extra_sampling = set(sampling) - {"num_steps", "seed"}
        if extra_sampling:
            raise ValueError(f"unsupported sampling fields: {sorted(extra_sampling)}")
        num_steps = sampling.get("num_steps", DIFFUSION_STEPS)
        seed = sampling.get("seed", self.default_seed)
        if isinstance(num_steps, bool) or num_steps != DIFFUSION_STEPS:
            raise ValueError(f"sampling.num_steps must be {DIFFUSION_STEPS}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("sampling.seed must be an integer")
        _seed_u64(seed)

        exact_noise = None
        if noise is not None:
            exact_noise = np.asarray(noise, dtype=np.float32)
            if exact_noise.shape != (ACTION_HORIZON, MODEL_ACTION_DIM):
                raise ValueError(
                    f"DM05 noise must have shape {(ACTION_HORIZON, MODEL_ACTION_DIM)}"
                )
            if not np.isfinite(exact_noise).all():
                raise ValueError("DM05 noise must contain finite values")
            exact_noise = np.ascontiguousarray(exact_noise)

        token_ids = self._tokens(prompt, images)
        rgb_u8 = np.ascontiguousarray(
            np.stack([np.asarray(image, dtype=np.uint8) for image in images])
        )
        normalized, model_ms = self.backend.infer(
            rgb_u8, token_ids, seed=seed, noise=exact_noise
        )
        normalized = np.asarray(normalized, dtype=np.float32)
        if normalized.shape != (ACTION_HORIZON, MODEL_ACTION_DIM):
            raise RuntimeError(
                f"native DM05 returned {normalized.shape}, expected "
                f"{(ACTION_HORIZON, MODEL_ACTION_DIM)}"
            )
        if not np.isfinite(normalized).all() or not math.isfinite(model_ms) or model_ms <= 0:
            raise RuntimeError("native DM05 returned invalid actions or latency")
        active = normalized[:, :ACTION_DIM]
        actions = ((active + 1.0) / 2.0) * (
            self.action_high - self.action_low + 1e-6
        ) + self.action_low
        return {
            "actions": np.ascontiguousarray(actions, dtype=np.float32),
            "normalized_actions": np.ascontiguousarray(normalized),
            "timing": {
                "model_ms": float(model_ms),
                "total_ms": (time.perf_counter() - started) * 1000.0,
            },
            "sampling": {
                "num_steps": DIFFUSION_STEPS,
                "seed": seed,
                "noise_source": (
                    "provided" if exact_noise is not None else "apxinf-philox-box-muller-v1"
                ),
            },
        }

    def close(self) -> None:
        self.backend.close()
        self.processor = None
