"""GR00T N1.7 policy: raw camera/state observations to deployable actions.

The CUDA binding deliberately exposes a small tensor-level ``Gr00tModel``.
This module supplies the user-facing layer around it: NVIDIA's checkpoint
processor builds the exact multimodal tensors, ApxInf executes the model core,
and the same processor decodes normalized actions back to the robot domain.

Heavy optional dependencies (``torch``, ``transformers`` and Isaac-GR00T) are
imported only by :meth:`Gr00tPolicy.from_pretrained`; importing ``apxinf`` stays
CUDA- and GR00T-free.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence

import numpy as np

from ...processors.transforms import has_key, lookup_key
from ..registry import register_policy

__all__ = ["Gr00tPolicy"]

_DEFAULT_IMAGE_KEYS = ("observation/image", "observation/wrist_image")
_STATE_KEY = "observation/state"
_PROMPT_KEY = "prompt"


class _ProcessorAdapter(Protocol):
    """Small injectable seam around NVIDIA's processor, also used by tests."""

    image_keys: tuple[str, ...]
    action_horizon: int
    action_dim: int

    def encode(self, observation: Mapping[str, Any]) -> dict[str, Any]: ...

    def decode(self, normalized: np.ndarray, encoded: Mapping[str, Any]) -> np.ndarray: ...


@register_policy("gr00t")
@register_policy("gr00tn1d7")
class Gr00tPolicy:
    """Compose NVIDIA preprocessing, ApxInf GR00T Model Core and action decode."""

    def __init__(
        self,
        model: Any,
        *,
        processor: _ProcessorAdapter,
        seed: int = 0,
        noise_mode: str = "stream",
        action_horizon: Optional[int] = None,
        action_dim: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if noise_mode not in ("fixed", "stream"):
            raise ValueError("Gr00tPolicy: noise_mode must be 'fixed' or 'stream'")
        self.model = model
        self.processor = processor
        self.image_keys = tuple(processor.image_keys)
        self.noise_mode = noise_mode
        self.action_horizon_out = (
            int(action_horizon) if action_horizon is not None else int(model.action_horizon)
        )
        self.action_dim_out = int(action_dim) if action_dim is not None else int(model.action_dim)
        self._rng = np.random.default_rng(seed)
        self._fixed_noise = self._sample_noise()
        # Constructing the fixed control tensor must not consume the first draw
        # of stream mode; this mirrors the closed-loop validation harness.
        self._rng = np.random.default_rng(seed)
        self.metadata = {
            "model_type": "gr00t",
            "action_horizon": self.action_horizon_out,
            "model_action_horizon": int(model.action_horizon),
            "model_action_dim": int(model.action_dim),
            "action_dim": self.action_dim_out,
            "image_keys": list(self.image_keys),
            "noise_mode": noise_mode,
            "seed": int(seed),
            **(dict(metadata) if metadata else {}),
        }

    @classmethod
    def from_pretrained(
        cls,
        model_dir,
        *,
        backbone=None,
        model: Optional[Any] = None,
        device: str = "cuda:0",
        precision: str = "bf16",
        calibration=None,
        tactics=None,
        embodiment: str = "libero_sim",
        image_keys: Sequence[str] = _DEFAULT_IMAGE_KEYS,
        state_key: str = _STATE_KEY,
        prompt_key: str = _PROMPT_KEY,
        action_key: Optional[str] = None,
        action_horizon: Optional[int] = None,
        action_dim: Optional[int] = None,
        seed: int = 0,
        noise_mode: str = "stream",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "Gr00tPolicy":
        """Load a checkpoint and its official processor.

        ``backbone`` is the local Cosmos-Reason2-2B directory. FP8 additionally
        requires ``calibration``. Raw observations use the same friendly keys as
        Pi0.5 by default: ``observation/image``, ``observation/wrist_image``,
        ``observation/state`` and ``prompt``.
        """
        model_dir = Path(model_dir)
        if backbone is None:
            raise ValueError(
                "Gr00tPolicy: backbone= must point to the local Cosmos-Reason2-2B directory"
            )
        backbone = Path(backbone)
        if precision not in ("bf16", "fp8", "int8"):
            raise ValueError(
                "Gr00tPolicy: precision must be 'bf16', 'fp8', or 'int8'"
            )
        if precision == "fp8" and calibration is None:
            raise ValueError("Gr00tPolicy: precision='fp8' requires calibration=")

        adapter = _NvidiaProcessorAdapter.load(
            model_dir,
            backbone=backbone,
            embodiment=embodiment,
            image_keys=image_keys,
            state_key=state_key,
            prompt_key=prompt_key,
            action_key=action_key,
        )
        resolved_action_dim = action_dim if action_dim is not None else adapter.action_dim
        resolved_action_horizon = (
            action_horizon if action_horizon is not None else adapter.action_horizon
        )
        if model is None:
            import apxinf_py  # lazy optional native dependency

            model = apxinf_py.Gr00tModel.load(
                model_dir,
                backbone,
                device,
                precision,
                Path(calibration) if calibration is not None else None,
                Path(tactics) if tactics is not None else None,
            )
        return cls(
            model,
            processor=adapter,
            seed=seed,
            noise_mode=noise_mode,
            action_horizon=resolved_action_horizon,
            action_dim=resolved_action_dim,
            metadata={"precision": precision, "embodiment": embodiment, **(dict(metadata) if metadata else {})},
        )

    def infer(
        self,
        observation: Mapping[str, Any],
        *,
        noise: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        """Return deployable actions for one raw RGB/state/prompt observation."""
        if not isinstance(observation, Mapping):
            raise TypeError(f"observation must be a mapping, got {type(observation)!r}")
        started = time.perf_counter()
        encoded = self.processor.encode(observation)
        if noise is None:
            noise = self._fixed_noise.copy() if self.noise_mode == "fixed" else self._sample_noise()
        else:
            noise = np.ascontiguousarray(noise, dtype=np.float32)
            expected_unbatched = (
                int(self.model.action_horizon),
                int(self.model.action_dim),
            )
            expected_noise = (
                1,
                *expected_unbatched,
            )
            if noise.shape == expected_unbatched:
                noise = noise[None, ...]
            if noise.shape != expected_noise:
                raise ValueError(
                    f"noise has shape {noise.shape}, expected {expected_unbatched} "
                    f"or {expected_noise}"
                )
            if not np.isfinite(noise).all():
                raise ValueError("noise must contain only finite values")
            # The native model consumes BF16 noise. Return the exact rounded
            # control tensor used by inference rather than the pre-rounding
            # float32 input supplied by the caller.
            noise = _round_to_bf16(noise)

        model_started = time.perf_counter()
        normalized = np.asarray(
            self.model.infer(
                encoded["pixel_values"],
                encoded["image_grid_thw"],
                encoded["token_ids"],
                encoded["attention_mask"],
                encoded["state"],
                int(encoded["embodiment_id"]),
                noise,
            ),
            dtype=np.float32,
        )
        model_ms = (time.perf_counter() - model_started) * 1000.0
        expected = (int(self.model.action_horizon), int(self.model.action_dim))
        if normalized.shape != expected:
            raise ValueError(f"model returned action shape {normalized.shape}, expected {expected}")
        if not np.isfinite(normalized).all():
            raise FloatingPointError("model returned non-finite normalized actions")

        actions = np.asarray(self.processor.decode(normalized, encoded), dtype=np.float32)
        if actions.ndim != 2 or not np.isfinite(actions).all():
            raise ValueError(f"processor returned invalid action array {actions.shape}")
        if actions.shape[0] != self.action_horizon_out:
            raise ValueError(
                f"processor returned action horizon {actions.shape[0]}, "
                f"expected {self.action_horizon_out}; pass action_horizon= for this embodiment"
            )
        if actions.shape[1] != self.action_dim_out:
            raise ValueError(
                f"processor returned action width {actions.shape[1]}, "
                f"expected {self.action_dim_out}; pass action_dim= for this embodiment"
            )
        total_ms = (time.perf_counter() - started) * 1000.0
        return {
            "actions": actions,
            "normalized_actions": normalized,
            "noise": noise,
            "timing": {"model_ms": model_ms, "total_ms": total_ms},
            "metadata": dict(self.metadata),
        }

    __call__ = infer

    @property
    def action_dim(self) -> int:
        return self.action_dim_out

    @property
    def action_horizon(self) -> int:
        return self.action_horizon_out

    def close(self) -> None:
        close = getattr(self.model, "close", None)
        if callable(close):
            close()

    def _sample_noise(self) -> np.ndarray:
        # Match NVIDIA/ApxInf validation: sample f32, round through BF16, then
        # hand contiguous f32 values to the native binding.
        values = self._rng.standard_normal(
            (1, int(self.model.action_horizon), int(self.model.action_dim)), dtype=np.float32
        )
        return _round_to_bf16(values)


class _NvidiaProcessorAdapter:
    """Adapter for the pinned Isaac-GR00T AutoProcessor contract."""

    def __init__(
        self,
        processor: Any,
        *,
        embodiment_tag: Any,
        message_type: Any,
        step_type: Any,
        image_keys: Sequence[str],
        state_key: str,
        prompt_key: str,
        action_key: Optional[str],
    ) -> None:
        self.processor = processor
        self.embodiment_tag = embodiment_tag
        self.message_type = message_type
        self.step_type = step_type
        self.image_keys = tuple(image_keys)
        self.state_key = state_key
        self.prompt_key = prompt_key
        configs = processor.get_modality_configs()[embodiment_tag.value]
        self.modality_configs = {key: value for key, value in configs.items() if key != "rl_info"}
        self.video_keys = list(self.modality_configs["video"].modality_keys)
        self.state_keys = list(self.modality_configs["state"].modality_keys)
        self.action_keys = list(self.modality_configs["action"].modality_keys)
        self.action_horizon = len(self.modality_configs["action"].delta_indices)
        statistics = processor.state_action_processor.statistics[embodiment_tag.value]
        self.state_dims = {
            key: _statistics_dim(statistics["state"][key])
            for key in self.state_keys
        }
        self.action_dims = {
            key: _statistics_dim(statistics["action"][key])
            for key in self.action_keys
        }
        if len(self.image_keys) != len(self.video_keys):
            raise ValueError(
                f"Gr00tPolicy: checkpoint expects {len(self.video_keys)} views "
                f"{self.video_keys}, but image_keys has {len(self.image_keys)} entries"
            )
        if action_key is not None and action_key not in self.action_keys:
            raise ValueError(f"Gr00tPolicy: unknown action_key {action_key!r}; expected {self.action_keys}")
        self.selected_action_keys = [action_key] if action_key is not None else self.action_keys
        self.action_dim = sum(self.action_dims[key] for key in self.selected_action_keys)

    @classmethod
    def load(cls, model_dir: Path, *, backbone: Path, embodiment: str, **kwargs):
        try:
            import gr00t.model  # noqa: F401
            from gr00t.data.embodiment_tags import EmbodimentTag
            from gr00t.data.types import MessageType, VLAStepData
            from transformers import AutoProcessor
        except ImportError as error:
            raise ImportError(
                "Gr00tPolicy requires NVIDIA Isaac-GR00T and transformers. "
                "Install the pinned Isaac-GR00T environment before loading a policy."
            ) from error
        processor_dir = (
            model_dir / "processor"
            if (model_dir / "processor").is_dir()
            and not (model_dir / "processor_config.json").exists()
            else model_dir
        )
        processor = AutoProcessor.from_pretrained(
            processor_dir,
            model_name=str(backbone.resolve()),
            local_files_only=True,
            trust_remote_code=True,
            transformers_loading_kwargs={"local_files_only": True, "trust_remote_code": True},
        )
        processor.eval()
        return cls(
            processor,
            embodiment_tag=EmbodimentTag.resolve(embodiment),
            message_type=MessageType,
            step_type=VLAStepData,
            **kwargs,
        )

    def encode(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        required = [*self.image_keys, self.state_key, self.prompt_key]
        missing = [key for key in required if not has_key(observation, key)]
        if missing:
            raise KeyError(f"Gr00tPolicy.infer: missing observation keys: {missing}")
        prompt = lookup_key(observation, self.prompt_key)
        if not isinstance(prompt, str):
            raise TypeError(f"{self.prompt_key} must be a string")

        images = {
            model_key: _as_video(lookup_key(observation, user_key), user_key)
            for user_key, model_key in zip(self.image_keys, self.video_keys)
        }
        states = self._states(lookup_key(observation, self.state_key))
        step = self.step_type(
            images=images,
            states=states,
            actions={},
            text=prompt,
            embodiment=self.embodiment_tag,
        )
        processed = self.processor(
            [{"type": self.message_type.EPISODE_STEP.value, "content": step}]
        )
        inputs = self.processor.collator([processed])["inputs"]
        encoded = {
            "pixel_values": _numpy(inputs["pixel_values"], np.float32),
            "image_grid_thw": _numpy(inputs["image_grid_thw"], np.uint32),
            "token_ids": _numpy(inputs["input_ids"], np.uint32).reshape(-1),
            "attention_mask": _numpy(inputs["attention_mask"], np.uint8).reshape(-1),
            "state": _numpy(inputs["state"], np.float32),
            "embodiment_id": int(_numpy(inputs["embodiment_id"], np.int64).reshape(-1)[0]),
            "raw_states": states,
        }
        return encoded

    def decode(self, normalized: np.ndarray, encoded: Mapping[str, Any]) -> np.ndarray:
        batched_states = {
            key: np.expand_dims(value, axis=0)
            for key, value in encoded["raw_states"].items()
        }
        decoded = self.processor.decode_action(
            normalized[None], self.embodiment_tag, batched_states
        )
        components = []
        for key in self.selected_action_keys:
            value = np.asarray(decoded[key], dtype=np.float32)
            if value.ndim == 3 and value.shape[0] == 1:
                value = value[0]
            if value.ndim == 1:
                value = value[:, None]
            if value.ndim != 2:
                raise ValueError(f"decoded action field {key!r} has invalid shape {value.shape}")
            components.append(value)
        return np.ascontiguousarray(np.concatenate(components, axis=-1), dtype=np.float32)

    def _states(self, value: Any) -> dict[str, np.ndarray]:
        if isinstance(value, Mapping):
            missing = [key for key in self.state_keys if key not in value]
            if missing:
                raise KeyError(f"Gr00tPolicy.infer: missing state fields: {missing}")
            return {key: _as_state(value[key], key) for key in self.state_keys}
        flat = np.asarray(value, dtype=np.float32)
        expected_dim = sum(self.state_dims.values())
        if flat.ndim == 1 and flat.size == expected_dim:
            states = {}
            offset = 0
            for key in self.state_keys:
                width = self.state_dims[key]
                states[key] = np.ascontiguousarray(flat[None, offset : offset + width])
                offset += width
            return states
        if len(self.state_keys) != 1:
            raise ValueError(
                "Gr00tPolicy: this checkpoint has multiple state fields; pass a flat "
                f"vector of length {expected_dim} or a mapping with keys "
                f"{self.state_keys}"
            )
        return {self.state_keys[0]: _as_state(value, self.state_keys[0])}


def _numpy(value: Any, dtype: np.dtype) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.ascontiguousarray(value, dtype=dtype)


def _as_video(value: Any, key: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim not in (3, 4) or array.shape[-1] != 3:
        raise ValueError(f"{key} must be uint8 HWC or THWC RGB, got {array.shape}/{array.dtype}")
    if array.ndim == 3:
        array = array[None]
    return np.ascontiguousarray(array)


def _as_state(value: Any, key: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    if array.ndim == 1:
        array = array[None]
    if array.ndim != 2:
        raise ValueError(f"state field {key!r} must be D or T×D, got {array.shape}")
    return np.ascontiguousarray(array)


def _statistics_dim(statistics: Mapping[str, Any]) -> int:
    """Read a modality width before or after NVIDIA statistics conversion."""
    if "dim" in statistics:
        return int(np.asarray(statistics["dim"]).item())
    for name in ("mean", "std", "min", "max", "q01", "q99"):
        if name in statistics:
            return int(np.asarray(statistics[name]).size)
    raise ValueError("GR00T checkpoint statistics do not describe the modality width")


def _round_to_bf16(value: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(value, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)
