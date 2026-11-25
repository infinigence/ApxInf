"""Native MolmoAct2 policy (LIBERO, plain continuous-action checkpoint).

The native runner executes ``uint8`` images + prompt token ids -> normalized, padded actions
``[10, 32]``, by default through one captured whole-model CUDA graph (``execution="eager"`` selects prepared
eager: the same traversal enqueued from the host on prepared, reused native executions, without a graph;
``"require-graph"`` fails instead of falling back; see ``doc/molmoact2.md``). Everything that
is *not* a network is done here, following the official
``predict_action`` path of the checkpoint:

* the two cameras (``observation.images.image`` then ``observation.images.wrist_image``) are resized on
  the host to 378x378 with the checkpoint's own CPU resize (torchvision ``Resize``, bilinear,
  ``antialias=False``, on ``uint8``). torch/torchvision are therefore a host-preprocessing dependency
  (``pip install 'apxinf[molmoact2]'``); without them the policy raises instead of approximating;
* the 8-D robot state is normalized (q01/q99, first seven dimensions; the gripper entry is kept),
  clipped, discretized to 256 bins and written into the prompt as ``<state_N>`` tokens;
* the prompt is built from the task text, the setup / control strings of the ``libero`` statistics tag
  and the two image placeholders, expanded to image tokens, tokenized, and the BOS token is inserted;
* the normalized padded actions are cropped to seven dimensions, clipped to ``[-1, 1]`` and mapped back
  to physical units (the first six dimensions through q01/q99; the gripper dimension stays in
  ``[-1, 1]``) in float32, then rounded to BF16 (round-to-nearest-even) and returned as float32: the
  official BF16 model returns its physical actions in the trajectory dtype, and this runner is that
  BF16 profile.

The policy accepts only checkpoints whose ``config.json`` has the continuous action expert and the
discrete-state prompt this path implements; another checkpoint with the same ``model_type`` is rejected
before anything is loaded.

Provenance: the prompt, state-discretization and normalization helpers below follow the functions of the
same names in ``modeling_molmoact2.py`` / ``processing_molmoact2.py`` / ``image_processing_molmoact2.py`` shipped
with ``allenai/MolmoAct2-LIBERO`` revision ``0d24a92bd1faf321ef497c3bbd5681af97c65aa2`` (Apache-2.0 per the
publisher's ``allenai/molmoact2`` repository); see ``NOTICE``.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ...processors.transforms import has_key, lookup_key
from ..registry import register_policy

__all__ = ["MolmoAct2Policy"]

NORM_TAG = "libero"
ACTION_DIM = 7
STATE_DIM = 8
IMAGE_SIZE = 378
NUM_STATE_BINS = 256

# --- prompt vocabulary of the checkpoint (``modeling_molmoact2.py`` / ``processing_molmoact2.py``)
_IMAGE_PROMPT = "<|image|>"
_IM_START, _IM_END, _IM_PATCH = "<im_start>", "<im_end>", "<im_patch>"
_ACTION_OUTPUT = "<action_output>"
_STATE_START, _STATE_END, _STATE_PREFIX = "<state_start>", "<state_end>", "<state_"
_SETUP_START, _SETUP_END = "<setup_start>", "<setup_end>"
_CONTROL_START, _CONTROL_END = "<control_start>", "<control_end>"

_TRAILING_PUNCTUATION = ".,!?;:,…"
_TRAILING_CLOSERS = "\"'”’)]}"
_SURROUNDING_DELIMITERS = "\"'`“”‘’[](){}"
_PREFIX_PATTERNS = tuple(
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"^(?:task|instruction|language[_ ]instruction|goal)\s*[:\-]\s*",
        r"^(?:the\s+task\s+is\s+to|your\s+task\s+is\s+to)\s+",
    )
)


def normalize_task_text(text: str) -> str:
    """The official ``_normalize_question_text``."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return ""
    previous = None
    while normalized and normalized != previous:
        previous = normalized
        normalized = normalized.strip().strip(_SURROUNDING_DELIMITERS).strip()
        for pattern in _PREFIX_PATTERNS:
            normalized = pattern.sub("", normalized, count=1).strip()
        normalized = normalized.rstrip(_TRAILING_PUNCTUATION).rstrip()
        normalized = normalized.rstrip(_TRAILING_CLOSERS).rstrip()
        normalized = normalized.rstrip(_TRAILING_PUNCTUATION).rstrip()
    chunks = [chunk.strip() for chunk in re.split(r"[.!?]+", normalized) if chunk.strip()]
    if len(chunks) > 1:
        normalized = "; ".join(chunks)
    return normalized.lower()


def discretize_state(normalized_state: np.ndarray, num_bins: int = NUM_STATE_BINS) -> np.ndarray:
    """The official ``_discretize_normalized_state`` (float32 arithmetic, round-half-even)."""
    arr = np.asarray(normalized_state, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    arr = np.clip(arr, -1.0, 1.0)
    scaled = (arr + 1.0) / 2.0 * float(num_bins - 1)
    return np.clip(np.rint(scaled).astype(np.int64), 0, int(num_bins) - 1)


def round_to_bf16(value: np.ndarray) -> np.ndarray:
    """float32 -> nearest-even BF16 -> float32 (finite inputs; what ``tensor.to(torch.bfloat16)`` does)."""
    array = np.ascontiguousarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise FloatingPointError("round_to_bf16 takes finite values")
    bits = array.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def build_prompt(
    *,
    task: str,
    state_bins: np.ndarray,
    setup_type: str,
    control_mode: str,
    add_setup_tokens: bool,
    add_control_tokens: bool,
    num_images: int,
) -> str:
    """The official ``_build_robot_text`` for the ``robot_action`` style (before image expansion)."""
    setup = f"{_SETUP_START}{setup_type}{_SETUP_END}" if setup_type and add_setup_tokens else setup_type
    control = (
        f"{_CONTROL_START}{control_mode}{_CONTROL_END}" if control_mode and add_control_tokens else control_mode
    )
    state = "".join(f"{_STATE_PREFIX}{int(b)}>" for b in np.asarray(state_bins).reshape(-1))
    state_clause = f" The current state of the robot is {_STATE_START}{state}{_STATE_END}." if state else ""
    prompt = (
        f"The task is to {task}. The setup is {setup}.{state_clause} "
        f"The expected control mode is {control}. Given these, what action should the robot take "
        "to complete the task?"
    )
    prefix = "".join(f"Image {i + 1}{_IMAGE_PROMPT}" for i in range(num_images))
    return f"{prefix}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{_ACTION_OUTPUT}"


def expand_image_tokens(text: str, tokens_per_side: int) -> str:
    """Replace each image placeholder by its single-crop token string (``crop_mode='resize'``).

    ``<im_start>`` + ``tokens_per_side`` rows of ``tokens_per_side`` patch tokens (no column tokens for
    the single crop) + ``<im_end>``.
    """
    image = _IM_START + _IM_PATCH * (tokens_per_side * tokens_per_side) + _IM_END
    return text.replace(_IMAGE_PROMPT, image)


def insert_bos(ids: Sequence[int], bos_id: int) -> List[int]:
    """The processor's ``insert_bos`` for one sequence without padding."""
    ids = list(ids)
    if not ids:
        return [bos_id]
    return ids if ids[0] == bos_id else [bos_id] + ids


# ----------------------------------------------------------------------------- image resize


def _resize_backend():
    """``(torch, torchvision)`` for the host resize; a missing install is an explicit error.

    The checkpoint's reference preprocessing (``resize_image`` in ``image_processing_molmoact2.py``) is
    torchvision's CPU ``uint8`` bilinear resize, whose rounding is produced by torch's kernel. There is
    no approximation to fall back to: a NumPy re-implementation was tried and differed from it by single
    codes on real frames, and a one-code input difference changes the actions. torch is used only for
    this host image resize; none of its model code runs.
    """
    try:
        import torch
        import torchvision
    except ImportError as error:
        raise ImportError(
            "MolmoAct2Policy resizes camera frames with torchvision (the checkpoint's reference preprocessing); "
            "install torch and torchvision (pip install 'apxinf[molmoact2]'), or pass image_transform= with "
            "an equivalent exact resize. No approximate fallback is used."
        ) from error
    return torch, torchvision


def resize_bilinear_u8(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """``uint8`` HWC -> ``uint8`` ``size x size`` HWC through the official CPU resize.

    The same calls, in the same tensor layout (a channel-first *view* of the HWC array), as the
    checkpoint's ``resize_image`` for integer images: ``torchvision.transforms.Resize([size, size],
    BILINEAR, antialias=False)`` followed by a clip to ``[0, 255]``.
    """
    torch, torchvision = _resize_backend()
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"resize_bilinear_u8 takes uint8 HWC with 3 channels, got {image.dtype} {image.shape}")
    hwc = np.require(image, requirements=["C", "W"])  # contiguous like the reference's decoded frame; torch needs writable memory
    tensor = torch.permute(torch.from_numpy(hwc), [2, 0, 1])
    resized = torchvision.transforms.Resize(
        [int(size), int(size)], torchvision.transforms.InterpolationMode.BILINEAR, antialias=False
    )(tensor)
    resized = torch.clip(resized, 0, 255).to(torch.uint8)
    return np.ascontiguousarray(torch.permute(resized, [1, 2, 0]).numpy())


# ----------------------------------------------------------------------------- statistics


class _Q01Q99:
    """The official ``_FeatureNormalizer`` for ``norm_mode='q01_q99'`` (float32 NumPy arithmetic)."""

    def __init__(self, stats: Mapping[str, Any], width: int, what: str):
        arrays = {}
        for key in ("q01", "q99"):
            value = np.asarray(stats[key], dtype=np.float32)
            if value.shape != (width,):
                raise ValueError(f"MolmoAct2 {what} statistics {key!r} have shape {value.shape}, expected ({width},)")
            arrays[key] = value
        self.q_low, self.q_high = arrays["q01"], arrays["q99"]
        mask = stats.get("mask")
        self.mask = None if mask is None else np.asarray(mask, dtype=np.bool_)
        if self.mask is not None and self.mask.shape != (width,):
            raise ValueError(f"MolmoAct2 {what} statistics mask has shape {self.mask.shape}")
        lo, hi = stats.get("min"), stats.get("max")
        self.zero_mask = None if lo is None or hi is None else (np.asarray(lo, np.float32) == np.asarray(hi, np.float32))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        normed = 2.0 * (arr - self.q_low) / np.maximum(self.q_high - self.q_low, 1e-6) - 1.0
        normed = np.clip(normed, -1.0, 1.0)
        if self.mask is not None:
            normed = np.where(self.mask, normed, arr)
        if self.zero_mask is not None:
            normed = np.where(self.zero_mask, 0.0, normed)
        return np.asarray(normed, dtype=np.float32)

    def unnormalize(self, x: np.ndarray) -> np.ndarray:
        arr = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
        out = (arr + 1.0) * (self.q_high - self.q_low) / 2.0 + self.q_low
        if self.mask is not None:
            out = np.where(self.mask, out, arr)
        return np.asarray(out, dtype=np.float32)


def check_checkpoint_config(config: Mapping[str, Any]) -> None:
    """Refuse every checkpoint this policy does not implement (same rules as the native loader)."""
    problems = []
    if config.get("model_type") != "molmoact2":
        problems.append(f"model_type is {config.get('model_type')!r}, not 'molmoact2'")
    if config.get("add_action_expert") is not True:
        problems.append("add_action_expert is not true (no continuous action expert)")
    if config.get("action_mode") != "both":
        problems.append(f"action_mode is {config.get('action_mode')!r}, not 'both'")
    if config.get("state_format") != "discrete":
        problems.append(f"state_format is {config.get('state_format')!r}, not 'discrete'")
    if config.get("enable_depth_reasoning") is True or config.get("action_expert_depth_gate") is True:
        problems.append("enable_depth_reasoning / action_expert_depth_gate is set (depth reasoning is not supported)")
    if config.get("mask_action_dim_padding") is not True:
        problems.append("mask_action_dim_padding is not true")
    if config.get("n_obs_steps") != 1:
        problems.append(f"n_obs_steps is {config.get('n_obs_steps')!r}, not 1")
    if int(config.get("num_state_tokens") or 0) != NUM_STATE_BINS:
        problems.append(f"num_state_tokens is {config.get('num_state_tokens')!r}, not {NUM_STATE_BINS}")
    if problems:
        raise ValueError("MolmoAct2Policy supports the plain continuous-action LIBERO checkpoint only: " + "; ".join(problems))


class _Tokenizer:
    """``tokenizer.json`` through ApxInf's native Rust tokenizer."""

    def __init__(self, model_dir: Path):
        try:
            import apxinf_py
        except ImportError as error:
            raise ImportError("MolmoAct2Policy requires the native apxinf-py package") from error
        path = model_dir / "tokenizer.json"
        if not path.is_file():
            raise FileNotFoundError(f"MolmoAct2 tokenizer not found: {path}")
        self._tokenizer = apxinf_py.HfTokenizer.from_file(str(path))
        config_path = model_dir / "tokenizer_config.json"
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        resolved = {}
        for role in ("bos_token", "eos_token", "pad_token"):
            token = config.get(role)
            resolved[role] = None if token is None else self._tokenizer.token_to_id(token)
        # processor: bos = tokenizer.bos_token_id or tokenizer.eos_token_id
        self.bos_id = resolved["bos_token"] if resolved["bos_token"] is not None else resolved["eos_token"]
        if self.bos_id is None:
            raise ValueError("MolmoAct2 tokenizer config names no BOS/EOS token")

    def encode(self, text: str) -> List[int]:
        return list(self._tokenizer.encode_with_special_tokens(text, True))


@register_policy("molmoact2")
class MolmoAct2Policy:
    """Observation-to-action policy for the native BF16 MolmoAct2-LIBERO runner."""

    def __init__(
        self,
        model_runner,
        *,
        tokenizer,
        state_normalizer: _Q01Q99,
        action_normalizer: _Q01Q99,
        setup_type: str,
        control_mode: str,
        add_setup_tokens: bool,
        add_control_tokens: bool,
        image_keys: Sequence[str],
        state_key: str = "observation.state",
        prompt_key: str = "task",
        image_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ):
        self.model_runner = model_runner
        variant = getattr(model_runner, "model_variant", "bf16")
        if variant != "bf16":
            # postprocess() closes the output boundary of the BF16 profile; another precision needs its own rule
            raise ValueError(f"MolmoAct2Policy implements the bf16 profile only, the runner is {variant!r}")
        self.tokenizer = tokenizer
        self.state_normalizer = state_normalizer
        self.action_normalizer = action_normalizer
        self.setup_type = setup_type
        self.control_mode = control_mode
        self.add_setup_tokens = bool(add_setup_tokens)
        self.add_control_tokens = bool(add_control_tokens)
        self.image_keys = tuple(image_keys)
        self.state_key = state_key
        self.prompt_key = prompt_key
        self.image_transform = image_transform

        if len(self.image_keys) != int(model_runner.num_views):
            raise ValueError(
                f"MolmoAct2Policy requires {model_runner.num_views} image keys, got {len(self.image_keys)}"
            )
        self.image_size = int(model_runner.image_size)
        patch = int(model_runner.patch_size) if hasattr(model_runner, "patch_size") else 14
        # the connector pools 2x2 patches; the single crop covers ceil(side / 2) tokens per side
        self.tokens_per_side = -(-(self.image_size // patch) // 2)
        latent = tuple(getattr(model_runner, "initial_latent_shape", (1, 10, 32)))
        self.latent_shape = latent
        self._horizon = int(latent[-2])

        self.metadata = {
            "model_type": "molmoact2",
            "model_variant": "bf16",
            "action_output": "float32 widened from BF16 (official BF16 output dtype)",
            "norm_tag": NORM_TAG,
            "image_keys": list(self.image_keys),
            "state_key": self.state_key,
            "prompt_key": self.prompt_key,
            "action_horizon": self._horizon,
            "action_dim": ACTION_DIM,
            "state_dim": STATE_DIM,
            "internal_action_dim": int(model_runner.action_dim),
            "num_views": int(model_runner.num_views),
            "image_size": self.image_size,
            "initial_latent_shape": list(latent),
            **(dict(metadata) if metadata else {}),
        }

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_pretrained(
        cls,
        model_dir,
        *,
        device: str = "cuda:0",
        seed: int = 0,
        model_runner=None,
        image_keys: Optional[Sequence[str]] = None,
        state_key: Optional[str] = None,
        prompt_key: str = "task",
        image_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        tokenizer=None,
        tactics: Optional[Path | str] = None,
        autotune: bool = False,
        execution: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "MolmoAct2Policy":
        model_dir = Path(model_dir)
        if image_transform is None:
            _resize_backend()  # fail before any weight is loaded when the host resize is unavailable
        config = json.loads((model_dir / "config.json").read_text())
        check_checkpoint_config(config)
        stats = json.loads((model_dir / "norm_stats.json").read_text())
        if stats.get("norm_mode") != "q01_q99":
            raise ValueError(f"MolmoAct2 norm_mode {stats.get('norm_mode')!r} is not supported (q01_q99 only)")
        tag = (stats.get("metadata_by_tag") or {}).get(NORM_TAG)
        if tag is None:
            raise ValueError(f"MolmoAct2 norm_stats.json has no {NORM_TAG!r} tag")
        state_normalizer = _Q01Q99(tag["state_stats"], STATE_DIM, "state")
        action_normalizer = _Q01Q99(tag["action_stats"], ACTION_DIM, "action")
        if int(tag.get("action_horizon", 0)) != int(tag.get("n_action_steps", 0)):
            raise ValueError("MolmoAct2 tag executes a different number of steps than it predicts")

        if model_runner is None:
            try:
                import apxinf_py
            except ImportError as error:
                raise ImportError("MolmoAct2Policy requires the native apxinf-py package") from error
            model_runner = apxinf_py.ModelRunner.load(
                "molmoact2",
                model_dir,
                device=device,
                sampling_seed=int(seed),
                tactics=str(tactics) if tactics is not None else None,
                autotune=bool(autotune),
                execution=execution,
            )
        if tokenizer is None:
            tokenizer = _Tokenizer(model_dir)
        keys = tuple(image_keys) if image_keys is not None else tuple(tag.get("camera_keys") or ())
        return cls(
            model_runner,
            tokenizer=tokenizer,
            state_normalizer=state_normalizer,
            action_normalizer=action_normalizer,
            setup_type=str(tag.get("setup_type") or ""),
            control_mode=str(tag.get("control_mode") or ""),
            add_setup_tokens=bool(config.get("add_setup_tokens")),
            add_control_tokens=bool(config.get("add_control_tokens")),
            image_keys=keys,
            state_key=state_key if state_key is not None else str(tag.get("state_key") or "observation.state"),
            prompt_key=prompt_key,
            image_transform=image_transform,
            metadata=metadata,
        )

    # ------------------------------------------------------------------ inference

    def infer(self, observation: Mapping[str, Any], *, noise: Optional[np.ndarray] = None) -> dict:
        started = time.perf_counter()
        required = list(self.image_keys) + [self.state_key]
        if not has_key(observation, self.prompt_key):
            required.append(self.prompt_key)
        missing = [key for key in required if not has_key(observation, key)]
        if missing:
            raise KeyError(
                f"MolmoAct2Policy.infer: missing observation keys: {missing}; "
                f"observation has {sorted(observation)}"
            )

        raw_state = self._state(lookup_key(observation, self.state_key))
        token_ids = self._token_ids(lookup_key(observation, self.prompt_key), raw_state)
        rgb = self._rgb(observation)
        selected_noise = self._noise(noise)
        prepared = time.perf_counter()

        normalized = np.asarray(
            self.model_runner.infer_rgb(rgb, "nhwc", token_ids, selected_noise), dtype=np.float32
        )
        model_done = time.perf_counter()
        width = int(self.model_runner.action_dim)
        if normalized.shape != (self._horizon, width):
            raise ValueError(f"MolmoAct2 runner returned {normalized.shape}, expected {(self._horizon, width)}")
        if not np.isfinite(normalized).all():
            raise FloatingPointError("MolmoAct2 runner returned non-finite normalized actions")
        actions = self.postprocess(normalized)
        finished = time.perf_counter()
        return {
            "actions": actions,
            "normalized_actions": normalized,
            "token_ids": token_ids,
            "noise": selected_noise,
            "execution_mode": self.execution_mode,
            "timing": {
                "model_ms": (model_done - prepared) * 1000.0,
                "total_ms": (finished - started) * 1000.0,
                "preprocess_ms": (prepared - started) * 1000.0,
                "postprocess_ms": (finished - model_done) * 1000.0,
            },
            "metadata": self.metadata,
        }

    __call__ = infer

    def postprocess(self, normalized: np.ndarray) -> np.ndarray:
        """Normalized padded ``[horizon, width]`` -> physical ``[horizon, 7]`` float32 actions.

        Crop, clip and unnormalize in float32 (the official ``_FeatureNormalizer.unnormalize`` computes in
        NumPy float32), then round to BF16 and widen again: the official BF16 model returns
        ``torch.as_tensor(out, dtype=x.dtype)`` of its BF16 trajectory.
        """
        cropped = np.ascontiguousarray(np.asarray(normalized, dtype=np.float32)[:, :ACTION_DIM])
        physical = self.action_normalizer.unnormalize(cropped)
        return np.ascontiguousarray(round_to_bf16(physical), dtype=np.float32)

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    @property
    def execution_mode(self) -> str:
        """What the native runner executes right now: ``"unprepared"`` (graph runner before its first call),
        ``"graph"`` (the captured whole-model CUDA graph) or ``"eager"`` (prepared eager: ``execution="eager"``, or the
        fallback of ``"prefer-graph"``; never the unprepared raw path)."""
        mode = getattr(self.model_runner, "execution_mode", None)
        return str(mode() if callable(mode) else mode) if mode is not None else "unknown"

    @property
    def action_horizon(self) -> int:
        return self._horizon

    def close(self) -> None:
        close = getattr(self.model_runner, "close", None)
        if callable(close):
            close()

    # ------------------------------------------------------------------ input preparation

    def _state(self, raw_state) -> np.ndarray:
        state = np.asarray(raw_state, dtype=np.float32)
        if state.shape != (STATE_DIM,):
            raise ValueError(f"MolmoAct2 state must have shape ({STATE_DIM},), got {state.shape}")
        if not np.isfinite(state).all():
            raise FloatingPointError("MolmoAct2 state contains non-finite values")
        return state

    def prompt_text(self, prompt, raw_state: np.ndarray) -> str:
        if not isinstance(prompt, str):
            raise TypeError(f"MolmoAct2 prompt must be a string, got {type(prompt)!r}")
        bins = discretize_state(self.state_normalizer.normalize(raw_state))
        text = build_prompt(
            task=normalize_task_text(prompt),
            state_bins=bins,
            setup_type=self.setup_type,
            control_mode=self.control_mode,
            add_setup_tokens=self.add_setup_tokens,
            add_control_tokens=self.add_control_tokens,
            num_images=len(self.image_keys),
        )
        return expand_image_tokens(text, self.tokens_per_side)

    def _token_ids(self, prompt, raw_state: np.ndarray) -> np.ndarray:
        ids = insert_bos(self.tokenizer.encode(self.prompt_text(prompt, raw_state)), int(self.tokenizer.bos_id))
        return np.asarray(ids, dtype=np.uint32)

    def _rgb(self, observation: Mapping[str, Any]) -> np.ndarray:
        frames = []
        for key in self.image_keys:
            image = np.asarray(lookup_key(observation, key))
            if image.dtype != np.uint8:
                raise TypeError(f"MolmoAct2 image {key!r} must be uint8, got {image.dtype}; convert it explicitly")
            if image.ndim != 3 or 3 not in (image.shape[0], image.shape[-1]):
                raise ValueError(f"MolmoAct2 image {key!r} must be HWC or CHW with 3 channels, got {image.shape}")
            if image.shape[-1] != 3:
                image = np.ascontiguousarray(image.transpose(1, 2, 0))
            if self.image_transform is None:
                frame = resize_bilinear_u8(image, self.image_size)
            else:
                # a caller-supplied resize replaces the default (the default is already the official one)
                frame = np.asarray(self.image_transform(image))
                if frame.shape != (self.image_size, self.image_size, 3) or frame.dtype != np.uint8:
                    raise ValueError(
                        f"MolmoAct2 image_transform must return uint8 {(self.image_size, self.image_size, 3)}, "
                        f"got {frame.dtype} {frame.shape}"
                    )
            frames.append(frame)
        return np.ascontiguousarray(np.stack(frames))

    def _noise(self, noise: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if noise is None:
            return None
        value = np.ascontiguousarray(noise, dtype=np.float32)
        if value.shape != tuple(self.latent_shape):
            raise ValueError(f"MolmoAct2 noise shape {value.shape} != {tuple(self.latent_shape)}")
        if not np.isfinite(value).all():
            raise FloatingPointError("MolmoAct2 noise contains non-finite values")
        return value
