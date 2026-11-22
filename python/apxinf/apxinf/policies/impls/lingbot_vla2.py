"""LingBot-VLA 2.0 L2 policy: robot-config state/action mapping, Qwen3-VL patches, native model call.

Registers ``LingbotVla2Policy`` under ``model_type="lingbot_vla2"``. The published
checkpoint's ``config.json`` is just ``{"vlm_family": "qwen3_vl"}``, so
:class:`~apxinf.policies.auto.AutoPolicy` needs an explicit ``model_type="lingbot_vla2"``.

The policy reproduces the reference deployment path
(``deploy/lingbot_vla_v2_policy.py`` + ``lingbotvla/data/vla_data/{utils,transform}.py`` of
``github.com/robbyant/lingbot-vla-v2``, Apache-2.0) in plain numpy:

* **state / action layout.** A robot config (YAML, as in the upstream ``configs/robot_configs``)
  maps raw observation keys onto named joint groups (``arm.position`` ...). Each group is
  normalised with the post-training norm statistics, zero-padded to its slot width, and the slots
  are concatenated in the training ``joints`` order and padded to ``max_state_dim`` (55).
  Model output goes the reverse way: select the real columns, split per group, un-normalise,
  add the state for ``subtract_state`` groups, and write back to the raw action keys.
* **images.** ``torchvision.transforms.v2.Resize((256, 256))`` semantics (bilinear, antialias) on a
  *float32* CHW image, then ``Qwen2VLImageProcessorFast`` arithmetic for a 256x256 input:
  ``(x - 127.5) / 127.5``, temporal duplication, 16x16 patches, merge-2 block ordering,
  1536 features per row. Three views -> ``pixel_values [3*256, 1536]``, ``image_grid_thw``
  ``[[1,16,16]] * 3``.
* **prompt.** The Qwen3-VL chat template for one user turn, tokenised without padding. Only the
  valid tokens are sent (the reference pads to 72 and masks; the native model takes the
  unpadded sequence).
* **noise.** ``_infer_preprocessed`` has no device-RNG path, so the policy draws the flow-matching
  noise on the host and rounds it through BF16, which is the precision the model consumes.

Everything here is host-side preprocessing/postprocessing. No torch, transformers or ``apxinf_py``
import happens at module import; ``apxinf_py`` is loaded lazily by ``from_pretrained``.

Divergences from the reference (all documented, none silent):

* Missing cameras are an error. The reference zero-pads a missing view and masks it; the native
  runner only has a fixed three-view profile.
* ``sincos`` state normalisation and ``convert_from_state`` actions are rejected.
* A joint group whose norm statistics are absent raises. The reference silently skips normalisation
  for a key that has no statistics entry.
* The reference deploy default (``use_bf16=True``) rounds the state to BF16 before adding it back to
  ``subtract_state`` actions. This policy adds the float32 state, which is what
  ``FeatureTransform.apply/unapply`` itself computes.
* Resize arithmetic is float64 internally and rounded to float32 once; torch accumulates in float32,
  so frames agree to ~1e-4 (in 0..255 units), not bit-for-bit.

Quirk reproduced on purpose: the reference casts every frame to float32 *before* resizing, and its
``_to_uint8`` then reads any frame whose maximum is <= 2.0 as [0, 1]-scaled and multiplies by 255.
A real uint8 frame that is almost black (max <= 2) is therefore amplified, exactly as upstream does.
"""

from __future__ import annotations

import ast
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ..base import CANONICAL_PROMPT_KEY
from ..registry import register_policy
from ._lingbot_vla2_robots import BUNDLED_ROBOTS

__all__ = ["LingbotVla2Policy", "load_robot_config", "default_joints"]

MODEL_TYPE = "lingbot_vla2"

#: Joint groups of the unified 55-dim vector, in the order the released training configs use
#: (``configs/vla/real_robot/real_robot.yaml``). The remaining 4 dims are reserved padding.
_DEFAULT_JOINTS: Tuple[Tuple[str, int], ...] = (
    ("arm.position", 14),
    ("end.position", 14),
    ("effector.position", 2),
    ("waist.position", 4),
    ("head.position", 2),
    ("base.position", 3),
    ("hand.position", 12),
)
_DEFAULT_CAMERAS = ("camera_top", "camera_wrist_left", "camera_wrist_right")

_PATCH = 16
_MERGE = 2
_TEMPORAL = 2
_MEAN_STD = 127.5  # mean 0.5 / std 0.5 fused with the 1/255 rescale, as the fast HF processor does

_NORM_TYPES = (
    "meanstd", "std", "bounds_98", "bounds_99", "bounds_98_woclip", "bounds_99_woclip",
    "minmax", "minmax_woclip", "identity",
)

_CHAT_PREFIX = "<|im_start|>user\n"
_CHAT_SUFFIX = "<|im_end|>\n"


def default_joints() -> List[Tuple[str, int]]:
    """The default ``(joint group, slot width)`` layout of the 55-dim state/action vector."""
    return list(_DEFAULT_JOINTS)


# --------------------------------------------------------------------------------------------
# Robot config
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Piece:
    """One slice ``origin[start:end]`` of a raw key, placed at ``target[t_start:t_end]``."""

    origin: str
    start: Optional[int]
    end: Optional[int]
    t_start: int
    t_end: int


@dataclass
class _Feature:
    target: str
    pieces: List[_Piece]  # a plain (unsliced) key is one piece with start=end=None
    subtract_state: bool = False
    relative_type: str = "quaternion_local"

    @property
    def joint(self) -> str:
        for prefix in ("observation.state.", "action."):
            if self.target.startswith(prefix):
                return self.target[len(prefix):]
        return self.target

    @property
    def width(self) -> Optional[int]:
        """Width implied by the slices, or ``None`` when a whole raw key is used."""
        if any(p.start is None for p in self.pieces):
            return None
        return sum(p.end - p.start for p in self.pieces)


@dataclass
class _Robot:
    states: Dict[str, _Feature]
    actions: Dict[str, _Feature]
    images: Dict[str, str]  # target "observation.images.<cam>" -> raw observation key
    raw_actions: List[str]  # raw action keys in first-appearance order


def _parse_features(entries: Sequence[Any], category: str) -> Dict[str, _Feature]:
    out: Dict[str, _Feature] = {}
    for entry in entries or ():
        if isinstance(entry, str):
            # A bare name maps the same raw key onto itself.
            out[entry] = _Feature(entry, [_Piece(entry, None, None, 0, 0)])
            continue
        if not isinstance(entry, Mapping) or len(entry) != 1:
            raise ValueError(f"robot config {category}: each entry must be a name or a one-key mapping, got {entry!r}")
        target, info = next(iter(entry.items()))
        info = dict(info or {})
        if info.pop("convert_from_state", False):
            raise NotImplementedError(
                f"robot config {category} {target}: convert_from_state actions are not supported"
            )
        subtract = bool(info.pop("subtract_state", False))
        relative = info.pop("relative_type", "quaternion_local")
        origin = info.pop("origin_keys", None)
        if origin is None:
            pieces = [_Piece(target, None, None, 0, 0)]
        elif isinstance(origin, str):
            pieces = [_Piece(origin, None, None, 0, 0)]
        else:
            pieces, cursor = [], 0
            for item in origin:
                if not isinstance(item, Mapping) or len(item) != 1:
                    raise ValueError(f"robot config {category} {target}: bad origin_keys entry {item!r}")
                key, span = next(iter(item.items()))
                start, end = int(span["start"]), int(span["end"])
                if end <= start:
                    raise ValueError(f"robot config {category} {target}: empty slice {key}[{start}:{end}]")
                pieces.append(_Piece(str(key), start, end, cursor, cursor + end - start))
                cursor += end - start
        if category == "actions" and subtract and "effector.position" in target:
            raise ValueError(f"robot config actions {target}: effector.position cannot be subtracted from state")
        if category == "actions" and subtract and _is_quaternion_relative(str(relative)) and "end.position" not in target:
            raise ValueError(
                f"robot config actions {target}: subtract_state with relative_type {relative!r} "
                "(the default) is a quaternion pose delta, which upstream only defines for end.position. "
                "For plain joint deltas set an explicit relative_type, e.g. 'joint'."
            )
        out[target] = _Feature(target, pieces, subtract, str(relative))
    return out


def load_robot_config(robot: Union[str, Path, Mapping[str, Any]]) -> _Robot:
    """Parse a bundled robot name, a YAML path, or an already-loaded mapping."""
    if isinstance(robot, Mapping):
        document = dict(robot)
    else:
        text = str(robot)
        path = Path(text)
        if text in BUNDLED_ROBOTS:
            document = json.loads(json.dumps(BUNDLED_ROBOTS[text]))
        elif path.is_file():
            import yaml

            document = yaml.safe_load(path.read_text())
        else:
            raise FileNotFoundError(
                f"robot {text!r} is neither a bundled config {sorted(BUNDLED_ROBOTS)} nor a YAML file"
            )
    states = _parse_features(document.get("states"), "states")
    actions = _parse_features(document.get("actions"), "actions")
    images: Dict[str, str] = {}
    for entry in document.get("images") or ():
        if isinstance(entry, str):
            images[entry] = entry
        else:
            target, info = next(iter(entry.items()))
            raw = (info or {}).get("origin_keys", target)
            if not isinstance(raw, str):
                raise ValueError(f"robot config images {target}: origin_keys must be one raw key")
            images[target] = raw
    for target in states:
        if not target.startswith("observation.state."):
            raise ValueError(f"robot config state {target!r} must start with 'observation.state.'")
    for target in actions:
        if not target.startswith("action."):
            raise ValueError(f"robot config action {target!r} must start with 'action.'")
    raw_actions: List[str] = []
    for feature in actions.values():
        for piece in feature.pieces:
            if piece.origin not in raw_actions:
                raw_actions.append(piece.origin)
    return _Robot(states, actions, images, raw_actions)


# --------------------------------------------------------------------------------------------
# Normalisation (Normalizer in lingbotvla/data/vla_data/transform.py)
# --------------------------------------------------------------------------------------------


def _norm_stat(stats: Mapping[str, np.ndarray], key: str, name: str, value: np.ndarray) -> np.ndarray:
    if name not in stats:
        raise KeyError(f"norm stats for {key!r} have no {name!r}")
    stat = stats[name]
    # Per-step action statistics ([horizon, dim]) are cut to the actual horizon.
    if stat.ndim == 2 and value.ndim == 2:
        if stat.shape[-1] != value.shape[-1]:
            raise ValueError(f"norm dim mismatch for {key}.{name}: stat {stat.shape}, value {value.shape}")
        if value.shape[0] > stat.shape[0]:
            raise ValueError(
                f"norm horizon mismatch for {key}.{name}: stat horizon {stat.shape[0]}, value {value.shape[0]}"
            )
        stat = stat[: value.shape[0]]
    return stat


def _normalize(value: np.ndarray, norm: str, stats: Optional[Mapping[str, np.ndarray]], key: str) -> np.ndarray:
    if norm == "identity" or stats is None:
        if norm != "identity":
            raise ValueError(f"{key}: norm type {norm!r} needs norm statistics")
        return value
    s = lambda n: _norm_stat(stats, key, n, value)  # noqa: E731
    if norm == "meanstd":
        return (value - s("mean")) / (s("std") + 1e-6)
    if norm == "std":
        return value / (s("std") + 1e-6)
    if norm in ("bounds_98", "bounds_99", "bounds_98_woclip", "bounds_99_woclip"):
        lo_name, hi_name = ("q02", "q98") if "98" in norm else ("q01", "q99")
        low, high = s(lo_name), s(hi_name)
        out = (value - low) / (high - low + 1e-6) * 2.0 - 1.0
        return out if norm.endswith("woclip") else np.clip(out, -1.5, 1.5)
    if norm in ("minmax", "minmax_woclip"):
        low, high = s("min"), s("max")
        out = (value - low) / (high - low + 1e-6) * 2 - 1
        return out if norm.endswith("woclip") else np.clip(out, -1, 1)
    raise ValueError(f"unknown norm type {norm!r}; supported: {_NORM_TYPES}")


def _unnormalize(value: np.ndarray, norm: str, stats: Optional[Mapping[str, np.ndarray]], key: str) -> np.ndarray:
    if norm == "identity" or stats is None:
        return value
    s = lambda n: _norm_stat(stats, key, n, value)  # noqa: E731
    if norm == "meanstd":
        return value * (s("std") + 1e-6) + s("mean")
    if norm == "std":
        return value * (s("std") + 1e-6)
    if norm in ("bounds_98", "bounds_99", "bounds_98_woclip", "bounds_99_woclip"):
        lo_name, hi_name = ("q02", "q98") if "98" in norm else ("q01", "q99")
        low, high = s(lo_name), s(hi_name)
        return ((value + 1.0) / 2.0) * (high - low + 1e-6) + low
    if norm in ("minmax", "minmax_woclip"):
        low, high = s("min"), s("max")
        return (value + 1) / 2.0 * (high - low + 1e-6) + low
    raise ValueError(f"unknown norm type {norm!r}; supported: {_NORM_TYPES}")


def _load_norm_stats(source: Union[None, str, Path, Mapping[str, Any]]) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    if source is None:
        return None
    document = source
    if not isinstance(source, Mapping):
        document = json.loads(Path(source).read_text())
    if "norm_stats" in document and isinstance(document["norm_stats"], Mapping):
        document = document["norm_stats"]
    return {
        key: {stat: np.asarray(values, dtype=np.float64) for stat, values in entry.items()}
        for key, entry in document.items()
        if isinstance(entry, Mapping)
    }


# --------------------------------------------------------------------------------------------
# Pose helpers (ee_pose_transform.py), numpy float64
# --------------------------------------------------------------------------------------------


def _quat_normalize(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), eps)


def _quat_inverse(q: np.ndarray) -> np.ndarray:
    inv = _quat_normalize(q).copy()
    inv[..., :3] = -inv[..., :3]
    return inv


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = np.moveaxis(a, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        axis=-1,
    )


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = _quat_normalize(q)
    vq = np.concatenate([v, np.zeros_like(v[..., :1])], axis=-1)
    return _quat_multiply(_quat_multiply(q, vq), _quat_inverse(q))[..., :3]


def _quat_canonicalize(q: np.ndarray) -> np.ndarray:
    return q * np.where(q[..., 3:4] < 0, -1.0, 1.0)


def _is_quaternion_relative(relative_type: Optional[str]) -> bool:
    return relative_type in ("quaternion", "quaternion_world", "quaternion_local")


def _absolute_pose(rel: np.ndarray, state: np.ndarray, relative_type: Optional[str], pose_dim: int = 7) -> np.ndarray:
    local = relative_type in ("local", "quaternion_local")
    if relative_type not in (None, "world", "quaternion", "quaternion_world", "local", "quaternion_local"):
        raise ValueError(f"unsupported quaternion relative type: {relative_type}")
    total = rel.shape[-1]
    if total % pose_dim:
        raise ValueError(f"end-effector width {total} is not a multiple of {pose_dim}")
    parts = []
    for i in range(total // pose_dim):
        s = i * pose_dim
        r_xyz, r_q = rel[..., s:s + 3], _quat_normalize(rel[..., s + 3:s + pose_dim])
        s_xyz, s_q = state[..., s:s + 3], _quat_normalize(state[..., s + 3:s + pose_dim])
        xyz = s_xyz + (_quat_rotate(s_q, r_xyz) if local else r_xyz)
        q = _quat_canonicalize(_quat_normalize(_quat_multiply(s_q, r_q)))
        parts.append(np.concatenate([xyz, q], axis=-1))
    return np.concatenate(parts, axis=-1)


# --------------------------------------------------------------------------------------------
# Images (torchvision v2.Resize antialias bilinear + Qwen2VLImageProcessorFast arithmetic)
# --------------------------------------------------------------------------------------------


def _aa_matrix(in_size: int, out_size: int) -> np.ndarray:
    """Dense ``[out, in]`` weights of torch's antialiased bilinear (``align_corners=False``)."""
    scale = in_size / out_size
    support = scale if scale >= 1.0 else 1.0
    invscale = 1.0 / scale if scale >= 1.0 else 1.0
    matrix = np.zeros((out_size, in_size), dtype=np.float64)
    for i in range(out_size):
        center = scale * (i + 0.5)
        lo = max(int(center - support + 0.5), 0)
        hi = min(int(center + support + 0.5), in_size)
        j = np.arange(hi - lo)
        w = np.maximum(0.0, 1.0 - np.abs((j + lo - center + 0.5) * invscale))
        total = w.sum()
        matrix[i, lo:hi] = w / total if total > 0 else w
    return matrix


def _resize_chw(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """``Resize((height, width))`` of a float32 CHW image: bilinear, antialias, no rounding."""
    _, h, w = image.shape
    if (h, w) == (height, width):
        return image
    wy, wx = _aa_matrix(h, height), _aa_matrix(w, width)
    out = np.einsum("oy,cyx,px->cop", wy, image.astype(np.float64), wx, optimize=True)
    return out.astype(np.float32)


def _prepare_view(value: Any, name: str, size: int) -> np.ndarray:
    """Raw camera frame -> float32 CHW ``[3, size, size]`` in the 0..255 range."""
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"{name} must be a rank-3 image, got shape {array.shape}")
    if array.shape[-1] != 3 and array.shape[0] == 3:
        array = array.transpose(1, 2, 0)
    if array.shape[-1] != 3:
        raise ValueError(f"{name} must have three RGB channels, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.floating) and array.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8 or floating point, got {array.dtype}")
    chw = np.ascontiguousarray(array.transpose(2, 0, 1), dtype=np.float32)
    chw = _resize_chw(chw, size, size)
    if float(chw.max(initial=0.0)) <= 2.0:
        # Reference ``_to_uint8``: a floating-point frame with max <= 2 is taken as [0, 1] and scaled to
        # uint8. The reference's frames are always float32 by then, so this applies to uint8 inputs too.
        chw = np.clip(np.round(chw * 255.0), 0, 255).astype(np.float32)
    return chw


def _patchify(chw: np.ndarray) -> np.ndarray:
    """``[3, H, W]`` float32 0..255 -> ``[(H/16)*(W/16), 3*2*16*16]`` Qwen3-VL patches."""
    c, h, w = chw.shape
    if h % (_PATCH * _MERGE) or w % (_PATCH * _MERGE):
        raise ValueError(f"image {h}x{w} is not a multiple of {_PATCH * _MERGE}")
    x = (chw.astype(np.float32) - np.float32(_MEAN_STD)) / np.float32(_MEAN_STD)
    x = np.broadcast_to(x, (_TEMPORAL, c, h, w))  # a single frame is repeated along time
    gh, gw = h // _PATCH, w // _PATCH
    x = x.reshape(_TEMPORAL, c, gh // _MERGE, _MERGE, _PATCH, gw // _MERGE, _MERGE, _PATCH)
    x = x.transpose(2, 5, 3, 6, 1, 0, 4, 7)  # (Hb, Wb, mh, mw, C, T, ph, pw)
    return np.ascontiguousarray(x.reshape(gh * gw, c * _TEMPORAL * _PATCH * _PATCH), dtype=np.float32)


# --------------------------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------------------------


class _Tokenizer:
    """``tokenizer.json`` encoder returning plain ids (no special-token post-processing)."""

    def __init__(self, model_dir: Path):
        path = Path(model_dir) / "tokenizer.json"
        if not path.is_file():
            raise FileNotFoundError(f"LingbotVla2Policy: {path} not found")
        self._native = None
        self._fast = None
        try:
            import apxinf_py

            self._native = apxinf_py.HfTokenizer.from_file(str(path))
        except ImportError:
            from tokenizers import Tokenizer

            self._fast = Tokenizer.from_file(str(path))

    def encode(self, text: str) -> List[int]:
        if self._native is not None:
            return [int(i) for i in self._native.encode(text)]
        return [int(i) for i in self._fast.encode(text, add_special_tokens=False).ids]


# --------------------------------------------------------------------------------------------
# Training-config discovery
# --------------------------------------------------------------------------------------------


def _literal(entry: Any) -> Mapping[str, Any]:
    return ast.literal_eval(entry) if isinstance(entry, str) else entry


def _read_training_config(path: Path) -> Dict[str, Any]:
    """``joints`` / ``norm_type`` / ``cameras`` / sizes from an upstream ``lingbotvla_cli.yaml``."""
    import yaml

    document = yaml.safe_load(path.read_text())
    data, train = document.get("data", {}), document.get("train", {})
    joints = [(k, int(v)) for entry in data.get("joints", []) for k, v in _literal(entry).items() if int(v) > 0]
    norm = {k: str(v) for entry in data.get("norm_type", []) for k, v in _literal(entry).items()}
    out: Dict[str, Any] = {"joints": joints, "norm_type": norm}
    if data.get("cameras"):
        out["cameras"] = tuple(data["cameras"])
    if data.get("img_size"):
        out["image_size"] = int(data["img_size"])
    for key in ("max_state_dim", "max_action_dim", "chunk_size", "tokenizer_max_length"):
        if train.get(key):
            out[key] = int(train[key])
    return out


def _find_training_config(model_dir: Path) -> Optional[Path]:
    # The reference looks three levels up from ``<run>/checkpoints/global_step_N/hf_ckpt``.
    for candidate in (model_dir / "lingbotvla_cli.yaml", model_dir.parent.parent.parent / "lingbotvla_cli.yaml"):
        if candidate.is_file():
            return candidate
    return None


def _round_to_bf16(value: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(value, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


# --------------------------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------------------------


@register_policy(MODEL_TYPE)
class LingbotVla2Policy:
    """Raw robot observation -> deployable action chunk for LingBot-VLA 2.0."""

    def __init__(
        self,
        model_runner,
        *,
        robot: Union[str, Path, Mapping[str, Any]],
        tokenizer,
        norm_stats: Union[None, str, Path, Mapping[str, Any]] = None,
        norm_type: Union[None, str, Mapping[str, str]] = None,
        joints: Optional[Sequence[Tuple[str, int]]] = None,
        joint_dims: Optional[Mapping[str, int]] = None,
        cameras: Sequence[str] = _DEFAULT_CAMERAS,
        image_size: int = 256,
        max_state_dim: int = 55,
        tokenizer_max_length: int = 72,
        prompt_key: str = CANONICAL_PROMPT_KEY,
        seed: int = 0,
    ) -> None:
        self.model_runner = model_runner
        self.tokenizer = tokenizer
        self.robot = load_robot_config(robot)
        self.stats = _load_norm_stats(norm_stats)
        self.joints: List[Tuple[str, int]] = [(str(k), int(v)) for k, v in (joints or _DEFAULT_JOINTS) if int(v) > 0]
        self.cameras = tuple(cameras)
        self.image_size = int(image_size)
        self.max_state_dim = int(max_state_dim)
        self.max_action_dim = int(model_runner.action_dim)
        self.tokenizer_max_length = int(tokenizer_max_length)
        self.prompt_key = prompt_key
        self._rng = np.random.default_rng(seed)
        self.seed = int(seed)

        if len(self.cameras) != 3:
            raise ValueError(f"LingBot-VLA 2.0 consumes exactly 3 camera views, got {self.cameras}")
        if self.image_size % (_PATCH * _MERGE):
            raise ValueError(f"image_size must be a multiple of {_PATCH * _MERGE}, got {self.image_size}")
        if sum(width for _, width in self.joints) > min(self.max_state_dim, self.max_action_dim):
            raise ValueError(f"joint widths {self.joints} exceed the {self.max_state_dim}-dim state")
        names = [k for k, _ in self.joints]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate joint groups in {names}")
        self._widths = dict(self.joints)
        self._joint_dims = {str(k): int(v) for k, v in (joint_dims or {}).items()}

        missing_cameras = [c for c in self.cameras if f"observation.images.{c}" not in self.robot.images]
        if missing_cameras:
            raise ValueError(
                f"robot config provides cameras {sorted(self.robot.images)} but the checkpoint needs "
                f"observation.images.{{{', '.join(self.cameras)}}}; missing views are not supported"
            )
        self._norm = self._resolve_norm_types(norm_type)
        self._validate_features()

        self._horizon = int(model_runner.action_horizon)
        parts = self._action_layout()
        self._part_widths = {key: width for key, width in parts}
        self._action_dim_out = sum(width for _, width in parts)
        self.metadata: Dict[str, Any] = {
            "model_type": MODEL_TYPE,
            "cameras": list(self.cameras),
            "image_keys": [self.robot.images[f"observation.images.{c}"] for c in self.cameras],
            "state_keys": sorted({p.origin for f in self.robot.states.values() for p in f.pieces}),
            "action_keys": list(self.robot.raw_actions),
            "prompt_key": self.prompt_key,
            "joints": [list(j) for j in self.joints],
            "max_state_dim": self.max_state_dim,
            "action_horizon": self._horizon,
            "action_dim": self._action_dim_out,
            "image_size": self.image_size,
        }

    # ---------------------------------------------------------------- configuration

    def _resolve_norm_types(self, norm_type) -> Dict[str, str]:
        if norm_type is None:
            raise ValueError(
                "LingbotVla2Policy needs norm_type=: a name applied to every joint group or a mapping "
                f"{{joint group: type}}, one of {_NORM_TYPES}. It is a property of the post-training run "
                "(its lingbotvla_cli.yaml data.norm_type), not of the weights."
            )
        names = {f.joint for f in list(self.robot.states.values()) + list(self.robot.actions.values())}
        if isinstance(norm_type, str):
            resolved = {name: norm_type for name in names}
        else:
            resolved = {str(k): str(v) for k, v in norm_type.items()}
            absent = sorted(names - set(resolved))
            if absent:
                raise ValueError(f"norm_type has no entry for joint groups {absent}")
        for name, kind in resolved.items():
            if kind == "sincos":
                raise NotImplementedError("sincos state normalisation is not supported")
            if kind not in _NORM_TYPES:
                raise ValueError(f"unknown norm type {kind!r} for {name}; supported: {_NORM_TYPES}")
            if kind != "identity" and self.stats is None:
                raise ValueError(f"norm type {kind!r} for {name} needs norm_stats=")
        return resolved

    def _validate_features(self) -> None:
        for feature in list(self.robot.states.values()) + list(self.robot.actions.values()):
            if feature.joint not in self._widths:
                raise ValueError(
                    f"{feature.target}: joint group {feature.joint!r} is not in the configured joints "
                    f"{[k for k, _ in self.joints]}"
                )
            if self.stats is not None and self._norm[feature.joint] != "identity" and feature.target not in self.stats:
                raise KeyError(f"norm stats have no entry for {feature.target!r} (have {sorted(self.stats)})")

    def _joint_dim(self, feature: _Feature) -> int:
        """Real (unpadded) width of one action joint group.

        Taken, in order, from ``joint_dims=``, the ``mean`` of its norm statistics (what the reference
        does), or the slices of the robot config. A whole raw key with none of these has no known width.
        """
        if feature.joint in self._joint_dims:
            return self._joint_dims[feature.joint]
        if self.stats is not None and feature.target in self.stats and "mean" in self.stats[feature.target]:
            return int(self.stats[feature.target]["mean"].shape[-1])
        width = feature.width
        if width is None:
            raise ValueError(
                f"{feature.target}: cannot tell how many values {feature.joint!r} has; pass joint_dims="
                f"{{{feature.joint!r}: N}} or norm_stats= containing its 'mean'"
            )
        return width

    def _action_layout(self) -> List[Tuple[str, int]]:
        widths: Dict[str, int] = {}
        for feature in self.robot.actions.values():
            for piece in feature.pieces:
                piece_width = (
                    piece.end - piece.start if piece.start is not None else self._joint_dim(feature)
                )
                widths[piece.origin] = widths.get(piece.origin, 0) + piece_width
        return [(key, widths[key]) for key in self.robot.raw_actions]

    # ---------------------------------------------------------------- construction

    @classmethod
    def from_pretrained(
        cls,
        model_dir,
        *,
        robot: Union[str, Path, Mapping[str, Any]],
        norm_stats: Union[None, str, Path, Mapping[str, Any]] = None,
        norm_type: Union[None, str, Mapping[str, str]] = None,
        joints: Optional[Sequence[Tuple[str, int]]] = None,
        joint_dims: Optional[Mapping[str, int]] = None,
        cameras: Optional[Sequence[str]] = None,
        training_config: Union[None, str, Path] = None,
        model_variant: Optional[str] = None,
        tactics=None,
        autotune: bool = False,
        device: str = "cuda:0",
        seed: int = 0,
        prompt_key: str = CANONICAL_PROMPT_KEY,
    ) -> "LingbotVla2Policy":
        """Load the native model and the host-side processors.

        ``robot`` is a bundled name (``"robotwin"``, ``"agilex_cobot_magic"``), a robot-config YAML path,
        or a mapping. ``training_config`` (default: ``lingbotvla_cli.yaml`` next to the checkpoint or three
        levels above it, as the reference looks) supplies ``joints``, ``norm_type`` and ``cameras``;
        explicit arguments win. The pre-trained base checkpoint has no such file, so there ``norm_type=``
        (and ``norm_stats=`` unless it is ``"identity"``) must be given.
        """
        from apxinf import ModelRunner

        model_dir = Path(model_dir)
        found = Path(training_config) if training_config is not None else _find_training_config(model_dir)
        defaults = _read_training_config(found) if found is not None else {}
        runner = ModelRunner.load(
            MODEL_TYPE, model_dir, device=device, model_variant=model_variant, tactics=tactics, autotune=autotune
        )
        return cls(
            runner,
            robot=robot,
            tokenizer=_Tokenizer(model_dir),
            norm_stats=norm_stats,
            norm_type=norm_type if norm_type is not None else defaults.get("norm_type"),
            joints=joints if joints is not None else defaults.get("joints"),
            joint_dims=joint_dims,
            cameras=tuple(cameras) if cameras is not None else defaults.get("cameras", _DEFAULT_CAMERAS),
            image_size=defaults.get("image_size", 256),
            max_state_dim=defaults.get("max_state_dim", 55),
            tokenizer_max_length=defaults.get("tokenizer_max_length", 72),
            prompt_key=prompt_key,
            seed=seed,
        )

    # ---------------------------------------------------------------- preprocessing

    def _tokens(self, observation: Mapping[str, Any]) -> np.ndarray:
        prompt = observation.get(self.prompt_key)
        if prompt is None and self.prompt_key == CANONICAL_PROMPT_KEY:
            prompt = observation.get("task")  # the reference's own key
        if prompt is None:
            raise KeyError(f"observation has no prompt under {self.prompt_key!r}")
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be a string, got {type(prompt).__name__}")
        ids = self.tokenizer.encode(_CHAT_PREFIX + prompt + _CHAT_SUFFIX)[: self.tokenizer_max_length]
        return np.asarray(ids, dtype=np.uint32)

    def _vision(self, observation: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        rows = []
        for camera in self.cameras:
            raw_key = self.robot.images[f"observation.images.{camera}"]
            if raw_key not in observation:
                raise KeyError(f"observation has no image {raw_key!r} (camera {camera}); got {sorted(observation)}")
            rows.append(_patchify(_prepare_view(observation[raw_key], raw_key, self.image_size)))
        grid = self.image_size // _PATCH
        grids = np.asarray([[1, grid, grid]] * len(rows), dtype=np.uint32)
        return np.ascontiguousarray(np.concatenate(rows, axis=0), dtype=np.float32), grids

    @staticmethod
    def _gather(feature: _Feature, observation: Mapping[str, Any]) -> np.ndarray:
        parts = []
        for piece in feature.pieces:
            if piece.origin not in observation:
                raise KeyError(f"observation has no {piece.origin!r} (needed for {feature.target})")
            raw = np.asarray(observation[piece.origin], dtype=np.float64)
            if raw.ndim != 1:
                raise ValueError(f"{piece.origin} must be one-dimensional, got shape {raw.shape}")
            parts.append(raw if piece.start is None else raw[piece.start:piece.end])
        return np.concatenate(parts, axis=-1)

    def _state(self, observation: Mapping[str, Any]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Normalised, per-group padded and concatenated state ``[max_state_dim]`` plus the per-group values."""
        slots: List[np.ndarray] = []
        groups: Dict[str, np.ndarray] = {}
        for joint, width in self.joints:
            target = f"observation.state.{joint}"
            feature = self.robot.states.get(target)
            if feature is None:
                slots.append(np.zeros(width, dtype=np.float64))
                continue
            value = self._gather(feature, observation)
            value = _normalize(value, self._norm[joint], self.stats and self.stats.get(target), target)
            pad = width - value.shape[-1]
            if pad < 0:
                raise ValueError(f"{target}: {value.shape[-1]} values exceed the {width}-wide slot")
            groups[target] = value
            slots.append(np.pad(value, (0, pad)))
        state = np.concatenate(slots).astype(np.float32)
        if state.shape[0] > self.max_state_dim:
            raise ValueError(f"state of {state.shape[0]} dims exceeds max_state_dim {self.max_state_dim}")
        state = np.pad(state, (0, self.max_state_dim - state.shape[0]))
        return state, {k: v.astype(np.float32) for k, v in groups.items()}

    def encode(self, observation: Mapping[str, Any]) -> Dict[str, Any]:
        """Raw observation -> the native model inputs plus what ``decode`` needs."""
        if not isinstance(observation, Mapping):
            raise TypeError(f"observation must be a mapping, got {type(observation)!r}")
        pixel_values, grids = self._vision(observation)
        token_ids = self._tokens(observation)
        state, groups = self._state(observation)
        return {
            "pixel_values": pixel_values,
            "image_grid_thw": grids,
            "token_ids": token_ids,
            "attention_mask": np.ones(token_ids.shape, dtype=np.uint8),
            "state": np.ascontiguousarray(state.reshape(1, -1), dtype=np.float32),
            "state_groups": groups,
        }

    # ---------------------------------------------------------------- postprocessing

    def decode(self, normalized: np.ndarray, state_groups: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Normalised ``[horizon, max_action_dim]`` chunk -> ``{raw action key: [horizon, width]}``."""
        normalized = np.asarray(normalized, dtype=np.float32)
        if normalized.ndim != 2 or normalized.shape[1] < sum(w for _, w in self.joints):
            raise ValueError(f"model output has shape {normalized.shape}")
        values: Dict[str, np.ndarray] = {}
        cursor = 0
        for joint, width in self.joints:
            target = f"action.{joint}"
            feature = self.robot.actions.get(target)
            if feature is not None:
                dim = self._joint_dim(feature)
                if dim > width:
                    raise ValueError(f"{target}: {dim} values exceed the {width}-wide slot")
                values[target] = normalized[:, cursor:cursor + dim].astype(np.float64)
            cursor += width

        out: Dict[str, np.ndarray] = {}
        decoded: Dict[str, np.ndarray] = {}
        for target, feature in self.robot.actions.items():
            if target not in values:
                continue
            joint = feature.joint
            value = _unnormalize(values[target], self._norm[joint], self.stats and self.stats.get(target), target)
            if feature.subtract_state:
                state_target = target.replace("action.", "observation.state.", 1)
                if state_target not in state_groups:
                    raise ValueError(f"{target} is relative to {state_target}, which the observation did not provide")
                state_stats = self.stats and self.stats.get(state_target)
                state = _unnormalize(state_groups[state_target].astype(np.float64), self._norm[joint], state_stats, state_target)
                if _is_quaternion_relative(feature.relative_type):
                    if "end.position" not in target:
                        raise ValueError(f"{target}: quaternion-relative actions are only defined for end.position")
                    value = _absolute_pose(value, state, feature.relative_type)
                else:
                    value = value + state
            decoded[target] = value

        for raw_key in self.robot.raw_actions:
            pieces = []
            for target, feature in self.robot.actions.items():
                for piece in feature.pieces:
                    if piece.origin == raw_key and target in decoded:
                        pieces.append((piece, target))
            if not pieces:
                continue
            if all(p.start is None for p, _ in pieces):
                out[raw_key] = decoded[pieces[0][1]].astype(np.float32)
                continue
            # Slice-built keys are re-assembled in order of their end offset, as the reference does.
            pieces.sort(key=lambda item: item[0].end)
            out[raw_key] = np.concatenate(
                [decoded[t][:, p.t_start:p.t_end] for p, t in pieces], axis=-1
            ).astype(np.float32)
        return out

    # ---------------------------------------------------------------- inference

    def _sample_noise(self) -> np.ndarray:
        shape = self._latent_shape()
        return _round_to_bf16(self._rng.standard_normal(shape, dtype=np.float32))

    def _latent_shape(self) -> Tuple[int, ...]:
        shape = getattr(self.model_runner, "initial_latent_shape", None)
        if shape is None:
            return (self._horizon, self.max_action_dim)
        return tuple(int(v) for v in shape)

    def infer(self, observation: Mapping[str, Any], *, noise: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Return deployable actions for one raw observation.

        ``actions`` is ``float32 [horizon, action_dim]``: the single raw action key's array, or all raw
        action keys concatenated in robot-config order. ``action_parts`` has them separately.
        """
        started = time.perf_counter()
        encoded = self.encode(observation)
        latent_shape = self._latent_shape()
        if noise is None:
            noise = self._sample_noise()
        else:
            noise = np.ascontiguousarray(noise, dtype=np.float32)
            if noise.shape == (1, *latent_shape):
                noise = noise[0]
            if noise.shape != latent_shape:
                raise ValueError(f"noise has shape {noise.shape}, expected {latent_shape}")
            if not np.isfinite(noise).all():
                raise ValueError("noise must contain only finite values")
            noise = _round_to_bf16(noise)

        model_started = time.perf_counter()
        normalized = np.asarray(
            self.model_runner._infer_preprocessed(
                encoded["pixel_values"],
                encoded["image_grid_thw"],
                encoded["token_ids"],
                encoded["attention_mask"],
                encoded["state"],
                None,
                noise,
            ),
            dtype=np.float32,
        )
        model_ms = (time.perf_counter() - model_started) * 1000.0
        expected = (self._horizon, self.max_action_dim)
        if normalized.shape != expected:
            raise ValueError(f"model returned action shape {normalized.shape}, expected {expected}")
        if not np.isfinite(normalized).all():
            raise FloatingPointError("model returned non-finite normalized actions")

        parts = self.decode(normalized, encoded["state_groups"])
        actions = np.ascontiguousarray(
            np.concatenate([parts[key] for key in self.robot.raw_actions if key in parts], axis=-1), dtype=np.float32
        )
        return {
            "actions": actions,
            "action_parts": parts,
            "normalized_actions": normalized,
            "token_ids": encoded["token_ids"],
            "noise": noise,
            "timing": {"model_ms": model_ms, "total_ms": (time.perf_counter() - started) * 1000.0},
            "metadata": dict(self.metadata),
        }

    __call__ = infer

    @property
    def action_dim(self) -> int:
        return self._action_dim_out

    @property
    def action_horizon(self) -> int:
        return self._horizon

    def close(self) -> None:
        close = getattr(self.model_runner, "close", None)
        if callable(close):
            close()
