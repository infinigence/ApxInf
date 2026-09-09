"""Qwen-Drive L2 policy: driving-scene preprocessing + native model calls.

Registers ``QwenDrivePolicy`` under ``model_type="qwen_drive"`` so
:class:`~apxinf.policies.auto.AutoPolicy` can dispatch to it.

The policy owns canonical preprocessing and postprocessing, mirroring the
reference ``QwenDriveProcessor`` contract exactly: PIL bicubic resize
(torchvision dispatches PIL inputs to PIL resize), ``smart_resize`` factor
snapping with Python ``round()``, pixel normalization ``(x/255 - 0.5) / 0.5``,
block-ordered patchify (permute ``(1,3,6,4,7,0,2,5,8)``), ChatML prompt
composition, history re-referencing/normalization, detokenization with
think-block stripping, and trajectory denormalization with heading wrap.

All model computation runs in the Rust/CUDA executor through
``apxinf_py.QwenDriveModel``. No torch/Transformers model execution, no CPU
model hot path, no subprocess or remote inference fallback happens here.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..registry import register_policy

__all__ = ["QwenDrivePolicy"]

CAMERA_VIEWS = ("<FRONT VIEW>", "<FRONT LEFT VIEW>", "<FRONT RIGHT VIEW>")
NAV_COMMANDS = ("GO STRAIGHT", "TURN LEFT", "TURN RIGHT")
HISTORY_FRAME_LABELS = ("t-1.5s", "t-1.0s", "t-0.5s", "t-0s")
REASONING_REQUEST = (
    "\n\nGive a one-sentence brief reasoning of the ego's future driving decision ONLY."
)
THINK_CLOSE = "</think>"

_INSTRUCTION_HEADER = (
    "The input images are organized by camera view. Each view contains {num_frames} temporal "
    "frames captured at {interval:g}s intervals (frame 0 at {first_label}, frame {last_frame} is "
    "the current frame at t=0s).\n"
    "1. Historical trajectories (x, y, heading) in the current frame's ego coordinate system. "
    "Positive x points forward, positive y points left, and a positive heading indicates a left "
    "turn\uff1a\n"
)


def _strip_thinking(text: str) -> str:
    """Drop a leading thinking block, keeping the answer that follows it."""
    return text.split(THINK_CLOSE)[-1].strip()


def _round_by_factor(value: float, factor: int) -> int:
    return round(value / factor) * factor


def smart_resize(
    height: int, width: int, factor: int, min_pixels: int, max_pixels: int
) -> Tuple[int, int]:
    """Snap a resolution to a multiple of ``factor`` inside a pixel budget.

    Bit-identical to the reference: Python ``round()`` is banker's rounding,
    and the shrink/enlarge branches match its floor/ceil behavior.
    """
    bar_h = _round_by_factor(height, factor)
    bar_w = _round_by_factor(width, factor)
    pixels = bar_h * bar_w
    if pixels > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        bar_h = math.floor(height / beta / factor) * factor
        bar_w = math.floor(width / beta / factor) * factor
    elif pixels < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        bar_h = math.ceil(height * beta / factor) * factor
        bar_w = math.ceil(width * beta / factor) * factor
    return bar_h, bar_w


def _wrap_heading(trajectory: np.ndarray) -> np.ndarray:
    heading = np.remainder(trajectory[..., 2:3] + math.pi, 2 * math.pi) - math.pi
    return np.concatenate([trajectory[..., :2], heading], axis=-1)


class _Tokenizer:
    """Uniform facade over the fast `tokenizers` backend or AutoTokenizer.

    Tokenization is a preprocessing utility only; no model code is involved.
    """

    def __init__(self, model_dir: Path):
        self._fast = None
        self._slow = None
        tokenizer_json = model_dir / "tokenizer.json"
        if tokenizer_json.is_file():
            try:
                from tokenizers import Tokenizer

                self._fast = Tokenizer.from_file(str(tokenizer_json))
            except ImportError:
                self._fast = None
        if self._fast is None:
            from transformers import AutoTokenizer

            self._slow = AutoTokenizer.from_pretrained(str(model_dir))

    def encode(self, text: str) -> list:
        if self._fast is not None:
            return list(self._fast.encode(text, add_special_tokens=False).ids)
        return list(self._slow.encode(text, add_special_tokens=False))

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        if self._fast is not None:
            return self._fast.decode(list(ids), skip_special_tokens=skip_special_tokens)
        return self._slow.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def token_id(self, token: str) -> int:
        if self._fast is not None:
            value = self._fast.token_to_id(token)
        else:
            value = self._slow.convert_tokens_to_ids(token)
        if value is None:
            raise KeyError(f"token {token!r} is not in the tokenizer vocabulary")
        return int(value)


@register_policy("qwen_drive")
class QwenDrivePolicy:
    """Scene dict -> native Qwen-Drive outputs, per the frozen public contract."""

    def __init__(
        self,
        model,
        *,
        config: Mapping[str, Any],
        tokenizer: _Tokenizer,
        mode: str,
        eos_token_ids: Sequence[int],
        seed: int,
        max_new_tokens: int,
        min_new_tokens: int,
        num_steps: int,
    ):
        self.model = model
        self.config = dict(config)
        self.tokenizer = tokenizer
        self.mode = mode
        self.eos_token_ids = list(eos_token_ids)
        self.seed = int(seed)
        self.max_new_tokens = int(max_new_tokens)
        self.min_new_tokens = int(min_new_tokens)
        self.num_steps = int(num_steps)

        vlm = self.config["vlm_config"]
        self.image_token_id = int(vlm["image_token_id"])
        self.vision_start_id = int(vlm["vision_start_token_id"])
        self.vision_end_id = int(vlm["vision_end_token_id"])
        self.im_start_id = self.tokenizer.token_id("<|im_start|>")
        self.im_end_id = self.tokenizer.token_id("<|im_end|>")
        self.newline_ids = self.tokenizer.encode("\n")

        self.patch_size = int(self.config["image_patch_size"])
        self.merge = int(self.config["image_spatial_merge_size"])
        self.temporal = int(self.config["image_temporal_patch_size"])
        self.factor = self.patch_size * self.merge
        self.min_pixels = 4 * self.factor**2
        self.grid_pixel_limit = 12800 * self.factor**2
        self.history_pixels = int(self.config["history_image_pixels"])
        self.current_pixels = int(self.config["current_image_pixels"])
        self.scale = np.asarray(self.config["trajectory_scale"], dtype=np.float32)
        self.metadata = {
            "model_type": "qwen_drive",
            "mode": self.mode,
            "action_horizon": int(self.config["num_future_points"]),
            "action_dim": int(self.config["trajectory_point_dim"]),
            "num_inference_steps": self.num_steps,
            "native_executor": "apxinf-cuda",
        }

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_pretrained(
        cls,
        model_dir,
        *,
        precision: str = "bf16",
        mode: str = "direct_planning",
        planner=None,
        device: str = "cuda:0",
        seed: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        min_new_tokens: Optional[int] = None,
        num_steps: Optional[int] = None,
        eos_token_ids: Optional[Sequence[int]] = None,
        **kwargs,
    ) -> "QwenDrivePolicy":
        if kwargs:
            raise TypeError(
                f"QwenDrivePolicy.from_pretrained: unexpected kwargs {sorted(kwargs)}"
            )
        if precision not in ("bf16", "auto"):
            raise ValueError(
                f"QwenDrivePolicy: qwen_drive is bf16-native, got precision={precision!r}"
            )
        import apxinf_py  # lazy: processor-only users never import the binding

        model_dir = Path(model_dir)
        config = json.loads((model_dir / "config.json").read_text())
        planner_path = Path(planner) if planner is not None else None
        model = apxinf_py.QwenDriveModel.load(
            model_dir,
            planner_path,
            device=device,
            precision="bf16",
        )
        tokenizer = _Tokenizer(model_dir)

        if eos_token_ids is not None:
            eos = list(eos_token_ids)
        else:
            # Exactly what the reference's generate() consumed: the checkpoint's
            # own generation_config eos list (fall back to the text config eos).
            eos = []
            gen_cfg = model_dir / "generation_config.json"
            if gen_cfg.is_file():
                value = json.loads(gen_cfg.read_text()).get("eos_token_id")
                if isinstance(value, list):
                    eos = [int(v) for v in value]
                elif value is not None:
                    eos = [int(value)]
            if not eos:
                eos = [int(config["vlm_config"]["text_config"]["eos_token_id"])]

        return cls(
            model,
            config=config,
            tokenizer=tokenizer,
            mode=mode,
            eos_token_ids=eos,
            seed=int(config.get("noise_seed", 42)) if seed is None else int(seed),
            max_new_tokens=max_new_tokens
            if max_new_tokens is not None
            else int(config.get("max_reasoning_tokens", 256)),
            min_new_tokens=min_new_tokens
            if min_new_tokens is not None
            else int(config.get("min_reasoning_tokens", 10)),
            num_steps=num_steps
            if num_steps is not None
            else int(config.get("num_inference_steps", 10)),
        )

    # ------------------------------------------------------------------ preprocessing

    def _patchify(self, image: np.ndarray, target_size, budget: int):
        """Resize one frame onto the patch grid and flatten it into patches."""
        from PIL import Image

        array = np.ascontiguousarray(image, dtype=np.uint8)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(
                f"QwenDrivePolicy: expected an RGB uint8 image, got shape {array.shape}"
            )
        pil = Image.fromarray(array, mode="RGB")
        max_pixels = budget
        if target_size is not None:
            width, height = int(target_size[0]), int(target_size[1])
            pil = pil.resize((width, height), resample=Image.BICUBIC)
            max_pixels = self.grid_pixel_limit
        width, height = pil.size
        grid_h, grid_w = smart_resize(
            height, width, self.factor, self.min_pixels, max_pixels
        )
        if (grid_h, grid_w) != (height, width):
            pil = pil.resize((grid_w, grid_h), resample=Image.BICUBIC)
        pixels = np.asarray(pil, dtype=np.float32)
        pixels = pixels / 255.0
        pixels = (pixels - 0.5) / 0.5
        rows = grid_h // self.patch_size
        cols = grid_w // self.patch_size
        merge = self.merge
        x = pixels.transpose(2, 0, 1)[:, None, :, :]
        x = np.broadcast_to(x, (3, self.temporal, grid_h, grid_w))
        x = x.reshape(
            3,
            1,
            self.temporal,
            rows // merge,
            merge,
            self.patch_size,
            cols // merge,
            merge,
            self.patch_size,
        )
        x = x.transpose(1, 3, 6, 4, 7, 0, 2, 5, 8)
        patches = np.ascontiguousarray(x.reshape(rows * cols, -1), dtype=np.float32)
        return patches, (rows, cols)

    def _scene_views(self, observation: Mapping[str, Any]):
        views = observation.get("views")
        if views is None:
            raise KeyError("QwenDrivePolicy: observation is missing 'views'")
        if isinstance(views, Mapping):
            names = [view for view in CAMERA_VIEWS if view in views]
            names += [name for name in views.keys() if name not in names]
            return [(name, views[name]) for name in names]
        raise TypeError(
            f"QwenDrivePolicy: 'views' must be a mapping of camera name -> frames, got {type(views)!r}"
        )

    def _scene_frames(self, views) -> list:
        """All frames grouped by view then timestamp: (image, target_size, is_current)."""
        frames = []
        for _name, view_frames in views:
            view_frames = list(view_frames)
            per_view = len(view_frames)
            for index, frame in enumerate(view_frames):
                if isinstance(frame, Mapping):
                    image = frame["image"]
                    target = frame.get("target_size")
                else:
                    image = frame
                    target = None
                frames.append((image, target, index == per_view - 1))
        return frames

    def _instruction(self, observation, num_frames, history, nav_command) -> str:
        text = observation.get("instruction_text")
        if text is not None:
            return str(text)
        labels = HISTORY_FRAME_LABELS[-num_frames:]
        stride = max(1, (len(history) - 1) // max(1, num_frames - 1))
        lines = "".join(
            " -{}: ({:.4f}, {:.4f}, {:.4f});\n".format(label, *history[index * stride])
            for index, label in enumerate(labels)
        )
        header = _INSTRUCTION_HEADER.format(
            num_frames=num_frames,
            interval=1.5 / max(1, num_frames - 1),
            first_label=labels[0],
            last_frame=num_frames - 1,
        )
        command = NAV_COMMANDS[int(nav_command)]
        return f"{header}{lines}2. Active navigation command: [{command}]"

    def _build_scene_ids(self, observation, views, token_counts, with_reasoning) -> list:
        per_view = len(views[0][1])
        body: list = []
        for view_index, (view_name, _frames) in enumerate(views):
            body += self.tokenizer.encode(view_name)
            for frame_index in range(per_view):
                body += self.tokenizer.encode(f"frame: {frame_index}")
                count = token_counts[view_index * per_view + frame_index]
                body += [self.vision_start_id] + [self.image_token_id] * count + [self.vision_end_id]
        nav_command = int(observation["nav_command"])
        history = np.asarray(observation["history"], dtype=np.float64)
        instruction = self._instruction(observation, per_view, history, nav_command)
        if with_reasoning:
            instruction = instruction + REASONING_REQUEST
        body += self.tokenizer.encode(instruction)
        assistant_header = [self.im_start_id] + self.tokenizer.encode("assistant") + self.newline_ids
        prompt = (
            [self.im_start_id]
            + self.tokenizer.encode("user")
            + self.newline_ids
            + body
            + [self.im_end_id]
            + self.newline_ids
            + assistant_header
        )
        if not with_reasoning:
            prompt += [self.im_end_id] + self.newline_ids
        return prompt

    def _encode_vqa(self, observation) -> Tuple[list, np.ndarray, list]:
        question = observation.get("question")
        if not isinstance(question, str):
            raise KeyError("QwenDrivePolicy: the VQA mode needs a 'question' string")
        if "images" in observation:
            raw_frames = []
            for frame in observation["images"]:
                if isinstance(frame, Mapping):
                    raw_frames.append((frame["image"], None, True))
                else:
                    raw_frames.append((frame, None, True))
        else:
            views = self._scene_views(observation)
            raw_frames = [
                (image, None, True) for (image, _target, _cur) in self._scene_frames(views)
            ]
        # encode_vqa applies the current-frame budget to every frame and never
        # uses the planning target sizes (reference CameraFrame(image)).
        patch_list, grids, token_counts = [], [], []
        for image, _target, _cur in raw_frames:
            patches, (rows, cols) = self._patchify(image, None, self.current_pixels)
            patch_list.append(patches)
            grids.append([1, rows, cols])
            token_counts.append(rows * cols // self.merge**2)
        body: list = []
        for count in token_counts:
            body += [self.vision_start_id] + [self.image_token_id] * count + [self.vision_end_id]
        body += self.tokenizer.encode(question)
        prompt = (
            [self.im_start_id]
            + self.tokenizer.encode("user")
            + self.newline_ids
            + body
            + [self.im_end_id]
            + self.newline_ids
            + [self.im_start_id]
            + self.tokenizer.encode("assistant")
            + self.newline_ids
        )
        return (
            prompt,
            np.ascontiguousarray(np.concatenate(patch_list, axis=0), dtype=np.float32),
            grids,
        )

    def _conditioning(self, observation):
        history = np.asarray(observation["history"], dtype=np.float32)
        shifted = _wrap_heading(history - history[0:1, :])
        normalized = _wrap_heading(shifted[1:, :]) / self.scale
        velocity = np.asarray(observation["history_velocity"], dtype=np.float32)
        acceleration = np.asarray(observation["history_acceleration"], dtype=np.float32)
        ego = np.concatenate(
            [
                np.asarray(observation["ego_velocity"], dtype=np.float32),
                np.asarray(observation["ego_acceleration"], dtype=np.float32),
                np.asarray(observation["driving_command"], dtype=np.float32),
            ]
        )
        return (
            np.ascontiguousarray(normalized.reshape(-1), dtype=np.float32),
            np.ascontiguousarray(velocity.reshape(-1), dtype=np.float32),
            np.ascontiguousarray(acceleration.reshape(-1), dtype=np.float32),
            np.ascontiguousarray(ego, dtype=np.float32),
            int(observation["nav_command"]),
        )

    def _noise(self, observation, noise) -> np.ndarray:
        selected = noise
        if selected is None:
            selected = observation.get("noise")
        if selected is not None:
            array = np.ascontiguousarray(selected, dtype=np.float32)
        else:
            # Non-reference fallback: the acceptance contract always supplies
            # exact noise; without it we draw a deterministic numpy sample.
            rng = np.random.default_rng(self.seed)
            array = rng.standard_normal((1, 50, 3), dtype=np.float32)
        if array.size != 150:
            raise ValueError(
                f"QwenDrivePolicy: noise must hold 50x3 values, got shape {array.shape}"
            )
        return array

    # ------------------------------------------------------------------ inference

    def infer(self, observation: Mapping[str, Any], *, noise: Optional[np.ndarray] = None) -> dict:
        started = time.perf_counter()
        mode = observation.get("mode", self.mode)
        if mode == "vqa":
            return self._infer_vqa(observation, started)
        if mode in ("direct", "direct_planning"):
            return self._infer_planning(observation, False, noise, started)
        if mode in ("reasoning", "reasoning_planning"):
            return self._infer_planning(observation, True, noise, started)
        if mode == "perception":
            raise RuntimeError(
                "QwenDrivePolicy: perception is a declared pending gap in this revision - "
                "the BEV stack (SimpleFPN/DepthNet/voxel pool/BEVFormer/deformable "
                "attention/occupancy/map heads) requires the conv2d/conv3d/GroupNorm/"
                "grid_sample/ms_deform_attn/voxel_pool kernel families, which are not "
                "yet in the native kernel set"
            )
        raise ValueError(f"QwenDrivePolicy: unknown mode {mode!r}")

    __call__ = infer

    def _infer_vqa(self, observation, started) -> dict:
        if bool(observation.get("do_sample", False)):
            raise ValueError(
                "QwenDrivePolicy: do_sample=True is unsupported; the frozen VQA protocol "
                "is top_k=1 (greedy)"
            )
        token_ids, pixel_values, grid_thw = self._encode_vqa(observation)
        max_new = int(observation.get("max_new_tokens", 2048))
        # TEMP-DIAG (implement_r1): clamp the decode cap so the diagnostic receipt
        # returns with heartbeat data under the 300s job limit; revert in the
        # acceptance-bound revision (mode=all).
        max_new = min(max_new, 64)
        model_started = time.perf_counter()
        generated = self.model.generate_tokens(
            token_ids,
            pixel_values,
            grid_thw,
            max_new,
            0,
            self.eos_token_ids,
        )
        model_ms = (time.perf_counter() - model_started) * 1000.0
        text = _strip_thinking(self.tokenizer.decode(generated, skip_special_tokens=True))
        return {
            "token_ids": list(generated),
            "text": text,
            "timing": {"model_ms": model_ms, "total_ms": (time.perf_counter() - started) * 1000.0},
            "metadata": self.metadata,
        }

    def _infer_planning(self, observation, with_reasoning: bool, noise, started) -> dict:
        views = self._scene_views(observation)
        frames = self._scene_frames(views)
        patch_list, grids, token_counts = [], [], []
        for image, target, is_current in frames:
            budget = self.current_pixels if is_current else self.history_pixels
            patches, (rows, cols) = self._patchify(image, target, budget)
            patch_list.append(patches)
            grids.append([1, rows, cols])
            token_counts.append(rows * cols // self.merge**2)
        pixel_values = np.ascontiguousarray(np.concatenate(patch_list, axis=0), dtype=np.float32)
        token_ids = self._build_scene_ids(observation, views, token_counts, with_reasoning)
        history, velocity, acceleration, ego, nav_command = self._conditioning(observation)
        noise_array = self._noise(observation, noise)
        model_started = time.perf_counter()
        if with_reasoning:
            terminators = [self.im_end_id] + [
                token for token in self.eos_token_ids if token != self.im_end_id
            ]
            generated, trajectory = self.model.plan_reasoning(
                token_ids,
                pixel_values,
                grids,
                history,
                velocity,
                acceleration,
                ego,
                nav_command,
                noise_array,
                self.max_new_tokens,
                self.min_new_tokens,
                terminators,
                self.im_end_id,
                self.newline_ids,
                None,
            )
            content = list(generated)
            for position, token in enumerate(generated):
                if token in terminators:
                    content = list(generated[:position])
                    break
            reasoning = _strip_thinking(
                self.tokenizer.decode(content, skip_special_tokens=True)
            )
        else:
            generated = None
            reasoning = None
            trajectory = self.model.plan_direct(
                token_ids,
                pixel_values,
                grids,
                history,
                velocity,
                acceleration,
                ego,
                nav_command,
                noise_array,
                None,
            )
        model_ms = (time.perf_counter() - model_started) * 1000.0
        actions = _wrap_heading(
            np.asarray(trajectory, dtype=np.float32) * self.scale
        )[None]
        result = {
            "actions": np.ascontiguousarray(actions, dtype=np.float32),
            "timing": {"model_ms": model_ms, "total_ms": (time.perf_counter() - started) * 1000.0},
            "metadata": self.metadata,
        }
        if with_reasoning:
            result["reasoning"] = reasoning
            result["token_ids"] = list(generated)
        return result

    @property
    def action_dim(self) -> int:
        return int(self.config["trajectory_point_dim"])

    @property
    def action_horizon(self) -> int:
        return int(self.config["num_future_points"])

    def close(self) -> None:
        self.model = None

    def __repr__(self) -> str:
        return f"QwenDrivePolicy(mode={self.mode!r}, steps={self.num_steps})"
