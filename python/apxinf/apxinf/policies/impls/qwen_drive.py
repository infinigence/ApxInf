"""Qwen-Drive L2 policy: driving-scene preprocessing + native model calls.

Registers ``QwenDrivePolicy`` under ``model_type="qwen_drive"`` so
:class:`~apxinf.policies.auto.AutoPolicy` can dispatch to it.

The policy owns canonical preprocessing and postprocessing, mirroring the
reference ``QwenDriveProcessor`` contract exactly: PIL bicubic resize
(torchvision dispatches PIL inputs to PIL resize), ``smart_resize`` factor
snapping with Python ``round()``, pixel normalization ``(x/255 - 0.5) / 0.5``,
block-ordered patchify (permute ``(1,3,6,4,7,0,2,5,8)``), ChatML prompt
composition, history re-referencing/normalization, and trajectory
denormalization with heading wrap. Reasoning stays internal to planning.

All model computation runs in the Rust/CUDA executor through
``apxinf.ModelRunner``. No torch/Transformers model execution, no CPU
model hot path, no subprocess or remote inference fallback happens here.
"""

from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
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

_INSTRUCTION_HEADER = (
    "The input images are organized by camera view. Each view contains {num_frames} temporal "
    "frames captured at {interval:g}s intervals (frame 0 at {first_label}, frame {last_frame} is "
    "the current frame at t=0s).\n"
    "1. Historical trajectories (x, y, heading) in the current frame's ego coordinate system. "
    "Positive x points forward, positive y points left, and a positive heading indicates a left "
    "turn\uff1a\n"
)


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
    """Driving scene -> trajectory, optionally with internal reasoning text."""

    def __init__(
        self,
        model_runner,
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
        self.model_runner = model_runner
        self.config = dict(config)
        self.tokenizer = tokenizer
        self.mode = self._validate_mode(mode)
        self.eos_token_ids = list(eos_token_ids)
        self.seed = int(seed)
        self.max_new_tokens = int(max_new_tokens)
        self.min_new_tokens = int(min_new_tokens)
        self.num_steps = int(num_steps)
        if self.num_steps <= 0:
            raise ValueError("QwenDrivePolicy: num_steps must be positive")
        if self.max_new_tokens <= 0 or not 0 <= self.min_new_tokens <= self.max_new_tokens:
            raise ValueError("QwenDrivePolicy: invalid reasoning token bounds")

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

    @staticmethod
    def _validate_mode(mode: str) -> str:
        if mode in ("direct", "direct_planning", "reasoning", "reasoning_planning"):
            return mode
        # Distinguish "not carried over yet" from "never existed": both paths shipped
        # before this family moved to the shared VLA organization, and both are
        # blocked on decisions rather than deleted outright. Reject at load time
        # rather than at the first request.
        if mode == "vqa":
            raise ValueError(
                "QwenDrivePolicy: VQA/text generation is not exposed by the planning "
                "runtime; the Qwen backbone supplies scene context to the planner and "
                "is not a separate language model. Re-introducing it is a VLM/LLM "
                "interface decision, not a gap in this policy"
            )
        if mode == "perception":
            raise ValueError(
                "QwenDrivePolicy: perception is a declared pending gap - the BEV stack "
                "(SimpleFPN/DepthNet/voxel pool/BEVFormer/deformable attention/occupancy/"
                "map heads) requires the conv2d/conv3d/GroupNorm/grid_sample/"
                "ms_deform_attn/voxel_pool kernel families, which are not yet in the "
                "native kernel set"
            )
        raise ValueError(f"QwenDrivePolicy supports only direct/reasoning planning, got {mode!r}")

    @classmethod
    def from_pretrained(
        cls,
        model_dir,
        *,
        model_variant: str = "bf16",
        tactics=None,
        autotune: bool = False,
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
        if model_variant not in ("bf16", "auto"):
            raise ValueError(
                f"QwenDrivePolicy: qwen_drive is bf16-native, got model_variant={model_variant!r}"
            )
        mode = cls._validate_mode(mode)
        if num_steps is not None and int(num_steps) <= 0:
            raise ValueError("QwenDrivePolicy: num_steps must be positive")
        from apxinf import ModelRunner

        model_dir = Path(model_dir)
        config = json.loads((model_dir / "config.json").read_text())
        planner_path = Path(planner) if planner is not None else None
        model_runner = ModelRunner.load(
            "qwen_drive", model_dir, device=device, model_variant=model_variant,
            assets={"planner": planner_path} if planner_path is not None else None,
            tactics=tactics, autotune=autotune,
        )
        tokenizer = _Tokenizer(model_dir)

        if eos_token_ids is not None:
            eos = list(eos_token_ids)
        else:
            # The reference calls the VLM created from its nested config.
            # The outer Drive generation_config.json is not applied to it.
            vlm_config = config["vlm_config"]
            value = vlm_config.get("eos_token_id")
            if value is None:
                value = vlm_config["text_config"]["eos_token_id"]
            eos = [int(v) for v in value] if isinstance(value, list) else [int(value)]

        return cls(
            model_runner,
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

    def _patchify_batch(self, items):
        """Patchify several frames, in order, on as many cores as are useful.

        Frames do not interact: each call reads only its own image and the
        policy's immutable grid constants, so running them concurrently
        produces the same bytes in the same order as the loop it replaces --
        checked by hashing the concatenated output against the serial result,
        not assumed. It is worth doing because this is the largest single item
        in a scene that is not GPU work: on the RTX 4090 the twelve frames of a
        multi-camera scene cost 219 ms of PIL bicubic resize and numpy permutation
        before the first kernel launches, which is 15% of the scene. Twelve
        workers take that to 50 ms. Both PIL's resampling and numpy's copies
        drop the GIL, which is why threads rather than processes: no image is
        pickled and no array is copied between address spaces.

        ``APXINF_QWEN_PREPROC_THREADS`` overrides the worker count; 0 or 1
        restores the serial loop.
        """
        if len(items) < 2:
            return [self._patchify(image, target, budget) for image, target, budget in items]
        workers = self._preproc_workers(len(items))
        if workers < 2:
            return [self._patchify(image, target, budget) for image, target, budget in items]
        pool = self._preproc_pool(workers)
        return list(pool.map(lambda item: self._patchify(*item), items))

    def _preproc_workers(self, frames: int) -> int:
        override = os.environ.get("APXINF_QWEN_PREPROC_THREADS")
        if override is not None:
            try:
                return max(0, int(override))
            except ValueError:
                pass
        # One worker per frame, bounded by the cores. Measured on all three
        # boards and all three want it: at twelve workers against eight, the
        # RTX 4090 is 4.36x against 3.36x, Orin 4.89x against 3.86x, Thor 4.25x
        # against 3.28x. Frames are unequal -- the tail is one large one -- so
        # there is nothing to gain by giving a worker two of them.
        return max(1, min(frames, os.cpu_count() or 1))

    def _preproc_pool(self, workers: int) -> ThreadPoolExecutor:
        pool = getattr(self, "_patch_pool", None)
        if pool is None or getattr(self, "_patch_pool_workers", 0) != workers:
            if pool is not None:
                pool.shutdown(wait=False)
            pool = ThreadPoolExecutor(max_workers=workers,
                                      thread_name_prefix="apxinf-patchify")
            self._patch_pool = pool
            self._patch_pool_workers = workers
        return pool

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
            array = rng.standard_normal((1, self.action_horizon, self.action_dim), dtype=np.float32) * float(self.config.get("noise_init_std", 1.0))
        if array.shape not in ((self.action_horizon, self.action_dim), (1, self.action_horizon, self.action_dim)) or not np.isfinite(array).all():
            raise ValueError(
                f"QwenDrivePolicy: noise must be finite [horizon, dim] or [1, horizon, dim], got shape {array.shape}"
            )
        return array

    # ------------------------------------------------------------------ inference

    def infer(self, observation: Mapping[str, Any], *, noise: Optional[np.ndarray] = None) -> dict:
        started = time.perf_counter()
        mode = self._validate_mode(observation.get("mode", self.mode))
        if mode in ("direct", "direct_planning"):
            return self._infer_planning(observation, False, noise, started)
        if mode in ("reasoning", "reasoning_planning"):
            return self._infer_planning(observation, True, noise, started)
        raise ValueError(f"QwenDrivePolicy: unknown mode {mode!r}")

    __call__ = infer

    def _infer_planning(self, observation, with_reasoning: bool, noise, started) -> dict:
        views = self._scene_views(observation)
        frames = self._scene_frames(views)
        patch_list, grids, token_counts = [], [], []
        for patches, (rows, cols) in self._patchify_batch(
            [(image, target,
              self.current_pixels if is_current else self.history_pixels)
             for image, target, is_current in frames]
        ):
            patch_list.append(patches)
            grids.append([1, rows, cols])
            token_counts.append(rows * cols // self.merge**2)
        pixel_values = np.ascontiguousarray(np.concatenate(patch_list, axis=0), dtype=np.float32)
        token_ids = self._build_scene_ids(observation, views, token_counts, with_reasoning)
        history, velocity, acceleration, ego, nav_command = self._conditioning(observation)
        noise_array = self._noise(observation, noise)
        model_started = time.perf_counter()
        # Canonical state layout is validated by the native planning runner.
        state = np.ascontiguousarray(
            np.concatenate([history, velocity, acceleration, ego, [nav_command]]),
            dtype=np.float32,
        )
        options = {"num_steps": self.num_steps}
        if with_reasoning:
            terminators = [self.im_end_id] + [
                token for token in self.eos_token_ids if token != self.im_end_id
            ]
            options.update(
                max_new_tokens=self.max_new_tokens, min_new_tokens=self.min_new_tokens,
                terminator_ids=terminators, closing_ids=[self.im_end_id, *self.newline_ids],
            )
        trajectory = self.model_runner._infer_preprocessed(
            pixel_values, np.ascontiguousarray(grids, dtype=np.uint32),
            np.ascontiguousarray(token_ids, dtype=np.uint32),
            np.ones(len(token_ids), dtype=np.uint8), state, None, noise_array,
            **options,
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
        return result

    @property
    def action_dim(self) -> int:
        return int(self.config["trajectory_point_dim"])

    @property
    def action_horizon(self) -> int:
        return int(self.config["num_future_points"])

    def close(self) -> None:
        self.model_runner = None
        pool = getattr(self, "_patch_pool", None)
        if pool is not None:
            pool.shutdown(wait=False)
            self._patch_pool = None
            self._patch_pool_workers = 0

    def __repr__(self) -> str:
        return f"QwenDrivePolicy(mode={self.mode!r}, steps={self.num_steps})"
