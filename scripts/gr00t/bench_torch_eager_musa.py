#!/usr/bin/env python3
"""Benchmark NVIDIA GR00T N1.7 PyTorch eager inference on MUSA."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
from PIL import Image

from formats import load_noise, normalize_actions, normalize_noise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--base-image", type=Path, required=True)
    parser.add_argument("--wrist-image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--state", type=float, nargs=8, required=True)
    parser.add_argument("--noise-input", type=Path, required=True,
                        help="NPY or NPZ noise: floating values or raw BF16 uint16 bits")
    parser.add_argument("--device", default="musa:0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50,
                        help="number of measured runs (at least 2); outputs must match exactly")
    parser.add_argument(
        "--chrome-trace",
        type=Path,
        help="export one measured model call as a torch.profiler Chrome trace",
    )
    parser.add_argument(
        "--compile-action-head",
        action="store_true",
        help="wrap action_head.model.forward with torch.compile for a probe",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def require_path(path: Path, label: str, *, directory: bool = False) -> Path:
    path = path.expanduser().resolve()
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"{label} {kind} does not exist: {path}")
    return path


def load_rgb(path: Path, label: str) -> np.ndarray:
    path = require_path(path, label)
    with Image.open(path) as image:
        return np.ascontiguousarray(np.asarray(image.convert("RGB")), dtype=np.uint8)


def timing_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(array.max()),
        "samples": values,
    }


def make_observation(base_rgb: np.ndarray, wrist_rgb: np.ndarray, state: np.ndarray, prompt: str):
    return {
        "observation/image": base_rgb,
        "observation/wrist_image": wrist_rgb,
        "observation/state": {
            "x": state[0:1],
            "y": state[1:2],
            "z": state[2:3],
            "roll": state[3:4],
            "pitch": state[4:5],
            "yaw": state[5:6],
            "gripper": state[6:8],
        },
        "prompt": prompt,
    }


def validate_repeat_outputs(measured: list[dict[str, Any]]) -> dict[str, Any]:
    if len(measured) < 2:
        raise ValueError("at least two measured runs are required to verify repeatability")
    for item in measured:
        for name in ("normalized_actions", "actions"):
            item[name] = normalize_actions(item[name], name)
    for name in ("normalized_actions", "actions"):
        if any(not np.array_equal(measured[0][name], item[name]) for item in measured[1:]):
            raise RuntimeError(f"repeat captures differ ({name}); refusing to publish reference")
    return measured[0]


def main() -> int:
    args = parse_args()
    if args.warmup < 1 or args.repeat < 2:
        raise ValueError("--warmup must be at least 1 and --repeat must be at least 2")
    checkpoint = require_path(args.checkpoint, "checkpoint", directory=True)
    backbone = require_path(args.backbone, "backbone", directory=True)
    base_rgb = load_rgb(args.base_image, "base image")
    wrist_rgb = load_rgb(args.wrist_image, "wrist image")
    state = np.asarray(args.state, dtype=np.float32)
    if not np.isfinite(state).all():
        raise ValueError("state must contain finite values")
    noise = load_noise(args.noise_input)
    if args.output.suffix != ".npz":
        raise ValueError("--output must end in .npz")

    import torch
    import torch_musa  # noqa: F401 - registers the MUSA device with PyTorch.
    import gr00t.model  # noqa: F401 - registers the N1.7 AutoModel/AutoProcessor classes.
    from transformers import AutoConfig, AutoModel, Qwen3VLForConditionalGeneration

    from apxinf.policies.impls.gr00t import _NvidiaProcessorAdapter
    from gr00t.model.gr00t_n1d7 import processing_gr00t_n1d7

    adapter = _NvidiaProcessorAdapter.load(
        checkpoint,
        backbone=backbone,
        embodiment="libero_sim",
        image_keys=("observation/image", "observation/wrist_image"),
        state_key="observation/state",
        prompt_key="prompt",
        action_key=None,
    )

    cosmos_config = AutoConfig.from_pretrained(
        backbone, local_files_only=True, trust_remote_code=True
    )
    cosmos_config._attn_implementation = "sdpa"

    def construct_local_backbone(*_args, **_kwargs):
        return Qwen3VLForConditionalGeneration(cosmos_config)

    load_started = time.perf_counter()
    with (
        patch.object(
            Qwen3VLForConditionalGeneration,
            "from_pretrained",
            side_effect=construct_local_backbone,
        ),
        patch.object(
            processing_gr00t_n1d7,
            "build_processor",
            return_value=adapter.processor.processor,
        ),
    ):
        model, loading_info = AutoModel.from_pretrained(
            checkpoint,
            local_files_only=True,
            output_loading_info=True,
        )

    load_problems = {
        key: loading_info.get(key, [])
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if loading_info.get(key)
    }
    if load_problems:
        raise RuntimeError(f"checkpoint did not load strictly: {load_problems}")
    model.eval().to(device=args.device, dtype=torch.bfloat16)
    torch.musa.synchronize()
    if args.compile_action_head:
        if not hasattr(torch, "compile"):
            raise RuntimeError("this PyTorch build has no torch.compile")
        print("torch.compile(action_head.model.forward) starting", file=sys.stderr, flush=True)
        model.action_head.model.forward = torch.compile(
            model.action_head.model.forward, mode="max-autotune"
        )
        print("torch.compile wrapper installed", file=sys.stderr, flush=True)
    load_ms = (time.perf_counter() - load_started) * 1000.0

    observation = make_observation(base_rgb, wrist_rgb, state, args.prompt)
    fixed_noise = torch.from_numpy(noise).to(device=args.device, dtype=torch.bfloat16)
    noise = normalize_noise(fixed_noise.float().cpu().numpy())
    original_randn = torch.randn
    random_calls = 0

    def fixed_randn(*randn_args, **randn_kwargs):
        nonlocal random_calls
        size = randn_kwargs.get("size")
        if size is None and randn_args:
            size = randn_args[0]
        try:
            normalized_size = tuple(size) if size is not None else None
        except TypeError:
            # Inductor may use a SymInt/Integer while generating benchmark
            # inputs; that is not the model's fixed-noise call.
            normalized_size = None
        if normalized_size == tuple(fixed_noise.shape):
            random_calls += 1
            return fixed_noise.clone()
        return original_randn(*randn_args, **randn_kwargs)

    profiler = None
    if args.chrome_trace is not None:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )

    profiled_once = False

    def infer_once() -> dict[str, Any]:
        nonlocal profiled_once
        total_started = time.perf_counter()
        encoded = adapter.encode(observation)
        inputs = {
            "pixel_values": torch.from_numpy(encoded["pixel_values"]),
            "image_grid_thw": torch.from_numpy(encoded["image_grid_thw"].astype(np.int64)),
            "input_ids": torch.from_numpy(encoded["token_ids"].astype(np.int64))[None],
            "attention_mask": torch.from_numpy(encoded["attention_mask"].astype(np.int64))[None],
            "state": torch.from_numpy(encoded["state"]),
            "embodiment_id": torch.tensor([encoded["embodiment_id"]], dtype=torch.int64),
        }
        model_started = time.perf_counter()
        should_profile = profiler is not None and not profiled_once
        if should_profile:
            profiler.__enter__()
        try:
            with patch.object(torch, "randn", side_effect=fixed_randn), torch.inference_mode():
                prediction = model.get_action(inputs=inputs)["action_pred"]
        finally:
            if should_profile:
                torch.musa.synchronize()
                profiler.__exit__(None, None, None)
                profiled_once = True
        torch.musa.synchronize()
        model_ms = (time.perf_counter() - model_started) * 1000.0
        normalized = prediction.float().cpu().numpy()[0]
        actions = adapter.decode(normalized, encoded)
        total_ms = (time.perf_counter() - total_started) * 1000.0
        return {
            "normalized_actions": normalized,
            "actions": actions,
            "model_ms": model_ms,
            "total_ms": total_ms,
        }

    for _ in range(args.warmup):
        infer_once()
    measured = [infer_once() for _ in range(args.repeat)]
    if profiler is not None:
        args.chrome_trace.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(args.chrome_trace))
    if random_calls != args.warmup + args.repeat:
        raise RuntimeError(
            f"expected one fixed-noise torch.randn call per inference, got {random_calls}"
        )

    result = validate_repeat_outputs(measured)
    model_samples = [float(item["model_ms"]) for item in measured]
    total_samples = [float(item["total_ms"]) for item in measured]
    report: dict[str, Any] = {
        "backend": "nvidia-pytorch-eager-musa",
        "device": args.device,
        "checkpoint": str(checkpoint),
        "backbone_config": str(backbone),
        "load_ms": load_ms,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "model_ms": timing_summary(model_samples),
        "total_ms": timing_summary(total_samples),
        "repeat_normalized_max_abs": 0.0,
        "repeat_exact": True,
        "normalized_shape": list(result["normalized_actions"].shape),
        "actions_shape": list(result["actions"].shape),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        normalized_actions=result["normalized_actions"],
        actions=result["actions"],
        noise=noise,
        model_ms=np.asarray(model_samples, dtype=np.float64),
        total_ms=np.asarray(total_samples, dtype=np.float64),
        report=np.asarray(json.dumps(report)),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
