#!/usr/bin/env python3
"""Fixed-input WallOSS latency, parity hash and stage-profile benchmark."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_APXINF_PKG = _REPO_ROOT / "python" / "apxinf"
if _APXINF_PKG.is_dir() and str(_APXINF_PKG) not in sys.path:
    sys.path.insert(0, str(_APXINF_PKG))


@contextlib.contextmanager
def _cuda_profiler_capture():
    """Optionally delimit steady-state CUDA work for Nsight Systems."""
    if os.environ.get("APXINF_BENCH_CUDA_PROFILER_API") != "1":
        yield
        return

    cudart = ctypes.CDLL("libcudart.so")
    start = cudart.cudaProfilerStart
    stop = cudart.cudaProfilerStop
    start.argtypes = []
    start.restype = ctypes.c_int
    stop.argtypes = []
    stop.restype = ctypes.c_int
    status = start()
    if status != 0:
        raise RuntimeError(f"cudaProfilerStart failed with status {status}")
    try:
        yield
    finally:
        status = stop()
        if status != 0:
            raise RuntimeError(f"cudaProfilerStop failed with status {status}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=pathlib.Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--height", type=int, default=256, help="raw camera height")
    parser.add_argument("--width", type=int, default=256, help="raw camera width")
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="override the training resolution of every camera (-1: none)",
    )
    parser.add_argument("--tactics", type=pathlib.Path)
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--eager", action="store_true", help="disable CUDA graph capture")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="also report CUDA-event vision/prefix/denoise times and graph replay time",
    )
    parser.add_argument("--dump-actions", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmup < 0 or args.views <= 0:
        parser.error("--iterations and --views must be positive, --warmup non-negative")
    return args


def stats(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {
        "samples": len(ordered),
        "min": ordered[0],
        "p50": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    args = parse_args()
    if args.eager:
        os.environ["APXINF_WALLOSS_NO_GRAPH"] = "1"
    from apxinf import WallossPolicy

    names = ("face_view", "right_wrist_view", "left_wrist_view", "move1_view")
    if args.views > len(names):
        sys.exit(f"--views supports up to {len(names)} cameras")
    image_keys = tuple(f"observation/camera{index}" for index in range(args.views))
    policy = WallossPolicy.from_pretrained(
        args.model_dir,
        device=args.device,
        precision=args.precision,
        tactics=args.tactics,
        image_keys=image_keys,
        camera_names=names[: args.views],
        seed=args.seed,
        **({"autotune": True} if args.autotune else {}),
    )
    if args.resolution is not None:
        policy.processor.resolutions = (args.resolution,) * args.views

    rng = np.random.default_rng(42)
    observation = {
        key: rng.integers(0, 256, (args.height, args.width, 3), dtype=np.uint8)
        for key in image_keys
    }
    observation["observation/state"] = rng.standard_normal(7, dtype=np.float32)
    observation["prompt"] = "pick up the red cube"
    noise = rng.standard_normal(
        (policy.action_horizon, policy.model_runner.action_dim), dtype=np.float32
    )

    first = policy.infer(observation, noise=noise)
    grids = policy.processor.image_processor.resize(
        [observation[key] for key in image_keys], policy.processor.resolutions
    )["image_grid_thw"]
    merge = policy.processor.image_processor.merge_size
    actions = np.ascontiguousarray(first["normalized_actions"], np.float32)
    if args.dump_actions:
        np.save(args.dump_actions, actions)

    for _ in range(args.warmup):
        policy.infer(observation, noise=noise)
    latencies = []
    with _cuda_profiler_capture():
        for _ in range(args.iterations):
            started = time.perf_counter()
            policy.infer(observation, noise=noise)
            latencies.append((time.perf_counter() - started) * 1e3)

    report = {
        "commit": git_commit(),
        "model_dir": str(args.model_dir),
        "precision": args.precision,
        "raw_image": [args.views, args.height, args.width],
        "image_grid_thw": np.asarray(grids).tolist(),
        "vision_tokens": int(sum(t * h * w for t, h, w in grids) // (merge * merge)),
        "token_count": int(first["token_ids"].size),
        "normalized_actions_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
        "policy_ms": stats(latencies),
    }
    # Check the effective runtime mode, not just the requested CLI flag.
    # Profiling exposes graph_ms only when a captured graph was replayed.
    # Run this outside the latency samples even when --profile was omitted.
    rgb, token_ids, action_mask = policy.processor(observation)
    views, height, width = rgb.shape[:3]
    patch = int(policy.model_runner.patch_size)
    grid = np.full((views, 3), (1, height // patch, width // patch), np.uint32)
    samples = []
    for _ in range(args.iterations if args.profile else 1):
        _, phases = policy.model_runner.infer_rgb_profiled(
            rgb, "nhwc", token_ids, noise=noise, action_mask=action_mask,
            image_grid_thw=grid,
        )
        samples.append(phases)
    actual_mode = "graph" if "graph_ms" in samples[0] else "eager"
    expected_mode = "eager" if args.eager else "graph"
    if actual_mode != expected_mode:
        raise RuntimeError(
            f"requested {expected_mode} benchmark but runtime used {actual_mode}; "
            "check APXINF_WALLOSS_NO_GRAPH and graph-capture diagnostics"
        )
    report["mode"] = actual_mode
    if args.profile:
        report["profile_ms"] = {
            key: stats([sample[key] for sample in samples]) for key in sorted(samples[0])
        }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
