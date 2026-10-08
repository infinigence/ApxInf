#!/usr/bin/env python3
"""Fixed-input native SmolVLA latency benchmark."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys
import time
from typing import Callable

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_APXINF_PKG = _REPO_ROOT / "python" / "apxinf"
if _APXINF_PKG.is_dir() and str(_APXINF_PKG) not in sys.path:
    sys.path.insert(0, str(_APXINF_PKG))

from apxinf.policies.impls.smolvla import SmolVlaPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=pathlib.Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--model-variant",
        choices=("bf16", "fp16"),
        default="bf16",
        help="compute stream: BF16, or FP16 tensor-core GEMMs with BF16 residual stream",
    )
    parser.add_argument("--prompt", default="pick up the black bowl on the stove")
    parser.add_argument("--num-views", type=int, choices=(2, 3), default=3)
    parser.add_argument("--image-size", type=int, choices=(256, 512), default=512)
    parser.add_argument(
        "--tactics",
        type=pathlib.Path,
        help="hardware GEMM tactics JSON; with --autotune, missing keys are appended here",
    )
    parser.add_argument(
        "--autotune",
        action="store_true",
        help="tune missing exact GEMM keys from the first real inference request",
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="report CUDA event timings for VLM prefix and action expert phases",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=_REPO_ROOT
        / "devlocal"
        / "smolvla-integration"
        / "results"
        / "smolvla-benchmark.json",
    )
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmup < 0:
        parser.error("--iterations must be positive and --warmup must be non-negative")
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
        "std": statistics.pstdev(ordered) if len(ordered) > 1 else 0.0,
    }


def profile_stats(samples: list[dict]) -> dict:
    keys = sorted(set().union(*(sample["phases"] for sample in samples)))
    return {
        key: stats([sample["phases"][key] for sample in samples]) for key in keys
    }


def git_commit() -> str:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], cwd=_REPO_ROOT, stderr=subprocess.DEVNULL
        )
        return revision + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def time_loop(callable: Callable[[], dict], warmup: int, iterations: int) -> list[dict]:
    for _ in range(warmup):
        callable()
    return [callable() for _ in range(iterations)]


def main() -> None:
    args = parse_args()
    image_keys = tuple(
        f"observation.images.camera{index + 1}" for index in range(args.num_views)
    )
    policy = SmolVlaPolicy.from_pretrained(
        args.model_dir,
        device=args.device,
        seed=args.seed,
        image_keys=image_keys,
        num_views=args.num_views,
        tactics=args.tactics,
        autotune=args.autotune,
        model_variant=args.model_variant,
        state_key="observation.state",
        prompt_key="task",
    )
    observation = {
        key: np.full(
            (args.image_size, args.image_size, 3),
            value,
            dtype=np.uint8,
        )
        for key, value in zip(
            image_keys,
            (127, 180, 210),
        )
    }
    observation.update(
        {
            "observation.state": np.zeros(8, dtype=np.float32),
            "task": args.prompt,
        }
    )

    def run_once() -> dict:
        started = time.perf_counter()
        result = policy.infer(observation, profile=args.profile)
        elapsed = (time.perf_counter() - started) * 1000.0
        timing = result["timing"]
        return {
            "model_ms": float(timing["model_ms"]),
            "total_ms": elapsed,
            "processor_ms": max(0.0, elapsed - float(timing["model_ms"])),
            "action_shape": list(result["actions"].shape),
            **({"phases": dict(timing["phases"])} if args.profile else {}),
        }

    samples = time_loop(run_once, args.warmup, args.iterations)
    report = {
        "schema": "apxinf.smolvla.latency.v1",
        "model_type": "smolvla_libero",
        "model_variant": args.model_variant,
        "device": args.device,
        "checkpoint": str(args.model_dir.resolve()),
        "git_commit": git_commit(),
        "prompt": args.prompt,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "num_views": args.num_views,
        "image_size": args.image_size,
        "tactics": str(args.tactics) if args.tactics is not None else None,
        "autotune": bool(args.autotune),
        "model_ms": stats([sample["model_ms"] for sample in samples]),
        "total_ms": stats([sample["total_ms"] for sample in samples]),
        "processor_ms": stats([sample["processor_ms"] for sample in samples]),
        **({"phase_ms": profile_stats(samples)} if args.profile else {}),
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"SmolVLA {args.model_variant.upper()}: {report['model_ms']['p50']:.1f} ms model p50, "
        f"{report['total_ms']['p50']:.1f} ms end-to-end p50"
    )
    print(f"report: {args.output}")
    if args.profile:
        for name, values in report["phase_ms"].items():
            print(f"{name}: {values['p50']:.1f} ms p50")
    policy.close()


if __name__ == "__main__":
    main()
