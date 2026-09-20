#!/usr/bin/env python3
"""Validate a PI0.5 production calibration profile on target hardware.

The runner stays on public seams: Observation NPZs enter ``Pi05Policy``, the
manifest is checked against its published plan, and BF16/FP8 policies are loaded
normally.  It emits the raw business actions as well as aggregate accuracy and
steady-state timing evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import pathlib
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_APXINF_PKG = _REPO_ROOT / "python" / "apxinf"
if _APXINF_PKG.is_dir() and str(_APXINF_PKG) not in sys.path:
    sys.path.insert(0, str(_APXINF_PKG))

SCHEMA = "apxinf.pi05.fp8-calibration-validation.v1"
CALIBRATION_SCHEMA = "apxinf.pi05.fp8-calibration.v1"
DEFAULT_IMAGE_KEYS = ("observation/image", "observation/wrist_image")


def _canonical_without_environment(document: Mapping[str, Any]) -> str:
    comparable = dict(document)
    comparable.pop("device", None)
    return json.dumps(comparable, sort_keys=True, separators=(",", ":"))


def compare_manifests(manifests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare repeated profiles, excluding only declared environment metadata."""
    if len(manifests) < 2:
        raise ValueError("reproducibility requires at least two calibration manifests")
    canonical = [_canonical_without_environment(document) for document in manifests]
    hashes = [hashlib.sha256(value.encode()).hexdigest() for value in canonical]
    return {
        "runs": len(manifests),
        "equivalent": len(set(canonical)) == 1,
        "content_sha256_without_device": hashes,
        "ignored_fields": ["device"],
    }


def validate_manifest(document: Mapping[str, Any]) -> dict[str, Any]:
    """Require an exact required/observed/consumed static-scale site set."""
    if document.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError(f"unexpected calibration schema: {document.get('schema')!r}")
    calibration_data = document.get("calibration_data", {})
    if not calibration_data.get("production", False):
        raise ValueError("validation requires a representative production calibration")
    if int(calibration_data.get("sample_count", 0)) < 1:
        raise ValueError("calibration manifest has no samples")

    required = set(document.get("plan", {}).get("sites", ()))
    observed = set(document.get("observed_sites", ()))
    generated = set(document.get("scales", {}))
    missing = sorted(required - observed)
    unknown = sorted(observed - required)
    unused = sorted(generated - required)
    missing_scales = sorted(required - generated)
    if missing or unknown or unused or missing_scales:
        raise ValueError(
            "calibration site coverage mismatch: "
            f"missing={missing}, unknown={unknown}, unused={unused}, "
            f"missing_scales={missing_scales}"
        )
    return {
        "required": len(required),
        "observed": len(observed),
        "generated_scales": len(generated),
        "missing": missing,
        "unknown": unknown,
        "unused": unused,
        "missing_scales": missing_scales,
        "complete": True,
    }


def summarize_errors(
    reference: Sequence[np.ndarray], candidate: Sequence[np.ndarray]
) -> dict[str, Any]:
    if len(reference) != len(candidate) or not reference:
        raise ValueError("accuracy comparison requires equal non-empty output sets")
    lhs = np.concatenate([np.asarray(value, dtype=np.float64).ravel() for value in reference])
    rhs = np.concatenate([np.asarray(value, dtype=np.float64).ravel() for value in candidate])
    if lhs.shape != rhs.shape:
        raise ValueError(f"business output shapes differ: {lhs.shape} vs {rhs.shape}")
    finite = np.isfinite(lhs) & np.isfinite(rhs)
    non_finite = int(finite.size - np.count_nonzero(finite))
    if not np.any(finite):
        return {
            "elements": int(lhs.size),
            "non_finite": non_finite,
            "max_abs": float("inf"),
            "mean_abs": float("inf"),
            "rmse": float("inf"),
            "relative_l2": float("inf"),
        }
    delta = rhs[finite] - lhs[finite]
    reference_norm = float(np.linalg.norm(lhs[finite]))
    return {
        "elements": int(lhs.size),
        "non_finite": non_finite,
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "relative_l2": float(
            np.linalg.norm(delta) / max(reference_norm, np.finfo(np.float64).tiny)
        ),
    }


def accuracy_gate(metrics: Mapping[str, Any], max_relative_l2: float) -> bool:
    return (
        int(metrics["non_finite"]) == 0
        and np.isfinite(float(metrics["relative_l2"]))
        and float(metrics["relative_l2"]) <= max_relative_l2
    )


def load_observation(
    path: pathlib.Path,
    *,
    image_keys: Sequence[str],
    prompt_key: str,
    state_key: str,
    require_state: bool,
) -> dict[str, Any]:
    """Load only public business fields; internal tensors are never accepted."""
    with np.load(path, allow_pickle=False) as data:
        required = [*image_keys, prompt_key]
        if require_state:
            required.append(state_key)
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"{path}: missing Observation field(s): {missing}")
        observation: dict[str, Any] = {
            key: np.asarray(data[key]).copy() for key in image_keys
        }
        prompt = np.asarray(data[prompt_key])
        if prompt.ndim != 0:
            raise ValueError(f"{path}: {prompt_key} must be a scalar string")
        observation[prompt_key] = str(prompt.item())
        if require_state:
            observation[state_key] = np.asarray(data[state_key], dtype=np.float32).copy()
    return observation


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stats(samples: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in samples)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = int(np.floor(position))
        upper = int(np.ceil(position))
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "samples": len(ordered),
        "min": ordered[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "std": statistics.pstdev(ordered) if len(ordered) > 1 else 0.0,
    }


def _noise(seed: int, index: int, shape: tuple[int, int]) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    return np.ascontiguousarray(rng.standard_normal(shape), dtype=np.float32)


def _run_precision(
    *,
    precision: str,
    model_dir: pathlib.Path,
    calibration: pathlib.Path,
    observations: Sequence[Mapping[str, Any]],
    image_keys: Sequence[str],
    prompt_key: str,
    state_key: str,
    discrete_state: bool,
    state_norm_key: str,
    action_dim: Optional[int],
    action_horizon: Optional[int],
    seed: int,
    device: str,
    tactics: Optional[pathlib.Path],
    warmup: int,
    samples: int,
) -> dict[str, Any]:
    from apxinf import Pi05Policy

    options: dict[str, Any] = {
        "device": device,
        "model_variant": {"bf16": "bf16", "fp8": "fp8_static"}[precision],
        "seed": seed,
        "image_keys": tuple(image_keys),
        "num_views": len(image_keys),
        "prompt_key": prompt_key,
        "state_key": state_key,
        "discrete_state": discrete_state,
        "state_norm_key": state_norm_key,
    }
    if precision == "fp8":
        options["calibration"] = calibration
    if action_dim is not None:
        options["action_dim"] = action_dim
    if action_horizon is not None:
        options["action_horizon"] = action_horizon
    if tactics is not None:
        options["tactics"] = tactics

    load_started = time.perf_counter()
    policy = Pi05Policy.from_pretrained(model_dir, **options)
    load_seconds = time.perf_counter() - load_started
    try:
        shape = (int(policy.model_runner.action_horizon), int(policy.model_runner.action_dim))
        noises = [_noise(seed, index, shape) for index in range(len(observations))]
        raw_outputs = [
            policy.infer(observation, noise=noise)
            for observation, noise in zip(observations, noises)
        ]

        for index in range(warmup):
            sample_index = index % len(observations)
            policy.infer(observations[sample_index], noise=noises[sample_index])

        wall_ms: list[float] = []
        model_ms: list[float] = []
        for index in range(samples):
            sample_index = index % len(observations)
            started = time.perf_counter()
            result = policy.infer(observations[sample_index], noise=noises[sample_index])
            wall_ms.append((time.perf_counter() - started) * 1000.0)
            model_ms.append(float(result["timing"]["model_ms"]))
        return {
            "load_seconds": load_seconds,
            "actions": [np.asarray(result["actions"]).tolist() for result in raw_outputs],
            "normalized_actions": [
                np.asarray(result["normalized_actions"]).tolist() for result in raw_outputs
            ],
            "timing_ms": {
                "policy_wall": _stats(wall_ms),
                "model": _stats(model_ms),
                "raw_policy_wall": wall_ms,
                "raw_model": model_ms,
            },
        }
    finally:
        policy.close()


def _run_precision_from_options(options: Mapping[str, Any]) -> dict[str, Any]:
    return _run_precision(**options)


def _run_precision_isolated(**options: Any) -> dict[str, Any]:
    """Use a fresh process because the CUDA tactic store is process-global."""
    context = multiprocessing.get_context("spawn")
    with context.Pool(processes=1) as pool:
        return pool.apply(_run_precision_from_options, (options,))


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _command_output(command: Sequence[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=pathlib.Path)
    parser.add_argument(
        "--profile",
        required=True,
        action="append",
        type=pathlib.Path,
        help="repeat for independently generated profiles; the first is loaded by FP8",
    )
    parser.add_argument("--input", required=True, action="append", type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--max-relative-l2", required=True, type=float)
    parser.add_argument(
        "--validator-revision",
        help="validator source commit/version (required outside a Git checkout)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-key", action="append", default=[])
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--state-key", default="observation/state")
    parser.add_argument("--discrete-state", action="store_true")
    parser.add_argument("--state-norm-key", default="state")
    parser.add_argument("--action-dim", type=int)
    parser.add_argument("--action-horizon", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tactics", type=pathlib.Path)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    return parser.parse_args(argv)


def _validate_args(args) -> None:
    paths = [args.model_dir, *args.profile, *args.input]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise ValueError(f"input path does not exist: {missing[0]}")
    if len(args.profile) < 2:
        raise ValueError("pass at least two independently generated --profile files")
    if not np.isfinite(args.max_relative_l2) or args.max_relative_l2 < 0.0:
        raise ValueError("--max-relative-l2 must be finite and non-negative")
    if args.seed < 0 or args.warmup < 0 or args.samples < 1:
        raise ValueError("seed/warmup must be non-negative and samples must be positive")
    if args.validator_revision is not None and not args.validator_revision.strip():
        raise ValueError("--validator-revision must not be empty")


def main(argv=None) -> int:
    args = parse_args(argv)
    _validate_args(args)
    image_keys = tuple(args.image_key or DEFAULT_IMAGE_KEYS)
    manifests = [json.loads(path.read_text()) for path in args.profile]
    coverage = validate_manifest(manifests[0])
    reproducibility = compare_manifests(manifests)
    if not reproducibility["equivalent"]:
        raise ValueError("repeated calibration manifests are not reproducible")

    observations = [
        load_observation(
            path,
            image_keys=image_keys,
            prompt_key=args.prompt_key,
            state_key=args.state_key,
            require_state=args.discrete_state,
        )
        for path in args.input
    ]
    common = {
        "model_dir": args.model_dir,
        "calibration": args.profile[0],
        "observations": observations,
        "image_keys": image_keys,
        "prompt_key": args.prompt_key,
        "state_key": args.state_key,
        "discrete_state": args.discrete_state,
        "state_norm_key": args.state_norm_key,
        "action_dim": args.action_dim,
        "action_horizon": args.action_horizon,
        "seed": args.seed,
        "device": args.device,
        "tactics": args.tactics,
        "warmup": args.warmup,
        "samples": args.samples,
    }
    bf16 = _run_precision_isolated(precision="bf16", **common)
    fp8 = _run_precision_isolated(precision="fp8", **common)
    bf16_actions = [np.asarray(value, dtype=np.float32) for value in bf16["actions"]]
    fp8_actions = [np.asarray(value, dtype=np.float32) for value in fp8["actions"]]
    accuracy = summarize_errors(bf16_actions, fp8_actions)
    passed = accuracy_gate(accuracy, args.max_relative_l2)

    report = {
        "schema": SCHEMA,
        "calibration_source_revision": manifests[0].get("source_revision", "unknown"),
        "validator_source_revision": args.validator_revision or _git_revision(),
        "environment": {
            "host": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "device": args.device,
            "gpu": _command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,compute_cap,driver_version",
                    "--format=csv,noheader",
                ]
            ),
            "cuda_toolkit": _command_output(["nvcc", "--version"]),
        },
        "inputs": {
            "model_dir": str(args.model_dir),
            "profiles": [
                {"path": str(path), "sha256": _sha256(path)} for path in args.profile
            ],
            "observations": [
                {"path": str(path), "sha256": _sha256(path)} for path in args.input
            ],
            "seed": args.seed,
            "image_keys": list(image_keys),
            "prompt_key": args.prompt_key,
            "state_key": args.state_key,
            "discrete_state": args.discrete_state,
        },
        "coverage": {
            **coverage,
            "fp8_runtime_accepted": bool(coverage["complete"]),
        },
        "reproducibility": reproducibility,
        "accuracy": {
            **accuracy,
            "acceptance": {
                "metric": "relative_l2",
                "maximum": args.max_relative_l2,
                "passed": passed,
            },
        },
        "benchmark_protocol": {
            "warmup": args.warmup,
            "samples": args.samples,
            "aligned_inputs": True,
            "aligned_explicit_noise": True,
            "timing_boundary": "Pi05Policy.infer wall time and its model subspan",
            "synchronization": (
                "infer returns host action arrays after the CUDA device-to-host copy"
            ),
            "collector_enabled": False,
        },
        "bf16": bf16,
        "fp8": fp8,
        "passed": passed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"wrote {args.out}: relative_l2={accuracy['relative_l2']:.6g} "
        f"(maximum {args.max_relative_l2:.6g}) {'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
