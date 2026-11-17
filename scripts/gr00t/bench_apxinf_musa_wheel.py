#!/usr/bin/env python3
"""Benchmark GR00T through the installed MUSA Python wheel, without a subprocess."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from compare_outputs import GATES, compare_array, load_archive, noise_sha256
from formats import fixture_bf16_bits, load_metadata, load_noise
from prompt_fixture import prepare_fixture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--prompt", help="replace fixture prompt before timing; save a derived fixture beside --output")
    parser.add_argument("--precision", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--execution", choices=("eager", "prefer-graph", "require-graph"),
                        default="require-graph")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--tactics", type=Path)
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--reference", type=Path, help="normalized-action NPY/NPZ or benchmark JSON")
    parser.add_argument("--reference-noise", type=Path)
    parser.add_argument("--allow-missing-noise", action="store_true")
    parser.add_argument("--expect-model", default="nvidia/GR00T-N1.7-LIBERO")
    parser.add_argument("--output", type=Path,
                        default=Path("devlocal/gr00t-n1d7/results/gr00t-musa-wheel.json"))
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations <= 0:
        parser.error("--warmup must be at least 1 and --iterations must be positive")
    if args.autotune and args.tactics is None:
        parser.error("--autotune requires --tactics")
    if args.precision == "fp8" and args.calibration is None:
        parser.error("--precision fp8 requires --calibration")
    if args.output.suffix != ".json":
        parser.error("--output must end in .json")
    return args


def main() -> None:
    args = parse_args()
    args.fixture, manifest, arrays = prepare_fixture(args)
    reference = (
        load_archive(args.reference, expect_model=args.expect_model)
        if args.reference is not None
        else None
    )
    reference_metadata = load_metadata(args.reference) if args.reference is not None else {}
    reference_noise = reference.get("noise") if reference is not None else None
    if args.reference_noise is not None:
        external = load_noise(args.reference_noise)
        if reference_noise is not None and not np.array_equal(reference_noise, external):
            raise ValueError("reference noise differs from --reference-noise")
        reference_noise = external
    if reference_noise is not None and not np.array_equal(arrays["noise"], reference_noise):
        raise ValueError("fixture and reference noise differ")
    if (
        args.reference is not None
        and reference_noise is None
        and not args.allow_missing_noise
    ):
        raise ValueError(
            "reference has no noise; pass --reference-noise or --allow-missing-noise"
        )
    import apxinf_py

    started = time.perf_counter()
    model = apxinf_py.ModelRunner.load(
        "gr00t", str(args.checkpoint), device=f"musa:{args.device}",
        precision=args.precision, assets={"backbone": str(args.backbone)},
        calibration=str(args.calibration) if args.calibration else None,
        tactics=str(args.tactics) if args.tactics else None,
        autotune=args.autotune, execution=args.execution,
    )
    load_ms = (time.perf_counter() - started) * 1000
    infer_timed = getattr(model, "_infer_preprocessed_timed", None)
    if not callable(infer_timed):
        raise RuntimeError("rebuild the wheel: _infer_preprocessed_timed is required for model timing")
    inputs = (
        fixture_bf16_bits(arrays["pixel_values"]), arrays["image_grid_thw"], arrays["token_ids"],
        arrays["attention_mask"], fixture_bf16_bits(arrays["state"]), manifest["embodiment_id"],
        fixture_bf16_bits(arrays["noise"]),
    )
    for _ in range(args.warmup):
        infer_timed(*inputs)

    samples = []
    first_output = None
    repeat_max_abs = 0.0
    deterministic = True
    for iteration in range(args.iterations):
        started = time.perf_counter()
        output, elapsed_ms = infer_timed(*inputs)
        api_ms = (time.perf_counter() - started) * 1000
        if not np.isfinite(elapsed_ms) or elapsed_ms <= 0:
            raise ValueError("model returned invalid internal timing")
        if not np.all(np.isfinite(output)):
            raise ValueError("model output contains non-finite values")
        if first_output is None:
            first_output = output.copy()
        else:
            deterministic = deterministic and np.array_equal(output, first_output)
            repeat_max_abs = max(repeat_max_abs, float(np.max(np.abs(output - first_output))))
        samples.append({"iteration": iteration, "fixture": manifest["fixture"],
                        "token_count": arrays["token_ids"].size, "latency_ms": elapsed_ms,
                        "api_latency_ms": api_ms})
    execution = model.execution_mode
    allowed = {"eager": {"no_graph"}, "prefer-graph": {"no_graph", "musa-graph"},
               "require-graph": {"musa-graph"}}
    if execution not in allowed[args.execution]:
        raise RuntimeError(f"requested {args.execution}, runtime reported {execution}")
    latencies = np.asarray([sample["latency_ms"] for sample in samples])
    api_latencies = np.asarray([sample["api_latency_ms"] for sample in samples])
    report = {
        "schema": "apxinf.gr00t-n1.7.benchmark.v4",
        "frontend": "python-wheel",
        "extension": apxinf_py.__file__,
        "checkpoint": str(args.checkpoint), "backbone": str(args.backbone),
        "fixture": manifest["fixture"], "device": args.device, "precision": args.precision,
        "requested_execution": args.execution, "execution": execution,
        "timing_boundary": "Rust infer_host_f32 through synchronized model core and action D2H; excludes NumPy/PyO3 conversion",
        "api_timing_boundary": "Python call including NumPy/PyO3 input conversion, model core, action D2H and NumPy output",
        "warmup": args.warmup, "iterations": args.iterations, "load_ms": load_ms,
        "latency_ms": {
            "min": float(latencies.min()), "mean": float(latencies.mean()),
            "p50": float(np.median(latencies)), "p95": float(np.percentile(latencies, 95)),
            "max": float(latencies.max()),
        },
        "model_hz": 1000.0 / float(np.median(latencies)),
        "api_latency_ms": {
            "min": float(api_latencies.min()), "mean": float(api_latencies.mean()),
            "p50": float(np.median(api_latencies)), "p95": float(np.percentile(api_latencies, 95)),
            "max": float(api_latencies.max()),
        },
        "samples_detail": samples, "deterministic": deterministic,
        "repeat_normalized_max_abs": repeat_max_abs,
        "model": "gr00t",
        "noise": {
            "shape": list(arrays["noise"].shape),
            "values": arrays["noise"].reshape(-1).tolist(),
        },
        "input": {
            "float_tensor_dtype": "bfloat16",
            "prompt": manifest.get("prompt"), "fixture_path": str(args.fixture),
            "pixel_values": list(arrays["pixel_values"].shape),
            "image_grid_thw": arrays["image_grid_thw"].tolist(),
            "token_count": arrays["token_ids"].size,
            "state": list(arrays["state"].shape),
            "embodiment_id": manifest["embodiment_id"],
        },
        "output": {"shape": list(first_output.shape),
                   "values": first_output.reshape(-1).tolist()},
    }
    passed = deterministic
    if args.reference:
        metrics = compare_array(first_output, reference["normalized_actions"])
        gate = GATES[args.precision]
        noise_checked = reference_noise is not None
        noise_exact_match = (
            bool(np.array_equal(arrays["noise"], reference_noise))
            if noise_checked
            else None
        )
        failures = []
        if metrics["cosine"] < gate["minimum_cosine"]:
            failures.append(f"cosine={metrics['cosine']:.9f} < {gate['minimum_cosine']}")
        if metrics["relative_l2"] > gate["maximum_relative_l2"]:
            failures.append(
                f"relative_l2={metrics['relative_l2']:.9f} > {gate['maximum_relative_l2']}"
            )
        if metrics["max_abs"] > gate["maximum_abs"]:
            failures.append(
                f"max_abs={metrics['max_abs']:.9f} > {gate['maximum_abs']}"
            )
        if noise_checked and not noise_exact_match:
            failures.append("ApxInf and reference noise differ")
        if not noise_checked and not args.allow_missing_noise:
            failures.append(
                "noise is required; pass --reference-noise or --allow-missing-noise"
            )
        parity = {
            "reference": str(args.reference),
            "reference_model": reference_metadata.get("model"),
            "noise_checked": noise_checked,
            "noise_exact_match": noise_exact_match,
            "noise_sha256": {
                "apxinf": noise_sha256(arrays["noise"]),
                "reference": noise_sha256(reference_noise)
                if reference_noise is not None
                else None,
            },
            **metrics,
            "passed": not failures,
            "failures": failures,
        }
        report["parity"] = parity
        passed = passed and not failures
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"latency_ms": report["latency_ms"], "model_hz": report["model_hz"],
                      "api_latency_ms": report["api_latency_ms"], "execution": execution,
                      "parity": report.get("parity"), "deterministic": deterministic}, indent=2))
    if not passed:
        raise SystemExit("correctness or repeatability gate failed")


if __name__ == "__main__":
    main()
