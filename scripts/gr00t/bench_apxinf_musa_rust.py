#!/usr/bin/env python3
"""Launch the MUSA GR00T Rust executable benchmark and optional parity gate."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from compare_outputs import GATES, compare_array, load_archive, noise_sha256
from formats import load_metadata, load_noise
from prompt_fixture import prepare_fixture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--prompt", help="replace fixture prompt before timing; save a derived fixture beside --output")
    parser.add_argument("--precision", choices=("bf16", "fp8", "int8"), required=True)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--execution",
        choices=("eager", "prefer-graph", "require-graph"),
        default="require-graph",
        help="prepared execution policy (default: require-graph)",
    )
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--tactics", type=Path)
    parser.add_argument(
        "--autotune",
        action="store_true",
        help="tune missing GEMM tactics and persist them at --tactics",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("devlocal/gr00t-n1d7/results/gr00t-musa-bench.json"),
    )
    parser.add_argument(
        "--binary",
        type=Path,
        help="prebuilt gr00t_bench_musa binary; otherwise cargo run --release is used",
    )
    parser.add_argument(
        "--reference", type=Path, help="reference .npy, NVIDIA .npz, or JSON output"
    )
    parser.add_argument(
        "--reference-noise",
        type=Path,
        help="noise .npy/.npz (for example the Orin inputs.npz)",
    )
    parser.add_argument(
        "--allow-missing-noise",
        action="store_true",
        help="explicitly accept references without a noise array",
    )
    parser.add_argument(
        "--expect-model",
        default="nvidia/GR00T-N1.7-LIBERO",
        help="expected model name when the reference JSON carries one",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup <= 0 or args.iterations <= 0:
        raise SystemExit("--warmup and --iterations must be positive")
    if args.output.suffix != ".json":
        raise ValueError("--output must end in .json")
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    executable = (
        [str(args.binary)]
        if args.binary is not None
        else [
            "cargo",
            "run",
            "--release",
            "-p",
            "apxinf-model",
            "--features",
            "musa",
            "--example",
            "gr00t_bench_musa",
            "--",
        ]
    )
    command = [
        *executable,
        str(args.checkpoint),
        str(args.backbone),
        str(args.fixture),
        args.precision,
        str(args.device),
        str(args.warmup),
        str(args.iterations),
        str(args.calibration) if args.calibration is not None else "-",
        str(args.tactics) if args.tactics is not None else "-",
        str(args.output),
        "--execution",
        args.execution,
    ]
    if args.autotune:
        if args.tactics is None:
            raise SystemExit("--autotune requires an explicit --tactics output path")
        command.append("--autotune")
    subprocess.run(command, check=True)
    report = json.loads(args.output.read_text())
    actual_archive = load_archive(
        args.output, expect_schema_prefix="apxinf.gr00t-n1.7.benchmark."
    )
    actual = actual_archive["normalized_actions"]
    actual_noise = actual_archive.get("noise")
    if actual_noise is not None and not np.array_equal(actual_noise, arrays["noise"]):
        raise ValueError("benchmark report noise differs from the fixture noise")
    if actual_noise is None and not args.allow_missing_noise:
        raise ValueError(
            "benchmark report has no noise; pass --allow-missing-noise to accept it"
        )
    report["model_hz"] = 1000.0 / report["latency_ms"]["p50"]
    report["input"].update(prompt=manifest.get("prompt"), fixture_path=str(args.fixture))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.reference is not None:
        metrics = compare_array(actual, reference["normalized_actions"])
        gate = GATES[args.precision]
        noise_checked = actual_noise is not None and reference_noise is not None
        noise_exact_match = (
            bool(np.array_equal(actual_noise, reference_noise))
            if noise_checked
            else None
        )
        failures = []
        if metrics["cosine"] < gate["minimum_cosine"]:
            failures.append(
                f"cosine={metrics['cosine']:.9f} < {gate['minimum_cosine']}"
            )
        if metrics["relative_l2"] > gate["maximum_relative_l2"]:
            failures.append(
                f"relative_l2={metrics['relative_l2']:.9f} "
                f"> {gate['maximum_relative_l2']}"
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
                "apxinf": noise_sha256(actual_noise) if actual_noise is not None else None,
                "reference": noise_sha256(reference_noise)
                if reference_noise is not None
                else None,
            },
            **metrics,
            "passed": not failures,
            "failures": failures,
        }
        report["parity"] = parity
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if failures:
            raise SystemExit("parity gate failed: " + "; ".join(failures))
    print(json.dumps(report, indent=2))


def load_reference(path: Path) -> np.ndarray:
    return load_archive(path)["normalized_actions"]


if __name__ == "__main__":
    main()
