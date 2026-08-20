#!/usr/bin/env python3
"""Score PI0.5 FP8 calibration profiles against a fixed BF16 LIBERO fixture.

Runs in-process through the ``apxinf_py`` PyO3 binding's **L0** patches path: for
each calibration profile a model is loaded at FP8 with that profile's scales and
the shared tactics file, then every fixture's patches/tokens/noise are scored
against the preserved BF16 reference. This replaces the old stdio subprocess
server (whose ``patches`` image-input mode was the same L0 path); the numerics
are identical (the binding feeds f32 patches/noise that the runtime normalizes to
the same FP16 the stdio path used).

Internal-tooling note: L0 is exposed to Python only as the private
``Model._infer_patches`` (fixtures store pre-computed patches, so L1 ``infer_rgb``
cannot be used here). This is a first-party ``scripts/`` workflow that ships and
evolves with the binding, so depending on that private name is intentional — if
the L0 signature changes, this script changes with it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

import apxinf_py


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float | None]:
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    expected = np.asarray(expected, dtype=np.float64).reshape(-1)
    if actual.shape != expected.shape or actual.size == 0:
        raise ValueError(f"comparison shape mismatch: {actual.shape} versus {expected.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise FloatingPointError("comparison contains non-finite values")
    error = actual - expected
    actual_l2 = np.linalg.norm(actual)
    expected_l2 = np.linalg.norm(expected)
    cosine = (
        float(np.dot(actual, expected) / (actual_l2 * expected_l2))
        if actual_l2 and expected_l2
        else float(actual_l2 == expected_l2)
    )
    return {
        "cosine": cosine,
        "relative_l2": float(np.linalg.norm(error) / expected_l2) if expected_l2 else None,
        "max_abs": float(np.max(np.abs(error))),
        "mean_abs": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "actual_abs_checksum": float(np.abs(actual).sum()),
        "reference_abs_checksum": float(np.abs(expected).sum()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path)
    parser.add_argument("--tactics", required=True, type=pathlib.Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--fixture",
        action="append",
        default=[],
        type=pathlib.Path,
        help="BF16 fixture; repeat to score multiple fixtures",
    )
    parser.add_argument(
        "--fixture-dir",
        type=pathlib.Path,
        help="also score every *.npz fixture in this directory",
    )
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--fp8-output-dir", type=pathlib.Path)
    parser.add_argument("--min-cosine", type=float, default=0.997)
    parser.add_argument("--max-relative-l2", type=float, default=0.10)
    parser.add_argument("calibrations", nargs="+", type=pathlib.Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixture_paths = list(args.fixture)
    if args.fixture_dir is not None:
        fixture_paths.extend(sorted(args.fixture_dir.glob("*.npz")))
    fixture_paths = list(dict.fromkeys(fixture_paths))
    if not fixture_paths:
        raise ValueError("provide at least one --fixture or --fixture-dir")

    fixtures = []
    for path in fixture_paths:
        fixture = np.load(path, allow_pickle=False)
        patches = np.asarray(fixture["patches"], dtype=np.float16)
        tokens = np.asarray(fixture["tokens"], dtype=np.uint32)
        noise = np.asarray(fixture["noise"], dtype=np.float16)
        reference = np.asarray(fixture["normalized_actions"], dtype=np.float32)
        if patches.shape != (512, 588):
            raise ValueError(f"{path}: expected patches [512,588], got {patches.shape}")
        if noise.shape != (10, 32) or reference.shape != (10, 32):
            raise ValueError(
                f"{path}: expected noise/reference [10,32], "
                f"got {noise.shape}/{reference.shape}"
            )
        fixtures.append((path, patches, tokens, noise, reference))

    if args.fp8_output_dir is not None:
        args.fp8_output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, calibration in enumerate(args.calibrations):
        label = calibration.stem
        started = time.perf_counter()
        # Feed f32 patches/noise: the runtime normalizes them to the same FP16 the
        # old stdio server received, so scores match the legacy sweep.
        model = apxinf_py.Model.load(
            "pi05",
            str(args.checkpoint),
            device=args.device,
            precision="fp8",
            calibration=str(calibration),
            tactics=str(args.tactics),
        )
        comparisons = []
        try:
            for path, patches, tokens, noise, reference in fixtures:
                # L0 is intentionally private (Model._infer_patches); see the
                # module docstring for why this internal script depends on it.
                actual = np.asarray(
                    model._infer_patches(
                        patches.astype(np.float32), tokens, noise.astype(np.float32)
                    ),
                    dtype=np.float32,
                )
                measured = metrics(actual, reference)
                relative_l2 = measured["relative_l2"]
                passed = measured["cosine"] >= args.min_cosine and (
                    relative_l2 is not None and relative_l2 <= args.max_relative_l2
                )
                comparison = {
                    "fixture": str(path),
                    "passed": passed,
                    **measured,
                }
                comparisons.append(comparison)
                if args.fp8_output_dir is not None:
                    np.savez_compressed(
                        args.fp8_output_dir / f"{index:02d}_{label}_{path.name}",
                        patches=patches,
                        tokens=tokens,
                        noise=noise,
                        normalized_actions=actual,
                    )
                print(
                    f"{label}/{path.name}: cosine={measured['cosine']:.9f} "
                    f"relative_l2={relative_l2:.6f} passed={passed}",
                    flush=True,
                )
        finally:
            close = getattr(model, "close", None)
            if callable(close):
                close()
        relative_l2_values = [row["relative_l2"] for row in comparisons]
        row = {
            "calibration": str(calibration),
            "elapsed_seconds": time.perf_counter() - started,
            "passed": all(item["passed"] for item in comparisons),
            "minimum_cosine": min(item["cosine"] for item in comparisons),
            "maximum_relative_l2": max(
                value for value in relative_l2_values if value is not None
            ),
            "fixtures": comparisons,
        }
        results.append(row)
        print(
            f"{label}: minimum_cosine={row['minimum_cosine']:.9f} "
            f"maximum_relative_l2={row['maximum_relative_l2']:.6f} "
            f"passed={row['passed']}",
            flush=True,
        )

    document = {
        "schema": "apxinf.pi05.libero-calibration-sweep.v1",
        "fixtures": [str(path) for path in fixture_paths],
        "min_cosine": args.min_cosine,
        "max_relative_l2": args.max_relative_l2,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    if not all(row["passed"] for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
