#!/usr/bin/env python3
"""Compare fixed-input ApxInf GR00T outputs against an explicit reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from formats import load_archive, load_metadata, load_noise


GATES = {
    "bf16": {"minimum_cosine": 0.999, "maximum_relative_l2": 0.05, "maximum_abs": 0.125},
    "fp8": {"minimum_cosine": 0.997, "maximum_relative_l2": 0.10, "maximum_abs": 0.125},
    "int8": {"minimum_cosine": 0.995, "maximum_relative_l2": 0.10, "maximum_abs": 0.125},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apxinf", type=Path, required=True)
    parser.add_argument("--reference", "--torch-eager", dest="reference", type=Path, required=True)
    parser.add_argument("--precision", choices=tuple(GATES), default="bf16")
    parser.add_argument("--reference-noise", type=Path)
    parser.add_argument("--allow-missing-noise", action="store_true")
    parser.add_argument("--expect-model", default="nvidia/GR00T-N1.7-LIBERO")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def compare_array(actual: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    if actual.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: ApxInf {actual.shape}, reference {reference.shape}"
        )
    if actual.size == 0:
        raise ValueError("comparison inputs must not be empty")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(reference)):
        raise ValueError("comparison inputs must contain only finite values")

    actual64 = np.asarray(actual, dtype=np.float64).reshape(-1)
    reference64 = np.asarray(reference, dtype=np.float64).reshape(-1)
    difference = actual64 - reference64
    denominator = np.linalg.norm(actual64) * np.linalg.norm(reference64)
    reference_norm = np.linalg.norm(reference64)
    if reference_norm == 0 and np.any(difference):
        raise ValueError("nonzero output does not match a zero reference")
    if not np.isfinite(denominator) or not np.isfinite(reference_norm):
        raise ValueError("comparison norms must be finite")
    return {
        "shape": list(actual.shape),
        "cosine": float(np.dot(actual64, reference64) / denominator)
        if denominator
        else (1.0 if not np.any(difference) else 0.0),
        "relative_l2": float(np.linalg.norm(difference) / reference_norm)
        if reference_norm
        else 0.0,
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
    }


def noise_sha256(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values, dtype="<f4")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def main() -> int:
    args = parse_args()
    apxinf = load_archive(
        args.apxinf, expect_schema_prefix="apxinf.gr00t-n1.7.benchmark."
    )
    reference = load_archive(args.reference, expect_model=args.expect_model)
    reference_metadata = load_metadata(args.reference)

    failures: list[str] = []
    actual_noise = apxinf.get("noise")
    reference_noise = reference.get("noise")
    if args.reference_noise is not None:
        external = load_noise(args.reference_noise)
        if reference_noise is not None and not np.array_equal(reference_noise, external):
            failures.append("reference noise differs between the archive and --reference-noise")
        reference_noise = external
    noise_checked = actual_noise is not None and reference_noise is not None
    noise_exact_match = None
    if noise_checked:
        noise_exact_match = bool(np.array_equal(actual_noise, reference_noise))
        if not noise_exact_match:
            failures.append("ApxInf and reference noise differ")
    elif not args.allow_missing_noise:
        failures.append("noise is required; pass --reference-noise or --allow-missing-noise")

    gate = GATES[args.precision]
    keys = ["normalized_actions"]
    if "actions" in apxinf and "actions" in reference:
        keys.append("actions")
    comparisons = {key: compare_array(apxinf[key], reference[key]) for key in keys}
    for key, result in comparisons.items():
        if result["cosine"] < gate["minimum_cosine"]:
            failures.append(
                f"{key} cosine={result['cosine']:.9f} < {gate['minimum_cosine']}"
            )
        if result["relative_l2"] > gate["maximum_relative_l2"]:
            failures.append(
                f"{key} relative_l2={result['relative_l2']:.9f} "
                f"> {gate['maximum_relative_l2']}"
            )
        if result["max_abs"] > gate["maximum_abs"]:
            failures.append(
                f"{key} max_abs={result['max_abs']:.9f} > {gate['maximum_abs']}"
            )

    report = {
        "passed": not failures,
        "precision": args.precision,
        "apxinf": str(args.apxinf.expanduser().resolve()),
        "reference": str(args.reference.expanduser().resolve()),
        "reference_model": reference_metadata.get("model"),
        "noise_checked": noise_checked,
        "noise_exact_match": noise_exact_match,
        "noise_sha256": {
            "apxinf": noise_sha256(actual_noise) if actual_noise is not None else None,
            "reference": noise_sha256(reference_noise) if reference_noise is not None else None,
        },
        "gate": gate,
        "comparisons": comparisons,
        "failures": failures,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    if failures:
        raise SystemExit("correctness gate failed: " + "; ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
