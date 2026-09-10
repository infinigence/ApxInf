#!/usr/bin/env python3
"""Compare ApxInf GR00T output JSON with the NVIDIA reference NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--apxinf", required=True, type=Path)
    parser.add_argument(
        "--max-abs",
        type=float,
        default=None,
        help="Optional diagnostic gate; disabled by default because Pi0.5 does not gate it.",
    )
    parser.add_argument(
        "--mean-abs",
        type=float,
        default=None,
        help="Optional diagnostic gate; disabled by default because Pi0.5 does not gate it.",
    )
    parser.add_argument("--min-cosine", type=float, default=0.997)
    parser.add_argument("--max-relative-l2", type=float, default=0.10)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reference.suffix == ".npz":
        with np.load(args.reference, allow_pickle=False) as reference:
            expected = np.asarray(reference["final_action"], dtype=np.float32)
            metadata = json.loads(str(reference["metadata_json"]))
    elif args.reference.suffix == ".json":
        reference = json.loads(args.reference.read_text())
        expected = np.asarray(reference["output"], dtype=np.float32).reshape(
            reference["output_shape"]
        )
        metadata = reference
    else:
        raise RuntimeError(
            f"unsupported reference format {args.reference.suffix!r}; expected .npz or .json"
        )
    actual_document = json.loads(args.apxinf.read_text())
    if actual_document.get("fixture") != metadata.get("fixture"):
        raise RuntimeError(
            "fixture mismatch: "
            f"reference={metadata.get('fixture')!r}, "
            f"apxinf={actual_document.get('fixture')!r}"
        )
    actual = np.asarray(actual_document["output"], dtype=np.float32).reshape(
        actual_document["output_shape"]
    )
    if actual.shape != expected.shape:
        raise RuntimeError(
            f"shape mismatch: reference={expected.shape}, apxinf={actual.shape}"
        )
    if actual.shape != (1, 40, 132):
        raise RuntimeError(f"unexpected GR00T N1.7 action shape {actual.shape}")

    finite = np.isfinite(actual) & np.isfinite(expected)
    absolute = np.abs(actual - expected)
    denominator = np.maximum(np.abs(expected), 1.0e-6)
    relative = absolute / denominator
    expected_flat = expected.astype(np.float64, copy=False).reshape(-1)
    actual_flat = actual.astype(np.float64, copy=False).reshape(-1)
    expected_l2 = float(np.linalg.norm(expected_flat))
    actual_l2 = float(np.linalg.norm(actual_flat))
    error_l2 = float(np.linalg.norm(actual_flat - expected_flat))
    if expected_l2 == 0.0:
        relative_l2 = 0.0 if error_l2 == 0.0 else float("inf")
    else:
        relative_l2 = error_l2 / expected_l2
    if expected_l2 == 0.0 or actual_l2 == 0.0:
        cosine = 1.0 if expected_l2 == actual_l2 == 0.0 else 0.0
    else:
        cosine = float(np.dot(expected_flat, actual_flat) / (expected_l2 * actual_l2))
    report = {
        "schema": "apxinf.gr00t-n1.7.parity-report.v2",
        "shape": list(actual.shape),
        "finite": bool(finite.all()),
        "max_abs": float(absolute.max()),
        "mean_abs": float(absolute.mean()),
        "max_relative": float(relative.max()),
        "relative_l2": relative_l2,
        "cosine": cosine,
        "reference_sum": float(expected.sum(dtype=np.float64)),
        "apxinf_sum": float(actual.sum(dtype=np.float64)),
        "thresholds": {
            "max_abs": args.max_abs,
            "mean_abs": args.mean_abs,
            "min_cosine": args.min_cosine,
            "max_relative_l2": args.max_relative_l2,
        },
    }
    report["passed"] = bool(
        report["finite"]
        and (args.max_abs is None or report["max_abs"] <= args.max_abs)
        and (args.mean_abs is None or report["mean_abs"] <= args.mean_abs)
        and report["cosine"] >= args.min_cosine
        and report["relative_l2"] <= args.max_relative_l2
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(encoded)
        temporary.replace(args.output)
    print(encoded, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
