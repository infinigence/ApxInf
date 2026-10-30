#!/usr/bin/env python3
"""Compare two π0.5 stage dumps, e.g. CUDA (`pi05_stage_dump`) against HIP
(`pi05_hip_bench --dump-stages`), in pipeline order.

The first stage whose agreement drops sharply is where two backends diverge;
later stages inherit the difference. Metrics match `pi05_bench`:

    python scripts/compare_pi05_stages.py <reference-dir> <candidate-dir> [--json out.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ORDER = ["patches", "vision", "prefix"]


def stage_names(directory: Path) -> list[str]:
    names = {p.stem for p in directory.glob("*.npy")}
    layers = sorted(n for n in names if n[:2] in ("k_", "v_"))
    tail = [n for n in ("state_step00", "actions") if n in names]
    head = [n for n in ORDER if n in names]
    rest = sorted(names - set(head) - set(layers) - set(tail))
    return head + layers + rest + tail


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    a = actual.astype(np.float64).ravel()
    e = expected.astype(np.float64).ravel()
    error = a - e
    denominator = np.linalg.norm(a) * np.linalg.norm(e)
    return {
        "cosine": float(a @ e / denominator) if denominator else float(np.array_equal(a, e)),
        "relative_l2": float(np.linalg.norm(error) / np.linalg.norm(e)) if np.any(e) else float(np.any(error)) * np.inf,
        "max_abs": float(np.abs(error).max()),
        "mean_abs": float(np.abs(error).mean()),
        "bitwise_equal": bool(np.array_equal(actual, expected)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--json", type=Path, help="also write the table as JSON")
    args = parser.parse_args()

    rows = {}
    print(f"{'stage':<14} {'shape':<14} {'cosine':>12} {'rel_l2':>10} {'max_abs':>10} {'mean_abs':>10}")
    for name in stage_names(args.reference):
        candidate = args.candidate / f"{name}.npy"
        if not candidate.is_file():
            print(f"{name:<14} missing in candidate")
            continue
        reference = np.load(args.reference / f"{name}.npy")
        actual = np.load(candidate)
        if reference.shape != actual.shape:
            print(f"{name:<14} shape {actual.shape} != reference {reference.shape}")
            continue
        m = metrics(actual, reference)
        rows[name] = {"shape": list(reference.shape), **m}
        exact = "  (bit-exact)" if m["bitwise_equal"] else ""
        print(
            f"{name:<14} {str(list(reference.shape)):<14} {m['cosine']:>12.8f} {m['relative_l2']:>10.2e} "
            f"{m['max_abs']:>10.2e} {m['mean_abs']:>10.2e}{exact}"
        )
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
