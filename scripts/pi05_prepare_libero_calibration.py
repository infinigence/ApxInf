#!/usr/bin/env python3
"""Turn the zero-fixture PI0.5 calibration into a safe LIBERO smoke profile.

This profile is deliberately conservative and is only a bootstrap for collecting
real rollout fixtures. Behavioral evaluation and representative BF16 comparisons
must validate or replace it before it is treated as a production calibration.
"""

from __future__ import annotations

import argparse
import json
import pathlib


FP8_MAX = 448.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--margin", type=float, default=2.0)
    args = parser.parse_args()
    if args.margin < 1.0:
        raise ValueError("--margin must be >= 1")

    document = json.loads(args.input.read_text())
    scales = document["scales"]
    for name, entry in scales.items():
        amax = float(entry.get("amax", float(entry["scale"]) * FP8_MAX))
        amax *= args.margin
        if name == "vision.patch_input":
            amax = max(amax, 1.0)
        elif name == "action.input":
            # A 320-value N(0,1) diffusion seed normally reaches roughly 3-4.
            amax = max(amax, 5.0)
        entry["amax"] = amax
        entry["scale"] = max(amax / FP8_MAX, 1.0e-8)

    document["schema"] = "apxinf.pi05.fp8-calibration.v1"
    document["source"] = "conservative LIBERO bootstrap derived from BF16 zero fixture"
    document["fixture"] = (
        "bootstrap only: real image input floor, Gaussian-noise floor, and global margin"
    )
    document["bootstrap_margin"] = args.margin
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
