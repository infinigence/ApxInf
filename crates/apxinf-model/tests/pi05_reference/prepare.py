#!/usr/bin/env python3
"""Pin existing PI0.5 NPZ requests for the CPU/GPU reference gate.

Inputs must contain images (NHWC uint8), token_ids and explicit noise (FP32).
Checkpoints remain in place. Outputs belong under devlocal/<feature>/.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", nargs="+", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--flow-steps", type=int, default=10)
    args = parser.parse_args()
    if args.flow_steps <= 0:
        parser.error("--flow-steps must be positive")
    rows, images, geometry = [], [], None
    for path in args.cases:
        with np.load(path, allow_pickle=False) as data:
            rgb, tokens, noise = (data[k] for k in ("images", "token_ids", "noise"))
            if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[1:] != (224, 224, 3):
                raise ValueError(f"{path}: expected [views,224,224,3] uint8 RGB")
            if (
                noise.dtype != np.float32
                or noise.ndim != 2
                or noise.shape[1] != 32
                or not np.isfinite(noise).all()
            ):
                raise ValueError(f"{path}: expected finite [horizon,32] float32 noise")
            if (
                tokens.ndim != 1
                or tokens.size == 0
                or tokens.dtype.kind not in "iu"
                or np.any(tokens < 0)
                or np.any(tokens >= 257152)
            ):
                raise ValueError(f"{path}: invalid token IDs")
            current = (rgb.shape[0], noise.shape[0])
            if geometry is not None and geometry != current:
                raise ValueError("all cases must have the same view count and horizon")
            geometry = current
            name = path.stem
            if any(row["name"] == name for row in rows):
                raise ValueError(f"duplicate case name: {name}")
            raw = rgb.tobytes(order="C")
            images.append((name + ".rgb", raw))
            rows.append(
                {
                    "name": name,
                    "images": name + ".rgb",
                    "images_sha256": hashlib.sha256(raw).hexdigest(),
                    "token_ids": tokens.tolist(),
                    "latent": noise.ravel().tolist(),
                }
            )
    manifest = {
        "checkpoint_sha256": sha256(args.checkpoint),
        "num_views": geometry[0],
        "action_horizon": geometry[1],
        "num_flow_steps": args.flow_steps,
        "cases": rows,
        "limits": {},
        "calibration_sha256": sha256(args.calibration) if args.calibration else None,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    for name, raw in images:
        (args.output / name).write_bytes(raw)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "sources.json").write_text(
        json.dumps({str(path.resolve()): sha256(path) for path in args.cases}, indent=2)
        + "\n"
    )
    print(args.output / "manifest.json")


if __name__ == "__main__":
    main()
