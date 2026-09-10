#!/usr/bin/env python3
"""Load GR00T N1.7 and infer one action chunk from raw RGB/state/prompt."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from apxinf import Gr00tPolicy


def _load_state(path: Path):
    value = np.load(path, allow_pickle=True)
    if isinstance(value, np.ndarray) and value.shape == () and value.dtype == object:
        value = value.item()
    if not isinstance(value, (np.ndarray, Mapping)):
        raise ValueError(
            f"{path} must contain a NumPy state vector or a mapping of state fields"
        )
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--backbone", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--wrist-image", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--state", required=True, type=Path, help=".npy state vector or mapping")
    parser.add_argument(
        "--precision", choices=("bf16", "fp8", "int8"), default="bf16"
    )
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--tactics", type=Path)
    parser.add_argument(
        "--action-dim",
        type=int,
        help="Override the checkpoint processor's decoded action width.",
    )
    args = parser.parse_args()

    from PIL import Image

    policy = Gr00tPolicy.from_pretrained(
        args.model_dir,
        backbone=args.backbone,
        precision=args.precision,
        calibration=args.calibration,
        tactics=args.tactics,
        action_dim=args.action_dim,
        noise_mode="fixed",
    )
    try:
        result = policy.infer(
            {
                "observation/image": np.asarray(Image.open(args.image).convert("RGB")),
                "observation/wrist_image": np.asarray(
                    Image.open(args.wrist_image).convert("RGB")
                ),
                "observation/state": _load_state(args.state),
                "prompt": args.prompt,
            }
        )
        print("metadata:", result["metadata"])
        print("actions:", result["actions"].shape, result["actions"].dtype)
        print("timing:", result["timing"])
    finally:
        policy.close()


if __name__ == "__main__":
    main()
