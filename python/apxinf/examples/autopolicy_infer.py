#!/usr/bin/env python3
"""Minimal ``AutoPolicy`` example: dispatch a checkpoint by its config type.

Shows the *generic* entry point — ``AutoPolicy`` reads ``config.json``'s model
type and constructs the matching registered policy (``Pi05Policy`` today) without
your code naming the class. Use this when the model type is data-driven; use
``pi05policy_infer.py`` when you want model-specific knobs.

Requires the ``apxinf_py`` CUDA binding (``maturin develop`` of crates/apxinf-py)
and a checkpoint directory whose ``config.json`` names a registered model type.

    python examples/autopolicy_infer.py --model-dir /path/to/checkpoint
"""

from __future__ import annotations

import argparse
import pathlib

from _common import synthetic_observation  # noqa: E402 (path shim in _common)

from apxinf import AutoPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=pathlib.Path)
    parser.add_argument("--precision", choices=("auto", "fp8", "bf16", "int8"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--action-dim", type=int, default=7, help="0 keeps the full vector")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Generic construction: model type comes from config.json, not from code.
    policy = AutoPolicy.from_pretrained(
        args.model_dir,
        device=args.device,
        precision=args.precision,
        action_dim=(args.action_dim or None),
    )
    try:
        print("dispatched model_type:", policy.metadata.get("model_type"))

        observation = synthetic_observation(image_keys=policy.image_keys)
        result = policy.infer(observation)

        actions = result["actions"]
        print(f"actions: shape={actions.shape} dtype={actions.dtype}")
        print(f"timing:  {result['timing']}")
    finally:
        policy.close()


if __name__ == "__main__":
    main()
