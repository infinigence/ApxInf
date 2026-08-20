#!/usr/bin/env python3
"""Minimal ``Pi05Policy`` example: load a checkpoint, run one inference.

Shows the *concrete* entry point — use it when you want model-specific knobs
(``action_dim`` trim, ``image_keys``, ``discrete_state``, custom pipeline steps).
For config-driven dispatch that does not name the model class, see
``autopolicy_infer.py``.

Requires the ``apxinf_py`` CUDA binding (``maturin develop`` of crates/apxinf-py)
and a pi05 checkpoint directory (model + tokenizer + norm_stats.json).

    python examples/pi05policy_infer.py --model-dir /path/to/checkpoint
"""

from __future__ import annotations

import argparse
import pathlib

from _common import synthetic_observation  # noqa: E402 (path shim in _common)

from apxinf import Pi05Policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=pathlib.Path)
    parser.add_argument("--precision", choices=("auto", "fp8", "bf16", "int8"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--action-dim",
        type=int,
        default=7,
        help="deployable action width to trim to (LIBERO=7; 0 keeps the full vector)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Concrete construction: Pi05Policy knows pi05-specific knobs directly.
    policy = Pi05Policy.from_pretrained(
        args.model_dir,
        device=args.device,
        precision=args.precision,
        action_dim=(args.action_dim or None),
    )
    try:
        print("metadata:", policy.metadata)

        observation = synthetic_observation(image_keys=policy.image_keys)
        result = policy.infer(observation)

        actions = result["actions"]
        print(f"actions: shape={actions.shape} dtype={actions.dtype}")
        print(f"timing:  {result['timing']}")
        print(f"first action vector: {actions[0]}")
    finally:
        policy.close()


if __name__ == "__main__":
    main()
