#!/usr/bin/env python3
"""Minimal ``AutoPolicy`` example: dispatch a checkpoint by its config type.

Shows the *generic* entry point — ``AutoPolicy`` reads ``config.json``'s model
type and constructs the matching registered policy without your code naming the
class. Model-specific constructor options can be passed as one JSON object.

Requires the ``apxinf_py`` CUDA binding (``maturin develop`` of crates/apxinf-py)
and a checkpoint directory whose ``config.json`` names a registered model type.

    python examples/autopolicy_infer.py --model-dir /path/to/checkpoint \
        --policy-options '{"norm_key":"x2_normal"}'
"""

from __future__ import annotations

import argparse
import pathlib

from _common import json_object, policy_kwargs, synthetic_observation  # noqa: E402

from apxinf import AutoPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=pathlib.Path)
    parser.add_argument("--tokenizer", type=pathlib.Path)
    parser.add_argument("--precision", choices=("auto", "fp8", "bf16", "int8"), default=None)
    parser.add_argument("--model-variant", choices=("auto", "bf16", "fp8_static", "int8_dynamic"), default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--action-dim",
        type=int,
        default=0,
        help="deployable action width; 0 keeps the checkpoint's full vector",
    )
    parser.add_argument(
        "--policy-options",
        type=json_object,
        default={},
        metavar="JSON",
        help="extra concrete-policy options as a JSON object",
    )
    return parser.parse_args()


def synthetic_observation_for(policy, *, prompt: str = "pick up the block"):
    """Build a smoke input using only the public :class:`Policy` contract."""
    try:
        image_keys = policy.metadata["image_keys"]
    except (AttributeError, KeyError) as error:
        raise ValueError("policy metadata must declare image_keys") from error
    if (
        not isinstance(image_keys, (list, tuple))
        or not image_keys
        or not all(isinstance(key, str) and key for key in image_keys)
    ):
        raise ValueError(f"policy metadata image_keys must be non-empty strings, got {image_keys!r}")
    state_key = policy.metadata.get("state_key")
    state_dim = policy.metadata.get("state_dim")
    if state_key is not None and not isinstance(state_dim, int):
        raise ValueError("policy metadata must declare state_dim when state_key is set")
    return synthetic_observation(
        image_keys=tuple(image_keys),
        state_key=state_key,
        prompt_key=policy.metadata.get("prompt_key", "prompt"),
        state_dim=state_dim or 0,
        prompt=prompt,
    )


def main() -> None:
    args = parse_args()

    # Generic construction: model type comes from config.json, not from code.
    options = policy_kwargs(
        args.policy_options,
        device=args.device,
        precision=args.precision,
        model_variant=getattr(args, "model_variant", None),
        action_dim=args.action_dim,
    )
    if getattr(args, "tokenizer", None) is not None:
        options["tokenizer_path"] = args.tokenizer
    policy = AutoPolicy.from_pretrained(args.model_dir, **options)
    try:
        print("dispatched model_type:", policy.metadata.get("model_type"))

        observation = synthetic_observation_for(policy)
        result = policy.infer(observation)

        actions = result["actions"]
        print(f"actions: shape={actions.shape} dtype={actions.dtype}")
        print(f"timing:  {result['timing']}")
    finally:
        policy.close()


if __name__ == "__main__":
    main()
