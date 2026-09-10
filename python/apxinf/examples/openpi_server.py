#!/usr/bin/env python3
"""Minimal websocket policy server (openpi-compatible wire) over a ``apxinf`` policy.

Loads an in-process ``AutoPolicy`` and serves it with
``apxinf.serving.WebsocketPolicyServer`` — a model-agnostic transport shell over
any ``Policy``, reduced here to the essentials. An unmodified ``openpi_client``
connects to it; see ``openpi_client.py`` for the other end.

The wire keys are the caller's to name: this engine holds no dataset's dialect,
so ``--image-keys`` / ``--state-key`` say what your client sends. Omit them and
the policy falls back to its own view-slot vocabulary (``base_0_rgb``, ...),
which is a fallback rather than a contract. A deployment that wants named robot
contracts, presets, and simulator glue wants ``apxinf-robo`` on top of this.

Requires the ``apxinf_py`` CUDA binding plus the transport deps
(``websockets`` / ``msgpack``; see scripts/requirements-pi05-websocket.txt).

    python examples/openpi_server.py --model-dir /path/to/checkpoint \
        --image-keys observation/image,observation/wrist_image \
        --state-key observation/state \
        --policy-options '{"norm_key":"x2_normal"}'
"""

from __future__ import annotations

import argparse
import logging
import pathlib

from _common import json_object, policy_kwargs  # noqa: E402 (also installs source path shim)

from apxinf import AutoPolicy
from apxinf.serving import WebsocketPolicyServer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=pathlib.Path)
    parser.add_argument("--tokenizer", type=pathlib.Path)
    parser.add_argument("--precision", choices=("auto", "fp8", "bf16", "int8"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--action-dim", type=int, default=0, help="0 keeps the full vector")
    parser.add_argument(
        "--policy-options",
        type=json_object,
        default={},
        metavar="JSON",
        help="extra concrete-policy options as a JSON object",
    )
    parser.add_argument(
        "--image-keys",
        default=None,
        help=(
            "comma-separated camera wire keys, in model view-slot order. Omitted, "
            "the policy names them after its own view slots (base_0_rgb, ...) — a "
            "real deployment states its robot's keys."
        ),
    )
    parser.add_argument(
        "--state-key",
        default=None,
        help="wire key your client sends state under; required only if state is read",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    options = policy_kwargs(
        args.policy_options,
        device=args.device,
        precision=args.precision,
        action_dim=(args.action_dim or None),
        metadata={"protocol": "openpi.websocket_policy", "precision": args.precision},
    )
    if getattr(args, "tokenizer", None) is not None:
        options["tokenizer_path"] = args.tokenizer
    image_keys = getattr(args, "image_keys", None)
    if image_keys:
        options["image_keys"] = tuple(
            key.strip() for key in image_keys.split(",") if key.strip()
        )
    if getattr(args, "state_key", None):
        options["state_key"] = args.state_key
    policy = AutoPolicy.from_pretrained(args.model_dir, **options)
    server = WebsocketPolicyServer(policy, args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("shutting down")
    finally:
        policy.close()


if __name__ == "__main__":
    main()
