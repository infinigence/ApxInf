#!/usr/bin/env python3
"""Minimal websocket policy server (openpi-compatible wire) over a ``apxinf`` policy.

Loads an in-process ``AutoPolicy`` and serves it with
``apxinf.serving.WebsocketPolicyServer`` — the same transport shell the
production launcher (scripts/pi05_openpi_websocket_server.py) uses, reduced to
the essentials. An unmodified ``openpi_client`` connects to it; see
``openpi_client.py`` for the other end.

Requires the ``apxinf_py`` CUDA binding plus the transport deps
(``websockets`` / ``msgpack``; see scripts/requirements-pi05-websocket.txt).

    python examples/openpi_server.py --model-dir /path/to/checkpoint
"""

from __future__ import annotations

import argparse
import logging
import pathlib

import _common  # noqa: F401  (path shim so ``import apxinf`` works from a checkout)

from apxinf import AutoPolicy
from apxinf.serving import WebsocketPolicyServer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=pathlib.Path)
    parser.add_argument("--precision", choices=("auto", "fp8", "bf16", "int8"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--action-dim", type=int, default=7, help="0 keeps the full vector")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    policy = AutoPolicy.from_pretrained(
        args.model_dir,
        device=args.device,
        precision=args.precision,
        action_dim=(args.action_dim or None),
        # Extra metadata is sent to the client on connect (get_server_metadata).
        metadata={"protocol": "openpi.websocket_policy", "precision": args.precision},
    )
    server = WebsocketPolicyServer(policy, args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("shutting down")
    finally:
        policy.close()


if __name__ == "__main__":
    main()
