#!/usr/bin/env python3
"""Minimal OpenPI websocket client: connect, read metadata, send one observation.

The distilled client half of the LIBERO eval (scripts/eval_pi05_libero_openpi.py)
with the simulator and resumable ledger stripped away — just how to reach a
``apxinf.serving`` server (or any OpenPI ``WebsocketPolicyServer``) and get an
action chunk back. Pair it with ``openpi_server.py``.

The client sends *raw* camera frames + prompt; the server's policy owns resize /
tokenize / normalize. Only OpenPI-contract keys come back on the wire
(``actions`` + ``policy_timing``); ``server_timing`` is the server's tolerated
diagnostic namespace.

Requires ``openpi_client`` (the upstream OpenPI client package).

    python examples/openpi_client.py --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from openpi_client import websocket_client_policy

from _common import synthetic_observation  # noqa: E402 (path shim in _common)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default="pick up the block")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # WebsocketClientPolicy talks to the proxy by default; exempt a local host so
    # the connection is direct (the eval does the same).
    for variable in ("NO_PROXY", "no_proxy"):
        entries = [item for item in os.environ.get(variable, "").split(",") if item]
        if args.host not in entries:
            entries.append(args.host)
        os.environ[variable] = ",".join(entries)

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    print("server metadata:", metadata)

    observation = synthetic_observation(prompt=args.prompt)
    response = client.infer(observation)

    actions = np.asarray(response["actions"], dtype=np.float32)
    print(f"actions: shape={actions.shape} dtype={actions.dtype}")
    print(f"policy_timing: {response.get('policy_timing')}")
    print(f"server_timing: {response.get('server_timing')}")


if __name__ == "__main__":
    main()
