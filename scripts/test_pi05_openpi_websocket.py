#!/usr/bin/env python3
"""Smoke-test a ApxInf PI0.5 server with OpenPI's official websocket client."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import statistics
import time
import urllib.request

import numpy as np
from openpi_client import websocket_client_policy


ACTION_SHAPE = (10, 7)


def bypass_proxy_for_host(host: str) -> None:
    for variable in ("NO_PROXY", "no_proxy"):
        entries = [item for item in os.environ.get(variable, "").split(",") if item]
        if host not in entries:
            entries.append(host)
        os.environ[variable] = ",".join(entries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--expected-precision", choices=("fp8", "bf16", "int8"))
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--prompt", default="put the black bowl on the plate")
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def main() -> None:
    args = parse_args()
    if args.requests <= 0:
        raise ValueError("--requests must be positive")
    bypass_proxy_for_host(args.host)

    health_url = f"http://{args.host}:{args.port}/healthz"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(health_url, timeout=10) as response:
        health_status = response.status
        health_body = response.read().decode("utf-8")
    if health_status != 200 or health_body != "OK\n":
        raise RuntimeError(
            f"unexpected health response status={health_status}, body={health_body!r}"
        )

    rng = np.random.default_rng(args.seed)
    base = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    wrist = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    state = np.zeros(8, dtype=np.float32)
    observation = {
        "observation/image": base,
        "observation/wrist_image": wrist,
        "observation/state": state,
        "prompt": args.prompt,
    }

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    if args.expected_precision is not None:
        actual_precision = metadata.get("precision")
        if actual_precision != args.expected_precision:
            raise RuntimeError(
                f"server precision is {actual_precision!r}, expected {args.expected_precision!r}"
            )

    latencies = []
    checksums = []
    response_dtypes = []
    try:
        for _ in range(args.requests):
            started = time.perf_counter_ns()
            result = client.infer(observation)
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
            actions = np.asarray(result["actions"])
            if actions.shape != ACTION_SHAPE:
                raise ValueError(
                    f"OpenPI response actions have shape {actions.shape}, expected {ACTION_SHAPE}"
                )
            if not np.issubdtype(actions.dtype, np.floating):
                raise TypeError(
                    f"OpenPI response actions must be floating point, got {actions.dtype}"
                )
            if not np.isfinite(actions).all():
                raise FloatingPointError("OpenPI response contains non-finite actions")
            if "server_timing" not in result or "policy_timing" not in result:
                raise KeyError("OpenPI response is missing timing dictionaries")
            latencies.append(elapsed_ms)
            checksums.append(float(np.abs(actions).sum()))
            response_dtypes.append(str(actions.dtype))
    finally:
        client._ws.close()  # The upstream client currently exposes no public close().

    document = {
        "schema": "apxinf.pi05.openpi-websocket-smoke.v1",
        "endpoint": f"ws://{args.host}:{args.port}",
        "metadata": metadata,
        "healthz": {"status": health_status, "body": health_body},
        "requests": args.requests,
        "prompt": args.prompt,
        "action_shape": list(ACTION_SHAPE),
        "action_dtypes": response_dtypes,
        "action_abs_checksums": checksums,
        "latency_ms": {
            "min": min(latencies),
            "p50": statistics.median(latencies),
            "p95": percentile(latencies, 0.95),
            "max": max(latencies),
            "mean": statistics.fmean(latencies),
            "samples": latencies,
        },
        "passed": True,
    }
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered)
        temporary.replace(args.output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
