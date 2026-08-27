#!/usr/bin/env python3
"""Exercise fail-closed Qwen2.5-Omni HTTP generation contracts."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "apxinf.qwen25_omni.contract_gate.v1"


def request_json(
    method: str, url: str, body: dict[str, Any] | None, timeout: float
) -> tuple[int, Any, float]:
    data = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method=method,
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as error:
        raw = error.read()
        status = error.code
    elapsed = time.perf_counter() - started
    try:
        return status, json.loads(raw), elapsed
    except json.JSONDecodeError:
        return status, {"raw": raw.decode("utf-8", errors="replace")}, elapsed


def evaluation_body(prompt_tokens: int, output_tokens: int) -> dict[str, Any]:
    return {
        "input_ids": [1000 + index % 17 for index in range(prompt_tokens)],
        "max_new_tokens": output_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": False,
    }


def probe_passes(status: int, payload: Any, message_fragment: str) -> bool:
    if status != 400 or not isinstance(payload, dict):
        return False
    error = payload.get("error")
    return (
        isinstance(error, dict)
        and error.get("type") == "invalid_request"
        and message_fragment in str(error.get("message", ""))
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    health_status, health, _ = request_json(
        "GET", f"{base_url}/health", None, args.timeout
    )
    if health_status != 200 or not isinstance(health, dict):
        raise RuntimeError(f"health endpoint returned HTTP {health_status}")
    max_model_len = int(health["max_model_len"])
    max_output_tokens = int(health["max_output_tokens"])
    endpoint = f"{base_url}/v1/evaluations/generate"

    definitions = [
        (
            "combined_context_over_limit",
            evaluation_body(max_model_len, 1),
            f"exceeds context {max_model_len}",
        ),
        (
            "completion_over_limit",
            evaluation_body(1, max_output_tokens + 1),
            f"max_new_tokens must be in 1..={max_output_tokens}",
        ),
        (
            "non_greedy_temperature",
            {**evaluation_body(1, 1), "temperature": 0.1},
            "temperature supports only 0",
        ),
        (
            "streaming_evaluation",
            {**evaluation_body(1, 1), "stream": True},
            "evaluation v1 requires stream=false",
        ),
    ]
    probes = []
    for case_id, body, fragment in definitions:
        status, payload, elapsed = request_json(
            "POST", endpoint, body, args.timeout
        )
        probes.append(
            {
                "case_id": case_id,
                "status": status,
                "elapsed_seconds": elapsed,
                "expected_error_fragment": fragment,
                "error": payload.get("error") if isinstance(payload, dict) else payload,
                "passed": probe_passes(status, payload, fragment),
            }
        )
    post_status, post_health, _ = request_json(
        "GET", f"{base_url}/health", None, args.timeout
    )
    report = {
        "schema": SCHEMA,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "endpoint": "/v1/evaluations/generate",
        "health": health,
        "probes": probes,
        "post_probe_health": post_health,
        "passed": all(probe["passed"] for probe in probes)
        and post_status == 200
        and isinstance(post_health, dict)
        and post_health.get("status") == "ok",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "passed": report["passed"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
