#!/usr/bin/env python3
"""Verify that the resident Omni processor survives a malformed media request."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark_contract import request_json
from benchmark_multimodal import request_body, reference_tokens, sha256_file


SCHEMA = "apxinf.qwen25_omni.processor_recovery.v1"


def invalid_media_passes(status: int, payload: Any) -> bool:
    if status != 422 or not isinstance(payload, dict):
        return False
    error = payload.get("error")
    return isinstance(error, dict) and error.get("type") == "unprocessable_media"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8002")
    parser.add_argument(
        "--image", type=Path, default=Path("scripts/roofline_decode_throughput.png")
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--binary-path", type=Path)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    image = args.image.resolve(strict=True)
    reference = args.reference.resolve(strict=True)
    accepted_binary, expected = reference_tokens(reference)
    expected_tokens = expected.get("real_png_chart_title")
    if expected_tokens is None:
        raise ValueError("reference is missing real_png_chart_title")

    health_url = f"{base_url}/health"
    chat_url = f"{base_url}/v1/chat/completions"
    before_status, before_health, _ = request_json(
        "GET", health_url, None, args.timeout
    )

    valid_body = request_body(
        "image", image, "Read the chart title. Answer with the title only."
    )
    invalid_body = json.loads(json.dumps(valid_body))
    invalid_body["messages"][0]["content"][0]["image_url"]["url"] = (
        "data:image/png;base64,AA=="
    )
    invalid_status, invalid_payload, invalid_wall = request_json(
        "POST", chat_url, invalid_body, args.timeout
    )

    middle_status, middle_health, _ = request_json(
        "GET", health_url, None, args.timeout
    )
    valid_status, valid_payload, valid_wall = request_json(
        "POST", chat_url, valid_body, args.timeout
    )
    after_status, after_health, _ = request_json(
        "GET", health_url, None, args.timeout
    )

    apxinf = valid_payload.get("apxinf", {}) if isinstance(valid_payload, dict) else {}
    output_tokens = apxinf.get("tokens")
    health_passed = all(
        status == 200
        and isinstance(payload, dict)
        and payload.get("status") == "ok"
        and payload.get("processor_mode") == "persistent"
        for status, payload in (
            (before_status, before_health),
            (middle_status, middle_health),
            (after_status, after_health),
        )
    )
    valid_passed = (
        valid_status == 200
        and output_tokens == expected_tokens
        and apxinf.get("fallback_active") is False
    )
    report = {
        "schema": SCHEMA,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "input": {
            "path": str(image),
            "sha256": sha256_file(image),
        },
        "reference": {
            "path": str(reference),
            "binary_sha256": accepted_binary,
            "output_tokens": expected_tokens,
        },
        "candidate_binary_sha256": (
            sha256_file(args.binary_path.resolve(strict=True))
            if args.binary_path is not None
            else None
        ),
        "before_health": before_health,
        "malformed_request": {
            "status": invalid_status,
            "wall_seconds": invalid_wall,
            "error": (
                invalid_payload.get("error")
                if isinstance(invalid_payload, dict)
                else invalid_payload
            ),
            "passed": invalid_media_passes(invalid_status, invalid_payload),
        },
        "post_error_health": middle_health,
        "valid_request": {
            "status": valid_status,
            "wall_seconds": valid_wall,
            "output_tokens": output_tokens,
            "exact_token_agreement": output_tokens == expected_tokens,
            "fallback_active": apxinf.get("fallback_active"),
            "passed": valid_passed,
        },
        "final_health": after_health,
        "health_passed": health_passed,
        "passed": (
            invalid_media_passes(invalid_status, invalid_payload)
            and health_passed
            and valid_passed
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "passed": report["passed"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
