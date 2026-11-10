#!/usr/bin/env python3
"""Run the frozen HTTP protocol gates without third-party dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CONTRACT = "apxinf.qwen38_27b.inference_interface.v1"
MODEL_REVISION = "63768c10df38c0395e12ef49edac1bd539eaeeea"
MODEL_VOCAB_SIZE = 248_320


def build_cases(max_model_len: int, token: int = 1) -> list[dict[str, Any]]:
    common = {
        "max_new_tokens": 1,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": False,
    }
    return [
        {"id": "malformed_json", "body": "{not-json", "mode": "raw"},
        {"id": "empty_input_ids", "body": {**common, "input_ids": []}},
        {"id": "negative_token_id", "body": {**common, "input_ids": [-1]}},
        {
            "id": "out_of_vocabulary_token_id",
            "body": {**common, "input_ids": [4294967295]},
        },
        {
            "id": "unsupported_temperature",
            "body": {**common, "input_ids": [token], "temperature": 0.1},
        },
        {
            "id": "over_budget",
            "body": {**common, "input_ids": [token], "max_new_tokens": max_model_len},
        },
        {
            "id": "unsupported_modality_field",
            "body": {**common, "input_ids": [token], "images": ["x"]},
        },
        {
            "id": "valid_short_nostream_request",
            "body": {**common, "input_ids": list(range(1, 9))},
        },
        {"id": "health_after_invalid_requests", "method": "GET", "path": "/health"},
        {"id": "health_contract_identity", "method": "GET", "path": "/health"},
    ]


def _request(base_url: str, case: dict[str, Any], timeout: float) -> dict[str, Any]:
    method = case.get("method", "POST")
    path = case.get("path", "/v1/evaluations/generate")
    body_value = case.get("body")
    if isinstance(body_value, str):
        payload = body_value.encode("utf-8")
    elif body_value is None:
        payload = b""
    else:
        payload = json.dumps(body_value, separators=(",", ":"), sort_keys=True).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=payload if method != "GET" else None,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as error:
        raw = error.read()
        status = error.code
    elapsed_ms = (time.monotonic() - started) * 1000.0
    text = raw.decode("utf-8", errors="replace")
    try:
        parsed: Any = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    return {
        "request_method": method,
        "request_path": path,
        "request_body": payload.decode("utf-8", errors="replace"),
        "request_body_raw": payload.decode("utf-8", errors="replace"),
        "status_code": status,
        "response": parsed if parsed is not None else text[:4000],
        "response_raw": text,
        "elapsed_ms": round(elapsed_ms, 3),
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def evaluate_row(case: dict[str, Any], result: dict[str, Any]) -> bool:
    case_id = case["id"]
    status = result.get("status_code")
    response = result.get("response")
    if case_id == "malformed_json":
        return status == 400
    if case_id in {
        "empty_input_ids",
        "negative_token_id",
        "out_of_vocabulary_token_id",
        "unsupported_temperature",
        "over_budget",
        "unsupported_modality_field",
    }:
        return status == 400 and isinstance(response, dict) and isinstance(response.get("error"), dict)
    if case_id == "valid_short_nostream_request":
        usage = response.get("usage") if isinstance(response, dict) else None
        return bool(
            status == 200
            and isinstance(response, dict)
            and response.get("type") == "result"
            and isinstance(response.get("output_ids"), list)
            and len(response["output_ids"]) == 1
            and isinstance(usage, dict)
            and usage.get("prompt_tokens") == 8
            and usage.get("completion_tokens") == 1
            and usage.get("total_tokens") == 9
        )
    if case_id == "health_after_invalid_requests":
        return status == 200 and isinstance(response, dict) and response.get("status") == "ok"
    if case_id == "health_contract_identity":
        try:
            validate_health_identity(result)
        except ValueError:
            return False
        return True
    raise ValueError(f"unknown protocol gate {case_id}")


def health_max_model_len(health_result: dict[str, Any]) -> int:
    response = health_result.get("response")
    value = response.get("max_model_len") if isinstance(response, dict) else None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("/health.max_model_len must be a positive integer")
    return value


def validate_health_identity(health_result: dict[str, Any]) -> str:
    """Validate the frozen evaluator identity and classify fixture evidence."""
    if health_result.get("status_code") != 200:
        raise ValueError("/health must return HTTP 200")
    response = health_result.get("response")
    if not isinstance(response, dict) or response.get("status") != "ok":
        raise ValueError("/health response must have status=ok")
    if response.get("evaluation_contract") != CONTRACT:
        raise ValueError("unexpected evaluation contract")
    if response.get("model_revision") != MODEL_REVISION:
        raise ValueError("unexpected model revision")
    if response.get("vocab_size") != MODEL_VOCAB_SIZE:
        raise ValueError("/health.vocab_size must equal model vocab 248320")
    if response.get("fallback_active") is not False:
        raise ValueError("fallback_active must be false")
    capabilities = response.get("capabilities")
    if not isinstance(capabilities, dict) or capabilities.get("multimodal") is not False:
        raise ValueError("multimodal capability must be false")
    stub = response.get("stub")
    if not isinstance(stub, bool):
        raise ValueError("/health.stub must be a boolean")
    return "stub_fixture" if stub else "production_runtime"


def run_gates(base_url: str, timeout: float) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    initial_health = _request(
        base_url,
        {"id": "initial_health", "method": "GET", "path": "/health"},
        timeout,
    )
    runtime_kind = validate_health_identity(initial_health)
    max_model_len = health_max_model_len(initial_health)
    rows = []
    for case in build_cases(max_model_len):
        result = _request(base_url, case, timeout)
        passed = evaluate_row(case, result)
        rows.append({**case, **result, "passed": passed})
    ending_health = _request(
        base_url,
        {"id": "ending_health", "method": "GET", "path": "/health"},
        timeout,
    )
    ending_kind = validate_health_identity(ending_health)
    return {
        "schema": "apxinf.protocol-gates.v2",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "base_url": base_url.rstrip("/"),
        "runtime_kind": runtime_kind,
        "stub": runtime_kind == "stub_fixture",
        "max_model_len": max_model_len,
        "initial_health": initial_health,
        "ending_health": ending_health,
        "passed": all(row["passed"] for row in rows) and ending_kind == runtime_kind,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = run_gates(args.base_url, args.timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    hash_path = args.output.with_suffix(args.output.suffix + ".sha256")
    hash_path.write_text(f"{digest}  {args.output.name}\n", encoding="utf-8")
    print(json.dumps({"passed": evidence["passed"], "rows": len(evidence["rows"]), "output": str(args.output), "sha256": digest}))
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
