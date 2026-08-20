#!/usr/bin/env python3
"""Run the frozen image-input capability suite against ApxInf or vLLM."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPORT_SCHEMA = "apxinf.qwen38_27b.multimodal_report.v1"
ACCEPTED_UNSUPPORTED_STATUSES = {400, 415, 422, 501}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request_json(
    method: str,
    url: str,
    body: dict[str, Any] | None,
    timeout: float,
) -> tuple[int, Any, float]:
    payload = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {} if payload is None else {"Content-Type": "application/json"}
    request = urllib.request.Request(url, data=payload, headers=headers, method=method)
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as error:
        raw = error.read()
        status = error.code
    elapsed = time.perf_counter() - start
    if not raw:
        return status, {}, elapsed
    try:
        return status, json.loads(raw), elapsed
    except json.JSONDecodeError:
        return status, {"raw": raw.decode("utf-8", errors="replace")}, elapsed


def normalize_answer(value: str) -> str:
    text = value.strip(" \t\r\n")
    if text.startswith("<think>") and "</think>" in text:
        text = text.split("</think>", 1)[1].strip(" \t\r\n")
    return text


def response_text(response: Any) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("response is missing choices[0].message.content") from error
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        return "".join(parts)
    raise ValueError("message.content must be a string or content-part list")


def load_suite(suite_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = suite_dir / "manifest.json"
    cases_path = suite_dir / "cases.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["cases_jsonl_sha256"] != sha256_file(cases_path):
        raise ValueError("cases.jsonl hash does not match manifest")
    rows = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != manifest["case_count"]:
        raise ValueError("case count does not match manifest")
    for row in rows:
        image_path = suite_dir / row["image"]
        image_hash = sha256_file(image_path)
        if image_hash != row["image_sha256"] or image_hash != manifest["images"].get(row["id"]):
            raise ValueError(f"{row['id']}: image hash does not match manifest")
    return manifest, rows


def image_request(case: dict[str, Any], suite_dir: Path, model: str | None) -> dict[str, Any]:
    encoded = base64.b64encode((suite_dir / case["image"]).read_bytes()).decode("ascii")
    body: dict[str, Any] = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                    {"type": "text", "text": case["prompt"]},
                ],
            }
        ],
        "temperature": 0.0,
        "max_completion_tokens": case["max_completion_tokens"],
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if model:
        body["model"] = model
    return body


def unsupported_probe_valid(status: int, payload: Any) -> bool:
    if status not in ACCEPTED_UNSUPPORTED_STATUSES or not isinstance(payload, dict):
        return False
    error = payload.get("error")
    return isinstance(error, dict) and error.get("type") == "unsupported_capability"


def run(args: argparse.Namespace) -> dict[str, Any]:
    contract_path = args.contract.resolve(strict=True)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    manifest, cases = load_suite(args.suite_dir.resolve(strict=True))
    if manifest["contract_sha256"] != sha256_file(contract_path):
        raise ValueError("suite manifest belongs to a different multimodal contract")

    health_status, health, _ = request_json("GET", f"{args.base_url.rstrip('/')}/health", None, args.timeout)
    if health_status != 200:
        raise RuntimeError(f"health endpoint returned HTTP {health_status}")
    capabilities = health.get("capabilities", {}) if isinstance(health, dict) else {}
    declared = capabilities.get("multimodal") if isinstance(capabilities, dict) else None
    fallback_active = health.get("fallback_active") if isinstance(health, dict) else None
    force_supported = args.api_mode == "vllm-chat"
    if force_supported:
        # vLLM's /health body does not advertise model capabilities. The
        # teacher selects this mode only for the named reference backend.
        declared = True
        fallback_active = False

    records: list[dict[str, Any]] = []
    probe_status: int | None = None
    fail_closed: bool | None = None
    if declared is False and not force_supported:
        probe_status, payload, elapsed = request_json(
            "POST",
            f"{args.base_url.rstrip('/')}/v1/chat/completions",
            image_request(cases[0], args.suite_dir, args.served_model_name),
            args.timeout,
        )
        fail_closed = unsupported_probe_valid(probe_status, payload)
        records.append(
            {
                "id": "unsupported-capability-probe",
                "status": probe_status,
                "success": False,
                "passed": fail_closed,
                "e2e_s": elapsed,
                "error_type": payload.get("error", {}).get("type") if isinstance(payload, dict) else None,
            }
        )
    else:
        for case in cases:
            status, payload, elapsed = request_json(
                "POST",
                f"{args.base_url.rstrip('/')}/v1/chat/completions",
                image_request(case, args.suite_dir, args.served_model_name),
                args.timeout,
            )
            output = ""
            error: str | None = None
            if status == 200:
                try:
                    output = normalize_answer(response_text(payload))
                except ValueError as exc:
                    error = str(exc)
            else:
                error = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            passed = status == 200 and error is None and output == case["expected"]
            records.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "status": status,
                    "success": status == 200 and error is None,
                    "passed": passed,
                    "output": output,
                    "expected": case["expected"],
                    "e2e_s": elapsed,
                    "error": error,
                }
            )

    final_health_status, final_health, _ = request_json(
        "GET", f"{args.base_url.rstrip('/')}/health", None, args.timeout
    )
    service_healthy = final_health_status == 200 and (
        not isinstance(final_health, dict) or final_health.get("status", "ok") == "ok"
    )
    case_records = [row for row in records if row["id"] != "unsupported-capability-probe"]
    successes = sum(bool(row["success"]) for row in case_records)
    passed = sum(bool(row["passed"]) for row in case_records)
    latencies = [float(row["e2e_s"]) for row in case_records if row["success"]]
    return {
        "schema": REPORT_SCHEMA,
        "implementation": {
            "name": args.implementation_name,
            "revision": args.implementation_revision,
            "backend": args.backend,
        },
        "split": manifest["split"],
        "capability_declared": declared,
        "fallback_active": fallback_active,
        "fail_closed": fail_closed,
        "probe_status": probe_status,
        "cases_passed": passed,
        "cases_total": len(cases),
        "request_success_rate": successes / len(cases) if case_records else 0.0,
        "median_e2e_s": statistics.median(latencies) if latencies else None,
        "service_healthy_after_run": service_healthy,
        "records": records,
        "evidence": {
            "contract_sha256": sha256_file(contract_path),
            "manifest_sha256": sha256_file(args.suite_dir / "manifest.json"),
            "cases_jsonl_sha256": manifest["cases_jsonl_sha256"],
        },
        "contract_status": contract["status"],
        "capability_source": "runner_mode" if force_supported else "health",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-mode", choices=("apxinf-chat", "vllm-chat"), default="apxinf-chat")
    parser.add_argument("--served-model-name")
    parser.add_argument("--implementation-name", required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).with_name("multimodal-contract-v1.json"),
    )
    args = parser.parse_args()
    if args.api_mode == "vllm-chat" and not args.served_model_name:
        parser.error("--served-model-name is required in vllm-chat mode")
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
