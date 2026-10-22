#!/usr/bin/env python3
"""Check exact fact retrieval at measured local Messages input lengths."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import secrets

from http_benchmark import local_url, metal_measurement_lock, metrics_snapshot, request_json


MAX_NOTES = 8192
IDENTITY_FIELDS = ("model_revision", "capability_revision", "worker_epoch",
                   "model_manifest", "capabilities", "runtime")


def request_body(model, facts, notes, max_tokens):
    lines = [f"START FACT: start={facts['start']}"]
    for index in range(notes):
        if index == notes // 2:
            lines.append(f"MIDDLE FACT: middle={facts['middle']}")
        lines.append(f"Note {index:05d}: the ceramic marker belongs to shelf {index % 97:02d}.")
    if notes == 0:
        lines.append(f"MIDDLE FACT: middle={facts['middle']}")
    lines.extend((f"END FACT: end={facts['end']}",
                  "Return only one JSON object with keys start, middle, end. "
                  "Copy each value from its matching FACT line exactly."))
    return {"model": model, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": "\n".join(lines)}]}


def counted_body(base_url, model, facts, notes, max_tokens, request):
    body = request_body(model, facts, notes, max_tokens)
    status, response = request(base_url, "/v1/messages/count_tokens", body)
    error = response.get("error") if isinstance(response, dict) else None
    if status == 400 and isinstance(error, dict) and error.get("code") == "context_limit":
        return None, body
    count = response.get("input_tokens") if isinstance(response, dict) else None
    if status != 200 or type(count) is not int or count < 1:
        raise ValueError(f"Token count failed: HTTP {status}: {response!r}")
    return count, body


def calibrate(base_url, model, facts, target, max_tokens, request):
    calls = 0

    def measure(notes):
        nonlocal calls
        calls += 1
        return counted_body(base_url, model, facts, notes, max_tokens, request)

    best_count, best_body = measure(0)
    if best_count is None:
        raise ValueError("The base prompt exceeds the context limit.")
    if best_count > target:
        return best_count, best_body, calls
    low, high = 0, MAX_NOTES
    while low < high:
        middle = (low + high + 1) // 2
        count, body = measure(middle)
        if count is not None and count <= target:
            low, best_count, best_body = middle, count, body
        else:
            high = middle - 1
    final_count, final_body = measure(low)
    if final_count != best_count or final_body != best_body:
        raise ValueError("The token count changed during calibration.")
    return final_count, final_body, calls


def identity(ready):
    worker = ready.get("worker") if isinstance(ready, dict) else None
    if not isinstance(worker, dict):
        raise ValueError("Readiness lacks a worker.")
    if any(field not in worker for field in IDENTITY_FIELDS):
        raise ValueError("Readiness lacks worker identity fields.")
    return {field: worker[field] for field in IDENTITY_FIELDS}


def read_ready(base_url, request):
    status, ready = request(base_url, "/readyz")
    if status != 200 or not isinstance(ready, dict) or ready.get("ready") is not True:
        raise ValueError(f"Service is not ready: HTTP {status}: {ready!r}")
    if ready.get("profile") != "apxinf-serial-v0.1":
        raise ValueError("The service has an unsupported profile.")
    identity(ready)
    return ready


def output_facts(response):
    if not isinstance(response, dict):
        return None
    blocks = response.get("content")
    if not isinstance(blocks, list) or len(blocks) != 1:
        return None
    block = blocks[0]
    if not isinstance(block, dict) or block.get("type") != "text":
        return None
    try:
        result = json.loads(block["text"])
    except (KeyError, TypeError, ValueError):
        return None
    if type(result) is not dict or set(result) != {"start", "middle", "end"}:
        return None
    return result if all(type(value) is str for value in result.values()) else None


def run_case(base_url, target, max_tokens, request=request_json, metrics=metrics_snapshot):
    facts = {name: secrets.token_hex(8) for name in ("start", "middle", "end")}
    case = {"target_input_tokens": target, "status": "failed", "reason": None,
            "nonce_facts": facts, "notes": None, "ready_before": None, "ready_after": None,
            "metrics_before": metrics(base_url), "metrics_after": None}
    try:
        ready = read_ready(base_url, request)
        case["ready_before"] = ready
        worker = ready["worker"]
        capabilities = worker["capabilities"]
        context = capabilities.get("max_context")
        output = capabilities.get("max_output_tokens")
        if type(context) is not int or context < 1 or type(output) is not int or output < 1:
            raise ValueError("Readiness lacks positive context and output limits.")
        if max_tokens > output or target + max_tokens > context:
            case.update(status="unsupported", reason="Target or output allowance exceeds readiness limits.")
            return case
        count, body, calls = calibrate(base_url, ready["model"], facts, target, max_tokens, request)
        case.update(actual_input_tokens=count, count_calls=calls, notes=body["messages"][0]["content"].count("\nNote "))
        case["body_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                                     ensure_ascii=False).encode("utf-8")).hexdigest()
        if count < target * .95 or count > target:
            case.update(status="unsupported", reason="The measured input did not reach 95 percent of the target.")
            return case
        if count + max_tokens > context:
            case.update(status="unsupported", reason="The measured input and output allowance exceed context.")
            return case
        status, response = request(base_url, "/v1/messages", body)
        case.update(http_status=status, raw_response=response)
        if isinstance(response, dict):
            usage = response.get("usage", {})
            case["response_input_tokens"] = usage.get("input_tokens") if isinstance(usage, dict) else None
            case["stop_reason"] = response.get("stop_reason")
        case["observed_facts"] = output_facts(response)
        if status != 200:
            case["reason"] = f"Generation returned HTTP {status}."
        elif response.get("model") != ready["model"]:
            case["reason"] = "The response model differs from readiness."
        elif case["response_input_tokens"] != count:
            case["reason"] = "The response input count differs from the measured count."
        elif case["stop_reason"] != "end_turn":
            case["reason"] = "Generation did not end normally."
        elif case["observed_facts"] != facts:
            case["reason"] = "The response has missing or incorrect nonce facts."
        else:
            case["status"] = "passed"
    except Exception as exc:
        case["reason"] = str(exc)
    finally:
        case["metrics_after"] = metrics(base_url)
        try:
            after = read_ready(base_url, request)
            case["ready_after"] = after
            if case["ready_before"] is not None and (
                after.get("model") != case["ready_before"].get("model")
                or identity(after) != identity(case["ready_before"])
            ):
                case.update(status="failed", reason="Readiness identity changed during the case.")
        except Exception as exc:
            case.update(status="failed", reason=f"Final readiness failed: {exc}")
    return case


def run(args):
    local_url(args.base_url)
    if not args.input_tokens or any(target < 1 for target in args.input_tokens) or args.max_tokens < 1:
        raise ValueError("Input targets and max_tokens must be positive.")
    report = {"format": "apxinf-context-probe-v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "base_url": args.base_url, "max_tokens": args.max_tokens, "cases": []}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with metal_measurement_lock(args.metal_lock):
        for target in args.input_tokens:
            report["cases"].append(run_case(args.base_url, target, args.max_tokens,
                                            request_json, metrics_snapshot))
            (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return all(case["status"] == "passed" for case in report["cases"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--input-tokens", type=int, nargs="+", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metal-lock", type=Path)
    raise SystemExit(0 if run(parser.parse_args()) else 1)


if __name__ == "__main__":
    main()
