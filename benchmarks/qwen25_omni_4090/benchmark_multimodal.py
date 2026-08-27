#!/usr/bin/env python3
"""Run the frozen real-image and real-audio Qwen2.5-Omni service gate."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
SCHEMA = "apxinf.qwen25_omni.multimodal_gate.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def post_json(url: str, body: dict[str, Any], timeout: float) -> tuple[int, Any, float]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
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


def reference_tokens(path: Path) -> tuple[str | None, dict[str, list[int]]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    tokens: dict[str, list[int]] = {}
    for case in report.get("cases", []):
        direct = case.get("output_tokens")
        if isinstance(direct, list):
            tokens[case["case_id"]] = direct
            continue
        owner = case.get("candidate") or case.get("deployed") or case.get("baseline")
        if isinstance(owner, dict) and isinstance(owner.get("output_tokens"), list):
            tokens[case["case_id"]] = owner["output_tokens"]
    binary_hash = report.get("candidate_binary_sha256") or report.get(
        "deployed_binary_sha256"
    )
    return binary_hash, tokens


def request_body(kind: str, media: Path, prompt: str) -> dict[str, Any]:
    encoded = base64.b64encode(media.read_bytes()).decode("ascii")
    if kind == "image":
        media_part = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encoded}"},
        }
    elif kind == "audio":
        media_part = {
            "type": "input_audio",
            "input_audio": {"format": "wav", "data": encoded},
        }
    else:
        raise ValueError(f"unsupported media kind: {kind}")
    return {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [media_part, {"type": "text", "text": prompt}],
            }
        ],
        "temperature": 0,
        "max_tokens": 16,
        "stream": False,
    }


def run_case(
    base_url: str,
    case_id: str,
    kind: str,
    media: Path,
    prompt: str,
    expected: list[int] | None,
    timeout: float,
) -> dict[str, Any]:
    status, payload, wall_seconds = post_json(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        request_body(kind, media, prompt),
        timeout,
    )
    apxinf = payload.get("apxinf", {}) if isinstance(payload, dict) else {}
    choices = payload.get("choices", []) if isinstance(payload, dict) else []
    message = choices[0].get("message", {}) if choices else {}
    tokens = apxinf.get("tokens")
    exact = expected is None or tokens == expected
    passed = (
        status == 200
        and isinstance(tokens, list)
        and apxinf.get("fallback_active") is False
        and exact
    )
    record: dict[str, Any] = {
        "case_id": case_id,
        "input": {
            "kind": "image/png" if kind == "image" else "audio/wav",
            "path": str(media),
            "sha256": sha256_file(media),
            "prompt": prompt,
        },
        "status": status,
        "wall_seconds": wall_seconds,
        "prompt_tokens": payload.get("usage", {}).get("prompt_tokens")
        if isinstance(payload, dict)
        else None,
        "output_tokens": tokens,
        "text": message.get("content"),
        "ttft_seconds": apxinf.get("ttft_seconds"),
        "tpot_seconds": apxinf.get("tpot_seconds"),
        "fallback_active": apxinf.get("fallback_active"),
        "expected_output_tokens": expected,
        "exact_token_agreement": exact if expected is not None else None,
        "passed": passed,
    }
    if status != 200:
        record["error"] = payload.get("error", payload) if isinstance(payload, dict) else payload
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument(
        "--image", type=Path, default=Path("scripts/roofline_decode_throughput.png")
    )
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--binary-path", type=Path)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image = args.image.resolve(strict=True)
    audio = args.audio.resolve(strict=True)
    reference = args.reference.resolve(strict=True)
    accepted_binary, expected = reference_tokens(reference)
    required = {"real_png_chart_title", "real_wav_sine_description"}
    missing = sorted(required - expected.keys())
    if missing:
        raise ValueError(f"reference is missing cases: {', '.join(missing)}")

    started_at = datetime.now(timezone.utc).isoformat()
    cases = [
        run_case(
            args.base_url,
            "real_png_chart_title",
            "image",
            image,
            "Read the chart title. Answer with the title only.",
            expected["real_png_chart_title"],
            args.timeout,
        ),
        run_case(
            args.base_url,
            "real_wav_sine_description",
            "audio",
            audio,
            "Describe this audio signal briefly.",
            expected["real_wav_sine_description"],
            args.timeout,
        ),
    ]
    report = {
        "schema": SCHEMA,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_ID,
        "endpoint": "/v1/chat/completions",
        "contract": {
            "single_request": True,
            "sampling": "greedy",
            "temperature": 0,
            "stream": False,
            "max_tokens": 16,
            "fallback_required": False,
            "comparison": "complete output token sequence",
        },
        "reference": {
            "path": str(reference),
            "binary_sha256": accepted_binary,
        },
        "candidate_binary_sha256": sha256_file(args.binary_path.resolve(strict=True))
        if args.binary_path
        else None,
        "cases": cases,
        "passed": all(case["passed"] for case in cases),
        "limitations": [
            "Single image and single audio requests only",
            "Text output only; video and speech generation remain unsupported",
            "Single observations are correctness and path coverage, not timing admission samples",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "passed": report["passed"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
