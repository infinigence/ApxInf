#!/usr/bin/env python3
"""Run the canonical Qwen3.8-27B evaluator and emit a submission artifact.

Official runs use client-observed, no-profiler timing. This program preserves
one raw row per request and derives every aggregate in submission.json from
those rows; submission fields are never accepted as command-line input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUBMISSION_SCHEMA = "apxinf.qwen38_27b.leaderboard_submission.v1"
TRAJECTORY_SCHEMA = "apxinf.qwen38_27b.trajectory_reference.v1"


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return value


def load_tokenizer(model_dir: Path):
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("transformers is required by the evaluator") from error
    return AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
        use_fast=True,
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: Sequence[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * p / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(
        ordered[lower]
        + (ordered[upper] - ordered[lower]) * (position - lower)
    )


def coefficient_of_variation(values: Sequence[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    mean = statistics.fmean(values)
    return statistics.stdev(values) / mean if mean else None


def ascii_strip(value: str) -> str:
    return value.strip(" \t\n\r\f\v")


def validate_output(case: dict[str, Any], output: str) -> tuple[bool, str]:
    mode = case["validation"]
    actual = ascii_strip(output)
    expected_value = case.get("expected")
    expected = "" if expected_value is None else str(expected_value)
    if mode == "normalized_exact":
        passed = actual == expected
    elif mode == "normalized_prefix":
        passed = actual.startswith(expected)
    elif mode == "nonempty":
        passed = bool(actual)
    elif mode == "qualitative_unscored":
        passed = bool(actual)
    else:
        raise ValueError(f"{case['id']}: unsupported validator {mode!r}")
    return passed, f"validator={mode} expected={expected!r} actual={actual!r}"


def read_dataset(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = directory / "manifest.json"
    manifest = load_json(manifest_path)
    cases_path = directory / manifest["cases_jsonl"]
    if sha256_file(cases_path) != manifest["cases_jsonl_sha256"]:
        raise ValueError(f"{cases_path}: dataset hash mismatch")
    cases: list[dict[str, Any]] = []
    with cases_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            case = json.loads(line)
            if not isinstance(case, dict):
                raise ValueError(f"{cases_path}:{line_number}: case must be an object")
            stored_hash = case.pop("case_sha256", None)
            actual_hash = sha256_bytes(canonical_json(case))
            case["case_sha256"] = stored_hash
            if stored_hash != actual_hash:
                raise ValueError(f"{cases_path}:{line_number}: case hash mismatch")
            input_ids = case.get("input_ids")
            if (
                not isinstance(input_ids, list)
                or not input_ids
                or any(
                    isinstance(token_id, bool)
                    or not isinstance(token_id, int)
                    or token_id < 0
                    for token_id in input_ids
                )
            ):
                raise ValueError(f"{case.get('id')}: invalid input_ids")
            if len(input_ids) != case.get("actual_prompt_tokens"):
                raise ValueError(f"{case.get('id')}: prompt token count mismatch")
            if sha256_bytes(canonical_json(input_ids)) != case.get("input_ids_sha256"):
                raise ValueError(f"{case.get('id')}: input IDs hash mismatch")
            cases.append(case)
    if len(cases) != manifest["case_count"]:
        raise ValueError(f"{directory}: case count does not match manifest")
    if [case["id"] for case in cases] != manifest["case_ids"]:
        raise ValueError(f"{directory}: ordered case IDs do not match manifest")
    return manifest, cases


def http_json(
    method: str,
    url: str,
    *,
    body: Any | None = None,
    timeout: float = 30.0,
    raw_body: bytes | None = None,
) -> tuple[int, dict[str, Any] | None, str]:
    payload = raw_body
    headers: dict[str, str] = {}
    if body is not None:
        payload = canonical_json(body)
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=payload, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
            value = json.loads(raw) if raw.strip() else None
            return response.status, value if isinstance(value, dict) else None, raw
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8", "replace")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = None
        return error.code, value if isinstance(value, dict) else None, raw


def health_check(base_url: str, timeout: float) -> dict[str, Any]:
    status, value, raw = http_json(
        "GET", f"{base_url.rstrip('/')}/health", timeout=min(timeout, 30.0)
    )
    if status != 200:
        raise RuntimeError(f"health check returned HTTP {status}: {raw[:500]}")
    return value or {"status": "ok", "adapter_health_body": raw}


@dataclass
class HardwareSample:
    monotonic_s: float
    gpu_util_pct: float
    memory_controller_util_pct: float
    memory_used_mib: float
    power_w: float
    temperature_c: float


@dataclass
class HardwareSampler:
    interval_ms: int = 200
    samples: list[HardwareSample] = field(default_factory=list)
    process: subprocess.Popen[str] | None = None
    thread: threading.Thread | None = None
    error: str | None = None

    def start(self) -> None:
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
            "-lms",
            str(self.interval_ms),
        ]
        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except (FileNotFoundError, OSError) as error:
            self.error = f"{type(error).__name__}: {error}"
            return
        self.thread = threading.Thread(target=self._consume, daemon=True)
        self.thread.start()

    def _consume(self) -> None:
        assert self.process and self.process.stdout
        reader = csv.reader(self.process.stdout)
        for row in reader:
            if len(row) != 5:
                continue
            try:
                self.samples.append(
                    HardwareSample(
                        monotonic_s=time.perf_counter(),
                        gpu_util_pct=float(row[0]),
                        memory_controller_util_pct=float(row[1]),
                        memory_used_mib=float(row[2]),
                        power_w=float(row[3]),
                        temperature_c=float(row[4]),
                    )
                )
            except ValueError:
                continue

    def stop(self) -> None:
        if not self.process:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=3)
        if self.thread:
            self.thread.join(timeout=3)

    def window(self, start: float, end: float) -> dict[str, Any]:
        rows = [row for row in self.samples if start <= row.monotonic_s <= end]
        if not rows:
            return {"sample_count": 0, "sampler_error": self.error}
        return {
            "sample_count": len(rows),
            "gpu_util_mean_pct": statistics.fmean(row.gpu_util_pct for row in rows),
            "gpu_util_max_pct": max(row.gpu_util_pct for row in rows),
            "memory_controller_util_mean_pct": statistics.fmean(
                row.memory_controller_util_pct for row in rows
            ),
            "memory_controller_util_max_pct": max(
                row.memory_controller_util_pct for row in rows
            ),
            "memory_used_peak_mib": max(row.memory_used_mib for row in rows),
            "power_mean_w": statistics.fmean(row.power_w for row in rows),
            "power_max_w": max(row.power_w for row in rows),
            "temperature_max_c": max(row.temperature_c for row in rows),
        }


def _iter_sse_lines(response) -> Iterable[str]:
    while True:
        raw = response.readline()
        if not raw:
            break
        line = raw.decode("utf-8", "replace").strip()
        if line.startswith("data:"):
            yield line[5:].strip()


def _open_stream(url: str, body: dict[str, Any], timeout: float):
    request = urllib.request.Request(
        url,
        data=canonical_json(body),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=timeout)


def request_evaluation_api(
    case: dict[str, Any],
    base_url: str,
    timeout: float,
    tokenizer,
    sampler: HardwareSampler,
) -> dict[str, Any]:
    body = {
        "input_ids": case["input_ids"],
        "max_new_tokens": case["max_new_tokens"],
        "temperature": case["temperature"],
        "ignore_eos": case["ignore_eos"],
        "stream": True,
    }
    url = f"{base_url.rstrip('/')}/v1/evaluations/generate"
    start = time.perf_counter()
    first_token_at: float | None = None
    last_token_at: float | None = None
    token_times: list[float] = []
    output_ids: list[int] = []
    request_id: str | None = None
    done_usage: dict[str, Any] | None = None
    saw_done_sentinel = False
    status_code: int | None = None
    error: str | None = None
    try:
        with _open_stream(url, body, timeout) as response:
            status_code = response.status
            for data in _iter_sse_lines(response):
                if data == "[DONE]":
                    saw_done_sentinel = True
                    break
                event = json.loads(data)
                event_type = event.get("type")
                event_request_id = event.get("request_id")
                if not isinstance(event_request_id, str) or not event_request_id:
                    raise ValueError("SSE event has no valid request_id")
                if request_id is None:
                    request_id = event_request_id
                elif request_id != event_request_id:
                    raise ValueError("request_id changed within one SSE stream")
                if event_type == "token":
                    token_id = event.get("token_id")
                    index = event.get("index")
                    if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
                        raise ValueError("token event has invalid token_id")
                    if index != len(output_ids):
                        raise ValueError(
                            f"token index gap: expected {len(output_ids)}, got {index}"
                        )
                    now = time.perf_counter()
                    if first_token_at is None:
                        first_token_at = now
                    last_token_at = now
                    token_times.append(now)
                    output_ids.append(token_id)
                elif event_type == "done":
                    usage = event.get("usage")
                    if not isinstance(usage, dict):
                        raise ValueError("done event has no usage object")
                    done_usage = usage
                else:
                    raise ValueError(f"unknown SSE event type {event_type!r}")
    except urllib.error.HTTPError as http_error:
        status_code = http_error.code
        payload = http_error.read().decode("utf-8", "replace")
        error = f"HTTPError: {http_error.code}: {payload[:1000]}"
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    end = time.perf_counter()
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    functional_pass, validation_detail = validate_output(case, output_text)
    usage_valid = bool(
        done_usage
        and done_usage.get("prompt_tokens") == len(case["input_ids"])
        and done_usage.get("completion_tokens") == len(output_ids)
        and done_usage.get("total_tokens") == len(case["input_ids"]) + len(output_ids)
    )
    exact_budget = len(output_ids) == int(case["max_new_tokens"])
    if case["ignore_eos"] and not exact_budget:
        error = error or "incomplete output token budget"
    protocol_valid = bool(request_id and done_usage and saw_done_sentinel and usage_valid)
    if not protocol_valid:
        error = error or "incomplete or invalid SSE terminal protocol"
    success = status_code == 200 and error is None and bool(output_ids)
    return request_row(
        case,
        start,
        end,
        first_token_at,
        last_token_at,
        token_times,
        output_ids,
        output_text,
        status_code,
        error,
        success,
        success and functional_pass,
        validation_detail,
        done_usage or {},
        request_id,
        sampler,
    )


def request_vllm_completions(
    case: dict[str, Any],
    base_url: str,
    served_model_name: str,
    timeout: float,
    tokenizer,
    sampler: HardwareSampler,
) -> dict[str, Any]:
    body = {
        "model": served_model_name,
        "prompt": case["input_ids"],
        "max_tokens": case["max_new_tokens"],
        "temperature": case["temperature"],
        "stream": True,
        "ignore_eos": case["ignore_eos"],
        "return_token_ids": True,
        "stream_options": {"include_usage": True},
    }
    url = f"{base_url.rstrip('/')}/v1/completions"
    start = time.perf_counter()
    first_token_at: float | None = None
    last_token_at: float | None = None
    token_times: list[float] = []
    chunks: list[str] = []
    output_ids: list[int] = []
    usage: dict[str, Any] = {}
    status_code: int | None = None
    error: str | None = None
    request_id: str | None = None
    try:
        with _open_stream(url, body, timeout) as response:
            status_code = response.status
            for data in _iter_sse_lines(response):
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if isinstance(event.get("id"), str):
                    request_id = event["id"]
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                choices = event.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                choice = choices[0]
                prompt_token_ids = choice.get("prompt_token_ids")
                if prompt_token_ids is not None and prompt_token_ids != case["input_ids"]:
                    raise ValueError("vLLM returned prompt_token_ids that differ from the dataset")
                delta_token_ids = choice.get("token_ids")
                if not isinstance(delta_token_ids, list) or any(
                    isinstance(token_id, bool)
                    or not isinstance(token_id, int)
                    or token_id < 0
                    for token_id in delta_token_ids
                ):
                    raise ValueError("vLLM stream chunk has no valid token_ids")
                text = choice.get("text")
                if isinstance(text, str) and text:
                    chunks.append(text)
                if delta_token_ids:
                    now = time.perf_counter()
                    if first_token_at is None:
                        first_token_at = now
                    last_token_at = now
                    output_ids.extend(delta_token_ids)
                    token_times.extend([now] * len(delta_token_ids))
    except urllib.error.HTTPError as http_error:
        status_code = http_error.code
        payload = http_error.read().decode("utf-8", "replace")
        error = f"HTTPError: {http_error.code}: {payload[:1000]}"
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    end = time.perf_counter()
    output_text = "".join(chunks)
    functional_pass, validation_detail = validate_output(case, output_text)
    if case["ignore_eos"] and len(output_ids) != int(case["max_new_tokens"]):
        error = error or "incomplete output token budget"
    success = status_code == 200 and error is None and bool(output_ids)
    return request_row(
        case,
        start,
        end,
        first_token_at,
        last_token_at,
        token_times,
        output_ids,
        output_text,
        status_code,
        error,
        success,
        success and functional_pass,
        validation_detail,
        usage,
        request_id,
        sampler,
    )


def request_row(
    case: dict[str, Any],
    start: float,
    end: float,
    first_token_at: float | None,
    last_token_at: float | None,
    token_times: Sequence[float],
    output_ids: Sequence[int],
    output_text: str,
    status_code: int | None,
    error: str | None,
    success: bool,
    functional_pass: bool,
    validation_detail: str,
    usage: dict[str, Any],
    request_id: str | None,
    sampler: HardwareSampler,
) -> dict[str, Any]:
    completion_tokens = len(output_ids)
    ttft = first_token_at - start if first_token_at is not None else None
    e2e = end - start
    tpot = None
    decode_duration = None
    if first_token_at is not None and last_token_at is not None:
        decode_duration = last_token_at - first_token_at
        if completion_tokens > 1:
            tpot = decode_duration / (completion_tokens - 1)
    return {
        "timestamp_utc": utc_now(),
        "case_id": case["id"],
        "suite": case["suite"],
        "roles": case.get("roles", []),
        "category": case["category"],
        "input_ids_sha256": case["input_ids_sha256"],
        "prompt_tokens": len(case["input_ids"]),
        "requested_completion_tokens": case["max_new_tokens"],
        "completion_tokens": completion_tokens,
        "status_code": status_code,
        "success": success,
        "functional_pass": functional_pass,
        "validation_detail": validation_detail,
        "error": error,
        "request_id": request_id,
        "ttft_s": ttft,
        "tpot_s": tpot,
        "decode_duration_s": decode_duration,
        "e2e_s": e2e,
        "output_ids": list(output_ids),
        "output_ids_sha256": sha256_bytes(canonical_json(list(output_ids))),
        "output_text": output_text,
        "event_count": len(token_times),
        "usage": usage,
        "hardware": sampler.window(start, end),
    }


class Evaluator:
    def __init__(
        self,
        *,
        base_url: str,
        api_mode: str,
        served_model_name: str | None,
        timeout: float,
        tokenizer,
        sampler: HardwareSampler,
    ) -> None:
        self.base_url = base_url
        self.api_mode = api_mode
        self.served_model_name = served_model_name
        self.timeout = timeout
        self.tokenizer = tokenizer
        self.sampler = sampler

    def request(self, case: dict[str, Any]) -> dict[str, Any]:
        if self.api_mode == "evaluation":
            return request_evaluation_api(
                case,
                self.base_url,
                self.timeout,
                self.tokenizer,
                self.sampler,
            )
        if not self.served_model_name:
            raise ValueError("--served-model-name is required in vllm-completions mode")
        return request_vllm_completions(
            case,
            self.base_url,
            self.served_model_name,
            self.timeout,
            self.tokenizer,
            self.sampler,
        )


def protocol_checks(
    contract: dict[str, Any],
    base_url: str,
    health: dict[str, Any],
    small_case: dict[str, Any],
    timeout: float,
) -> tuple[bool, list[dict[str, Any]]]:
    endpoint = f"{base_url.rstrip('/')}/v1/evaluations/generate"
    max_model_len = health.get("max_model_len")
    if not isinstance(max_model_len, int) or max_model_len <= 0:
        max_model_len = 32768
    common = {
        "max_new_tokens": 1,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": False,
    }
    negative_cases = [
        ("empty_input_ids", {**common, "input_ids": []}),
        ("negative_token_id", {**common, "input_ids": [-1]}),
        ("out_of_vocabulary_token_id", {**common, "input_ids": [4294967295]}),
        (
            "unsupported_temperature",
            {**common, "input_ids": [small_case["input_ids"][0]], "temperature": 0.1},
        ),
        (
            "over_budget",
            {
                **common,
                "input_ids": [small_case["input_ids"][0]],
                "max_new_tokens": max_model_len,
            },
        ),
        (
            "unsupported_modality_field",
            {**common, "input_ids": [small_case["input_ids"][0]], "images": ["x"]},
        ),
    ]
    rows: list[dict[str, Any]] = []
    passed = True
    status, _, raw = http_json(
        "POST", endpoint, raw_body=b"{not-json", timeout=min(timeout, 30.0)
    )
    malformed_pass = status == 400
    rows.append(
        {
            "type": "protocol",
            "id": "malformed_json",
            "status_code": status,
            "passed": malformed_pass,
            "response": raw[:1000],
        }
    )
    passed &= malformed_pass
    for check_id, body in negative_cases:
        status, value, raw = http_json(
            "POST", endpoint, body=body, timeout=min(timeout, 30.0)
        )
        check_pass = status == 400 and isinstance(value, dict) and "error" in value
        rows.append(
            {
                "type": "protocol",
                "id": check_id,
                "status_code": status,
                "passed": check_pass,
                "response": value or raw[:1000],
            }
        )
        passed &= check_pass
    short_body = {
        **common,
        "input_ids": small_case["input_ids"][:8],
    }
    status, value, raw = http_json(
        "POST", endpoint, body=short_body, timeout=min(timeout, 120.0)
    )
    short_pass = bool(
        status == 200
        and isinstance(value, dict)
        and value.get("type") == "result"
        and isinstance(value.get("output_ids"), list)
        and len(value["output_ids"]) == 1
        and isinstance(value.get("usage"), dict)
        and value["usage"].get("prompt_tokens") == 8
        and value["usage"].get("completion_tokens") == 1
    )
    rows.append(
        {
            "type": "protocol",
            "id": "valid_short_nostream_request",
            "status_code": status,
            "passed": short_pass,
            "response": value or raw[:1000],
        }
    )
    passed &= short_pass
    recovered = False
    recovery_error: str | None = None
    try:
        after = health_check(base_url, timeout)
        recovered = after.get("status") == "ok"
    except Exception as exception:
        recovery_error = f"{type(exception).__name__}: {exception}"
    rows.append(
        {
            "type": "protocol",
            "id": "health_after_invalid_requests",
            "passed": recovered,
            "error": recovery_error,
        }
    )
    passed &= recovered
    expected_contract = contract["scenario"].get("interface_contract")
    if expected_contract and health.get("evaluation_contract") != expected_contract:
        passed = False
        rows.append(
            {
                "type": "protocol",
                "id": "health_contract_identity",
                "passed": False,
                "expected": expected_contract,
                "actual": health.get("evaluation_contract"),
            }
        )
    return bool(passed), rows


def run_functional(
    evaluator: Evaluator,
    cases: Sequence[dict[str, Any]],
    split: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        row = evaluator.request(case)
        row.update({"type": "request", "phase": "functional", "split": split})
        rows.append(row)
    return rows


def run_performance(
    evaluator: Evaluator,
    cases: Sequence[dict[str, Any]],
    warmups: int,
    repeats: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        for phase, count in (("warmup", warmups), ("measure", repeats)):
            for repeat in range(count):
                row = evaluator.request(case)
                row.update(
                    {
                        "type": "request",
                        "phase": phase,
                        "split": "public",
                        "repeat": repeat,
                    }
                )
                rows.append(row)
    return rows


def load_trajectory_reference(path: Path) -> dict[str, Any]:
    reference = load_json(path)
    if reference.get("schema") != TRAJECTORY_SCHEMA:
        raise ValueError(f"{path}: invalid trajectory-reference schema")
    if not isinstance(reference.get("cases"), dict):
        raise ValueError(f"{path}: trajectory reference has no cases object")
    return reference


def token_edit_distance(expected: Sequence[int], observed: Sequence[int]) -> int:
    """Unit-cost Levenshtein distance over categorical token IDs."""
    if len(expected) < len(observed):
        expected, observed = observed, expected
    previous = list(range(len(observed) + 1))
    for expected_index, expected_token in enumerate(expected, start=1):
        current = [expected_index]
        for observed_index, observed_token in enumerate(observed, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[observed_index] + 1,
                    previous[observed_index - 1]
                    + int(expected_token != observed_token),
                )
            )
        previous = current
    return previous[-1]


def trajectory_counts(
    rows: Sequence[dict[str, Any]],
    trajectory_case_ids: Sequence[str],
    reference: dict[str, Any],
) -> tuple[int, int, dict[str, Any]]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("phase") == "measure" and row.get("case_id") in trajectory_case_ids:
            by_case[row["case_id"]].append(row)
    passed = 0
    total = 0
    details: dict[str, Any] = {}
    for case_id in trajectory_case_ids:
        expected_entry = reference["cases"].get(case_id)
        observed_rows = by_case.get(case_id, [])
        if not isinstance(expected_entry, dict):
            raise ValueError(f"trajectory reference is missing frozen case {case_id}")
        expected = expected_entry.get("output_ids")
        if not isinstance(expected, list):
            raise ValueError(f"trajectory reference {case_id} has invalid output_ids")
        case_total = len(expected)
        total += case_total
        if not observed_rows:
            details[case_id] = {
                "passed": 0,
                "total": case_total,
                "reason": "missing observed trajectory",
            }
            continue
        observed = observed_rows[0]["output_ids"]
        distance = token_edit_distance(expected, observed)
        case_passed = max(0, case_total - distance)
        passed += case_passed
        details[case_id] = {
            "passed": case_passed,
            "total": case_total,
            "token_edit_distance": distance,
            "token_edit_similarity": case_passed / case_total if case_total else 1.0,
            "observed_tokens": len(observed),
            "expected_sha256": sha256_bytes(canonical_json(expected)),
            "observed_sha256": sha256_bytes(canonical_json(observed)),
        }
    return passed, total, details


def capture_trajectory_reference(
    contract: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    case_ids: Sequence[str],
    model_dir: Path,
) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    for case_id in case_ids:
        row = next(
            (
                item
                for item in rows
                if item.get("case_id") == case_id
                and item.get("phase") == "measure"
                and item.get("success")
            ),
            None,
        )
        if row is None:
            raise ValueError(f"cannot capture trajectory: no successful row for {case_id}")
        cases[case_id] = {
            "input_ids_sha256": row["input_ids_sha256"],
            "output_ids": row["output_ids"],
            "output_ids_sha256": row["output_ids_sha256"],
        }
    return {
        "schema": TRAJECTORY_SCHEMA,
        "model_repo_id": contract["model"]["repo_id"],
        "model_revision": contract["model"]["revision"],
        "tokenizer_config_sha256": sha256_file(model_dir / "tokenizer_config.json"),
        "cases": cases,
    }


def summarize_latency_cell(
    case: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    warmups: int,
) -> dict[str, Any]:
    measured = [
        row
        for row in rows
        if row.get("case_id") == case["id"] and row.get("phase") == "measure"
    ]
    successful = [row for row in measured if row.get("success")]

    def values(field: str) -> list[float]:
        return [float(row[field]) for row in successful if row.get(field) is not None]

    ttft = values("ttft_s")
    tpot = values("tpot_s")
    e2e = values("e2e_s")
    peak_vram = [
        float(row["hardware"]["memory_used_peak_mib"])
        for row in successful
        if row.get("hardware", {}).get("memory_used_peak_mib") is not None
    ]
    return {
        "actual_prompt_tokens": case["actual_prompt_tokens"],
        "completion_tokens": case["max_new_tokens"],
        "success_rate": len(successful) / len(measured) if measured else 0.0,
        "measured_repeats": len(measured),
        "warmup_repeats": warmups,
        "ttft_cv": coefficient_of_variation(ttft),
        "tpot_cv": coefficient_of_variation(tpot),
        "ttft_s": percentile(ttft, 50),
        "tpot_s": percentile(tpot, 50),
        "e2e_s": percentile(e2e, 50),
        "peak_vram_mib": max(peak_vram) if peak_vram else None,
    }


def run_context_staircase(
    evaluator: Evaluator,
    cases: Sequence[dict[str, Any]],
    base_url: str,
    timeout: float,
    recovery_case: dict[str, Any],
    full_validation_above_tokens: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_length[int(case["actual_prompt_tokens"])].append(case)
    rows: list[dict[str, Any]] = []
    max_verified = 0
    verified_count = 0
    verified_pass_rate: float | None = None
    first_failed: int | None = None
    failure_mode: str | None = None
    for length in sorted(by_length):
        group = by_length[length]
        early = next(
            (case for case in group if case["category"] == "retrieval_early"),
            group[0],
        )
        ordered = [early, *(case for case in group if case is not early)]
        if length <= full_validation_above_tokens:
            ordered = [early]
        length_rows: list[dict[str, Any]] = []
        for index, case in enumerate(ordered):
            row = evaluator.request(case)
            row.update(
                {
                    "type": "request",
                    "phase": "context",
                    "split": "public",
                    "context_probe_index": index,
                }
            )
            rows.append(row)
            length_rows.append(row)
            if index == 0 and not (row["success"] and row["functional_pass"]):
                break
        passed_count = sum(
            bool(row["success"] and row["functional_pass"]) for row in length_rows
        )
        if len(length_rows) == len(ordered) and passed_count == len(ordered):
            max_verified = length
            verified_count = len(ordered)
            verified_pass_rate = 1.0
            continue
        first_failed = length
        failure_row = next(
            (
                row
                for row in length_rows
                if not (row["success"] and row["functional_pass"])
            ),
            length_rows[-1],
        )
        failure_mode = failure_row.get("error") or "functional_failure"
        break

    healthy = False
    recovery_request_pass = False
    try:
        after_health = health_check(base_url, timeout)
        healthy = after_health.get("status") == "ok" or "adapter_health_body" in after_health
        recovery_row = evaluator.request(recovery_case)
        recovery_row.update(
            {
                "type": "request",
                "phase": "context_recovery",
                "split": "public",
            }
        )
        rows.append(recovery_row)
        recovery_request_pass = bool(
            recovery_row["success"] and recovery_row["functional_pass"]
        )
    except Exception as exception:
        rows.append(
            {
                "type": "context_recovery",
                "phase": "context_recovery",
                "success": False,
                "error": f"{type(exception).__name__}: {exception}",
            }
        )
    return (
        {
            "max_verified_prompt_tokens": max_verified,
            "verified_output_tokens": 128 if max_verified else 0,
            "verified_cases_at_max_context": verified_count,
            "pass_rate_at_max_context": verified_pass_rate,
            "first_failed_prompt_tokens": first_failed,
            "failure_mode": failure_mode,
            "service_healthy_after_failure": healthy and recovery_request_pass,
        },
        rows,
    )


def jain_fairness(values: Sequence[float]) -> float:
    if not values or any(value < 0 for value in values):
        return 0.0
    denominator = len(values) * sum(value * value for value in values)
    return (sum(values) ** 2 / denominator) if denominator else 0.0


def run_multi_cell(
    evaluator: Evaluator,
    case: dict[str, Any],
    definition: dict[str, Any],
    warmups: int,
    repeats: int,
    base_url: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    concurrency = int(definition["concurrency"])
    total_requests = int(definition["total_requests"])
    all_rows: list[dict[str, Any]] = []
    repeat_summaries: list[dict[str, Any]] = []
    for phase, count in (("warmup", warmups), ("measure", repeats)):
        for repeat in range(count):
            batch_start = time.perf_counter()
            batch_rows: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                pending = [executor.submit(evaluator.request, case) for _ in range(total_requests)]
                for future in as_completed(pending):
                    row = future.result()
                    row.update(
                        {
                            "type": "request",
                            "phase": f"multi_{phase}",
                            "split": "public",
                            "multi_cell_id": definition["id"],
                            "repeat": repeat,
                        }
                    )
                    batch_rows.append(row)
                    all_rows.append(row)
            makespan = time.perf_counter() - batch_start
            correct_tokens = sum(
                row["completion_tokens"]
                for row in batch_rows
                if row["success"] and row["functional_pass"]
            )
            per_request_rates = [
                row["completion_tokens"] / row["e2e_s"]
                for row in batch_rows
                if row["success"] and row["functional_pass"] and row["e2e_s"] > 0
            ]
            repeat_summaries.append(
                {
                    "phase": phase,
                    "repeat": repeat,
                    "makespan_s": makespan,
                    "goodput_tokens_per_s": correct_tokens / makespan if makespan else 0.0,
                    "success_rate": sum(bool(row["success"]) for row in batch_rows)
                    / len(batch_rows),
                    "correctness_rate": sum(
                        bool(row["success"] and row["functional_pass"])
                        for row in batch_rows
                    )
                    / len(batch_rows),
                    "p95_ttft_s": percentile(
                        [row["ttft_s"] for row in batch_rows if row["ttft_s"] is not None],
                        95,
                    ),
                    "p95_tpot_s": percentile(
                        [row["tpot_s"] for row in batch_rows if row["tpot_s"] is not None],
                        95,
                    ),
                    "jain_fairness_index": jain_fairness(per_request_rates),
                }
            )
    measured = [row for row in repeat_summaries if row["phase"] == "measure"]
    try:
        after_health = health_check(base_url, evaluator.timeout)
        healthy = after_health.get("status") == "ok" or "adapter_health_body" in after_health
    except Exception:
        after_health = {}
        healthy = False
    goodputs = [float(row["goodput_tokens_per_s"]) for row in measured]
    p95_ttft_values = [
        float(row["p95_ttft_s"])
        for row in measured
        if row.get("p95_ttft_s") is not None
    ]
    p95_tpot_values = [
        float(row["p95_tpot_s"])
        for row in measured
        if row.get("p95_tpot_s") is not None
    ]
    cell = {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "actual_prompt_tokens": case["actual_prompt_tokens"],
        "completion_tokens": case["max_new_tokens"],
        "success_rate": min(float(row["success_rate"]) for row in measured),
        "correctness_rate": min(float(row["correctness_rate"]) for row in measured),
        "measured_repeats": len(measured),
        "warmup_repeats": warmups,
        "goodput_tokens_per_s": percentile(goodputs, 50),
        "goodput_cv": coefficient_of_variation(goodputs),
        "p95_ttft_s": max(p95_ttft_values) if p95_ttft_values else None,
        "p95_tpot_s": max(p95_tpot_values) if p95_tpot_values else None,
        "jain_fairness_index": min(
            float(row["jain_fairness_index"]) for row in measured
        ),
        "no_fallback": evaluator.api_mode == "vllm-completions"
        or after_health.get("fallback_active") is False,
        "service_healthy_after_run": healthy,
    }
    all_rows.extend(
        {
            "type": "multi_repeat_summary",
            "multi_cell_id": definition["id"],
            **summary,
        }
        for summary in repeat_summaries
    )
    return cell, all_rows


def default_context() -> dict[str, Any]:
    return {
        "max_verified_prompt_tokens": 0,
        "verified_output_tokens": 0,
        "verified_cases_at_max_context": 0,
        "pass_rate_at_max_context": None,
        "first_failed_prompt_tokens": None,
        "failure_mode": None,
        "service_healthy_after_failure": True,
    }


def xid_events_since(start_epoch_s: float) -> dict[str, Any]:
    command = [
        "journalctl",
        "-k",
        "--since",
        f"@{int(start_epoch_s)}",
        "--no-pager",
        "-o",
        "cat",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as error:
        return {
            "available": False,
            "events": [],
            "error": f"{type(error).__name__}: {error}",
        }
    if completed.returncode != 0:
        return {
            "available": False,
            "events": [],
            "error": completed.stderr.strip() or f"journalctl exit {completed.returncode}",
        }
    events = [
        line
        for line in completed.stdout.splitlines()
        if "xid" in line.lower() and ("nvrm" in line.lower() or "nvidia" in line.lower())
    ]
    return {"available": True, "events": events, "error": None}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=here / "contract-v1.json")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--hidden-dataset", type=Path)
    parser.add_argument("--context-dataset", type=Path)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument(
        "--api-mode",
        choices=("evaluation", "vllm-completions"),
        default="evaluation",
    )
    parser.add_argument("--served-model-name")
    parser.add_argument(
        "--parallel-requests-capability",
        type=int,
        help="Teacher-declared vLLM adapter capacity; candidate services must advertise this in /health",
    )
    parser.add_argument("--implementation-name", required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument(
        "--profile",
        choices=("public_calibration", "midterm_leaderboard"),
        default="public_calibration",
    )
    parser.add_argument("--trajectory-reference", type=Path)
    parser.add_argument("--capture-trajectory-reference", type=Path)
    parser.add_argument("--run-context", action="store_true")
    parser.add_argument("--run-multi", action="store_true")
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--output-dir", type=Path, default=here / "runs")
    parser.add_argument(
        "--run-id",
        help="Stable artifact directory name; defaults to UTC timestamp plus backend",
    )
    parser.add_argument("--skip-protocol", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_started_epoch_s = time.time()
    contract = load_json(args.contract)
    profile = contract["score_profiles"][args.profile]
    measurement = profile["measurement"]
    warmups = (
        args.warmups
        if args.warmups is not None
        else int(measurement["required_warmup_repeats"])
    )
    repeats = (
        args.repeats
        if args.repeats is not None
        else int(measurement["minimum_measured_repeats_per_latency_cell"])
    )
    if warmups < 0 or repeats <= 0:
        raise ValueError("warmups must be non-negative and repeats must be positive")
    if args.parallel_requests_capability is not None and (
        args.api_mode != "vllm-completions" or args.parallel_requests_capability <= 0
    ):
        raise ValueError(
            "--parallel-requests-capability is a positive vLLM-adapter-only option"
        )

    public_manifest, public_cases = read_dataset(args.dataset)
    hidden_manifest: dict[str, Any] | None = None
    hidden_cases: list[dict[str, Any]] = []
    if args.hidden_dataset:
        hidden_manifest, hidden_cases = read_dataset(args.hidden_dataset)
    context_manifest: dict[str, Any] | None = None
    context_cases: list[dict[str, Any]] = []
    if args.context_dataset:
        context_manifest, context_cases = read_dataset(args.context_dataset)
    if args.run_context and not context_cases:
        raise ValueError("--run-context requires --context-dataset")

    public_ids = set(
        contract["correctness_workload"]["public_functional_suite"]["case_ids"]
    )
    functional_public = [case for case in public_cases if case["id"] in public_ids]
    if {case["id"] for case in functional_public} != public_ids:
        raise ValueError("public dataset does not contain the frozen functional case set")
    performance_ids = {
        item["id"] for item in contract["performance_scoring"]["ttft_cells"]
    }
    performance_cases = [case for case in public_cases if case["id"] in performance_ids]
    if {case["id"] for case in performance_cases} != performance_ids:
        raise ValueError("public dataset does not contain every frozen performance cell")
    functional_hidden = [case for case in hidden_cases if case["suite"] == "functional"]
    hidden_trajectory_cases = [
        case
        for case in hidden_cases
        if "hidden_trajectory" in case.get("roles", [])
    ]

    tokenizer = load_tokenizer(args.model_dir)
    health = health_check(args.base_url, args.timeout)
    if args.api_mode == "evaluation":
        if health.get("model_revision") != contract["model"]["revision"]:
            raise ValueError("service model revision does not match the contract")
        capabilities = health.get("capabilities")
        if not isinstance(capabilities, dict) or not all(
            capabilities.get(field) is True
            for field in ("pretokenized_input_ids", "token_id_output")
        ):
            raise ValueError("service does not advertise the required exact-token capabilities")

    if args.run_id is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_id):
        raise ValueError("--run-id must contain only letters, digits, dot, underscore, or hyphen")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_dir / (args.run_id or f"{timestamp}-{args.backend}")
    run_dir.mkdir(parents=True, exist_ok=False)
    sampler = HardwareSampler()
    sampler.start()
    evaluator = Evaluator(
        base_url=args.base_url,
        api_mode=args.api_mode,
        served_model_name=args.served_model_name,
        timeout=args.timeout,
        tokenizer=tokenizer,
        sampler=sampler,
    )
    raw_rows: list[dict[str, Any]] = []
    try:
        if args.api_mode == "evaluation" and not args.skip_protocol:
            protocol_pass, protocol_rows = protocol_checks(
                contract,
                args.base_url,
                health,
                functional_public[0],
                args.timeout,
            )
            raw_rows.extend(protocol_rows)
        else:
            protocol_pass = True
            raw_rows.append(
                {
                    "type": "protocol",
                    "id": "adapter_or_explicit_skip",
                    "passed": True,
                    "api_mode": args.api_mode,
                }
            )

        public_functional_rows = run_functional(
            evaluator, functional_public, "public"
        )
        raw_rows.extend(public_functional_rows)
        hidden_functional_rows = run_functional(
            evaluator, functional_hidden, "hidden"
        )
        raw_rows.extend(hidden_functional_rows)
        performance_rows = run_performance(
            evaluator, performance_cases, warmups, repeats
        )
        raw_rows.extend(performance_rows)
        hidden_trajectory_rows = run_performance(
            evaluator, hidden_trajectory_cases, 0, 1
        )
        for row in hidden_trajectory_rows:
            row["split"] = "hidden"
        raw_rows.extend(hidden_trajectory_rows)
        all_trajectory_rows = [*performance_rows, *hidden_trajectory_rows]

        public_trajectory_ids = contract["correctness_workload"][
            "public_token_trajectory_suite"
        ]["case_ids"]
        hidden_trajectory_ids = [
            case["id"]
            for case in hidden_cases
            if "hidden_trajectory" in case.get("roles", [])
        ]
        capture: dict[str, Any] | None = None
        if args.capture_trajectory_reference:
            capture_ids = [*public_trajectory_ids, *hidden_trajectory_ids]
            capture = capture_trajectory_reference(
                contract,
                all_trajectory_rows,
                capture_ids,
                args.model_dir,
            )
            args.capture_trajectory_reference.parent.mkdir(parents=True, exist_ok=True)
            args.capture_trajectory_reference.write_text(
                json.dumps(capture, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            trajectory_reference = capture
        elif args.trajectory_reference:
            trajectory_reference = load_trajectory_reference(args.trajectory_reference)
        else:
            raise ValueError(
                "provide --trajectory-reference, or capture one from the vLLM control"
            )
        public_trajectory_passed, public_trajectory_total, public_trajectory_details = trajectory_counts(
            performance_rows,
            public_trajectory_ids,
            trajectory_reference,
        )
        hidden_trajectory_passed: int | None = None
        hidden_trajectory_total: int | None = None
        hidden_trajectory_details: dict[str, Any] = {}
        if hidden_trajectory_ids:
            hidden_trajectory_passed, hidden_trajectory_total, hidden_trajectory_details = trajectory_counts(
                hidden_trajectory_rows,
                hidden_trajectory_ids,
                trajectory_reference,
            )

        context_result = default_context()
        if args.run_context:
            context_result, context_rows = run_context_staircase(
                evaluator,
                context_cases,
                args.base_url,
                args.timeout,
                functional_public[0],
                int(contract["context_bonus"]["bonus_start_prompt_tokens"]),
            )
            raw_rows.extend(context_rows)

        multi_cells: dict[str, Any] = {}
        parallel_requests = (
            args.parallel_requests_capability
            if args.api_mode == "vllm-completions"
            and args.parallel_requests_capability is not None
            else int(health.get("parallel_requests", 1))
        )
        if args.run_multi and parallel_requests >= 2:
            base_case = next(
                case for case in public_cases if "multi" in case.get("roles", [])
            )
            for definition in contract["multi_request_bonus"]["cells"]:
                if parallel_requests < int(definition["concurrency"]):
                    continue
                cell, multi_rows = run_multi_cell(
                    evaluator,
                    base_case,
                    definition,
                    warmups,
                    repeats,
                    args.base_url,
                )
                multi_cells[definition["id"]] = cell
                raw_rows.extend(multi_rows)
        elif args.run_multi:
            raw_rows.append(
                {
                    "type": "multi_skip",
                    "reason": "health.parallel_requests_below_two",
                    "parallel_requests": parallel_requests,
                }
            )
    finally:
        sampler.stop()

    public_passed = sum(
        bool(row["success"] and row["functional_pass"])
        for row in public_functional_rows
    )
    hidden_passed = (
        sum(
            bool(row["success"] and row["functional_pass"])
            for row in hidden_functional_rows
        )
        if hidden_functional_rows
        else None
    )
    cells = {
        case["id"]: summarize_latency_cell(case, performance_rows, warmups)
        for case in performance_cases
    }
    base_scored_requests = [
        row
        for row in raw_rows
        if row.get("type") == "request"
        and row.get("phase")
        not in (
            "warmup",
            "context",
            "context_recovery",
            "multi_warmup",
            "multi_measure",
        )
    ]
    success_rate = (
        sum(bool(row.get("success")) for row in base_scored_requests)
        / len(base_scored_requests)
        if base_scored_requests
        else 0.0
    )
    unexpected_oom = any(
        "oom" in str(row.get("error", "")).lower()
        or "out of memory" in str(row.get("error", "")).lower()
        for row in base_scored_requests
    )
    nan_seen = any(
        re_token in str(row.get("output_text", "")).lower().split()
        for row in base_scored_requests
        for re_token in ("nan", "+nan", "-nan")
    )
    post_health_ok = False
    try:
        final_health = health_check(args.base_url, args.timeout)
        post_health_ok = final_health.get("status") == "ok" or "adapter_health_body" in final_health
    except Exception as exception:
        final_health = {"error": f"{type(exception).__name__}: {exception}"}
    xid_evidence = xid_events_since(run_started_epoch_s)
    no_fallback = (
        args.api_mode == "vllm-completions"
        or (
            health.get("fallback_active") is False
            and final_health.get("fallback_active") is False
        )
    ) and not any(
        "fallback" in str(row.get("error", "")).lower()
        or "fallback" in str(row.get("output_text", "")).lower()
        for row in base_scored_requests
    )

    raw_path = run_dir / "raw.jsonl"
    raw_payload = b"".join(canonical_json(row) + b"\n" for row in raw_rows)
    raw_path.write_bytes(raw_payload)
    environment = {
        "schema": "apxinf.qwen38_27b.run_environment.v1",
        "started_health": health,
        "finished_health": final_health,
        "base_url": args.base_url,
        "api_mode": args.api_mode,
        "profile": args.profile,
        "warmups": warmups,
        "repeats": repeats,
        "timeout_s": args.timeout,
        "sampler": {
            "sample_count": len(sampler.samples),
            "error": sampler.error,
        },
        "xid_evidence": xid_evidence,
    }
    environment_path = run_dir / "environment.json"
    environment_path.write_text(
        json.dumps(environment, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    submission = {
        "schema": SUBMISSION_SCHEMA,
        "implementation": {
            "name": args.implementation_name,
            "revision": args.implementation_revision,
            "backend": args.backend,
        },
        "correctness": {
            "protocol_pass": protocol_pass,
            "public_cases_passed": public_passed,
            "public_cases_total": len(functional_public),
            "hidden_cases_passed": hidden_passed,
            "hidden_cases_total": len(functional_hidden) if functional_hidden else None,
            "public_trajectory_tokens_passed": public_trajectory_passed,
            "public_trajectory_tokens_total": public_trajectory_total,
            "hidden_trajectory_tokens_passed": hidden_trajectory_passed,
            "hidden_trajectory_tokens_total": hidden_trajectory_total,
        },
        "cells": cells,
        "context": context_result,
        "multi_request": {"cells": multi_cells},
        "reliability": {
            "request_success_rate": success_rate,
            "no_unexpected_oom": not unexpected_oom,
            "no_nan": not nan_seen,
            "no_fallback": no_fallback,
            "no_xid": bool(xid_evidence["available"] and not xid_evidence["events"]),
            "service_healthy_after_failure": post_health_ok,
        },
        "evidence": {
            "run_id": run_dir.name,
            "contract_sha256": sha256_file(args.contract),
            "public_manifest_sha256": sha256_file(args.dataset / "manifest.json"),
            "hidden_manifest_sha256": sha256_file(args.hidden_dataset / "manifest.json")
            if args.hidden_dataset
            else None,
            "context_manifest_sha256": sha256_file(args.context_dataset / "manifest.json")
            if args.context_dataset
            else None,
            "raw_jsonl": raw_path.name,
            "raw_jsonl_sha256": sha256_bytes(raw_payload),
            "environment_json": environment_path.name,
            "environment_json_sha256": sha256_file(environment_path),
            "trajectory_reference_sha256": sha256_file(args.trajectory_reference)
            if args.trajectory_reference
            else (
                sha256_file(args.capture_trajectory_reference)
                if args.capture_trajectory_reference
                else None
            ),
            "trajectory_details": {
                "public": public_trajectory_details,
                "hidden": hidden_trajectory_details,
            },
        },
    }
    submission_path = run_dir / "submission.json"
    submission_path.write_text(
        json.dumps(submission, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = {
        "run_dir": str(run_dir),
        "submission": str(submission_path),
        "raw_sha256": submission["evidence"]["raw_jsonl_sha256"],
        "public_correctness": f"{public_passed}/{len(functional_public)}",
        "public_trajectory": f"{public_trajectory_passed}/{public_trajectory_total}",
        "request_success_rate": success_rate,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
