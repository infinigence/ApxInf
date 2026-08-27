#!/usr/bin/env python3
"""Single-request external-engine baseline using an OpenAI streaming API."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark_service import (
    Case,
    HardwareSampler,
    coefficient_of_variation,
    percentile,
    utc_now,
)


VLLM_SCHEMA = "apxinf.qwen25_omni.vllm_omni_benchmark.v1"
EXTERNAL_SCHEMA = "apxinf.qwen25_omni.external_engine_benchmark.v1"
CHAT_TEMPLATE_OVERHEAD = 20


def get_json(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def health_status(base_url: str, timeout: float) -> int:
    with urllib.request.urlopen(
        f"{base_url.rstrip('/')}/health", timeout=timeout
    ) as response:
        return response.status


def engine_version(
    base_url: str,
    version_path: str,
    version_key: str | None,
    engine_name: str,
    timeout: float,
) -> tuple[str, Any]:
    normalized_path = "/" + version_path.lstrip("/")
    response = get_json(f"{base_url.rstrip('/')}{normalized_path}", timeout)
    if version_key is None:
        return normalized_path, response
    if version_key not in response:
        raise RuntimeError(f"{engine_name} version response omitted {version_key}")
    return normalized_path, {version_key: response[version_key]}


def exact_text_prompt(prompt_tokens: int) -> str:
    """Build a prompt whose Qwen2.5-Omni chat-template length is exact.

    The pinned processor maps ``"x " * n`` to ``n + 20`` prompt tokens,
    including the default system/user/assistant template. Every response's
    usage record is still checked; a tokenizer/template drift fails closed.
    """
    repeats = prompt_tokens - CHAT_TEMPLATE_OVERHEAD
    if repeats < 1:
        raise ValueError(
            f"prompt_tokens must be at least {CHAT_TEMPLATE_OVERHEAD + 1}"
        )
    return "x " * repeats


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stream_openai_chat(
    base_url: str,
    body: dict[str, Any],
    expected_prompt_tokens: int,
    expected_output_tokens: int,
    timeout: float,
    sampler: HardwareSampler,
) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    token_times: list[float] = []
    output_parts: list[str] = []
    usage: dict[str, Any] | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                event = json.loads(data)
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                choices = event.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                delta = choices[0].get("delta")
                if not isinstance(delta, dict) or "content" not in delta:
                    continue
                # The initial role event carries an empty content string but
                # is not a generated token. Later content events are one token
                # each, including any token whose decoded delta is empty.
                if "role" in delta:
                    continue
                token_times.append(time.perf_counter())
                content = delta.get("content")
                if isinstance(content, str):
                    output_parts.append(content)
        ended = time.perf_counter()
        if usage is None:
            raise RuntimeError("stream ended without a usage record")
        prompt_count = usage.get("prompt_tokens")
        completion_count = usage.get("completion_tokens")
        if prompt_count != expected_prompt_tokens:
            raise RuntimeError(
                f"prompt token count {prompt_count} != {expected_prompt_tokens}"
            )
        if completion_count != expected_output_tokens:
            raise RuntimeError(
                f"completion token count {completion_count} != {expected_output_tokens}"
            )
        if len(token_times) != expected_output_tokens:
            raise RuntimeError(
                f"stream token event count {len(token_times)} != {expected_output_tokens}"
            )
        ttft = token_times[0] - started
        tpot = (
            (token_times[-1] - token_times[0]) / (expected_output_tokens - 1)
            if expected_output_tokens > 1
            else 0.0
        )
        wall = ended - started
        output_text = "".join(output_parts)
        return {
            "passed": True,
            "error": None,
            "wall_seconds": wall,
            "ttft_seconds": ttft,
            "tpot_seconds": tpot,
            "prefill_tokens_per_second_proxy": expected_prompt_tokens / ttft,
            "decode_tokens_per_second": 1.0 / tpot if tpot > 0 else None,
            "e2e_output_tokens_per_second": expected_output_tokens / wall,
            "prompt_token_count": prompt_count,
            "output_token_count": completion_count,
            "stream_token_event_count": len(token_times),
            "trajectory_sha256": text_sha256(output_text),
            "output_text": output_text,
            "hardware": sampler.window(started, ended),
        }
    except urllib.error.HTTPError as error:
        ended = time.perf_counter()
        detail = error.read().decode("utf-8", errors="replace")
        return {
            "passed": False,
            "error": f"HTTP {error.code}: {detail}",
            "wall_seconds": ended - started,
            "hardware": sampler.window(started, ended),
        }
    except Exception as error:
        ended = time.perf_counter()
        return {
            "passed": False,
            "error": str(error),
            "wall_seconds": ended - started,
            "hardware": sampler.window(started, ended),
        }


def stream_chat(
    base_url: str,
    model: str,
    case: Case,
    timeout: float,
    sampler: HardwareSampler,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": exact_text_prompt(case.prompt_tokens)}
                ],
            }
        ],
        "modalities": ["text"],
        "max_tokens": case.output_tokens,
        "temperature": 0,
        "top_p": 1,
        "seed": 42,
        "repetition_penalty": 1.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    return stream_openai_chat(
        base_url,
        body,
        case.prompt_tokens,
        case.output_tokens,
        timeout,
        sampler,
    )


def summarize(case: Case, trials: list[dict[str, Any]]) -> dict[str, Any]:
    passed = [trial for trial in trials if trial["passed"]]
    summary: dict[str, Any] = {
        **asdict(case),
        "trial_count": len(trials),
        "passed_trials": len(passed),
        "failed_trials": len(trials) - len(passed),
    }
    for field in (
        "wall_seconds",
        "ttft_seconds",
        "tpot_seconds",
        "prefill_tokens_per_second_proxy",
        "decode_tokens_per_second",
        "e2e_output_tokens_per_second",
    ):
        values = [
            float(trial[field])
            for trial in passed
            if trial.get(field) is not None
        ]
        summary[field] = {
            "mean": statistics.fmean(values) if values else None,
            "p50": percentile(values, 50),
            "p90": percentile(values, 90),
            "cv": coefficient_of_variation(values),
        }
    trajectories = sorted(
        {
            str(trial["trajectory_sha256"])
            for trial in passed
            if "trajectory_sha256" in trial
        }
    )
    summary["trajectory_sha256s"] = trajectories
    summary["trajectory_stable"] = len(trajectories) == 1 and bool(trajectories)
    return summary


def parse_lengths(raw: str) -> list[int]:
    values = [int(item) for item in raw.split(",") if item.strip()]
    if not values or any(value <= CHAT_TEMPLATE_OVERHEAD for value in values):
        raise argparse.ArgumentTypeError(
            f"lengths must exceed {CHAT_TEMPLATE_OVERHEAD}"
        )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8003")
    parser.add_argument("--model")
    parser.add_argument("--engine-name", default="vLLM-Omni")
    parser.add_argument("--version-path", default="/version")
    parser.add_argument("--version-key")
    parser.add_argument(
        "--suite", choices=("quick", "context", "decode", "all"), default="quick"
    )
    parser.add_argument(
        "--lengths",
        type=parse_lengths,
        default=parse_lengths("1024,2048,4096,8192,12288,16384,24576,32760"),
    )
    parser.add_argument("--context-output-tokens", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--sample-interval-ms", type=int, default=100)
    parser.add_argument("--continue-context-on-failure", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def cases_for(args: argparse.Namespace) -> list[Case]:
    cases: list[Case] = []
    if args.suite in ("quick", "all"):
        cases.append(Case("prefill_1k_decode_32", 1024, 32, "fixed"))
    if args.suite in ("decode", "all"):
        cases.append(Case("decode_128", 128, 128, "fixed"))
    if args.suite in ("context", "all"):
        cases.extend(
            Case(f"context_{length}", length, args.context_output_tokens, "context")
            for length in args.lengths
        )
    return cases


def main() -> int:
    args = parse_args()
    if args.warmups < 0 or args.repeats < 1:
        raise SystemExit("warmups must be nonnegative and repeats must be positive")
    base_url = args.base_url.rstrip("/")
    if health_status(base_url, args.timeout) != 200:
        raise SystemExit(f"{args.engine_name} health endpoint is not ready")
    try:
        version_path, version = engine_version(
            base_url,
            args.version_path,
            args.version_key,
            args.engine_name,
            args.timeout,
        )
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    models = get_json(f"{base_url}/v1/models", args.timeout)
    model = args.model
    if model is None:
        rows = models.get("data")
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
            raise SystemExit(f"{args.engine_name} returned no served model")
        model = rows[0].get("id")
    if not isinstance(model, str) or not model:
        raise SystemExit("served model id is empty")
    max_model_len = int(models["data"][0]["max_model_len"])
    cases = cases_for(args)
    for case in cases:
        if case.prompt_tokens + case.output_tokens > max_model_len:
            raise SystemExit(
                f"{case.case_id}: prompt {case.prompt_tokens} + output "
                f"{case.output_tokens} > max_model_len {max_model_len}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sampler = HardwareSampler(args.sample_interval_ms)
    sampler.start()
    started_at = utc_now()
    raw_cases: list[dict[str, Any]] = []
    try:
        for case in cases:
            for _ in range(args.warmups):
                warmup = stream_chat(base_url, model, case, args.timeout, sampler)
                if not warmup["passed"]:
                    raw_cases.append(
                        {"case": asdict(case), "warmup_failure": warmup, "trials": []}
                    )
                    break
            else:
                trials = [
                    stream_chat(base_url, model, case, args.timeout, sampler)
                    for _ in range(args.repeats)
                ]
                raw_cases.append(
                    {"case": asdict(case), "warmup_failure": None, "trials": trials}
                )
                if (
                    case.family == "context"
                    and not args.continue_context_on_failure
                    and not all(trial["passed"] for trial in trials)
                ):
                    break
                continue
            if case.family == "context" and not args.continue_context_on_failure:
                break
    finally:
        sampler.stop()

    summaries = [
        summarize(Case(**item["case"]), item["trials"])
        for item in raw_cases
        if item["trials"]
    ]
    report = {
        "schema": (
            VLLM_SCHEMA if args.engine_name == "vLLM-Omni" else EXTERNAL_SCHEMA
        ),
        "started_at": started_at,
        "completed_at": utc_now(),
        "engine": args.engine_name,
        "engine_version": version,
        "version_path": version_path,
        "base_url": base_url,
        "model": model,
        "models_response": models,
        "suite": args.suite,
        "contract": {
            "api": "/v1/chat/completions",
            "single_request": True,
            "sampling": "greedy",
            "temperature": 0,
            "top_p": 1,
            "seed": 42,
            "repetition_penalty": 1.0,
            "ignore_eos": True,
            "stream": True,
            "output_modalities": ["text"],
            "prompt_pattern": "Qwen2.5-Omni chat template over 'x ' repetitions",
            "chat_template_overhead": CHAT_TEMPLATE_OVERHEAD,
            "usage_token_count_must_match": True,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "timeout_seconds": args.timeout,
            "sample_interval_ms": args.sample_interval_ms,
        },
        "raw_cases": raw_cases,
        "summaries": summaries,
        "passed": all(
            item["warmup_failure"] is None
            and item["trials"]
            and all(trial["passed"] for trial in item["trials"])
            for item in raw_cases
        ),
        "limitations": [
            "TTFT/TPOT are client-observed from true SSE token events",
            "ApxInf comparison uses its server-emitted model timing because its current SSE response is buffered",
            "Prefill tokens/s remains a proxy because TTFT includes first-token and service overhead",
        ],
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"output": str(args.output), "passed": report["passed"], "summaries": summaries},
            ensure_ascii=False,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
