#!/usr/bin/env python3
"""Deterministic no-profiler baseline for the native Qwen2.5-Omni service."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "apxinf.qwen25_omni.service_benchmark.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def coefficient_of_variation(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    mean = statistics.fmean(values)
    return statistics.stdev(values) / mean if mean else None


@dataclass(frozen=True)
class Case:
    case_id: str
    prompt_tokens: int
    output_tokens: int
    family: str


@dataclass
class HardwareSample:
    monotonic_s: float
    gpu_util_pct: float
    memory_util_pct: float
    memory_used_mib: float
    memory_total_mib: float
    power_w: float
    sm_clock_mhz: float
    memory_clock_mhz: float


@dataclass
class HardwareSampler:
    interval_ms: int
    samples: list[HardwareSample] = field(default_factory=list)
    process: subprocess.Popen[str] | None = None
    thread: threading.Thread | None = None

    fields = (
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.total",
        "power.draw",
        "clocks.current.sm",
        "clocks.current.memory",
    )

    def start(self) -> None:
        self.process = subprocess.Popen(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(self.fields)}",
                "--format=csv,noheader,nounits",
                "-lms",
                str(self.interval_ms),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.thread = threading.Thread(target=self._consume, daemon=True)
        self.thread.start()

    def _consume(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for row in csv.reader(self.process.stdout):
            if len(row) != len(self.fields):
                continue
            try:
                values = [float(item.strip()) for item in row]
            except ValueError:
                continue
            self.samples.append(
                HardwareSample(
                    monotonic_s=time.perf_counter(),
                    gpu_util_pct=values[0],
                    memory_util_pct=values[1],
                    memory_used_mib=values[2],
                    memory_total_mib=values[3],
                    power_w=values[4],
                    sm_clock_mhz=values[5],
                    memory_clock_mhz=values[6],
                )
            )

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=3)
        if self.thread is not None:
            self.thread.join(timeout=3)

    def window(self, start: float, end: float) -> dict[str, Any]:
        rows = [sample for sample in self.samples if start <= sample.monotonic_s <= end]
        if not rows:
            return {"sample_count": 0}
        peak_memory = max(row.memory_used_mib for row in rows)
        total_memory = rows[0].memory_total_mib
        return {
            "sample_count": len(rows),
            "gpu_util_mean_pct": statistics.fmean(row.gpu_util_pct for row in rows),
            "gpu_util_max_pct": max(row.gpu_util_pct for row in rows),
            "memory_util_mean_pct": statistics.fmean(
                row.memory_util_pct for row in rows
            ),
            "memory_util_max_pct": max(row.memory_util_pct for row in rows),
            "memory_used_peak_mib": peak_memory,
            "memory_total_mib": total_memory,
            "memory_headroom_min_mib": total_memory - peak_memory,
            "power_mean_w": statistics.fmean(row.power_w for row in rows),
            "power_max_w": max(row.power_w for row in rows),
            "sm_clock_mean_mhz": statistics.fmean(row.sm_clock_mhz for row in rows),
            "memory_clock_mean_mhz": statistics.fmean(
                row.memory_clock_mhz for row in rows
            ),
        }


def request_json(url: str, body: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    data = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error


def prompt_ids(length: int, seed_token: int) -> list[int]:
    return [seed_token + index % 17 for index in range(length)]


def trajectory_sha256(tokens: list[int]) -> str:
    payload = json.dumps(tokens, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def run_request(
    base_url: str,
    case: Case,
    token_id: int,
    timeout: float,
    sampler: HardwareSampler,
) -> dict[str, Any]:
    body = {
        "input_ids": prompt_ids(case.prompt_tokens, token_id),
        "max_new_tokens": case.output_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": False,
    }
    start = time.perf_counter()
    try:
        response = request_json(
            f"{base_url.rstrip('/')}/v1/evaluations/generate", body, timeout
        )
        end = time.perf_counter()
        output_ids = response.get("output_ids")
        if not isinstance(output_ids, list) or len(output_ids) != case.output_tokens:
            raise RuntimeError(
                f"output token count {len(output_ids) if isinstance(output_ids, list) else None} "
                f"!= {case.output_tokens}"
            )
        ttft = float(response["ttft_seconds"])
        tpot = float(response["tpot_seconds"])
        wall = end - start
        return {
            "passed": True,
            "error": None,
            "wall_seconds": wall,
            "ttft_seconds": ttft,
            "tpot_seconds": tpot,
            "prefill_tokens_per_second_proxy": case.prompt_tokens / ttft,
            "decode_tokens_per_second": 1.0 / tpot if tpot > 0 else None,
            "e2e_output_tokens_per_second": case.output_tokens / wall,
            "output_token_count": len(output_ids),
            "trajectory_sha256": trajectory_sha256(output_ids),
            "output_head": output_ids[:8],
            "fallback_active": response.get("fallback_active"),
            "hardware": sampler.window(start, end),
        }
    except Exception as error:
        end = time.perf_counter()
        return {
            "passed": False,
            "error": str(error),
            "wall_seconds": end - start,
            "hardware": sampler.window(start, end),
        }


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
        values = [float(trial[field]) for trial in passed if trial.get(field) is not None]
        summary[field] = {
            "mean": statistics.fmean(values) if values else None,
            "p50": percentile(values, 50),
            "p90": percentile(values, 90),
            "cv": coefficient_of_variation(values),
        }
    trajectories = sorted(
        {str(trial["trajectory_sha256"]) for trial in passed if "trajectory_sha256" in trial}
    )
    summary["trajectory_sha256s"] = trajectories
    summary["trajectory_stable"] = len(trajectories) == 1 and bool(trajectories)
    return summary


def parse_lengths(raw: str) -> list[int]:
    values = [int(item) for item in raw.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("lengths must be positive comma-separated integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--suite", choices=("quick", "context", "decode", "all"), default="quick")
    parser.add_argument(
        "--lengths",
        type=parse_lengths,
        default=parse_lengths("1024,2048,4096,8192,12288,16384,24576,32760"),
    )
    parser.add_argument("--context-output-tokens", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--token-id", type=int, default=1000)
    parser.add_argument("--sample-interval-ms", type=int, default=100)
    parser.add_argument("--continue-context-on-failure", action="store_true")
    parser.add_argument("--output", type=Path)
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
    health = request_json(f"{args.base_url.rstrip('/')}/health", None, args.timeout)
    max_model_len = int(health["max_model_len"])
    cases = cases_for(args)
    for case in cases:
        if case.prompt_tokens + case.output_tokens > max_model_len:
            raise SystemExit(
                f"{case.case_id}: prompt {case.prompt_tokens} + output {case.output_tokens} "
                f"> max_model_len {max_model_len}"
            )

    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = Path(__file__).resolve().parent / "results" / f"{stamp}-{args.suite}.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    sampler = HardwareSampler(args.sample_interval_ms)
    sampler.start()
    started_at = utc_now()
    raw_cases: list[dict[str, Any]] = []
    try:
        for case in cases:
            for _ in range(args.warmups):
                warmup = run_request(args.base_url, case, args.token_id, args.timeout, sampler)
                if not warmup["passed"]:
                    raw_cases.append({"case": asdict(case), "warmup_failure": warmup, "trials": []})
                    break
            else:
                trials = [
                    run_request(args.base_url, case, args.token_id, args.timeout, sampler)
                    for _ in range(args.repeats)
                ]
                raw_cases.append({"case": asdict(case), "warmup_failure": None, "trials": trials})
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
    result = {
        "schema": SCHEMA,
        "started_at": started_at,
        "completed_at": utc_now(),
        "base_url": args.base_url,
        "suite": args.suite,
        "contract": {
            "api": "/v1/evaluations/generate",
            "single_request": True,
            "sampling": "greedy",
            "temperature": 0,
            "ignore_eos": True,
            "stream": False,
            "token_pattern": f"{args.token_id} + index % 17",
            "warmups": args.warmups,
            "repeats": args.repeats,
            "timeout_seconds": args.timeout,
            "sample_interval_ms": args.sample_interval_ms,
        },
        "health": health,
        "raw_cases": raw_cases,
        "summaries": summaries,
        "passed": all(
            item["warmup_failure"] is None
            and item["trials"]
            and all(trial["passed"] for trial in item["trials"])
            for item in raw_cases
        ),
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "passed": result["passed"], "summaries": summaries}, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
