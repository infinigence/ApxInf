#!/usr/bin/env python3
"""Measure local serving requests with client timestamps and reported token counts."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
import fcntl
import hashlib
import http.client
import ipaddress
import json
import math
from pathlib import Path
import platform
import re
import socket
import statistics
import threading
import time
from urllib.parse import urlsplit


class ProtocolError(ValueError):
    pass


class DeadlineConnection(http.client.HTTPConnection):
    """Keep the owned socket available while HTTPResponse reads its body."""

    def connect(self):
        super().connect()
        self.deadline_socket = self.sock
        if getattr(self, "deadline_expired", None) is not None and self.deadline_expired.is_set():
            self.abort()
            raise TimeoutError("The request exceeded its absolute deadline.")

    def abort(self):
        owned = getattr(self, "deadline_socket", None)
        if owned is not None:
            with suppress(OSError):
                owned.shutdown(socket.SHUT_RDWR)


@contextmanager
def absolute_deadline(connection, timeout):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("The HTTP deadline must be finite and positive.")
    expires = time.perf_counter() + timeout
    expired = threading.Event()
    connection.deadline_expired = expired

    def interrupt():
        expired.set()
        getattr(connection, "abort", connection.close)()

    timer = threading.Timer(timeout, interrupt)
    timer.daemon = True
    timer.start()
    try:
        try:
            yield
        except Exception:
            if expired.is_set() or time.perf_counter() >= expires:
                raise TimeoutError("The request exceeded its absolute deadline.") from None
            raise
        if expired.is_set() or time.perf_counter() >= expires:
            raise TimeoutError("The request exceeded its absolute deadline.")
    finally:
        timer.cancel()
        timer.join()


def local_url(value: str):
    parsed = urlsplit(value)
    if parsed.scheme != "http" or parsed.username or parsed.password:
        raise ValueError("Use an HTTP loopback URL without credentials.")
    host = parsed.hostname
    if host != "localhost":
        try:
            if not ipaddress.ip_address(host or "").is_loopback:
                raise ValueError("Use a loopback host.")
        except ValueError as exc:
            raise ValueError("Use a loopback host.") from exc
    if parsed.query or parsed.fragment or parsed.path not in ("", "/"):
        raise ValueError("Use a base URL without an API path.")
    return parsed


def connect(base_url: str, timeout: float):
    parsed = local_url(base_url)
    return DeadlineConnection(parsed.hostname, parsed.port or 80, timeout=timeout)


def request_json(base_url: str, path: str, body=None, timeout: float = 120):
    connection = connect(base_url, timeout)
    headers = {"anthropic-version": "2023-06-01", "content-type": "application/json"}
    try:
        with absolute_deadline(connection, timeout):
            connection.request("POST" if body is not None else "GET", path,
                               None if body is None else json.dumps(body), headers)
            response = connection.getresponse()
            raw = response.read()
            return response.status, json.loads(raw)
    finally:
        connection.close()


def request_text(base_url: str, path: str, timeout: float = 5):
    connection = connect(base_url, timeout)
    try:
        with absolute_deadline(connection, timeout):
            connection.request("GET", path)
            response = connection.getresponse()
            text = response.read().decode("utf-8")
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status} from {path}")
            return text
    finally:
        connection.close()


def sse_events(response):
    """Decode complete SSE frames without treating network chunks as events."""
    event = "message"
    data = []
    frame_bytes = 0
    while True:
        raw = response.readline(1024 * 1024 + 1)
        if not raw:
            if data:
                raise ProtocolError("SSE ended inside a frame.")
            return
        frame_bytes += len(raw)
        if frame_bytes > 4 * 1024 * 1024 or len(raw) > 1024 * 1024:
            raise ProtocolError("SSE frame exceeds the client limit.")
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield event, "\n".join(data)
            event, data, frame_bytes = "message", [], 0
        elif line.startswith("event:"):
            event = line[6:].lstrip(" ")
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))


class StreamResult:
    """Track wire order, visible output events, usage, and complete termination."""

    def __init__(self, api: str):
        self.api = api
        self.started = False
        self.finished = False
        self.stop_reason = None
        self.blocks = {}
        self.closed_blocks = set()
        self.content_times = []
        self.first_event_time = None
        self.usage = {}
        self.text = ""
        self.tool_calls = []

    def feed(self, event: str, raw: str, elapsed: float):
        if self.finished:
            raise ProtocolError("The stream contains events after its terminal event.")
        if self.first_event_time is None:
            self.first_event_time = elapsed
        if self.api == "openai" and raw == "[DONE]":
            if self.stop_reason is None:
                raise ProtocolError("The OpenAI stream has no finish reason.")
            self.finished = True
            return
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ProtocolError("An SSE payload must be an object.")
        if obj.get("type") == "error" or "error" in obj:
            raise ProtocolError(f"The server returned a stream error: {obj.get('error')}")
        if self.api == "openai":
            self._openai(obj, elapsed)
        else:
            if event != obj.get("type"):
                raise ProtocolError("The SSE event name differs from its payload type.")
            self._anthropic(obj, elapsed)

    def _anthropic(self, obj, elapsed):
        kind = obj["type"]
        if kind == "ping":
            return
        if kind == "message_start":
            if self.started:
                raise ProtocolError("Duplicate message_start.")
            self.started = True
            self.usage.update(obj["message"].get("usage", {}))
        elif kind in ("content_block_start", "content_block_delta", "content_block_stop"):
            if not self.started:
                raise ProtocolError("A content event precedes message_start.")
            index = obj["index"]
            if type(index) is not int or index < 0:
                raise ProtocolError("Invalid content index.")
            if kind == "content_block_start":
                if index in self.blocks or index != len(self.blocks):
                    raise ProtocolError("Duplicate or non-contiguous content index.")
                self.blocks[index] = dict(obj["content_block"], json_fragments="")
            elif index not in self.blocks or index in self.closed_blocks:
                raise ProtocolError("An event refers to an absent or closed content block.")
            elif kind == "content_block_delta":
                delta = obj["delta"]
                text = delta.get("text", delta.get("thinking", delta.get("partial_json", "")))
                if text:
                    self.content_times.append(elapsed)
                if delta.get("type") == "text_delta":
                    self.text += text
                elif delta.get("type") == "input_json_delta":
                    self.blocks[index]["json_fragments"] += text
            else:
                self.closed_blocks.add(index)
                block = self.blocks[index]
                if block.get("type") == "tool_use":
                    value = json.loads(block["json_fragments"]) if block["json_fragments"] else block.get("input", {})
                    if not isinstance(value, dict):
                        raise ProtocolError("Tool input must be an object.")
                    self.tool_calls.append({"id": block["id"], "name": block["name"], "input": value})
        elif kind == "message_delta":
            if not self.started:
                raise ProtocolError("message_delta precedes message_start.")
            self.stop_reason = obj.get("delta", {}).get("stop_reason", self.stop_reason)
            self.usage.update(obj.get("usage", {}))
        elif kind == "message_stop":
            if not self.started or self.stop_reason is None or len(self.blocks) != len(self.closed_blocks):
                raise ProtocolError("message_stop precedes complete content and a stop reason.")
            self.finished = True

    def _openai(self, obj, elapsed):
        self.started = True
        if obj.get("usage"):
            self.usage.update(obj["usage"])
        for choice in obj.get("choices", []):
            if choice.get("index", 0) != 0:
                raise ProtocolError("The benchmark expects one output sequence.")
            delta = choice.get("delta", {})
            text = delta.get("content") or ""
            tools = delta.get("tool_calls") or []
            if text or any(call.get("function", {}).get("arguments") for call in tools):
                self.content_times.append(elapsed)
            self.text += text
            if choice.get("finish_reason") is not None:
                self.stop_reason = choice["finish_reason"]

    def require_finished(self):
        if not self.finished:
            raise ProtocolError("The stream ended without its terminal event.")


def stream_request(base_url, model, prompt, max_tokens=64, api="anthropic", timeout=120, index=0):
    path = "/v1/messages" if api == "anthropic" else "/v1/chat/completions"
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "stream": True, "temperature": 0}
    if api == "openai":
        body["stream_options"] = {"include_usage": True}
    result = {"index": index, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
              "success": False, "http_status": None, "error": None}
    connection = connect(base_url, timeout)
    state = StreamResult(api)
    start = time.perf_counter()
    result["start_monotonic_s"] = start
    try:
        with absolute_deadline(connection, timeout):
            connection.request("POST", path, json.dumps(body), {
                "content-type": "application/json", "anthropic-version": "2023-06-01"})
            response = connection.getresponse()
            result["http_status"] = response.status
            result["request_id"] = response.getheader("x-request-id")
            result["model_revision"] = response.getheader("x-apxinf-model-revision")
            result["service_profile"] = response.getheader("x-apxinf-profile")
            if response.status != 200:
                raise ProtocolError(f"HTTP {response.status}: {response.read(4096).decode('utf-8', 'replace')}")
            if "text/event-stream" not in response.getheader("Content-Type", ""):
                raise ProtocolError("Expected text/event-stream.")
            for event, raw in sse_events(response):
                state.feed(event, raw, time.perf_counter() - start)
            state.require_finished()
        result["success"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["e2e_s"] = time.perf_counter() - start
        connection.close()
    output_tokens = state.usage.get("output_tokens", state.usage.get("completion_tokens"))
    input_tokens = state.usage.get("input_tokens", state.usage.get("prompt_tokens"))
    if type(output_tokens) is not int or output_tokens < 0:
        output_tokens = None
    if type(input_tokens) is not int or input_tokens < 0:
        input_tokens = None
    ttft = state.content_times[0] if state.content_times else None
    result.update({"first_event_s": state.first_event_time, "ttft_s": ttft,
                   "output_event_times_s": state.content_times,
                   "inter_output_s": [b - a for a, b in zip(state.content_times, state.content_times[1:])],
                   "input_tokens": input_tokens, "output_tokens": output_tokens,
                   "stop_reason": state.stop_reason, "usage": state.usage,
                   "output_characters": len(state.text), "tool_calls": state.tool_calls,
                   "output_sha256": hashlib.sha256(state.text.encode()).hexdigest()})
    result["tpot_s"] = ((result["e2e_s"] - ttft) / (output_tokens - 1)
                         if ttft is not None and output_tokens is not None and output_tokens > 1 else None)
    return result


def distribution(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {"n": len(values), "mean": statistics.fmean(values), "p50": statistics.median(values),
            "p95": values[max(0, math.ceil(0.95 * len(values)) - 1)], "max": values[-1]}


def summarize(samples, wall_s, slo_ttft_s=None, slo_e2e_s=None):
    success = [sample for sample in samples if sample["success"]]
    counted = [sample for sample in success if sample["output_tokens"] is not None]
    targets = slo_ttft_s is not None or slo_e2e_s is not None
    good = [sample for sample in success
            if (slo_ttft_s is None or (sample["ttft_s"] is not None and sample["ttft_s"] <= slo_ttft_s))
            and (slo_e2e_s is None or sample["e2e_s"] <= slo_e2e_s)]
    result = {"attempts": len(samples), "completed": len(success), "errors": len(samples) - len(success),
            "error_rate": (len(samples) - len(success)) / len(samples) if samples else None,
            "wall_s": wall_s, "attempt_rate_per_s": len(samples) / wall_s,
            "completed_requests_per_s": len(success) / wall_s,
            "goodput_requests_per_s": len(good) / wall_s if targets else None,
            "slo_passing_requests": len(good) if targets else None,
            "usage_coverage": len(counted) / len(success) if success else None,
            "output_tokens_per_s": sum(sample["output_tokens"] for sample in counted) / wall_s if counted else None,
            "ttft_s": distribution([sample["ttft_s"] for sample in success]),
            "e2e_s": distribution([sample["e2e_s"] for sample in success]),
            "tpot_s": distribution([sample["tpot_s"] for sample in success]),
            "inter_output_s": distribution([gap for sample in success for gap in sample["inter_output_s"]])}
    if samples and "scheduled_monotonic_s" in samples[0]:
        arrival_good = [sample for sample in success
                        if (slo_ttft_s is None or (sample["arrival_ttft_s"] is not None
                                                  and sample["arrival_ttft_s"] <= slo_ttft_s))
                        and (slo_e2e_s is None or sample["arrival_to_completion_s"] <= slo_e2e_s)]
        result.update({
            "offered_requests": len(samples),
            "dispatched_requests": sum(sample["dispatch_monotonic_s"] is not None for sample in samples),
            "client_capacity_drops": sum(sample["client_capacity_drop"] for sample in samples),
            "dispatch_delay_s": distribution([sample["dispatch_delay_s"] for sample in samples]),
            "arrival_ttft_s": distribution([sample["arrival_ttft_s"] for sample in success]),
            "arrival_to_completion_s": distribution([sample["arrival_to_completion_s"] for sample in samples]),
            "arrival_slo_passing_requests": len(arrival_good) if targets else None,
            "arrival_goodput_requests_per_s": len(arrival_good) / wall_s if targets else None,
        })
    return result


def failed_arrival(index, error):
    return {"index": index, "success": False, "http_status": None, "error": error,
            "start_monotonic_s": None, "first_event_s": None, "ttft_s": None, "e2e_s": None,
            "inter_output_s": [], "output_event_times_s": [], "input_tokens": None,
            "output_tokens": None, "stop_reason": None, "usage": {}, "output_characters": 0,
            "tool_calls": [], "output_sha256": None, "tpot_s": None}


def open_loop_requests(request, samples, arrival_rate, max_in_flight, *,
                       clock=time.perf_counter, sleep=time.sleep, executor_factory=ThreadPoolExecutor):
    """Dispatch fixed arrivals without a waiting queue at client capacity."""
    results = []
    pending = set()
    start = clock()

    def execute(index, scheduled):
        dispatched = clock()
        try:
            result = request(index)
        except Exception as error:
            result = failed_arrival(index, f"{type(error).__name__}: {error}")
            result["start_monotonic_s"] = dispatched
            result["e2e_s"] = clock() - dispatched
        completed = clock()
        request_started = result.get("start_monotonic_s")
        if request_started is None:
            request_started = dispatched
        result.update({"scheduled_monotonic_s": scheduled, "scheduled_offset_s": index / arrival_rate,
                       "dispatch_monotonic_s": dispatched, "dispatch_delay_s": dispatched - scheduled,
                       "client_capacity_drop": False, "arrival_to_completion_s": completed - scheduled,
                       "arrival_ttft_s": (request_started - scheduled + result["ttft_s"]
                                          if result["ttft_s"] is not None else None)})
        return result

    with executor_factory(max_workers=max_in_flight) as executor:
        for index in range(samples):
            scheduled = start + index / arrival_rate
            while (remaining := scheduled - clock()) > 0:
                sleep(remaining)
            completed = {future for future in pending if future.done()}
            results.extend(future.result() for future in completed)
            pending.difference_update(completed)
            if len(pending) == max_in_flight:
                result = failed_arrival(index, "client_capacity_drop")
                result.update({"scheduled_monotonic_s": scheduled, "scheduled_offset_s": index / arrival_rate,
                               "dispatch_monotonic_s": None, "dispatch_delay_s": None,
                               "client_capacity_drop": True, "arrival_ttft_s": None,
                               "arrival_to_completion_s": clock() - scheduled})
                results.append(result)
            else:
                pending.add(executor.submit(execute, index, scheduled))
        results.extend(future.result() for future in pending)
    wall = max(clock() - start, 1e-9)
    return sorted(results, key=lambda sample: sample["index"]), wall


def load_settings(args):
    arrival_rate = getattr(args, "arrival_rate", None)
    max_in_flight = getattr(args, "max_in_flight", None)
    concurrency = getattr(args, "concurrency", None)
    if (arrival_rate is None) != (max_in_flight is None):
        raise ValueError("Supply both --arrival-rate and --max-in-flight.")
    if arrival_rate is not None:
        if not math.isfinite(arrival_rate) or arrival_rate <= 0:
            raise ValueError("The arrival rate must be finite and positive.")
        if type(max_in_flight) is not int or max_in_flight <= 0:
            raise ValueError("The in-flight limit must be a positive integer.")
        if concurrency is not None and concurrency != [max_in_flight]:
            raise ValueError("Open-loop concurrency must contain only the in-flight limit.")
        return "open_loop", [max_in_flight], arrival_rate, max_in_flight
    return "closed_loop", concurrency or [1, 2, 4], None, None


def prompt_for(index, workload):
    instruction = f"Request {index}. Explain why a local inference service needs a bounded queue. Use five short sentences."
    if workload == "short" or (workload == "mixed" and index % 4):
        return instruction
    context = "\n".join(f"Record {n}: the request enters a queue, receives a worker credit, and releases its state after completion."
                        for n in range(48))
    return context + "\n" + instruction


@contextmanager
def metal_measurement_lock(path):
    if path is None:
        yield
        return
    with open(path, "rb") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process owns the Metal measurement lock.") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def metrics_snapshot(base_url):
    try:
        text = request_text(base_url, "/metrics")
        return {"text": text, "values": metric_values(text), "error": None}
    except Exception as exc:
        return {"text": None, "values": {}, "error": str(exc)}


def metric_values(text):
    values = {}
    for line in text.splitlines():
        match = re.fullmatch(r'([a-zA-Z_:][a-zA-Z0-9_:]*(?:\{.*\})?)\s+([-+0-9.eE]+)', line)
        if match:
            try:
                values[match[1]] = float(match[2])
            except ValueError:
                pass
    return values


def summarize_metrics(samples):
    counters = {}
    for sample in samples:
        for name, value in sample.get("values", {}).items():
            counters.setdefault(name, []).append(value)
    return {name: {"first": values[0], "last": values[-1], "min": min(values), "max": max(values)}
            for name, values in counters.items()}


def run(args):
    local_url(args.base_url)
    load_mode, concurrency_levels, arrival_rate, max_in_flight = load_settings(args)
    if args.samples < 1 or args.warmup < 0 or any(value < 1 for value in concurrency_levels):
        raise ValueError("Sample counts and concurrency must be positive.")
    if arrival_rate is not None and not math.isfinite((args.samples - 1) / arrival_rate):
        raise ValueError("The arrival schedule exceeds the finite clock range.")
    if args.max_tokens < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("Output limits and timeouts must be positive.")
    if any(value is not None and (not math.isfinite(value) or value <= 0)
           for value in (args.slo_ttft_ms, args.slo_e2e_ms)):
        raise ValueError("Latency targets must be finite and positive.")
    status, model_document = request_json(args.base_url, "/v1/models", timeout=args.timeout)
    if status != 200:
        raise RuntimeError("The model discovery request failed.")
    model = args.model or model_document["data"][0]["id"]
    report = {"format": "apxinf-serving-benchmark-v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "host": platform.platform(), "model": model, "models": model_document,
              "api": args.api, "workload": args.workload, "max_tokens": args.max_tokens,
              "warmup_count": args.warmup, "sample_count_per_concurrency": args.samples,
              "slo_ttft_s": args.slo_ttft_ms / 1000 if args.slo_ttft_ms is not None else None,
              "slo_e2e_s": args.slo_e2e_ms / 1000 if args.slo_e2e_ms is not None else None,
              "load_mode": load_mode, "arrival_rate_per_s": arrival_rate, "max_in_flight": max_in_flight,
              "load_model": ("fixed arrivals with bounded client in-flight requests" if load_mode == "open_loop"
                             else "closed loop with fixed client concurrency"), "runs": [],
              "limitations": ["Warmup requests are excluded from measured samples.",
                              "TTFT measures the first nonempty text, thinking, or tool argument delta.",
                              "Inter-output latency measures SSE content events, not individual model tokens.",
                              "TPOT uses server token usage and includes the final response tail.",
                              "Output throughput excludes failed requests and missing token usage.",
                              "P95 uses nearest rank. Small sample sets do not establish tail SLOs.",
                              "Server memory counters do not measure whole-system physical memory.",
                              "The workload measures serving performance, not model task quality."]}
    with metal_measurement_lock(args.metal_lock):
        warmup = [stream_request(args.base_url, model, prompt_for(i, args.workload), args.max_tokens,
                                 args.api, args.timeout, i) for i in range(args.warmup)]
        report["warmup"] = warmup
        warmup_errors = sum(not sample["success"] for sample in warmup)
        report["warmup_summary"] = {"attempts": len(warmup), "completed": len(warmup) - warmup_errors,
                                    "errors": warmup_errors}
        for concurrency in concurrency_levels:
            memory_samples = []
            stop = threading.Event()
            def sample_metrics():
                while not stop.is_set():
                    memory_samples.append({"monotonic_s": time.perf_counter(), **metrics_snapshot(args.base_url)})
                    stop.wait(0.5)
            monitor = threading.Thread(target=sample_metrics, daemon=True)
            monitor.start()
            start = time.perf_counter()
            try:
                def request(index):
                    return stream_request(args.base_url, model, prompt_for(index, args.workload),
                                          args.max_tokens, args.api, args.timeout, index)
                if load_mode == "open_loop":
                    samples, wall = open_loop_requests(request, args.samples, arrival_rate, max_in_flight)
                else:
                    with ThreadPoolExecutor(max_workers=concurrency) as executor:
                        samples = list(executor.map(request, range(args.samples)))
                    wall = time.perf_counter() - start
            finally:
                stop.set()
                monitor.join(timeout=6)
            report["runs"].append({"concurrency": concurrency, "samples": samples,
                                   "metrics_samples": memory_samples,
                                   "metrics_summary": summarize_metrics(memory_samples),
                                   "summary": summarize(samples, wall, report["slo_ttft_s"], report["slo_e2e_s"])})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output.resolve()),
                      "warmup_summary": report["warmup_summary"],
                      "runs": [{"concurrency": item["concurrency"], **item["summary"]} for item in report["runs"]]}, indent=2))
    return int(bool(warmup_errors) or any(not sample["success"]
                                         for item in report["runs"] for sample in item["samples"]))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--base-url", default="http://127.0.0.1:8080")
    result.add_argument("--model")
    result.add_argument("--api", choices=("anthropic", "openai"), default="anthropic")
    result.add_argument("--samples", type=int, default=32)
    result.add_argument("--warmup", type=int, default=2)
    result.add_argument("--concurrency", type=int, nargs="+")
    result.add_argument("--arrival-rate", type=float)
    result.add_argument("--max-in-flight", type=int)
    result.add_argument("--workload", choices=("short", "long", "mixed"), default="mixed")
    result.add_argument("--max-tokens", type=int, default=64)
    result.add_argument("--timeout", type=float, default=120)
    result.add_argument("--slo-ttft-ms", type=float)
    result.add_argument("--slo-e2e-ms", type=float)
    result.add_argument("--metal-lock", type=Path)
    result.add_argument("--output", type=Path, required=True)
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
