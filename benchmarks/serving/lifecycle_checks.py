#!/usr/bin/env python3
"""Check local queue, cancellation, and explicitly authorized worker failure behavior."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import threading
import time
import uuid

from http_benchmark import connect, local_url, metrics_snapshot, request_json, sse_events, StreamResult


CANCELLED = 'apxinf_requests_total{status="cancelled"}'
IDLE = ("apxinf_active_requests", "apxinf_queued_requests", "apxinf_capacity_reserved_bytes")


def idle(snapshot):
    return not snapshot["error"] and all(snapshot["values"].get(key) == 0 for key in IDLE)


def metric_wait(base_url, predicate, timeout):
    start = time.perf_counter()
    samples = []
    while time.perf_counter() - start < timeout:
        sample = {"elapsed_s": time.perf_counter() - start, **metrics_snapshot(base_url)}
        samples.append(sample)
        if predicate(sample):
            return {"passed": True, "elapsed_s": time.perf_counter() - start, "samples": samples}
        time.sleep(.05)
    return {"passed": False, "elapsed_s": time.perf_counter() - start, "samples": samples}


class OwnedRequest:
    """Own one connection, reader thread, and the evidence from that connection."""

    def __init__(self, base_url, model, max_tokens, timeout):
        self.connection = connect(base_url, timeout)
        self.model = model
        self.max_tokens = max_tokens
        self.socket = None
        self.done = threading.Event()
        self.first_content = threading.Event()
        self.headers = threading.Event()
        self.closed_by_harness = False
        self.record = {"probe_id": str(uuid.uuid4()), "request_id": None, "http_status": None,
                       "error": None, "first_content_s": None, "terminal_received": False}
        self.thread = threading.Thread(target=self._read, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _read(self):
        start = time.perf_counter()
        state = StreamResult("anthropic")
        body = {"model": self.model, "max_tokens": self.max_tokens, "stream": True, "temperature": 0,
                "messages": [{"role": "user", "content": "List the integers from 1 to 2000. "
                              "Write one integer on each line. Continue until the list reaches 2000."}]}
        try:
            if self.closed_by_harness:
                return
            self.connection.request("POST", "/v1/messages", json.dumps(body), {
                "content-type": "application/json", "anthropic-version": "2023-06-01",
                "x-apxinf-probe-id": self.record["probe_id"]})
            self.socket = self.connection.sock
            if self.closed_by_harness:
                self.close()
                return
            response = self.connection.getresponse()
            self.record.update({"http_status": response.status,
                                "request_id": response.getheader("x-request-id"),
                                "service_profile": response.getheader("x-apxinf-profile")})
            self.headers.set()
            if response.status != 200:
                self.record["error_body"] = response.read(4096).decode("utf-8", "replace")
                return
            for event, raw in sse_events(response):
                state.feed(event, raw, time.perf_counter() - start)
                if state.content_times and not self.first_content.is_set():
                    self.record["first_content_s"] = state.content_times[0]
                    self.first_content.set()
            state.require_finished()
            self.record["terminal_received"] = True
        except Exception as error:
            self.record["error"] = f"{type(error).__name__}: {error}"
        finally:
            self.record.update({"elapsed_s": time.perf_counter() - start,
                                "closed_by_harness": self.closed_by_harness,
                                "observed_content_events": len(state.content_times)})
            self.connection.close()
            self.done.set()

    def close(self):
        self.closed_by_harness = True
        endpoint = self.socket or self.connection.sock
        if endpoint is not None:
            try:
                endpoint.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.connection.close()


def disconnect_check(args, model, queue_capacity=None):
    before = metrics_snapshot(args.base_url)
    result = {"check": "queue_overflow" if queue_capacity is not None else "disconnect_cancel",
              "metrics_before": before, "requests": [], "passed": False}
    if not idle(before):
        result["error"] = "The service must be idle before this check."
        return result
    requests = []
    try:
        active = OwnedRequest(args.base_url, model, args.max_tokens, args.timeout)
        requests.append(active)
        active.start()
        if not active.first_content.wait(args.timeout) or active.done.is_set():
            raise RuntimeError("The first request did not remain active after a content delta.")
        active_snapshot = metrics_snapshot(args.base_url)
        result["metrics_active"] = active_snapshot
        if active_snapshot["values"].get("apxinf_active_requests") != 1:
            raise RuntimeError("The service did not report one active request.")
        if queue_capacity is not None:
            result["expected_queue_capacity"] = queue_capacity
            for _ in range(queue_capacity):
                request = OwnedRequest(args.base_url, model, args.max_tokens, args.timeout)
                requests.append(request)
                request.start()
            filling = metric_wait(args.base_url, lambda sample:
                                  sample["values"].get("apxinf_queued_requests") == queue_capacity and
                                  sample["values"].get("apxinf_active_requests") == 1, args.timeout)
            result["queue_fill"] = filling
            if not filling["passed"] or active.done.is_set():
                raise RuntimeError("The queue did not reach its declared capacity behind one active request.")
            overflow = OwnedRequest(args.base_url, model, args.max_tokens, args.timeout)
            requests.append(overflow)
            overflow.start()
            if not overflow.done.wait(args.timeout):
                raise RuntimeError("The overflow request did not finish within the test deadline.")
            if overflow.record["http_status"] != 429:
                raise RuntimeError("The overflow request did not receive HTTP 429.")
            error = json.loads(overflow.record.get("error_body", "{}"))
            if error.get("error", {}).get("code") != "queue_full":
                raise RuntimeError("HTTP 429 did not identify queue_full.")
            result["overflow_observed"] = True
        result["protocol_assertions_passed"] = True
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        for request in requests:
            request.close()
        for request in requests:
            if request.thread.ident is not None:
                request.thread.join(timeout=2)
        result["requests"] = [request.record for request in requests]
        result["readers_stopped"] = all(not request.thread.is_alive() for request in requests)
        previous = before["values"].get(CANCELLED, 0)
        result["settlement"] = metric_wait(args.base_url,
            lambda sample: idle(sample) and sample["values"].get(CANCELLED, 0) > previous, args.timeout)
        try:
            status, readiness = request_json(args.base_url, "/readyz", timeout=5)
            result["readiness_after"] = {"http_status": status, "ready": readiness.get("ready")}
        except Exception as error:
            result["readiness_after"] = {"http_status": None, "error": str(error)}
        result["passed"] = bool(result.get("protocol_assertions_passed") and result["readers_stopped"] and
                                result["settlement"]["passed"] and
                                result["readiness_after"].get("http_status") == 200 and
                                result["readiness_after"].get("ready") is True)
    return result


def process_identity(pid):
    command = ["/bin/ps", "-p", str(pid), "-o", "pid=,ppid=,uid=,lstart=,args="]
    response = subprocess.run(command, check=True, text=True, capture_output=True, timeout=5)
    fields = response.stdout.strip().split(maxsplit=8)
    if len(fields) != 9:
        raise RuntimeError("The explicit process identity is unavailable.")
    identity = {"pid": int(fields[0]), "parent_pid": int(fields[1]), "uid": int(fields[2]),
                "started": " ".join(fields[3:8]), "command": fields[8]}
    if identity["pid"] != pid or identity["uid"] != os.getuid():
        raise RuntimeError("The explicit process does not belong to the current user.")
    return identity


def argument_value(arguments, name):
    if name not in arguments:
        return None
    index = arguments.index(name)
    return arguments[index + 1] if index + 1 < len(arguments) else None


def verify_worker_target(args, ready):
    if not args.allow_worker_termination or not args.service_pid or not args.worker_pid:
        raise RuntimeError("Worker termination needs its flag and both explicit PIDs.")
    if args.service_pid <= 1 or args.worker_pid <= 1 or args.worker_pid == args.service_pid:
        raise RuntimeError("The service and worker PIDs must be distinct process IDs greater than one.")
    service = process_identity(args.service_pid)
    worker = process_identity(args.worker_pid)
    service_command = shlex.split(service["command"])
    worker_command = shlex.split(worker["command"])
    if not service_command or Path(service_command[0]).name != "apxinf-serve":
        raise RuntimeError("The explicit service PID does not run apxinf-serve.")
    if worker["parent_pid"] != service["pid"]:
        raise RuntimeError("The explicit worker is not a child of the explicit service.")
    if not any(Path(part).name == "text_worker.py" for part in worker_command):
        raise RuntimeError("The explicit worker does not run text_worker.py.")
    if argument_value(worker_command, "--protocol") != "apxinf-worker/2.0":
        raise RuntimeError("The explicit worker protocol does not match.")
    epoch = ready.get("worker", {}).get("worker_epoch")
    model_path = ready.get("worker", {}).get("model_path")
    if not epoch or argument_value(worker_command, "--worker-epoch") != epoch:
        raise RuntimeError("The explicit worker epoch does not match readiness.")
    if not model_path or argument_value(worker_command, "--model") != model_path:
        raise RuntimeError("The explicit worker model does not match readiness.")
    port = local_url(args.base_url).port or 80
    listing = subprocess.run(["/usr/sbin/lsof", "-a", "-p", str(args.service_pid),
                              f"-iTCP:{port}", "-sTCP:LISTEN", "-Fp"],
                             check=True, capture_output=True, text=True, timeout=5)
    process_fields = [field for field in listing.stdout.splitlines() if field.startswith("p")]
    if process_fields != [f"p{args.service_pid}"]:
        raise RuntimeError("The explicit service PID does not own the endpoint listener.")
    return {"service": service, "worker": worker, "listener_port": port}


def worker_failure_check(args):
    result = {"check": "idle_worker_failure", "passed": False, "signal_sent": False}
    before = metrics_snapshot(args.base_url)
    result["metrics_before"] = before
    if not idle(before):
        result["error"] = "The service must be idle before worker termination."
        return result
    status, readiness = request_json(args.base_url, "/readyz", timeout=5)
    if status != 200:
        result["error"] = "The service must be ready before worker termination."
        return result
    try:
        identity = verify_worker_target(args, readiness)
        result["verified_target"] = identity
        if process_identity(args.worker_pid) != identity["worker"]:
            raise RuntimeError("The worker identity changed before the signal.")
        os.kill(args.worker_pid, signal.SIGTERM)
        result["signal_sent"] = True
        start = time.perf_counter()
        result["readiness_samples"] = []
        while time.perf_counter() - start < args.timeout:
            status, body = request_json(args.base_url, "/readyz", timeout=5)
            result["readiness_samples"].append({"elapsed_s": time.perf_counter() - start,
                                               "http_status": status, "ready": body.get("ready")})
            if status == 503 and body.get("ready") is False:
                break
            time.sleep(.05)
        health_status, health = request_json(args.base_url, "/healthz", timeout=5)
        result["health"] = {"http_status": health_status, "body": health}
        result["settlement"] = metric_wait(args.base_url, idle, args.timeout)
        result["service_identity_after"] = process_identity(args.service_pid)
        result["passed"] = (status == 503 and body.get("ready") is False and health_status == 200 and
                            result["settlement"]["passed"] and
                            result["service_identity_after"] == identity["service"])
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def run(args):
    local_url(args.base_url)
    if not 0 < args.timeout <= 60 or args.max_tokens < 1:
        raise ValueError("Use a timeout of at most 60 seconds and a positive output limit.")
    if "worker-failure" in args.checks and not (args.allow_worker_termination and args.worker_pid and args.service_pid):
        raise ValueError("Worker failure requires explicit authorization and both process IDs.")
    status, readiness = request_json(args.base_url, "/readyz", timeout=5)
    if status != 200:
        raise RuntimeError("The service is not ready.")
    capacity = args.queue_capacity or readiness.get("queue_capacity")
    if "queue" in args.checks and (type(capacity) is not int or not 1 <= capacity <= 64):
        raise ValueError("Supply the deployment queue capacity, from 1 to 64, for the overflow check.")
    model = args.model or readiness["model"]
    if args.max_tokens > readiness["worker"]["capabilities"]["max_output_tokens"]:
        raise ValueError("The requested output limit exceeds the deployment limit.")
    report = {"format": "apxinf-lifecycle-checks-v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "base_url": args.base_url, "model": model, "readiness_before": readiness, "checks": [],
              "limitations": ["Run without other clients and under the service operator's device lock.",
                              "Queued requests can close before the server exposes their request IDs.",
                              "Local probe IDs do not replace server request IDs.",
                              "Worker termination leaves the service unavailable and requires an operator restart.",
                              "PID checks narrow the target but cannot eliminate the final operating-system PID reuse race."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in ("cancel", "queue", "worker-failure"):
        if name not in args.checks:
            continue
        result = worker_failure_check(args) if name == "worker-failure" else disconnect_check(
            args, model, capacity if name == "queue" else None)
        report["checks"].append(result)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"check": result["check"], "passed": result["passed"], "error": result.get("error")}), flush=True)
        if not result["passed"]:
            break
    return 0 if report["checks"] and all(item["passed"] for item in report["checks"]) else 1


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--base-url", default="http://127.0.0.1:8080")
    result.add_argument("--model")
    result.add_argument("--checks", nargs="+", choices=("cancel", "queue", "worker-failure"), default=["cancel"])
    result.add_argument("--queue-capacity", type=int)
    result.add_argument("--max-tokens", type=int, default=512)
    result.add_argument("--timeout", type=float, default=15)
    result.add_argument("--service-pid", type=int)
    result.add_argument("--worker-pid", type=int)
    result.add_argument("--allow-worker-termination", action="store_true")
    result.add_argument("--output", type=Path, required=True)
    return result


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
