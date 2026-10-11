#!/usr/bin/env python3
"""Collect bounded production HTTP evidence without device sampling or service ownership."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import time
import uuid

import host_memory
import http_benchmark as http
from lifecycle_checks import argument_value, process_identity

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "python/apxinf/apxinf/serving/contracts.py"
SPEC = importlib.util.spec_from_file_location("production_memory_contracts", CONTRACT_PATH)
contracts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contracts)
CHECKS = {"identity", "execution", "input_band", "full_output", "settlement",
          "host_pressure", "external_memory", "runtime_peak"}
CORE_CHECKS = {"identity", "execution", "settlement", "host_pressure"}
GAUGES = ("apxinf_active_requests", "apxinf_queued_requests", "apxinf_capacity_reserved_bytes",
          'apxinf_cleanup_pending{operation="generate"}', 'apxinf_cleanup_pending{operation="count_tokens"}')
OUTCOMES = tuple(f'apxinf_request_outcomes_total{{operation="{operation}",status="{status}"}}'
                 for operation in ("generate", "count_tokens")
                 for status in ("completed", "cancelled", "failed", "expired"))
COMPLETED = 'apxinf_request_outcomes_total{operation="generate",status="completed"}'
MAX_PLAN_BYTES = 4 * 1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, name, low=0, high=2**53 - 1):
    require(type(value) is int and low <= value <= high, f"Invalid integer: {name}.")
    return value


def fields(value, expected, name):
    require(type(value) is dict and set(value) == set(expected), f"Invalid fields: {name}.")


def strict_json(payload):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, f"Duplicate JSON key: {key}.")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Invalid JSON constant: {value}.")

    def number(value):
        result = float(value)
        require(math.isfinite(result), "Invalid finite JSON number.")
        return result

    return json.loads(payload, object_pairs_hook=pairs, parse_constant=constant, parse_float=number)


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(encoded(value) + b"\n")
    temporary.replace(path)


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_plan(path):
    with Path(path).open("rb") as source:
        raw = source.read(MAX_PLAN_BYTES + 1)
    require(len(raw) <= MAX_PLAN_BYTES, "The plan exceeds its byte limit.")
    plan = strict_json(raw)
    fields(plan, ("format", "expected", "limits", "environment", "cases"), "plan")
    require(plan["format"] == "apxinf-production-memory-plan-v1", "Unsupported plan format.")
    fields(plan["expected"], ("model_revision", "capability_revision", "worker_epoch", "service_binary_sha256"), "expected")
    for key, value in plan["expected"].items():
        if key == "worker_epoch":
            require(type(value) is str and str(uuid.UUID(value)) == value, "Invalid worker epoch.")
        else:
            require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value), f"Invalid digest: {key}.")
    bounds = {"request_timeout_ms": (1, 1800000), "settlement_timeout_ms": (1, 120000),
              "observation_interval_ms": (10, 60000), "observation_budget_ms": (1, 30000),
              "max_sample_gap_ms": (10, 120000), "max_response_bytes": (1024, 16 * 1024**2),
              "max_samples": (2, 10000), "max_diagnostic_bytes": (1024, 16 * 1024**2)}
    fields(plan["limits"], bounds, "limits")
    for key, (low, high) in bounds.items():
        integer(plan["limits"][key], key, low, high)
    require(plan["limits"]["max_sample_gap_ms"] >= plan["limits"]["observation_interval_ms"],
            "The sample gap limit is smaller than the observation interval.")
    fields(plan["environment"], ("process_limits", "runtime_settings", "competing_workload_policy"), "environment")
    for value in plan["environment"].values():
        require(type(value) is str and 1 <= len(value) <= 4096, "Environment declarations must be bounded strings.")
    require(type(plan["cases"]) is list and 1 <= len(plan["cases"]) <= 64, "Supply one to 64 cases.")
    identifiers = set()
    for case in plan["cases"]:
        fields(case, ("case_id", "body", "input_tokens_min", "input_tokens_max", "required_checks"), "case")
        name = case["case_id"]
        require(type(name) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name)
                and name not in identifiers, "Case IDs must be safe and unique.")
        identifiers.add(name)
        integer(case["input_tokens_min"], "minimum input", 1, 131072)
        integer(case["input_tokens_max"], "maximum input", case["input_tokens_min"], 131072)
        requested = case["required_checks"]
        require(type(requested) is list and requested and all(type(item) is str and item in CHECKS for item in requested)
                and len(set(requested)) == len(requested), "Invalid required checks.")
        require(CORE_CHECKS <= set(requested), "Each case must require identity, execution, settlement, and host pressure.")
        body = case["body"]
        fields(body, ("model", "messages", "max_tokens", "stream", "temperature"), "request body")
        require(type(body["model"]) is str and bool(body["model"]), "A model alias is required.")
        require(body["stream"] is True and type(body["temperature"]) is int and body["temperature"] == 0,
                "The runner requires greedy streaming requests.")
        integer(body["max_tokens"], "output allowance", 1, 65536)
        require(type(body["messages"]) is list and 1 <= len(body["messages"]) <= 128, "Invalid messages.")
        for message in body["messages"]:
            fields(message, ("role", "content"), "message")
            require(message["role"] in ("user", "assistant") and type(message["content"]) is str, "Use plain text messages.")
        require(len(encoded(body)) <= 1000000, "The request body exceeds the service byte limit.")
    return plan, raw


def verify_lock(path, descriptor):
    expected, actual = Path(path).stat(), os.fstat(descriptor)
    require((expected.st_dev, expected.st_ino) == (actual.st_dev, actual.st_ino), "The inherited lock inode differs.")
    with Path(path).open("rb") as rival:
        try:
            fcntl.flock(rival, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(rival, fcntl.LOCK_UN)
            raise ValueError("The inherited descriptor does not hold an exclusive lock.")
    # A lock held through a different open description cannot pass this operation.
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return {"device": actual.st_dev, "inode": actual.st_ino, "inherited_fd": descriptor}


def fetch(base_url, path, limit, timeout):
    connection = http.connect(base_url, timeout)
    try:
        with http.absolute_deadline(connection, timeout):
            connection.request("GET", path)
            response = connection.getresponse()
            raw = response.read(limit + 1)
            require(len(raw) <= limit, f"The response exceeds its byte limit: {path}.")
            return {"status": response.status, "body": raw.decode("utf-8")}
    finally:
        connection.close()


def metric_values(text):
    values = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, raw = line.rsplit(None, 1)
        require(name not in values, "Duplicate metric sample.")
        value = float(raw)
        # NaN is permitted for unrelated unavailable gauges, but cannot pass a required check.
        values[name] = value if math.isfinite(value) else None
    return values


def drained(values):
    return all(type(values.get(key)) in (int, float) and values[key] == 0 for key in GAUGES)


def outcomes(values):
    result = {}
    for key in OUTCOMES:
        value = values.get(key)
        require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 2**53 - 1 and value == int(value),
                f"Missing or invalid outcome metric: {key}.")
        result[key] = int(value)
    return result


def executable_stat(path):
    value = path.stat()
    return {"device": value.st_dev, "inode": value.st_ino, "size_bytes": value.st_size, "modified_ns": value.st_mtime_ns}


class ProcessObserver:
    """Collect macOS process RSS only while the expected process identities remain stable."""
    def __init__(self, args, ready):
        require(sys.platform == "darwin", "The production process collector requires macOS.")
        self.args = args
        self.identities = {"service": process_identity(args.service_pid), "worker": process_identity(args.worker_pid)}
        service, worker = self.identities["service"], self.identities["worker"]
        command = shlex.split(service["command"])
        require(command and Path(command[0]).resolve() == args.service_binary.resolve(), "The service executable differs.")
        require(worker["parent_pid"] == service["pid"], "The worker parent differs.")
        arguments = shlex.split(worker["command"])
        require(argument_value(arguments, "--worker-epoch") == ready["worker"]["worker_epoch"], "The process epoch differs.")
        require(argument_value(arguments, "--model") == ready["worker"]["model_path"], "The process model differs.")
        require(argument_value(arguments, "--protocol") == "apxinf-worker/2.0"
                and any(Path(part).name == "text_worker.py" for part in arguments), "The worker command differs.")
        port = http.local_url(args.base_url).port or 80
        listing = subprocess.run(["/usr/sbin/lsof", "-a", "-p", str(args.service_pid), f"-iTCP:{port}",
                                  "-sTCP:LISTEN", "-Fp"], capture_output=True, text=True, check=True, timeout=5)
        require([line for line in listing.stdout.splitlines() if line.startswith("p")] == [f"p{args.service_pid}"],
                "The service does not own the endpoint listener.")
        self.executable = executable_stat(args.service_binary)
        self.check_executable()
        self.executable["sha256"] = sha(args.service_binary)
        self.check_executable()

    def check_executable(self):
        require(all(self.executable[key] == value for key, value in executable_stat(self.args.service_binary).items()),
                "The executable file identity changed.")
        listing = subprocess.run(["/usr/sbin/lsof", "-a", "-p", str(self.args.service_pid), "-d", "txt", "-F", "fDin"],
                                 capture_output=True, text=True, check=True, timeout=5)
        records, current = [], {}
        for line in listing.stdout.splitlines():
            if line.startswith("f"):
                current = {}
                records.append(current)
            elif line and line[0] in "Din":
                current[line[0]] = line[1:]
        matches = [record for record in records if "n" in record
                   and Path(record["n"]).resolve() == self.args.service_binary.resolve()]
        require(len(matches) == 1 and int(matches[0].get("D", "-1"), 16) == self.executable["device"]
                and int(matches[0].get("i", "-1")) == self.executable["inode"],
                "The mapped executable identity differs from the hashed file.")

    def sample(self):
        self.check_executable()
        result = {}
        for name, expected in self.identities.items():
            require(process_identity(expected["pid"]) == expected, "The process identity changed before RSS collection.")
            response = subprocess.run(["/bin/ps", "-p", str(expected["pid"]), "-o", "rss="],
                                      capture_output=True, text=True, check=True, timeout=5)
            raw = response.stdout.strip()
            require(raw.isdecimal(), "The process RSS observation is invalid.")
            require(process_identity(expected["pid"]) == expected, "The process identity changed during RSS collection.")
            result[name] = {"identity": expected, "rss_kib": int(raw), "rss_bytes": int(raw) * 1024,
                            "source": "macos ps -o rss", "error": None}
        return result


class Diagnostics:
    def __init__(self, path, destination, limit):
        self.path, self.limit, self.total = path, limit, 0
        self.source = path.open("rb")
        self.identity = os.fstat(self.source.fileno())
        self.source.seek(0, os.SEEK_END)
        self.start = self.source.tell()
        self.output = destination.open("xb")
        self.pending = b""
        self.records = []
        self.errors = []

    def collect(self):
        try:
            actual = self.path.stat()
            require((actual.st_dev, actual.st_ino) == (self.identity.st_dev, self.identity.st_ino)
                    and actual.st_size >= self.source.tell(), "The diagnostic source changed or shrank.")
            data = self.source.read(self.limit - self.total + 1)
            exceeded = self.total + len(data) > self.limit
            data = data[:self.limit - self.total]
            self.output.write(data)
            self.output.flush()
            self.total += len(data)
            lines = (self.pending + data).split(b"\n")
            self.pending = lines.pop()
            for line in lines:
                try:
                    value = strict_json(line)
                    if type(value) is dict:
                        self.records.append(value)
                except (ValueError, UnicodeError):
                    pass  # Raw non-JSON stderr remains in the retained byte slice.
            require(not exceeded, "The diagnostic byte limit was exceeded.")
        except (OSError, ValueError) as error:
            self.errors.append(str(error))

    def close(self):
        self.source.close()
        self.output.close()


def check_ready(ready, expected):
    require(type(ready) is dict, "Readiness must be an object.")
    worker = ready.get("worker", {})
    contracts.validate_frame(worker)
    require(worker["kind"] == "ready", "Expected worker readiness.")
    for name in ("model_revision", "capability_revision", "worker_epoch"):
        require(worker.get(name) == expected[name], f"Readiness identity differs: {name}.")
    require(type(worker.get("model_manifest")) is dict and type(worker.get("capabilities")) is dict,
            "Readiness lacks model or capability evidence.")
    require(worker["capabilities"].get("max_active_sequences") == 1, "The runner requires one active sequence.")
    require(ready.get("profile") == "apxinf-serial-v0.1", "Unsupported service profile.")
    pressure = ready.get("host_pressure", {})
    require(pressure.get("policy") == "macos", "Production pressure checks must remain enabled.")


def normal_pressure(ready):
    pressure = ready.get("host_pressure", {})
    age = pressure.get("sample_age_seconds")
    return (ready.get("ready") is True and ready.get("worker_available") is True
            and pressure.get("state") == "normal" and pressure.get("admission_allowed") is True
            and pressure.get("recovering") is False and type(age) in (int, float) and 0 <= age < 3)


def stream_case(args, case, limits, path):
    state = http.StreamResult("anthropic")
    result = {"success": False, "http_status": None, "request_id": None, "model_revision": None,
              "error": None, "input_tokens": None, "output_tokens": None}
    connection = http.connect(args.base_url, limits["request_timeout_ms"] / 1000)
    started = time.monotonic_ns()
    total = 0
    try:
        with path.open("xb") as destination, http.absolute_deadline(connection, limits["request_timeout_ms"] / 1000):
            connection.request("POST", "/v1/messages", encoded(case["body"]),
                               {"content-type": "application/json", "anthropic-version": "2023-06-01"})
            response = connection.getresponse()
            result.update(http_status=response.status, request_id=response.getheader("x-request-id"),
                          model_revision=response.getheader("x-apxinf-model-revision"))
            if response.status != 200:
                raw = response.read(min(4096, limits["max_response_bytes"]))
                destination.write(raw)
                raise ValueError(f"HTTP {response.status}: {raw.decode('utf-8', 'replace')}")
            require("text/event-stream" in response.getheader("Content-Type", ""), "Expected an SSE response.")
            require(type(result["request_id"]) is str and str(uuid.UUID(result["request_id"])) == result["request_id"],
                    "Invalid response request identity.")

            class Recorder:
                def readline(self, size):
                    nonlocal total
                    raw = response.readline(min(size, limits["max_response_bytes"] - total + 1))
                    require(total + len(raw) <= limits["max_response_bytes"], "The SSE response exceeded its byte limit.")
                    destination.write(raw)
                    destination.flush()
                    total += len(raw)
                    return raw

            for event, raw in http.sse_events(Recorder()):
                state.feed(event, raw, (time.monotonic_ns() - started) / 1e9)
            state.require_finished()
            for key in ("input_tokens", "output_tokens"):
                result[key] = integer(state.usage.get(key), key)
            require(0 < result["input_tokens"] and result["output_tokens"] <= case["body"]["max_tokens"],
                    "Response usage exceeds the request allowance.")
            result["success"] = True
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        connection.close()
    result.update(start_monotonic_ns=started, end_monotonic_ns=time.monotonic_ns(), stop_reason=state.stop_reason,
                  ttft_seconds=state.content_times[0] if state.content_times else None, usage=state.usage,
                  output_sha256=hashlib.sha256(state.text.encode()).hexdigest(), response_file=path.name)
    return result


def runtime_peak(diagnostics, request):
    matches = [record for record in diagnostics.records if record.get("event") == "request_settled"
               and record.get("request_id") == request.get("request_id")]
    unknown = {"status": "unknown", "bytes": None, "reason": "No unique valid diagnostic peak.", "matching_records": matches}
    if diagnostics.errors or diagnostics.pending or len(matches) != 1 or not request["success"] or not request["output_tokens"]:
        return unknown
    record = matches[0]
    metrics = record.get("worker_metrics")
    peak = metrics.get("peak_memory_bytes") if type(metrics) is dict else None
    if (record.get("worker_status") != "completed" or record.get("status") != "completed"
            or record.get("input_tokens") != request["input_tokens"] or record.get("output_tokens") != request["output_tokens"]
            or type(peak) is not int or not 0 < peak <= 2**53 - 1):
        return unknown
    return {"status": "passed", "bytes": peak, "scope": "worker terminal before settlement", "matching_records": matches}


def run(args, *, observer_factory=ProcessObserver):
    plan, raw_plan = read_plan(args.plan_file)
    http.local_url(args.base_url)
    require(args.service_pid > 1 and args.worker_pid > 1 and args.service_pid != args.worker_pid, "Supply distinct process IDs.")
    lock = verify_lock(args.metal_lock, args.lock_fd)
    require(sha(args.service_binary) == plan["expected"]["service_binary_sha256"], "The binary digest differs.")
    destination = args.output_dir.resolve()
    if destination.is_relative_to(ROOT):
        require(destination.is_relative_to(ROOT / "devlocal") and len(destination.relative_to(ROOT / "devlocal").parts) >= 2,
                "Use a new devlocal feature directory or a directory outside the checkout.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "source-plan.json").write_bytes(raw_plan)
    report = {"format": "apxinf-production-memory-evidence-v1", "validation": "incomplete",
              "admission_approved": False, "process_reaping": {"status": "unknown", "reason": "Attached service is not owned."},
              "candidate_validation": {"status": "unknown"}, "status": "running", "error": None,
              "cases": [{"case_id": case["case_id"], "status": "skipped", "checks": {}} for case in plan["cases"]],
              "limitations": ["External collection can change execution timing and miss memory peaks.",
                              "Environment fields are declarations, not verified service settings.",
                              "This run cannot establish process reaping or approve an admission envelope."], "files": {}}
    save(args.output_dir / "report.json", report)
    limits, expected = plan["limits"], plan["expected"]
    diagnostics = None
    observations = []
    try:
        first = fetch(args.base_url, "/readyz", limits["max_response_bytes"], limits["observation_budget_ms"] / 1000)
        ready = strict_json(first["body"])
        check_ready(ready, expected)
        observer = observer_factory(args, ready)
        require(observer.executable["sha256"] == expected["service_binary_sha256"], "The mapped binary digest differs.")
        diagnostics = Diagnostics(args.diagnostics_file, args.output_dir / "diagnostics.jsonl", limits["max_diagnostic_bytes"])
        tool_paths = [Path(__file__), Path(http.__file__), Path(host_memory.__file__),
                      ROOT / "benchmarks/serving/lifecycle_checks.py", CONTRACT_PATH]
        frozen = {"plan": plan, "plan_sha256": hashlib.sha256(raw_plan).hexdigest(), "ready": first,
                  "created_at": datetime.now(timezone.utc).isoformat(), "argv": sys.argv,
                  "processes": observer.identities, "lock": lock, "service_binary": str(args.service_binary.resolve()),
                  "executable": observer.executable,
                  "tools": {path.name: sha(path) for path in tool_paths},
                  "collector": {"python": platform.python_version(), "system": platform.platform(),
                                "hardware": host_memory.hardware_snapshot(), "device_identity": "unknown"},
                  "diagnostics": {"path": str(args.diagnostics_file.resolve()), "start_offset": diagnostics.start,
                                  "device": diagnostics.identity.st_dev, "inode": diagnostics.identity.st_ino}}
        save(args.output_dir / "frozen-plan.json", frozen)

        with (args.output_dir / "observations.jsonl").open("xb") as journal:
            def observe(case_id):
                require(len(observations) < limits["max_samples"], "The observation sample limit was reached.")
                started = time.monotonic_ns()
                sample = {"index": len(observations), "case_id": case_id, "started_monotonic_ns": started,
                          "error": None, "processes": None, "ready": None, "metrics": None}
                try:
                    sample["ready"] = fetch(args.base_url, "/readyz", limits["max_response_bytes"], limits["observation_budget_ms"] / 1000)
                    document = strict_json(sample["ready"]["body"])
                    check_ready(document, expected)
                    sample["pressure_normal"] = normal_pressure(document)
                    sample["metrics"] = fetch(args.base_url, "/metrics", limits["max_response_bytes"], limits["observation_budget_ms"] / 1000)
                    require(sample["metrics"]["status"] == 200, "Metrics are unavailable.")
                    sample["values"] = metric_values(sample["metrics"]["body"])
                    outcomes(sample["values"])
                    sample["processes"] = observer.sample()
                    host = host_memory.snapshot()
                    sample["host"] = {"pressure": host["pressure"], "swap": host["swap"], "monotonic_ns": host["monotonic_ns"]}
                    sample["pressure_normal"] = sample["pressure_normal"] and host["pressure"] == {
                        "state": "normal", "dispatch_value": 1, "error": None}
                except Exception as error:
                    sample["error"] = f"{type(error).__name__}: {error}"
                sample["finished_monotonic_ns"] = time.monotonic_ns()
                sample["duration_ms"] = (sample["finished_monotonic_ns"] - started) / 1e6
                sample["gap_ms"] = (started - observations[-1]["started_monotonic_ns"]) / 1e6 if observations else None
                observations.append(sample)
                journal.write(encoded(sample) + b"\n")
                journal.flush()
                diagnostics.collect()
                return sample

            for case, result in zip(plan["cases"], report["cases"]):
                verify_lock(args.metal_lock, args.lock_fd)
                before = observe(case["case_id"])
                require(before["error"] is None, before["error"] or "Invalid startup observation.")
                if not before["pressure_normal"]:
                    result.update(status="blocked", checks={"host_pressure": "blocked"})
                    break
                require(drained(before["values"]), "The attached service is not drained before the case.")
                require(case["body"]["model"] == ready["model"], "The request model alias differs.")
                first_index = before["index"]
                result["status"] = "running"
                collection_error = None
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(stream_case, args, case, limits, args.output_dir / (case["case_id"] + ".sse"))
                    completed_at = None
                    while True:
                        client_finished = future.done()
                        if client_finished and completed_at is None:
                            completed_at = time.monotonic()
                        try:
                            sample = observe(case["case_id"])
                        except Exception as error:
                            collection_error = f"{type(error).__name__}: {error}"
                            break
                        if client_finished:
                            if not sample["error"] and drained(sample["values"]):
                                break
                            if time.monotonic() - completed_at >= limits["settlement_timeout_ms"] / 1000:
                                break
                        time.sleep(limits["observation_interval_ms"] / 1000)
                    request = future.result()
                diagnostics.collect()
                samples = observations[first_index:]
                checks = {name: "unknown" for name in CHECKS}
                checks["identity"] = "passed" if not collection_error and all(item["error"] is None for item in samples) and request["model_revision"] == expected["model_revision"] else "failed"
                checks["execution"] = "passed" if request["success"] else "failed"
                checks["host_pressure"] = "passed" if all(item.get("pressure_normal") for item in samples) else "blocked"
                checks["input_band"] = ("passed" if request["input_tokens"] is not None
                                        and case["input_tokens_min"] <= request["input_tokens"] <= case["input_tokens_max"] else "failed")
                checks["full_output"] = "passed" if request["success"] and request["output_tokens"] == case["body"]["max_tokens"] else "unreached"
                if not collection_error and not sample["error"] and drained(sample["values"]):
                    delta = {key: outcomes(sample["values"])[key] - outcomes(before["values"])[key] for key in OUTCOMES}
                    checks["settlement"] = "passed" if all(value == (1 if key == COMPLETED else 0) for key, value in delta.items()) and request["success"] else "unknown"
                    result["outcome_deltas"] = delta
                measurements_valid = all(item["processes"] is not None and item["duration_ms"] <= limits["observation_budget_ms"]
                                         and (item["gap_ms"] is None or item["gap_ms"] <= limits["max_sample_gap_ms"]) for item in samples)
                checks["external_memory"] = "passed" if measurements_valid and not collection_error else "unknown"
                peak = runtime_peak(diagnostics, request)
                if checks["identity"] != "passed":
                    peak = {"status": "unknown", "bytes": None, "reason": "Request identity was not established.",
                            "matching_records": peak["matching_records"]}
                checks["runtime_peak"] = peak["status"]
                result.update(request=request, checks=checks, runtime_peak=peak,
                              collection_error=collection_error,
                              post_settlement_allocator={"status": "unknown"}, cache_payload={"status": "unknown"},
                              sample_range={"first": first_index, "count": len(samples)},
                              required_checks_passed=all(checks[key] == "passed" for key in case["required_checks"]))
                result["status"] = ("blocked" if checks["host_pressure"] == "blocked" else
                                    "failed" if "failed" in checks.values() else
                                    "passed" if result["required_checks_passed"] else
                                    "unknown" if any(checks[key] == "unknown" for key in case["required_checks"]) else "unreached")
                save(args.output_dir / "report.json", report)
                if result["status"] in ("failed", "blocked", "unknown"):
                    break
        report["status"] = "collected" if all(item["status"] == "passed" for item in report["cases"]) else "incomplete"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        if diagnostics is not None:
            report["diagnostic_errors"] = diagnostics.errors
            diagnostics.close()
        for path in args.output_dir.iterdir():
            if path.is_file() and path.name not in ("report.json", "report.tmp"):
                report["files"][path.name] = {"size_bytes": path.stat().st_size, "sha256": sha(path)}
        save(args.output_dir / "report.json", report)
    return 0 if report["status"] == "collected" else 1


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan-file", type=Path, required=True)
    result.add_argument("--base-url", required=True)
    result.add_argument("--service-pid", type=int, required=True)
    result.add_argument("--worker-pid", type=int, required=True)
    result.add_argument("--service-binary", type=Path, required=True)
    result.add_argument("--diagnostics-file", type=Path, required=True)
    result.add_argument("--metal-lock", type=Path, required=True)
    result.add_argument("--lock-fd", type=int, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    return result


def main():
    try:
        return run(parser().parse_args())
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
