#!/usr/bin/env python3
"""Check a running Clef decision service through bounded loopback HTTP requests."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python/apxinf/apxinf/serving"))
import clef_contracts as clef
import contracts
from http_benchmark import absolute_deadline, connect, local_url

SCHEMA = "apxinf-clef-acceptance-registration-v1"
PROFILE = "apxinf-clef-text-v0.1"
MAX_BODY = 262144
QUESTIONS = {
    "down": {"type": "noul", "instructions": "Is the database currently unavailable?"},
    "route": {"type": "choice", "instructions": "Which component needs repair?",
              "criteria": {"database": "Database service", "none": "No repair needed",
                           "renderer": "Page rendering service"}},
    "severity": {"type": "score", "instructions": "How urgent is restoring service?",
                 "criteria": ["Routine", "Soon", "Immediate"]}}
OUTAGE = ("The database stopped accepting every connection. All database operations fail. "
          "The renderer is healthy. Restore the database immediately.")
HEALTHY = ("The database accepts every connection and passes all checks. The renderer is healthy. "
           "There are no failures and no repair is needed.")
STATUSES = ("completed", "cancelled", "failed", "expired")
OPERATIONS = ("generate", "count_tokens", "decision")
GAUGES = ("apxinf_active_requests", "apxinf_queued_requests", "apxinf_capacity_reserved_bytes")
COUNTERS = ("apxinf_input_tokens_total", "apxinf_output_tokens_total",
            "apxinf_worker_rotations_total", "apxinf_worker_faults_total")
METRICS = (*GAUGES, *COUNTERS,
           *(f'apxinf_requests_total{{status="{status}"}}' for status in STATUSES),
           *(f'apxinf_cleanup_pending{{operation="{op}"}}' for op in OPERATIONS),
           *(f'apxinf_request_outcomes_total{{operation="{op}",status="{status}"}}'
             for op in OPERATIONS for status in STATUSES))
COUNTER_METRICS = tuple(name for name in METRICS if name not in GAUGES and not name.startswith("apxinf_cleanup_pending"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class Case:
    name: str
    body: bytes
    status: int
    error_code: str | None
    consumes_worker: bool
    encoding_name: str | None


def cases(model):
    base = {"model": model, "state": OUTAGE, "questions": QUESTIONS}
    return (
        Case("invalid_stream", canonical(dict(base, stream=True)), 400, "invalid_request", False, None),
        Case("wrong_alias", canonical(dict(base, model=model + "-unknown")), 404, "model_not_found", False, None),
        Case("image_field", canonical(dict(base, images=[])), 400, "invalid_request", False, None),
        Case("outage", canonical(base), 200, None, True, "outage"),
        Case("healthy", canonical(dict(base, state=HEALTHY)), 200, None, True, "healthy"),
        Case("outage_repeat", canonical(base), 200, None, True, "outage"),
        Case("boundary", canonical(dict(base, state="x " * 3768 + OUTAGE)), 200, None, True, "boundary"),
        Case("overflow", canonical(dict(base, state="x " * 3769 + OUTAGE)), 400, "context_limit", True, "overflow"),
        Case("post_error", canonical(base), 200, None, True, "outage"))


@dataclass(frozen=True)
class Config:
    base_url: str
    registration: Path
    output_dir: Path
    diagnostic_disabled_pressure: bool = False
    request_timeout: float = 120
    settlement_timeout: float = 180


@dataclass(frozen=True)
class Exchange:
    status: int | None
    headers: tuple[tuple[str, str], ...]
    raw: bytes
    error: str | None


@dataclass(frozen=True)
class Snapshot:
    ready: dict
    metrics: dict[str, int]
    epoch: str

    def document(self):
        return {"ready": self.ready, "metrics": self.metrics, "epoch": self.epoch}


def registration_document(path):
    with path.open("rb") as handle:
        raw = handle.read(65537)
    require(len(raw) <= 65536, "The registration exceeds its byte limit.")
    value = contracts.parse_document(raw)
    keys(value, {"schema", "model", "model_revision", "capability_revision", "encodings"})
    require(value["schema"] == SCHEMA, "The registration schema differs.")
    clef.text(value["model"], 1, 128)
    suite = cases(value["model"])
    require(len((value["model"] + "-unknown").encode("utf-8")) <= 128, "The model alias is too long for the suite.")
    for field in ("model_revision", "capability_revision"):
        digest(value[field])
    keys(value["encodings"], {"outage", "healthy", "boundary", "overflow"})
    for name, entry in value["encodings"].items():
        keys(entry, {"request_sha256", "input_tokens", "input_ids_sha256", "source_receipt_sha256"})
        integer(entry["input_tokens"], minimum=1)
        for field in ("request_sha256", "input_ids_sha256", "source_receipt_sha256"):
            digest(entry[field])
        case = next(item for item in suite if item.encoding_name == name)
        clef.parse_decision_request(case.body)
        require(entry["request_sha256"] == hashlib.sha256(case.body).hexdigest(), "A registered request binding differs.")
    require(value["encodings"]["boundary"]["input_tokens"] == 4096, "Register exactly the 4096-token boundary.")
    require(value["encodings"]["overflow"]["input_tokens"] == 4097, "Register exactly the 4097-token overflow.")
    require(all(value["encodings"][name]["input_tokens"] <= 4096 for name in ("outage", "healthy")),
            "An ordinary registered input exceeds the context limit.")
    return value


def keys(value, expected):
    require(type(value) is dict and set(value) == expected, "Object fields differ from the expected schema.")


def digest(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "A SHA-256 digest is invalid.")


def integer(value, minimum=0):
    require(type(value) is int and minimum <= value <= 9007199254740991, "A safe integer is required.")


def exchange(base_url, method, path, body, timeout):
    connection = connect(base_url, timeout)
    raw = bytearray()
    status, headers, error = None, (), None
    try:
        with absolute_deadline(connection, timeout):
            connection.request(method, path, body, {"content-type": "application/json", "connection": "close"})
            response = connection.getresponse()
            status, headers = response.status, tuple(response.getheaders())
            while True:
                chunk = response.read1(min(4096, MAX_BODY + 1 - len(raw)))
                raw.extend(chunk)
                require(len(raw) <= MAX_BODY, "The HTTP body exceeds the observation limit.")
                if not chunk:
                    require(response.length in (None, 0), "The HTTP body ended before its declared length.")
                    break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        connection.close()
    return Exchange(status, headers, bytes(raw[:MAX_BODY]), error)


def json_response(result):
    require(result.error is None, result.error or "The HTTP exchange failed.")
    types = [value.split(";", 1)[0].strip().lower() for key, value in result.headers if key.lower() == "content-type"]
    require(types == ["application/json"], "The response must have one JSON content type.")
    return contracts.parse_document(result.raw)


def probability(value):
    require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1, "A probability is invalid.")


def check_response(result, case, registered):
    require(result.error is None, result.error or "The HTTP exchange failed.")
    require(result.status == case.status, f"Expected HTTP {case.status}, received {result.status}.")
    value = json_response(result)
    if case.error_code is not None:
        keys(value, {"type", "error"})
        require(value["type"] == "error", "The public error envelope differs.")
        keys(value["error"], {"code", "message", "type"})
        require(value["error"]["code"] == case.error_code, "The public error code differs.")
        require(value["error"]["type"] == ("not_found_error" if case.status == 404 else "invalid_request_error"),
                "The public error type differs.")
        clef.text(value["error"]["message"], 1, 4096)
        return value
    keys(value, {"model", "answers", "usage"})
    require(value["model"] == registered["model"], "The public model alias differs.")
    headers = {}
    for key, content in result.headers:
        key = key.lower()
        if key in {"x-request-id", "x-apxinf-model-revision", "x-apxinf-profile"}:
            require(key not in headers, "A response identity header is duplicated.")
            headers[key] = content
    contracts._uuid(headers.get("x-request-id"))
    require(headers.get("x-apxinf-model-revision") == registered["model_revision"], "The response revision header differs.")
    require(headers.get("x-apxinf-profile") == PROFILE, "The response profile header differs.")
    keys(value["usage"], {"input_tokens", "output_tokens"})
    for count in value["usage"].values():
        integer(count)
    require(value["usage"] == {"input_tokens": registered["encodings"][case.encoding_name]["input_tokens"],
                               "output_tokens": 0}, "Usage differs from the independent registration.")
    keys(value["answers"], set(QUESTIONS))
    for qid, question in QUESTIONS.items():
        answer = value["answers"][qid]
        kind = question["type"]
        fields = {"type", "noul"} if kind == "noul" else {"type", "confidence", "probabilities"}
        fields |= {"choice"} if kind == "choice" else {"score", "legend"} if kind == "score" else set()
        keys(answer, fields)
        require(answer["type"] == kind, "An answer type differs.")
        if kind == "noul":
            probability(answer["noul"])
            continue
        options = sorted(question["criteria"]) if kind == "choice" else [str(n) for n in range(len(question["criteria"]))]
        vector = answer["probabilities"]
        keys(vector, set(options))
        for p in vector.values():
            probability(p)
        require(abs(sum(vector.values()) - 1) <= 0.00001, "The probability vector does not normalize.")
        probability(answer["confidence"])
        require(answer["confidence"] == max(vector.values()), "Confidence differs from the maximum probability.")
        if kind == "choice":
            require(answer["choice"] == max(options, key=lambda key: vector[key]), "The choice projection differs.")
        else:
            require(type(answer["score"]) in (int, float) and math.isfinite(answer["score"]), "The score is invalid.")
            projected = sum(n * vector[str(n)] for n in range(len(options)))
            require(abs(answer["score"] - projected) <= 1e-12, "The score projection differs.")
            keys(answer["legend"], set(options))
            require(answer["legend"] == dict(zip(options, question["criteria"])), "The score legend differs.")
    return value


def check_ready(value, registered, diagnostic):
    keys(value, {"ready", "worker_available", "host_pressure", "model", "worker", "profile", "public_capabilities"})
    require(value["ready"] is True and value["worker_available"] is True, "The service is not ready.")
    require(value["model"] == registered["model"] and value["profile"] == PROFILE, "The public readiness identity differs.")
    pressure = value["host_pressure"]
    keys(pressure, {"policy", "state", "admission_allowed", "recovering", "sample_age_seconds"})
    require(pressure["policy"] == ("disabled" if diagnostic else "macos"), "The observed pressure policy differs.")
    require(pressure["admission_allowed"] is True and pressure["recovering"] is False, "Host admission is not allowed.")
    if not diagnostic:
        age = pressure["sample_age_seconds"]
        require(pressure["state"] == "normal" and type(age) in (int, float) and math.isfinite(age) and 0 <= age < 3,
                "The default pressure sample is invalid.")
    else:
        require(pressure["state"] == "disabled" and pressure["sample_age_seconds"] is None, "The disabled pressure state differs.")
    worker = value["worker"]
    clef.validate_frame(worker)
    require(worker["kind"] == "ready", "The worker frame is not ready.")
    for field in ("model_revision", "capability_revision"):
        require(worker[field] == registered[field], "The pinned worker revision differs.")
    require(worker["capabilities"]["max_context"] == 4096, "The suite requires the 4096-token profile.")
    expected = {"workload": "decision", "systemone": True, "text": True, "noul": True, "choice": True,
                "score": True, "images": False, "streaming": False, "token_events": False, "kv_reuse": False,
                "max_context": 4096, "max_output_tokens": 0, "max_active_requests": 1,
                "cancel_granularity": "process_exit", "sessions": False, "batching": False,
                "prompt_cache": False, "anthropic_messages": False, "openai_chat_completions": False,
                "count_tokens": False, "state_schema": "none"}
    require(canonical(value["public_capabilities"]) == canonical(expected), "The public capabilities differ.")
    return worker["worker_epoch"]


def metric_identity(name):
    match = re.fullmatch(r'([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?', name)
    require(match is not None, "A required metric name is invalid.")
    labels = []
    if match[2] is not None:
        for label in match[2].split(","):
            pair = re.fullmatch(r'([a-zA-Z_][a-zA-Z0-9_]*)="([a-zA-Z_]+)"', label)
            require(pair is not None, "A required metric label is invalid.")
            labels.append((pair[1], pair[2]))
        require(len({key for key, _ in labels}) == len(labels), "A required metric label is duplicated.")
    return match[1], tuple(sorted(labels))


def checked_metrics(raw):
    expected = {metric_identity(name): name for name in METRICS}
    families = {name.split("{", 1)[0] for name in METRICS}
    found = {}
    for line in raw.decode("utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0].split("{", 1)[0] not in families:
            continue
        identity = metric_identity(parts[0])
        require(identity in expected, "A required metric has unexpected labels.")
        name = expected[identity]
        require(len(parts) == 2 and re.fullmatch(r"[0-9]+", parts[1]) is not None, "A required metric is not an integer.")
        require(name not in found, "A required metric is duplicated.")
        found[name] = int(parts[1])
        integer(found[name])
    require(set(found) == set(METRICS), "Required metrics are missing.")
    for status in STATUSES:
        require(found[f'apxinf_requests_total{{status="{status}"}}'] == sum(
            found[f'apxinf_request_outcomes_total{{operation="{op}",status="{status}"}}'] for op in OPERATIONS),
            "Legacy and operation outcome totals differ.")
    return found


def settled(metrics):
    return all(metrics[key] == 0 for key in (*GAUGES, *(f'apxinf_cleanup_pending{{operation="{op}"}}' for op in OPERATIONS)))


def expected_metrics(before, case, registered):
    result = dict(before)
    if case.consumes_worker:
        status = "completed" if case.status == 200 else "failed"
        result[f'apxinf_requests_total{{status="{status}"}}'] += 1
        result[f'apxinf_request_outcomes_total{{operation="decision",status="{status}"}}'] += 1
        result["apxinf_worker_rotations_total"] += 1
        if case.status == 200:
            result["apxinf_input_tokens_total"] += registered["encodings"][case.encoding_name]["input_tokens"]
    return result


def save_report(path, report):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(config):
    parsed = local_url(config.base_url)
    address = ipaddress.ip_address(parsed.hostname)
    require(address.is_loopback and getattr(address, "scope_id", None) is None,
            "Use a numeric loopback address without DNS or a scope identifier.")
    require(parsed.port is None or 1 <= parsed.port <= 65535, "Use a positive TCP port.")
    for value in (config.request_timeout, config.settlement_timeout):
        require(type(value) in (int, float) and math.isfinite(value) and value > 0, "Timeouts must be finite and positive.")
    config.output_dir.mkdir(parents=True, exist_ok=False)
    path = config.output_dir / "report.json"
    report = {"schema": "apxinf-clef-acceptance-v1", "passed": False, "default_pressure_qualified": False,
              "phase": "registration", "post_attempts": 0, "base_url": config.base_url,
              "request_timeout_s": config.request_timeout, "settlement_timeout_s": config.settlement_timeout,
              "diagnostic_disabled_pressure": config.diagnostic_disabled_pressure,
              "cases": [], "exchanges": [], "error": None,
              "limitations": ["Registration binds requests but cannot prove historical independence or input ID counts.",
                              "The tool checks public projections without independent logits or hidden noul false probabilities.",
                              "Semantic routing and repeated answer equality are observations, not acceptance gates.",
                              "HTTP observations cannot prove operating system process reaping or external pressure monitoring."]}
    save_report(path, report)

    def observe(method, endpoint, body=None, deadline=None):
        timeout = config.request_timeout if deadline is None else min(config.request_timeout, deadline - time.monotonic())
        if timeout <= 0:
            raise TimeoutError("The settlement deadline expired.")
        result = exchange(config.base_url, method, endpoint, body, timeout)
        observed_at = time.monotonic()
        index = len(report["exchanges"])
        filename = f"exchange-{index:04d}.body"
        (config.output_dir / filename).write_bytes(result.raw)
        report["exchanges"].append({"method": method, "path": endpoint, "status": result.status,
                                    "headers": [list(pair) for pair in result.headers],
                                    "raw_body": filename, "error": result.error})
        save_report(path, report)
        if deadline is not None and observed_at >= deadline:
            raise TimeoutError("The settlement deadline expired.")
        return result

    def snapshot(registered, previous=None, case=None, seen=()):
        deadline = time.monotonic() + config.settlement_timeout
        expected = (expected_metrics(previous.metrics, case, registered) if case else previous.metrics) if previous else None
        while time.monotonic() < deadline:
            first = observe("GET", "/readyz", deadline=deadline)
            require(first.error is None, first.error or "Readiness transport failed.")
            require(first.status in (200, 503), "The readiness HTTP status is unexpected.")
            if first.status == 200:
                ready = json_response(first)
                epoch = check_ready(ready, registered, config.diagnostic_disabled_pressure)
                metrics_result = observe("GET", "/metrics", deadline=deadline)
                require(metrics_result.status == 200 and metrics_result.error is None, "Metrics observation failed.")
                metrics = checked_metrics(metrics_result.raw)
                if previous is not None:
                    require(all(previous.metrics[name] <= metrics[name] <= expected[name] for name in COUNTER_METRICS),
                            "An observed counter decreased or exceeded its expected delta.")
                second = observe("GET", "/readyz", deadline=deadline)
                require(second.error is None, second.error or "Readiness transport failed.")
                require(second.status in (200, 503), "The readiness HTTP status is unexpected.")
                if second.status == 200:
                    after = json_response(second)
                    final_epoch = check_ready(after, registered, config.diagnostic_disabled_pressure)
                    stable = (epoch == final_epoch and ready["worker"] == after["worker"]
                              and ready["host_pressure"]["policy"] == after["host_pressure"]["policy"])
                    epoch_ok = previous is None or (epoch not in seen if case and case.consumes_worker else epoch == previous.epoch)
                    if stable and settled(metrics) and epoch_ok and (expected is None or metrics == expected):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("The settlement deadline expired.")
                        return Snapshot(after, metrics, epoch)
            time.sleep(min(0.05, max(0, deadline - time.monotonic())))
        raise TimeoutError("The joint readiness, identity, counters, epoch, and resource settlement did not converge.")

    try:
        registered = registration_document(config.registration)
        suite = cases(registered["model"])
        report["registration"] = registered
        report["registration_sha256"] = hashlib.sha256(canonical(registered)).hexdigest()
        report["phase"] = "initial_settlement"
        save_report(path, report)
        before = snapshot(registered)
        report["initial"] = before.document()
        seen = {before.epoch}
        request_ids = set()
        answers = {}
        for case in suite:
            report["phase"] = "case_precondition"
            save_report(path, report)
            before = snapshot(registered, before, seen=seen)
            row = {"name": case.name, "passed": False, "post_attempted": False,
                   "post_state": "not_started", "before": before.document()}
            report["cases"].append(row)
            report["phase"] = "post_attempt"
            row["post_attempted"] = True
            row["post_state"] = "dispatch_started"
            report["post_attempts"] += 1
            save_report(path, report)
            result = observe("POST", "/v1/systemone", case.body)
            row["exchange_index"] = len(report["exchanges"]) - 1
            row["post_state"] = "transport_failed" if result.error else "response_received"
            row["response"] = check_response(result, case, registered)
            if case.status == 200:
                request_id = next(value for key, value in result.headers if key.lower() == "x-request-id")
                require(request_id not in request_ids, "A successful response reused a request ID.")
                request_ids.add(request_id)
            report["phase"] = "case_settlement"
            save_report(path, report)
            after = snapshot(registered, before, case, seen)
            row["after"], row["passed"] = after.document(), True
            seen.add(after.epoch)
            if case.status == 200:
                answers[case.name] = row["response"]
            before = after
            save_report(path, report)
        report["observations"] = {"outage_route": answers["outage"]["answers"]["route"]["choice"],
                                  "healthy_route": answers["healthy"]["answers"]["route"]["choice"],
                                  "repeat_equal": answers["outage"] == answers["outage_repeat"]}
        report["passed"] = True
        report["default_pressure_qualified"] = not config.diagnostic_disabled_pressure
        report["phase"] = "complete"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        save_report(path, report)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--base-url", required=True)
    result.add_argument("--registration", required=True, type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--diagnostic-disabled-pressure", action="store_true")
    result.add_argument("--request-timeout", type=float, default=120)
    result.add_argument("--settlement-timeout", type=float, default=180)
    return result


if __name__ == "__main__":
    report = run(Config(**vars(parser().parse_args())))
    print(json.dumps({key: report[key] for key in ("passed", "default_pressure_qualified", "phase", "post_attempts", "error")}))
    raise SystemExit(0 if report["passed"] else 1)
