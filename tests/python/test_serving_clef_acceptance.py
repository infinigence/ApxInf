from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/serving"))
import clef_acceptance as a


def ready_document(policy="macos"):
    def file_record(path):
        return {"path": path, "sha256": "a" * 64}

    manifest = {"checkpoint": a.clef.CHECKPOINT, "checkpoint_revision": a.clef.CHECKPOINT_REVISION,
                "adapter_revision": a.clef.ADAPTER_REVISION, "platform": "macos-metal",
                "artifacts": a.clef.approved_artifacts(),
                "loader": {"path": "/fixtures/clef_mlx.py", "sha256": a.clef.LOADER_SHA256},
                "runtime": a.clef.RUNTIME,
                "sources": {name: file_record("/fixtures/" + name) for name in a.clef.SOURCE_FILES},
                "python": file_record("/fixtures/python"),
                "runtime_files": {"modules": {name: file_record("/fixtures/" + name) for name in
                                  ("mlx_core", "mlx_lm", "transformers", "numpy", "pillow", "tokenizers")},
                                  "libraries": [file_record("/fixtures/libmlx.dylib")]},
                "execution": a.clef.execution(4096, 8 * 1024**3)}
    caps = a.clef.capabilities(248320, 4096)
    worker = {"protocol": a.clef.PROTOCOL, "kind": "ready", "worker_epoch": "00000000-0000-4000-8000-000000000001",
              "model_revision": a.contracts.canonical_identity_digest("apxinf-model-identity-v1", manifest),
              "capability_revision": a.contracts.canonical_identity_digest("apxinf-capability-v1", caps),
              "model_manifest": manifest, "capabilities": caps, "runtime": a.clef.RUNTIME,
              "limits": {"max_frame_bytes": 1048576, "max_depth": 16, "max_commands": 10000},
              "model_path": "/fixtures/model", "memory": {"active_bytes": 7, "peak_bytes": 11, "cache_bytes": 0}}
    public = {"workload": "decision", "systemone": True, "text": True, "noul": True, "choice": True, "score": True,
              "images": False, "streaming": False, "token_events": False, "kv_reuse": False, "max_context": 4096,
              "max_output_tokens": 0, "max_active_requests": 1, "cancel_granularity": "process_exit", "sessions": False,
              "batching": False, "prompt_cache": False, "anthropic_messages": False, "openai_chat_completions": False,
              "count_tokens": False, "state_schema": "none"}
    return {"ready": True, "worker_available": True, "model": "clef-flash", "profile": a.PROFILE,
            "worker": worker, "public_capabilities": public,
            "host_pressure": {"policy": policy, "state": "normal" if policy == "macos" else "disabled",
                              "admission_allowed": True, "recovering": False,
                              "sample_age_seconds": 0.1 if policy == "macos" else None}}


def registration(ready):
    value = {"schema": a.SCHEMA, "model": "clef-flash", "model_revision": ready["worker"]["model_revision"],
             "capability_revision": ready["worker"]["capability_revision"], "encodings": {}}
    counts = {"outage": 328, "healthy": 331, "boundary": 4096, "overflow": 4097}
    for case in a.cases(value["model"]):
        if case.encoding_name is not None:
            value["encodings"][case.encoding_name] = {"request_sha256": hashlib.sha256(case.body).hexdigest(),
                "input_tokens": counts[case.encoding_name], "input_ids_sha256": "b" * 64,
                "source_receipt_sha256": "c" * 64}
    return value


def answer(input_tokens):
    return {"model": "clef-flash", "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            "answers": {"down": {"type": "noul", "noul": 0.75},
                        "route": {"type": "choice", "choice": "database", "confidence": 0.5,
                                  "probabilities": {"renderer": 0.25, "none": 0.25, "database": 0.5}},
                        "severity": {"type": "score", "score": 1.5, "confidence": 0.625,
                                     "legend": {"0": "Routine", "1": "Soon", "2": "Immediate"},
                                     "probabilities": {"0": 0.125, "1": 0.25, "2": 0.625}}}}


def response(value, status=200, revision=None):
    headers = [("content-type", "application/json")]
    if status == 200:
        headers += [("x-request-id", "00000000-0000-4000-8000-000000000099"),
                    ("x-apxinf-profile", a.PROFILE), ("x-apxinf-model-revision", revision)]
    return a.Exchange(status, tuple(headers), a.canonical(value), None)


class ServiceFixture:
    def __init__(self, policy="macos", fault=None, delay=False):
        self.ready = ready_document(policy)
        self.registration = registration(self.ready)
        self.metrics = dict.fromkeys(a.METRICS, 0)
        self.posts = []
        self.fault = fault
        self.delay = delay
        self.pending = 0
        self.epoch = 1
        self.ready_reads = 0
        self.post_while_pending = False

    def get_ready(self):
        self.ready_reads += 1
        if self.pending:
            self.pending -= 1
            if self.pending == 0:
                self.epoch += 1
                if self.fault != "no_rotation":
                    self.ready["worker"]["worker_epoch"] = f"00000000-0000-4000-8000-{self.epoch:012d}"
                self.metrics["apxinf_worker_rotations_total"] += 1
                self.metrics["apxinf_capacity_reserved_bytes"] = 0
                self.metrics['apxinf_cleanup_pending{operation="decision"}'] = 0
        return self.ready

    def post(self, raw):
        self.post_while_pending |= bool(self.pending)
        self.posts.append(raw)
        case = a.cases("clef-flash")[len(self.posts) - 1]
        if case.consumes_worker:
            self.metrics = a.expected_metrics(self.metrics, case, self.registration)
            self.metrics["apxinf_worker_rotations_total"] -= 1
            self.metrics["apxinf_capacity_reserved_bytes"] = 1024
            self.metrics['apxinf_cleanup_pending{operation="decision"}'] = 1
            self.pending = 4 if self.delay else 1
            if self.fault == "counter":
                self.metrics["apxinf_worker_faults_total"] += 1
            if self.fault == "counter_decrease":
                self.metrics["apxinf_output_tokens_total"] = 0
            if self.fault == "other_operation":
                self.metrics['apxinf_request_outcomes_total{operation="generate",status="completed"}'] += 1
                self.metrics['apxinf_requests_total{status="completed"}'] += 1
        if case.status != 200:
            return case.status, {"type": "error", "error": {"code": case.error_code, "message": "Fixture refusal.",
                    "type": "not_found_error" if case.status == 404 else "invalid_request_error"}}
        count = self.registration["encodings"][case.encoding_name]["input_tokens"]
        return 200, answer(count + (1 if self.fault == "usage" else 0))


@contextmanager
def running_service(fixture):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, status, raw, content_type):
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(raw)))
            index = 99 if fixture.fault == "duplicate_uuid" else len(fixture.posts)
            self.send_header("x-request-id", f"00000000-0000-4000-8000-{index:012d}")
            self.send_header("x-apxinf-model-revision", fixture.registration["model_revision"])
            self.send_header("x-apxinf-profile", a.PROFILE)
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/readyz":
                self.send(200, a.canonical(fixture.get_ready()), "application/json")
            else:
                raw = "".join(f"{key} {value}\n" for key, value in fixture.metrics.items()).encode()
                self.send(200, raw, "text/plain")

        def do_POST(self):
            raw = self.rfile.read(int(self.headers["content-length"]))
            status, value = fixture.post(raw)
            if fixture.fault == "disconnect":
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            if fixture.fault == "partial":
                self.send_response(400)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", "100")
                self.end_headers()
                self.wfile.write(b'{"error":')
                self.wfile.flush()
                self.close_connection = True
                return
            self.send(status, a.canonical(value), "application/json")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class ClefAcceptanceTests(unittest.TestCase):
    def run_fixture(self, fixture, diagnostic=False, mutate_registration=None):
        with tempfile.TemporaryDirectory() as directory, running_service(fixture) as url:
            root = Path(directory)
            registered = copy.deepcopy(fixture.registration)
            if mutate_registration:
                mutate_registration(registered)
            path = root / "registration.json"
            path.write_bytes(a.canonical(registered))
            result = a.run(a.Config(url, path, root / "result", diagnostic, 0.5, 0.3))
            self.assertEqual(result, json.loads((root / "result/report.json").read_text()))
            result["raw_bodies"] = [(root / "result" / item["raw_body"]).read_bytes() for item in result["exchanges"]]
            return result

    def test_nine_real_http_cases_require_joint_settlement(self):
        fixture = ServiceFixture(delay=True)
        result = self.run_fixture(fixture)
        self.assertTrue(result["passed"], result["error"])
        self.assertTrue(result["default_pressure_qualified"])
        self.assertEqual(fixture.posts, [case.body for case in a.cases("clef-flash")])
        self.assertFalse(fixture.post_while_pending)
        self.assertGreater(fixture.ready_reads, 20)
        self.assertEqual(len({row["after"]["epoch"] for row in result["cases"]}), 7)
        self.assertEqual(fixture.metrics["apxinf_worker_rotations_total"], 6)
        self.assertEqual(fixture.metrics['apxinf_requests_total{status="completed"}'], 5)
        self.assertEqual(fixture.metrics['apxinf_requests_total{status="failed"}'], 1)
        self.assertEqual(fixture.metrics["apxinf_input_tokens_total"], 328 * 3 + 331 + 4096)

    def test_explicit_diagnostic_pass_does_not_qualify_default_pressure(self):
        result = self.run_fixture(ServiceFixture("disabled"), diagnostic=True)
        self.assertTrue(result["passed"], result["error"])
        self.assertFalse(result["default_pressure_qualified"])

    def test_pressure_policy_mismatch_stops_before_any_post(self):
        for policy, diagnostic in (("disabled", False), ("macos", True)):
            with self.subTest(policy=policy):
                fixture = ServiceFixture(policy)
                result = self.run_fixture(fixture, diagnostic)
                self.assertFalse(result["passed"])
                self.assertEqual(fixture.posts, [])
                self.assertIn("pressure policy", result["error"])

    def test_independent_count_mismatch_is_not_derived_from_response(self):
        fixture = ServiceFixture()
        def wrong_count(value):
            value["encodings"]["outage"]["input_tokens"] += 1
        result = self.run_fixture(fixture, mutate_registration=wrong_count)
        self.assertFalse(result["passed"])
        self.assertIn("independent registration", result["error"])
        self.assertEqual(len(fixture.posts), 4)

    def test_transport_failure_does_not_replay_post(self):
        fixture = ServiceFixture(fault="disconnect")
        result = self.run_fixture(fixture)
        self.assertFalse(result["passed"])
        self.assertEqual(len(fixture.posts), 1)
        self.assertEqual(result["post_attempts"], 1)
        self.assertEqual(result["phase"], "post_attempt")
        self.assertTrue(result["cases"][0]["post_attempted"])
        self.assertTrue(result["exchanges"][-1]["error"])

    def test_failed_partial_body_is_retained(self):
        result = self.run_fixture(ServiceFixture(fault="partial"))
        self.assertFalse(result["passed"])
        self.assertEqual(result["raw_bodies"][-1], b'{"error":')
        self.assertIn("declared length", result["exchanges"][-1]["error"])

    def test_wrong_request_binding_fails_before_network(self):
        fixture = ServiceFixture()
        def wrong_binding(value):
            value["encodings"]["healthy"]["request_sha256"] = "0" * 64
        result = self.run_fixture(fixture, mutate_registration=wrong_binding)
        self.assertEqual(result["exchanges"], [])
        self.assertFalse(result["passed"])

    def test_missing_rotation_fault_and_other_operations_fail(self):
        for fault in ("no_rotation", "counter", "counter_decrease", "other_operation"):
            with self.subTest(fault=fault):
                fixture = ServiceFixture(fault=fault)
                if fault == "counter_decrease":
                    fixture.metrics["apxinf_output_tokens_total"] = 1
                result = self.run_fixture(fixture)
                self.assertFalse(result["passed"])
                self.assertIn("did not converge" if fault == "no_rotation" else "counter", result["error"])
                self.assertEqual(len(fixture.posts), 4)

    def test_duplicate_successful_request_id_fails(self):
        fixture = ServiceFixture(fault="duplicate_uuid")
        result = self.run_fixture(fixture)
        self.assertFalse(result["passed"])
        self.assertIn("reused a request ID", result["error"])
        self.assertEqual(len(fixture.posts), 5)

    def test_ready_validates_outer_and_pinned_worker_identity(self):
        original = ready_document()
        registered = registration(original)
        for pointer, replacement in (("profile", "another"), ("worker", None), ("public_capabilities", {})):
            with self.subTest(pointer=pointer):
                value = copy.deepcopy(original)
                value[pointer] = replacement
                with self.assertRaises((ValueError, a.contracts.ContractError)):
                    a.check_ready(value, registered, False)
        for mutate in (
            lambda v: v["worker"].update(model_revision="0" * 64),
            lambda v: v["worker"]["model_manifest"]["execution"].update(kv_reuse=True),
            lambda v: v["public_capabilities"].update(max_output_tokens=False),
            lambda v: v["host_pressure"].update(state="warning"),
            lambda v: v["host_pressure"].update(admission_allowed=1),
            lambda v: v["host_pressure"].update(sample_age_seconds=3),
        ):
            value = copy.deepcopy(original)
            mutate(value)
            with self.assertRaises((ValueError, a.contracts.ContractError)):
                a.check_ready(value, registered, False)

    def test_required_metrics_reject_missing_duplicate_noninteger_and_boolean(self):
        valid = "".join(f"{key} 0\n" for key in a.METRICS).encode()
        self.assertEqual(a.checked_metrics(valid), dict.fromkeys(a.METRICS, 0))
        for raw in (valid + b"apxinf_active_requests 0\n", valid.replace(b"apxinf_active_requests 0\n", b""),
                    valid.replace(b"apxinf_active_requests 0", b"apxinf_active_requests NaN"),
                    valid.replace(b"apxinf_active_requests 0", b"apxinf_active_requests 0.0"),
                    valid.replace(b"apxinf_active_requests 0", b"apxinf_active_requests true")):
            with self.subTest(raw=raw[:60]), self.assertRaises(ValueError):
                a.checked_metrics(raw)
        reordered = b'apxinf_request_outcomes_total{status="completed",operation="decision"} 0\n'
        with self.assertRaises(ValueError):
            a.checked_metrics(valid + reordered)
        original = b'apxinf_request_outcomes_total{operation="decision",status="completed"} 0\n'
        self.assertEqual(a.checked_metrics(valid.replace(original, reordered)), dict.fromkeys(a.METRICS, 0))

    def test_response_rejects_shape_probability_projection_and_usage_errors(self):
        registered = registration(ready_document())
        case = a.cases("clef-flash")[3]
        revision = registered["model_revision"]
        self.assertEqual(a.check_response(response(answer(328), revision=revision), case, registered), answer(328))
        mutations = (
            lambda v: v["usage"].update(input_tokens=True),
            lambda v: v["usage"].update(output_tokens=1),
            lambda v: v["answers"]["down"].update(noul=True),
            lambda v: v["answers"]["route"].update(choice="none"),
            lambda v: v["answers"]["severity"].update(score=1.4),
            lambda v: v["answers"]["severity"]["legend"].update({"1": "Wrong"}),
            lambda v: v["answers"]["route"]["probabilities"].update(database=0.6),
            lambda v: v.update(extra=0),
        )
        for mutate in mutations:
            value = answer(328)
            mutate(value)
            with self.assertRaises((ValueError, a.contracts.ContractError)):
                a.check_response(response(value, revision=revision), case, registered)

    def test_strict_json_rejects_duplicate_nonfinite_and_unsafe_integer(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e999}', b'{"a":9007199254740992}'):
            with self.subTest(raw=raw), self.assertRaises(a.contracts.ContractError):
                a.json_response(a.Exchange(200, (("content-type", "application/json"),), raw, None))

    def test_choice_tie_uses_canonical_order(self):
        registered = registration(ready_document())
        value = answer(328)
        value["answers"]["route"].update(confidence=0.5, probabilities={"none": 0.5, "database": 0.5, "renderer": 0.0})
        a.check_response(response(value, revision=registered["model_revision"]), a.cases("clef-flash")[3], registered)
        value["answers"]["route"]["choice"] = "none"
        with self.assertRaises(ValueError):
            a.check_response(response(value, revision=registered["model_revision"]), a.cases("clef-flash")[3], registered)

    def test_score_projection_tolerates_only_small_rounding_difference(self):
        registered = registration(ready_document())
        case = a.cases("clef-flash")[3]
        value = answer(328)
        value["answers"]["severity"]["score"] += 1e-13
        a.check_response(response(value, revision=registered["model_revision"]), case, registered)
        value["answers"]["severity"]["score"] += 1e-10
        with self.assertRaises(ValueError):
            a.check_response(response(value, revision=registered["model_revision"]), case, registered)

    def test_trickled_body_uses_absolute_deadline_and_retains_partial_bytes(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("content-length", "100")
                self.end_headers()
                try:
                    for _ in range(100):
                        self.wfile.write(b"x")
                        self.wfile.flush()
                        time.sleep(0.02)
                except OSError:
                    pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        try:
            started = time.monotonic()
            result = a.exchange(f"http://127.0.0.1:{server.server_port}", "GET", "/", None, 0.12)
            self.assertLess(time.monotonic() - started, 0.8)
            self.assertIn("absolute deadline", result.error)
            self.assertGreater(len(result.raw), 0)
            self.assertLess(len(result.raw), 100)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_http_body_cap_retains_only_bounded_raw_bytes(self):
        with running_service(ServiceFixture()) as url, patch.object(a, "MAX_BODY", 32):
            result = a.exchange(url, "GET", "/metrics", None, 0.5)
        self.assertEqual(len(result.raw), 32)
        self.assertIn("observation limit", result.error)

    def test_numeric_loopback_and_new_output_directory_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for url in ("http://localhost:8080", "http://example.com", "http://127.0.0.1/path", "https://127.0.0.1"):
                with self.subTest(url=url), self.assertRaises(ValueError):
                    a.run(a.Config(url, root / "missing", root / "out"))
            with self.assertRaises(FileExistsError):
                a.run(a.Config("http://127.0.0.1", root / "missing", root))
            for timeout in (float("nan"), float("inf"), 0, -1, True):
                with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                    a.run(a.Config("http://127.0.0.1", root / "missing", root / "out", request_timeout=timeout))

    def test_cli_help_imports_without_model_dependencies(self):
        command = [sys.executable, "-S", str(ROOT / "benchmarks/serving/clef_acceptance.py"), "--help"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--registration", result.stdout)


if __name__ == "__main__":
    unittest.main()
