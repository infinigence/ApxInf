"""Exercise production evidence collection with original CPU HTTP fixtures."""

from contextlib import contextmanager
import copy
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/serving"))
import production_memory_validation as probe


def ready_fixture():
    capabilities = {"text": True, "image": False, "greedy": True, "sampling": False,
                    "token_events": True, "cancel_granularity": "token_or_prefill_chunk",
                    "max_active_sequences": 1, "max_context": 16384, "max_output_tokens": 2048,
                    "exact_append": False, "state_schema": "none"}
    manifest = {"provider": "cpu-http-fixture", "precision": "fixture"}
    worker = {"protocol": "apxinf-worker/2.0", "kind": "ready", "worker_epoch": str(uuid.uuid4()),
              "model_manifest": manifest, "capabilities": capabilities,
              "model_revision": probe.contracts.canonical_identity_digest("apxinf-model-identity-v1", manifest),
              "capability_revision": probe.contracts.canonical_identity_digest("apxinf-capability-v1", capabilities),
              "model_path": "/original/cpu/fixture", "eos_token_ids": [2], "vocab_size": 32,
              "runtime": {"python": "fixture", "mlx": "not-loaded", "mlx_lm": "not-loaded"},
              "memory": {"active_bytes": 700, "peak_bytes": 900, "cache_bytes": 200},
              "limits": {"max_frame_bytes": 1048576, "max_depth": 16, "max_commands": 10000}}
    return {"ready": True, "worker_available": True, "model": "fixture", "worker": worker,
            "profile": "apxinf-serial-v0.1", "host_pressure": {"policy": "macos", "state": "normal",
                "admission_allowed": True, "recovering": False, "sample_age_seconds": 0.01}}


def event(kind, **body):
    return f"event: {kind}\ndata: {json.dumps({'type': kind, **body})}\n\n".encode()


class ServiceFixture:
    def __init__(self, directory, **changes):
        self.directory = directory
        self.ready = ready_fixture()
        self.active = self.completed = self.requests = 0
        self.output_tokens = 4
        self.cleanup_delay = .035
        self.skip_cleanup = self.omit_peak = self.duplicate_peak = False
        self.bad_epoch = self.bad_header = self.incomplete_stream = self.extra_outcome = False
        self.pressure_after_start = False
        self.missing_metric = None
        self.freeze_present = []
        self.diagnostics = directory / "service.log"
        self.diagnostics.write_bytes(b"historical diagnostic bytes\n")
        self.__dict__.update(changes)

    def metrics(self):
        values = {key: 0 for key in probe.GAUGES + probe.OUTCOMES}
        values[probe.GAUGES[0]] = self.active
        values[probe.GAUGES[3]] = self.active
        values[probe.COMPLETED] = self.completed
        values['apxinf_worker_peak_bytes'] = 99999999
        values['apxinf_diagnostic_write_errors_total'] = 0
        if self.missing_metric:
            del values[self.missing_metric]
        return "".join(f"{key} {value}\n" for key, value in values.items()).encode()

    def handler(self):
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path == "/readyz":
                    ready = copy.deepcopy(fixture.ready)
                    if fixture.requests and fixture.bad_epoch:
                        ready["worker"]["worker_epoch"] = str(uuid.uuid4())
                    if fixture.requests and fixture.pressure_after_start:
                        ready["host_pressure"].update(state="critical", admission_allowed=False)
                    raw = json.dumps(ready).encode()
                elif self.path == "/metrics":
                    raw = fixture.metrics()
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_POST(self):
                if self.path != "/v1/messages":
                    self.send_error(404)
                    return
                fixture.freeze_present.append((fixture.directory / "output/frozen-plan.json").is_file())
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                assert body["stream"] is True
                fixture.requests += 1
                fixture.active = 1
                request_id = str(uuid.uuid4())
                frames = [event("message_start", message={"content": [], "usage": {"input_tokens": 9, "output_tokens": 0}}),
                          event("content_block_start", index=0, content_block={"type": "text", "text": ""}),
                          event("content_block_delta", index=0, delta={"type": "text_delta", "text": "Fixture result."})]
                if not fixture.incomplete_stream:
                    frames.extend([event("content_block_stop", index=0),
                                   event("message_delta", delta={"stop_reason": "end_turn"}, usage={"output_tokens": fixture.output_tokens}),
                                   event("message_stop")])
                raw = b"".join(frames)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("x-request-id", request_id)
                self.send_header("x-apxinf-model-revision", "f" * 64 if fixture.bad_header else fixture.ready["worker"]["model_revision"])
                self.end_headers()
                self.wfile.write(raw)
                self.wfile.flush()
                time.sleep(fixture.cleanup_delay)
                if not fixture.omit_peak:
                    record = {"event": "request_settled", "request_id": request_id, "status": "completed",
                              "worker_status": "completed", "input_tokens": 9, "output_tokens": fixture.output_tokens,
                              "worker_metrics": {"elapsed_ns": 12000, "ttft_ns": 8000, "peak_memory_bytes": 12345}}
                    with fixture.diagnostics.open("ab") as destination:
                        destination.write((json.dumps(record) + "\n").encode() * (2 if fixture.duplicate_peak else 1))
                if not fixture.skip_cleanup:
                    fixture.active = 0
                    fixture.completed += 2 if fixture.extra_outcome else 1

        return Handler


class ProcessFixture:
    def __init__(self, args, ready):
        self.identities = {"service": {"pid": args.service_pid, "started": "CPU fixture"},
                           "worker": {"pid": args.worker_pid, "started": "CPU fixture"}}
        self.executable = {**probe.executable_stat(args.service_binary), "sha256": probe.sha(args.service_binary)}

    def sample(self):
        return {name: {"identity": identity, "rss_kib": 10, "rss_bytes": 10240,
                       "source": "CPU fixture", "error": None} for name, identity in self.identities.items()}


class ProductionMemoryTests(unittest.TestCase):
    @contextmanager
    def fixture(self, **changes):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            fixture = ServiceFixture(directory, **changes)
            binary = directory / "service"
            binary.write_bytes(b"Original CPU executable identity fixture.\n")
            lock_path = directory / "metal.lock"
            lock_path.touch()
            with lock_path.open("rb") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                expected = {key: fixture.ready["worker"][key] for key in ("model_revision", "capability_revision", "worker_epoch")}
                expected["service_binary_sha256"] = probe.sha(binary)
                plan = {"format": "apxinf-production-memory-plan-v1", "expected": expected,
                        "environment": {"process_limits": "CPU fixture", "runtime_settings": "CPU fixture",
                                        "competing_workload_policy": "No device execution."},
                        "limits": {"request_timeout_ms": 1000, "settlement_timeout_ms": 100,
                                   "observation_interval_ms": 10, "observation_budget_ms": 1000,
                                   "max_sample_gap_ms": 1000, "max_response_bytes": 65536,
                                   "max_samples": 100, "max_diagnostic_bytes": 65536},
                        "cases": [{"case_id": "original-case", "input_tokens_min": 8, "input_tokens_max": 10,
                                   "required_checks": sorted(probe.CHECKS), "body": {"model": "fixture", "stream": True,
                                       "temperature": 0, "max_tokens": 4,
                                       "messages": [{"role": "user", "content": "Write an original sentence."}]}}]}
                server = ThreadingHTTPServer(("127.0.0.1", 0), fixture.handler())
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                args = SimpleNamespace(plan_file=directory / "plan.json", base_url=f"http://127.0.0.1:{server.server_port}",
                                       service_pid=101, worker_pid=102, service_binary=binary, diagnostics_file=fixture.diagnostics,
                                       metal_lock=lock_path, lock_fd=lock.fileno(), output_dir=directory / "output")
                try:
                    yield fixture, plan, args
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join()

    def collect(self, fixture, plan, args):
        args.plan_file.write_text(json.dumps(plan))
        host = {"pressure": {"state": "normal", "dispatch_value": 1, "error": None}, "swap": {"used_bytes": 0},
                "process_peak_rss_bytes": 999999, "monotonic_ns": time.monotonic_ns()}
        with patch.object(probe.host_memory, "snapshot", return_value=host), \
             patch.object(probe.host_memory, "hardware_snapshot", return_value={"model": "CPU fixture", "memory_bytes": 1024, "errors": {}}):
            code = probe.run(args, observer_factory=ProcessFixture)
        return code, json.loads((args.output_dir / "report.json").read_text())

    def test_success_waits_for_cleanup_and_retains_bounded_source_evidence(self):
        with self.fixture() as (fixture, plan, args):
            code, report = self.collect(fixture, plan, args)
            self.assertEqual(code, 0, report)
            self.assertEqual(fixture.freeze_present, [True])
            self.assertEqual(report["validation"], "incomplete")
            self.assertEqual(report["process_reaping"]["status"], "unknown")
            self.assertFalse(report["admission_approved"])
            result = report["cases"][0]
            self.assertEqual(result["checks"], dict.fromkeys(probe.CHECKS, "passed"))
            self.assertEqual(result["runtime_peak"]["bytes"], 12345)
            self.assertEqual(result["post_settlement_allocator"]["status"], "unknown")
            self.assertEqual(result["outcome_deltas"][probe.COMPLETED], 1)
            observations = [json.loads(line) for line in (args.output_dir / "observations.jsonl").read_text().splitlines()]
            self.assertTrue(any(item["values"][probe.GAUGES[3]] == 1 for item in observations))
            self.assertEqual(observations[-1]["values"][probe.GAUGES[3]], 0)
            self.assertGreaterEqual(observations[-1]["started_monotonic_ns"], result["request"]["end_monotonic_ns"])
            self.assertNotIn("process_peak_rss_bytes", observations[0]["host"])
            self.assertNotIn(b"historical diagnostic", (args.output_dir / "diagnostics.jsonl").read_bytes())
            for name, identity in report["files"].items():
                self.assertEqual(identity["sha256"], probe.sha(args.output_dir / name))

    def test_missing_or_duplicate_peak_stays_unknown_without_aggregate_substitution(self):
        for changes in ({"omit_peak": True}, {"duplicate_peak": True}):
            with self.subTest(changes), self.fixture(cleanup_delay=.15, **changes) as (fixture, plan, args):
                code, report = self.collect(fixture, plan, args)
                self.assertEqual(code, 1)
                result = report["cases"][0]
                self.assertEqual(result["checks"]["settlement"], "passed")
                self.assertEqual(result["checks"]["runtime_peak"], "unknown")
                self.assertIsNone(result["runtime_peak"]["bytes"])

    def test_cleanup_after_settlement_deadline_stays_unknown_and_stops_progression(self):
        with self.fixture(cleanup_delay=.15) as (fixture, plan, args):
            plan["limits"]["settlement_timeout_ms"] = 50
            second = copy.deepcopy(plan["cases"][0])
            second["case_id"] = "must-not-run"
            plan["cases"].append(second)
            code, report = self.collect(fixture, plan, args)
            self.assertEqual(code, 1)
            self.assertEqual(fixture.requests, 1)
            self.assertEqual(report["cases"][0]["checks"]["settlement"], "unknown")
            self.assertEqual(report["cases"][1]["status"], "skipped")
            observations = [json.loads(line) for line in (args.output_dir / "observations.jsonl").read_text().splitlines()]
            self.assertEqual(observations[-1]["values"][probe.GAUGES[3]], 1)
            self.assertEqual(observations[-1]["values"][probe.COMPLETED], 0)

    def test_early_eos_remains_unreached_and_next_settled_case_can_run(self):
        with self.fixture(output_tokens=2) as (fixture, plan, args):
            second = copy.deepcopy(plan["cases"][0])
            second["case_id"] = "second-case"
            plan["cases"].append(second)
            code, report = self.collect(fixture, plan, args)
            self.assertEqual(code, 1)
            self.assertEqual(fixture.requests, 2)
            self.assertTrue(all(case["checks"]["full_output"] == "unreached" for case in report["cases"]))
            self.assertTrue(all(case["checks"]["settlement"] == "passed" for case in report["cases"]))

    def test_identity_pressure_execution_and_settlement_faults_stop_progression(self):
        changes_and_check = [({"bad_epoch": True}, "identity"), ({"bad_header": True}, "identity"),
                             ({"pressure_after_start": True}, "host_pressure"), ({"incomplete_stream": True}, "execution"),
                             ({"skip_cleanup": True}, "settlement"), ({"extra_outcome": True}, "settlement")]
        for changes, check in changes_and_check:
            with self.subTest(changes), self.fixture(**changes) as (fixture, plan, args):
                second = copy.deepcopy(plan["cases"][0])
                second["case_id"] = "must-not-run"
                plan["cases"].append(second)
                code, report = self.collect(fixture, plan, args)
                self.assertEqual(code, 1)
                self.assertEqual(fixture.requests, 1)
                self.assertNotEqual(report["cases"][0]["checks"][check], "passed")
                self.assertEqual(report["cases"][1]["status"], "skipped")
                self.assertTrue((args.output_dir / "original-case.sse").is_file())

    def test_missing_cleanup_metric_never_establishes_drained_service(self):
        with self.fixture(missing_metric=probe.GAUGES[-1]) as (fixture, plan, args):
            code, report = self.collect(fixture, plan, args)
            self.assertEqual(code, 1)
            self.assertEqual(fixture.requests, 0)
            self.assertIn("not drained", report["error"])

    def test_observation_budget_and_sample_limit_leave_unknown_memory(self):
        for key, value in (("observation_budget_ms", 100), ("max_samples", 2)):
            with self.subTest(key), self.fixture() as (fixture, plan, args):
                plan["limits"][key] = value
                if key == "observation_budget_ms":
                    original = ProcessFixture.sample
                    def slow_sample(instance):
                        time.sleep(.12)
                        return original(instance)
                    with patch.object(ProcessFixture, "sample", slow_sample):
                        code, report = self.collect(fixture, plan, args)
                else:
                    code, report = self.collect(fixture, plan, args)
                self.assertEqual(code, 1)
                if report["cases"][0]["checks"]:
                    self.assertNotEqual(report["cases"][0]["checks"]["external_memory"], "passed")
                self.assertTrue((args.output_dir / "observations.jsonl").is_file())

    def test_changed_manifest_and_existing_output_fail_before_execution(self):
        with self.fixture() as (fixture, plan, args):
            fixture.ready["worker"]["model_manifest"]["provider"] = "unexpected"
            code, report = self.collect(fixture, plan, args)
            self.assertEqual(code, 1)
            self.assertEqual(fixture.requests, 0)
            self.assertIn("model revision mismatch", report["error"])
            sentinel = (args.output_dir / "report.json").read_bytes()
            with self.assertRaises(FileExistsError):
                self.collect(fixture, plan, args)
            self.assertEqual((args.output_dir / "report.json").read_bytes(), sentinel)

    def test_lock_requires_matching_inode_and_actual_open_description_ownership(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lock"
            path.touch()
            other = Path(temporary) / "other"
            other.touch()
            with path.open("rb") as owner, path.open("rb") as foreign, other.open("rb") as wrong:
                with self.assertRaises(ValueError):
                    probe.verify_lock(path, owner.fileno())
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertEqual(probe.verify_lock(path, owner.fileno())["inode"], path.stat().st_ino)
                with self.assertRaises(BlockingIOError):
                    probe.verify_lock(path, foreign.fileno())
                with self.assertRaises(ValueError):
                    probe.verify_lock(path, wrong.fileno())

    def test_plan_cannot_omit_safety_checks_or_use_unbounded_json_numbers(self):
        with self.fixture() as (fixture, plan, args):
            plan["cases"][0]["required_checks"] = ["full_output"]
            args.plan_file.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "Each case must require"):
                probe.read_plan(args.plan_file)
            for payload in ('{"a": 1, "a": 2}', '{"a": 1e999}', '{"a": NaN}'):
                with self.subTest(payload), self.assertRaises(ValueError):
                    probe.strict_json(payload)

    def test_process_collector_rejects_replacement_during_rss_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / "service"
            binary.write_bytes(b"Original executable inode fixture.")
            ready = ready_fixture()
            epoch = ready["worker"]["worker_epoch"]
            service = {"pid": 101, "parent_pid": 50, "uid": 500, "started": "first", "command": str(binary)}
            worker = {"pid": 102, "parent_pid": 101, "uid": 500, "started": "first",
                      "command": f"python /worker/text_worker.py --protocol apxinf-worker/2.0 --worker-epoch {epoch} --model /original/cpu/fixture"}
            args = SimpleNamespace(service_pid=101, worker_pid=102, service_binary=binary, base_url="http://127.0.0.1:8080")
            response = SimpleNamespace(stdout="p101\n", returncode=0)
            mapped = SimpleNamespace(stdout=f"p101\nftxt\nD{binary.stat().st_dev:#x}\ni{binary.stat().st_ino}\nn{binary}\n", returncode=0)
            with patch.object(probe.sys, "platform", "darwin"), \
                 patch.object(probe, "process_identity", side_effect=[service, worker]), \
                 patch.object(probe.subprocess, "run", side_effect=[response, mapped, mapped]):
                observer = probe.ProcessObserver(args, ready)
            replacement = {**service, "started": "second"}
            with patch.object(probe, "process_identity", side_effect=[service, replacement]), \
                 patch.object(probe.subprocess, "run", side_effect=[mapped, SimpleNamespace(stdout="400\n", returncode=0)]):
                with self.assertRaisesRegex(ValueError, "during RSS"):
                    observer.sample()
            replacement = binary.with_suffix(".replacement")
            replacement.write_bytes(binary.read_bytes())
            replacement.replace(binary)
            with self.assertRaisesRegex(ValueError, "executable file identity changed"):
                observer.check_executable()

    def test_metrics_before_client_completion_cannot_end_settlement_observation(self):
        with self.fixture() as (fixture, plan, args):
            gate = threading.Event()
            sample_count = 0
            original_stream, original_sample = probe.stream_case, ProcessFixture.sample

            def delayed_stream(*arguments):
                self.assertTrue(gate.wait(1))
                return original_stream(*arguments)

            def delayed_sample(instance):
                nonlocal sample_count
                sample_count += 1
                if sample_count == 2:
                    gate.set()
                    time.sleep(.15)
                return original_sample(instance)

            with patch.object(probe, "stream_case", delayed_stream), patch.object(ProcessFixture, "sample", delayed_sample):
                code, report = self.collect(fixture, plan, args)
            self.assertEqual(code, 0, report)
            self.assertEqual(report["cases"][0]["checks"]["settlement"], "passed")
            observations = [json.loads(line) for line in (args.output_dir / "observations.jsonl").read_text().splitlines()]
            self.assertEqual(observations[1]["values"][probe.COMPLETED], 0)
            self.assertEqual(observations[-1]["values"][probe.COMPLETED], 1)

    def test_runtime_peak_rejects_invalid_nested_metrics_and_preserves_raw_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "source"
            source.touch()
            diagnostic = probe.Diagnostics(source, directory / "retained", 128)
            try:
                diagnostic.records = [{"event": "request_settled", "request_id": "fixture", "worker_metrics": []}]
                request = {"request_id": "fixture", "success": True, "output_tokens": 1, "input_tokens": 1}
                self.assertEqual(probe.runtime_peak(diagnostic, request)["status"], "unknown")
                with source.open("ab") as output:
                    output.write(b"x" * 129)
                diagnostic.collect()
                self.assertEqual((directory / "retained").read_bytes(), b"x" * 128)
                self.assertTrue(diagnostic.errors)
            finally:
                diagnostic.close()


if __name__ == "__main__":
    unittest.main()
