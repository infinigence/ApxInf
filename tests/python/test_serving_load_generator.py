"""Original CPU tests for fixed arrivals and absolute HTTP deadlines."""

from __future__ import annotations

import argparse
from concurrent.futures import Future
from contextlib import contextmanager, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("apxinf_load_benchmark", ROOT / "benchmarks/serving/http_benchmark.py")
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class DeferredFuture:
    def __init__(self, clock, action):
        self.clock = clock
        self.action = action
        self.value = None

    def done(self):
        return self.value is not None

    def result(self):
        if self.value is None:
            self.clock.sleep(0.25)
            self.value = self.action()
        return self.value


class ControlledExecutor:
    def __init__(self, clock, *, immediate=False):
        self.clock = clock
        self.immediate = immediate
        self.submitted = 0
        self.max_workers = None

    def __call__(self, *, max_workers):
        self.max_workers = max_workers
        return self

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        return False

    def submit(self, action, *args):
        self.submitted += 1
        if self.immediate:
            future = Future()
            future.set_result(action(*args))
            return future
        return DeferredFuture(self.clock, lambda: action(*args))


class LoadTests(unittest.TestCase):
    def request(self, clock):
        def run(index):
            started = clock()
            clock.sleep(0.5)
            return {"index": index, "success": True, "start_monotonic_s": started,
                    "ttft_s": 0.1, "e2e_s": 0.5, "tpot_s": 0.1,
                    "inter_output_s": [0.1], "output_tokens": 5}
        return run

    def test_capacity_drops_do_not_enter_an_executor_waiting_queue(self):
        clock = Clock()
        executor = ControlledExecutor(clock)
        samples, wall = bench.open_loop_requests(self.request(clock), 6, 2, 2,
                                                  clock=clock, sleep=clock.sleep,
                                                  executor_factory=executor)
        self.assertEqual(executor.submitted, 2)
        self.assertEqual(executor.max_workers, 2)
        self.assertEqual([sample["index"] for sample in samples], list(range(6)))
        self.assertEqual(sum(sample["client_capacity_drop"] for sample in samples), 4)
        self.assertEqual([sample["scheduled_offset_s"] for sample in samples], [0, .5, 1, 1.5, 2, 2.5])
        self.assertEqual(wall, 4)
        dispatched = [sample for sample in samples if not sample["client_capacity_drop"]]
        self.assertTrue(all(sample["dispatch_delay_s"] >= 2 for sample in dispatched))
        self.assertTrue(all(sample["dispatch_monotonic_s"] == sample["start_monotonic_s"] for sample in dispatched))
        summary = bench.summarize(samples, wall, .2, 1)
        self.assertEqual(summary["slo_passing_requests"], 2)
        self.assertEqual(summary["goodput_requests_per_s"], .5)
        self.assertEqual(summary["arrival_slo_passing_requests"], 0)
        self.assertEqual(summary["arrival_goodput_requests_per_s"], 0)
        self.assertEqual(summary["errors"], 4)
        self.assertEqual(summary["arrival_to_completion_s"]["n"], 6)

    def test_completed_requests_release_client_capacity(self):
        clock = Clock()
        executor = ControlledExecutor(clock, immediate=True)
        samples, wall = bench.open_loop_requests(self.request(clock), 3, 1, 1,
                                                  clock=clock, sleep=clock.sleep,
                                                  executor_factory=executor)
        self.assertEqual(executor.submitted, 3)
        self.assertEqual(wall, 2.5)
        self.assertTrue(all(sample["dispatch_delay_s"] == 0 for sample in samples))
        self.assertTrue(all(sample["arrival_to_completion_s"] == .5 for sample in samples))
        summary = bench.summarize(samples, wall)
        self.assertIsNone(summary["goodput_requests_per_s"])
        self.assertIsNone(summary["arrival_goodput_requests_per_s"])

    def test_worker_exception_is_a_dispatched_failure(self):
        clock = Clock()
        executor = ControlledExecutor(clock, immediate=True)

        def fail(index):
            clock.sleep(.25)
            raise OSError("fixture connection failure")

        samples, wall = bench.open_loop_requests(fail, 1, 1, 1, clock=clock,
                                                  sleep=clock.sleep, executor_factory=executor)
        self.assertFalse(samples[0]["success"])
        self.assertFalse(samples[0]["client_capacity_drop"])
        self.assertEqual(samples[0]["arrival_to_completion_s"], .25)
        self.assertEqual(bench.summarize(samples, wall)["dispatched_requests"], 1)

    def test_options_keep_closed_loop_default_and_reject_ambiguous_load(self):
        defaults = bench.parser().parse_args(["--output", "/tmp/unused.json"])
        self.assertEqual(bench.load_settings(defaults), ("closed_loop", [1, 2, 4], None, None))
        for rate, capacity, concurrency in [
            (1, None, None), (None, 1, None), (0, 1, None), (-1, 1, None),
            (float("nan"), 1, None), (float("inf"), 1, None), (1, 0, None),
            (1, -1, None), (1, 2, [1, 2]), (1, 2, [1]),
        ]:
            with self.subTest(rate=rate, capacity=capacity, concurrency=concurrency):
                with self.assertRaises(ValueError):
                    bench.load_settings(argparse.Namespace(arrival_rate=rate, max_in_flight=capacity,
                                                           concurrency=concurrency))
        self.assertEqual(bench.load_settings(argparse.Namespace(arrival_rate=1, max_in_flight=2,
                                                                concurrency=[2])),
                         ("open_loop", [2], 1, 2))

    def test_report_preserves_existing_fields_and_declares_open_loop(self):
        with tempfile.TemporaryDirectory(prefix="apxinf-load-report-") as directory:
            output = Path(directory) / "report.json"
            args = bench.parser().parse_args([
                "--output", str(output), "--samples", "2", "--warmup", "0",
                "--arrival-rate", "100", "--max-in-flight", "1",
                "--slo-ttft-ms", "100", "--slo-e2e-ms", "1000",
            ])

            def complete(*args):
                return {"index": args[-1], "success": True, "start_monotonic_s": time.perf_counter(),
                        "ttft_s": 0, "e2e_s": 0, "tpot_s": None,
                        "inter_output_s": [], "output_tokens": 1}

            with patch.object(bench, "request_json", return_value=(200, {"data": [{"id": "fixture"}]})), \
                    patch.object(bench, "metrics_snapshot", return_value={"values": {}}), \
                    patch.object(bench, "stream_request", side_effect=complete), redirect_stdout(io.StringIO()):
                status = bench.run(args)
            report = json.loads(output.read_text())
            self.assertEqual(status, 0)
            self.assertEqual(report["format"], "apxinf-serving-benchmark-v1")
            self.assertEqual(report["load_mode"], "open_loop")
            self.assertEqual(report["arrival_rate_per_s"], 100)
            self.assertEqual(report["max_in_flight"], 1)
            self.assertEqual(report["runs"][0]["concurrency"], 1)
            summary = report["runs"][0]["summary"]
            self.assertEqual(summary["offered_requests"], 2)
            self.assertEqual(summary["dispatched_requests"], 2)
            self.assertEqual(summary["arrival_slo_passing_requests"], 2)
            self.assertIn("goodput_requests_per_s", summary)

    def test_invalid_slo_targets_fail_before_http_requests(self):
        for option in ("--slo-ttft-ms", "--slo-e2e-ms"):
            for value in ("nan", "inf", "0", "-1"):
                with self.subTest(option=option, value=value):
                    args = bench.parser().parse_args(["--output", "/tmp/unused.json", option, value])
                    with patch.object(bench, "request_json") as request:
                        with self.assertRaisesRegex(ValueError, "finite and positive"):
                            bench.run(args)
                        request.assert_not_called()

    def test_failed_warmup_keeps_evidence_and_fails_the_run(self):
        with tempfile.TemporaryDirectory(prefix="apxinf-warmup-report-") as directory:
            output = Path(directory) / "report.json"
            args = bench.parser().parse_args(["--output", str(output), "--samples", "1",
                                              "--warmup", "1", "--concurrency", "1"])
            failure = bench.failed_arrival(0, "fixture warmup failure")
            success = {"index": 0, "success": True, "ttft_s": .1, "e2e_s": .2,
                       "tpot_s": None, "inter_output_s": [], "output_tokens": 1}
            printed = io.StringIO()
            with patch.object(bench, "request_json", return_value=(200, {"data": [{"id": "fixture"}]})), \
                    patch.object(bench, "metrics_snapshot", return_value={"values": {}}), \
                    patch.object(bench, "stream_request", side_effect=[failure, success]), redirect_stdout(printed):
                status = bench.run(args)
            report = json.loads(output.read_text())
            self.assertEqual(status, 1)
            self.assertEqual(report["warmup_summary"], {"attempts": 1, "completed": 0, "errors": 1})
            self.assertEqual(report["warmup"][0]["error"], "fixture warmup failure")
            self.assertEqual(report["runs"][0]["summary"]["completed"], 1)
            self.assertEqual(report["runs"][0]["summary"]["errors"], 0)
            self.assertEqual(json.loads(printed.getvalue())["warmup_summary"]["errors"], 1)


def content_frames():
    messages = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 2}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    ]
    return [f"event: {message['type']}\ndata: {json.dumps(message)}\n\n".encode() for message in messages]


@contextmanager
def local_server(mode):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *unused):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            with self.server.count_lock:
                self.server.requests += 1
            try:
                if mode == "headers":
                    self.connection.sendall(b"HTTP/1.1 200 OK\r\nX-Slow: ")
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Connection", "close")
                    self.end_headers()
                if mode == "complete":
                    self.wfile.write(b"".join(content_frames()))
                    self.wfile.flush()
                    return
                if mode == "content_then_heartbeat":
                    self.wfile.write(b"".join(content_frames()[:3]))
                    self.wfile.flush()
                if mode == "partial":
                    self.wfile.write(b"data: ")
                until = time.monotonic() + 1
                while time.monotonic() < until:
                    self.wfile.write(b"x" if mode in {"partial", "headers"} else b": keepalive\n\n")
                    self.wfile.flush()
                    time.sleep(.01)
            except OSError:
                pass
            finally:
                self.close_connection = True
                self.server.handler_done.set()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    server.requests = 0
    server.count_lock = threading.Lock()
    server.handler_done = threading.Event()
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


class DeadlineTests(unittest.TestCase):
    def test_heartbeats_and_partial_lines_cannot_extend_deadline(self):
        for mode in ("heartbeat", "partial", "headers", "content_then_heartbeat"):
            with self.subTest(mode=mode), local_server(mode) as (url, server):
                owned = bench.connect(url, .08)
                with patch.object(bench, "connect", return_value=owned):
                    result = bench.stream_request(url, "fixture", "hello", timeout=.08)
                self.assertFalse(result["success"])
                self.assertIn("absolute deadline", result["error"])
                self.assertLess(result["e2e_s"], .5)
                self.assertEqual(owned.deadline_socket.fileno(), -1)
                self.assertTrue(server.handler_done.wait(.5))
                if mode == "content_then_heartbeat":
                    self.assertIsNotNone(result["ttft_s"])

    def test_complete_terminal_frame_succeeds_and_closes_owned_connection(self):
        with local_server("complete") as (url, server):
            owned = bench.connect(url, 1)
            with patch.object(bench, "connect", return_value=owned):
                result = bench.stream_request(url, "fixture", "hello", timeout=1)
            self.assertTrue(result["success"], result["error"])
            self.assertEqual(result["output_tokens"], 1)
            self.assertEqual(owned.deadline_socket.fileno(), -1)
            self.assertTrue(server.handler_done.wait(.5))

    def test_absolute_deadline_interrupts_request_transmission(self):
        sending, receiving = socket.socketpair()
        connection = bench.DeadlineConnection("127.0.0.1")
        connection.deadline_socket = sending
        sending.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        try:
            with self.assertRaisesRegex(TimeoutError, "absolute deadline"):
                with bench.absolute_deadline(connection, .05):
                    sending.sendall(b"x" * 1_000_000)
        finally:
            sending.close()
            receiving.close()

    def test_open_loop_drain_is_bounded_by_request_deadlines(self):
        with local_server("heartbeat") as (url, server):
            samples, wall = bench.open_loop_requests(
                lambda index: bench.stream_request(url, "fixture", "hello", timeout=.08, index=index),
                4, 100, 2)
            self.assertEqual(len(samples), 4)
            self.assertLess(wall, .5)
            self.assertTrue(all(not sample["success"] for sample in samples))
            self.assertEqual(sum(sample["client_capacity_drop"] for sample in samples), 2)
            self.assertEqual(server.requests, 2)


if __name__ == "__main__":
    unittest.main()
