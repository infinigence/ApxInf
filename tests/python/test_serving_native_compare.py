import contextlib
import hashlib
import fcntl
from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/serving"))
SPEC = importlib.util.spec_from_file_location("native_runtime_compare", ROOT / "benchmarks/serving/native_runtime_compare.py")
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


class NativeComparisonTests(unittest.TestCase):
    @contextlib.contextmanager
    def deployment(self, directory, content="中文 answer", digest=None):
        worker = directory / "worker"
        worker.write_bytes(b"original worker fixture")
        model = directory / "model"
        model.mkdir()
        ready = {"ready": True, "model": "minicpm5", "worker": {
            "protocol": "apxinf-worker/2.7", "model_path": str(model),
            "runtime": {"executable_sha256": digest or hashlib.sha256(worker.read_bytes()).hexdigest()},
            "model_manifest": {"execution": {"allocator_cache_limit_bytes": 2147483648, "memory_limit_bytes": 10737418240}}}}
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(ready).encode())

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append(body)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                events = [
                    {"choices": [{"delta": {"content": content}, "finish_reason": None}], "usage": None},
                    {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": None},
                    {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}},
                ]
                for event in events:
                    self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            yield worker, model, f"http://127.0.0.1:{server.server_port}", requests
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def invoke(self, directory, worker, model, base, receipt_mode="valid"):
        corpus = directory / "corpus.json"
        corpus.write_text(json.dumps({"cases": [{"id": "question", "messages": [{"role": "user", "content": "Explain caching."}], "max_tokens": 8}]}))
        probe = directory / "probe"
        probe.write_bytes(b"original probe fixture")
        direct = directory / "direct.jsonl"
        direct.write_text(json.dumps({"kind": "metadata", "checkout_commit": "test-source", "checkout_dirty": False,
                                      "executable_sha256": hashlib.sha256(probe.read_bytes()).hexdigest(),
                                      "model_path": str(model), "allocator_cache_limit_bytes": 2147483648,
                                      "memory_limit_bytes": 10737418240}) + "\n" +
                          json.dumps({"kind": "result", "id": "question", "repeat": 0, "input_tokens": 5,
                                      "completion_tokens": 3, "content": "中文 answer", "finish_reason": "stop"}) + "\n")
        result = directory / "comparison.json"
        lock = directory / "metal.lock"
        argv = ["native_runtime_compare", "--base-url", base, "--corpus", str(corpus), "--direct-jsonl", str(direct),
                "--native-worker", str(worker), "--native-probe", str(probe), "--output", str(result),
                "--metal-lock", str(lock), "--lock-owned-by-parent"]
        if receipt_mode != "none":
            receipt = {"source_commit": "test-source", "source_dirty": False, "binaries": {
                "metal_text_probe": {"bytes": probe.stat().st_size, "sha256": hashlib.sha256(probe.read_bytes()).hexdigest()},
                "apxinf-native-worker": {"bytes": worker.stat().st_size, "sha256": hashlib.sha256(worker.read_bytes()).hexdigest()}}}
            if receipt_mode == "stale-probe":
                receipt["binaries"]["metal_text_probe"]["sha256"] = "0" * 64
            receipt_path = directory / "build-receipt.json"
            receipt_path.write_text(json.dumps(receipt))
            argv.extend(["--build-receipt", str(receipt_path)])
        with lock.open("w") as owned_lock:
            fcntl.flock(owned_lock, fcntl.LOCK_EX)
            with patch.object(sys, "argv", argv), patch.object(comparison, "checkout_commit", return_value="test-source"):
                comparison.main()
        return result

    def test_actual_http_stream_matches_complete_output_and_server_usage(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.deployment(directory) as (worker, model, base, requests):
                result = self.invoke(directory, worker, model, base)
            report = json.loads(result.read_text())
            self.assertTrue(report["all_outputs_equal"])
            self.assertEqual(report["source_binding"]["status"], "verified_external_build_receipt")
            self.assertFalse(report["model_binding"]["artifact_equality_verified"])
            self.assertEqual(report["rows"][0]["http"]["usage"]["completion_tokens"], 3)
            self.assertEqual(requests[0]["stream_options"], {"include_usage": True})

    def test_changed_http_answer_is_preserved_and_fails_the_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.deployment(directory, content="different answer") as (worker, model, base, _):
                with self.assertRaises(SystemExit) as failure:
                    self.invoke(directory, worker, model, base)
            self.assertEqual(failure.exception.code, 1)
            report = json.loads((directory / "comparison.json").read_text())
            self.assertFalse(report["all_outputs_equal"])
            self.assertEqual(report["rows"][0]["http"]["content"], "different answer")

    def test_wrong_worker_identity_fails_before_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.deployment(directory, digest="0" * 64) as (worker, model, base, requests):
                with self.assertRaisesRegex(ValueError, "worker differs"):
                    self.invoke(directory, worker, model, base)
                self.assertEqual(requests, [])
            self.assertFalse((directory / "comparison.json").exists())

    def test_stale_probe_in_build_receipt_fails_before_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.deployment(directory) as (worker, model, base, requests):
                with self.assertRaisesRegex(ValueError, "metal_text_probe differs"):
                    self.invoke(directory, worker, model, base, receipt_mode="stale-probe")
                self.assertEqual(requests, [])
            self.assertFalse((directory / "comparison.json").exists())

    def test_checkout_metadata_without_receipt_is_not_source_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.deployment(directory) as (worker, model, base, _):
                result = self.invoke(directory, worker, model, base, receipt_mode="none")
            report = json.loads(result.read_text())
            self.assertEqual(report["source_binding"]["status"], "unverified")
            self.assertNotIn("source_commit", report)
