from __future__ import annotations

import importlib.util
import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_evaluation_protocol_test", HERE / "run_evaluation.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = load_runner()


class FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        del token_ids, skip_special_tokens
        return "42"


class FakeSampler:
    def window(self, start, end):
        del start, end
        return {"sample_count": 0}


class BoundaryEvaluator:
    def request(self, case):
        passed = case["id"] == "recovery" or case["actual_prompt_tokens"] == 32640
        return {
            "success": passed,
            "functional_pass": passed,
            "error": None if passed else "HTTPError: 400 capacity exceeded",
        }


class EvaluationHandler(BaseHTTPRequestHandler):
    mode = "happy"

    def log_message(self, format, *args):
        del format, args

    def do_GET(self):
        if self.path != "/health":
            self.send_error(404)
            return
        payload = json.dumps({"status": "ok", "fallback_active": False}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        if self.path != "/v1/evaluations/generate":
            self.send_error(404)
            return
        size = int(self.headers.get("Content-Length", "0"))
        json.loads(self.rfile.read(size))
        second_index = 2 if self.mode == "index_gap" else 1
        second_request_id = "req-b" if self.mode == "crossed_request" else "req-a"
        events = [
            {"type": "token", "request_id": "req-a", "index": 0, "token_id": 7},
            {
                "type": "token",
                "request_id": second_request_id,
                "index": second_index,
                "token_id": 8,
            },
            {
                "type": "done",
                "request_id": "req-a",
                "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
            },
        ]
        body = "".join(
            f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
            for event in events
        ) + "data: [DONE]\n\n"
        payload = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class RunnerProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), EvaluationHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"
        cls.case = {
            "id": "fake-case",
            "suite": "functional",
            "category": "exact_retrieval",
            "roles": [],
            "input_ids": list(range(8)),
            "input_ids_sha256": "0" * 64,
            "max_new_tokens": 2,
            "temperature": 0.0,
            "ignore_eos": True,
            "validation": "normalized_exact",
            "expected": "42",
        }

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=3)

    def request(self, mode):
        EvaluationHandler.mode = mode
        return runner.request_evaluation_api(
            self.case,
            self.base_url,
            5.0,
            FakeTokenizer(),
            FakeSampler(),
        )

    def test_accepts_complete_exact_sse_trajectory(self):
        row = self.request("happy")
        self.assertTrue(row["success"])
        self.assertTrue(row["functional_pass"])
        self.assertEqual(row["output_ids"], [7, 8])
        self.assertEqual(row["usage"]["total_tokens"], 10)

    def test_rejects_token_index_gap(self):
        row = self.request("index_gap")
        self.assertFalse(row["success"])
        self.assertIn("token index gap", row["error"])

    def test_rejects_crossed_request_ids(self):
        row = self.request("crossed_request")
        self.assertFalse(row["success"])
        self.assertIn("request_id changed", row["error"])

    def test_context_boundary_failure_is_recovered_and_not_overclaimed(self):
        cases = [
            {
                "id": f"context-{length}",
                "actual_prompt_tokens": length,
                "category": "retrieval_early",
            }
            for length in (32640, 32768)
        ]
        recovery = {
            "id": "recovery",
            "actual_prompt_tokens": 1024,
            "category": "exact_retrieval",
        }
        result, rows = runner.run_context_staircase(
            BoundaryEvaluator(),
            cases,
            self.base_url,
            5.0,
            recovery,
            32768,
        )
        self.assertEqual(result["max_verified_prompt_tokens"], 32640)
        self.assertEqual(result["first_failed_prompt_tokens"], 32768)
        self.assertEqual(result["verified_cases_at_max_context"], 1)
        self.assertTrue(result["service_healthy_after_failure"])
        self.assertEqual(rows[-1]["phase"], "context_recovery")


if __name__ == "__main__":
    unittest.main()
