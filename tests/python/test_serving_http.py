"""Check HTTP measurement and client task boundaries without loading a model."""

import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "benchmarks" / "serving"
sys.path.insert(0, str(BENCHMARKS))
import http_benchmark as bench
import claude_code_tasks as client
import lifecycle_checks as lifecycle


def feed(state, kind, elapsed=0, **fields):
    state.feed(kind, json.dumps({"type": kind, **fields}), elapsed)


def begin(state):
    feed(state, "message_start", message={"content": [], "usage": {"input_tokens": 12, "output_tokens": 0}})


def finish(state, reason="end_turn", tokens=2):
    feed(state, "message_delta", delta={"stop_reason": reason}, usage={"output_tokens": tokens})
    feed(state, "message_stop")


class FakeResponse(io.BytesIO):
    def __init__(self, body, status=200, content_type="text/event-stream"):
        super().__init__(body)
        self.status = status
        self.content_type = content_type

    def getheader(self, name, default=None):
        return self.content_type if name.lower() == "content-type" else default


class FakeConnection:
    def __init__(self, response):
        self.response = response
        self.closed = False

    def request(self, *args, **kwargs):
        pass

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True


def original_text_frames():
    documents = [
        {"type": "message_start", "message": {"content": [], "usage": {"input_tokens": 9, "output_tokens": 0}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Local result"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
        {"type": "message_stop"},
    ]
    return [f"event: {document['type']}\ndata: {json.dumps(document)}\n\n".encode() for document in documents]


class HttpFaultTests(unittest.TestCase):
    def measure(self, response):
        connection = FakeConnection(response)
        with patch.object(bench, "connect", return_value=connection):
            result = bench.stream_request("http://127.0.0.1:8080", "test-model", "Original test request.")
        self.assertTrue(connection.closed)
        return result

    def test_rejection_preserves_http_status_and_error(self):
        result = self.measure(FakeResponse(b'{"error":{"type":"rate_limit_error"}}', status=429))
        self.assertFalse(result["success"])
        self.assertEqual(result["http_status"], 429)
        self.assertIn("rate_limit_error", result["error"])
        self.assertIsNone(result["ttft_s"])

    def test_wrong_content_type_never_counts_as_a_stream(self):
        result = self.measure(FakeResponse(b'{}', content_type="application/json"))
        self.assertFalse(result["success"])
        self.assertIn("text/event-stream", result["error"])

    def test_disconnect_after_text_preserves_partial_observation(self):
        result = self.measure(FakeResponse(b"".join(original_text_frames()[:3])))
        self.assertFalse(result["success"])
        self.assertIsNotNone(result["ttft_s"])
        self.assertGreater(result["output_characters"], 0)
        self.assertIn("terminal", result["error"])

    def test_error_after_text_is_failure_even_with_http_200(self):
        body = b"".join(original_text_frames()[:3])
        body += b'event: error\ndata: {"type":"error","error":{"type":"worker_lost"}}\n\n'
        result = self.measure(FakeResponse(body))
        self.assertFalse(result["success"])
        self.assertEqual(result["http_status"], 200)
        self.assertIn("worker_lost", result["error"])

    def test_malformed_json_is_failure(self):
        result = self.measure(FakeResponse(b"event: message_start\ndata: {broken}\n\n"))
        self.assertFalse(result["success"])
        self.assertIn("JSONDecodeError", result["error"])

    def test_extra_event_after_terminal_is_failure(self):
        body = b"".join(original_text_frames()) + b'event: ping\ndata: {"type":"ping"}\n\n'
        result = self.measure(FakeResponse(body))
        self.assertFalse(result["success"])
        self.assertIn("after its terminal", result["error"])

    def test_complete_response_records_reported_usage(self):
        result = self.measure(FakeResponse(b"".join(original_text_frames())))
        self.assertTrue(result["success"], result["error"])
        self.assertEqual(result["input_tokens"], 9)
        self.assertEqual(result["output_tokens"], 2)
        self.assertIsNotNone(result["tpot_s"])


class StreamTests(unittest.TestCase):
    def test_sse_frame_boundaries_and_multiline_data(self):
        response = io.BytesIO(b": keepalive\r\nevent: ping\r\ndata: {\r\ndata: \"type\":\"ping\"}\r\n\r\n")
        self.assertEqual(list(bench.sse_events(response)), [("ping", '{\n"type":"ping"}')])

    def test_partial_frame_is_failure(self):
        with self.assertRaises(bench.ProtocolError):
            list(bench.sse_events(io.BytesIO(b"event: message_stop\ndata: {}")))

    def test_text_order_and_cumulative_usage(self):
        state = bench.StreamResult("anthropic")
        begin(state)
        feed(state, "content_block_start", index=0, content_block={"type": "text", "text": ""})
        feed(state, "content_block_delta", .2, index=0, delta={"type": "text_delta", "text": "One"})
        feed(state, "content_block_delta", .5, index=0, delta={"type": "text_delta", "text": " two"})
        feed(state, "content_block_stop", index=0)
        finish(state)
        state.require_finished()
        self.assertEqual(state.text, "One two")
        self.assertEqual(state.content_times, [.2, .5])
        self.assertEqual(state.usage, {"input_tokens": 12, "output_tokens": 2})

    def test_tool_json_is_reconstructed_after_stop(self):
        state = bench.StreamResult("anthropic")
        begin(state)
        feed(state, "content_block_start", index=0,
             content_block={"type": "tool_use", "id": "call_original", "name": "Read", "input": {}})
        feed(state, "content_block_delta", .1, index=0,
             delta={"type": "input_json_delta", "partial_json": '{"file_'})
        feed(state, "content_block_delta", .2, index=0,
             delta={"type": "input_json_delta", "partial_json": 'path":"marker.txt"}'})
        self.assertEqual(state.tool_calls, [])
        feed(state, "content_block_stop", index=0)
        finish(state, reason="tool_use", tokens=8)
        self.assertEqual(state.tool_calls[0]["input"], {"file_path": "marker.txt"})

    def test_stream_error_never_counts_as_completion(self):
        state = bench.StreamResult("anthropic")
        begin(state)
        with self.assertRaises(bench.ProtocolError):
            feed(state, "error", error={"type": "worker_error", "message": "Original test failure."})
        with self.assertRaises(bench.ProtocolError):
            state.require_finished()

    def test_terminal_requires_closed_content_and_stop_reason(self):
        state = bench.StreamResult("anthropic")
        begin(state)
        feed(state, "content_block_start", index=0, content_block={"type": "text", "text": ""})
        with self.assertRaises(bench.ProtocolError):
            finish(state)

    def test_event_payload_type_must_match(self):
        state = bench.StreamResult("anthropic")
        with self.assertRaises(bench.ProtocolError):
            state.feed("ping", '{"type":"message_stop"}', 0)

    def test_openai_terminal_and_usage(self):
        state = bench.StreamResult("openai")
        state.feed("message", json.dumps({"choices": [{"index": 0, "delta": {"content": "Hello"}}]}), .3)
        state.feed("message", json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}), .4)
        state.feed("message", json.dumps({"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 1}}), .4)
        state.feed("message", "[DONE]", .4)
        state.require_finished()
        self.assertEqual(state.content_times, [.3])
        self.assertEqual(state.usage["completion_tokens"], 1)

    def test_done_without_finish_reason_is_failure(self):
        with self.assertRaises(bench.ProtocolError):
            bench.StreamResult("openai").feed("message", "[DONE]", .1)


class AccountingTests(unittest.TestCase):
    def test_error_and_missing_usage_are_not_success_throughput(self):
        base = {"success": True, "ttft_s": .2, "e2e_s": 1, "tpot_s": .1,
                "inter_output_s": [.1], "output_tokens": 4}
        samples = [base, {**base, "output_tokens": None}, {**base, "success": False, "output_tokens": 100}]
        result = bench.summarize(samples, 2, slo_ttft_s=.3, slo_e2e_s=2)
        self.assertEqual(result["output_tokens_per_s"], 2)
        self.assertEqual(result["completed_requests_per_s"], 1)
        self.assertEqual(result["goodput_requests_per_s"], 1)
        self.assertEqual(result["usage_coverage"], .5)
        self.assertEqual(result["errors"], 1)

    def test_goodput_requires_an_explicit_target(self):
        self.assertIsNone(bench.summarize([], 1)["goodput_requests_per_s"])

    def test_metric_labels_and_memory_ranges(self):
        first = bench.metric_values('# ignored\napxinf_active_requests 1\napxinf_worker_active_bytes 2048\n'
                                    'apxinf_requests_total{status="cancelled"} 3\n')
        self.assertEqual(first['apxinf_requests_total{status="cancelled"}'], 3)
        summary = bench.summarize_metrics([{"values": first}, {"values": {"apxinf_worker_active_bytes": 4096}}])
        self.assertEqual(summary["apxinf_worker_active_bytes"]["max"], 4096)

    def test_loopback_policy_rejects_remote_and_credentials(self):
        for value in ("https://127.0.0.1", "http://example.com", "http://user@localhost", "http://localhost/v1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                bench.local_url(value)
        self.assertEqual(bench.local_url("http://127.0.0.1:8080").port, 8080)


class ClientTaskTests(unittest.TestCase):
    def test_ui_event_string_is_not_a_model_delta(self):
        self.assertFalse(client.has_content_delta({"type": "system", "subtype": "ui_invalidate", "event": "ui.render"}))
        self.assertTrue(client.has_content_delta({"type": "stream_event", "event": {"delta": {"type": "text_delta", "text": "Original"}}}))

    def test_environment_does_not_inherit_credentials_or_proxy(self):
        with patch.dict(os.environ, {"ANTHROPIC_AUTH_TOKEN": "secret-test", "HTTP_PROXY": "proxy-test"}):
            environment = client.client_environment("http://127.0.0.1:8080", "local", Path("/tmp/config-test"))
        self.assertNotIn("ANTHROPIC_AUTH_TOKEN", environment)
        self.assertNotIn("HTTP_PROXY", environment)
        self.assertEqual(environment["CLAUDE_CODE_DISABLE_THINKING"], "1")
        self.assertEqual(environment["ANTHROPIC_API_KEY"], "apxinf-local-test-only")

    def test_context_and_output_controls_use_documented_environment_keys(self):
        environment = client.client_environment("http://127.0.0.1:8080", "apxinf-local", Path("/tmp/config-test"), 2048, 16384)
        self.assertEqual(environment["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "16384")
        self.assertEqual(environment["CLAUDE_CODE_MAX_OUTPUT_TOKENS"], "2048")
        self.assertNotIn("DISABLE_COMPACT", environment)

    def test_edit_oracle_checks_behavior_without_running_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "score.py"
            path.write_text("def normalize_score(value):\n    return value / 10\n")
            self.assertFalse(client.check_edit(path))
            path.write_text("def normalize_score(value):\n    return max(0, min(1, value / 10))\n")
            self.assertTrue(client.check_edit(path))
            path.write_text("import os\ndef normalize_score(value):\n    return 0\n")
            self.assertFalse(client.check_edit(path))

    def test_claimed_marker_without_tool_does_not_pass(self):
        capture = {"events": [{"elapsed_s": 1, "event": {"type": "result", "result": "MARKER", "is_error": False}}],
                   "exit_code": 0}
        self.assertFalse(client.task_outcome("read", {"marker": "MARKER"}, Path("/tmp"), capture)["passed"])


class LifecycleGuardTests(unittest.TestCase):
    def identities(self, parent=400, epoch="original-epoch"):
        return [{"pid": 400, "parent_pid": 300, "uid": 501, "started": "original-start",
                 "command": "target/release/apxinf-serve --port 8080"},
                {"pid": 401, "parent_pid": parent, "uid": 501, "started": "original-start",
                 "command": "python -u /original/text_worker.py --protocol apxinf-worker/2.0 "
                            f"--worker-epoch {epoch} --model /original/model"}]

    def arguments(self, allowed=True):
        return SimpleNamespace(allow_worker_termination=allowed, service_pid=400, worker_pid=401,
                               base_url="http://127.0.0.1:8080")

    def readiness(self):
        return {"worker": {"worker_epoch": "original-epoch", "model_path": "/original/model"}}

    def test_worker_fault_requires_explicit_flag_before_process_inspection(self):
        with patch.object(lifecycle, "process_identity") as inspection:
            with self.assertRaisesRegex(RuntimeError, "termination needs"):
                lifecycle.verify_worker_target(self.arguments(False), self.readiness())
            inspection.assert_not_called()

    def test_unrelated_parent_or_worker_epoch_prevents_a_target(self):
        for identities in (self.identities(parent=900), self.identities(epoch="another-epoch")):
            with patch.object(lifecycle, "process_identity", side_effect=identities):
                with self.assertRaises(RuntimeError):
                    lifecycle.verify_worker_target(self.arguments(), self.readiness())

    def test_matching_worker_still_requires_service_listener_ownership(self):
        with patch.object(lifecycle, "process_identity", side_effect=self.identities()):
            with patch.object(lifecycle.subprocess, "run", return_value=SimpleNamespace(stdout="p999\n")):
                with self.assertRaisesRegex(RuntimeError, "listener"):
                    lifecycle.verify_worker_target(self.arguments(), self.readiness())

    def test_matching_processes_and_listener_produce_an_auditable_target(self):
        with patch.object(lifecycle, "process_identity", side_effect=self.identities()):
            with patch.object(lifecycle.subprocess, "run", return_value=SimpleNamespace(stdout="p400\n")):
                target = lifecycle.verify_worker_target(self.arguments(), self.readiness())
        self.assertEqual(target["worker"]["pid"], 401)
        self.assertEqual(target["service"]["pid"], 400)
        self.assertEqual(target["listener_port"], 8080)

    def test_listener_file_descriptor_fields_do_not_change_process_ownership(self):
        with patch.object(lifecycle, "process_identity", side_effect=self.identities()):
            with patch.object(lifecycle.subprocess, "run", return_value=SimpleNamespace(stdout="p400\nf9\n")):
                target = lifecycle.verify_worker_target(self.arguments(), self.readiness())
        self.assertEqual(target["service"]["pid"], 400)

    def test_idle_requires_zero_reservations_and_present_metrics(self):
        sample = {"error": None, "values": {key: 0 for key in lifecycle.IDLE}}
        self.assertTrue(lifecycle.idle(sample))
        sample["values"]["apxinf_capacity_reserved_bytes"] = 1
        self.assertFalse(lifecycle.idle(sample))
        self.assertFalse(lifecycle.idle({"error": None, "values": {}}))


@unittest.skipUnless(os.environ.get("APXINF_SERVING_URL"), "Set APXINF_SERVING_URL to run live model tests.")
class LiveServingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = os.environ["APXINF_SERVING_URL"]
        bench.local_url(cls.url)
        status, models = bench.request_json(cls.url, "/v1/models")
        if status != 200 or not models.get("data"):
            raise AssertionError("The service did not expose a model.")
        cls.model = os.environ.get("APXINF_SERVING_MODEL") or models["data"][0]["id"]

    def test_health_readiness_and_metrics(self):
        for path in ("/healthz", "/readyz"):
            self.assertEqual(bench.request_json(self.url, path)[0], 200)
        self.assertIn("apxinf_", bench.request_text(self.url, "/metrics"))

    def test_count_tokens_matches_repeated_input(self):
        body = {"model": self.model, "messages": [{"role": "user", "content": "Count this original sentence."}]}
        first = bench.request_json(self.url, "/v1/messages/count_tokens", body)
        second = bench.request_json(self.url, "/v1/messages/count_tokens", body)
        self.assertEqual(first, second)
        self.assertEqual(first[0], 200)
        self.assertGreater(first[1]["input_tokens"], 0)

    def test_anthropic_and_openai_stream_completion(self):
        for api in ("anthropic", "openai"):
            with self.subTest(api=api):
                result = bench.stream_request(self.url, self.model, "Say hello briefly.", 16, api)
                self.assertTrue(result["success"], result["error"])
                self.assertIsNotNone(result["ttft_s"])
                self.assertGreater(result["output_tokens"], 0)

    def test_unsupported_sampling_rejects_before_generation(self):
        body = {"model": self.model, "max_tokens": 16, "temperature": .8,
                "messages": [{"role": "user", "content": "This request must fail validation."}]}
        status, error = bench.request_json(self.url, "/v1/messages", body)
        self.assertEqual(status, 400)
        self.assertIn("error", error)

    def test_invalid_model_and_unsupported_content_return_json_errors(self):
        original = {"model": self.model, "max_tokens": 16, "stream": True,
                    "messages": [{"role": "user", "content": "Original rejected request."}]}
        cases = [
            ({**original, "model": "absent-test-model"}, 404),
            ({**original, "tool_choice": {"type": "any"}}, 400),
            ({**original, "thinking": {"type": "adaptive"}}, 400),
            ({**original, "messages": [{"role": "user", "content": [{"type": "image", "source": {}}]}]}, 400),
        ]
        for body, expected in cases:
            with self.subTest(body=body):
                status, error = bench.request_json(self.url, "/v1/messages", body)
                self.assertEqual(status, expected)
                self.assertEqual(error["type"], "error")
                self.assertIsInstance(error["error"]["message"], str)
        self.assertEqual(bench.request_json(self.url, "/readyz")[0], 200)

    def test_malformed_and_duplicate_json_do_not_start_a_stream(self):
        for raw in ('{"model":', '{"model":"first","model":"second"}'):
            with self.subTest(raw=raw):
                connection = bench.connect(self.url, 10)
                try:
                    connection.request("POST", "/v1/messages", raw, {"content-type": "application/json"})
                    response = connection.getresponse()
                    self.assertEqual(response.status, 400)
                    self.assertIn("application/json", response.getheader("content-type"))
                    self.assertEqual(json.loads(response.read())["type"], "error")
                finally:
                    connection.close()


if __name__ == "__main__":
    unittest.main()
