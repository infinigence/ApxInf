"""Check the real HTTP binary with an original CPU protocol fixture.

No model, MLX package, GPU, or external service is required.
Each test starts its own process group and retains diagnostics under devlocal.
"""

import argparse
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import uuid


ROOT = Path(__file__).resolve().parents[3]
BINARY = None
RUN_ROOT = None
MODEL = "original-cpu-fixture"


def body(scenario="text", **fields):
    return {"model": MODEL, "max_tokens": 8,
            "messages": [{"role": "user", "content": "fixture:" + scenario}], **fields}


def until(probe, description, seconds=4):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        value = probe()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError("Timed out: " + description)


def frames(response):
    event, data = "message", []
    while line := response.readline():
        line = line.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                text = "\n".join(data)
                yield event, text if text == "[DONE]" else json.loads(text)
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:
        raise AssertionError("The SSE response ended inside a frame.")


class Gateway:
    def __init__(self, name, timeout=5):
        self.root = Path(tempfile.mkdtemp(prefix=name + "-", dir=RUN_ROOT))
        adapter = self.root / "python/apxinf/apxinf/serving/text_worker.py"
        adapter.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / "tests/fixtures/serving/http_worker.py", adapter)
        shutil.copyfile(ROOT / "python/apxinf/apxinf/serving/contracts.py", adapter.with_name("contracts.py"))
        pins = self.root / "scripts/apxinf_mlx_generate.py"
        pins.parent.mkdir()
        shutil.copyfile(ROOT / "scripts/apxinf_mlx_generate.py", pins)
        model = self.root / "model"
        model.mkdir()
        (model / "fixture.txt").write_text("Original CPU HTTP fixture. No model weights.\n")
        self.connections = []
        self.forced_cleanup = False
        self.closed = False
        self.log = (self.root / "gateway.log").open("wb")
        self.process = subprocess.Popen([
            BINARY, "--model", str(model), "--model-id", MODEL,
            "--python", str(Path(sys.executable).resolve()), "--worker", str(adapter),
            "--port", "0", "--host-pressure-policy", "disabled",
            "--max-context", "128", "--max-output-tokens", "64",
            "--queue-capacity", "1", "--timeout-seconds", str(timeout),
            "--memory-budget-bytes", "8192", "--sequence-reservation-bytes", "1024",
        ], stdout=self.log, stderr=self.log, start_new_session=True, cwd=ROOT)
        try:
            self.port = until(self.listener, "gateway startup", seconds=15)
            until(lambda: self.request("/readyz")[0] == 200, "HTTP readiness")
        except BaseException:
            self.close()
            raise

    def listener(self):
        text = (self.root / "gateway.log").read_text()
        if self.process.poll() is not None:
            raise AssertionError("Gateway startup failed:\n" + text)
        match = re.search(r"listener bound at http://127\.0\.0\.1:(\d+)", text)
        return int(match[1]) if match else None

    def trace(self):
        path = self.root / "http-worker-events.jsonl"
        result = []
        for line in path.read_text().splitlines() if path.exists() else []:
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # A concurrent append can expose the final incomplete line.
        return result

    def send(self, path, document=None, raw=None, method=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=4)
        self.connections.append(connection)
        payload = raw if raw is not None else json.dumps(document) if document is not None else None
        connection.request(method or ("GET" if payload is None else "POST"), path, payload,
                           {"Content-Type": "application/json"})
        return connection

    def request(self, path, document=None, raw=None, method=None):
        connection = self.send(path, document, raw, method)
        try:
            response = connection.getresponse()
            content = response.read()
            headers = dict(response.getheaders())
            if "application/json" in response.getheader("content-type", ""):
                content = json.loads(content)
            else:
                content = content.decode()
            return response.status, headers, content
        finally:
            connection.close()

    def stream(self, path, document):
        connection = self.send(path, document)
        response = connection.getresponse()
        assert response.status == 200, response.read()
        assert response.getheader("content-type").startswith("text/event-stream")
        return connection, response

    def metrics(self):
        status, _, text = self.request("/metrics")
        assert status == 200
        return {line.rsplit(" ", 1)[0]: float(line.rsplit(" ", 1)[1])
                for line in text.splitlines() if line and not line.startswith("#")}

    def idle(self):
        names = ["apxinf_active_requests", "apxinf_queued_requests", "apxinf_capacity_reserved_bytes",
                 'apxinf_cleanup_pending{operation="generate"}',
                 'apxinf_cleanup_pending{operation="count_tokens"}']
        def settled():
            metrics = self.metrics()
            return all(metrics.get(name) == 0 for name in names)
        return until(settled, "resource settlement")

    def close(self):
        if self.closed:
            return
        self.closed = True
        for connection in self.connections:
            connection.close()
        try:
            if self.process.poll() is None:
                self.process.send_signal(signal.SIGINT)
                try:
                    self.process.wait(timeout=6)
                except subprocess.TimeoutExpired:
                    self.forced_cleanup = True
        finally:
            # The owned process group can survive an unexpected gateway exit.
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.forced_cleanup = True
            except ProcessLookupError:
                pass
            finally:
                self.process.wait(timeout=3)
                self.log.close()


class HttpGatewayTests(unittest.TestCase):
    def setUp(self):
        self.gateway = Gateway(self._testMethodName, timeout=1 if "deadline" in self._testMethodName else 5)
        self.addCleanup(self.gateway.close)

    def assert_error(self, response, status, code):
        actual, headers, content = response
        self.assertEqual(actual, status, content)
        self.assertIn("application/json", headers["content-type"])
        self.assertEqual(content["type"], "error")
        self.assertEqual(content["error"]["code"], code)

    def stream_all(self, path, document):
        connection, response = self.gateway.stream(path, document)
        try:
            return dict(response.getheaders()), list(frames(response))
        finally:
            response.close()
            connection.close()

    def test_routes_counting_and_nonstreamed_responses(self):
        gateway = self.gateway
        self.assertEqual(gateway.request("/healthz")[2], {"status": "alive"})
        status, _, ready = gateway.request("/readyz")
        self.assertEqual(status, 200)
        self.assertTrue(ready["ready"])
        self.assertEqual(ready["worker"]["runtime"]["mlx"], "absent")
        self.assertFalse(ready["worker"]["capabilities"]["exact_append"])
        self.assertEqual(ready["host_pressure"]["policy"], "disabled")
        models = gateway.request("/v1/models")[2]["data"]
        self.assertEqual([item["id"] for item in models], [MODEL])
        self.assertEqual(gateway.request("/v1/messages/count_tokens", body())[2], {"input_tokens": 2})
        self.assertFalse(any(item["event"] == "accepted" for item in gateway.trace()))
        for path in ("/v1/messages", "/v1/chat/completions"):
            status, headers, reply = gateway.request(path, body())
            self.assertEqual(status, 200, reply)
            uuid.UUID(headers["x-request-id"])
            self.assertEqual(headers["x-apxinf-model-revision"], ready["worker"]["model_revision"])
            self.assertEqual(reply["model"], MODEL)
            if path.endswith("messages"):
                self.assertEqual(reply["content"], [{"type": "text", "text": "Original reply."}])
                self.assertEqual(reply["stop_reason"], "end_turn")
                self.assertEqual(reply["usage"], {"input_tokens": 2, "output_tokens": 3})
            else:
                self.assertEqual(reply["choices"][0]["message"]["content"], "Original reply.")
                self.assertEqual(reply["choices"][0]["finish_reason"], "stop")
                self.assertEqual(reply["usage"], {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5})
        gateway.idle()
        self.assertEqual(gateway.metrics()['apxinf_requests_total{status="completed"}'], 3)

    def test_anthropic_stream_indices_usage_and_terminal(self):
        headers, events = self.stream_all("/v1/messages", body(stream=True))
        self.assertEqual([name for name, _ in events], ["message_start", "content_block_start",
                         "content_block_delta", "content_block_delta", "content_block_stop",
                         "message_delta", "message_stop"])
        for name, value in events:
            self.assertEqual(value["type"], name)
            if "index" in value:
                self.assertEqual(value["index"], 0)
        self.assertEqual(events[0][1]["message"]["id"], "msg_" + headers["x-request-id"])
        self.assertEqual(events[-2][1]["usage"], {"output_tokens": 3})
        self.assertEqual(events[-2][1]["delta"]["stop_reason"], "end_turn")
        self.gateway.idle()

    def test_openai_stream_usage_modes(self):
        for include in (None, False, True):
            with self.subTest(include_usage=include):
                request = body(stream=True)
                if include is not None:
                    request["stream_options"] = {"include_usage": include}
                headers, events = self.stream_all("/v1/chat/completions", request)
                self.assertEqual(events[-1], ("message", "[DONE]"))
                chunks = [value for _, value in events[:-1]]
                self.assertTrue(all(chunk["id"] == "chatcmpl-" + headers["x-request-id"] for chunk in chunks))
                text = "".join(choice["delta"].get("content", "") for chunk in chunks for choice in chunk["choices"])
                self.assertEqual(text, "Original reply.")
                if include:
                    self.assertEqual(chunks[-1]["choices"], [])
                    self.assertEqual(chunks[-1]["usage"], {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5})
                    self.assertTrue(all(chunk["usage"] is None for chunk in chunks[:-1]))
                    self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "stop")
                else:
                    self.assertTrue(all("usage" not in chunk for chunk in chunks))
                    self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    def test_invalid_input_never_reaches_worker(self):
        gateway = self.gateway
        for path in ("/v1/messages", "/v1/chat/completions", "/v1/messages/count_tokens"):
            for raw in ('{"model":', '{"model":"a","model":"b"}'):
                self.assert_error(gateway.request(path, raw=raw), 400, "invalid_request")
            self.assert_error(gateway.request(path, body(model="absent")), 404, "model_not_found")
            self.assert_error(gateway.request(path, body(temperature=0.5)), 400, "unsupported_feature")
        self.assert_error(gateway.request("/v1/messages", body(thinking={"type": "adaptive"})), 400, "unsupported_feature")
        self.assert_error(gateway.request("/v1/chat/completions", body(max_completion_tokens=4)), 400, "invalid_request")
        self.assertEqual(gateway.request("/v1/messages", raw=" " * 1_000_001)[0], 413)
        self.assertEqual(gateway.request("/not-a-route")[0], 404)
        self.assertEqual(gateway.request("/v1/messages", method="GET")[0], 405)
        self.assertEqual([entry["event"] for entry in gateway.trace()], ["started"])
        self.assertEqual(gateway.request("/readyz")[0], 200)

    def test_tools_and_stop_sequences_use_public_completion(self):
        schema = {"type": "object", "properties": {"path": {"type": "string"}},
                  "required": ["path"], "additionalProperties": False}
        request = body("tool", stream=True, tools=[{"name": "read", "input_schema": schema}])
        _, events = self.stream_all("/v1/messages", request)
        starts = [value for name, value in events if name == "content_block_start"]
        self.assertEqual(len(starts), 1)
        self.assertEqual(starts[0]["index"], 0)
        self.assertEqual(starts[0]["content_block"]["type"], "tool_use")
        self.assertEqual(starts[0]["content_block"]["name"], "read")
        arguments = "".join(value["delta"]["partial_json"] for name, value in events if name == "content_block_delta")
        self.assertEqual(json.loads(arguments), {"path": "marker.txt"})
        self.assertEqual(events[-2][1]["delta"]["stop_reason"], "tool_use")
        reply = self.gateway.request("/v1/messages", body("stop", stop_sequences=["STOP"]))[2]
        self.assertEqual(reply["content"], [{"type": "text", "text": "visible "}])
        self.assertEqual((reply["stop_reason"], reply["stop_sequence"]), ("stop_sequence", "STOP"))
        self.gateway.idle()

    def test_preparation_and_stream_failures_keep_worker_usable(self):
        self.assert_error(self.gateway.request("/v1/messages", body("prepare-error")), 400, "invalid_request")
        for path in ("/v1/messages", "/v1/chat/completions"):
            _, events = self.stream_all(path, body("fail", stream=True))
            self.assertEqual(events[-1][0], "error")
            self.assertEqual(events[-1][1]["error"]["code"], "internal_error")
            self.assertNotIn(("message", "[DONE]"), events)
            self.assertFalse(any(name == "message_stop" for name, _ in events))
        self.gateway.idle()
        self.assertEqual(self.gateway.request("/readyz")[0], 200)
        self.assertEqual(self.gateway.request("/v1/messages", body())[0], 200)

    def test_disconnects_reclaim_waiting_queue_and_active_reservation(self):
        gateway = self.gateway
        active, response = gateway.stream("/v1/messages", body("slow", stream=True, max_tokens=64))
        events = frames(response)
        until(lambda: next(events)[0] == "content_block_delta", "first active output")
        queued = gateway.send("/v1/messages", body())
        until(lambda: gateway.metrics()["apxinf_queued_requests"] == 1, "queued request")
        self.assert_error(gateway.request("/v1/messages", body()), 429, "queue_full")
        queued.close()
        until(lambda: gateway.metrics()["apxinf_queued_requests"] == 0, "queued disconnect")
        self.assertEqual(len([item for item in gateway.trace() if item["event"] == "prepared"]), 1)
        response.close()
        active.close()
        gateway.idle()
        self.assertEqual(gateway.metrics()['apxinf_requests_total{status="cancelled"}'], 2)
        controls = [item for item in gateway.trace() if item["event"] == "control"]
        self.assertEqual([item["kind"] for item in controls], ["cancel_request"])
        self.assertEqual(gateway.request("/v1/messages", body())[0], 200)

    def test_deadline_emits_error_and_releases_resources(self):
        _, events = self.stream_all("/v1/messages", body("slow", stream=True, max_tokens=64))
        self.assertEqual(events[-1][0], "error")
        self.assertEqual(events[-1][1]["error"]["code"], "deadline_exceeded")
        self.assertFalse(any(name == "message_stop" for name, _ in events))
        self.gateway.idle()
        self.assertEqual(self.gateway.metrics()['apxinf_requests_total{status="expired"}'], 1)
        self.assertEqual(self.gateway.request("/readyz")[0], 200)

    def test_worker_exit_fences_readiness_but_keeps_health(self):
        _, events = self.stream_all("/v1/messages", body("exit", stream=True))
        self.assertEqual(events[-1][0], "error")
        self.assertEqual(events[-1][1]["error"]["code"], "worker_lost")
        until(lambda: self.gateway.request("/readyz")[0] == 503, "worker failure readiness")
        self.gateway.idle()
        self.assertEqual(self.gateway.request("/healthz")[0], 200)
        self.assert_error(self.gateway.request("/v1/messages", body()), 503, "worker_lost")

    def test_count_tokens_protocol_fault_keeps_specific_error(self):
        self.assert_error(self.gateway.request("/v1/messages/count_tokens", body("prepare-fault")),
                          500, "protocol_fault")
        self.assertEqual(self.gateway.request("/readyz")[0], 503)
        self.gateway.idle()

    def test_stream_protocol_fault_has_one_error_without_success(self):
        for path in ("/v1/messages", "/v1/chat/completions"):
            if path.endswith("completions"):
                self.gateway.close()
                self.gateway = Gateway(self._testMethodName + "-openai")
                self.addCleanup(self.gateway.close)
            _, events = self.stream_all(path, body("stream-fault", stream=True))
            errors = [value for name, value in events if name == "error"]
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0]["error"]["code"], "protocol_fault")
            self.assertFalse(any(name == "message_stop" or value == "[DONE]" for name, value in events))
            self.assertEqual(self.gateway.request("/readyz")[0], 503)
            self.gateway.idle()

    def test_cli_shutdown_reaps_worker(self):
        gateway = self.gateway
        gateway.close()
        self.assertEqual(gateway.process.returncode, 0)
        self.assertFalse(gateway.forced_cleanup, "The CLI must reap its worker without fixture intervention.")
        worker_pid = next(item["pid"] for item in gateway.trace() if item["event"] == "started")
        with self.assertRaises(ProcessLookupError):
            os.kill(worker_pid, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    args = parser.parse_args()
    BINARY = str(Path(args.binary).resolve(strict=True))
    base = Path(os.environ.get("APXINF_HTTP_TEST_OUTPUT", str(ROOT / "devlocal/serving-http-integration"))).resolve()
    base.mkdir(parents=True, exist_ok=True)
    RUN_ROOT = Path(tempfile.mkdtemp(prefix="http-gateway-", dir=base))
    print("CPU HTTP fixture evidence: " + str(RUN_ROOT), file=sys.stderr)
    unittest.main(argv=[sys.argv[0]], verbosity=2)
