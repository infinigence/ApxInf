"""Exercise the CPU lifecycle fixture through its real subprocess pipes."""

from __future__ import annotations

import json
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/apxinf/apxinf/serving"))
import contracts


class LifecycleWorkerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="apxinf-lifecycle-")
        self.model = Path(self.directory.name) / "model"
        self.model.mkdir()
        self.trace = Path(self.directory.name) / "trace.jsonl"
        self.epoch = str(uuid.uuid4())
        self.process = None
        self.reader = None

    def tearDown(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
            self.process.wait(timeout=5)
            self.reader.join(timeout=5)
            for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                stream.close()
        self.directory.cleanup()

    def start(self, **settings):
        settings["trace_path"] = str(self.trace)
        (self.model / "lifecycle.json").write_text(json.dumps(settings))
        self.process = subprocess.Popen(
            [sys.executable, "-u", str(ROOT / "tests/fixtures/serving/lifecycle_worker.py"),
             "--model", str(self.model), "--worker-epoch", self.epoch,
             "--protocol", contracts.PROTOCOL, "--max-context", "64",
             "--max-output-tokens", "16"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self.frames = queue.Queue()

        def receive():
            try:
                for line in self.process.stdout:
                    self.frames.put(contracts.decode_frame(line))
            except Exception as error:
                self.frames.put(error)
            finally:
                self.frames.put(None)

        self.reader = threading.Thread(target=receive, daemon=True)
        self.reader.start()
        self.ready = self.read()
        self.assertEqual(self.ready["kind"], "ready")
        self.assertEqual(self.ready["limits"]["max_commands"], 10000)

    def read(self):
        frame = self.frames.get(timeout=3)
        if frame is None or isinstance(frame, Exception):
            self.fail(f"The fixture ended or produced an invalid frame: {frame!r}")
        return frame

    def command(self, kind, **fields):
        return {"protocol": contracts.PROTOCOL, "worker_epoch": self.epoch,
                "kind": kind, "command_id": str(uuid.uuid4()), **fields}

    def send(self, command):
        self.process.stdin.write(contracts.encode_frame(command))
        self.process.stdin.flush()

    def prepare(self):
        return self.command("prepare_input", model_revision=self.ready["model_revision"],
                            messages=[{"role": "user", "content": "fixture input"}],
                            tools=[], template_options={"enable_thinking": False})

    def submit(self):
        return self.command("submit", request_id=str(uuid.uuid4()), attempt=1,
                            model_revision=self.ready["model_revision"],
                            capability_revision=self.ready["capability_revision"],
                            token_ids=[3, 4], max_tokens=8, remaining_timeout_ms=2000,
                            eos_token_ids=[2], capacity_lease_ids=[str(uuid.uuid4())])

    def trace_records(self):
        text = self.trace.read_text()
        lines = text.splitlines()
        if text and not text.endswith("\n"):
            lines.pop()
        return [json.loads(line) for line in lines if line]

    def wait_for_trace(self, event):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if any(record["event"] == event for record in self.trace_records()):
                return
            time.sleep(0.005)
        self.fail(f"The fixture did not record {event}.")

    def finish_attempt(self, submit, first=()):
        tracker = contracts.AttemptTracker(submit)
        seen = []
        for frame in first:
            if frame["kind"] in contracts.ATTEMPT_EVENTS:
                tracker.observe(frame)
                seen.append(frame)
        while not tracker.done:
            frame = self.read()
            if frame["kind"] in contracts.ATTEMPT_EVENTS:
                tracker.observe(frame)
                seen.append(frame)
        return seen

    def shutdown(self):
        command = self.command("shutdown")
        self.send(command)
        reply = self.read()
        self.assertEqual((reply["kind"], reply["command_id"], reply["status"]),
                         ("command_result", command["command_id"], "accepted"))
        self.assertEqual(self.read()["kind"], "drained")
        status = self.process.wait(timeout=3)
        self.assertFalse(self.process.stdin.closed)
        self.assertEqual(status, 0, self.process.stderr.read().decode())

    def test_slow_and_rejected_preparations_preserve_command_matching(self):
        self.start(prepare_delays_ms=[100], prepare_error_indices=[1])
        first = self.prepare()
        self.send(first)
        self.wait_for_trace("prepare_started")
        self.assertTrue(self.frames.empty())
        self.assertEqual(self.read()["command_id"], first["command_id"])
        rejected = self.prepare()
        self.send(rejected)
        result = self.read()
        self.assertEqual((result["command_id"], result["status"], result["error"]["code"]),
                         (rejected["command_id"], "error", "invalid_request"))
        next_prepare = self.prepare()
        self.send(next_prepare)
        result = self.read()
        self.assertEqual((result["kind"], result["command_id"], result["token_ids"]),
                         ("prepared_input", next_prepare["command_id"], [3, 4]))
        self.shutdown()

    def test_paused_generation_cancels_and_settles_through_real_worker(self):
        self.start(generation_pause_ms=1000)
        submit = self.submit()
        self.send(submit)
        self.wait_for_trace("generate_started")
        self.send(self.command("cancel_request", request_id=submit["request_id"],
                               attempt=1, reason="user_cancel"))
        frames = self.finish_attempt(submit)
        self.assertEqual(frames[-2]["status"], "cancelled")
        self.assertEqual(frames[-1]["released_lease_ids"], submit["capacity_lease_ids"])
        self.assertTrue(any(record["event"] == "settled" for record in self.trace_records()))
        self.shutdown()

    def test_token_pause_remains_cancellable(self):
        self.start(token_delay_ms=1000)
        submit = self.submit()
        self.send(submit)
        first = []
        while not first or first[-1]["kind"] != "prefill_progress" or first[-1]["consumed_tokens"] != 2:
            first.append(self.read())
        self.send(self.command("cancel_request", request_id=submit["request_id"],
                               attempt=1, reason="client_disconnect"))
        frames = self.finish_attempt(submit, first)
        self.assertEqual(frames[-2]["cause"], "client_disconnect")
        self.assertEqual(frames[-2]["usage"]["output_tokens"], 0)
        self.shutdown()

    def test_normal_generation_and_shutdown_keep_terminal_order(self):
        self.start()
        submit = self.submit()
        self.send(submit)
        frames = self.finish_attempt(submit)
        self.assertEqual(frames[-2]["cause"], "eos")
        self.assertEqual(frames[-2]["usage"]["output_tokens"], 3)
        self.assertEqual([frame["kind"] for frame in frames][-2:],
                         ["terminal", "resources_released"])
        self.shutdown()

    def test_shutdown_during_generation_settles_before_exit_with_open_input(self):
        self.start(generation_pause_ms=100)
        submit = self.submit()
        self.send(submit)
        self.wait_for_trace("generate_started")
        self.send(self.command("shutdown"))
        frames = self.finish_attempt(submit)
        self.assertEqual(frames[-2]["status"], "completed")
        self.assertEqual(frames[-2]["cause"], "eos")
        self.assertEqual(self.read()["kind"], "drained")
        status = self.process.wait(timeout=3)
        self.assertFalse(self.process.stdin.closed)
        self.assertEqual(status, 0, self.process.stderr.read().decode())


if __name__ == "__main__":
    unittest.main()
