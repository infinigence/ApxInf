"""Run the real serial protocol with an original CPU lifecycle test runtime.

The model directory can contain lifecycle.json with prepare_delays_ms,
prepare_error_indices, generation_pause_ms, token_delay_ms, and trace_path.
Preparation indices start at zero. All durations use milliseconds.
The trace path is absolute and outside the model directory.
This fixture does not establish model identity or inference correctness.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python/apxinf/apxinf/serving"))
import contracts
from text_worker import ByteOutbox, TextWorker, WorkerFailure


class LifecycleRuntime:
    def __init__(self, model: str, max_context: int, epoch: str):
        self.owner = threading.get_ident()
        self.model_path = str(Path(model).resolve(strict=True))
        self.max_context = max_context
        self.vocab_size = 128
        self.eos_token_ids = [2]
        self.runtime = {"python": "cpu-fixture", "mlx": "absent", "mlx_lm": "absent"}
        self.manifest = {"fixture": "apxinf-lifecycle-cpu-1", "runtime": self.runtime}
        self.consumed_position = 0
        self.epoch = epoch
        self.prepare_count = 0
        settings_path = Path(self.model_path) / "lifecycle.json"
        settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
        allowed = {"prepare_delays_ms", "prepare_error_indices", "generation_pause_ms",
                   "token_delay_ms", "trace_path"}
        if type(settings) is not dict or settings.keys() - allowed:
            raise ValueError("The fixture configuration has invalid fields.")
        self.prepare_delays_ms = settings.get("prepare_delays_ms", [])
        self.prepare_error_indices = settings.get("prepare_error_indices", [])
        self.generation_pause_ms = settings.get("generation_pause_ms", 0)
        self.token_delay_ms = settings.get("token_delay_ms", 0)
        if type(self.prepare_delays_ms) is not list:
            raise ValueError("Preparation delays must be an array.")
        for delay in [*self.prepare_delays_ms, self.generation_pause_ms, self.token_delay_ms]:
            if type(delay) is not int or not 0 <= delay <= 60_000:
                raise ValueError("Fixture delays must be integers from zero through 60000.")
        if type(self.prepare_error_indices) is not list or any(
            type(index) is not int or index < 0 for index in self.prepare_error_indices
        ):
            raise ValueError("Preparation error indices must be nonnegative integers.")
        model_path = Path(self.model_path)
        trace_name = settings.get("trace_path", str(model_path.with_name(model_path.name + "-trace.jsonl")))
        self.trace_path = Path(trace_name)
        if not self.trace_path.is_absolute() or self.trace_path.resolve().is_relative_to(model_path):
            raise ValueError("The trace path must be absolute and outside the model directory.")
        self.trace("started", pid=os.getpid())

    def trace(self, event: str, **fields) -> None:
        if threading.get_ident() != self.owner:
            raise RuntimeError("The runtime has a different execution owner.")
        record = {"event": event, "worker_epoch": self.epoch, **fields}
        with self.trace_path.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(record, separators=(",", ":")) + "\n")

    def prepare(self, command):
        self.trace("prepare_started", command_id=command["command_id"])
        index = self.prepare_count
        self.prepare_count += 1
        if index < len(self.prepare_delays_ms):
            time.sleep(self.prepare_delays_ms[index] / 1000)
        if index in self.prepare_error_indices:
            self.trace("prepare_rejected", command_id=command["command_id"])
            raise WorkerFailure("invalid_request", "The fixture rejected preparation.")
        self.trace("prepare_finished", command_id=command["command_id"])
        return [3, 4]

    @staticmethod
    def pause(milliseconds, progress, consumed):
        pause_until = time.monotonic() + milliseconds / 1000
        while True:
            progress(consumed)
            remaining = pause_until - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 0.005))

    def generate(self, command, progress):
        self.trace("generate_started", request_id=command["request_id"])
        self.consumed_position = 0
        self.pause(self.generation_pause_ms, progress, 0)
        prompt_length = len(command["token_ids"])
        self.consumed_position = prompt_length
        progress(self.consumed_position)
        for index in range(command["max_tokens"]):
            self.pause(self.token_delay_ms, progress, prompt_length)
            self.consumed_position += 1
            yield 2 if index == 2 else 10 + index % 2

    def decode(self, token, eos=False):
        self.trace("decoded", token=token)
        return "" if eos else "x"

    def finish_text(self):
        self.trace("text_finished")
        return ""

    def memory(self):
        self.trace("memory_read")
        return {"active_bytes": 1024, "peak_bytes": 2048, "cache_bytes": 0}

    def settle(self):
        self.trace("settled")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--worker-epoch", required=True)
    parser.add_argument("--protocol", default=contracts.PROTOCOL)
    parser.add_argument("--max-context", type=int, default=16384)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--prefill-step-size", type=int, default=256)
    parser.add_argument("--output-batch-tokens", type=int, default=1)
    parser.add_argument("--memory-limit-bytes", type=int, default=10 * 1024**3)
    args = parser.parse_args(argv)
    contracts.validate_frame({"protocol": args.protocol, "kind": "drained",
                              "worker_epoch": args.worker_epoch})
    if not 1 <= args.output_batch_tokens <= 256:
        parser.error("The output batch size is invalid.")
    runtime = LifecycleRuntime(args.model, args.max_context, args.worker_epoch)

    def write_all(payload):
        pending = memoryview(payload)
        while pending:
            count = os.write(sys.stdout.fileno(), pending)
            pending = pending[count:]

    outbox = ByteOutbox(write_all)

    def emit(frame, critical=False):
        contracts.validate_frame(frame)
        runtime.trace("emitted", kind=frame["kind"],
                      **{name: frame[name] for name in ("command_id", "request_id", "status", "cause")
                         if name in frame})
        outbox.put(frame, critical=critical)

    worker = TextWorker(runtime, args.worker_epoch, emit,
                        max_output_tokens=args.max_output_tokens,
                        output_batch_tokens=args.output_batch_tokens)
    try:
        worker.run(sys.stdin.buffer)
    finally:
        outbox.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
