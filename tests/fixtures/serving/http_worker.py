"""Emit original CPU protocol responses for real HTTP gateway checks.

This fixture does not load MLX or establish model quality.
The launch manifest checks real fixture files and the actual interpreter.
Execution labels match the current launch schema, not an inference claim.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time

import contracts


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def error(code):
    return {"code": code, "message": "Original HTTP fixture failure.",
            "scope": "request", "state_validity": "none"}


class Worker:
    def __init__(self, args):
        self.args = args
        self.root = Path(__file__).resolve().parents[4]
        self.commands = queue.Queue()
        self.trace_path = self.root / "http-worker-events.jsonl"

        artifacts = [{"path": path.name, "size_bytes": path.stat().st_size,
                      "sha256": digest(path)} for path in sorted(Path(args.model).iterdir())]
        manifest = {
            "fixture": "original-http-cpu-1", "artifacts": artifacts,
            "adapter_sha256": digest(__file__), "python_sha256": digest(sys.executable),
            "contracts_sha256": digest(Path(__file__).with_name("contracts.py")),
            "runtime_pins_sha256": digest(self.root / "scripts/apxinf_mlx_generate.py"),
            "execution": {"provider": "mlx-lm", "precision": "bundle",
                          "prefill_step_size": args.prefill_step_size,
                          "output_batch_tokens": args.output_batch_tokens,
                          "memory_limit_bytes": args.memory_limit_bytes},
        }
        capabilities = {
            "text": True, "greedy": True, "token_events": True,
            "image": False, "sampling": False, "exact_append": False,
            "cancel_granularity": "token", "max_active_sequences": 1,
            "max_context": args.max_context, "max_output_tokens": args.max_output_tokens,
            "state_schema": "none",
        }
        self.revision = contracts.canonical_identity_digest("apxinf-model-identity-v1", manifest)
        self.trace("started", pid=os.getpid())
        self.emit("ready", model_revision=self.revision, model_manifest=manifest,
                  capability_revision=contracts.canonical_identity_digest("apxinf-capability-v1", capabilities),
                  capabilities=capabilities, model_path=args.model, eos_token_ids=[2], vocab_size=128,
                  limits={"max_frame_bytes": contracts.MAX_FRAME_BYTES,
                          "max_depth": contracts.MAX_DEPTH, "max_commands": contracts.MAX_COMMANDS},
                  runtime={"python": "cpu-fixture", "mlx": "absent", "mlx_lm": "absent"},
                  memory={"active_bytes": 1024, "cache_bytes": 0, "peak_bytes": 2048})

    def trace(self, event, **fields):
        with self.trace_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps({"event": event, **fields}) + "\n")

    def emit(self, kind, **fields):
        frame = {"protocol": self.args.protocol, "worker_epoch": self.args.worker_epoch,
                 "kind": kind, **fields}
        sys.stdout.buffer.write(contracts.encode_frame(frame))
        sys.stdout.buffer.flush()

    def read_commands(self):
        try:
            for line in sys.stdin.buffer:
                self.commands.put(contracts.decode_frame(line))
        finally:
            self.commands.put(None)

    def prepare(self, command):
        self.trace("prepared", messages=command["messages"], tools=command["tools"])
        text = command["messages"][-1]["content"]
        names = ["text", "slow", "fail", "tool", "stop", "exit", "prepare-error", "prepare-fault", "stream-fault"]
        scenario = text.removeprefix("fixture:")
        index = names.index(scenario) if scenario in names else 0
        if scenario == "prepare-error":
            self.emit("command_result", command_id=command["command_id"], status="error",
                      error=error("invalid_request"))
        else:
            tokens = [20 + index, 4]
            self.emit("prepared_input", command_id=("00000000-0000-4000-8000-000000000001"
                      if scenario == "prepare-fault" else command["command_id"]),
                      model_revision=self.revision, token_ids=tokens, effective_input_tokens=2,
                      prompt_digest=contracts.token_prefix_digest(tokens))

    def generate(self, command):
        identity = {"request_id": command["request_id"], "attempt": command["attempt"]}
        sequence = 0
        count = 0
        started = time.monotonic_ns()

        def event(kind, **fields):
            nonlocal sequence
            self.emit(kind, **identity, event_seq=sequence, **fields)
            sequence += 1

        def control(seconds):
            try:
                incoming = self.commands.get(timeout=seconds)
            except queue.Empty:
                return None
            if incoming is None:
                raise RuntimeError("The supervisor closed its input during execution.")
            assert incoming["kind"] in {"cancel_request", "stop_generation"}, incoming
            assert incoming["request_id"] == command["request_id"]
            self.trace("control", kind=incoming["kind"], request_id=incoming["request_id"])
            self.emit("command_result", command_id=incoming["command_id"], status="accepted")
            return incoming

        event("accepted")
        self.trace("accepted", request_id=command["request_id"])
        event("prefill_progress", consumed_tokens=2)
        scenario = command["token_ids"][0] - 20
        if scenario == 5:
            os._exit(19)
        if scenario == 8:
            sequence += 1
        pieces = ["Original ", "reply.", ""]
        if scenario == 3:
            pieces = ['<tool_call>{"name":"read","arguments":{"path":"marker.txt"}}</tool_call>']
            pieces.append("")

        elif scenario == 4:
            pieces = ["visible ST", "OP hidden"]
        elif scenario == 1:
            pieces = ["waiting "] * command["max_tokens"]
        elif scenario == 2:
            pieces = ["partial "]
        status, cause = "completed", "length"
        for index, piece in enumerate(pieces[:command["max_tokens"]]):
            received = control(0.15 if scenario == 1 and index else 0.025)
            if received:
                if received["kind"] == "stop_generation":
                    cause = received["cause"]
                else:
                    cause = received["reason"]
                    status = "expired" if cause == "deadline_exceeded" else "cancelled"
                break
            token = 2 if piece == "" else 10
            event("tokens", output_index=count, token_ids=[token], text_delta=piece)
            count += 1
            if token == 2:
                cause = "eos"
        else:
            if scenario == 4:
                received = control(2)
                assert received and received["kind"] == "stop_generation"
                cause = received["cause"]
        if scenario == 2:
            status, cause = "failed", "internal_error"
        fields = {"error": error(cause)} if status == "failed" else {}
        event("terminal", status=status, cause=cause,
              usage={"input_tokens": 2, "output_tokens": count},
              state_result={"validity": "none", "consumed_position": 2 + count,
                            "history_count": 2 + count}, text_delta="",
              metrics={"elapsed_ns": time.monotonic_ns() - started, "ttft_ns": 0,
                       "peak_memory_bytes": 2048}, **fields)
        event("resources_released", released_lease_ids=command["capacity_lease_ids"], retained_lease_ids=[])
        self.trace("settled", request_id=command["request_id"], status=status)

    def run(self):
        threading.Thread(target=self.read_commands, daemon=True).start()
        while (command := self.commands.get()) is not None:
            if command["kind"] == "prepare_input":
                self.prepare(command)
            elif command["kind"] == "submit":
                self.generate(command)
            elif command["kind"] in {"cancel_request", "stop_generation"}:
                self.emit("command_result", command_id=command["command_id"], status="already_terminal")
            elif command["kind"] == "shutdown":
                self.emit("command_result", command_id=command["command_id"], status="accepted")
                self.emit("drained")
                self.trace("stopped")
                return
            else:
                raise RuntimeError(f"Unexpected fixture command: {command['kind']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "worker-epoch", "protocol"):
        parser.add_argument("--" + name, required=True)
    for name in ("max-context", "max-output-tokens", "prefill-step-size",
                 "output-batch-tokens", "memory-limit-bytes"):
        parser.add_argument("--" + name, type=int, required=True)
    Worker(parser.parse_args()).run()
