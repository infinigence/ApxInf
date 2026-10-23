from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

if __package__:
    from . import contracts
    from . import clef_contracts as schema
    from .clef_identity import ClefError
else:
    import contracts
    import clef_contracts as schema
    from clef_identity import ClefError


class ClefWorker:
    def __init__(self, runtime: Any, worker_epoch: str, emit: Any):
        contracts._uuid(worker_epoch)
        self.runtime = runtime
        self.worker_epoch = worker_epoch
        self.emit = emit
        self.preparation_bytes = None
        self.prepared_record = None
        self.prepared_encoding = None
        self.terminal = False

    def frame(self, kind: str, **fields: Any) -> dict:
        return {"protocol": schema.PROTOCOL, "kind": kind, "worker_epoch": self.worker_epoch, **fields}

    def send(self, kind: str, **fields: Any) -> None:
        self.emit(contracts.encode_frame(self.frame(kind, **fields)))

    def ready(self) -> None:
        self.send("ready", model_revision=self.runtime.model_revision,
                  capability_revision=self.runtime.capability_revision, model_manifest=self.runtime.manifest,
                  capabilities=self.runtime.capabilities,
                  limits={"max_frame_bytes": contracts.MAX_FRAME_BYTES, "max_depth": 16, "max_commands": 10000},
                  model_path=self.runtime.model_path, runtime=schema.RUNTIME, memory=self.runtime.memory())

    def _attempt(self, command: dict) -> dict:
        return {key: command[key] for key in ("command_id", "request_id", "attempt")}

    def finish(self, command: dict, *, code: str | None = None, message: str = "", accepted: bool = False) -> int:
        if code is not None:
            self.send("failure", **self._attempt(command), event_seq=1 if accepted else 0, code=code, message=message)
        self.send("drained", command_id=command["command_id"], reason="attempt_complete")
        self.terminal = True
        return 0 if code in (None, "context_limit", "invalid_request") else 1

    def handle(self, command: dict) -> int | None:
        schema.validate_frame(command)
        if command["worker_epoch"] != self.worker_epoch or self.terminal:
            raise ClefError("protocol_fault", "The decision command belongs to an unavailable worker epoch.")
        if command["kind"] == "shutdown":
            self.send("drained", command_id=command["command_id"], reason="shutdown")
            self.terminal = True
            return 0
        if command["kind"] not in {"prepare_decision", "submit_decision"}:
            raise ClefError("protocol_fault", "The decision worker received an event as a command.")
        if (command["model_revision"] != self.runtime.model_revision
                or command["capability_revision"] != self.runtime.capability_revision):
            return self.finish(command, code="protocol_fault", message="The approved decision identity does not match.")
        accepted = False
        try:
            if command["kind"] == "prepare_decision":
                if self.preparation_bytes is not None:
                    raise ClefError("protocol_fault", "The decision worker already owns one preparation.")
                record = schema.validate_record(command["record"])
                encoding = self.runtime.encode(record)
                preparation = schema.make_preparation(command, record, encoding)
                self.preparation_bytes = json.dumps(preparation, ensure_ascii=False, sort_keys=True,
                                                     separators=(",", ":")).encode("utf-8")
                self.prepared_record = record
                self.prepared_encoding = encoding
                self.send("prepared_decision", **self._attempt(command), model_revision=self.runtime.model_revision,
                          capability_revision=self.runtime.capability_revision, preparation=preparation)
                return None
            submitted = json.dumps(command["preparation"], ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")).encode("utf-8")
            if self.preparation_bytes is None or submitted != self.preparation_bytes:
                raise ClefError("protocol_fault", "The submitted preparation differs from the owned preparation.")
            self.send("accepted", **self._attempt(command), event_seq=0, model_revision=self.runtime.model_revision,
                      capability_revision=self.runtime.capability_revision)
            accepted = True
            probabilities, metrics = self.runtime.decide(self.prepared_record, self.prepared_encoding)
            self.send("decision_result", **self._attempt(command), event_seq=1,
                      preparation_digest=command["preparation"]["preparation_digest"], probabilities=probabilities,
                      usage={"input_tokens": len(self.prepared_encoding.input_ids), "output_tokens": 0}, metrics=metrics)
            return self.finish(command)
        except ClefError as error:
            return self.finish(command, code=error.code, message=str(error), accepted=accepted)
        except contracts.ContractError:
            return self.finish(command, code="protocol_fault", message="The decision runtime returned invalid data.", accepted=accepted)
        except Exception:
            return self.finish(command, code="internal_error", message="The decision runtime failed.", accepted=accepted)

    def run(self, source: Any) -> int:
        self.ready()
        for _ in range(contracts.MAX_COMMANDS):
            raw = source.readline(contracts.MAX_FRAME_BYTES + 1)
            if not raw:
                return 1
            if len(raw) > contracts.MAX_FRAME_BYTES or not raw.endswith(b"\n"):
                return 1
            command = contracts.decode_frame(raw)
            result = self.handle(command)
            if result is not None:
                return result
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one pinned Clef text decision attempt.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--worker-epoch", required=True)
    parser.add_argument("--clef-loader", required=True)
    parser.add_argument("--protocol", default=schema.PROTOCOL)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--max-output-tokens", type=int, default=0)
    parser.add_argument("--prefill-step-size", type=int, default=256)
    parser.add_argument("--output-batch-tokens", type=int, default=1)
    parser.add_argument("--memory-limit-bytes", type=int, required=True)
    args = parser.parse_args(argv)
    if (args.protocol != schema.PROTOCOL or not 1 <= args.max_context <= 4096 or args.max_output_tokens != 0
            or args.prefill_step_size != 256 or args.output_batch_tokens != 1
            or not 1 <= args.memory_limit_bytes <= contracts.MAX_SAFE_INTEGER):
        parser.error("The Clef execution limits are unsupported.")
    try:
        contracts._uuid(args.worker_epoch)
    except contracts.ContractError:
        parser.error("The worker epoch is invalid.")
    output_fd = sys.stdout.fileno()

    def emit(payload: bytes) -> None:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(output_fd, remaining)
            if written <= 0:
                raise BrokenPipeError("The worker output closed.")
            remaining = remaining[written:]

    runtime = None
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        if __package__:
            from .clef_runtime import ClefRuntime
        else:
            from clef_runtime import ClefRuntime
        runtime = ClefRuntime(args.model, args.clef_loader, args.max_context, args.memory_limit_bytes)
        return ClefWorker(runtime, args.worker_epoch, emit).run(sys.stdin.buffer)
    except Exception as error:
        print(f"Clef worker stopped. {type(error).__name__}.", file=sys.stderr)
        return 1
    finally:
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
