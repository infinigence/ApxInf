from __future__ import annotations

import argparse
import os
import sys
from typing import Any

if __package__:
    from . import contracts
    from .owned_media import MediaBatch, MediaError
    from .text_worker import Attempt, ByteOutbox, TextWorker, WorkerFailure
    from .vision_preparation import PreparationError, PreparationIdentity
else:
    import contracts
    from owned_media import MediaBatch, MediaError
    from text_worker import Attempt, ByteOutbox, TextWorker, WorkerFailure
    from vision_preparation import PreparationError, PreparationIdentity


def capabilities(max_context: int, max_output_tokens: int) -> dict[str, Any]:
    return {
        "text": True, "image": True, "greedy": True, "sampling": True, "token_events": True,
        "cancel_granularity": "yield_or_process_exit", "max_active_sequences": 1,
        "max_context": max_context, "max_output_tokens": max_output_tokens,
        "exact_append": False, "state_schema": "none",
        "media_limits": {"max_images": 4, "max_image_bytes": 131072, "max_total_bytes": 262144,
                         "max_dimension": 1024, "max_host_array_bytes": 67108864},
    }


class VisionWorker(TextWorker):
    def __init__(self, runtime: Any, worker_epoch: str, emit: Any, *, max_output_tokens: int = 2048):
        super().__init__(runtime, worker_epoch, emit, max_output_tokens=max_output_tokens,
                         protocol=contracts.VISION_PROTOCOL,
                         capabilities=capabilities(runtime.max_context, max_output_tokens))
        runtime.activate(PreparationIdentity(worker_epoch, self.model_revision, self.capability_revision))

    def _prepare(self, command: dict[str, Any]) -> None:
        if command["model_revision"] != self.model_revision:
            self._reply(command, "error", self.error("invalid_request", "The model revision does not match."))
            return
        try:
            fields = self.runtime.prepare(command)
            frame = self.frame("prepared_input", command_id=command["command_id"],
                               model_revision=self.model_revision, **fields)
            contracts.validate_frame(frame)
            self._send(frame)
        except (PreparationError, MediaError, WorkerFailure) as error:
            self._reply(command, "error", self.error(error.code, str(error)))
        except Exception:
            self._reply(command, "error", self.error("invalid_request", "The model processor rejected the input."))

    def _terminal_metrics(self, attempt: Attempt, mutated: bool) -> dict[str, int]:
        try:
            replay = attempt.command.get("replay", {})
            batch = MediaBatch.from_descriptors(replay.get("media", []))
            source_bytes = sum(len(image.data) for image in batch.images)
        except (MediaError, AttributeError, TypeError):
            source_bytes = 0
        return {"source_media_bytes": source_bytes,
                "peak_host_array_bytes": self.runtime.peak_host_array_bytes if mutated else 0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the pinned serial MiniCPM-V vision worker.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--worker-epoch", required=True)
    parser.add_argument("--protocol", default=contracts.VISION_PROTOCOL)
    parser.add_argument("--max-context", type=int, default=16384)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--prefill-step-size", type=int, default=256)
    parser.add_argument("--output-batch-tokens", type=int, default=1)
    parser.add_argument("--memory-limit-bytes", type=int, default=10 * 1024**3)
    arguments = parser.parse_args(argv)
    if (arguments.protocol != contracts.VISION_PROTOCOL or arguments.prefill_step_size != 256
            or arguments.output_batch_tokens != 1 or not 1 <= arguments.max_context <= 16384
            or not 0 <= arguments.max_output_tokens <= 8192
            or not 1 <= arguments.memory_limit_bytes <= contracts.MAX_SAFE_INTEGER):
        parser.error("The vision execution limits are unsupported.")
    try:
        contracts.validate_frame(dict(protocol=arguments.protocol, kind="drained", worker_epoch=arguments.worker_epoch))
    except contracts.ContractError:
        parser.error("The worker epoch is invalid.")
    output_fd = sys.stdout.fileno()

    def write_all(payload: bytes) -> None:
        view = memoryview(payload)
        while view:
            written = os.write(output_fd, view)
            view = view[written:]

    outbox = ByteOutbox(write_all)
    try:
        if __package__:
            from .vision_runtime import VisionRuntime
        else:
            from vision_runtime import VisionRuntime
        runtime = VisionRuntime(arguments.model, arguments.max_context,
                                arguments.max_output_tokens, arguments.memory_limit_bytes)
        worker = VisionWorker(runtime, arguments.worker_epoch, outbox.put,
                              max_output_tokens=arguments.max_output_tokens)
        worker.run(sys.stdin.buffer)
        outbox.close()
        return 0
    except Exception as error:
        print(f"Vision worker stopped: {type(error).__name__}.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
