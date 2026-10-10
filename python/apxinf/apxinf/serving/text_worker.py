"""Run one resident MLX text model through the worker/2.0 protocol.

The main thread owns all MLX objects. Reader and writer threads handle bytes only.
This file also supports direct execution without importing the policy package.
"""

from __future__ import annotations

import argparse
from collections import deque
import contextlib
from dataclasses import dataclass
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import queue
import stat
import sys
import threading
import time
from typing import Any, Callable

if __package__:
    from . import contracts
else:
    import contracts


ADAPTER_REVISION = "apxinf-serial-v0.1"
PINNED_RUNTIME = {"python": "3.14.3", "mlx": "0.32.1", "mlx_lm": "0.31.3"}
MAX_FRAME_BYTES = 1_048_576
MAX_COMMANDS = 10_000


class WorkerFailure(Exception):
    """Report a stable error without dependency details or input text."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class GenerationStopped(Exception):
    """Return control from generation at a safe boundary."""


class ChannelFailure(Exception):
    """Stop a worker whose protocol channel cannot remain valid."""


@dataclass
class InvalidCommand:
    """Keep a valid command envelope with invalid request content."""

    frame: dict[str, Any]


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def model_manifest(model_path: Path, runtime: dict[str, str], execution: dict[str, Any]) -> dict[str, Any]:
    """Bind local model artifacts, the interpreter, and this adapter by content."""
    artifacts = []
    entries = sorted(model_path.iterdir(), key=lambda path: path.name)
    if not entries or len(entries) > 4096:
        raise WorkerFailure("model_unavailable", "The model file count is invalid.")
    for path in entries:
        resolved = path.resolve(strict=True)
        before = resolved.stat()
        if not stat.S_ISREG(before.st_mode):
            raise WorkerFailure("model_unavailable", "Each model artifact must resolve to a regular file.")
        digest = _file_digest(resolved)
        after = resolved.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns
        ) or path.resolve(strict=True) != resolved:
            raise WorkerFailure("model_unavailable", "A model file changed during identity checks.")
        artifacts.append({"path": path.name, "size_bytes": before.st_size, "sha256": digest})
    return {
        "artifacts": artifacts,
        "runtime": runtime,
        "adapter_revision": ADAPTER_REVISION,
        "adapter_sha256": _file_digest(Path(__file__).resolve()),
        "python_sha256": _file_digest(Path(sys.executable).resolve()),
        "execution": execution,
    }


class MLXRuntime:
    """Use public MLX APIs on the thread that constructs this object."""

    def __init__(self, model_path: str, max_context: int, prefill_step_size: int,
                 output_batch_tokens: int = 1, memory_limit_bytes: int = 10 * 1024**3):
        self.owner = threading.get_ident()
        self.model_path = str(Path(model_path).resolve(strict=True))
        self.runtime = {
            "python": platform.python_version(),
            "mlx": importlib.metadata.version("mlx"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
        }
        if self.runtime != PINNED_RUNTIME:
            raise WorkerFailure("model_unavailable", "The runtime does not match the pinned versions.")
        helper_path = Path(__file__).resolve().parents[4] / "scripts/apxinf_mlx_generate.py"
        helper_spec = importlib.util.spec_from_file_location("apxinf_existing_runtime_pins", helper_path)
        if helper_spec is None or helper_spec.loader is None:
            raise WorkerFailure("model_unavailable", "The runtime pin definitions are unavailable.")
        helper = importlib.util.module_from_spec(helper_spec)
        helper_spec.loader.exec_module(helper)
        packages = helper._pinned_toolchain_versions()
        config_path = Path(self.model_path) / "config.json"
        if config_path.stat().st_size > 2 * 1024 * 1024:
            raise WorkerFailure("model_unavailable", "The model configuration exceeds the byte limit.")
        source_config = helper._parse_json(config_path.read_bytes(), "config.json")
        if source_config.get("model_file") is not None or source_config.get("auto_map") is not None:
            raise WorkerFailure("model_unavailable", "The model configuration requests custom code.")
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
                     "HF_HUB_DISABLE_TELEMETRY", "HF_HUB_DISABLE_IMPLICIT_TOKEN"):
            os.environ[name] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.manifest = model_manifest(Path(self.model_path), self.runtime, {
            "provider": "mlx-lm", "precision": "bundle",
            "prefill_step_size": prefill_step_size, "output_batch_tokens": output_batch_tokens,
            "memory_limit_bytes": memory_limit_bytes,
        })
        self.manifest["packages"] = packages
        self.manifest["runtime_pins_sha256"] = _file_digest(helper_path)
        self.manifest["contracts_sha256"] = _file_digest(Path(contracts.__file__).resolve())
        with contextlib.redirect_stdout(sys.stderr):
            import mlx.core as mx
            from mlx_lm import load
            from mlx_lm.generate import generate_step
            from mlx_lm.models.cache import make_prompt_cache

            self.mx = mx
            self._load_stream = mx.default_stream(mx.default_device())
            self._request_stream = None
            self._generation_stream = None
            self._peak_epoch = 0
            mx.set_memory_limit(memory_limit_bytes)
            self.generate_step = generate_step
            self.make_prompt_cache = make_prompt_cache
            self.model, self.tokenizer, config = load(
                self.model_path,
                tokenizer_config={"local_files_only": True, "trust_remote_code": False},
                lazy=False,
                return_config=True,
            )
            mx.synchronize(self._load_stream)
        text_config = config.get("text_config", config)
        model_limit = text_config.get("max_position_embeddings", max_context)
        self.max_context = min(max_context, int(model_limit))
        self.vocab_size = int(text_config["vocab_size"])
        self.eos_token_ids = sorted(int(item) for item in self.tokenizer.eos_token_ids)
        self.prefill_step_size = prefill_step_size
        self.consumed_position = 0
        self._cache = None
        self._detokenizer = None

    def _check_owner(self) -> None:
        if threading.get_ident() != self.owner:
            raise RuntimeError("Only the execution owner can use MLX objects.")

    def prepare(self, command: dict[str, Any]) -> list[int]:
        self._check_owner()
        with contextlib.redirect_stdout(sys.stderr):
            tokens = self.tokenizer.apply_chat_template(
                command["messages"],
                tools=command["tools"] or None,
                tokenize=True,
                add_generation_prompt=True,
                **command["template_options"],
            )
        return list(tokens)

    def generate(self, command: dict[str, Any], progress: Callable[[int], None]):
        self._check_owner()
        self._request_stream = self.mx.default_stream(self.mx.default_device())
        self.mx.reset_peak_memory()
        self._peak_epoch += 1
        self._generation_stream = None
        self.consumed_position = 0
        self._cache = self.make_prompt_cache(self.model)
        self._detokenizer = self.tokenizer.detokenizer
        self._detokenizer.reset()
        prompt_length = len(command["token_ids"])

        def prompt_progress(consumed: int, total: int) -> None:
            if self._generation_stream is None:
                if consumed != 0:
                    raise RuntimeError("The generation stream was not captured before prefill.")
                self._generation_stream = self.mx.default_stream(self.mx.default_device())
            self.consumed_position = max(self.consumed_position, consumed)
            progress(consumed)

        with contextlib.redirect_stdout(sys.stderr):
            generator = self.generate_step(
                self.mx.array(command["token_ids"]),
                self.model,
                max_tokens=command["max_tokens"],
                sampler=lambda logits: self.mx.argmax(logits, axis=-1),
                prompt_cache=self._cache,
                prefill_step_size=self.prefill_step_size,
                prompt_progress_callback=prompt_progress,
            )
            try:
                for index, (token, _) in enumerate(generator):
                    if self._generation_stream is None:
                        raise RuntimeError("The generation stream was not captured before output.")
                    # The pinned generator consumes the emitted token before yield.
                    self.consumed_position = prompt_length + index + 1
                    yield int(token)
            finally:
                generator.close()

    def decode(self, token: int, eos: bool = False) -> str:
        self._check_owner()
        if eos:
            return ""
        self._detokenizer.add_token(token)
        return self._detokenizer.last_segment

    def finish_text(self) -> str:
        self._check_owner()
        if self._detokenizer is None:
            return ""
        self._detokenizer.finalize()
        return self._detokenizer.last_segment

    def memory(self) -> dict[str, int]:
        self._check_owner()
        return {
            "active_bytes": int(self.mx.get_active_memory()),
            "peak_bytes": int(self.mx.get_peak_memory()),
            "cache_bytes": int(self.mx.get_cache_memory()),
        }

    def _synchronize_owned_streams(self) -> None:
        streams = []
        for stream in (self._generation_stream, self._request_stream, self._load_stream):
            if stream is not None and stream not in streams:
                self.mx.synchronize(stream)
                streams.append(stream)

    def inspect_memory(self) -> dict[str, Any]:
        """Observe logical cache metadata after submitted work completes."""
        self._check_owner()
        self._synchronize_owned_streams()

        def optional_integer(layer: Any, name: str) -> int | None:
            try:
                value = getattr(layer, name)
            except (AttributeError, NotImplementedError):
                return None
            # Do not materialize array-valued metadata or invent unsupported values.
            return value if type(value) is int and value >= 0 else None

        layers = []
        for layer in self._cache or ():
            layers.append({"type": type(layer).__name__,
                           "offset": optional_integer(layer, "offset"),
                           "nbytes": optional_integer(layer, "nbytes")})
        sizes = [layer["nbytes"] for layer in layers]
        return {
            "allocator": self.memory(),
            "peak_epoch": self._peak_epoch,
            "runtime_consumed_position": self.consumed_position,
            "cache_payload_bytes": sum(sizes) if all(size is not None for size in sizes) else None,
            "layers": layers,
        }

    def settle(self) -> None:
        self._check_owner()
        self._synchronize_owned_streams()
        self._cache = None
        self._detokenizer = None
        self.mx.clear_cache()
        self._synchronize_owned_streams()
        self._generation_stream = None
        self._request_stream = None


class ByteOutbox:
    """Bound output bytes and reserve room for control and terminal events."""

    def __init__(self, sink: Callable[[bytes], None], limit: int = 4 * MAX_FRAME_BYTES):
        self.sink = sink
        self.limit = limit
        self.pending_bytes = 0
        self.items: deque[bytes] = deque()
        self.condition = threading.Condition()
        self.failure: BaseException | None = None
        self.closed = False
        self.thread = threading.Thread(target=self._write, name="apxinf-output", daemon=True)
        self.thread.start()

    def put(self, frame: dict[str, Any], critical: bool = False) -> None:
        payload = contracts.encode_frame(frame)
        with self.condition:
            if self.failure or self.closed:
                raise ChannelFailure("The output channel is closed.")
            ceiling = self.limit if critical else self.limit - MAX_FRAME_BYTES
            if self.pending_bytes + len(payload) > ceiling:
                if critical:
                    raise ChannelFailure("The reserved output capacity is exhausted.")
                raise WorkerFailure("slow_consumer", "The output consumer cannot keep pace.")
            self.items.append(payload)
            self.pending_bytes += len(payload)
            self.condition.notify()

    def _write(self) -> None:
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.items or self.closed)
                if not self.items:
                    return
                payload = self.items.popleft()
            try:
                self.sink(payload)
            except BaseException as error:
                with self.condition:
                    self.failure = error
                    self.condition.notify_all()
                return
            with self.condition:
                self.pending_bytes -= len(payload)
                self.condition.notify_all()

    def close(self, timeout: float = 2.0) -> None:
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.thread.join(timeout)
        if self.thread.is_alive() or self.failure:
            raise ChannelFailure("The output channel did not finish.")


@dataclass
class Attempt:
    command: dict[str, Any]
    started_ns: int
    event_seq: int = 0
    output_count: int = 0
    first_token_ns: int = 0
    stop_status: str | None = None
    stop_cause: str | None = None
    terminal: bool = False
    payload_digest: str = ""

    def __post_init__(self) -> None:
        content = {key: value for key, value in self.command.items() if key != "command_id"}
        self.payload_digest = hashlib.sha256(
            json.dumps(content, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def key(self) -> tuple[str, int]:
        return (self.command["request_id"], self.command["attempt"])


class TextWorker:
    """Serialize model work and terminal decisions on one execution owner."""

    def __init__(
        self,
        runtime: Any,
        worker_epoch: str,
        emit: Callable[..., None],
        *,
        max_output_tokens: int = 2048,
        output_batch_tokens: int = 1,
    ):
        self.runtime = runtime
        self.worker_epoch = worker_epoch
        self.emit = emit
        self.max_output_tokens = max_output_tokens
        self.output_batch_tokens = output_batch_tokens
        self.commands: queue.Queue = queue.Queue(maxsize=8)
        self.controls: queue.Queue = queue.Queue(maxsize=32)
        self.input_done = threading.Event()
        self.input_fault: BaseException | None = None
        self.active: Attempt | None = None
        self.attempts: dict[tuple[str, int], Attempt] = {}
        self.command_records: dict[str, tuple[bytes, dict[str, Any] | None]] = {}
        self.completed_requests = 0
        self.cancelled_requests = 0
        self.draining = False
        self.shutdown_requested = False
        self.drained_sent = False
        self.capabilities = {
            "text": True, "image": False, "greedy": True, "sampling": False,
            "token_events": True, "cancel_granularity": "token_or_prefill_chunk",
            "max_active_sequences": 1, "max_context": runtime.max_context,
            "max_output_tokens": max_output_tokens, "exact_append": False,
            "state_schema": "none",
        }
        self.model_revision = contracts.canonical_identity_digest(
            "apxinf-model-identity-v1", runtime.manifest
        )
        self.capability_revision = contracts.canonical_identity_digest(
            "apxinf-capability-v1", self.capabilities
        )

    def frame(self, kind: str, **fields: Any) -> dict[str, Any]:
        return {"protocol": "apxinf-worker/2.0", "kind": kind,
                "worker_epoch": self.worker_epoch, **fields}

    @staticmethod
    def error(code: str, message: str, scope: str = "request") -> dict[str, str]:
        return {"code": code, "message": message, "scope": scope, "state_validity": "none"}

    def ready(self) -> dict[str, Any]:
        return self.frame(
            "ready", model_revision=self.model_revision,
            capability_revision=self.capability_revision,
            model_manifest=self.runtime.manifest, capabilities=self.capabilities,
            limits={"max_frame_bytes": MAX_FRAME_BYTES, "max_depth": 16,
                    "max_commands": MAX_COMMANDS},
            model_path=self.runtime.model_path, eos_token_ids=self.runtime.eos_token_ids,
            vocab_size=self.runtime.vocab_size, runtime=self.runtime.runtime,
            memory=self.runtime.memory(),
        )

    def read_commands(self, source: Any) -> None:
        """Read bytes independently. Do not access the runtime from this thread."""
        try:
            while line := source.readline(MAX_FRAME_BYTES + 1):
                if len(line) > MAX_FRAME_BYTES or not line.endswith(b"\n"):
                    raise ChannelFailure("The input frame is too large or incomplete.")
                invalid = False
                try:
                    command = contracts.decode_frame(line)
                except contracts.ContractError as error:
                    if error.frame is None:
                        raise
                    contracts.validate_command_envelope(error.frame)
                    command = error.frame
                    invalid = True
                if command["worker_epoch"] != self.worker_epoch:
                    raise ChannelFailure("The command has a different worker epoch.")
                if command["kind"] not in {
                    "submit", "prepare_input", "cancel_request", "stop_generation",
                    "query_stats", "drain", "shutdown",
                }:
                    raise ChannelFailure("The worker received an event instead of a command.")
                selected = self.commands if command["kind"] in {"submit", "prepare_input"} else self.controls
                try:
                    selected.put_nowait(InvalidCommand(command) if invalid else command)
                except queue.Full as error:
                    raise ChannelFailure("The command mailbox is full.") from error
                if command["kind"] == "shutdown" and not invalid:
                    # A shutdown ends input. Do not hold stdin's buffered lock at exit.
                    break
        except BaseException as error:
            self.input_fault = error
        finally:
            self.input_done.set()

    def _send(self, frame: dict[str, Any], critical: bool = True) -> None:
        self.emit(frame, critical=critical)

    def _event(self, attempt: Attempt, kind: str, **fields: Any) -> None:
        frame = self.frame(kind, request_id=attempt.key[0], attempt=attempt.key[1],
                           event_seq=attempt.event_seq, **fields)
        self._send(frame, critical=kind not in {"tokens", "prefill_progress"})
        attempt.event_seq += 1

    def _reply(self, command: dict[str, Any], status: str, error: dict | None = None) -> None:
        fields = {"command_id": command["command_id"], "status": status}
        if error is not None:
            fields["error"] = error
        frame = self.frame("command_result", **fields)
        self._send(frame)
        record = self.command_records.get(command["command_id"])
        if record is not None:
            self.command_records[command["command_id"]] = (record[0], frame)

    def _register_command(self, command: dict[str, Any]) -> bool:
        identity = hashlib.sha256(json.dumps(
            command, sort_keys=True, ensure_ascii=True, separators=(",", ":")
        ).encode()).digest()
        previous = self.command_records.get(command["command_id"])
        if previous is not None:
            if previous[0] != identity:
                self._send(self.frame("command_result", command_id=command["command_id"],
                                     status="error", error=self.error(
                                         "invalid_request", "The command ID has conflicting content.")))
            elif command["kind"] in {"cancel_request", "stop_generation"} and (
                registered := self.attempts.get((command["request_id"], command["attempt"]))
            ) is not None and registered.terminal:
                self._reply(command, "already_terminal")
            elif previous[1] is not None:
                self._send(previous[1])
            else:
                self._reply(command, "already_registered")
            return False
        if len(self.command_records) >= MAX_COMMANDS:
            raise ChannelFailure("The command history limit requires a new worker epoch.")
        self.command_records[command["command_id"]] = (identity, None)
        return True

    def _control(self, command: dict[str, Any]) -> None:
        if not self._register_command(command):
            return
        kind = command["kind"]
        if kind in {"drain", "shutdown"}:
            self.draining = True
            self.shutdown_requested |= kind == "shutdown"
            self._reply(command, "accepted")
            return
        if kind == "query_stats":
            frame = self.frame("stats", command_id=command["command_id"],
                               active_requests=int(self.active is not None),
                               completed_requests=self.completed_requests,
                               cancelled_requests=self.cancelled_requests)
            self._send(frame)
            old = self.command_records[command["command_id"]]
            self.command_records[command["command_id"]] = (old[0], frame)
            return
        attempt = self.attempts.get((command["request_id"], command["attempt"]))
        if attempt is None:
            self._reply(command, "error", self.error("invalid_request", "The attempt is not registered."))
        elif attempt.terminal:
            self._reply(command, "already_terminal")
        elif kind == "stop_generation" and command["output_token_count"] > attempt.output_count:
            self._reply(command, "error", self.error("invalid_request", "The output boundary is not available."))
        else:
            if attempt.stop_status is None:
                if kind == "stop_generation":
                    attempt.stop_status, attempt.stop_cause = "completed", command["cause"]
                elif command["reason"] == "deadline_exceeded":
                    attempt.stop_status, attempt.stop_cause = "expired", "deadline_exceeded"
                else:
                    attempt.stop_status, attempt.stop_cause = "cancelled", command["reason"]
            self._reply(command, "accepted")

    def _check_controls(self, attempt: Attempt | None = None) -> None:
        if self.input_fault is not None:
            raise ChannelFailure("The input channel failed validation.")
        while True:
            try:
                command = self.controls.get_nowait()
            except queue.Empty:
                break
            if isinstance(command, InvalidCommand):
                self._invalid(command.frame)
            else:
                self._control(command)
        if attempt is None:
            return
        elapsed = time.monotonic_ns() - attempt.started_ns
        if attempt.stop_status is None and elapsed >= attempt.command["remaining_timeout_ms"] * 1_000_000:
            attempt.stop_status, attempt.stop_cause = "expired", "deadline_exceeded"
        if attempt.stop_status is not None:
            raise GenerationStopped()

    def _prepare(self, command: dict[str, Any]) -> None:
        if command["model_revision"] != self.model_revision:
            self._reply(command, "error", self.error("invalid_request", "The model revision does not match."))
            return
        try:
            tokens = self.runtime.prepare(command)
            if not tokens or len(tokens) > self.runtime.max_context:
                raise WorkerFailure("context_limit", "The prepared input exceeds the context limit.")
            if any(type(token) is not int or not 0 <= token < self.runtime.vocab_size for token in tokens):
                raise WorkerFailure("internal_error", "The tokenizer returned an invalid token.")
            frame = self.frame(
                "prepared_input", command_id=command["command_id"],
                model_revision=self.model_revision, token_ids=tokens,
                effective_input_tokens=len(tokens), prompt_digest=contracts.token_prefix_digest(tokens),
            )
            self._send(frame)
            # Retain the digest, not a second copy of a potentially large prompt.
        except WorkerFailure as error:
            self._reply(command, "error", self.error(error.code, str(error)))
        except Exception:
            self._reply(command, "error", self.error("invalid_request", "The model template rejected the input."))

    def _submit_error(self, command: dict[str, Any]) -> dict | None:
        if self.draining:
            return self.error("model_unavailable", "The worker is draining.")
        if command["model_revision"] != self.model_revision or command["capability_revision"] != self.capability_revision:
            return self.error("invalid_request", "The request revisions do not match the worker.")
        if command["max_tokens"] > self.max_output_tokens:
            return self.error("unsupported_feature", "The requested output exceeds the deployment limit.")
        if len(command["token_ids"]) + command["max_tokens"] > self.runtime.max_context:
            return self.error("context_limit", "The input and output allowance exceed the context limit.")
        if any(token >= self.runtime.vocab_size for token in command["token_ids"] + command["eos_token_ids"]):
            return self.error("invalid_request", "A token exceeds the model vocabulary.")
        return None

    def _submit(self, command: dict[str, Any]) -> None:
        key = (command["request_id"], command["attempt"])
        previous = self.attempts.get(key)
        if previous is not None:
            if previous.payload_digest == Attempt(command, 0).payload_digest:
                self._reply(command, "already_registered")
            else:
                self._reply(command, "error", self.error("invalid_request", "The attempt has conflicting content."))
            return
        attempt = Attempt(command, time.monotonic_ns())
        self.attempts[key] = attempt
        error = self._submit_error(command)
        if error:
            self._event(attempt, "rejected", error=error)
            self._finish(attempt, "failed", error["code"], error, mutated=False)
            return
        self.active = attempt
        self._event(attempt, "accepted")
        status, cause = "completed", "length"
        error = None
        pending_tokens: list[int] = []
        pending_text: list[str] = []
        generator = None
        mutated = False
        decoder_valid = True
        output_rejected = False

        def flush() -> None:
            nonlocal output_rejected
            if pending_tokens:
                try:
                    self._event(attempt, "tokens", output_index=attempt.output_count,
                                token_ids=list(pending_tokens), text_delta="".join(pending_text))
                except Exception:
                    output_rejected = True
                    raise
                attempt.output_count += len(pending_tokens)
                pending_tokens.clear()
                pending_text.clear()

        last_progress = -1

        def progress(consumed: int) -> None:
            nonlocal last_progress
            self._check_controls(attempt)
            if consumed > last_progress:
                self._event(attempt, "prefill_progress", consumed_tokens=consumed)
                last_progress = consumed

        try:
            self._check_controls(attempt)
            if command["max_tokens"]:
                generator = self.runtime.generate(command, progress)
                mutated = True
                for token in generator:
                    if not 0 <= token < self.runtime.vocab_size:
                        raise WorkerFailure("internal_error", "The model returned an invalid token.")
                    if not attempt.first_token_ns:
                        attempt.first_token_ns = time.monotonic_ns()
                    eos = token in command["eos_token_ids"]
                    try:
                        text = self.runtime.decode(token, eos=eos)
                    except Exception:
                        decoder_valid = False
                        raise
                    pending_tokens.append(token)
                    pending_text.append(text)
                    if len(pending_tokens) >= self.output_batch_tokens or attempt.output_count == 0 or eos:
                        flush()
                    self._check_controls(attempt)
                    if eos:
                        cause = "eos"
                        break
                flush()
                if cause == "length" and attempt.output_count != command["max_tokens"]:
                    raise WorkerFailure("internal_error", "Generation ended before the output allowance.")
        except GenerationStopped:
            status, cause = attempt.stop_status, attempt.stop_cause
            try:
                flush()
            except WorkerFailure:
                status, cause = "cancelled", "slow_consumer"
        except WorkerFailure as failure:
            if failure.code == "slow_consumer":
                status, cause = "cancelled", "slow_consumer"
            else:
                status, cause = "failed", failure.code
                error = self.error(failure.code, str(failure))
        except ChannelFailure:
            raise
        except Exception:
            status, cause = "failed", "internal_error"
            error = self.error("internal_error", "Model execution failed.")
        finally:
            if generator is not None:
                generator.close()
        if status == "failed" and not output_rejected:
            try:
                flush()
            except ChannelFailure:
                raise
            except Exception:
                # An output rejection cannot replace the observed execution failure.
                pass
        self._finish(attempt, status, cause, error, mutated=mutated,
                     finalize_text=decoder_valid and not output_rejected)
        self.active = None

    def _finish(self, attempt: Attempt, status: str, cause: str,
                error: dict | None, *, mutated: bool, finalize_text: bool = True) -> None:
        tail = ""
        consumed = 0
        peak = 0
        if mutated:
            try:
                tail = self.runtime.finish_text() if finalize_text and cause != "slow_consumer" else ""
                consumed = self.runtime.consumed_position
                peak = self.runtime.memory()["peak_bytes"]
            except Exception:
                status, cause = "failed", "internal_error"
                error = self.error("internal_error", "The output finalization failed.")
        history = len(attempt.command["token_ids"]) + attempt.output_count
        # An aborted internal lookahead cannot establish reusable state.
        if consumed > history:
            consumed = 0
        now = time.monotonic_ns()
        fields = {
            "status": status, "cause": cause,
            "usage": {"input_tokens": len(attempt.command["token_ids"]), "output_tokens": attempt.output_count},
            "state_result": {"validity": "none", "consumed_position": consumed, "history_count": history},
            "text_delta": tail,
            "metrics": {"elapsed_ns": now - attempt.started_ns,
                        "ttft_ns": (attempt.first_token_ns - attempt.started_ns) if attempt.output_count else 0,
                        "peak_memory_bytes": peak},
        }
        if error is not None:
            fields["error"] = error
        self._event(attempt, "terminal", **fields)
        attempt.terminal = True
        if status == "completed":
            self.completed_requests += 1
        if status == "cancelled":
            self.cancelled_requests += 1
        if mutated:
            try:
                self.runtime.settle()
            except Exception as failure:
                # Do not return leases when device settlement did not complete.
                raise ChannelFailure("Device resource settlement failed.") from failure
        self._event(attempt, "resources_released",
                    released_lease_ids=attempt.command["capacity_lease_ids"], retained_lease_ids=[])
        # Keep only duplicate-detection and terminal metadata within the epoch.
        attempt.command = {"request_id": attempt.key[0], "attempt": attempt.key[1]}

    def _invalid(self, command: dict[str, Any]) -> None:
        if not self._register_command(command):
            return
        error = self.error("invalid_request", "The command content does not match the profile.")
        if command["kind"] != "submit":
            self._reply(command, "error", error)
            return
        key = (command["request_id"], command["attempt"])
        if key in self.attempts:
            self._reply(command, "error", error)
            return
        safe_command = dict(command)
        tokens = command.get("token_ids")
        safe_command["token_ids"] = tokens if type(tokens) is list else []
        attempt = Attempt(safe_command, time.monotonic_ns())
        self.attempts[key] = attempt
        self._event(attempt, "rejected", error=error)
        self._finish(attempt, "failed", "invalid_request", error, mutated=False)

    def handle(self, command: dict[str, Any] | InvalidCommand) -> None:
        if isinstance(command, InvalidCommand):
            self._invalid(command.frame)
            return
        contracts.validate_frame(command)
        if command["worker_epoch"] != self.worker_epoch:
            raise ChannelFailure("The command has a different worker epoch.")
        if command["kind"] not in {"submit", "prepare_input"}:
            self._control(command)
            return
        if not self._register_command(command):
            return
        if command["kind"] == "prepare_input":
            self._prepare(command)
        else:
            self._submit(command)

    def run(self, source: Any) -> None:
        self._send(self.ready())
        reader = threading.Thread(target=self.read_commands, args=(source,), name="apxinf-input", daemon=True)
        reader.start()
        try:
            while True:
                # Register a queued submit before applying its later control.
                try:
                    command = self.commands.get(timeout=0.01)
                except queue.Empty:
                    command = None
                if command is not None:
                    self.handle(command)
                self._check_controls()
                if self.draining and not self.drained_sent and self.commands.empty():
                    self._send(self.frame("drained"))
                    self.drained_sent = True
                if (self.shutdown_requested or self.input_done.is_set()) and self.commands.empty():
                    break
        except ChannelFailure:
            # Process exit fences attempts whose resources remain unsettled.
            with contextlib.suppress(Exception):
                self._send(self.frame("worker_fault", error=self.error(
                    "protocol_fault", "The worker channel cannot continue.", "worker")))
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--worker-epoch", required=True)
    parser.add_argument("--protocol", default="apxinf-worker/2.0")
    parser.add_argument("--max-context", type=int, default=16384)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--prefill-step-size", type=int, default=256)
    parser.add_argument("--output-batch-tokens", type=int, default=1)
    parser.add_argument("--memory-limit-bytes", type=int, default=10 * 1024**3)
    arguments = parser.parse_args(argv)
    if arguments.protocol != "apxinf-worker/2.0":
        parser.error("The protocol version is not supported.")
    try:
        contracts.validate_frame({"protocol": arguments.protocol, "kind": "drained",
                                  "worker_epoch": arguments.worker_epoch})
    except contracts.ContractError:
        parser.error("The worker epoch is invalid.")
    if not 1 <= arguments.prefill_step_size <= 8192 or not 1 <= arguments.output_batch_tokens <= 256:
        parser.error("The prefill or output batch size is invalid.")
    if not 1 <= arguments.max_context <= 131072 or not 0 <= arguments.max_output_tokens <= 65536:
        parser.error("The deployment token limits are invalid.")
    if not 1 <= arguments.memory_limit_bytes <= 9007199254740991:
        parser.error("The memory limit is invalid.")
    output_fd = sys.stdout.fileno()

    def write_all(payload: bytes) -> None:
        view = memoryview(payload)
        while view:
            written = os.write(output_fd, view)
            view = view[written:]

    outbox = ByteOutbox(write_all)
    try:
        runtime = MLXRuntime(arguments.model, arguments.max_context, arguments.prefill_step_size,
                             arguments.output_batch_tokens, arguments.memory_limit_bytes)
        worker = TextWorker(runtime, arguments.worker_epoch, outbox.put,
                            max_output_tokens=arguments.max_output_tokens,
                            output_batch_tokens=arguments.output_batch_tokens)
        worker.run(sys.stdin.buffer)
        outbox.close()
        return 0
    except Exception as error:
        print(f"Text worker stopped: {type(error).__name__}.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
