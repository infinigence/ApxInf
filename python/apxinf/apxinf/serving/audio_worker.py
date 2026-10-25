"""Persistent single-task audio worker with an independent control reader."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import stat
import sys
import threading

from audio_contracts import (Accepted, AsrInput, AudioLimits, Cancel, Cancelled, ControlAck, Failed,
    Frame, MAX_FRAME, MediaGrant, PROTOCOL, Ready, ResourcesReleased, Shutdown, Speech, Stopped,
    Submit, Terminal, Transcript, parse_frame, strict_json)
from audio_provider import CpuFixtureProvider, MlxAudioProvider
from audio_wav import inspect


class BoundedLog:
    def write(self, message):
        raw = str(message).encode("utf-8", "replace")[:2048]
        try:
            os.write(2, raw)
        except OSError:
            pass
        return len(message)

    def flush(self):
        return None


class Worker:
    def __init__(self, args):
        self.args = args
        self.protocol_fd = os.dup(1)
        os.set_blocking(2, False)
        os.dup2(2, 1)
        sys.stdout = BoundedLog()
        sys.stderr = BoundedLog()
        self.write_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.pending = queue.Queue(maxsize=1)
        self.cancelled = threading.Event()
        self.stopping = threading.Event()
        self.active = None
        self.seen_commands = set()
        self.seen_keys = set()

    def emit(self, message):
        raw = json.dumps(asdict(Frame(PROTOCOL, self.args.epoch, message)), ensure_ascii=False, separators=(",", ":")).encode()
        parse_frame(raw)
        with self.write_lock:
            view = memoryview(raw + b"\n")
            while view:
                count = os.write(self.protocol_fd, view)
                view = view[count:]

    def stop(self, *_):
        self.stopping.set()
        self.cancelled.set()

    def controls(self):
        try:
            while True:
                raw = sys.stdin.buffer.readline(MAX_FRAME + 2)
                if not raw:
                    self.stop()
                    return
                if not raw.endswith(b"\n") or len(raw) > MAX_FRAME + 1:
                    raise ValueError("invalid control framing")
                frame = parse_frame(raw[:-1])
                if frame.worker_epoch != self.args.epoch:
                    raise ValueError("stale worker epoch")
                message = frame.message
                if not isinstance(message, (Submit, Cancel, Shutdown)):
                    raise ValueError("unexpected control message")
                with self.state_lock:
                    if message.command_id in self.seen_commands or len(self.seen_commands) >= 1024:
                        raise ValueError("duplicate command or exhausted command history")
                    self.seen_commands.add(message.command_id)
                    if isinstance(message, Submit):
                        execution = (message.key.request_id, message.key.attempt)
                        if execution in self.seen_keys or self.active is not None or self.stopping.is_set():
                            raise ValueError("duplicate or concurrent submit")
                        self.seen_keys.add(execution)
                        self.active = message.key
                        self.cancelled = threading.Event()
                        self.pending.put_nowait(message)
                    elif isinstance(message, Cancel):
                        active = self.active == message.key
                        if active:
                            self.cancelled.set()
                        self.emit(ControlAck(message.command_id, "cancel_requested" if active else "already_terminal"))
                    else:
                        self.stop()
                        self.emit(ControlAck(message.command_id, "shutdown_requested"))
        except Exception as error:
            print("audio control fault: " + str(error), file=sys.stderr)
            os._exit(2)

    def read_input(self, grant):
        path = Path(self.args.media_dir) / grant.handle
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size != grant.bytes:
                raise ValueError("invalid input file")
            data = source.read(grant.bytes + 1)
            after = os.fstat(source.fileno())
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError("input changed during read")
        if len(data) != grant.bytes or hashlib.sha256(data).hexdigest() != grant.sha256:
            raise ValueError("input digest mismatch")
        info = inspect(data, True)
        if (info.frames, info.sample_rate) != (grant.frames, grant.sample_rate):
            raise ValueError("input shape mismatch")
        return data

    def write_output(self, submit, data):
        info = inspect(data, False)
        handle = submit.operation.output_handle
        fd = os.open(Path(self.args.media_dir) / handle, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as target:
            target.write(data)
            target.flush()
            os.fchmod(target.fileno(), 0o400)
        return MediaGrant(handle, self.args.epoch, submit.key, len(data), hashlib.sha256(data).hexdigest(), info.frames, info.sample_rate)

    def run(self):
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        threading.Thread(target=self.controls, name="audio-control", daemon=True).start()
        with open(self.args.manifest, "rb") as source:
            raw = source.read(MAX_FRAME + 1)
        revision = hashlib.sha256(raw).hexdigest()
        if revision != self.args.manifest_revision:
            raise ValueError("manifest changed before provider construction")
        manifest = strict_json(raw)
        if self.stopping.is_set():
            self.emit(Stopped())
            return
        if manifest["provider"] == "cpu_fixture":
            if not self.args.allow_cpu_fixture or not self.args.disable_pressure_check:
                raise ValueError("CPU fixture requires explicit test policy")
            provider = CpuFixtureProvider(manifest, self.args.fixture_delay_ms)
        else:
            if self.args.allow_cpu_fixture or self.args.disable_pressure_check:
                raise ValueError("production worker requires pressure policy")
            provider = MlxAudioProvider(manifest)
        capabilities = {"task": manifest["task"], "languages": manifest["languages"],
            "voices": sorted(manifest["voices"]), "limits": asdict(AudioLimits()),
            "cancellation": "provider_boundary_or_process_reap"}
        cap_revision = hashlib.sha256(json.dumps(capabilities, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        if not self.stopping.is_set():
            self.emit(Ready(manifest["task"], manifest["model"], revision, cap_revision, revision,
                manifest["provider"], manifest["languages"], sorted(manifest["voices"]),
                "provider_boundary_or_process_reap", AudioLimits()))
        while not self.stopping.is_set():
            try:
                submit = self.pending.get(timeout=0.05)
            except queue.Empty:
                continue
            self.emit(Accepted(submit.key, 0))
            cancelled = self.cancelled
            deadline = threading.Timer(submit.timeout_ms / 1000, cancelled.set)
            deadline.daemon = True
            deadline.start()
            try:
                op = submit.operation
                if submit.model_revision != revision or op.task != manifest["task"]:
                    raise ValueError("task or model identity mismatch")
                if op.language not in manifest["languages"]:
                    raise ValueError("unsupported language")
                if isinstance(op, AsrInput):
                    result = provider.transcribe(op, self.read_input(op.input), cancelled)
                    result = Cancelled() if result is None else Transcript(result)
                else:
                    if op.voice not in manifest["voices"]:
                        raise ValueError("unsupported voice")
                    audio = provider.synthesize(op, cancelled)
                    result = Cancelled() if audio is None else Speech(self.write_output(submit, audio))
            except Exception as error:
                code = "output_limit" if "output_limit" in str(error) else "provider_error"
                print("audio provider error: " + str(error), file=sys.stderr)
                result = Failed(code, "The audio provider could not complete the request.")
            finally:
                deadline.cancel()
            self.emit(Terminal(submit.key, 1, result))
            # Cleanup runs on the execution owner before settlement can release admission.
            provider.settle()
            with self.state_lock:
                self.active = None
            self.emit(ResourcesReleased(submit.key, 2, submit.lease_id))
        provider.settle()
        self.emit(Stopped())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-revision", required=True)
    parser.add_argument("--epoch", required=True)
    parser.add_argument("--media-dir", required=True)
    parser.add_argument("--allow-cpu-fixture", action="store_true")
    parser.add_argument("--disable-pressure-check", action="store_true")
    parser.add_argument("--fixture-delay-ms", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.fixture_delay_ms <= 130000 or (args.fixture_delay_ms and not args.allow_cpu_fixture):
        parser.error("fixture delay requires the CPU fixture")
    worker = Worker(args)
    try:
        worker.run()
    except Exception as error:
        print("audio worker startup or settlement fault: " + str(error), file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
