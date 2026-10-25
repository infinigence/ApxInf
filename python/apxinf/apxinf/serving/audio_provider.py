"""Adapters call pinned MLX Audio public APIs without embedding dependency implementations."""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import struct
import sys
import time

from audio_contracts import AsrInput, MAX_OUTPUT, TtsInput
from audio_wav import encode_pcm16, inspect

AUDIO_COMMIT = "f976eab8b47a3cada5aeb3b52a78451ee06117bc"


def file_digest(path):
    result = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as source:
        before = os.fstat(source.fileno())
        while block := source.read(1024 * 1024):
            result.update(block)
        after = os.fstat(source.fileno())
        if (before.st_ino,before.st_size,before.st_mtime_ns) != (after.st_ino,after.st_size,after.st_mtime_ns):
            raise ValueError("artifact changed during identity check")
    return result.hexdigest()


def verify_local_profile(manifest):
    runtime = manifest["runtime"]
    if platform.python_version() != runtime["python"]:
        raise ValueError("Python runtime identity mismatch")
    if runtime["mlx"] != "0.32.4" or runtime["mlx_audio"] != "0.6.1" or runtime["mlx_audio_commit"] != AUDIO_COMMIT:
        raise ValueError("unapproved MLX Audio runtime")
    if any(re.sub(r"[-_.]+", "-", name).lower() in ("mlx", "mlx-audio") for name in runtime["packages"]):
        raise ValueError("extra package identities cannot override core runtime pins")
    for package, expected in {**runtime["packages"], "mlx": "0.32.4", "mlx-audio": "0.6.1"}.items():
        if importlib.metadata.version(package) != expected:
            raise ValueError("dependency runtime identity mismatch: " + package)
    direct = json.loads(importlib.metadata.distribution("mlx-audio").read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != AUDIO_COMMIT:
        raise ValueError("MLX Audio installation lacks the pinned commit identity")
    declared = {}
    for artifact in manifest["artifacts"]:
        path = Path(artifact["path"])
        if not path.is_absolute() or path.resolve() != path or not path.is_file() or path.is_symlink():
            raise ValueError("missing_resource: artifact must be a canonical regular file")
        if file_digest(path) != artifact["sha256"] or str(path) in declared:
            raise ValueError("artifact identity mismatch")
        declared[str(path)] = artifact["role"]
    model_dir = Path(manifest["model_dir"])
    if model_dir.resolve() != model_dir or not model_dir.is_dir():
        raise ValueError("missing_resource: local model directory")
    for path in model_dir.rglob("*"):
        if path.is_symlink() or (path.is_file() and str(path) not in declared):
            raise ValueError("model directory contains an unfrozen artifact")
    for voice in manifest["voices"].values():
        if declared.get(voice) != "voice" or not voice.endswith(".safetensors"):
            raise ValueError("missing_resource: frozen local voice")
    if manifest["family"] == "kokoro":
        if manifest["languages"] != ["en"]:
            raise ValueError("unsupported Kokoro language profile")
        for module in ("misaki", "en_core_web_sm"):
            spec = importlib.util.find_spec(module)
            if spec is None or spec.origin is None or declared.get(str(Path(spec.origin).resolve())) != "frontend":
                raise ValueError("missing_resource: frozen English frontend " + module)
        # The frontend can execute package data, so the manifest covers its entire package directory.
        for module in ("misaki", "en_core_web_sm"):
            root = Path(importlib.util.find_spec(module).origin).resolve().parent
            for path in root.rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts and declared.get(str(path.resolve())) != "frontend":
                    raise ValueError("missing_resource: unfrozen frontend package data")


def deny_network(event, args):
    if event in ("socket.connect", "socket.getaddrinfo", "subprocess.Popen", "os.system"):
        raise RuntimeError("offline audio worker forbids network and subprocess fallback")


class MlxAudioProvider:
    def __init__(self, manifest):
        self.manifest = manifest
        verify_local_profile(manifest)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        sys.addaudithook(deny_network)
        self.np = importlib.import_module("numpy")
        self.mx = importlib.import_module("mlx.core")
        if manifest["family"] == "kokoro":
            # Exercise the same public English fallback before model readiness.
            fallback = importlib.import_module("misaki.espeak").EspeakFallback(british=False)
            if fallback is None:
                raise ValueError("missing_resource: English phonemizer fallback")
        path = Path(manifest["model_dir"])
        if manifest["family"] == "whisper":
            loader = importlib.import_module("mlx_audio.stt.utils").load_model
        else:
            loader = importlib.import_module("mlx_audio.tts.utils").load_model
        self.model = loader(path, lazy=False, strict=True)
        family = manifest["family"]
        if family not in type(self.model).__module__.split("."):
            raise ValueError("loaded model family differs from the manifest")

    def transcribe(self, operation: AsrInput, audio: bytes, cancelled):
        info = inspect(audio, True)
        samples = self.np.frombuffer(audio, dtype="<i2", count=info.frames, offset=info.data_offset).astype(self.np.float32) / 32768.0
        if cancelled.is_set():
            return None
        output = self.model.generate(audio=samples, language=operation.language, stream=False, temperature=0, return_timestamps=False)
        self.mx.synchronize()
        if cancelled.is_set():
            return None
        result = output.text
        if not isinstance(result, str) or len(result.encode("utf-8")) > 32768:
            raise ValueError("invalid provider transcript")
        return result

    def synthesize(self, operation: TtsInput, cancelled):
        pcm = bytearray()
        sample_rate = None
        voice = self.manifest["voices"][operation.voice]
        iterator = self.model.generate(text=operation.text, voice=voice, speed=1.0, lang_code="a")
        try:
            for result in iterator:
                if cancelled.is_set():
                    return None
                rate = result.sample_rate
                audio = result.audio
                if type(rate) is not int or not 8000 <= rate <= 48000 or (sample_rate is not None and rate != sample_rate):
                    raise ValueError("invalid provider sample rate")
                sample_rate = rate
                # Check shape and size before any device-to-host copy or PCM allocation.
                if len(audio.shape) != 1 or type(audio.shape[0]) is not int or audio.shape[0] <= 0:
                    raise ValueError("invalid provider waveform shape")
                count = audio.shape[0]
                total = len(pcm) // 2 + count
                if total > rate * 60 or total * 2 + 44 > MAX_OUTPUT:
                    raise ValueError("output_limit")
                samples = self.np.asarray(audio, dtype=self.np.float32)
                if samples.shape != (count,) or not self.np.isfinite(samples).all():
                    raise ValueError("invalid provider waveform")
                encoded = (self.np.clip(samples, -1.0, 32767.0 / 32768.0) * 32768.0).astype("<i2").tobytes()
                pcm.extend(encoded)
            self.mx.synchronize()
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
        if cancelled.is_set():
            return None
        if sample_rate is None:
            raise ValueError("provider produced no audio")
        return encode_pcm16(bytes(pcm), sample_rate)

    def settle(self):
        self.mx.synchronize()
        self.mx.clear_cache()


class CpuFixtureProvider:
    """Transport fixture, enabled only by explicit test configuration."""
    def __init__(self, manifest, delay_ms=0):
        self.delay_ms = delay_ms

    def _wait(self, cancelled):
        until = time.monotonic() + self.delay_ms / 1000
        while time.monotonic() < until:
            if cancelled.wait(min(0.01, max(0, until - time.monotonic()))):
                return False
        return not cancelled.is_set()

    def transcribe(self, operation, audio, cancelled):
        info = inspect(audio, True)
        return "fixture transcript: " + str(info.frames) + " frames" if self._wait(cancelled) else None

    def synthesize(self, operation, cancelled):
        if not self._wait(cancelled):
            return None
        pcm = bytearray()
        for index in range(2400):
            pcm.extend(struct.pack("<h", int(math.sin(index * 2 * math.pi * 440 / 24000) * 4000)))
        return encode_pcm16(bytes(pcm), 24000)

    def settle(self):
        return None
