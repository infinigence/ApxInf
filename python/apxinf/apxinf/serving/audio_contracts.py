from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Literal
import uuid

PROTOCOL = "apxinf-audio-worker/1.0"
MAX_FRAME = 65536
SAFE_INTEGER = 9007199254740991
MAX_UPLOAD = 2 * 1024 * 1024
MAX_OUTPUT = 8 * 1024 * 1024
MEDIA_QUOTA = 32 * 1024 * 1024
ERROR_CODES = frozenset(("provider_error", "output_limit", "missing_resource"))
AudioTask = Literal["asr", "tts"]


@dataclass(frozen=True)
class AttemptKey:
    request_id: str
    attempt: int


@dataclass(frozen=True)
class MediaGrant:
    handle: str
    worker_epoch: str
    key: AttemptKey
    bytes: int
    sha256: str
    frames: int
    sample_rate: int


@dataclass(frozen=True)
class AudioLimits:
    upload_bytes: int = MAX_UPLOAD
    asr_frames: int = 480000
    text_scalars: int = 4096
    text_bytes: int = 16384
    output_bytes: int = MAX_OUTPUT
    output_seconds: int = 60
    queue: int = 4
    uploads: int = 2
    media_bytes: int = MEDIA_QUOTA


@dataclass(frozen=True)
class AsrInput:
    input: MediaGrant
    language: str
    task: Literal["asr"] = "asr"


@dataclass(frozen=True)
class TtsInput:
    text: str
    voice: str
    language: str
    output_handle: str
    task: Literal["tts"] = "tts"


AudioOperation = AsrInput | TtsInput


@dataclass(frozen=True)
class Transcript:
    text: str
    status: Literal["transcript"] = "transcript"


@dataclass(frozen=True)
class Speech:
    output: MediaGrant
    status: Literal["speech"] = "speech"


@dataclass(frozen=True)
class Cancelled:
    status: Literal["cancelled"] = "cancelled"


@dataclass(frozen=True)
class Failed:
    code: str
    message: str
    status: Literal["failed"] = "failed"


AudioResult = Transcript | Speech | Cancelled | Failed


@dataclass(frozen=True)
class Ready:
    task: AudioTask
    model: str
    model_revision: str
    capability_revision: str
    manifest_revision: str
    provider: str
    languages: list[str]
    voices: list[str]
    cancellation: str
    limits: AudioLimits
    kind: Literal["ready"] = "ready"


@dataclass(frozen=True)
class Submit:
    command_id: str
    key: AttemptKey
    lease_id: str
    model_revision: str
    timeout_ms: int
    operation: AudioOperation
    kind: Literal["submit"] = "submit"


@dataclass(frozen=True)
class Cancel:
    command_id: str
    key: AttemptKey
    kind: Literal["cancel"] = "cancel"


@dataclass(frozen=True)
class Shutdown:
    command_id: str
    kind: Literal["shutdown"] = "shutdown"


@dataclass(frozen=True)
class Accepted:
    key: AttemptKey
    event_seq: int
    kind: Literal["accepted"] = "accepted"


@dataclass(frozen=True)
class Terminal:
    key: AttemptKey
    event_seq: int
    result: AudioResult
    kind: Literal["terminal"] = "terminal"


@dataclass(frozen=True)
class ResourcesReleased:
    key: AttemptKey
    event_seq: int
    lease_id: str
    kind: Literal["resources_released"] = "resources_released"


@dataclass(frozen=True)
class ControlAck:
    command_id: str
    status: str
    kind: Literal["control_ack"] = "control_ack"


@dataclass(frozen=True)
class Stopped:
    kind: Literal["stopped"] = "stopped"


Message = Ready | Submit | Cancel | Shutdown | Accepted | Terminal | ResourcesReleased | ControlAck | Stopped


@dataclass(frozen=True)
class Frame:
    protocol: str
    worker_epoch: str
    message: Message


def is_uuid(value):
    try:
        return isinstance(value, str) and str(uuid.UUID(value)) == value
    except ValueError:
        return False


def is_digest(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def is_text(value):
    whitespace = "\u0009\u000a\u000b\u000c\u000d\u0020\u0085\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000"
    return isinstance(value, str) and any(c not in whitespace for c in value) and len(value) <= 4096 and len(value.encode("utf-8")) <= 16384


def _object(value, fields):
    if not isinstance(value, dict) or set(value) != set(fields.split()):
        raise ValueError("unknown or missing fields")


def _require(condition):
    if not condition:
        raise ValueError("invalid audio frame")


def _integer(value, low=0, high=SAFE_INTEGER):
    return type(value) is int and low <= value <= high


def _label(value):
    return isinstance(value, str) and 0 < len(value.encode("utf-8")) <= 128 and not any(ord(c) < 32 or 127 <= ord(c) <= 159 for c in value)


def _key(value):
    _object(value, "request_id attempt")
    _require(is_uuid(value["request_id"]) and _integer(value["attempt"], 1))
    return AttemptKey(**value)


def _grant(value):
    _object(value, "handle worker_epoch key bytes sha256 frames sample_rate")
    k = _key(value["key"])
    _require(is_uuid(value["handle"]) and is_uuid(value["worker_epoch"]) and is_digest(value["sha256"]))
    _require(_integer(value["bytes"], 1, MAX_OUTPUT) and _integer(value["sample_rate"], 8000, 48000))
    _require(_integer(value["frames"], 1, value["sample_rate"] * 60))
    return MediaGrant(**{**value, "key": k})


def strict_json(raw):
    if len(raw) > MAX_FRAME:
        raise ValueError("frame limit")

    def pairs(items):
        result = {}
        for name, value in items:
            if name in result or len(result) == 64:
                raise ValueError("duplicate key or object limit")
            result[name] = value
        return result

    def integer(value):
        if value == "-0":
            raise ValueError("negative zero integer")
        result = int(value)
        if abs(result) > SAFE_INTEGER:
            raise ValueError("unsafe integer")
        return result

    def floating(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("nonfinite number")
        return result

    def constant(value):
        raise ValueError("nonfinite number")

    value = json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=pairs, parse_int=integer, parse_float=floating, parse_constant=constant)

    def check(item, depth):
        _require(depth <= 12)
        if isinstance(item, str):
            item.encode("utf-8")
        elif isinstance(item, list):
            _require(len(item) <= 256)
            for child in item:
                check(child, depth + 1)
        elif isinstance(item, dict):
            for name, child in item.items():
                name.encode("utf-8")
                check(child, depth + 1)
    check(value, 0)
    return value


def parse_frame(raw: bytes) -> Frame:
    value = strict_json(raw)
    _object(value, "protocol worker_epoch message")
    epoch = value["worker_epoch"]
    _require(value["protocol"] == PROTOCOL and is_uuid(epoch))
    m = value["message"]
    _require(isinstance(m, dict))
    kind = m.get("kind")
    fields = {
        "ready": "kind task model model_revision capability_revision manifest_revision provider languages voices cancellation limits",
        "submit": "kind command_id key lease_id model_revision timeout_ms operation",
        "cancel": "kind command_id key", "shutdown": "kind command_id",
        "accepted": "kind key event_seq", "terminal": "kind key event_seq result",
        "resources_released": "kind key event_seq lease_id", "control_ack": "kind command_id status", "stopped": "kind",
    }
    _require(isinstance(kind, str) and kind in fields)
    _object(m, fields[kind])
    args = {k: v for k, v in m.items() if k != "kind"}
    if "command_id" in args:
        _require(is_uuid(args["command_id"]))
    if "key" in args:
        args["key"] = _key(args["key"])
    if "event_seq" in args:
        _require(_integer(args["event_seq"]))
    if "lease_id" in args:
        _require(is_uuid(args["lease_id"]))
    if kind == "ready":
        _object(args["limits"], "upload_bytes asr_frames text_scalars text_bytes output_bytes output_seconds queue uploads media_bytes")
        _require(all(_integer(v) for v in args["limits"].values()))
        args["limits"] = AudioLimits(**args["limits"])
        _require(args["limits"] == AudioLimits() and args["task"] in ("asr", "tts"))
        _require(_label(args["model"]) and all(is_digest(args[k]) for k in ("model_revision", "capability_revision", "manifest_revision")))
        _require(args["provider"] in ("mlx_audio", "cpu_fixture"))
        for name in ("languages", "voices"):
            _require(isinstance(args[name], list) and len(args[name]) <= 32 and all(_label(v) for v in args[name]))
        _require(bool(args["languages"]) and (args["task"] != "tts" or bool(args["voices"])))
        _require(args["cancellation"] == "provider_boundary_or_process_reap")
        message = Ready(**args)
    elif kind == "submit":
        _require(is_digest(args["model_revision"]) and _integer(args["timeout_ms"], 1, 120000))
        op = args["operation"]
        _require(isinstance(op, dict))
        if op.get("task") == "asr":
            _object(op, "task input language")
            g = _grant(op["input"])
            _require(g.key == args["key"] and g.worker_epoch == epoch and g.sample_rate == 16000)
            _require(g.frames <= 480000 and g.bytes <= MAX_UPLOAD and _label(op["language"]))
            args["operation"] = AsrInput(g, op["language"])
        else:
            _object(op, "task text voice language output_handle")
            _require(op["task"] == "tts" and is_text(op["text"]) and _label(op["voice"]) and _label(op["language"]) and is_uuid(op["output_handle"]))
            args["operation"] = TtsInput(op["text"], op["voice"], op["language"], op["output_handle"])
        message = Submit(**args)
    elif kind == "terminal":
        r = args["result"]
        _require(isinstance(r, dict))
        status = r.get("status")
        if status == "transcript":
            _object(r, "status text")
            _require(isinstance(r["text"], str) and len(r["text"].encode("utf-8")) <= 32768)
            result = Transcript(r["text"])
        elif status == "speech":
            _object(r, "status output")
            g = _grant(r["output"])
            _require(g.key == args["key"] and g.worker_epoch == epoch)
            result = Speech(g)
        elif status == "cancelled":
            _object(r, "status")
            result = Cancelled()
        else:
            _object(r, "status code message")
            _require(status == "failed" and r["code"] in ERROR_CODES and isinstance(r["message"], str) and 0 < len(r["message"].encode("utf-8")) <= 512)
            result = Failed(r["code"], r["message"])
        message = Terminal(args["key"], args["event_seq"], result)
    elif kind == "control_ack":
        _require(args["status"] in ("cancel_requested", "already_terminal", "shutdown_requested"))
        message = ControlAck(**args)
    else:
        message = {"cancel": Cancel, "shutdown": Shutdown, "accepted": Accepted,
            "resources_released": ResourcesReleased, "stopped": Stopped}[kind](**args)
    return Frame(PROTOCOL, epoch, message)


class AttemptState:
    def __init__(self, key: AttemptKey, lease_id: str):
        self.key = key
        self.lease_id = lease_id
        self.phase = "submitted"
        self.next_seq = 0

    def event(self, message):
        _require(isinstance(message, (Accepted, Terminal, ResourcesReleased)))
        _require(message.key == self.key and message.event_seq == self.next_seq)
        if isinstance(message, Accepted) and self.phase == "submitted":
            self.phase = "accepted"
        elif isinstance(message, Terminal) and self.phase == "accepted":
            self.phase = "terminal"
        elif isinstance(message, ResourcesReleased) and self.phase == "terminal" and message.lease_id == self.lease_id:
            self.phase = "settled"
        else:
            raise ValueError("invalid attempt transition")
        self.next_seq += 1
