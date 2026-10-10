"""Handwritten validators for the serial apxinf-worker/2.0 profile."""

from __future__ import annotations

import hashlib
import json
import re
import struct
from typing import Any

PROTOCOL = "apxinf-worker/2.0"
MAX_FRAME_BYTES = 1_048_576
MAX_DEPTH = 16
MAX_COMMANDS = 10_000
MAX_PROMPT_TOKENS = 131_072
MAX_OUTPUT_TOKENS = 65_536
MAX_SAFE_INTEGER = 9_007_199_254_740_991
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")
DIGEST = re.compile(r"[0-9a-f]{64}\Z")
ERROR_CODES = frozenset({
    "invalid_request", "unsupported_feature", "context_limit", "model_not_found",
    "session_not_found", "session_conflict", "prefix_mismatch", "queue_full",
    "quota_exceeded", "capacity_unavailable", "model_unavailable", "worker_lost",
    "deadline_exceeded", "protocol_fault", "internal_error", "state_invalid",
})
BASE = {"protocol", "kind", "worker_epoch"}
COMMAND = BASE | {"command_id"}
REQUEST_COMMAND = COMMAND | {"request_id", "attempt"}
EVENT = BASE | {"request_id", "attempt", "event_seq"}
ATTEMPT_EVENTS = frozenset({"accepted", "rejected", "prefill_progress", "tokens", "terminal", "resources_released"})


class ContractError(ValueError):
    """An invalid wire value. The frame is diagnostic data, not validated input."""

    def __init__(self, message: str, *, frame: dict | None = None):
        super().__init__(message)
        self.frame = frame


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _keys(value: Any, required: set[str], optional: set[str] = frozenset()) -> None:
    _require(type(value) is dict, "expected object")
    _require(required <= value.keys() and value.keys() <= required | optional, "unexpected or missing fields")


def _int(value: Any, low: int = 0, high: int = MAX_SAFE_INTEGER) -> None:
    _require(type(value) is int and low <= value <= high, "integer outside allowed range")


def _str(value: Any) -> None:
    _require(type(value) is str, "expected string")
    _require(not any(0xD800 <= ord(c) <= 0xDFFF for c in value), "unpaired surrogate")


def _enum(value: Any, choices: tuple | set | frozenset) -> None:
    _str(value)
    _require(value in choices, "unsupported enum value")


def _uuid(value: Any) -> None:
    _str(value)
    _require(UUID.fullmatch(value) is not None, "invalid lowercase UUID")


def _digest(value: Any) -> None:
    _str(value)
    _require(DIGEST.fullmatch(value) is not None, "invalid SHA-256 digest")


def _tokens(value: Any, maximum: int, minimum: int = 0) -> None:
    _require(type(value) is list and minimum <= len(value) <= maximum, "invalid token array length")
    for token in value:
        _int(token, 0, 2_147_483_647)


def _leases(value: Any) -> None:
    _require(type(value) is list and len(value) <= 64, "invalid lease array")
    for lease in value:
        _uuid(lease)
    _require(len(set(value)) == len(value), "duplicate lease ID")


def _tree(value: Any, depth: int = 1, *, identity: bool = False) -> None:
    _require(depth <= MAX_DEPTH, "maximum JSON depth exceeded")
    if type(value) is dict:
        _require(len(value) <= MAX_PROMPT_TOKENS, "object too large")
        for key, child in value.items():
            _str(key)
            if identity:
                _require(key.isascii(), "identity keys must be ASCII")
            _tree(child, depth + 1, identity=identity)
    elif type(value) is list:
        _require(len(value) <= MAX_PROMPT_TOKENS, "array too large")
        for child in value:
            _tree(child, depth + 1, identity=identity)
    elif type(value) is str:
        _str(value)
    elif type(value) is int:
        _int(value, -MAX_SAFE_INTEGER if not identity else 0)
    elif type(value) is bool:
        pass
    elif value is None:
        _require(not identity, "identity cannot contain null")
    elif type(value) is float and not identity:
        _require(float("-inf") < value < float("inf"), "non-finite JSON number")
    else:
        raise ContractError("unsupported JSON value")


def _error(value: Any) -> None:
    _keys(value, {"code", "message", "scope", "state_validity"})
    _enum(value["code"], ERROR_CODES)
    _str(value["message"])
    _enum(value["scope"], {"request", "session", "worker", "service"})
    _enum(value["state_validity"], {"unchanged", "committed", "invalid", "none"})


def _messages(messages: Any, tools: Any) -> None:
    _require(type(messages) is list and 1 <= len(messages) <= 256, "invalid messages array")
    for message in messages:
        _keys(message, {"role", "content"}, {"tool_call_id", "name", "tool_calls"})
        _enum(message["role"], {"system", "user", "assistant", "tool"})
        _str(message["content"])
        for key in ("tool_call_id", "name"):
            if key in message:
                _str(message[key])
        if "tool_calls" in message:
            calls = message["tool_calls"]
            _require(type(calls) is list and len(calls) <= 256, "invalid tool calls")
            for call in calls:
                _keys(call, {"id", "type", "function"})
                _str(call["id"])
                _enum(call["type"], {"function"})
                _keys(call["function"], {"name", "arguments"})
                _str(call["function"]["name"])
                _require(type(call["function"]["arguments"]) in (dict, str), "invalid tool arguments")
    _require(type(tools) is list and len(tools) <= 256, "invalid tools array")
    for tool in tools:
        _keys(tool, {"type", "function"})
        _enum(tool["type"], {"function"})
        function = tool["function"]
        _keys(function, {"name"}, {"description", "parameters", "strict"})
        _str(function["name"])
        if "description" in function:
            _str(function["description"])
        if "parameters" in function:
            _require(type(function["parameters"]) is dict, "tool parameters must be an object")
        if "strict" in function:
            _require(type(function["strict"]) is bool, "tool strict must be Boolean")


def _capabilities(value: Any) -> None:
    _keys(value, {"text", "image", "greedy", "sampling", "token_events", "cancel_granularity", "max_active_sequences", "max_context", "max_output_tokens", "exact_append", "state_schema"})
    for name, expected in {"text": True, "image": False, "greedy": True, "sampling": False, "token_events": True, "exact_append": False}.items():
        _require(type(value[name]) is bool and value[name] is expected, "unsupported serial capability")
    _enum(value["cancel_granularity"], {"token", "token_or_prefill_chunk"})
    _int(value["max_active_sequences"], 1, 1)
    _int(value["max_context"], 1, MAX_PROMPT_TOKENS)
    _int(value["max_output_tokens"], 0, MAX_OUTPUT_TOKENS)
    _enum(value["state_schema"], {"none"})


def _counts(value: Any, fields: set[str]) -> None:
    _keys(value, fields)
    for count in value.values():
        _int(count)


def validate_command_envelope(frame: dict) -> None:
    """Check routing and settlement fields, without accepting request content."""
    _tree(frame)
    _require(type(frame) is dict, "expected command object")
    _require(frame.get("protocol") == PROTOCOL, "unsupported protocol")
    _uuid(frame.get("worker_epoch"))
    _uuid(frame.get("command_id"))
    kind = frame.get("kind")
    shapes = {
        "prepare_input": (COMMAND, {"model_revision", "messages", "tools", "template_options"}),
        "submit": (REQUEST_COMMAND, {"model_revision", "capability_revision", "token_ids", "max_tokens", "remaining_timeout_ms", "eos_token_ids", "capacity_lease_ids"}),
        "cancel_request": (REQUEST_COMMAND, {"reason"}),
        "stop_generation": (REQUEST_COMMAND, {"cause", "output_token_count", "text_byte_cutoff"}),
        "query_stats": (COMMAND, set()), "drain": (COMMAND, set()), "shutdown": (COMMAND, set()),
    }
    _str(kind)
    _require(kind in shapes, "unknown command kind")
    base, payload = shapes[kind]
    _keys(frame, base | payload)
    if kind in {"submit", "cancel_request", "stop_generation"}:
        _uuid(frame.get("request_id"))
        _int(frame.get("attempt"), 1)
    if kind == "submit":
        _leases(frame["capacity_lease_ids"])


def validate_frame(frame: dict) -> None:
    """Check complete frame shape without changing the supplied object."""
    _tree(frame)
    _require(type(frame) is dict and BASE <= frame.keys(), "missing frame envelope")
    _require(frame["protocol"] == PROTOCOL, "unsupported protocol")
    _uuid(frame["worker_epoch"])
    kind = frame["kind"]
    _str(kind)
    if "command_id" in frame:
        _uuid(frame["command_id"])
    if kind in ATTEMPT_EVENTS or kind in {"submit", "cancel_request", "stop_generation"}:
        _uuid(frame.get("request_id"))
        _int(frame.get("attempt"), 1)
    if kind in ATTEMPT_EVENTS:
        _int(frame.get("event_seq"))
    if kind == "ready":
        _keys(frame, BASE | {"model_revision", "capability_revision", "model_manifest", "capabilities", "limits", "model_path", "eos_token_ids", "vocab_size", "runtime", "memory"})
        _capabilities(frame["capabilities"])
        _keys(frame["limits"], {"max_frame_bytes", "max_depth", "max_commands"})
        _require(frame["limits"] == {"max_frame_bytes": MAX_FRAME_BYTES, "max_depth": MAX_DEPTH, "max_commands": MAX_COMMANDS}, "incompatible transport limits")
        for value in frame["limits"].values():
            _int(value)
        _keys(frame["runtime"], {"python", "mlx", "mlx_lm"})
        for value in frame["runtime"].values():
            _str(value)
        _counts(frame["memory"], {"active_bytes", "peak_bytes", "cache_bytes"})
        _str(frame["model_path"])
        _require(frame["model_path"].startswith("/"), "model path must be absolute")
        _int(frame["vocab_size"], 1, 2_147_483_648)
        _tokens(frame["eos_token_ids"], 256)
        _require(all(token < frame["vocab_size"] for token in frame["eos_token_ids"]), "EOS outside vocabulary")
        _require(type(frame["model_manifest"]) is dict, "model manifest must be an object")
        _require(frame["model_revision"] == canonical_identity_digest("apxinf-model-identity-v1", frame["model_manifest"]), "model revision mismatch")
        _require(frame["capability_revision"] == canonical_identity_digest("apxinf-capability-v1", frame["capabilities"]), "capability revision mismatch")
    elif kind == "prepare_input":
        _keys(frame, COMMAND | {"model_revision", "messages", "tools", "template_options"})
        _digest(frame["model_revision"])
        _messages(frame["messages"], frame["tools"])
        _keys(frame["template_options"], {"enable_thinking"})
        _require(type(frame["template_options"]["enable_thinking"]) is bool, "enable_thinking must be Boolean")
    elif kind == "prepared_input":
        _keys(frame, COMMAND | {"model_revision", "token_ids", "effective_input_tokens", "prompt_digest"})
        _digest(frame["model_revision"])
        _tokens(frame["token_ids"], MAX_PROMPT_TOKENS, 1)
        _int(frame["effective_input_tokens"], 1, MAX_PROMPT_TOKENS)
        _require(frame["effective_input_tokens"] == len(frame["token_ids"]), "prepared input count mismatch")
        _require(frame["prompt_digest"] == token_prefix_digest(frame["token_ids"]), "prepared input digest mismatch")
    elif kind == "submit":
        _keys(frame, REQUEST_COMMAND | {"model_revision", "capability_revision", "token_ids", "max_tokens", "remaining_timeout_ms", "eos_token_ids", "capacity_lease_ids"})
        _digest(frame["model_revision"])
        _digest(frame["capability_revision"])
        _tokens(frame["token_ids"], MAX_PROMPT_TOKENS, 1)
        _tokens(frame["eos_token_ids"], 256)
        _int(frame["max_tokens"], 0, MAX_OUTPUT_TOKENS)
        _int(frame["remaining_timeout_ms"], 1, 3_600_000)
        _leases(frame["capacity_lease_ids"])
    elif kind == "cancel_request":
        _keys(frame, REQUEST_COMMAND | {"reason"})
        _enum(frame["reason"], {"user_cancel", "client_disconnect", "slow_consumer", "deadline_exceeded"})
    elif kind == "stop_generation":
        _keys(frame, REQUEST_COMMAND | {"cause", "output_token_count", "text_byte_cutoff"})
        _enum(frame["cause"], {"stop_sequence", "tool_calls"})
        _int(frame["output_token_count"], 0, MAX_OUTPUT_TOKENS)
        _int(frame["text_byte_cutoff"])
    elif kind in {"query_stats", "drain", "shutdown"}:
        _keys(frame, COMMAND)
    elif kind == "accepted":
        _keys(frame, EVENT)
    elif kind == "rejected":
        _keys(frame, EVENT | {"error"})
        _error(frame["error"])
    elif kind == "prefill_progress":
        _keys(frame, EVENT | {"consumed_tokens"})
        _int(frame["consumed_tokens"], 0, MAX_PROMPT_TOKENS)
    elif kind == "tokens":
        _keys(frame, EVENT | {"output_index", "token_ids", "text_delta"})
        _int(frame["output_index"], 0, MAX_OUTPUT_TOKENS)
        _tokens(frame["token_ids"], 256, 1)
        _str(frame["text_delta"])
    elif kind == "terminal":
        _keys(frame, EVENT | {"status", "cause", "usage", "state_result", "text_delta", "metrics"}, {"error"})
        causes = {"completed": {"eos", "length", "stop_sequence", "tool_calls"}, "cancelled": {"user_cancel", "client_disconnect", "slow_consumer"}, "expired": {"deadline_exceeded"}, "failed": ERROR_CODES}
        _enum(frame["status"], set(causes))
        _enum(frame["cause"], causes[frame["status"]])
        _require(("error" in frame) == (frame["status"] == "failed"), "invalid terminal error presence")
        if "error" in frame:
            _error(frame["error"])
            _require(frame["error"]["code"] == frame["cause"], "terminal cause differs from error")
        _counts(frame["usage"], {"input_tokens", "output_tokens"})
        _keys(frame["state_result"], {"validity", "consumed_position", "history_count"})
        _enum(frame["state_result"]["validity"], {"none"})
        _int(frame["state_result"]["history_count"])
        _int(frame["state_result"]["consumed_position"], 0, frame["state_result"]["history_count"])
        _str(frame["text_delta"])
        _counts(frame["metrics"], {"elapsed_ns", "ttft_ns", "peak_memory_bytes"})
        _require(frame["metrics"]["ttft_ns"] <= frame["metrics"]["elapsed_ns"], "TTFT exceeds elapsed time")
    elif kind == "resources_released":
        _keys(frame, EVENT | {"released_lease_ids", "retained_lease_ids"})
        _leases(frame["released_lease_ids"])
        _leases(frame["retained_lease_ids"])
        _require(not frame["retained_lease_ids"], "stateless profile cannot retain leases")
    elif kind == "command_result":
        _keys(frame, COMMAND | {"status"}, {"error"})
        _enum(frame["status"], {"accepted", "already_registered", "already_terminal", "error"})
        _require(("error" in frame) == (frame["status"] == "error"), "invalid command error presence")
        if "error" in frame:
            _error(frame["error"])
    elif kind == "stats":
        _keys(frame, COMMAND | {"active_requests", "completed_requests", "cancelled_requests"})
        _int(frame["active_requests"], 0, 1)
        _int(frame["completed_requests"])
        _int(frame["cancelled_requests"])
    elif kind == "worker_fault":
        _keys(frame, BASE | {"error"})
        _error(frame["error"])
    elif kind == "drained":
        _keys(frame, BASE)
    else:
        raise ContractError("unknown frame kind")


def _pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def parse_document(payload: bytes | str) -> Any:
    """Parse bounded JSON without frame semantics or JSONL framing."""
    try:
        raw = payload.encode("utf-8") if isinstance(payload, str) else payload
        _require(type(raw) is bytes and len(raw) <= MAX_FRAME_BYTES, "document exceeds byte limit")
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                              parse_int=lambda value: -0.0 if value == "-0" else int(value),
                              parse_constant=lambda _: (_ for _ in ()).throw(ContractError("non-finite JSON number")))
        _tree(document)
        return document
    except ContractError:
        raise
    except (UnicodeError, ValueError, RecursionError) as error:
        raise ContractError("invalid UTF-8 JSON document") from error


def decode_frame(line: bytes | str) -> dict:
    try:
        raw = line.encode("utf-8") if isinstance(line, str) else line
        _require(type(raw) is bytes and len(raw) <= MAX_FRAME_BYTES, "frame exceeds byte limit")
        payload = raw[:-1] if raw.endswith(b"\n") else raw
        _require(b"\n" not in payload and b"\r" not in payload, "frame contains a line break")
        frame = parse_document(payload)
        try:
            validate_frame(frame)
        except ContractError as error:
            error.frame = frame if type(frame) is dict else None
            raise
        return frame
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ContractError("invalid UTF-8 JSON frame") from error


def encode_frame(frame: dict) -> bytes:
    validate_frame(frame)
    encoded = json.dumps(frame, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    _require(len(encoded) <= MAX_FRAME_BYTES, "frame exceeds byte limit")
    return encoded


def canonical_identity_digest(domain: str, document: Any) -> str:
    _str(domain)
    _require(domain.isascii() and "\0" not in domain, "invalid identity domain")
    _tree(document, identity=True)

    def encode(value: Any) -> str:
        if type(value) is str:
            text = '"'
            for char in value:
                if char in ('"', "\\"):
                    text += "\\" + char
                elif ord(char) < 32:
                    text += "\\u%04x" % ord(char)
                else:
                    text += char
            return text + '"'
        if type(value) is bool:
            return "true" if value else "false"
        if type(value) is int:
            return str(value)
        if type(value) is list:
            return "[" + ",".join(encode(item) for item in value) + "]"
        return "{" + ",".join(encode(key) + ":" + encode(value[key]) for key in sorted(value)) + "}"

    return hashlib.sha256(domain.encode("ascii") + b"\0" + encode(document).encode("utf-8")).hexdigest()


def token_prefix_digest(tokens: list[int]) -> str:
    _tokens(tokens, MAX_PROMPT_TOKENS + MAX_OUTPUT_TOKENS)
    digest = hashlib.sha256(b"apxinf-token-prefix-v1\0")
    digest.update(struct.pack("<Q", len(tokens)))
    for token in tokens:
        digest.update(struct.pack("<I", token))
    return digest.hexdigest()


class AttemptTracker:
    """Check one attempt stream. Invalid events do not change tracker state."""

    def __init__(self, submit: dict):
        validate_frame(submit)
        _require(submit["kind"] == "submit", "tracker requires submit")
        self.key = tuple(submit[key] for key in ("worker_epoch", "request_id", "attempt"))
        self.input_tokens = len(submit["token_ids"])
        self.max_tokens = submit["max_tokens"]
        self.leases = set(submit["capacity_lease_ids"])
        self.next_seq = 0
        self.output_tokens = 0
        self.prefill_tokens = 0
        self.phase = "new"

    @property
    def done(self) -> bool:
        return self.phase == "released"

    def observe(self, event: dict) -> None:
        validate_frame(event)
        _require(event["kind"] in ATTEMPT_EVENTS, "not an attempt event")
        _require(tuple(event[key] for key in ("worker_epoch", "request_id", "attempt")) == self.key, "attempt identity mismatch")
        _require(event["event_seq"] == self.next_seq, "event sequence mismatch")
        kind = event["kind"]
        phase = self.phase
        output = self.output_tokens
        prefill = self.prefill_tokens
        if phase == "new":
            _require(kind in {"accepted", "rejected"}, "first event must accept or reject")
            phase = kind
        elif kind == "prefill_progress" and phase == "accepted":
            _require(output == 0 and prefill <= event["consumed_tokens"] <= self.input_tokens, "invalid prefill progress")
            prefill = event["consumed_tokens"]
        elif kind == "tokens" and phase == "accepted":
            _require(event["output_index"] == output, "non-contiguous token range")
            output += len(event["token_ids"])
            _require(output <= self.max_tokens, "output exceeds allowance")
        elif kind == "terminal" and phase in {"accepted", "rejected"}:
            _require(phase != "rejected" or event["status"] == "failed", "rejection requires failed terminal")
            _require(event["usage"] == {"input_tokens": self.input_tokens, "output_tokens": output}, "terminal usage mismatch")
            _require(event["state_result"]["history_count"] == self.input_tokens + output, "history count mismatch")
            _require(output != 0 or event["metrics"]["ttft_ns"] == 0, "empty output must have zero TTFT")
            if event["status"] == "completed" and event["cause"] == "length":
                _require(output == self.max_tokens, "length completion requires full allowance")
            phase = "terminal"
        elif kind == "resources_released" and phase == "terminal":
            _require(set(event["released_lease_ids"]) == self.leases, "lease settlement mismatch")
            phase = "released"
        else:
            raise ContractError("event is invalid in current attempt phase")
        self.phase = phase
        self.output_tokens = output
        self.prefill_tokens = prefill
        self.next_seq += 1
