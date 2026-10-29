from __future__ import annotations

import hashlib
import json
import re
import struct
from typing import Any

PROTOCOL = "apxinf-worker/2.0"
VISION_PROTOCOL = "apxinf-worker/2.5"
NATIVE_PREPARED_PROTOCOL = "apxinf-worker/2.7"
NATIVE_PROTOCOL = "apxinf-worker/2.1"
NATIVE_SAMPLED_PROTOCOL = "apxinf-worker/2.2"
NATIVE_THINKING_PROTOCOL = "apxinf-worker/2.3"
NATIVE_BOUNDED_THINKING_PROTOCOL = "apxinf-worker/2.4"
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


def _messages(messages: Any, tools: Any, vision: bool = False) -> None:
    images = []
    _require(type(messages) is list and 1 <= len(messages) <= 256, "invalid messages array")
    for message in messages:
        _keys(message, {"role", "content"}, {"tool_call_id", "name", "tool_calls"})
        _enum(message["role"], {"system", "user", "assistant", "tool"})
        if vision:
            _vision_content(message["content"], message["role"], images)
        else:
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
    if vision:
        _vision_media(images)
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


def _sampling(value: Any) -> None:
    _require(type(value) is dict, "expected sampling object")
    _enum(value.get("mode"), {"greedy", "random"})
    if value["mode"] == "greedy":
        _keys(value, {"mode"})
        return
    _keys(value, {"mode", "temperature", "top_p", "seed"})
    for key, maximum in (("temperature", 2.0), ("top_p", 1.0)):
        number = value[key]
        _require(type(number) in (int, float) and 0 < number <= maximum,
                 "sampling number outside allowed range")
        narrowed = struct.unpack("!f", struct.pack("!f", number))[0]
        _require(0 < narrowed < float("inf"), "sampling number underflows binary32")
    _int(value["seed"])


def _submit_fields(frame: dict) -> set[str]:
    fields = {"model_revision", "capability_revision", "token_ids", "max_tokens", "remaining_timeout_ms", "eos_token_ids", "capacity_lease_ids"}
    if frame["protocol"] in (NATIVE_SAMPLED_PROTOCOL, NATIVE_THINKING_PROTOCOL, NATIVE_BOUNDED_THINKING_PROTOCOL, VISION_PROTOCOL):
        fields.add("sampling")
    if frame["protocol"] == VISION_PROTOCOL:
        fields.add("replay")
    return fields


def _vision_media(value: Any) -> None:
    if __package__:
        from .owned_media import MediaBatch, MediaError
    else:
        from owned_media import MediaBatch, MediaError
    try:
        MediaBatch.from_descriptors(value)
    except MediaError as error:
        raise ContractError(str(error)) from error


def _vision_content(value: Any, role: str, images: list) -> None:
    if type(value) is str:
        _str(value)
        return
    _enum(role, {"user", "tool"})
    _require(type(value) is list and 1 <= len(value) <= 256, "invalid content array")
    for part in value:
        _require(type(part) is dict, "expected content part")
        _enum(part.get("type"), {"text", "image"})
        if part["type"] == "text":
            _keys(part, {"type", "text"})
            _str(part["text"])
        else:
            _keys(part, {"type", "media"})
            images.append(part["media"])


def _vision_ready(frame: dict) -> None:
    runtime = {"python": "3.14.3", "mlx": "0.32.4", "mlx_vlm": "0.7.10", "transformers": "5.18.0"}
    _require(frame["runtime"] == runtime, "unsupported vision runtime")
    manifest = frame["model_manifest"]
    _require(manifest.get("runtime") == runtime
             and manifest.get("checkpoint") == "openbmb/MiniCPM-V-4.7-1B"
             and manifest.get("checkpoint_revision") == "379f120c2ff3588e6e06559ec6b9ec15cc05114e"
             and manifest.get("adapter_revision") == "apxinf-mlx-minicpmv47-vision-v1",
             "unsupported vision identity")
    _require(type(manifest.get("processor")) is dict, "expected processor identity")
    _require(manifest.get("processor_revision") == canonical_identity_digest("apxinf-vision-processor-v1", manifest["processor"]), "processor revision mismatch")
    artifacts = manifest.get("artifacts")
    _require(type(artifacts) is list and 1 <= len(artifacts) <= MAX_PROMPT_TOKENS, "invalid artifacts")
    previous = ""
    weights = False
    for artifact in artifacts:
        _keys(artifact, {"path", "size_bytes", "sha256"})
        path = artifact["path"]
        _str(path)
        _require(path > previous and not path.startswith("/") and all(part not in {"", ".", ".."} for part in path.split("/")), "invalid artifact path order")
        previous = path
        _int(artifact["size_bytes"])
        _digest(artifact["sha256"])
        if path == "model.safetensors":
            _require(artifact["size_bytes"] == 3109054480 and artifact["sha256"] == "8c4921ac83182c85190ad10c3aeb118d51e385d2d98ec13b88b514bbf6cac002", "vision weight identity mismatch")
            weights = True
    _require(weights, "missing vision weight identity")
    execution = manifest.get("execution")
    _keys(execution, {"provider", "precision", "enable_thinking", "kv_reuse", "max_context", "max_output_tokens", "prefill_step_size", "output_batch_tokens", "memory_limit_bytes", "selection"})
    _require(execution["provider"] == "mlx-vlm-minicpmv47" and execution["precision"] == "bf16"
             and execution["enable_thinking"] is False and execution["kv_reuse"] is False,
             "unsupported vision execution")
    _int(execution["prefill_step_size"], 256, 256)
    _int(execution["output_batch_tokens"], 1, 1)
    _int(execution["memory_limit_bytes"], 1)
    _int(execution["max_context"], 1, 16384)
    _int(execution["max_output_tokens"], 0, 8192)
    selection = execution["selection"]
    _require(selection == {"algorithm": "mlx-vlm-fixed-controls-v1", "top_k": 100, "min_p": "0", "repetition_penalty": "1.03", "repetition_context_size": 8192, "presence_penalty": "0", "frequency_penalty": "0", "seed_policy": "request-id-sha256-v1"}, "unsupported vision selection")
    _int(selection["top_k"], 100, 100)
    _int(selection["repetition_context_size"], 8192, 8192)
    caps = frame["capabilities"]
    _keys(caps, {"text", "image", "greedy", "sampling", "token_events", "cancel_granularity", "max_active_sequences", "max_context", "max_output_tokens", "exact_append", "state_schema", "media_limits"})
    for key, expected in {"text": True, "image": True, "greedy": True, "sampling": True, "token_events": True, "exact_append": False}.items():
        _require(caps[key] is expected, "unsupported vision capability")
    _require(caps["cancel_granularity"] == "yield_or_process_exit" and caps["state_schema"] == "none", "unsupported vision lifecycle")
    _int(caps["max_active_sequences"], 1, 1)
    for key in ("max_context", "max_output_tokens"):
        _int(caps[key])
        _require(caps[key] == execution[key], "vision capability bound mismatch")
    _require(caps["media_limits"] == {"max_images": 4, "max_image_bytes": 131072, "max_total_bytes": 262144, "max_dimension": 1024, "max_host_array_bytes": 67108864}, "unsupported vision media limits")
    for value in caps["media_limits"].values():
        _int(value)
    _int(frame["vocab_size"], 248057, 2_147_483_648)


def _vision_replay(frame: dict) -> None:
    replay = frame.get("replay")
    _keys(replay, {"worker_epoch", "model_revision", "capability_revision", "preparation_id", "rendered_prompt", "processor_revision", "processing_digest", "media", "image_bindings", "host_array_bytes"})
    _uuid(replay["worker_epoch"])
    _digest(replay["model_revision"])
    _digest(replay["capability_revision"])
    _require(replay["worker_epoch"] == frame["worker_epoch"] and replay["model_revision"] == frame["model_revision"], "replay identity mismatch")
    _require(frame["kind"] != "submit" or replay["capability_revision"] == frame["capability_revision"], "replay capability mismatch")
    _uuid(replay["preparation_id"])
    _require((replay["preparation_id"] == frame["command_id"]) == (frame["kind"] == "prepared_input"), "invalid replay command identity")
    _str(replay["rendered_prompt"])
    _require(len(replay["rendered_prompt"].encode("utf-8")) <= 131072, "rendered prompt exceeds byte limit")
    _digest(replay["processor_revision"])
    _digest(replay["processing_digest"])
    _int(replay["host_array_bytes"], 0, 67108864)
    _vision_media(replay["media"])
    bindings = replay["image_bindings"]
    _require(type(bindings) is list and len(bindings) == len(replay["media"]), "image binding count mismatch")
    ids = frame["token_ids"]
    _tokens(ids, 16384, 1)
    previous = 0
    for index, binding in enumerate(bindings):
        _keys(binding, {"source_index", "token_start", "token_end", "pixel_shape", "target_size"})
        _int(binding["source_index"], index, index)
        start, end = binding["token_start"], binding["token_end"]
        _int(start, previous, len(ids))
        _int(end, start + 1, len(ids))
        _require(end - start <= 1024 and all(token != 248056 for token in ids[previous:start]) and all(token == 248056 for token in ids[start:end]), "image token correspondence mismatch")
        previous = end
        target = binding["target_size"]
        _require(type(target) is list and len(target) == 2, "invalid target size")
        for dimension in target:
            _int(dimension, 1, 1024)
        height, width = target
        _require(height * width <= 4096, "image target exceeds layout limit")
        shape = binding["pixel_shape"]
        _require(type(shape) is list and len(shape) == 3, "invalid pixel shape")
        for value, expected in zip(shape, (3, 14, 14 * height * width)):
            _int(value, expected, expected)
    _require(all(token != 248056 for token in ids[previous:]), "unbound image token")


def _capabilities(value: Any, native: bool = False, sampled: bool = False) -> None:
    _keys(value, {"text", "image", "greedy", "sampling", "token_events", "cancel_granularity", "max_active_sequences", "max_context", "max_output_tokens", "exact_append", "state_schema"})
    for name, expected in {"text": True, "image": False, "greedy": True, "sampling": sampled, "token_events": True, "exact_append": False}.items():
        _require(type(value[name]) is bool and value[name] is expected, "unsupported serial capability")
    cancellation = {"token_or_prefill_chunk"} if sampled else ({"end_prefill_or_token", "token_or_prefill_chunk"} if native else {"token", "token_or_prefill_chunk"})
    _enum(value["cancel_granularity"], cancellation)
    _int(value["max_active_sequences"], 1, 1)
    context_limit = (16384 if value["cancel_granularity"] == "token_or_prefill_chunk" else 4096) if native else MAX_PROMPT_TOKENS
    _int(value["max_context"], 1, context_limit)
    _int(value["max_output_tokens"], 0, MAX_OUTPUT_TOKENS)
    _enum(value["state_schema"], {"none"})


def _counts(value: Any, fields: set[str]) -> None:
    _keys(value, fields)
    for count in value.values():
        _int(count)


def validate_command_envelope(frame: dict) -> None:
    """Check routing and settlement fields, without accepting request content."""
    if type(frame) is dict and frame.get("protocol") == "apxinf-worker/2.6":
        if __package__:
            from .clef_contracts import validate_frame as validate_decision_frame
        else:
            from clef_contracts import validate_frame as validate_decision_frame
        validate_decision_frame(frame)
        _require(frame["kind"] in {"prepare_decision", "submit_decision", "shutdown"}, "unknown command kind")
        return
    _tree(frame)
    _require(type(frame) is dict, "expected command object")
    _require(frame.get("protocol") in (PROTOCOL, NATIVE_PROTOCOL, NATIVE_PREPARED_PROTOCOL, NATIVE_SAMPLED_PROTOCOL, NATIVE_THINKING_PROTOCOL, NATIVE_BOUNDED_THINKING_PROTOCOL, VISION_PROTOCOL), "unsupported protocol")
    _uuid(frame.get("worker_epoch"))
    _uuid(frame.get("command_id"))
    kind = frame.get("kind")
    shapes = {
        "prepare_input": (COMMAND, {"model_revision", "messages", "tools", "template_options"}),
        "submit": (REQUEST_COMMAND, _submit_fields(frame)),
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
    if type(frame) is dict and frame.get("protocol") == "apxinf-worker/2.6":
        if __package__:
            from .clef_contracts import validate_frame as validate_decision_frame
        else:
            from clef_contracts import validate_frame as validate_decision_frame
        validate_decision_frame(frame)
        return
    _tree(frame)
    _require(type(frame) is dict and BASE <= frame.keys(), "missing frame envelope")
    _require(frame["protocol"] in (PROTOCOL, NATIVE_PROTOCOL, NATIVE_PREPARED_PROTOCOL, NATIVE_SAMPLED_PROTOCOL, NATIVE_THINKING_PROTOCOL, NATIVE_BOUNDED_THINKING_PROTOCOL, VISION_PROTOCOL), "unsupported protocol")
    vision = frame["protocol"] == VISION_PROTOCOL
    bounded = frame["protocol"] == NATIVE_BOUNDED_THINKING_PROTOCOL
    thinking = frame["protocol"] == NATIVE_THINKING_PROTOCOL or bounded
    sampled = frame["protocol"] == NATIVE_SAMPLED_PROTOCOL or thinking
    prepared = frame["protocol"] == NATIVE_PREPARED_PROTOCOL
    native = frame["protocol"] == NATIVE_PROTOCOL or prepared or sampled
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
        manifest = frame["model_manifest"]
        _require(type(manifest) is dict, "model manifest must be an object")
        if not bounded:
            execution = manifest.get("execution")
            _require(manifest.get("adapter_revision") != "apxinf-native-minicpm5-bounded-thinking-v1"
                     and not (type(execution) is dict and (
                         execution.get("output_policy") == "minicpm5-initial-thinking-fixed512-v1"
                         or "initial_reasoning" in execution)),
                     "bounded thinking identity requires protocol 2.4")
        if vision:
            _vision_ready(frame)
        else:
            execution = manifest.get("execution")
            _require(manifest.get("adapter_revision") != "apxinf-mlx-minicpmv47-vision-v1"
                     and not (type(execution) is dict and execution.get("provider") == "mlx-vlm-minicpmv47"),
                     "vision identity requires protocol 2.5")
            _capabilities(frame["capabilities"], native, sampled)
        _keys(frame["limits"], {"max_frame_bytes", "max_depth", "max_commands"})
        _require(frame["limits"] == {"max_frame_bytes": MAX_FRAME_BYTES, "max_depth": MAX_DEPTH, "max_commands": MAX_COMMANDS}, "incompatible transport limits")
        for value in frame["limits"].values():
            _int(value)
        if native:
            _keys(frame["runtime"], {"kind", "apxinf", "mlx", "executable_sha256", "metal_library_sha256"})
            _require(frame["runtime"]["kind"] == "native_minicpm5" and frame["runtime"]["apxinf"] == "0.1.0" and frame["runtime"]["mlx"] in ("0.31.2", "0.32.3"), "unsupported native runtime")
            _digest(frame["runtime"]["executable_sha256"])
            _digest(frame["runtime"]["metal_library_sha256"])
            _require(type(frame["model_manifest"]) is dict, "model manifest must be an object")
            _require(frame["model_manifest"].get("runtime") == frame["runtime"], "native runtime mismatch")
            if prepared:
                execution = manifest.get("execution")
                _keys(execution, {"provider", "precision", "variant", "parser", "enable_thinking", "max_context",
                                  "max_output_tokens", "memory_limit_bytes", "preparation_policy",
                                  "allocator_cache_limit_bytes", "retained_memory_limit_bytes"})
                _require(manifest.get("adapter_revision") == "apxinf-native-minicpm5-prepared-v1"
                         and execution["provider"] == "metal_minicpm5" and execution["precision"] == "bf16"
                         and execution["variant"] == "bf16-compiled" and execution["parser"] == "minicpm5_xml"
                         and execution["enable_thinking"] is False
                         and execution["preparation_policy"] == "single-code-entry-v1"
                         and frame["runtime"]["mlx"] == "0.32.3", "unsupported prepared execution identity")
                _int(execution["memory_limit_bytes"], 1)
                limit = execution["memory_limit_bytes"]
                _int(execution["allocator_cache_limit_bytes"], 1, limit)
                _int(execution["retained_memory_limit_bytes"], 1, limit)
            else:
                _require(manifest.get("adapter_revision") != "apxinf-native-minicpm5-prepared-v1"
                         and "preparation_policy" not in manifest.get("execution", {}),
                         "prepared execution requires protocol 2.7")
            if sampled:
                manifest = frame["model_manifest"]
                execution = manifest.get("execution")
                _require(type(execution) is dict, "expected execution identity")
                if thinking:
                    required = {"provider", "precision", "variant", "parser", "enable_thinking", "max_context",
                                "max_output_tokens", "memory_limit_bytes", "sampling", "output_policy"}
                    if bounded:
                        required.add("initial_reasoning")
                    _keys(execution, required)
                    output_policy = "minicpm5-initial-thinking-fixed512-v1" if bounded else "minicpm5-initial-thinking-v1"
                    _require(execution["output_policy"] == output_policy,
                             "unsupported native thinking output policy")
                if bounded:
                    reasoning = execution["initial_reasoning"]
                    _keys(reasoning, {"budget_tokens", "close_token_id", "selection"})
                    _int(reasoning["budget_tokens"], 512, 512)
                    _int(reasoning["close_token_id"], 9, 9)
                    _enum(reasoning["selection"], {"cpu-constrained-close-v1"})
                adapter = ("apxinf-native-minicpm5-bounded-thinking-v1" if bounded else
                           "apxinf-native-minicpm5-thinking-v1" if thinking else "apxinf-native-minicpm5-sampled-v1")
                _require(manifest.get("adapter_revision") == adapter
                         and execution.get("provider") == "metal_minicpm5"
                         and execution.get("precision") == "bf16"
                         and execution.get("variant") == "bf16-compiled"
                         and execution.get("enable_thinking") is thinking
                         and execution.get("sampling") == {
                             "algorithm": "apxinf-cpu-categorical-v1", "temperature": "1",
                             "top_p": "0.95", "seed_policy": "request-id-sha256-v1",
                         }, "unsupported native sampling identity")
        elif not vision:
            _keys(frame["runtime"], {"python", "mlx", "mlx_lm"})
        for value in frame["runtime"].values():
            _str(value)
        _counts(frame["memory"], {"active_bytes", "peak_bytes", "cache_bytes"})
        _str(frame["model_path"])
        _require(frame["model_path"].startswith("/"), "model path must be absolute")
        _int(frame["vocab_size"], 1, 2_147_483_648)
        _tokens(frame["eos_token_ids"], 256)
        _require(all(token < frame["vocab_size"] for token in frame["eos_token_ids"]), "EOS outside vocabulary")
        if bounded:
            _require(9 < frame["vocab_size"] and 9 not in frame["eos_token_ids"],
                     "initial reasoning close token must be inside vocabulary and outside EOS")
        _require(type(frame["model_manifest"]) is dict, "model manifest must be an object")
        _require(frame["model_revision"] == canonical_identity_digest("apxinf-model-identity-v1", frame["model_manifest"]), "model revision mismatch")
        _require(frame["capability_revision"] == canonical_identity_digest("apxinf-capability-v1", frame["capabilities"]), "capability revision mismatch")
    elif kind == "prepare_input":
        _keys(frame, COMMAND | {"model_revision", "messages", "tools", "template_options"})
        _digest(frame["model_revision"])
        _messages(frame["messages"], frame["tools"], vision)
        _keys(frame["template_options"], {"enable_thinking"})
        _require(not vision or frame["template_options"]["enable_thinking"] is False, "vision preparation disables thinking")
        _require(type(frame["template_options"]["enable_thinking"]) is bool, "enable_thinking must be Boolean")
        _require(not thinking or frame["template_options"]["enable_thinking"] is True,
                 "protocol 2.4 requires thinking preparation" if bounded else "protocol 2.3 requires thinking preparation")
    elif kind == "prepared_input":
        _keys(frame, COMMAND | {"model_revision", "token_ids", "effective_input_tokens", "prompt_digest"} | ({"replay"} if vision else set()))
        _digest(frame["model_revision"])
        _tokens(frame["token_ids"], MAX_PROMPT_TOKENS, 1)
        _int(frame["effective_input_tokens"], 1, MAX_PROMPT_TOKENS)
        _require(frame["effective_input_tokens"] == len(frame["token_ids"]), "prepared input count mismatch")
        _require(frame["prompt_digest"] == token_prefix_digest(frame["token_ids"]), "prepared input digest mismatch")
    elif kind == "submit":
        _keys(frame, REQUEST_COMMAND | _submit_fields(frame))
        if sampled or vision:
            _sampling(frame["sampling"])
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
        _int(frame["consumed_tokens"], 0, 0 if vision else MAX_PROMPT_TOKENS)
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
        if vision:
            _require(frame["state_result"]["consumed_position"] is None, "vision consumed position must be null")
        else:
            _int(frame["state_result"]["consumed_position"], 0, frame["state_result"]["history_count"])
        _str(frame["text_delta"])
        _counts(frame["metrics"], {"elapsed_ns", "ttft_ns", "peak_memory_bytes"} | ({"source_media_bytes", "peak_host_array_bytes"} if vision else set()))
        _require(frame["metrics"]["ttft_ns"] <= frame["metrics"]["elapsed_ns"], "TTFT exceeds elapsed time")
    elif kind == "resources_released":
        _keys(frame, EVENT | {"released_lease_ids", "retained_lease_ids"} | ({"memory"} if prepared else set()))
        if prepared:
            _counts(frame["memory"], {"active_bytes", "cache_bytes", "peak_bytes"})
            _require(frame["memory"]["peak_bytes"] >= frame["memory"]["active_bytes"], "active memory exceeds recorded peak")
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
    if vision and kind in {"prepared_input", "submit"}:
        _vision_replay(frame)
        if kind == "submit":
            _int(frame["max_tokens"], 0, 8192)
            _require(frame["sampling"]["mode"] != "random" or frame["sampling"]["temperature"] >= 0.01,
                     "vision temperature is below the supported minimum")
    if vision and kind == "terminal":
        _int(frame["metrics"]["source_media_bytes"], 0, 262144)
        _int(frame["metrics"]["peak_host_array_bytes"], 0, 67108864)


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
        self.protocol = submit["protocol"]
        self.input_tokens = len(submit["token_ids"])
        self.max_tokens = submit["max_tokens"]
        self.source_media_bytes = (
            sum(source["byte_count"] for source in submit["replay"]["media"])
            if self.protocol == VISION_PROTOCOL else None
        )
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
        _require(event["protocol"] == self.protocol and tuple(event[key] for key in ("worker_epoch", "request_id", "attempt")) == self.key, "attempt identity mismatch")
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
            if self.source_media_bytes is not None:
                _require(event["metrics"]["source_media_bytes"] == self.source_media_bytes, "terminal source media byte count mismatch")
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
