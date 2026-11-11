from __future__ import annotations

from dataclasses import dataclass
import math
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

if __package__:
    from . import contracts as common
else:
    import contracts as common

PROTOCOL = "apxinf-worker/2.6"
RUNTIME = {"python": "3.12.10", "mlx": "0.32.3", "mlx_lm": "0.32.0",
           "transformers": "5.19.0", "numpy": "2.5.4", "pillow": "12.3.0"}
CHECKPOINT = "mlx-community/clef-flash-4bit"
CHECKPOINT_REVISION = "b5adbe2c3a95200d4bf04a0727c8ec457157d5e1"
ADAPTER_REVISION = "apxinf-python-clef-text-v1"
LOADER_SHA256 = "852223c944819a32fad5cf798d9d1dff30419820eaf5ad1f10cb9698afec97d5"
SOURCE_FILES = ("clef_worker.py", "clef_runtime.py", "clef_identity.py", "clef_contracts.py", "contracts.py", "clef_artifacts.json")
ID = re.compile(r"[A-Za-z0-9_.-]{1,64}\Z")
TYPE_CODES = {"noul": 0, "choice": 1, "score": 2}
ContractError = common.ContractError


def approved_artifacts() -> list[dict]:
    return json.loads(Path(__file__).with_name("clef_artifacts.json").read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    common._require(condition, message)


def text(value: Any, minimum: int, maximum: int) -> str:
    common._str(value)
    require(minimum <= len(value.encode("utf-8")) <= maximum, "text byte limit exceeded")
    return value


def identifier(value: Any) -> str:
    common._str(value)
    require(ID.fullmatch(value) is not None, "invalid decision identifier")
    return value


@dataclass(frozen=True)
class Question:
    question_id: str
    kind: str
    instructions: str
    criteria: tuple[tuple[str, str], ...] | None

    @property
    def option_ids(self) -> tuple[str, ...]:
        return ("true", "false") if self.kind == "noul" else tuple(key for key, _ in self.criteria)

    def document(self) -> dict:
        result = {"type": self.kind, "instructions": self.instructions}
        if self.criteria is not None:
            result["criteria"] = ([value for _, value in self.criteria] if self.kind == "score"
                                  else dict(self.criteria))
        return result


@dataclass(frozen=True)
class Record:
    state: str
    questions: tuple[Question, ...]

    def document(self) -> dict:
        return {"state": self.state, "questions": {q.question_id: q.document() for q in self.questions}}


@dataclass(frozen=True)
class DecisionRequest:
    model: str
    record: Record


def validate_record(value: Any) -> Record:
    common._keys(value, {"state", "questions"})
    state = text(value["state"], 0, 65536)
    questions = value["questions"]
    require(type(questions) is dict and 1 <= len(questions) <= 16, "invalid question count")
    parsed = []
    for qid in sorted(questions):
        identifier(qid)
        question = questions[qid]
        common._keys(question, {"type", "instructions"}, {"criteria"})
        common._enum(question["type"], set(TYPE_CODES))
        kind = question["type"]
        instructions = text(question["instructions"], 1, 4096)
        criteria = question.get("criteria")
        if kind == "noul" and "criteria" not in question:
            parsed.append(Question(qid, kind, instructions, None))
            continue
        if kind == "score":
            require(type(criteria) is list and 2 <= len(criteria) <= 16, "invalid score criteria")
            options = tuple((str(index), text(description, 1, 2048)) for index, description in enumerate(criteria))
        else:
            require(type(criteria) is dict, "criteria must be an object")
            if kind == "noul":
                common._keys(criteria, {"true", "false"})
                keys = ("true", "false")
            else:
                require(2 <= len(criteria) <= 16, "invalid choice criteria")
                keys = sorted(criteria)
            options = tuple((identifier(key), text(criteria[key], 1, 2048)) for key in keys)
        parsed.append(Question(qid, kind, instructions, options))
    return Record(state, tuple(parsed))


def parse_decision_request(payload: bytes | str | dict) -> DecisionRequest:
    if type(payload) in (bytes, str):
        raw = payload.encode("utf-8") if type(payload) is str else payload
        require(len(raw) <= 131072, "request byte limit exceeded")
        value = common.parse_document(raw)
    else:
        value = payload
        common._tree(value)
        require(len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) <= 131072,
                "request byte limit exceeded")
    common._keys(value, {"model", "state", "questions"})
    return DecisionRequest(text(value["model"], 1, 128), validate_record({k: value[k] for k in ("state", "questions")}))


@dataclass(frozen=True)
class EncodedQuestion:
    question_id: str
    question_type: int
    question_span: tuple[int, int]
    option_spans: tuple[tuple[int, int], ...]
    option_ids: tuple[str, ...]

    def document(self) -> dict:
        return {"question_id": self.question_id, "question_type": self.question_type,
                "question_span": list(self.question_span), "option_spans": [list(s) for s in self.option_spans],
                "option_ids": list(self.option_ids)}


@dataclass(frozen=True)
class Encoding:
    input_ids: tuple[int, ...]
    questions: tuple[EncodedQuestion, ...]

    def document(self) -> dict:
        return {"input_ids": list(self.input_ids), "questions": [q.document() for q in self.questions]}


def validate_encoding(value: Any, record: Record | dict, *, vocab_size: int = 2147483648,
                      max_context: int = 4096) -> Encoding:
    if type(record) is dict:
        record = validate_record(record)
    common._keys(value, {"input_ids", "questions"})
    common._int(vocab_size, 1, 2147483648)
    common._int(max_context, 1, 4096)
    ids = value["input_ids"]
    common._tokens(ids, max_context, 1)
    require(all(token < vocab_size for token in ids), "token outside loaded vocabulary")
    metadata = value["questions"]
    require(type(metadata) is list and len(metadata) == len(record.questions), "question metadata count mismatch")
    parsed = []
    previous = 0
    for entry, question in zip(metadata, record.questions):
        common._keys(entry, {"question_id", "question_type", "question_span", "option_spans", "option_ids"})
        common._int(entry["question_type"], 0, 2)
        require(entry["question_id"] == question.question_id and entry["question_type"] == TYPE_CODES[question.kind],
                "question metadata mismatch")
        require(type(entry["option_ids"]) is list and entry["option_ids"] == list(question.option_ids),
                "option metadata mismatch")
        spans = entry["option_spans"]
        require(type(spans) is list and len(spans) == len(question.option_ids), "option span count mismatch")
        checked = []
        for span in [entry["question_span"], *spans]:
            require(type(span) is list and len(span) == 2, "invalid span")
            common._int(span[0], 0, len(ids))
            common._int(span[1], 0, len(ids))
            require(previous <= span[0] < span[1], "overlapping or empty span")
            previous = span[1]
            checked.append(tuple(span))
        parsed.append(EncodedQuestion(question.question_id, entry["question_type"], checked[0],
                                     tuple(checked[1:]), question.option_ids))
    return Encoding(tuple(ids), tuple(parsed))


def validate_probabilities(value: Any, encoding: Encoding | dict) -> tuple[tuple[float, ...], ...]:
    questions = encoding.questions if isinstance(encoding, Encoding) else encoding["questions"]
    require(type(value) is list and len(value) == len(questions), "probability question count mismatch")
    vectors = []
    for entry, question in zip(value, questions):
        expected = question.document() if isinstance(question, EncodedQuestion) else question
        common._keys(entry, {"question_id", "option_ids", "values"})
        require(entry["question_id"] == expected["question_id"] and entry["option_ids"] == expected["option_ids"],
                "probability metadata mismatch")
        values = entry["values"]
        require(type(values) is list and len(values) == len(expected["option_ids"]), "probability cardinality mismatch")
        require(all(type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1 for p in values),
                "invalid probability")
        require(abs(sum(values) - 1) <= 0.00001, "probabilities do not sum to one")
        vectors.append(tuple(float(p) for p in values))
    return tuple(vectors)


def project_response(model: str, record: Record, encoding: Encoding, probabilities: Any) -> dict:
    vectors = validate_probabilities(probabilities, encoding)
    answers = {}
    for question, values in zip(record.questions, vectors):
        if question.kind == "noul":
            answer = {"type": "noul", "noul": values[0]}
        elif question.kind == "choice":
            selected = max(range(len(values)), key=values.__getitem__)
            answer = {"type": "choice", "choice": question.option_ids[selected], "confidence": values[selected],
                      "probabilities": dict(zip(question.option_ids, values))}
        else:
            answer = {"type": "score", "score": sum(i * p for i, p in enumerate(values)),
                      "confidence": max(values), "legend": dict(question.criteria),
                      "probabilities": dict(zip(question.option_ids, values))}
        answers[question.question_id] = answer
    return {"model": model, "answers": answers, "usage": {"input_tokens": len(encoding.input_ids), "output_tokens": 0}}


def preparation_identity(value: dict) -> dict:
    return {key: value[key] for key in ("worker_epoch", "request_id", "attempt", "model_revision",
            "capability_revision", "preparation_id", "record_digest", "encoding_digest")}


def make_preparation(command: dict, record: Record, encoding: Encoding) -> dict:
    result = {key: command[key] for key in ("worker_epoch", "request_id", "attempt", "model_revision", "capability_revision")}
    result.update(preparation_id=command["command_id"], record=record.document(), encoding=encoding.document(),
                  effective_input_tokens=len(encoding.input_ids),
                  record_digest=common.canonical_identity_digest("apxinf-clef-record-v1", record.document()),
                  encoding_digest=common.canonical_identity_digest("apxinf-clef-encoding-v1", encoding.document()))
    result["preparation_digest"] = common.canonical_identity_digest("apxinf-clef-preparation-v1", preparation_identity(result))
    return result


def validate_preparation(value: Any) -> tuple[Record, Encoding]:
    common._keys(value, {"worker_epoch", "request_id", "attempt", "model_revision", "capability_revision",
                        "preparation_id", "record", "encoding", "effective_input_tokens", "record_digest",
                        "encoding_digest", "preparation_digest"})
    for key in ("worker_epoch", "request_id", "preparation_id"):
        common._uuid(value[key])
    common._int(value["attempt"], 1)
    for key in ("model_revision", "capability_revision", "record_digest", "encoding_digest", "preparation_digest"):
        common._digest(value[key])
    record = validate_record(value["record"])
    require(record.document() == value["record"], "record is not canonical")
    encoding = validate_encoding(value["encoding"], record)
    common._int(value["effective_input_tokens"], 1, 4096)
    require(value["effective_input_tokens"] == len(encoding.input_ids), "effective count mismatch")
    for domain, key, document in (("apxinf-clef-record-v1", "record_digest", value["record"]),
                                   ("apxinf-clef-encoding-v1", "encoding_digest", value["encoding"]),
                                   ("apxinf-clef-preparation-v1", "preparation_digest", preparation_identity(value))):
        require(value[key] == common.canonical_identity_digest(domain, document), "preparation digest mismatch")
    return record, encoding


def _file_record(value: Any, *, sized: bool = False, absolute: bool = True) -> None:
    common._keys(value, {"path", "sha256"} | ({"size_bytes"} if sized else set()))
    path = text(value["path"], 1, 4096)
    require(PurePosixPath(path).is_absolute() == absolute, "invalid identity path")
    common._digest(value["sha256"])
    if sized:
        common._int(value["size_bytes"])


def capabilities(vocab_size: int, max_context: int) -> dict:
    return {"workload": "decision", "text": True, "noul": True, "choice": True, "score": True,
            "image": False, "streaming": False, "token_events": False, "kv_reuse": False,
            "state_schema": "none", "max_active_requests": 1, "vocab_size": vocab_size,
            "max_context": max_context, "max_output_tokens": 0, "cancel_granularity": "process_exit"}


def execution(max_context: int, memory_limit_bytes: int) -> dict:
    return {"provider": "python-clef", "precision": "backbone-4bit-head-bf16", "backend": "lm",
            "max_context": max_context, "max_output_tokens": 0, "idle_cache_limit_bytes": 0,
            "memory_limit_bytes": memory_limit_bytes, "truncate": False, "kv_reuse": False,
            "worker_lifetime": "one-attempt"}


def _ready(frame: dict) -> None:
    common._keys(frame, common.BASE | {"model_revision", "capability_revision", "model_manifest", "capabilities",
                                      "limits", "model_path", "runtime", "memory"})
    caps = frame["capabilities"]
    require(type(caps) is dict, "invalid capabilities")
    common._int(caps.get("vocab_size"), 1, 2147483648)
    common._int(caps.get("max_context"), 1, 4096)
    for key in ("text", "noul", "choice", "score", "image", "streaming", "token_events", "kv_reuse"):
        require(type(caps.get(key)) is bool, "capability must be Boolean")
    common._int(caps.get("max_active_requests"), 1, 1)
    common._int(caps.get("max_output_tokens"), 0, 0)
    require(caps == capabilities(caps["vocab_size"], caps["max_context"]), "incompatible decision capabilities")
    require(frame["runtime"] == RUNTIME, "incompatible decision runtime")
    require(frame["limits"] == {"max_frame_bytes": common.MAX_FRAME_BYTES, "max_depth": 16, "max_commands": 10000},
            "incompatible transport limits")
    for value in frame["limits"].values():
        common._int(value)
    text(frame["model_path"], 1, 4096)
    require(PurePosixPath(frame["model_path"]).is_absolute(), "model path must be absolute")
    common._counts(frame["memory"], {"active_bytes", "peak_bytes", "cache_bytes"})
    manifest = frame["model_manifest"]
    common._keys(manifest, {"checkpoint", "checkpoint_revision", "adapter_revision", "platform", "artifacts", "loader",
                            "runtime", "sources", "python", "runtime_files", "execution"})
    require((manifest["checkpoint"], manifest["checkpoint_revision"], manifest["adapter_revision"], manifest["platform"])
            == (CHECKPOINT, CHECKPOINT_REVISION, ADAPTER_REVISION, "macos-metal"), "incompatible model identity")
    require(manifest["runtime"] == RUNTIME, "manifest runtime mismatch")
    ex = manifest["execution"]
    require(type(ex) is dict, "invalid execution identity")
    common._int(ex.get("memory_limit_bytes"), 1)
    for key in ("max_context", "max_output_tokens", "idle_cache_limit_bytes"):
        common._int(ex.get(key))
    require(type(ex.get("truncate")) is bool and type(ex.get("kv_reuse")) is bool, "execution flag must be Boolean")
    require(ex == execution(caps["max_context"], ex["memory_limit_bytes"]), "incompatible execution identity")
    _file_record(manifest["loader"])
    require(manifest["loader"]["sha256"] == LOADER_SHA256, "external loader identity mismatch")
    _file_record(manifest["python"])
    common._keys(manifest["sources"], set(SOURCE_FILES))
    for name, source in manifest["sources"].items():
        _file_record(source)
        require(PurePosixPath(source["path"]).name == name, "source filename mismatch")
    files = manifest["runtime_files"]
    common._keys(files, {"modules", "libraries"})
    common._keys(files["modules"], {"mlx_core", "mlx_lm", "transformers", "numpy", "pillow", "tokenizers"})
    for source in files["modules"].values():
        _file_record(source)
    require(type(files["libraries"]) is list and len(files["libraries"]) == 1, "one MLX library required")
    _file_record(files["libraries"][0])
    artifacts = manifest["artifacts"]
    require(type(artifacts) is list and artifacts, "missing model artifacts")
    names = []
    for artifact in artifacts:
        _file_record(artifact, sized=True, absolute=False)
        path = PurePosixPath(artifact["path"])
        require(path.name == artifact["path"] and path.suffix in {".json", ".safetensors", ".jinja"}, "invalid artifact name")
        names.append(artifact["path"])
    require(names == sorted(set(names)), "artifacts must be sorted and unique")
    require(artifacts == approved_artifacts(), "artifacts differ from the approved checkpoint")
    require(frame["model_revision"] == common.canonical_identity_digest("apxinf-model-identity-v1", manifest),
            "model revision mismatch")
    require(frame["capability_revision"] == common.canonical_identity_digest("apxinf-capability-v1", caps),
            "capability revision mismatch")


def validate_frame(frame: Any) -> None:
    common._tree(frame)
    require(type(frame) is dict and common.BASE <= frame.keys(), "missing frame envelope")
    require(frame["protocol"] == PROTOCOL, "unsupported decision protocol")
    common._uuid(frame["worker_epoch"])
    kind = frame["kind"]
    common._str(kind)
    if kind == "ready":
        _ready(frame)
        return
    common._uuid(frame.get("command_id"))
    if kind in {"shutdown", "drained"}:
        common._keys(frame, common.COMMAND | ({"reason"} if kind == "drained" else set()))
        if kind == "drained":
            common._enum(frame["reason"], {"shutdown", "attempt_complete"})
        return
    common._uuid(frame.get("request_id"))
    common._int(frame.get("attempt"), 1)
    shapes = {"prepare_decision": {"model_revision", "capability_revision", "record"},
              "prepared_decision": {"model_revision", "capability_revision", "preparation"},
              "submit_decision": {"model_revision", "capability_revision", "preparation", "capacity_lease_ids"},
              "accepted": {"event_seq", "model_revision", "capability_revision"},
              "decision_result": {"event_seq", "preparation_digest", "probabilities", "usage", "metrics"},
              "failure": {"event_seq", "code", "message"}}
    require(type(kind) is str and kind in shapes, "unknown decision frame")
    common._keys(frame, common.REQUEST_COMMAND | shapes[kind])
    if "model_revision" in frame:
        common._digest(frame["model_revision"])
        common._digest(frame["capability_revision"])
    if kind == "prepare_decision":
        record = validate_record(frame["record"])
        require(record.document() == frame["record"], "noncanonical record")
    if kind in {"prepared_decision", "submit_decision"}:
        validate_preparation(frame["preparation"])
        for key in ("worker_epoch", "request_id", "attempt", "model_revision", "capability_revision"):
            require(frame[key] == frame["preparation"][key], "preparation envelope mismatch")
        if kind == "prepared_decision":
            require(frame["command_id"] == frame["preparation"]["preparation_id"], "preparation command mismatch")
        else:
            common._leases(frame["capacity_lease_ids"])
            require(frame["command_id"] != frame["preparation"]["preparation_id"], "submission needs a new command")
    if kind in {"accepted", "decision_result", "failure"}:
        common._int(frame["event_seq"], 0, 1)
    if kind == "accepted":
        require(frame["event_seq"] == 0, "acceptance sequence mismatch")
    if kind == "decision_result":
        require(frame["event_seq"] == 1, "result sequence mismatch")
        common._digest(frame["preparation_digest"])
        common._counts(frame["usage"], {"input_tokens", "output_tokens"})
        require(1 <= frame["usage"]["input_tokens"] <= 4096 and frame["usage"]["output_tokens"] == 0, "invalid decision usage")
        common._counts(frame["metrics"], {"elapsed_ms", "peak_device_bytes"})
        probabilities = frame["probabilities"]
        require(type(probabilities) is list and 1 <= len(probabilities) <= 16, "invalid probability count")
        seen = set()
        for entry in probabilities:
            common._keys(entry, {"question_id", "option_ids", "values"})
            qid = identifier(entry["question_id"])
            require(qid not in seen, "duplicate probability question")
            seen.add(qid)
            ids = entry["option_ids"]
            require(type(ids) is list and 2 <= len(ids) <= 16, "invalid probability options")
            require(len(set(identifier(i) for i in ids)) == len(ids), "duplicate probability options")
        validate_probabilities(probabilities, {"questions": probabilities})
    if kind == "failure":
        common._enum(frame["code"], {"invalid_request", "context_limit", "model_unavailable", "worker_lost", "protocol_fault", "internal_error"})
        text(frame["message"], 1, 4096)


class DecisionTracker:
    def __init__(self, worker_epoch: str | None = None, model_revision: str | None = None,
                 capability_revision: str | None = None):
        self.worker_epoch = worker_epoch
        self.model_revision = model_revision
        self.capability_revision = capability_revision
        self.state = "new"
        self.preparation = None
        self.submission_id = None
        self.terminal_command = None
        self.result = None
        self.drained = False
        self.cancelled = False
        self.public_success = False
        self.resources_released = False

    def observe(self, frame: dict) -> None:
        validate_frame(frame)
        if self.worker_epoch is not None:
            require(frame["worker_epoch"] == self.worker_epoch, "stale worker epoch")
        for key in ("model_revision", "capability_revision"):
            expected = getattr(self, key)
            if key in frame and expected is not None:
                require(frame[key] == expected, "stale ready identity")
        kind = frame["kind"]
        if self.preparation is not None and kind not in {"drained", "shutdown"}:
            for key in ("request_id", "attempt", "worker_epoch"):
                require(frame[key] == self.preparation[key], "attempt identity mismatch")
        if kind == "prepared_decision":
            require(self.state == "new", "duplicate preparation")
            self.preparation = frame["preparation"]
            self.worker_epoch = frame["worker_epoch"]
            self.model_revision = frame["model_revision"]
            self.capability_revision = frame["capability_revision"]
            self.state = "prepared"
        elif kind == "submit_decision":
            require(self.state == "prepared" and frame["preparation"] == self.preparation, "submission preparation mismatch")
            self.submission_id = frame["command_id"]
            self.state = "submitted"
        elif kind == "accepted":
            require(self.state == "submitted" and frame["command_id"] == self.submission_id, "unexpected acceptance")
            self.state = "accepted"
        elif kind == "decision_result":
            require(self.state == "accepted" and frame["command_id"] == self.submission_id, "unexpected result")
            require(frame["preparation_digest"] == self.preparation["preparation_digest"], "result preparation mismatch")
            require(frame["usage"]["input_tokens"] == self.preparation["effective_input_tokens"], "result usage mismatch")
            validate_probabilities(frame["probabilities"], self.preparation["encoding"])
            self.result = frame
            self.terminal_command = frame["command_id"]
            self.state = "terminal_pending_reap"
        elif kind == "failure":
            require(self.state in {"new", "prepared", "submitted", "accepted"}, "unexpected failure")
            require(frame["event_seq"] == (1 if self.state == "accepted" else 0), "failure sequence mismatch")
            if self.submission_id is not None:
                require(frame["command_id"] == self.submission_id, "failure command mismatch")
            elif self.preparation is not None:
                require(frame["command_id"] == self.preparation["preparation_id"], "failure preparation mismatch")
            self.terminal_command = frame["command_id"]
            self.state = "failed_pending_reap"
        elif kind == "drained":
            require(self.state in {"terminal_pending_reap", "failed_pending_reap"} and not self.drained, "unexpected drain")
            require(frame["command_id"] == self.terminal_command and frame["reason"] == "attempt_complete", "drain command mismatch")
            self.drained = True
        else:
            raise ContractError("frame cannot enter an attempt tracker")

    def reaped(self, exit_code: int, eof: bool, drained: bool | None = None) -> bool:
        self.resources_released = True
        checked_drain = self.drained if drained is None else drained and self.drained
        success = (self.state == "terminal_pending_reap" and checked_drain and eof and exit_code == 0 and not self.cancelled)
        ordinary_failure = self.state == "failed_pending_reap" and checked_drain and eof and not self.cancelled
        self.public_success = bool(success)
        self.state = "succeeded" if success else "failed"
        return bool(success or ordinary_failure)
