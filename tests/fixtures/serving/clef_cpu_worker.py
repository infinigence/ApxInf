"""Original decision byte-pipe fixture selected only by private Rust tests.

This fixture starts after identity validation. It proves no checkpoint identity.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def digest(domain, document):
    body = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(domain.encode() + b"\0" + body.encode()).hexdigest()


def emit(value):
    print(json.dumps(value, separators=(",", ":")), flush=True)


def trace(name):
    with (root / "trace.jsonl").open("a") as sink:
        sink.write(json.dumps({"event": name, "pid": os.getpid()}) + "\n")


def hold(name):
    trace(name)
    until = time.monotonic() + 15
    while not (root / "release").exists():
        if time.monotonic() >= until:
            raise RuntimeError("The fixture release marker did not arrive.")
        time.sleep(0.005)


parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
parser.add_argument("--epoch", required=True)
parser.add_argument("--mode", required=True)
args = parser.parse_args()
root = Path(args.root)
trace("started")

for line in sys.stdin:
    command = json.loads(line)
    base = {key: command[key] for key in ("protocol", "worker_epoch", "command_id")}
    if command["kind"] == "shutdown":
        emit({**base, "kind": "drained", "reason": "shutdown"})
        break
    attempt = {**base, **{key: command[key] for key in ("request_id", "attempt")}}
    if command["kind"] == "prepare_decision":
        if args.mode == "prepare_hold":
            hold("preparing")
        if args.mode == "context_limit":
            emit({**attempt, "kind": "failure", "event_seq": 0,
                  "code": "context_limit", "message": "The prepared input exceeds context."})
            emit({**base, "kind": "drained", "reason": "attempt_complete"})
            break
        record = command["record"]
        metadata = []
        cursor = 0
        for key, question in sorted(record["questions"].items()):
            kind = question["type"]
            ids = (["true", "false"] if kind == "noul" else
                   sorted(question["criteria"]) if kind == "choice" else
                   [str(index) for index in range(len(question["criteria"]))])
            entry = {"question_id": key, "question_type": ["noul", "choice", "score"].index(kind),
                     "question_span": [cursor, cursor + 1], "option_ids": ids,
                     "option_spans": [[cursor + index + 1, cursor + index + 2] for index in range(len(ids))]}
            metadata.append(entry)
            cursor += 1 + len(ids)
        encoding = {"input_ids": list(range(cursor)), "questions": metadata}
        identity = {key: command[key] for key in ("worker_epoch", "request_id", "attempt",
                                                "model_revision", "capability_revision")}
        identity.update(preparation_id=command["command_id"],
                        record_digest=digest("apxinf-clef-record-v1", record),
                        encoding_digest=digest("apxinf-clef-encoding-v1", encoding))
        preparation = {**identity, "record": record, "encoding": encoding,
                       "effective_input_tokens": cursor,
                       "preparation_digest": digest("apxinf-clef-preparation-v1", identity)}
        emit({**attempt, "kind": "prepared_decision", "model_revision": command["model_revision"],
              "capability_revision": command["capability_revision"], "preparation": preparation})
        trace("prepared")
    elif command["kind"] == "submit_decision":
        trace("submitted")
        emit({**attempt, "kind": "accepted", "event_seq": 0,
              "model_revision": command["model_revision"], "capability_revision": command["capability_revision"]})
        if args.mode == "accepted_hold":
            hold("accepted")
        preparation = command["preparation"]
        vectors = []
        for entry in preparation["encoding"]["questions"]:
            count = len(entry["option_ids"])
            vectors.append({"question_id": entry["question_id"], "option_ids": entry["option_ids"],
                            "values": [1.0 / count] * count})
        result = {**attempt, "kind": "decision_result", "event_seq": 1,
                  "preparation_digest": preparation["preparation_digest"], "probabilities": vectors,
                  "usage": {"input_tokens": preparation["effective_input_tokens"], "output_tokens": 0},
                  "metrics": {"elapsed_ms": 1, "peak_device_bytes": 128}}
        emit(result)
        if args.mode == "result_hold":
            hold("result_pending")
        emit({**base, "kind": "drained", "reason": "attempt_complete"})
        if args.mode == "extra_output":
            emit(result)
        if args.mode == "nonzero_exit":
            sys.exit(7)
        break
trace("exiting")
