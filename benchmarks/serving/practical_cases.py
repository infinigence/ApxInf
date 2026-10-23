"""Original daily file tasks and strict completion checks."""

import hashlib
import json
import os
from pathlib import Path
import stat


CASES = {
    "expenses": {
        "revision": "expenses-v1", "output": "expenses.json",
        "inputs": {
            "transactions.csv": (
                "id,date,description,category,currency,amount,status\n"
                "t01,2026-10-19,Metro pass,transport,CNY,120.00,posted\n"
                "t02,2026-10-19,Team lunch,meals,CNY,86.35,posted\n"
                "t03,2026-10-20,Cloud storage,software,CNY,29.90,posted\n"
                "t04,2026-10-20,Taxi refund,transport,CNY,-18.50,posted\n"
                "t05,2026-10-21,Coffee,meals,CNY,12.80,pending\n"
                "t06,2026-10-21,Train booking,transport,USD,44.00,posted\n"
                "t07,2026-10-21,Notebook,office,CNY,16.00,posted\n"
                "t08,2026-10-22,Dinner,meals,CNY,43.65,posted\n"
                "t09,2026-10-22,Editor subscription,software,CNY,55.10,posted\n"
            ),
            "expense-rules.txt": (
                "Summarize posted CNY transactions in transport, meals, and software only.\n"
                "Include negative refunds. Exclude pending rows, other currencies, and other categories.\n"
                "Convert amounts to integer cents exactly. Do not round. Preserve source order for included IDs.\n"
                "Write exactly currency, included_rows, category_totals_cents, and total_cents.\n"
                "currency is CNY. included_rows is a list of transaction IDs.\n"
                "category_totals_cents maps transport, meals, and software to integer totals.\n"
                "total_cents is the integer sum across those categories.\n"
            ),
        },
        "expected": {"currency": "CNY", "included_rows": ["t01", "t02", "t03", "t04", "t08", "t09"],
                     "category_totals_cents": {"transport": 10150, "meals": 13000, "software": 8500},
                     "total_cents": 31650},
    },
    "meeting-actions": {
        "revision": "meeting-actions-v1", "output": "meeting-actions.json",
        "inputs": {
            "meeting-notes.txt": (
                "Operations meeting, 2026-10-22.\n"
                "Action: Mira will send the revised onboarding checklist by 2026-10-24.\n"
                "Discussion: Jon suggested a dashboard redesign. No action was agreed.\n"
                "Decision: Keep the weekly release on Thursday.\n"
                "Action: Jon will test the backup restore. No due date was assigned.\n"
                "Action: Update the contact directory by 2026-10-27. No owner was assigned.\n"
                "Discussion: The group may review office chairs next month. This is not an action.\n"
                "Action: Lina will book the accessibility review by 2026-10-29.\n"
            ),
            "extraction-rules.txt": (
                "Extract explicit actions and confirmed decisions only. Preserve source order.\n"
                "Write exactly meeting_date, actions, and decisions.\n"
                "meeting_date is the ISO date from the notes.\n"
                "actions is a list of objects with exactly task, owner, and due_date.\n"
                "Use the task wording after 'will', without the deadline or final period.\n"
                "For an action without 'will', use its imperative wording without the deadline or final period.\n"
                "owner is the named person or JSON null when absent.\n"
                "due_date is an ISO date or JSON null when absent. Do not infer missing values.\n"
                "decisions is a list of exact decision sentences after 'Decision: ', including the final period.\n"
                "Exclude discussion and suggestions.\n"
            ),
        },
        "expected": {"meeting_date": "2026-10-22", "actions": [
            {"task": "send the revised onboarding checklist", "owner": "Mira", "due_date": "2026-10-24"},
            {"task": "test the backup restore", "owner": "Jon", "due_date": None},
            {"task": "Update the contact directory", "owner": None, "due_date": "2026-10-27"},
            {"task": "book the accessibility review", "owner": "Lina", "due_date": "2026-10-29"},
        ], "decisions": ["Keep the weekly release on Thursday."]},
    },
}


def prepare(name, directory):
    case = CASES[name]
    inputs = {}
    for relative, text in case["inputs"].items():
        original = text.encode("utf-8")
        (directory / relative).write_bytes(original)
        inputs[relative] = {"bytes": original, "sha256": hashlib.sha256(original).hexdigest()}
    target = directory / case["output"]
    target.write_bytes(b"")
    return {"prompt": "Use Read to inspect " + ", ".join(inputs) + f", and the empty {case['output']}. "
            "Follow the rules in the input files. Use Edit to save the complete JSON object. "
            f"Set file_path to {case['output']}, old_string to an empty string, and new_string to the complete JSON. "
            "Keep all input files unchanged. Do not run code. Reply after saving the file.",
            "tools": ["Read", "Edit"], "allowed": ["Read", "Edit"], "input_files": inputs,
            "output_path": case["output"], "expected": case["expected"],
            "output_limit_bytes": 8192, "task_revision": case["revision"]}


def read_regular(path, limit):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("The artifact is not a bounded regular file.")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            data = source.read(limit + 1)
        if len(data) > limit:
            raise ValueError("The artifact exceeds its byte limit.")
        return data
    finally:
        os.close(descriptor)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("The JSON object contains a duplicate field.")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"The JSON contains a nonfinite number: {value}.")


def exact_value(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(exact_value(actual[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(exact_value(left, right) for left, right in zip(actual, expected))
    return actual == expected


def artifact_check(spec, directory):
    try:
        data = read_regular(directory / spec["output_path"], spec["output_limit_bytes"])
        actual = json.loads(data.decode("utf-8"), object_pairs_hook=unique_object, parse_constant=reject_constant)
        if not exact_value(actual, spec["expected"]):
            return {"valid": False, "reason": "The structured result differs from the expected result."}
        return {"valid": True, "reason": None}
    except (OSError, ValueError, UnicodeError, RecursionError) as error:
        return {"valid": False, "reason": str(error)}


def tool_cycle(events, spec, directory):
    expected_paths = {(directory / name).resolve(): name for name in spec["input_files"]}
    output = (directory / spec["output_path"]).resolve()
    seen, pending, read = set(), {}, set()
    edited = initialized = finalized = False
    for record in events:
        event = record["event"]
        kind = event.get("type")
        if kind == "result":
            if finalized or not initialized or pending:
                return False
            finalized = True
        if kind == "system" and event.get("subtype") == "init":
            advertised = event.get("tools")
            if initialized or not isinstance(advertised, list) or not all(isinstance(item, str) for item in advertised):
                return False
            if not {"Read", "Edit"}.issubset(advertised):
                return False
            initialized = True
        if kind not in ("assistant", "user"):
            continue
        if not initialized or finalized or event.get("message", {}).get("role", kind) != kind:
            return False
        content = event.get("message", {}).get("content", [])
        for block in content if isinstance(content, list) else []:
            if not isinstance(block, dict):
                return False
            block_type = block.get("type")
            if block_type == "tool_use":
                tool_id, name, arguments = block.get("id"), block.get("name"), block.get("input")
                if (kind != "assistant" or not isinstance(tool_id, str) or not tool_id or tool_id in seen or
                        name not in ("Read", "Edit") or not isinstance(arguments, dict)):
                    return False
                file_path = arguments.get("file_path")
                if not isinstance(file_path, str) or not file_path:
                    return False
                path = Path(file_path)
                path = (path if path.is_absolute() else directory / path).resolve()
                if (name == "Read" and path not in expected_paths and path != output) or (
                        name == "Edit" and (path != output or read != set(spec["input_files"]))):
                    return False
                seen.add(tool_id)
                pending[tool_id] = (name, expected_paths.get(path))
            elif block_type == "tool_result":
                tool_id = block.get("tool_use_id")
                if kind != "user" or not isinstance(tool_id, str) or tool_id not in pending or block.get("is_error", False) is not False:
                    return False
                name, relative = pending.pop(tool_id)
                if name == "Read" and relative is not None:
                    read.add(relative)
                if name == "Edit":
                    edited = True
    return initialized and edited and not pending and read == set(spec["input_files"])


def outcome(spec, directory, capture):
    unchanged = True
    for relative, original in spec["input_files"].items():
        try:
            actual = read_regular(directory / relative, len(original["bytes"]))
            unchanged &= actual == original["bytes"] and hashlib.sha256(actual).hexdigest() == original["sha256"]
        except (OSError, ValueError):
            unchanged = False
    checked = artifact_check(spec, directory)
    cycle = tool_cycle(capture["events"], spec, directory)
    finals = [record["event"] for record in capture["events"] if record["event"].get("type") == "result"]
    final = finals[0] if len(finals) == 1 else None
    inventory = capture.get("tool_inventory", {})
    finished = bool(final and final.get("subtype") == "success" and final.get("is_error", False) is False and
                    capture.get("exit_code") == 0 and capture.get("timed_out") is False and
                    capture.get("cancel_sent") is False and inventory.get("status") == "verified" and
                    inventory.get("requested") == ["Edit", "Read"] and not capture.get("diagnostics"))
    usage = final.get("usage", {}) if final else {}
    observed = {key: usage[key] for key in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
                if isinstance(usage, dict) and type(usage.get(key)) is int and usage[key] >= 0}
    return {"passed": bool(finished and cycle and unchanged and checked["valid"]),
            "assertion_passed": bool(cycle and unchanged and checked["valid"]),
            "tool_cycle_passed": cycle, "inputs_unchanged": bool(unchanged), "artifact_check": checked,
            "observed_input_usage": observed or None, "result": final,
            "task_revision": spec["task_revision"],
            "input_sha256": {name: original["sha256"] for name, original in spec["input_files"].items()},
            "artifact_path": str(directory / spec["output_path"])}
