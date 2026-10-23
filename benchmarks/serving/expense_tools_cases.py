"""Original expense tasks with calculation receipts and ordered tool evidence."""

import hashlib
import json
import os
from pathlib import Path

from practical_cases import exact_value, read_regular, reject_constant, unique_object


COMMAND = "python3 -I expense_calculator.py"
PLAN = "calculation-plan.json"
OUTPUT = "expenses.json"
LIMITS = {"transactions.csv": 65536, "expense-rules.txt": 8192,
          "calculator-usage.txt": 8192, "expense_calculator.py": 32768, PLAN: 4096, OUTPUT: 8192}
REQUIRED_READS = {"transactions.csv", "expense-rules.txt", "calculator-usage.txt", PLAN, OUTPUT}
USAGE = (
    "Read transactions.csv, expense-rules.txt, and this usage file.\n"
    "Read the empty calculation-plan.json and expenses.json before any Edit.\n"
    "Use Edit once to insert the complete plan into calculation-plan.json.\n"
    "The plan has exactly currency, status, and categories.\n"
    "currency and status are nonempty strings. categories is a unique, nonempty list of strings.\n"
    "Select values from expense-rules.txt. Category order does not affect selection.\n"
    "Set old_string to an empty string. Set new_string to the complete JSON object.\n"
    "After the plan Edit succeeds, use Bash to run exactly python3 -I expense_calculator.py.\n"
    "The calculator reads fixed files and prints one JSON receipt. It writes no files.\n"
    "The receipt contains schema, source_sha256, rules_sha256, plan_sha256, calculator_sha256, selection, row_count, and summary.\n"
    "After Bash succeeds, use Edit once to insert the receipt's complete summary into expenses.json.\n"
    "Keep the four input files unchanged. Reply after the final Edit succeeds.\n"
)
CASES = {
    "expenses-tools-cny": {
        "revision": "expenses-tools-cny-v1", "row_count": 10,
        "selection": {"currency": "CNY", "status": "posted", "categories": ["transport", "meals", "software"]},
        "csv": (
            "id,date,description,category,currency,amount,status\n"
            "a01,2026-10-20,Commute pass,transport,CNY,23.40,posted\n"
            "a02,2026-10-20,Breakfast,meals,CNY,18.75,posted\n"
            "a03,2026-10-20,Backup service,software,CNY,49.99,posted\n"
            "a04,2026-10-21,Taxi refund,transport,CNY,-3.20,posted\n"
            "a05,2026-10-21,Lunch,meals,CNY,12.50,pending\n"
            "a06,2026-10-21,Office supplies,office,CNY,9.80,posted\n"
            "a07,2026-10-21,Foreign taxi,transport,USD,8.00,posted\n"
            "a08,2026-10-22,Dinner,meals,CNY,41.25,posted\n"
            "a09,2026-10-22,Editor add-on,software,CNY,10.01,posted\n"
            "a10,2026-10-22,Small software refund,software,CNY,-0.01,posted\n"
        ),
        "expected": {"currency": "CNY", "included_rows": ["a01", "a02", "a03", "a04", "a08", "a09", "a10"],
                     "category_totals_cents": {"transport": 2020, "meals": 6000, "software": 5999}, "total_cents": 14019},
    },
    "expenses-tools-usd": {
        "revision": "expenses-tools-usd-v1", "row_count": 12,
        "selection": {"currency": "USD", "status": "cleared", "categories": ["travel", "lodging", "supplies"]},
        "csv": (
            "id,date,description,category,currency,amount,status\n"
            "b40,2026-10-20,Train pass,travel,USD,37.45,cleared\n"
            'b12,2026-10-20,"Sandwich, tea",meals,USD,14.00,cleared\n'
            "b03,2026-10-20,Hotel,lodging,USD,128.70,cleared\n"
            "b22,2026-10-21,Train refund,travel,USD,-7.45,cleared\n"
            "b05,2026-10-21,Cable,supplies,USD,6.30,cleared\n"
            "b61,2026-10-21,Toll,travel,USD,4.20,posted\n"
            "b07,2026-10-21,Hotel refund,lodging,USD,-8.70,cleared\n"
            "b80,2026-10-21,Foreign taxi,travel,CNY,60.00,cleared\n"
            "b02,2026-10-22,Folder,supplies,USD,0.70,cleared\n"
            "b10,2026-10-22,Waived fare,travel,USD,0.00,cleared\n"
            "b04,2026-10-22,Hotel deposit,lodging,USD,25.00,pending\n"
            "b90,2026-10-22,Meal refund,meals,USD,-2.00,cleared\n"
        ),
        "expected": {"currency": "USD", "included_rows": ["b40", "b03", "b22", "b05", "b07", "b02", "b10"],
                     "category_totals_cents": {"travel": 3000, "lodging": 12000, "supplies": 700}, "total_cents": 15700},
    },
}


def prepare(name, directory):
    case = CASES[name]
    selection = case["selection"]
    categories = ", ".join(selection["categories"])
    rules = (
        f"Select only {selection['status']} {selection['currency']} transactions in these categories: {categories}.\n"
        "Include negative refunds and zero amounts. Exclude all other rows.\n"
        "Use exact integer cents. Do not round or convert currencies.\n"
        "Preserve CSV source order for included transaction IDs.\n"
        "Write exactly currency, included_rows, category_totals_cents, and total_cents.\n"
        f"currency is {selection['currency']}. included_rows is a list of selected transaction IDs.\n"
        "category_totals_cents maps each selected category to its integer total, including zero totals.\n"
        "total_cents is the integer sum of selected amounts.\n"
    )
    files = {"transactions.csv": case["csv"].encode(), "expense-rules.txt": rules.encode(),
             "calculator-usage.txt": USAGE.encode(),
             "expense_calculator.py": Path(__file__).with_name("expense_calculator.py").read_bytes()}
    snapshots = {}
    for relative, data in files.items():
        (directory / relative).write_bytes(data)
        snapshots[relative] = {"bytes": data, "sha256": hashlib.sha256(data).hexdigest()}
    for relative in (PLAN, OUTPUT):
        (directory / relative).write_bytes(b"")
    return {"prompt": "Read calculator-usage.txt and follow its complete workflow. "
            "Read the CSV, rules, empty plan, and empty expenses.json before any Edit. "
            "Use the calculator receipt to save the final summary with Edit. Keep all input files unchanged.",
            "tools": ["Read", "Edit", "Bash"], "allowed": ["Read", "Edit", f"Bash({COMMAND})"],
            "input_files": snapshots, "output_path": OUTPUT, "output_limit_bytes": 8192,
            "expected": case["expected"], "selection": selection, "row_count": case["row_count"],
            "task_revision": case["revision"]}


def strict_json(data, limit):
    if type(data) is not str or len(data.encode("utf-8")) > limit:
        raise ValueError("The JSON text is invalid or exceeds its byte limit.")
    return json.loads(data, object_pairs_hook=unique_object, parse_constant=reject_constant)


def plan_check(value, expected):
    if type(value) is not dict or value.keys() != {"currency", "status", "categories"}:
        raise ValueError("The plan fields differ from the selection contract.")
    categories = value["categories"]
    if (type(categories) is not list or not 1 <= len(categories) <= 16 or
            not all(type(item) is str for item in categories) or len(set(categories)) != len(categories) or
            set(categories) != set(expected["categories"]) or
            not exact_value(value["currency"], expected["currency"]) or
            not exact_value(value["status"], expected["status"])):
        raise ValueError("The plan selection differs from the rules.")


def receipt_check(event, block, spec, plan_text, plan):
    metadata = event.get("tool_use_result")
    if not isinstance(metadata, dict):
        raise ValueError("The Bash result has no structured execution evidence.")
    if (type(metadata.get("stdout")) is not str or type(metadata.get("stderr")) is not str or
            metadata["stderr"] != "" or any(metadata.get(key) is not False for key in
                                            ("interrupted", "isImage", "noOutputExpected")) or
            block.get("content") != metadata["stdout"]):
        raise ValueError("The Bash execution evidence is invalid.")
    receipt = strict_json(metadata["stdout"], 16384)
    expected = {"schema": "apxinf-expense-calculation-v1", "selection": plan,
                "row_count": spec["row_count"], "summary": spec["expected"],
                "plan_sha256": hashlib.sha256(plan_text.encode()).hexdigest()}
    for field, relative in (("source_sha256", "transactions.csv"), ("rules_sha256", "expense-rules.txt"),
                            ("calculator_sha256", "expense_calculator.py")):
        expected[field] = spec["input_files"][relative]["sha256"]
    if not exact_value(receipt, expected):
        raise ValueError("The calculator receipt differs from the snapshots, plan, or independent oracle.")
    return receipt


def command_check(arguments):
    if arguments.keys() - {"command", "description", "timeout"} or arguments.get("command") != COMMAND:
        return False
    if "description" in arguments and (type(arguments["description"]) is not str or
                                       len(arguments["description"].encode("utf-8")) > 4096):
        return False
    if "timeout" in arguments and (type(arguments["timeout"]) is not int or not 1 <= arguments["timeout"] <= 600000):
        return False
    return True


def tool_cycle(events, spec, directory):
    paths = {directory / name: name for name in LIMITS}
    seen, pending, read = set(), {}, set()
    phase = "reading"
    initialized = False
    plan_text = plan = receipt = final = None
    try:
        for record in events:
            event = record["event"]
            kind = event.get("type")
            if kind == "system" and event.get("subtype") == "init":
                advertised = event.get("tools")
                if initialized or type(advertised) is not list or not all(type(item) is str for item in advertised):
                    raise ValueError("The initial tool inventory is invalid or repeated.")
                if not {"Read", "Edit", "Bash"}.issubset(advertised):
                    raise ValueError("The initial tool inventory omits a required tool.")
                initialized = True
                continue
            if kind == "result":
                if not initialized or phase != "finish" or pending or final is not None:
                    raise ValueError("The final result is early, repeated, or has pending calls.")
                if event.get("subtype") != "success" or event.get("is_error", False) is not False:
                    raise ValueError("The final result is unsuccessful.")
                final, phase = event, "finalized"
                continue
            if kind not in ("assistant", "user", "stream_event"):
                continue
            if not initialized or phase == "finalized":
                raise ValueError("Conversation evidence occurs outside the active session.")
            if kind == "stream_event":
                continue
            message = event.get("message", {})
            if not isinstance(message, dict) or message.get("role") != kind or type(message.get("content")) is not list:
                raise ValueError("A conversation event has invalid role or content.")
            for block in message["content"]:
                if not isinstance(block, dict):
                    raise ValueError("A conversation block is invalid.")
                block_type = block.get("type")
                if block_type == "tool_use":
                    identifier, name, arguments = block.get("id"), block.get("name"), block.get("input")
                    if (kind != "assistant" or type(identifier) is not str or not identifier or identifier in seen or
                            name not in ("Read", "Edit", "Bash") or type(arguments) is not dict):
                        raise ValueError("A tool call has invalid identity, role, name, or arguments.")
                    if name == "Bash":
                        if phase != "calculate" or pending or not command_check(arguments):
                            raise ValueError("The calculator command is invalid or out of order.")
                        phase = "calculation pending"
                        relative, replacement = None, None
                    else:
                        file_path = arguments.get("file_path")
                        if type(file_path) is not str or not file_path:
                            raise ValueError("A file tool has no valid path.")
                        path = Path(file_path)
                        path = Path(os.path.abspath(path if path.is_absolute() else directory / path))
                        relative = paths.get(path)
                        if relative is None:
                            raise ValueError("A file tool addresses an unknown path.")
                        replacement = None
                        if name == "Read":
                            if arguments.keys() != {"file_path"} or phase.endswith("pending"):
                                raise ValueError("A Read is partial or overlaps a pending mutation.")
                        else:
                            if (pending or arguments.keys() - {"file_path", "old_string", "new_string", "replace_all"} or
                                    arguments.get("replace_all", False) is not False or
                                    arguments["old_string"] != "" or type(arguments["new_string"]) is not str):
                                raise ValueError("An Edit is invalid or overlaps a pending call.")
                            replacement = arguments["new_string"]
                            proposed = strict_json(replacement, LIMITS[relative])
                            if phase == "reading" and relative == PLAN and REQUIRED_READS <= read:
                                plan_check(proposed, spec["selection"])
                                phase = "plan pending"
                            elif phase == "artifact" and relative == OUTPUT:
                                if not exact_value(proposed, receipt["summary"]):
                                    raise ValueError("The output Edit differs from the calculator receipt.")
                                phase = "artifact pending"
                            else:
                                raise ValueError("An Edit has the wrong target or phase.")
                    seen.add(identifier)
                    pending[identifier] = (name, relative, replacement)
                elif block_type == "tool_result":
                    identifier = block.get("tool_use_id")
                    if (kind != "user" or type(identifier) is not str or identifier not in pending or
                            block.get("is_error", False) is not False):
                        raise ValueError("A tool result is failed, unknown, repeated, or has the wrong role.")
                    name, relative, replacement = pending.pop(identifier)
                    if name == "Read":
                        read.add(relative)
                    elif name == "Bash":
                        receipt = receipt_check(event, block, spec, plan_text, plan)
                        phase = "artifact"
                    elif relative == PLAN:
                        plan_text, plan = replacement, strict_json(replacement, 4096)
                        phase = "calculate"
                    else:
                        saved = read_regular(directory / OUTPUT, 8192).decode("utf-8")
                        if not exact_value(strict_json(saved, 8192), strict_json(replacement, 8192)):
                            raise ValueError("The saved output differs from the successful Edit.")
                        phase = "finish"
                elif block_type not in ("text", "thinking", "redacted_thinking") or kind != "assistant":
                    raise ValueError("A conversation block has an unsupported type or role.")
        if phase != "finalized" or pending:
            raise ValueError("The required tool workflow is incomplete.")
        if read_regular(directory / PLAN, 4096) != plan_text.encode():
            raise ValueError("The saved plan differs from the successful Edit.")
        return {"valid": True, "reason": None, "receipt": receipt, "result": final}
    except (OSError, ValueError, UnicodeError, RecursionError, KeyError, TypeError, AttributeError) as error:
        return {"valid": False, "reason": str(error), "receipt": receipt, "result": final}


def outcome(spec, directory, capture):
    directory = directory.absolute()
    unchanged = True
    for relative, original in spec["input_files"].items():
        try:
            unchanged &= read_regular(directory / relative, LIMITS[relative]) == original["bytes"]
        except (OSError, ValueError):
            unchanged = False
    cycle = tool_cycle(capture["events"], spec, directory)
    try:
        actual = strict_json(read_regular(directory / OUTPUT, 8192).decode("utf-8"), 8192)
        checked = {"valid": exact_value(actual, spec["expected"]), "reason": None}
        if not checked["valid"]:
            checked["reason"] = "The saved artifact differs from the independent oracle."
    except (OSError, ValueError, UnicodeError, RecursionError) as error:
        checked = {"valid": False, "reason": str(error)}
    inventory = capture.get("tool_inventory", {})
    finished = (capture.get("exit_code") == 0 and capture.get("timed_out") is False and
                capture.get("cancel_sent") is False and not capture.get("diagnostics") and
                inventory.get("status") == "verified" and inventory.get("requested") == ["Bash", "Edit", "Read"])
    finals = [record["event"] for record in capture["events"] if record["event"].get("type") == "result"]
    final = finals[0] if len(finals) == 1 else None
    usage = final.get("usage", {}) if final else {}
    observed = {key: usage[key] for key in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
                if isinstance(usage, dict) and type(usage.get(key)) is int and usage[key] >= 0}
    assertion = cycle["valid"] and unchanged and checked["valid"]
    return {"passed": bool(finished and assertion), "assertion_passed": bool(assertion),
            "tool_cycle_passed": cycle["valid"], "tool_cycle_reason": cycle["reason"],
            "calculation_receipt": cycle["receipt"], "inputs_unchanged": bool(unchanged),
            "artifact_check": checked, "result": final, "observed_input_usage": observed or None,
            "task_revision": spec["task_revision"], "artifact_path": str(directory / OUTPUT),
            "input_sha256": {name: original["sha256"] for name, original in spec["input_files"].items()}}
