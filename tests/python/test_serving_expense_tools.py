from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "serving"))
import expense_calculator as calculator
import claude_code_tasks as client
import expense_tools_cases as cases


CSV = (
    "id,date,description,category,currency,amount,status\n"
    'z9,2026-10-20,"Ticket, seat",travel,USD,10.10,cleared\n'
    "a2,2026-10-20,Refund,travel,USD,-0.11,cleared\n"
    "q1,2026-10-20,Free,supplies,USD,0.00,cleared\n"
    "c8,2026-10-20,Pending,travel,USD,1.00,pending\n"
    "d3,2026-10-20,Foreign,travel,CNY,2.00,cleared\n"
)
SELECTION = {"currency": "USD", "status": "cleared", "categories": ["travel", "supplies", "unused"]}
SUMMARY = {"currency": "USD", "included_rows": ["z9", "a2", "q1"],
           "category_totals_cents": {"travel": 999, "supplies": 0, "unused": 0}, "total_cents": 999}


class CalculatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def workspace(self):
        inputs = {"transactions.csv": CSV.encode(), "expense-rules.txt": b"Select cleared USD travel and supplies.\n",
                  "calculation-plan.json": json.dumps(SELECTION).encode(),
                  "expense_calculator.py": (ROOT / "benchmarks/serving/expense_calculator.py").read_bytes(),
                  "expenses.json": b""}
        for relative, data in inputs.items():
            (self.directory / relative).write_bytes(data)
        return inputs

    def run_calculator(self):
        return subprocess.run(["python3", "-I", "expense_calculator.py"], cwd=self.directory,
                              capture_output=True, text=True, timeout=5)

    def test_exact_signed_cents_quoted_csv_source_order_and_zero_categories(self):
        rows = calculator.parse_rows(CSV.encode())
        selection = calculator.parse_selection(json.dumps(SELECTION).encode())
        self.assertEqual(calculator.calculate(rows, selection), SUMMARY)
        self.assertEqual(calculator.parse_cents("+12.34"), 1234)
        self.assertEqual(calculator.parse_cents("-0.01"), -1)
        self.assertEqual(calculator.parse_cents("999999999999.99"), 99999999999999)

    def test_general_selection_changes_currency_status_and_categories(self):
        rows = calculator.parse_rows(CSV.encode())
        selected = calculator.Selection("CNY", "cleared", ("travel",))
        self.assertEqual(calculator.calculate(rows, selected), {
            "currency": "CNY", "included_rows": ["d3"], "category_totals_cents": {"travel": 200}, "total_cents": 200})
        selected = calculator.Selection("USD", "pending", ("travel",))
        self.assertEqual(calculator.calculate(rows, selected)["included_rows"], ["c8"])

    def test_actual_fixed_command_returns_only_receipt_and_writes_nothing(self):
        inputs = self.workspace()
        result = self.run_calculator()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        receipt = json.loads(result.stdout)
        self.assertEqual(receipt, {
            "schema": "apxinf-expense-calculation-v1", "selection": SELECTION, "row_count": 5, "summary": SUMMARY,
            "source_sha256": hashlib.sha256(inputs["transactions.csv"]).hexdigest(),
            "rules_sha256": hashlib.sha256(inputs["expense-rules.txt"]).hexdigest(),
            "plan_sha256": hashlib.sha256(inputs["calculation-plan.json"]).hexdigest(),
            "calculator_sha256": hashlib.sha256(inputs["expense_calculator.py"]).hexdigest()})
        self.assertEqual(set(path.name for path in self.directory.iterdir()), set(inputs))
        for relative, data in inputs.items():
            self.assertEqual((self.directory / relative).read_bytes(), data)

    def test_invalid_amounts_and_rows_fail_before_filtering(self):
        for value in ("1", "1.0", "1.000", "NaN", "1e2", " 1.00", "1000000000000.00", "1,00", "--1.00"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                calculator.parse_cents(value)
        for source in (CSV.replace("id,date", "ID,date"), CSV + "short,row\n",
                       CSV + "z9,2026-10-20,Duplicate,other,CNY,1.00,pending\n",
                       CSV + "x,2026-10-20,Hidden,other,CNY,broken,pending\n",
                       CSV.replace("Free,supplies", ",supplies"), CSV + '\"unterminated'):
            with self.subTest(source=source), self.assertRaises((ValueError, calculator.csv.Error)):
                calculator.parse_rows(source.encode())

    def test_plan_rejects_missing_extra_duplicate_nonfinite_and_wrong_types(self):
        for value in ("null", "{}", '{"currency":"USD","currency":"CNY","status":"cleared","categories":["travel"]}',
                      json.dumps({**SELECTION, "extra": 1}), json.dumps({**SELECTION, "categories": []}),
                      json.dumps({**SELECTION, "categories": ["travel", "travel"]}),
                      json.dumps({**SELECTION, "categories": [True]}), json.dumps({**SELECTION, "status": 1}),
                      json.dumps(SELECTION) + "{}", '{"currency":NaN}', " " * 4097):
            with self.subTest(value=value), self.assertRaises(ValueError):
                calculator.parse_selection(value.encode())

    def test_row_category_and_safe_integer_limits(self):
        header = ",".join(calculator.COLUMNS) + "\n"
        rows = "".join(f"x{index},2026-10-20,Item,travel,USD,1.00,cleared\n" for index in range(101))
        with self.assertRaises(ValueError):
            calculator.parse_rows((header + rows).encode())
        with self.assertRaises(ValueError):
            calculator.parse_selection(json.dumps({**SELECTION, "categories": [str(index) for index in range(17)]}).encode())
        with self.assertRaises(ValueError):
            calculator.checked_integer(2 ** 53)
        with self.assertRaises(ValueError):
            calculator.calculate([calculator.Row("x", "travel", "USD", 2 ** 53 - 1, "cleared"),
                                  calculator.Row("y", "travel", "USD", 1, "cleared")],
                                 calculator.Selection("USD", "cleared", ("travel",)))

    def test_failed_subprocess_has_stderr_without_receipt(self):
        self.workspace()
        (self.directory / "transactions.csv").write_text(CSV.replace("2.00", "invalid"))
        result = self.run_calculator()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertTrue(result.stderr)

    def test_regular_file_boundary_rejects_symlink_directory_fifo_and_size(self):
        self.workspace()
        target = self.directory / "transactions.csv"
        target.unlink()
        for kind in ("symlink", "directory", "fifo", "oversize"):
            with self.subTest(kind=kind):
                if kind == "symlink":
                    target.symlink_to(self.directory / "expense-rules.txt")
                elif kind == "directory":
                    target.mkdir()
                elif kind == "fifo":
                    os.mkfifo(target)
                else:
                    target.write_bytes(b" " * 65537)
                result = self.run_calculator()
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
                if kind == "directory":
                    target.rmdir()
                else:
                    target.unlink()


def event(kind, blocks=None, **fields):
    value = {"type": kind, **fields}
    if blocks is not None:
        value["message"] = {"role": kind, "content": blocks}
    return {"elapsed_s": .1, "event": value}


def call(identifier, name, **arguments):
    return event("assistant", [{"type": "tool_use", "id": identifier, "name": name, "input": arguments}])


def result(identifier, content="Completed.", **fields):
    return event("user", [{"type": "tool_result", "tool_use_id": identifier, "content": content, "is_error": False}], **fields)


def block(record):
    return record["event"]["message"]["content"][0]


class ExpenseToolCaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sequence = 0

    def fixture(self, name="expenses-tools-cny", categories=None):
        self.sequence += 1
        directory = self.root / str(self.sequence)
        spec = client.prepare_task(name, directory)
        plan = deepcopy(spec["selection"])
        if categories is not None:
            plan["categories"] = categories
        plan_text = json.dumps(plan, indent=2) + "\n"
        (directory / cases.PLAN).write_text(plan_text)
        process = subprocess.run(["python3", "-I", "expense_calculator.py"], cwd=directory,
                                 capture_output=True, text=True, timeout=5)
        self.assertEqual(process.returncode, 0, process.stderr)
        receipt = json.loads(process.stdout)
        self.assertEqual(receipt["summary"], spec["expected"])
        output = json.dumps(receipt["summary"], indent=2) + "\n"
        (directory / cases.OUTPUT).write_text(output)
        reads = [call(f"r{index}", "Read", file_path=relative) for index, relative in
                 enumerate(sorted(cases.REQUIRED_READS))]
        events = [event("system", subtype="init", tools=["Read", "Edit", "Bash", "Task"])]
        events += [event("assistant", [block(item) for item in reads])]
        events += [result(block(item)["id"]) for item in reads]
        events += [call("plan", "Edit", file_path=cases.PLAN, old_string="", new_string=plan_text, replace_all=False),
                   result("plan"), call("calculate", "Bash", command=cases.COMMAND, description="Calculate expense summary"),
                   result("calculate", process.stdout, tool_use_result={"stdout": process.stdout, "stderr": "",
                          "interrupted": False, "isImage": False, "noOutputExpected": False}),
                   call("output", "Edit", file_path=str(directory / cases.OUTPUT), old_string="", new_string=output),
                   result("output"), event("result", subtype="success", is_error=False, result="Saved.", usage={"input_tokens": 700})]
        capture = {"events": events, "exit_code": 0, "timed_out": False, "cancel_sent": False,
                   "tool_inventory": {"status": "verified", "requested": ["Bash", "Edit", "Read"]}, "diagnostics": []}
        return name, directory, spec, capture

    def outcome(self, fixture, capture=None):
        name, directory, spec, original = fixture
        return client.task_outcome(name, spec, directory, original if capture is None else capture)

    def rejected(self, fixture, mutate):
        capture = deepcopy(fixture[3])
        mutate(capture)
        checked = self.outcome(fixture, capture)
        self.assertFalse(checked["passed"])
        return checked

    def test_both_original_cases_use_actual_calculator_receipts(self):
        for name, total, identifiers in (
            ("expenses-tools-cny", 14019, ["a01", "a02", "a03", "a04", "a08", "a09", "a10"]),
            ("expenses-tools-usd", 15700, ["b40", "b03", "b22", "b05", "b07", "b02", "b10"]),
        ):
            with self.subTest(name=name):
                fixture = self.fixture(name)
                checked = self.outcome(fixture)
                self.assertTrue(checked["passed"], checked)
                self.assertEqual(checked["calculation_receipt"]["summary"]["total_cents"], total)
                self.assertEqual(checked["calculation_receipt"]["summary"]["included_rows"], identifiers)
                self.assertEqual(checked["observed_input_usage"], {"input_tokens": 700})
                self.assertTrue(checked["inputs_unchanged"])
                self.assertEqual(fixture[2]["task_revision"], name + "-v1")

    def test_preparation_keeps_answers_private_and_targets_empty(self):
        specs = []
        for name in cases.CASES:
            directory = self.root / name
            spec = client.prepare_task(name, directory)
            specs.append(spec)
            self.assertEqual((directory / cases.PLAN).read_bytes(), b"")
            self.assertEqual((directory / cases.OUTPUT).read_bytes(), b"")
            for relative in ("expense_calculator.py", "calculator-usage.txt", "expense-rules.txt"):
                text = (directory / relative).read_text()
                self.assertNotIn(str(spec["expected"]["total_cents"]), text)
                for identifier in spec["expected"]["included_rows"]:
                    self.assertNotIn(identifier, text)
            self.assertEqual(spec["allowed"], ["Read", "Edit", "Bash(python3 -I expense_calculator.py)"])
        self.assertEqual(specs[0]["input_files"]["expense_calculator.py"], specs[1]["input_files"]["expense_calculator.py"])

    def test_category_order_json_layout_and_optional_complete_reads_are_free(self):
        fixture = self.fixture(categories=["software", "transport", "meals"])
        capture = fixture[3]
        capture["events"][7:7] = [call("optional", "Read", file_path="expense_calculator.py"), result("optional")]
        capture["events"][-1:-1] = [call("later", "Read", file_path=cases.OUTPUT), result("later")]
        self.assertTrue(self.outcome(fixture)["passed"])

    def test_correct_artifact_without_calculator_is_rejected(self):
        fixture = self.fixture()
        checked = self.rejected(fixture, lambda value: value["events"].__delitem__(slice(9, 11)))
        self.assertTrue(checked["artifact_check"]["valid"])
        self.assertFalse(checked["tool_cycle_passed"])

    def test_wrong_selections_and_plan_fields_fail_even_with_correct_artifact(self):
        fixture = self.fixture()
        plans = ({**fixture[2]["selection"], "status": "pending"},
                 {**fixture[2]["selection"], "currency": "USD"},
                 {**fixture[2]["selection"], "categories": ["transport", "meals"]},
                 {**fixture[2]["selection"], "categories": ["transport", "meals", "software", "software"]},
                 {**fixture[2]["selection"], "extra": False}, {"currency": "CNY"})
        for plan in plans:
            with self.subTest(plan=plan):
                self.rejected(fixture, lambda value: block(value["events"][7])["input"].update(new_string=json.dumps(plan)))

    def test_receipt_hashes_fields_types_and_source_order_are_exact(self):
        fixture = self.fixture("expenses-tools-usd")
        mutations = [lambda value, field=field: value.update({field: "0" * 64}) for field in
                     ("source_sha256", "rules_sha256", "plan_sha256", "calculator_sha256")]
        mutations += [lambda value: value.update(row_count=True), lambda value: value.update(row_count=12.0),
                      lambda value: value.update(extra=0), lambda value: value.update(schema="unknown"),
                      lambda value: value["selection"].update(status="posted"),
                      lambda value: value["summary"].update(total_cents=15700.0),
                      lambda value: value["summary"].update(total_cents=True),
                      lambda value: value["summary"]["included_rows"].sort(),
                      lambda value: value["summary"]["category_totals_cents"].update(travel=3000.0)]
        for mutate in mutations:
            capture = deepcopy(fixture[3])
            bash = capture["events"][10]
            receipt = json.loads(block(bash)["content"])
            mutate(receipt)
            text = json.dumps(receipt)
            block(bash)["content"] = text
            bash["event"]["tool_use_result"]["stdout"] = text
            self.assertFalse(self.outcome(fixture, capture)["passed"])

    def test_entire_bounded_stdout_must_be_one_strict_receipt(self):
        fixture = self.fixture()
        original = block(fixture[3]["events"][10])["content"]
        variants = ("log\n" + original, original + "{}", original[:-2] + ',"row_count":10}',
                    original.replace('"total_cents": 14019', '"total_cents": NaN'), " " * 16385, "null")
        for text in variants:
            capture = deepcopy(fixture[3])
            block(capture["events"][10])["content"] = text
            capture["events"][10]["event"]["tool_use_result"]["stdout"] = text
            self.assertFalse(self.outcome(fixture, capture)["passed"])
        capture = deepcopy(fixture[3])
        text = " \n" + original + " \n"
        block(capture["events"][10])["content"] = text
        capture["events"][10]["event"]["tool_use_result"]["stdout"] = text
        self.assertTrue(self.outcome(fixture, capture)["passed"])

    def test_bash_metadata_requires_actual_pinned_shape_and_clean_execution(self):
        fixture = self.fixture()
        for field, value in (("stderr", "warning"), ("stderr", None), ("stdout", []), ("interrupted", True),
                             ("isImage", True), ("noOutputExpected", True), ("interrupted", 0), ("isImage", "false")):
            with self.subTest(field=field, value=value):
                self.rejected(fixture, lambda capture: capture["events"][10]["event"]["tool_use_result"].update({field: value}))
        for field in ("stdout", "stderr", "interrupted", "isImage", "noOutputExpected"):
            self.rejected(fixture, lambda capture: capture["events"][10]["event"]["tool_use_result"].pop(field))
        self.rejected(fixture, lambda capture: capture["events"][10]["event"].pop("tool_use_result"))
        self.rejected(fixture, lambda capture: block(capture["events"][10]).update(content="different"))

    def test_command_accepts_description_timeout_but_rejects_execution_changes(self):
        fixture = self.fixture()
        capture = deepcopy(fixture[3])
        block(capture["events"][9])["input"].update(timeout=5000)
        self.assertTrue(self.outcome(fixture, capture)["passed"])
        for arguments in ({"command": cases.COMMAND + " extra"}, {"command": cases.COMMAND + " > expenses.json"},
                          {"command": cases.COMMAND + " && echo done"}, {"command": cases.COMMAND + " &"},
                          {"run_in_background": True}, {"timeout": True}, {"timeout": 0}, {"timeout": 600001},
                          {"description": None}, {"description": "x" * 4097}):
            self.rejected(fixture, lambda value: block(value["events"][9])["input"].update(arguments))

    def test_missing_empty_target_reads_partial_reads_and_unknown_paths_fail(self):
        fixture = self.fixture()
        for index in range(5):
            capture = deepcopy(fixture[3])
            del capture["events"][1]["event"]["message"]["content"][index]
            del capture["events"][index + 2]
            self.assertFalse(self.outcome(fixture, capture)["passed"])
        for arguments in ({"limit": 1}, {"offset": 1}, {"pages": "1"}, {"file_path": "unknown.txt"}):
            self.rejected(fixture, lambda value: value["events"][1]["event"]["message"]["content"][0]["input"].update(arguments))

    def test_mutations_wait_for_all_dependent_results(self):
        fixture = self.fixture()
        for remove in (6, 8, 10, 12):
            self.rejected(fixture, lambda value: value["events"].pop(remove))
        for index, new_call in ((8, call("early-bash", "Bash", command=cases.COMMAND)),
                                (10, deepcopy(fixture[3]["events"][11])),
                                (12, call("early-read", "Read", file_path=cases.OUTPUT))):
            self.rejected(fixture, lambda value: value["events"].insert(index, new_call))
        capture = deepcopy(fixture[3])
        pending_read = capture["events"].pop(6)
        capture["events"].insert(8, pending_read)
        self.assertFalse(self.outcome(fixture, capture)["passed"])

    def test_failed_or_invalid_attempt_cannot_be_repaired_into_a_pass(self):
        fixture = self.fixture()
        for position in (2, 8, 10, 12):
            capture = deepcopy(fixture[3])
            failed = deepcopy(capture["events"][position])
            block(failed)["is_error"] = True
            capture["events"].insert(position, failed)
            self.assertFalse(self.outcome(fixture, capture)["passed"])
        invalid = call("bad-plan", "Edit", file_path=cases.PLAN, old_string="", new_string="{}")
        self.rejected(fixture, lambda value: value["events"].insert(7, invalid))

    def test_id_roles_duplicates_and_finalization_are_strict(self):
        fixture = self.fixture()
        mutations = (
            lambda value: block(value["events"][7]).update(id="r0"),
            lambda value: block(value["events"][8]).update(tool_use_id="unknown"),
            lambda value: value["events"][7]["event"].update(type="user"),
            lambda value: value["events"][8]["event"]["message"].update(role="assistant"),
            lambda value: value["events"].insert(9, deepcopy(value["events"][8])),
            lambda value: value["events"].append(deepcopy(value["events"][-1])),
            lambda value: value["events"].append(call("late", "Read", file_path=cases.OUTPUT)),
            lambda value: value["events"].append(event("stream_event", event={"type": "message_start"})),
            lambda value: value["events"][-1]["event"].update(subtype="error_max_turns"),
            lambda value: value["events"][-1]["event"].update(is_error=True),
            lambda value: value["events"].pop(),
        )
        for mutate in mutations:
            self.rejected(fixture, mutate)

    def test_capture_inventory_exit_and_diagnostics_are_required(self):
        fixture = self.fixture()
        mutations = (
            lambda value: value["events"].pop(0),
            lambda value: value["events"][0]["event"].update(tools=["Read", "Edit"]),
            lambda value: value["events"].insert(1, deepcopy(value["events"][0])),
            lambda value: value.pop("tool_inventory"),
            lambda value: value["tool_inventory"].update(requested=["Read", "Edit"]),
            lambda value: value["tool_inventory"].update(status="invalid"),
            lambda value: value.update(exit_code=1), lambda value: value.update(timed_out=True),
            lambda value: value.update(cancel_sent=True), lambda value: value["diagnostics"].append("truncated"),
        )
        for mutate in mutations:
            self.rejected(fixture, mutate)

    def test_saved_plan_output_and_immutable_inputs_are_bound_to_evidence(self):
        for relative in cases.LIMITS:
            with self.subTest(relative=relative):
                fixture = self.fixture()
                path = fixture[1] / relative
                path.write_bytes(path.read_bytes() + b" ")
                checked = self.outcome(fixture)
                if relative == cases.OUTPUT:
                    self.assertTrue(checked["passed"])
                else:
                    self.assertFalse(checked["passed"])
        fixture = self.fixture()
        (fixture[1] / cases.OUTPUT).write_text(json.dumps({**fixture[2]["expected"], "total_cents": 0}))
        self.assertFalse(self.outcome(fixture)["passed"])
        fixture = self.fixture()
        block(fixture[3]["events"][11])["input"]["new_string"] = json.dumps({**fixture[2]["expected"], "total_cents": 0})
        self.assertFalse(self.outcome(fixture)["passed"])

    def test_controller_rejects_symlink_fifo_directory_and_oversize_files(self):
        for relative in (cases.PLAN, cases.OUTPUT, "expense_calculator.py"):
            for kind in ("symlink", "fifo", "directory", "oversize"):
                with self.subTest(relative=relative, kind=kind):
                    fixture = self.fixture()
                    path = fixture[1] / relative
                    saved = path.read_bytes()
                    path.unlink()
                    if kind == "symlink":
                        outside = fixture[1] / "outside"
                        outside.write_bytes(saved)
                        path.symlink_to(outside)
                    elif kind == "fifo":
                        os.mkfifo(path)
                    elif kind == "directory":
                        path.mkdir()
                    else:
                        path.write_bytes(b" " * (cases.LIMITS[relative] + 1))
                    self.assertFalse(self.outcome(fixture)["passed"])

    def test_parser_preserves_defaults_and_exposes_new_cases(self):
        self.assertEqual(client.parser().parse_args(["--output-dir", "unused"]).tasks, ["read", "edit", "check"])
        parsed = client.parser().parse_args(["--output-dir", "unused", "--tasks", *cases.CASES])
        self.assertEqual(parsed.tasks, list(cases.CASES))


if __name__ == "__main__":
    unittest.main()
