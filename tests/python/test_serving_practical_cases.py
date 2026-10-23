from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "serving"))
import claude_code_tasks as client


EXPENSES = {
    "currency": "CNY", "included_rows": ["t01", "t02", "t03", "t04", "t08", "t09"],
    "category_totals_cents": {"transport": 10150, "meals": 13000, "software": 8500}, "total_cents": 31650,
}
MEETING = {
    "meeting_date": "2026-10-22",
    "actions": [
        {"task": "send the revised onboarding checklist", "owner": "Mira", "due_date": "2026-10-24"},
        {"task": "test the backup restore", "owner": "Jon", "due_date": None},
        {"task": "Update the contact directory", "owner": None, "due_date": "2026-10-27"},
        {"task": "book the accessibility review", "owner": "Lina", "due_date": "2026-10-29"},
    ], "decisions": ["Keep the weekly release on Thursday."],
}


def event(kind, blocks=None, **fields):
    item = {"type": kind, **fields}
    if blocks is not None:
        item["message"] = {"content": blocks}
    return {"elapsed_s": .1, "event": item}


def call(tool_id, name, path):
    return event("assistant", [{"type": "tool_use", "id": tool_id, "name": name, "input": {"file_path": str(path)}}])


def result(tool_id):
    return event("user", [{"type": "tool_result", "tool_use_id": tool_id, "is_error": False, "content": "Completed."}])


class PracticalCaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def fixture(self, name="expenses"):
        directory = self.root / name
        spec = client.prepare_task(name, directory)
        expected = EXPENSES if name == "expenses" else MEETING
        (directory / spec["output_path"]).write_text(json.dumps(expected))
        events = [event("system", subtype="init", tools=["Read", "Edit"])]
        for index, path in enumerate(spec["input_files"]):
            events += [call(f"r{index}", "Read", path), result(f"r{index}")]
        events += [call("e0", "Edit", directory / spec["output_path"]), result("e0"),
                   event("result", subtype="success", is_error=False, result="Saved.", usage={"input_tokens": 501})]
        capture = {"events": events, "exit_code": 0, "timed_out": False, "cancel_sent": False,
                   "tool_inventory": {"status": "verified", "requested": ["Edit", "Read"]}, "diagnostics": []}
        return directory, spec, capture

    def outcome(self, directory, spec, capture):
        name = "expenses" if spec["task_revision"] == "expenses-v1" else "meeting-actions"
        return client.task_outcome(name, spec, directory, capture)

    def test_original_exact_artifacts_and_real_tool_cycle_pass(self):
        for name in ("expenses", "meeting-actions"):
            with self.subTest(name=name):
                directory, spec, capture = self.fixture(name)
                outcome = self.outcome(directory, spec, capture)
                self.assertTrue(outcome["passed"])
                self.assertTrue(outcome["tool_cycle_passed"])
                self.assertTrue(outcome["inputs_unchanged"])
                self.assertEqual(outcome["observed_input_usage"], {"input_tokens": 501})
                self.assertNotIn("31650", spec["prompt"])
                self.assertNotIn(str(directory), spec["prompt"])
                self.assertIn(f"Set file_path to {spec['output_path']},", spec["prompt"])
                self.assertEqual(spec["tools"], ["Read", "Edit"])

    def test_tasks_precreate_empty_output_and_keep_original_input_bytes(self):
        directory = self.root / "fresh"
        spec = client.prepare_task("expenses", directory)
        self.assertEqual((directory / spec["output_path"]).read_bytes(), b"")
        self.assertEqual(spec["input_files"]["transactions.csv"]["bytes"], (directory / "transactions.csv").read_bytes())

    def test_wrong_totals_refund_inclusion_or_type_fails(self):
        directory, spec, capture = self.fixture()
        values = []
        wrong = deepcopy(EXPENSES)
        wrong["total_cents"] = 33500
        values.append(wrong)
        wrong = deepcopy(EXPENSES)
        wrong["category_totals_cents"]["transport"] = 12000
        values.append(wrong)
        wrong = deepcopy(EXPENSES)
        wrong["included_rows"].append("t05")
        values.append(wrong)
        wrong = deepcopy(EXPENSES)
        wrong["total_cents"] = 31650.0
        values.append(wrong)
        wrong = deepcopy(EXPENSES)
        wrong["category_totals_cents"]["meals"] = True
        values.append(wrong)
        for value in values:
            with self.subTest(value=value):
                (directory / spec["output_path"]).write_text(json.dumps(value))
                self.assertFalse(self.outcome(directory, spec, capture)["passed"])

    def test_wrong_missing_and_inferred_meeting_actions_fail(self):
        directory, spec, capture = self.fixture("meeting-actions")
        for mutate in (
            lambda value: value["actions"].pop(),
            lambda value: value["actions"][1].update(due_date="2026-10-30"),
            lambda value: value["actions"][2].update(owner="Jon"),
            lambda value: value["actions"].append({"task": "redesign dashboard", "owner": "Jon", "due_date": None}),
            lambda value: value.update(decisions=[]),
        ):
            value = deepcopy(MEETING)
            mutate(value)
            (directory / spec["output_path"]).write_text(json.dumps(value))
            self.assertFalse(self.outcome(directory, spec, capture)["artifact_check"]["valid"])

    def test_duplicate_unknown_nonfinite_and_trailing_json_fail(self):
        directory, spec, capture = self.fixture()
        valid = json.dumps(EXPENSES)
        for text in (valid[:-1] + ',"total_cents":31650}', valid[:-1] + ',"extra":0}',
                     valid.replace("31650", "NaN"), valid + "{}", "null", valid.replace("31650", "Infinity")):
            with self.subTest(text=text):
                (directory / spec["output_path"]).write_text(text)
                self.assertFalse(self.outcome(directory, spec, capture)["passed"])

    def test_changed_input_cannot_rebase_expected_hash(self):
        directory, spec, capture = self.fixture()
        (directory / "transactions.csv").write_text("replacement")
        outcome = self.outcome(directory, spec, capture)
        self.assertFalse(outcome["inputs_unchanged"])
        self.assertFalse(outcome["passed"])

    def test_output_symlink_directory_fifo_and_oversize_fail(self):
        directory, spec, capture = self.fixture()
        output = directory / spec["output_path"]
        output.unlink()
        outside = self.root / "outside.json"
        outside.write_text(json.dumps(EXPENSES))
        output.symlink_to(outside)
        self.assertFalse(self.outcome(directory, spec, capture)["artifact_check"]["valid"])
        output.unlink()
        output.mkdir()
        self.assertFalse(self.outcome(directory, spec, capture)["artifact_check"]["valid"])
        output.rmdir()
        import os
        os.mkfifo(output)
        self.assertFalse(self.outcome(directory, spec, capture)["artifact_check"]["valid"])
        output.unlink()
        output.write_bytes(b" " * 8193)
        self.assertFalse(self.outcome(directory, spec, capture)["artifact_check"]["valid"])

    def test_input_symlink_fails_even_for_identical_bytes(self):
        directory, spec, capture = self.fixture()
        original = directory / "transactions.csv"
        outside = self.root / "source.csv"
        original.rename(outside)
        original.symlink_to(outside)
        self.assertFalse(self.outcome(directory, spec, capture)["inputs_unchanged"])

    def test_wrong_paths_duplicate_ids_roles_and_error_results_fail(self):
        directory, spec, capture = self.fixture()
        mutations = (
            lambda records: records[1]["event"]["message"]["content"][0]["input"].update(file_path="wrong.csv"),
            lambda records: records[5]["event"]["message"]["content"][0]["input"].update(file_path="wrong.json"),
            lambda records: records[3]["event"]["message"]["content"][0].update(id="r0"),
            lambda records: records[1]["event"].update(type="user"),
            lambda records: records[2]["event"].update(type="assistant"),
            lambda records: records[1]["event"]["message"].update(role="user"),
            lambda records: records[2]["event"]["message"]["content"][0].update(is_error=True),
            lambda records: records[2]["event"]["message"]["content"][0].update(tool_use_id="unknown"),
            lambda records: records[5]["event"]["message"]["content"][0].update(name="Write"),
        )
        for mutate in mutations:
            current = deepcopy(capture)
            mutate(current["events"])
            self.assertFalse(self.outcome(directory, spec, current)["passed"])

    def test_edit_before_read_completion_missing_and_duplicate_results_fail(self):
        directory, spec, capture = self.fixture()
        reorder = deepcopy(capture)
        reorder["events"] = capture["events"][:1] + capture["events"][5:7] + capture["events"][1:5] + capture["events"][7:]
        missing = deepcopy(capture)
        del missing["events"][2]
        duplicate = deepcopy(capture)
        duplicate["events"].insert(3, result("r0"))
        after_final = deepcopy(capture)
        after_final["events"].extend([call("e1", "Edit", spec["output_path"]), result("e1")])
        for current in (reorder, missing, duplicate, after_final):
            self.assertFalse(self.outcome(directory, spec, current)["tool_cycle_passed"])

    def test_missing_actual_inventory_and_invalid_client_completion_fail(self):
        directory, spec, capture = self.fixture()
        mutations = (
            lambda value: value["events"].pop(0),
            lambda value: value.pop("tool_inventory"),
            lambda value: value["events"][0]["event"].update(tools=["Read"]),
            lambda value: value["events"].pop(),
            lambda value: value["events"][-1]["event"].update(subtype="error_max_turns"),
            lambda value: value["events"][-1]["event"].update(is_error=True),
            lambda value: value.update(exit_code=1),
            lambda value: value.update(timed_out=True),
            lambda value: value.update(cancel_sent=True),
            lambda value: value["diagnostics"].append("Truncated output."),
            lambda value: value["events"].append(deepcopy(value["events"][-1])),
        )
        for mutate in mutations:
            current = deepcopy(capture)
            mutate(current)
            self.assertFalse(self.outcome(directory, spec, current)["passed"])

    def test_existing_defaults_remain_unchanged(self):
        parsed = client.parser().parse_args(["--output-dir", "unused"])
        self.assertEqual(parsed.tasks, ["read", "edit", "check"])


if __name__ == "__main__":
    unittest.main()
