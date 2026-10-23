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


if __name__ == "__main__":
    unittest.main()
