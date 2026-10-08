"""Fail-closed PI05 receipts and numerical regression tests; no GPU/OpenPI needed."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_pi05_openpi as parity
import pi05_ci as ci


LIMITS = {"min_cosine": .999, "max_relative_l2": .05, "max_abs": .01, "zero_max_abs": .0001}
SHA = "a" * 40


class MetricsTest(unittest.TestCase):
    def test_cosine_does_not_detect_scale_but_gate_does(self):
        values = np.ones((2, 32))
        result = parity.metrics(values, values * 2)
        self.assertAlmostEqual(result["cosine"], 1)
        self.assertFalse(parity.meets_limits(result, LIMITS))

    def test_zero_is_absolute_only(self):
        values = np.zeros((2, 32))
        self.assertIsNone(parity.metrics(values, values)["cosine"])
        self.assertTrue(parity.meets_limits(parity.metrics(values, values), LIMITS))
        self.assertFalse(parity.meets_limits(parity.metrics(values, values + .001), LIMITS))

    def test_invalid_outputs(self):
        for actual in (np.array([np.nan]), np.array([np.inf]), np.zeros(2)):
            with self.assertRaises(ValueError):
                parity.metrics(np.ones(1), actual)


class ReceiptTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        entry = parity.save_case(self.root, "one", np.zeros((1, 224, 224, 3), np.uint8),
                                 (2, 108), np.zeros((2, 32), np.float32))
        parity.write_json(self.root / "manifest.json", {
            "schema": parity.SCHEMA, "cases": [entry], "image_keys": ["base"],
            "horizon": 2, "num_flow_steps": 10,
        })
        self.receipt = {"schema": parity.SCHEMA, "suite_sha256": parity.sha256(self.root / "manifest.json"),
                        "checkpoint_sha256": "b" * 64, "cases": [{"name": "one",
                        "input_sha256": entry["sha256"], "actions": np.ones((2, 32)).tolist()}]}

    def compare(self, actual):
        return parity.compare_results(self.root, self.receipt, actual, LIMITS)

    def test_valid_and_single_step_error(self):
        self.assertTrue(self.compare(self.receipt)[0]["passed"])
        actual = copy.deepcopy(self.receipt)
        actual["cases"][0]["actions"][1][0] += .02
        self.assertFalse(self.compare(actual)[0]["passed"])

    def test_changed_receipts(self):
        for key in ("schema", "suite_sha256", "checkpoint_sha256"):
            actual = copy.deepcopy(self.receipt)
            actual[key] = "changed"
            with self.assertRaises(ValueError):
                self.compare(actual)
        for actual_cases in ([], self.receipt["cases"] * 2):
            actual = dict(self.receipt, cases=actual_cases)
            with self.assertRaises(ValueError):
                self.compare(actual)

    def test_input_tampering(self):
        (self.root / "cases/one.npz").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.compare(self.receipt)

    def test_input_path_escape(self):
        manifest = json.loads((self.root / "manifest.json").read_text())
        manifest["cases"][0]["path"] = "../outside.npz"
        parity.write_json(self.root / "manifest.json", manifest, force=True)
        with self.assertRaises(ValueError):
            parity.load_suite(self.root)


class MatrixTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bank = {"schema": ci.SCHEMA, "cells": [
            {"id": f"{board}-{precision}-{views}", "hardware": board, "precision": precision,
             "views": views, "samples": 2, "budget_reason": "test fixture only",
             "reference_limits": LIMITS, "baseline_limits": LIMITS}
            for board, precisions in ci.PRECISIONS.items() for precision in precisions for views in (1, 2, 3)]}
        self.path = self.root / "bank.json"
        self.save_bank()
        self.reports = []
        for board in ci.PRECISIONS:
            path = self.root / f"{board}.json"
            parity.write_json(path, {"schema": ci.SCHEMA, "revision": SHA, "hardware": board,
                                    "bank_sha256": self.digest, "passed": True,
                                    "cells": [{"id": cell["id"], "passed": True}
                                              for cell in self.bank["cells"] if cell["hardware"] == board]})
            self.reports.append(path)

    def save_bank(self):
        parity.write_json(self.path, self.bank, force=True)
        self.digest = parity.sha256(self.path)

    def args(self, reports=None):
        return SimpleNamespace(bank=self.path, bank_sha256=self.digest, revision=SHA,
                               report=self.reports if reports is None else reports)

    def test_complete_matrix(self):
        self.assertEqual(ci.aggregate(self.args()), 0)

    def test_missing_bank_cell(self):
        self.bank["cells"].pop()
        self.save_bank()
        with self.assertRaises(ValueError):
            ci.load_bank(self.path, self.digest)

    def test_missing_or_duplicate_board(self):
        for reports in ([self.reports[0]], [self.reports[0], self.reports[0]]):
            with self.assertRaises(ValueError):
                ci.aggregate(self.args(reports))

    def test_stale_or_partial_report(self):
        for change in (lambda report: report.update(revision="c" * 40),
                       lambda report: report["cells"].pop()):
            original = ci.read_json(self.reports[0])
            changed = copy.deepcopy(original)
            change(changed)
            parity.write_json(self.reports[0], changed, force=True)
            with self.assertRaises(ValueError):
                ci.aggregate(self.args())
            parity.write_json(self.reports[0], original, force=True)

    def test_failed_cell_cannot_be_overridden_by_top_level_pass(self):
        report = ci.read_json(self.reports[0])
        report["cells"][0]["passed"] = False
        parity.write_json(self.reports[0], report, force=True)
        self.assertEqual(ci.aggregate(self.args()), 1)

    def test_uncalibrated_performance_is_not_green(self):
        rows = [{"name": "case", "latency_ms": [10, 11], "first_call_ms": 20}]
        self.assertEqual(ci.performance(rows, {})["state"], "uncalibrated")
        budgets = {"performance_limits": {"p50_ms": 10, "p95_ms": 12, "first_call_ms": 30}}
        self.assertEqual(ci.performance(rows, budgets)["state"], "fail")


if __name__ == "__main__":
    unittest.main()
