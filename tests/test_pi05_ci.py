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

    def test_identical_nonzero_vectors_pass_exact_equality_budget(self):
        values = np.array([.123, .456, .789])
        exact = {"min_cosine": 1, "max_relative_l2": 0, "max_abs": 0, "zero_max_abs": 0}
        self.assertTrue(parity.meets_limits(parity.metrics(values, values.copy()), exact))

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

    def test_replay_keeps_all_three_source_observations(self):
        paths = []
        for index in range(3):
            path = self.root / f"source-{index}.npz"
            np.savez(path, base=np.full((224, 224, 3), index, np.uint8))
            paths.append(path)
        suite = self.root / "replay"
        args = parity.parse_args(["prepare", "--suite-dir", str(suite), "--image-keys", "base"])
        args.source_npz = paths
        parity.prepare(args)
        manifest = parity.load_suite(suite)
        values = {int(parity.load_case(suite, case)["images"].flat[0])
                  for case in manifest["cases"] if case["name"] in ("replay-observation-0", "replay-observation-1", "replay-observation-2")}
        self.assertEqual(values, {0, 1, 2})

    def test_float_conversion_is_cpu_coverage(self):
        image = np.arange(224 * 224 * 3, dtype=np.uint32).reshape(224, 224, 3).astype(np.uint8)
        chw = np.moveaxis(image.astype(np.float32) / 255, -1, 0)
        np.testing.assert_array_equal(parity.image_to_uint8_hwc(chw), image)
        with self.assertRaises(TypeError):
            parity.image_to_uint8_hwc(image.astype(np.int32))

    def test_accuracy_keeps_first_output_and_stability_is_separate(self):
        class ChangingModel:
            calls = 0
            def infer(self, case):
                self.calls += 1
                return np.full((2, 32), self.calls, np.float32)
        model = ChangingModel()
        manifest = parity.load_suite(self.root)
        parity.collect(self.root, manifest, model, "fake", {}, force=False, stability_repeats=2)
        result = json.loads((self.root / "fake.json").read_text())
        self.assertEqual(result["cases"][0]["actions"][0][0], 1)
        self.assertNotIn("latency_ms", result["cases"][0])
        self.assertEqual(result["stability"]["repeats"], 2)
        self.assertEqual(result["stability"]["cases"][0]["repeat_max_abs"], 2)

    def test_revisit_detects_drift_separately_from_same_input_repeats(self):
        second = parity.save_case(self.root, "two", np.zeros((1, 224, 224, 3), np.uint8),
                                  (3, 108), np.zeros((2, 32), np.float32))
        manifest = parity.load_suite(self.root)
        manifest["cases"].append(second)
        parity.write_json(self.root / "manifest.json", manifest, force=True)
        class LeakyModel:
            other_calls = 0
            def infer(self, case):
                if case["token_ids"][0] == 3:
                    self.other_calls += 1
                value = 2 if self.other_calls >= 3 else 1
                return np.full((2, 32), value, np.float32)
        parity.collect(self.root, manifest, LeakyModel(), "fake", {},
                       force=False, stability_repeats=1)
        result = json.loads((self.root / "fake.json").read_text())
        self.assertEqual([r["repeat_max_abs"] for r in result["stability"]["cases"]], [0, 0])
        self.assertEqual(result["stability"]["revisit_max_abs"], 1)
        self.assertEqual(result["cases"][0]["actions"][0][0], 1)

    def test_diagnostics_separate_factors_and_cover_configured_token_boundary(self):
        suite = self.root / "diagnostic"
        args = parity.parse_args(["prepare", "--suite-dir", str(suite), "--diagnostic",
                                  "--image-keys", "base,wrist", "--max-token-len", "200"])
        parity.prepare(args)
        cases = {e["name"]: parity.load_case(suite, e) for e in parity.load_suite(suite)["cases"]}
        self.assertFalse(any("float-chw" in name or "typical" in name for name in cases))
        base = cases["gradient-t10-normal-noise"]
        np.testing.assert_array_equal(cases["black-normal-noise"]["noise"], base["noise"])
        np.testing.assert_array_equal(cases["gradient-zero-noise"]["images"], base["images"])
        self.assertEqual(len(cases["gradient-t200-boundary"]["token_ids"]), 200)


class MatrixTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bank = {"schema": ci.SCHEMA, "cells": [
            {"id": f"{board}-{precision}-{views}", "hardware": board, "precision": precision,
             "views": views, "stability_repeats": 2,
             "performance_protocol": {"warmup": 1, "samples": 2}, "budget_reason": "test fixture only",
             "reference_limits": LIMITS, "baseline_limits": LIMITS,
             "performance_limits": {"p50_ms": 12, "p95_ms": 12, "first_call_ms": 30}}
            for board, precisions in ci.PRECISIONS.items() for precision in precisions for views in (1, 2, 3)]}
        for cell in self.bank["cells"]:
            self.make_evidence(cell)
        self.path = self.root / "bank.json"
        self.save_bank()
        self.reports = []
        for board in ci.PRECISIONS:
            path = self.root / board / "summary.json"
            parity.write_json(path, {"schema": ci.SCHEMA, "revision": SHA, "hardware": board,
                                    "bank_sha256": self.digest, "passed": True,
                                    "cells": [{"id": cell["id"], "passed": True}
                                              for cell in self.bank["cells"] if cell["hardware"] == board]})
            self.reports.append(path)

    def make_evidence(self, cell):
        output = self.root / cell["hardware"]
        suite = output / "evidence" / cell["id"]
        entry = parity.save_case(suite, "one", np.zeros((cell["views"], 224, 224, 3), np.uint8),
                                 (2, 108), np.zeros((2, 32), np.float32))
        parity.write_json(suite / "manifest.json", {
            "schema": parity.SCHEMA, "cases": [entry], "image_keys": list(range(cell["views"])),
            "horizon": 2, "num_flow_steps": 10,
        })
        receipt = {"schema": parity.SCHEMA, "suite_sha256": parity.sha256(suite / "manifest.json"),
                   "engine": "apxinf", "revision": SHA, "precision": cell["precision"],
                   "hardware": cell["hardware"], "calibration_sha256": None,
                   "checkpoint_sha256": "b" * 64, "config_sha256": "c" * 64,
                   "extension_sha256": "d" * 64,
                   "stability": {"repeats": 2, "revisit_max_abs": 0,
                                 "cases": [{"name": "one", "repeat_max_abs": 0}]},
                   "cases": [{"name": "one", "input_sha256": entry["sha256"],
                              "actions": np.ones((2, 32)).tolist()}]}
        parity.write_json(output / f"{cell['id']}.json", receipt)
        parity.write_json(output / f"{cell['id']}.performance.json", dict(receipt,
            schema="apxinf.pi05.performance.v1", layer="l1", warmup=1, samples=2,
            cases=[{"name": "one", "input_sha256": entry["sha256"],
                    "first_call_ms": 20, "latency_ms": [10, 11]}]))
        parity.write_json(suite / "baseline.json", receipt)
        parity.write_json(suite / "reference.json", dict(receipt, engine="openpi"))
        for key, name in (("suite", "manifest.json"), ("reference", "reference.json"), ("baseline", "baseline.json")):
            cell[key] = {"path": str(suite / name), "sha256": parity.sha256(suite / name)}
        cell["checkpoint"] = {"path": "unused", "sha256": "b" * 64}
        cell["config"] = {"path": "unused", "sha256": "c" * 64}

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

    def test_raw_error_cannot_be_overridden_by_green_summary(self):
        cell = self.bank["cells"][0]
        path = self.root / cell["hardware"] / f"{cell['id']}.json"
        actual = ci.read_json(path)
        actual["cases"][0]["actions"][0][0] = 2
        parity.write_json(path, actual, force=True)
        self.assertEqual(ci.aggregate(self.args()), 1)

    def test_performance_protocol_and_raw_latency_cannot_be_forged(self):
        cell = self.bank["cells"][0]
        path = self.root / cell["hardware"] / f"{cell['id']}.performance.json"
        original = ci.read_json(path)
        changed = copy.deepcopy(original)
        changed["warmup"] = 2
        parity.write_json(path, changed, force=True)
        with self.assertRaises(ValueError):
            ci.aggregate(self.args())
        changed = copy.deepcopy(original)
        changed["cases"][0]["latency_ms"] = [100, 100]
        parity.write_json(path, changed, force=True)
        self.assertEqual(ci.aggregate(self.args()), 1)

    def test_uncalibrated_performance_is_not_green(self):
        rows = [{"name": "case", "latency_ms": [10, 11], "first_call_ms": 20}]
        self.assertEqual(ci.performance(rows, {})["state"], "uncalibrated")
        budgets = {"performance_limits": {"p50_ms": 9.9, "p95_ms": 12, "first_call_ms": 30}}
        self.assertEqual(ci.performance(rows, budgets)["state"], "fail")


if __name__ == "__main__":
    unittest.main()
