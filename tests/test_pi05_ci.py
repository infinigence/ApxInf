"""Fail-closed PI05 receipts and numerical regression tests; no GPU/OpenPI needed."""

import copy
import json
import hashlib
import os
import time
from unittest.mock import patch
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_pi05_openpi as parity
import pi05_ci as ci
import monitor_pi05_ci as activity
import bench_pi05 as bench


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

    def test_zero_candidate_cannot_bypass_nonzero_reference_relative_error(self):
        expected = np.full((2, 32), 1e-6)
        result = parity.metrics(expected, np.zeros_like(expected))
        self.assertIsNone(result["cosine"])
        self.assertEqual(result["relative_l2"], 1)
        self.assertFalse(parity.meets_limits(result, LIMITS))

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

    def test_native_loader_forwards_autotune(self):
        (self.root / "config.json").write_text("{}")
        options = []
        runner = SimpleNamespace(load=lambda *args, **kwargs: options.append(kwargs))
        with patch.dict(sys.modules, {"apxinf_py": SimpleNamespace(ModelRunner=runner)}):
            for enabled in (False, True):
                parity.ApxInfModel(self.root, parity.load_suite(self.root), "cuda:0", "bf16", None,
                                  autotune=enabled)
        self.assertEqual([entry["autotune"] for entry in options], [False, True])

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
        for name in ("gradient-t200-zero-noise-combined", "black-t200-zero-noise-combined"):
            np.testing.assert_array_equal(cases[name]["token_ids"], cases["gradient-t200-boundary"]["token_ids"])
            np.testing.assert_array_equal(cases[name]["noise"], np.zeros_like(base["noise"]))
        np.testing.assert_array_equal(cases["gradient-t200-zero-noise-combined"]["images"], base["images"])
        self.assertFalse(cases["black-t200-zero-noise-combined"]["images"].any())


class ActivityTest(unittest.TestCase):
    def test_thor_audit_includes_graphics_gpu_clients(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            for pid, name in ((101, "python"), (202, "Xorg")):
                process = proc / str(pid)
                process.mkdir()
                (process / "stat").write_text(f"{pid} ({name}) S 1 {pid} 1")
                (process / "comm").write_text(name)
            xml = "<nvidia_smi_log><gpu><processes>" + "".join(
                f"<process_info><pid>{pid}</pid><type>{kind}</type></process_info>"
                for pid, kind in ((101, "C"), (202, "G"))) + "</processes></gpu></nvidia_smi_log>"
            def query(command, **kwargs):
                return "101\n" if "--query-compute-apps=pid" in command else xml
            with patch.object(Path, "read_bytes", return_value=b"nvidia,tegra264"), patch.object(
                    activity.subprocess, "check_output", side_effect=query):
                receipt = activity.audit(proc)
            self.assertEqual({p["pid"] for p in receipt["processes"]}, {101, 202})

    def test_thor_audit_rejects_unavailable_process_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            for xml in ("<nvidia_smi_log/>", "<nvidia_smi_log><gpu/></nvidia_smi_log>",
                    "<nvidia_smi_log><gpu><processes>N/A</processes></gpu></nvidia_smi_log>"):
                with self.subTest(xml=xml), patch.object(Path, "read_bytes", return_value=b"tegra264"), patch.object(
                        activity.subprocess, "check_output", return_value=xml):
                    with self.assertRaisesRegex(ValueError, "inventory unavailable"):
                        activity.audit(Path(directory))

    def test_orin_audit_detects_current_driver_gpu_descriptors(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            process = proc / "123"
            (process / "fd").mkdir(parents=True)
            (process / "stat").write_text("123 (python) S 1 123 1")
            (process / "comm").write_text("python")
            for target, expected in (("/dev/nvhost-gpu", True),
                    ("/dev/nvgpu/igpu0/ctrl", True),
                    ("anon_inode:nvgpu-ga10b-tsg18", True),
                    ("anon_inode:nvhost-17000000.gpu-fd19", True),
                    ("anon_inode:[eventfd]", False), ("/dev/nvmap", False)):
                with self.subTest(target=target):
                    fd = process / "fd/10"
                    fd.symlink_to(target)
                    with patch.object(Path, "read_bytes", return_value=b"nvidia,tegra234"):
                        receipt = activity.audit(proc)
                    fd.unlink()
                    self.assertEqual(bool(receipt["processes"]), expected)

    def test_monitor_rejects_interference_and_terminates_own_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            auditor = root / "auditor"
            auditor.write_text("fixture executable")
            args = SimpleNamespace(output=root / "activity.json",
                command=[sys.executable, "-c", "import time; time.sleep(60)"])
            calls = 0
            def sample(*args, **kwargs):
                nonlocal calls
                calls += 1
                processes = [] if calls == 1 else [{"pid": 987654, "group": 987654,
                    "name": "python", "gpu": True, "compiler": False}]
                return json.dumps({"time": time.time(), "processes": processes})
            environment = {"APXINF_CI_ACTIVITY_AUDIT": str(auditor),
                "APXINF_CI_ACTIVITY_AUDIT_SHA256": hashlib.sha256(auditor.read_bytes()).hexdigest()}
            with patch.dict(os.environ, environment), patch.object(activity.subprocess, "check_output", side_effect=sample):
                self.assertEqual(activity.monitor(args), 1)
            receipt = ci.read_json(args.output)
            self.assertFalse(receipt["passed"])
            self.assertIn("foreign GPU", receipt["error"])
            with self.assertRaises(ProcessLookupError):
                os.kill(receipt["group"], 0)

    @unittest.skipUnless(os.name == "posix", "CI workers use POSIX process groups")
    def test_monitor_cancellation_terminates_own_process_and_saves_failure(self):
        self._monitor_cancellation(False)

    @unittest.skipUnless(os.name == "posix", "CI workers use POSIX process groups")
    def test_repeated_cancellation_still_kills_unresponsive_process(self):
        self._monitor_cancellation(True)

    @unittest.skipUnless(os.name == "posix", "CI workers use POSIX process groups")
    def test_cancellation_interrupts_blocked_activity_audit(self):
        self._monitor_cancellation(False, block_audit=True)

    def _monitor_cancellation(self, ignore_term, block_audit=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            auditor = root / "auditor"
            audited, audit_ready = root / "audited", root / "audit-ready"
            slow_sample = (
                f"from pathlib import Path\nimport os\np = Path({str(audited)!r})\n"
                f"if p.exists():\n Path({str(audit_ready)!r}).write_text(str(os.getpid())); time.sleep(60)\n"
                "else: p.touch()\n"
            ) if block_audit else ""
            auditor.write_text(f"#!{sys.executable}\nimport json, time\n" + slow_sample +
                "print(json.dumps({'time': time.time(), 'processes': []}))\n")
            auditor.chmod(0o755)
            ready, stopped, output = (root / name for name in ("ready", "stopped", "activity.json"))
            handler = ("signal.SIG_IGN" if ignore_term else
                f"lambda *_: (Path({str(stopped)!r}).touch(), sys.exit(0))")
            command = ("import os, signal, sys, time; from pathlib import Path; "
                f"signal.signal(signal.SIGTERM, {handler}); "
                f"Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(60)")
            environment = {**os.environ, "APXINF_CI_ACTIVITY_AUDIT": str(auditor),
                "APXINF_CI_ACTIVITY_AUDIT_SHA256": hashlib.sha256(auditor.read_bytes()).hexdigest()}
            monitor = activity.subprocess.Popen([sys.executable, activity.__file__, "monitor",
                "--output", str(output), "--", sys.executable, "-c", command], env=environment)
            try:
                deadline = time.monotonic() + 10
                while not ready.exists() and monitor.poll() is None and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(ready.exists(), "monitored child did not start")
                if block_audit:
                    while not audit_ready.exists() and time.monotonic() < deadline:
                        time.sleep(.01)
                    self.assertTrue(audit_ready.exists(), "activity auditor did not block")
                monitor.terminate()
                if ignore_term:
                    time.sleep(.1)
                    monitor.send_signal(activity.signal.SIGINT)
                self.assertNotEqual(monitor.wait(timeout=8 if block_audit else 15), 0)
                with self.assertRaises(ProcessLookupError):
                    os.kill(int(ready.read_text()), 0)
                if not ignore_term:
                    self.assertTrue(stopped.exists(), "child did not receive termination")
                receipt = ci.read_json(output)
                self.assertFalse(receipt["passed"])
                self.assertIn("signal", receipt["error"])
            finally:
                if monitor.poll() is None:
                    monitor.kill()
                monitor.wait()
                if audit_ready.exists():
                    try:
                        os.kill(int(audit_ready.read_text()), activity.signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if ready.exists():
                    try:
                        os.killpg(int(ready.read_text()), activity.signal.SIGKILL)
                    except ProcessLookupError:
                        pass


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
             "shape_switch_limit_ms": 30, "stability_max_abs": 0,
             "performance_limits": {name: {"p50_ms": 12, "p95_ms": 12, "first_call_ms": 30} for name in ("one", "other")}}
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
            parity.write_json(path.parent / "activity.json", {"schema": activity.SCHEMA,
                "passed": True, "auditor_sha256": "f" * 64, "start": 1, "finish": 2, "group": 123,
                "samples": [{"time": 1, "processes": []}, {"time": 2, "processes": []}]})

    def make_evidence(self, cell):
        output = self.root / cell["hardware"]
        suite = output / "evidence" / cell["id"]
        entry = parity.save_case(suite, "one", np.zeros((cell["views"], 224, 224, 3), np.uint8),
                                 (2, 108), np.zeros((2, 32), np.float32))
        other = parity.save_case(suite, "other", np.zeros((cell["views"], 224, 224, 3), np.uint8),
                                  (2, 108, 3), np.zeros((2, 32), np.float32))
        entries = [entry, other]
        parity.write_json(suite / "manifest.json", {
            "schema": parity.SCHEMA, "cases": entries, "image_keys": list(range(cell["views"])),
            "horizon": 2, "num_flow_steps": 10,
        })
        receipt = {"schema": parity.SCHEMA, "suite_sha256": parity.sha256(suite / "manifest.json"),
                   "engine": "apxinf", "revision": SHA, "precision": cell["precision"], "autotune": False,
                   "hardware": cell["hardware"], "calibration_sha256": None,
                   "checkpoint_sha256": "b" * 64, "config_sha256": "c" * 64,
                   "extension_sha256": "d" * 64,
                   "runtime_libraries": {name: {"path": f"/fixture/{name}.so", "sha256": "e" * 64}
                                         for name in ("libcuda", "libcudart", "libcublas", "libcublasLt")},
                   "stability": {"repeats": 2, "revisit_max_abs": 0,
                                 "cases": [{"name": item["name"], "repeat_max_abs": 0} for item in entries]},
                   "cases": [{"name": item["name"], "input_sha256": item["sha256"],
                              "actions": np.ones((2, 32)).tolist()} for item in entries]}
        parity.write_json(output / f"{cell['id']}.json", receipt)
        parity.write_json(output / f"{cell['id']}.performance.json", dict(receipt,
            schema="apxinf.pi05.performance.v2", layer="l1", warmup=1, samples=2,
            cases=[{"name": item["name"], "input_sha256": item["sha256"],
                    "first_call_ms": 20, "latency_ms": [10, 11]} for item in entries],
            shape_switch={"from": "other", "from_input_sha256": other["sha256"],
                          "to": "one", "to_input_sha256": entry["sha256"], "latency_ms": 25}))
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
                               activity_audit_sha256="f" * 64,
                               report=self.reports if reports is None else reports)

    def test_uncalibrated_bank_is_rejected_before_build(self):
        self.bank["cells"][0]["performance_limits"] = None
        self.save_bank()
        args = self.args()
        args.hardware = "thor"
        with self.assertRaisesRegex(ValueError, "uncalibrated"):
            ci.validate(args)

    def test_frozen_assets_must_be_qualified_before_build(self):
        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "model.safetensors").write_bytes(b"fixture weights")
        (checkpoint / "config.json").write_text("{}")
        for cell in self.bank["cells"]:
            for key, filename in (("checkpoint", "model.safetensors"), ("config", "config.json")):
                path = checkpoint / filename
                cell[key] = {"path": str(path), "sha256": parity.sha256(path)}
            for key in ("reference", "baseline"):
                path = Path(cell[key]["path"])
                receipt = ci.read_json(path)
                receipt.update(checkpoint_sha256=cell["checkpoint"]["sha256"], config_sha256=cell["config"]["sha256"])
                parity.write_json(path, receipt, force=True)
                cell[key]["sha256"] = parity.sha256(path)
        self.save_bank()
        args = self.args()
        args.hardware = "thor"
        self.assertEqual(ci.validate(args), 0)
        cell = self.bank["cells"][0]
        path = Path(cell["baseline"]["path"])
        receipt = ci.read_json(path)
        receipt["cases"][0]["actions"][0][0] = 2
        parity.write_json(path, receipt, force=True)
        cell["baseline"]["sha256"] = parity.sha256(path)
        self.save_bank()
        args.bank_sha256 = self.digest
        with self.assertRaisesRegex(ValueError, "baseline fails official"):
            ci.validate(args)
        receipt["hardware"] = "orin"
        parity.write_json(path, receipt, force=True)
        cell["baseline"]["sha256"] = parity.sha256(path)
        self.save_bank()
        args.bank_sha256 = self.digest
        with self.assertRaisesRegex(ValueError, "hardware differs"):
            ci.validate(args)

    def test_complete_matrix(self):
        self.assertEqual(ci.aggregate(self.args()), 0)

    def test_environment_interference_cannot_pass_green_summary(self):
        path = self.reports[0].parent / "activity.json"
        receipt = ci.read_json(path)
        receipt["samples"][1]["processes"] = [{"pid": 456, "group": 456,
            "name": "python", "gpu": True, "compiler": False}]
        parity.write_json(path, receipt, force=True)
        with self.assertRaisesRegex(ValueError, "foreign GPU"):
            ci.aggregate(self.args())

    def test_missing_activity_coverage_is_rejected(self):
        receipt = {"schema": activity.SCHEMA, "passed": True, "start": 1,
                   "finish": 2, "group": 123,
                   "samples": [{"time": 1, "processes": []}]}
        with self.assertRaisesRegex(ValueError, "cover"):
            activity.validate_receipt(receipt)

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

    def test_invalid_or_missing_stability_budget(self):
        cell = self.bank["cells"][0]
        for value in (-1, float("inf")):
            cell["stability_max_abs"] = value
            self.path.write_text(json.dumps(self.bank))
            self.digest = parity.sha256(self.path)
            with self.assertRaisesRegex(ValueError, "stability budget"):
                ci.load_bank(self.path, self.digest)
        cell["stability_max_abs"] = None
        self.save_bank()
        args = self.args()
        args.hardware = "thor"
        with self.assertRaisesRegex(ValueError, "uncalibrated stability"):
            ci.validate(args)

    def test_stability_uses_its_own_budget(self):
        cell = self.bank["cells"][0]
        path = self.root / cell["hardware"] / f"{cell['id']}.json"
        actual = ci.read_json(path)
        actual["stability"]["cases"][0]["repeat_max_abs"] = .005
        parity.write_json(path, actual, force=True)
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

    def test_daily_gate_rejects_tuning_during_measurement(self):
        cell = self.bank["cells"][0]
        suite = Path(cell["suite"]["path"]).parent
        reference = ci.read_json(suite / "reference.json")
        baseline = ci.read_json(suite / "baseline.json")
        candidate = ci.read_json(self.root / "thor" / f"{cell['id']}.json")
        latency = ci.read_json(self.root / "thor" / f"{cell['id']}.performance.json")
        for record in (baseline, candidate, latency):
            for invalid in (True, None):
                record["autotune"] = invalid
                with self.assertRaisesRegex(ValueError, "autotune must be disabled"):
                    ci.evaluate(cell, suite, reference, baseline, candidate, SHA, latency)
            record["autotune"] = False
        self.assertTrue(ci.evaluate(cell, suite, reference, baseline, candidate, SHA, latency)["passed"])

    def test_runtime_drift_and_missing_provenance_cannot_pass(self):
        cell = self.bank["cells"][0]
        suite = Path(cell["suite"]["path"]).parent
        candidate = json.loads((self.root / cell["hardware"] / f"{cell['id']}.json").read_text())
        reference = json.loads((suite / "reference.json").read_text())
        baseline = json.loads((suite / "baseline.json").read_text())
        latency = json.loads((self.root / cell["hardware"] / f"{cell['id']}.performance.json").read_text())
        latency["runtime_libraries"]["libcublas"]["path"] = "/relocated/libcublas.so"
        self.assertTrue(ci.evaluate(cell, suite, reference, baseline, candidate, SHA, latency)["passed"])
        latency["runtime_libraries"]["libcublas"]["sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "performance CUDA runtime"):
            ci.evaluate(cell, suite, reference, baseline, candidate, SHA, latency)
        latency["runtime_libraries"] = candidate["runtime_libraries"]
        baseline["runtime_libraries"]["libcublas"]["sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "approved baseline"):
            ci.evaluate(cell, suite, reference, baseline, candidate, SHA, latency)
        baseline["runtime_libraries"] = candidate["runtime_libraries"]
        reference["hardware"] = "orin" if cell["hardware"] == "thor" else "thor"
        with self.assertRaisesRegex(ValueError, "reference hardware"):
            ci.evaluate(cell, suite, reference, baseline, candidate, SHA, latency)
        reference["hardware"] = cell["hardware"]
        reference["runtime_libraries"]["libcublas"]["sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "official reference CUDA runtime"):
            ci.evaluate(cell, suite, reference, baseline, candidate, SHA, latency)
        reference.pop("runtime_libraries")
        with self.assertRaisesRegex(ValueError, "provenance missing"):
            ci.evaluate(cell, suite, reference, baseline, candidate, SHA, latency)

    def test_fast_case_regression_cannot_hide_under_slow_case_budget(self):
        rows = [{"name": "fast", "latency_ms": [15, 15], "first_call_ms": 20},
                {"name": "slow", "latency_ms": [90, 90], "first_call_ms": 100}]
        budgets = {"performance_limits": {
            "fast": {"p50_ms": 12, "p95_ms": 12, "first_call_ms": 30},
            "slow": {"p50_ms": 100, "p95_ms": 100, "first_call_ms": 120}}}
        self.assertEqual(ci.performance(rows, budgets)["state"], "fail")
        budgets["performance_limits"].pop("fast")
        with self.assertRaisesRegex(ValueError, "exactly"):
            ci.performance(rows, budgets)

    def test_benchmark_revisits_prior_shape_separately_from_warm_samples(self):
        cell = self.bank["cells"][0]
        suite = Path(cell["suite"]["path"]).parent
        extension = self.root / "extension.so"
        extension.write_bytes(b"fixture")
        (self.root / "model.safetensors").write_bytes(b"fixture weights")
        (self.root / "config.json").write_text("{}")
        calls, options = [], []
        class Model:
            def __init__(self, *args, **kwargs):
                options.append(kwargs)
            def infer(self, case):
                calls.append(len(case["token_ids"]))
                return np.ones((2, 32), np.float32)
        args = SimpleNamespace(model_dir=self.root, layer="l1", out=self.root / "bench.json",
            suite_dir=suite, warmup=1, samples=2, model_variant="bf16", calibration=None,
            device="cuda:0", revision=SHA, hardware="thor", autotune=True)
        with patch.dict(sys.modules, {"apxinf_py": SimpleNamespace(__file__=str(extension))}), patch.object(
                parity, "ApxInfModel", Model), patch.object(parity, "cuda_runtime_libraries", return_value={}):
            bench.benchmark_suite(args)
        self.assertEqual(options, [{"autotune": True}])
        self.assertEqual(calls, [2, 2, 2, 2, 3, 3, 3, 3, 3, 2])
        receipt = ci.read_json(args.out)
        self.assertEqual(receipt["shape_switch"]["from"], "other")
        self.assertEqual(receipt["shape_switch"]["to"], "one")
        self.assertEqual([len(row["latency_ms"]) for row in receipt["cases"]], [2, 2])

    def test_shape_switch_regression_and_forged_route_fail(self):
        path = self.reports[0].parent / f"{self.bank['cells'][0]['id']}.performance.json"
        receipt = ci.read_json(path)
        receipt["shape_switch"]["latency_ms"] = 31
        parity.write_json(path, receipt, force=True)
        self.assertEqual(ci.aggregate(self.args()), 1)
        receipt["shape_switch"]["latency_ms"] = 25
        receipt["shape_switch"]["from_input_sha256"] = "0" * 64
        parity.write_json(path, receipt, force=True)
        with self.assertRaisesRegex(ValueError, "shape-switch input"):
            ci.aggregate(self.args())

    def test_uncalibrated_performance_is_not_green(self):
        rows = [{"name": "case", "latency_ms": [10, 11], "first_call_ms": 20}]
        self.assertEqual(ci.performance(rows, {})["state"], "uncalibrated")
        budgets = {"performance_limits": {"case": {"p50_ms": 9.9, "p95_ms": 12, "first_call_ms": 30}}}
        self.assertEqual(ci.performance(rows, budgets)["state"], "fail")


if __name__ == "__main__":
    unittest.main()
