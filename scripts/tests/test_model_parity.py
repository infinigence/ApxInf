"""Consumer capture and failure gates, including historical failure patterns.

Small arrays isolate tool behavior; historical device evidence is replayed
separately and is not represented as a new GPU run by this suite.
"""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from scripts import model_parity as parity


REPO = Path(__file__).resolve().parents[2]


class ParityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.artifact = self.root / "native.so"
        self.artifact.write_bytes(b"synthetic artifact for CPU contract tests")
        self.plan = self.root / "contract.json"
        self.case = {"id": "a", "input_names": ["pixels", "noise"],
                     "grid": [[1, 18, 18]], "output_shape": [2, 3],
                     "mode": "eager", "domain": "normalized"}
        self.contract = {"version": 1, "comparison": {"kind": "byte"}, "cases": [self.case]}
        self.save_contract()
        self.inputs = {"pixels": np.zeros((3, 3), dtype=np.uint8),
                       "noise": np.arange(6, dtype=np.float32).reshape(2, 3)}
        self.outputs = np.arange(6, dtype=np.float32).reshape(2, 3)

    def save_contract(self):
        self.plan.write_text(json.dumps(self.contract))

    def capture(self, side, name="a", inputs=None, output=None, **kwargs):
        capture = parity.Capture(self.root / side / name, self.plan, name,
                                 self.inputs if inputs is None else inputs, self.artifact)
        values = dict(effective_grid=self.case["grid"], actual_mode="eager",
                      output_domain="normalized", board="CPU test", arch="cpu",
                      checkpoint="fixture", entry_point="test consumer")
        values.update(kwargs)
        return capture.finish(self.outputs if output is None else output, **values)

    def compare(self):
        return parity.compare(self.plan, self.root / "baseline", self.root / "candidate")

    def test_matching_tensors_do_not_certify_source(self):
        self.capture("baseline")
        self.capture("candidate")
        result = self.compare()
        self.assertEqual(result["checks"][0]["status"], "PASS")
        self.assertEqual(result["status"], "UNVERIFIED")
        command = [sys.executable, str(REPO / "scripts/model_parity.py"), "compare",
                   str(self.plan), str(self.root / "baseline"), str(self.root / "candidate")]
        run = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
        self.assertEqual(json.loads(run.stdout)["status"], "UNVERIFIED")

    def test_changed_noise_or_dtype_fails_despite_equal_outputs(self):
        for change in (lambda n: n + 1, lambda n: n.astype(np.float64)):
            with self.subTest(change=change):
                inputs = dict(self.inputs, noise=change(self.inputs["noise"]))
                side = "c" + str(inputs["noise"].dtype) + str(inputs["noise"][0, 0])
                if not (self.root / "baseline").exists():
                    self.capture("baseline")
                self.capture(side, inputs=inputs)
                result = parity.compare(self.plan, self.root / "baseline", self.root / side)
                self.assertEqual(result["status"], "FAIL")
                self.assertIn("noise", result["checks"][0]["reason"])

    def test_full_output_difference_and_shape_mismatch(self):
        self.capture("baseline")
        changed = self.outputs.copy()
        changed[0, 0] += 1
        self.capture("candidate", output=changed)
        self.assertEqual(self.compare()["status"], "FAIL")
        self.capture("wrongshape", output=self.outputs.reshape(3, 2))
        result = parity.compare(self.plan, self.root / "baseline", self.root / "wrongshape")
        self.assertEqual(result["status"], "FAIL")

    def test_effective_shape_and_fallback_not_requested_labels(self):
        self.case["grid"] = [[1, 32, 32]]
        self.case["mode"] = "graph"
        self.save_contract()
        self.capture("baseline", actual_mode="graph")
        self.capture("candidate", effective_grid=[[1, 18, 18]])
        self.assertEqual(self.compare()["status"], "FAIL")
        self.capture("fallback")
        result = parity.compare(self.plan, self.root / "baseline", self.root / "fallback")
        self.assertIn("mode", result["checks"][0]["reason"])

    def test_missing_tampered_or_rethresholded_evidence_never_passes(self):
        self.assertEqual(self.compare()["status"], "UNVERIFIED")
        self.capture("baseline")
        self.capture("candidate")
        data = self.root / "candidate/a/tensors.npz"
        data.write_bytes(data.read_bytes() + b"changed")
        self.assertEqual(self.compare()["status"], "FAIL")
        self.contract["comparison"] = {"kind": "tolerance", "min_cosine": 0, "max_relative_l2": 10}
        self.save_contract()
        self.assertIn("contract changed", self.compare()["checks"][0]["reason"])

    def test_capture_rejects_nan_and_midrun_changes(self):
        with self.assertRaisesRegex(ValueError, "NaN"):
            self.capture("bad", output=np.full((2, 3), np.nan))
        capture = parity.Capture(self.root / "changed", self.plan, "a", self.inputs, self.artifact)
        self.artifact.write_bytes(b"new library")
        with self.assertRaisesRegex(ValueError, "changed during"):
            capture.finish(self.outputs, effective_grid=self.case["grid"], actual_mode="eager",
                           output_domain="normalized", board="test", arch="cpu",
                           checkpoint="test", entry_point="test")

    def test_repeated_grid_is_valid_but_roundtrip_mutation_fails(self):
        final = dict(self.case, id="return")
        self.contract["cases"] += [dict(self.case, id="changed", grid=[[1, 32, 32]]), final]
        self.contract["roundtrips"] = [["a", "return"]]
        self.save_contract()
        for side in ("baseline", "candidate"):
            self.capture(side)
            self.capture(side, name="changed", effective_grid=[[1, 32, 32]])
            self.capture(side, name="return")
        result = self.compare()
        self.assertEqual(result["checks"][-1]["status"], "PASS")
        observations = [{k: c[k] for k in ("id", "grid", "output_shape", "mode", "domain")}
                        for c in self.contract["cases"]]
        self.assertEqual(parity.shapes(self.contract, observations)["status"], "PASS")
        observations[1]["grid"] = [[1, 18, 18]]
        self.assertEqual(parity.shapes(self.contract, observations)["status"], "FAIL")
        # Change both sides equally: A/B parity alone would miss the roundtrip regression.
        for side in ("mutant-a", "mutant-b"):
            self.capture(side)
            self.capture(side, name="changed", effective_grid=[[1, 32, 32]])
            self.capture(side, name="return", output=self.outputs + 1)
        result = parity.compare(self.plan, self.root / "mutant-a", self.root / "mutant-b")
        self.assertEqual(result["checks"][-1]["status"], "FAIL")

    def test_metrics_cover_zero_signed_zero_dtype_and_extreme_finite_values(self):
        metric = {"kind": "tolerance", "min_cosine": .99, "max_relative_l2": .02}
        for scale in (1e-300, 1, 1e300):
            a = np.array([1, 2, 3], dtype=np.float64) * scale
            result = parity.compare_arrays(a, a * 1.01, metric)
            self.assertEqual(result["status"], "PASS")
            self.assertAlmostEqual(result["relative_l2"], .01)
        self.assertEqual(parity.compare_arrays(np.zeros(2), np.zeros(2), metric)["status"], "PASS")
        self.assertEqual(parity.compare_arrays(np.zeros(2), np.ones(2), metric)["status"], "FAIL")
        self.assertEqual(parity.compare_arrays(np.array([0.]), np.array([-0.]), {"kind": "byte"})["status"], "FAIL")
        a = np.ones(2, dtype=np.float32)
        self.assertEqual(parity.compare_arrays(a, a.astype(np.float64), {"kind": "byte"})["status"], "FAIL")

    def test_tuning_requires_growth_after_reload_not_only_initial_stage(self):
        initial = {"shape252": {"candidate": "gemm", "time": 1.0}}
        self.assertEqual(parity.tuning({}, initial)["status"], "PASS")
        self.assertEqual(parity.tuning(initial, copy.deepcopy(initial))["status"], "FAIL")
        final = dict(initial, shape448={"candidate": "gemm", "time": 2.0})
        self.assertEqual(parity.tuning(initial, final)["new_keys"], ["shape448"])
        self.assertEqual(parity.tuning(initial, {"shape448": final["shape448"]})["status"], "FAIL")
        with self.assertRaises(ValueError):
            parity.tuning(initial, {"new": {}})

    def test_native_tuning_db_identity_and_duplicate_keys(self):
        before = dict(schema="apxinf.cuda.tuning.v1", device_name="test", sm=110,
                      cuda_version="13", cublas_version="13", kernel_build_id="fixture", records=[])
        after = dict(before, records=[dict(key={"m": 10, "n": 26, "k": 1024},
                     tactic=dict(backend="cublaslt", id=0, implementation_version=1), milliseconds=.01)])
        self.assertEqual(parity.persisted_tuning(before, after)["status"], "PASS")
        self.assertEqual(parity.persisted_tuning(after, after)["status"], "FAIL")
        with self.assertRaisesRegex(ValueError, "identity changed"):
            parity.persisted_tuning(before, dict(after, kernel_build_id="different"))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parity.persisted_tuning(before, dict(after, records=after["records"] * 2))

    def test_cache_metadata_cannot_substitute_for_build_receipt(self):
        receipt = self.capture("baseline")
        path = self.root / "old.json"
        path.write_text(json.dumps({"mtime": 123, "object_sha256": receipt["loaded_sha256"]}))
        with self.assertRaisesRegex(ValueError, "invalid"):
            parity.identity(receipt, path)

    def test_nonvisual_model_and_invalid_contracts(self):
        self.case["grid"] = []
        self.save_contract()
        self.capture("baseline")
        self.capture("candidate")
        self.assertEqual(self.compare()["checks"][0]["status"], "PASS")
        self.contract["comparison"] = {"kind": "tolerance", "min_cosine": float("nan"), "max_relative_l2": .01}
        self.save_contract()
        with self.assertRaises(ValueError):
            parity.load_contract(self.plan)


class ObservedBuildTests(unittest.TestCase):
    def test_real_fresh_cargo_build_loaded_bytes_and_source_change(self):
        # No dependency downloads, CUDA, fake cargo executable or mock build results.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            def run(*args):
                subprocess.run(args, cwd=root, check=True, capture_output=True)
            run("git", "init", "-q")
            run("git", "config", "user.email", "test@example.invalid")
            run("git", "config", "user.name", "Test")
            (root / ".gitignore").write_text("devlocal/\n")
            (root / "Cargo.toml").write_text('[package]\nname="parity_fixture"\nversion="0.1.0"\nedition="2021"\n')
            (root / "src").mkdir()
            source = root / "src/main.rs"
            source.write_text('fn main() { println!("actual executable"); }\n')
            run("cargo", "generate-lockfile", "--offline")
            run("git", "add", ".")
            run("git", "commit", "-qm", "fixture")
            destination = root / "devlocal/build"
            # Only change the current directory; cargo itself is executed unchanged.
            previous = Path.cwd()
            try:
                os.chdir(root)
                receipt = parity.build(destination, "debug/parity_fixture", ["--offline"])
                with self.assertRaises(FileExistsError):
                    parity.build(destination, "debug/parity_fixture", ["--offline"])
            finally:
                os.chdir(previous)
            binary = Path(receipt["artifact"])
            self.assertEqual(subprocess.check_output([str(binary)], text=True).strip(), "actual executable")
            observed = {"loaded_artifact": str(binary), "loaded_sha256": parity.digest(binary)}
            self.assertEqual(parity.identity(observed, destination / "build.json")["status"], "PASS")
            with self.assertRaisesRegex(ValueError, "loaded bytes"):
                parity.identity(dict(observed, loaded_sha256="0" * 64), destination / "build.json")
            source.write_text('fn main() { println!("new source"); }\n')
            with self.assertRaisesRegex(ValueError, "source changed"):
                parity.identity(observed, destination / "build.json")


if __name__ == "__main__":
    unittest.main()
