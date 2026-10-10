"""CPU checks for the matrix identity, lock, and comparison boundaries."""

import copy
import fcntl
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "benchmarks/serving"
spec = importlib.util.spec_from_file_location("apxinf_metal_matrix_test", DIRECTORY / "metal_matrix.py")
matrix = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(DIRECTORY))
try:
    spec.loader.exec_module(matrix)
finally:
    sys.path.pop(0)


class MatrixTests(unittest.TestCase):
    def ready(self, chunk=256, batch=1):
        return {"ready": True, "model": "apxinf-local", "worker": {
            "model_path": "/model", "model_manifest": {
                "artifacts": [{"path": "weights.safetensors", "size_bytes": 8, "sha256": "a" * 64}],
                "adapter_sha256": "b" * 64,
                "execution": {"provider": "mlx-lm", "precision": "bundle", "prefill_step_size": chunk,
                              "output_batch_tokens": batch, "memory_limit_bytes": 1024}},
            "capabilities": {"max_active_sequences": 1}, "runtime": {"mlx": "0.32.1"}}}

    def test_all_nine_pairs_are_unique(self):
        self.assertEqual(len(matrix.PROFILES), 9)
        self.assertEqual(len(set(matrix.PROFILES)), 9)

    def test_parent_lock_requires_acknowledgement_and_contention(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metal.lock"
            path.touch()
            with self.assertRaisesRegex(RuntimeError, "lock-owned-by-parent"):
                matrix.require_parent_lock(path, False)
            with self.assertRaisesRegex(RuntimeError, "lock is free"):
                matrix.require_parent_lock(path, True)
            with path.open("rb") as owner:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                matrix.require_parent_lock(path, True)

    def test_identity_allows_only_two_experiment_settings(self):
        reference = matrix.validate_ready(self.ready(), Path("/model"), 256, 1, 1024)
        changed = self.ready(64, 8)
        self.assertEqual(reference, matrix.validate_ready(changed, Path("/model"), 64, 8, 1024, reference))
        changed["worker"]["model_manifest"]["artifacts"][0]["sha256"] = "c" * 64
        with self.assertRaisesRegex(ValueError, "changed across profiles"):
            matrix.validate_ready(changed, Path("/model"), 64, 8, 1024, reference)

    def test_readiness_must_match_exact_execution_and_path(self):
        with self.assertRaisesRegex(ValueError, "execution settings"):
            matrix.validate_ready(self.ready(), Path("/model"), 64, 1, 1024)
        with self.assertRaisesRegex(ValueError, "different model path"):
            matrix.validate_ready(self.ready(), Path("/other"), 256, 1, 1024)

    def test_runtime_or_adapter_drift_fails(self):
        ready = self.ready()
        reference = matrix.validate_ready(ready, Path("/model"), 256, 1, 1024)
        changed = copy.deepcopy(ready)
        changed["worker"]["runtime"]["mlx"] = "different"
        with self.assertRaises(ValueError):
            matrix.validate_ready(changed, Path("/model"), 256, 1, 1024, reference)
        changed = copy.deepcopy(ready)
        changed["worker"]["model_manifest"]["adapter_sha256"] = "d" * 64
        with self.assertRaises(ValueError):
            matrix.validate_ready(changed, Path("/model"), 256, 1, 1024, reference)

    def test_owned_cpu_process_is_stopped_and_reaped(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 start_new_session=True)
        try:
            result = matrix.stop_owned_group(child)
            self.assertIsNotNone(child.returncode)
            self.assertEqual(result["pgid"], child.pid)
            self.assertFalse(any(not item["state"].startswith("Z") for item in result["remaining_members"]))
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()

    def test_inspection_failure_still_reaps_owned_cpu_process(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 start_new_session=True)
        try:
            with patch.object(matrix, "group_members", side_effect=PermissionError("inspection unavailable")):
                with self.assertRaises(PermissionError):
                    matrix.stop_owned_group(child)
            self.assertIsNotNone(child.returncode)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()

    def test_output_comparison_uses_index_and_records_failures(self):
        def sample(index, output="same", success=True):
            return {"index": index, "prompt_sha256": f"prompt-{index}",
                    "output_sha256": output, "success": success}
        profiles = [
            {"profile": "base", "samples": [sample(0), sample(1), sample(2)]},
            {"profile": "reordered", "samples": [sample(2), sample(0), sample(1)]},
            {"profile": "bad", "samples": [sample(0, "other"), sample(2, success=False)]},
        ]
        result = matrix.output_comparison(profiles)
        self.assertEqual(result["token_id_parity"], "not measured")
        self.assertTrue(result["profiles"][1]["equal"])
        self.assertEqual(result["profiles"][2]["mismatched_indices"], [0])
        self.assertEqual(result["profiles"][2]["missing_indices"], [1])
        self.assertEqual(result["profiles"][2]["failed_indices"], [2])


if __name__ == "__main__":
    unittest.main()
