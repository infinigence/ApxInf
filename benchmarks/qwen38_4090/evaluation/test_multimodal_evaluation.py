from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


generator = load_module("qwen38_mm_generator_test", HERE / "generate_multimodal_cases.py")
runner = load_module("qwen38_mm_runner_test", HERE / "run_multimodal.py")
scorer = load_module("qwen38_mm_scorer_test", HERE / "score_multimodal.py")


def report(split: str, total: int, *, declared=True, passed=None, fail_closed=None):
    passed = total if passed is None else passed
    return {
        "schema": runner.REPORT_SCHEMA,
        "implementation": {"name": "candidate", "revision": "a" * 40, "backend": "apxinf"},
        "split": split,
        "capability_declared": declared,
        "fallback_active": False,
        "fail_closed": fail_closed,
        "cases_passed": passed,
        "cases_total": total,
        "request_success_rate": 1.0 if passed == total and declared is not False else 0.0,
        "service_healthy_after_run": True,
        "evidence": {"contract_sha256": "1" * 64},
    }


class MultimodalEvaluationTest(unittest.TestCase):
    def test_public_generator_is_byte_reproducible(self):
        contract = HERE / "multimodal-contract-v1.json"
        with tempfile.TemporaryDirectory() as left_raw, tempfile.TemporaryDirectory() as right_raw:
            left = Path(left_raw)
            right = Path(right_raw)
            generator.write_suite(
                generator.generate_cases(generator.PUBLIC_SEED, "public-mm", 1),
                left,
                contract,
                "public",
            )
            generator.write_suite(
                generator.generate_cases(generator.PUBLIC_SEED, "public-mm", 1),
                right,
                contract,
                "public",
            )
            self.assertEqual((left / "manifest.json").read_bytes(), (right / "manifest.json").read_bytes())
            manifest = json.loads((left / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["case_count"], 4)
            self.assertEqual(set(manifest["categories"].values()), {1})
            for image in sorted((left / "images").glob("*.png")):
                self.assertEqual(image.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
                self.assertEqual(image.read_bytes(), (right / "images" / image.name).read_bytes())

    def test_normalization_removes_only_one_leading_think_block(self):
        self.assertEqual(runner.normalize_answer("  42\n"), "42")
        self.assertEqual(runner.normalize_answer("<think>ignored</think>\nBLUE"), "BLUE")
        self.assertEqual(runner.normalize_answer("prefix <think>x</think> BLUE"), "prefix <think>x</think> BLUE")

    def test_unsupported_path_requires_machine_readable_fail_closed_error(self):
        self.assertTrue(
            runner.unsupported_probe_valid(
                501, {"error": {"type": "unsupported_capability", "message": "not ready"}}
            )
        )
        self.assertFalse(runner.unsupported_probe_valid(500, {"error": {"type": "server_error"}}))
        self.assertFalse(runner.unsupported_probe_valid(200, {}))

    def test_badges_do_not_change_leaderboard_points(self):
        public = report("public", 4)
        hidden = report("hidden", 8)
        ready = scorer.score(public, hidden)
        self.assertEqual(ready["badge"], "multimodal-ready")
        self.assertEqual(ready["leaderboard_points"], 0.0)

        unsupported = report("public", 4, declared=False, passed=0, fail_closed=True)
        self.assertEqual(scorer.score(unsupported, None)["badge"], "declared-unsupported")


if __name__ == "__main__":
    unittest.main()
