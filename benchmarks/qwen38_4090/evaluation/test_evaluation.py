from __future__ import annotations

import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


score_module = load_module("score_submission", HERE / "score_submission.py")
efficiency_module = load_module("compute_efficiency", HERE / "compute_efficiency.py")
runner_module = load_module("run_evaluation", HERE / "run_evaluation.py")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def clone(value):
    return json.loads(json.dumps(value))


def rename(submission, suffix):
    submission["implementation"]["name"] += f" {suffix}"
    submission["implementation"]["revision"] += f"-{suffix}"
    submission["implementation"]["backend"] += f"-{suffix}"
    return submission


def make_submission(contract, name, backend, ttft_scale, tpot_1k, tpot_8k):
    cells = {}
    tpot_by_id = {
        "text-perf-1024": tpot_1k,
        "text-perf-8192": tpot_8k,
    }
    for index, definition in enumerate(contract["performance_scoring"]["ttft_cells"]):
        ttft = ttft_scale * (2**index)
        tpot = tpot_by_id.get(definition["id"], tpot_8k)
        cells[definition["id"]] = {
            "actual_prompt_tokens": definition["prompt_tokens"],
            "completion_tokens": definition["output_tokens"],
            "success_rate": 1.0,
            "measured_repeats": 1,
            "warmup_repeats": 0,
            "ttft_cv": 0.0,
            "tpot_cv": 0.0,
            "ttft_s": ttft,
            "tpot_s": tpot,
            "e2e_s": ttft + 127 * tpot,
            "peak_vram_mib": 20000.0,
        }
    return {
        "schema": "apxinf.qwen38_27b.leaderboard_submission.v1",
        "implementation": {"name": name, "revision": "synthetic-v1", "backend": backend},
        "correctness": {
            "protocol_pass": True,
            "public_cases_passed": 6,
            "public_cases_total": 6,
            "hidden_cases_passed": None,
            "hidden_cases_total": None,
            "public_trajectory_tokens_passed": 256,
            "public_trajectory_tokens_total": 256,
            "hidden_trajectory_tokens_passed": None,
            "hidden_trajectory_tokens_total": None,
        },
        "cells": cells,
        "context": {
            "max_verified_prompt_tokens": 32768,
            "verified_output_tokens": 128,
            "verified_cases_at_max_context": 6,
            "pass_rate_at_max_context": 1.0,
            "first_failed_prompt_tokens": 65536,
            "failure_mode": "capacity_rejected",
            "service_healthy_after_failure": True,
        },
        "multi_request": {"cells": {}},
        "reliability": {
            "request_success_rate": 1.0,
            "no_unexpected_oom": True,
            "no_nan": True,
            "no_fallback": True,
            "no_xid": True,
            "service_healthy_after_failure": True,
        },
        "evidence": {
            "run_id": "synthetic",
            "contract_sha256": "0" * 64,
            "public_manifest_sha256": "0" * 64,
            "hidden_manifest_sha256": None,
            "context_manifest_sha256": None,
            "raw_jsonl": "raw.jsonl",
            "raw_jsonl_sha256": "0" * 64,
            "environment_json": "environment.json",
            "environment_json_sha256": "0" * 64,
            "trajectory_reference_sha256": "0" * 64,
            "trajectory_details": {"public": {}, "hidden": {}},
        },
    }


def make_midterm_eligible(submission):
    submission = clone(submission)
    correctness = submission["correctness"]
    correctness.update(
        {
            "hidden_cases_passed": 11,
            "hidden_cases_total": 12,
            "hidden_trajectory_tokens_passed": 244,
            "hidden_trajectory_tokens_total": 256,
        }
    )
    for cell in submission["cells"].values():
        cell["measured_repeats"] = 5
        cell["warmup_repeats"] = 1
        if "ttft_s" in cell:
            cell["ttft_cv"] = 0.02
        if "tpot_s" in cell:
            cell["tpot_cv"] = 0.02
    return submission


def add_multi_cells(submission, c4_goodput, c8_goodput):
    submission["multi_request"] = {
        "cells": {
            "multi-c4-text-perf-1024": {
                "concurrency": 4,
                "total_requests": 32,
                "actual_prompt_tokens": 1024,
                "completion_tokens": 128,
                "success_rate": 1.0,
                "correctness_rate": 1.0,
                "measured_repeats": 1,
                "warmup_repeats": 0,
                "goodput_tokens_per_s": c4_goodput,
                "p95_ttft_s": 1.0,
                "p95_tpot_s": 0.1,
                "jain_fairness_index": 0.99,
                "no_fallback": True,
                "service_healthy_after_run": True,
            },
            "multi-c8-text-perf-1024": {
                "concurrency": 8,
                "total_requests": 32,
                "actual_prompt_tokens": 1024,
                "completion_tokens": 128,
                "success_rate": 1.0,
                "correctness_rate": 1.0,
                "measured_repeats": 1,
                "warmup_repeats": 0,
                "goodput_tokens_per_s": c8_goodput,
                "p95_ttft_s": 2.0,
                "p95_tpot_s": 0.1,
                "jain_fairness_index": 0.99,
                "no_fallback": True,
                "service_healthy_after_run": True,
            },
        }
    }


class EvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = read_json(HERE / "contract-v1.json")
        cls.vllm = make_submission(
            cls.contract, "synthetic-control", "vllm", 0.5, 0.08, 0.08
        )
        cls.apxinf = make_submission(
            cls.contract, "synthetic-candidate", "apxinf", 2.0, 0.025, 0.03
        )

    def test_public_cohort_calibration_uses_per_cell_dynamic_references(self):
        result = score_module.score_cohort(
            self.contract,
            [self.apxinf],
            "public_calibration",
            controls=[self.vllm],
        )
        apxinf = result["scores"][0]
        vllm = result["control_scores"][0]
        self.assertTrue(apxinf["eligible"])
        self.assertAlmostEqual(apxinf["leaderboard_score"], 73.75)
        self.assertAlmostEqual(apxinf["section_scores"]["ttft"], 8.75)
        self.assertEqual(apxinf["section_scores"]["tpot"], 25.0)
        self.assertAlmostEqual(vllm["leaderboard_score"], 83.75)
        self.assertEqual(vllm["section_scores"]["ttft"], 35.0)
        self.assertAlmostEqual(vllm["section_scores"]["tpot"], 8.75)

    def test_fast_ineligible_submission_cannot_set_reference(self):
        valid = rename(clone(self.vllm), "valid")
        invalid = rename(clone(self.vllm), "invalid")
        invalid["correctness"]["protocol_pass"] = False
        invalid["cells"]["text-perf-1024"]["ttft_s"] = 0.001
        result = score_module.score_cohort(
            self.contract, [valid, invalid], "public_calibration"
        )
        self.assertEqual(
            result["performance_references"]["ttft_s"]["text-perf-1024"],
            valid["cells"]["text-perf-1024"]["ttft_s"],
        )
        self.assertFalse(result["scores"][1]["eligible"])

    def test_midterm_requires_public_and_hidden_correctness(self):
        result = score_module.score_submission(
            self.contract, self.vllm, "midterm_leaderboard"
        )
        self.assertFalse(result["eligible"])
        self.assertIn(
            "hidden_pass_rate_below_threshold_or_missing",
            result["eligibility_failures"],
        )
        self.assertIn(
            "hidden_token_trajectory_rate_below_threshold_or_missing",
            result["eligibility_failures"],
        )

    def test_token_trajectory_uses_categorical_edit_distance_but_is_not_a_gate(self):
        self.assertEqual(
            runner_module.token_edit_distance([1, 2, 3, 4], [1, 3, 4, 5]),
            2,
        )
        candidate = rename(clone(self.vllm), "trajectory-diagnostic")
        candidate["correctness"]["public_trajectory_tokens_passed"] = 0
        result = score_module.score_submission(
            self.contract, candidate, "public_calibration"
        )
        self.assertTrue(result["eligible"])
        self.assertEqual(
            result["section_scores"]["correctness"],
            25.0,
        )
        with self.assertRaisesRegex(ValueError, "missing frozen case"):
            runner_module.trajectory_counts(
                [], ["missing-trajectory"], {"cases": {}}
            )

    def test_midterm_requires_vllm_control_and_stable_measurements(self):
        candidate = rename(make_midterm_eligible(self.apxinf), "midterm")
        control = make_midterm_eligible(self.vllm)
        with self.assertRaisesRegex(ValueError, "eligible --control-submission"):
            score_module.score_cohort(
                self.contract,
                [candidate],
                "midterm_leaderboard",
                require_official_control=True,
            )
        result = score_module.score_cohort(
            self.contract,
            [candidate],
            "midterm_leaderboard",
            controls=[control],
            require_official_control=True,
        )
        self.assertTrue(result["official_control_present"])
        self.assertFalse(result["provisional"])
        self.assertTrue(result["scores"][0]["eligible"])

    def test_context_is_zero_at_32k_and_rewards_progress_to_native_limit(self):
        candidate = rename(clone(self.vllm), "context")
        at_32k = score_module.score_submission(
            self.contract, candidate, "public_calibration"
        )
        self.assertEqual(at_32k["section_scores"]["context_bonus"], 0.0)

        candidate["context"].update(
            {
                "max_verified_prompt_tokens": 65536,
                "verified_cases_at_max_context": 6,
                "first_failed_prompt_tokens": 65537,
                "failure_mode": "capacity_rejected",
            }
        )
        at_64k = score_module.score_submission(
            self.contract, candidate, "public_calibration"
        )
        expected = 10.0 * math.log2(65536 / 32768) / math.log2(262016 / 32768)
        self.assertAlmostEqual(at_64k["section_scores"]["context_bonus"], expected)

        candidate["context"]["first_failed_prompt_tokens"] = None
        without_boundary = score_module.score_submission(
            self.contract, candidate, "public_calibration"
        )
        self.assertEqual(without_boundary["section_scores"]["context_bonus"], 0.0)
        self.assertIn(
            "missing_or_invalid_first_failed_prompt_tokens",
            without_boundary["context_bonus_details"]["reasons"],
        )

        candidate["context"].update(
            {
                "max_verified_prompt_tokens": 262016,
                "first_failed_prompt_tokens": None,
                "failure_mode": None,
            }
        )
        at_native_limit = score_module.score_submission(
            self.contract, candidate, "public_calibration"
        )
        self.assertEqual(at_native_limit["section_scores"]["context_bonus"], 10.0)

    def test_multi_request_bonus_uses_best_valid_goodput(self):
        slower = rename(clone(self.vllm), "slower")
        faster = rename(clone(self.vllm), "faster")
        add_multi_cells(slower, 100.0, 120.0)
        add_multi_cells(faster, 200.0, 240.0)
        faster["context"].update(
            {
                "max_verified_prompt_tokens": 262016,
                "verified_cases_at_max_context": 6,
                "first_failed_prompt_tokens": None,
                "failure_mode": None,
            }
        )
        result = score_module.score_cohort(
            self.contract, [slower, faster], "public_calibration"
        )
        self.assertEqual(
            result["scores"][0]["section_scores"]["multi_request_bonus"], 6.0
        )
        self.assertEqual(
            result["scores"][1]["section_scores"]["multi_request_bonus"], 10.0
        )
        self.assertEqual(result["scores"][1]["base_score"], 100.0)
        self.assertEqual(result["scores"][1]["bonus_score"], 20.0)
        self.assertEqual(result["scores"][1]["leaderboard_score"], 120.0)
        self.assertEqual(result["scores"][1]["automated_course_points"], 80.0)

    def test_efficiency_separates_proxy_from_profiled_metrics(self):
        result = efficiency_module.compute_efficiency(self.contract, self.apxinf)
        decode = result["cells"]["text-perf-1024"]["decode"]
        self.assertGreater(decode["minimum_model_bwu_pct"], 80.0)
        self.assertLess(decode["minimum_model_bwu_pct"], 100.0)
        self.assertLess(decode["estimated_mfu_bf16_equivalent_pct"], 5.0)
        self.assertIsNone(decode["profiled"])
        self.assertTrue(math.isfinite(decode["estimated_tflops"]))


if __name__ == "__main__":
    unittest.main()
