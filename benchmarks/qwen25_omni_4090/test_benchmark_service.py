import importlib.util
from pathlib import Path
import sys
import unittest


MODULE_PATH = Path(__file__).with_name("benchmark_service.py")
SPEC = importlib.util.spec_from_file_location("qwen25_omni_benchmark", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BenchmarkContractTests(unittest.TestCase):
    def test_prompt_pattern_is_deterministic_and_bounded(self):
        self.assertEqual(MODULE.prompt_ids(5, 1000), [1000, 1001, 1002, 1003, 1004])
        values = MODULE.prompt_ids(34, 1000)
        self.assertEqual(values[:17], values[17:])

    def test_summary_requires_stable_complete_trajectories(self):
        case = MODULE.Case("fixed", 1024, 32, "fixed")
        trial = {
            "passed": True,
            "wall_seconds": 1.0,
            "ttft_seconds": 0.5,
            "tpot_seconds": 0.02,
            "prefill_tokens_per_second_proxy": 2048.0,
            "decode_tokens_per_second": 50.0,
            "e2e_output_tokens_per_second": 32.0,
            "trajectory_sha256": "a" * 64,
        }
        summary = MODULE.summarize(case, [trial, dict(trial)])
        self.assertTrue(summary["trajectory_stable"])
        changed = dict(trial, trajectory_sha256="b" * 64)
        self.assertFalse(MODULE.summarize(case, [trial, changed])["trajectory_stable"])

    def test_percentiles_are_interpolated(self):
        self.assertEqual(MODULE.percentile([1.0, 3.0], 50), 2.0)
        self.assertEqual(MODULE.percentile([2.0], 90), 2.0)


if __name__ == "__main__":
    unittest.main()
