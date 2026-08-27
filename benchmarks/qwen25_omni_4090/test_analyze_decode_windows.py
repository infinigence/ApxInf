import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("analyze_decode_windows.py")
SPEC = importlib.util.spec_from_file_location("analyze_decode_windows", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class DecodeWindowAnalysisTest(unittest.TestCase):
    def test_merged_duration_unions_overlap_but_preserves_gaps(self):
        self.assertEqual(
            MODULE.merged_duration([(10, 20), (15, 30), (35, 40)]),
            25,
        )

    def test_merged_duration_is_order_independent(self):
        self.assertEqual(MODULE.merged_duration([(8, 9), (1, 3), (3, 5)]), 5)


if __name__ == "__main__":
    unittest.main()
