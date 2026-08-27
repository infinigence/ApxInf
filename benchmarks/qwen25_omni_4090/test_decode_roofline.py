import unittest

from benchmarks.qwen25_omni_4090.decode_roofline import DecodeShape, estimate


class DecodeRooflineTests(unittest.TestCase):
    def test_frozen_shape_reproduces_weight_lower_bound(self):
        result = estimate(DecodeShape(), 8.252485653543307, 128, 1008.0, None)
        self.assertEqual(result["weight_bytes_lower_bound"], 6_171_881_472)
        self.assertAlmostEqual(result["effective_weight_bandwidth_gbps"], 747.8815149)
        self.assertAlmostEqual(result["weight_bwu_pct"], 74.1945947)
        self.assertIsNone(result["linear_mfu_pct"])

    def test_mfu_requires_explicit_peak_and_invalid_inputs_fail(self):
        result = estimate(DecodeShape(), 10.0, 0, 1000.0, 100.0)
        self.assertIsNotNone(result["linear_mfu_pct"])
        with self.assertRaises(ValueError):
            estimate(DecodeShape(), 0.0, 0, 1000.0, None)
        with self.assertRaises(ValueError):
            estimate(DecodeShape(), 10.0, -1, 1000.0, None)


if __name__ == "__main__":
    unittest.main()
