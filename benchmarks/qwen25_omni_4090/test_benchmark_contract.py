import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("benchmark_contract.py")
SPEC = importlib.util.spec_from_file_location("benchmark_contract", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class BenchmarkContractTest(unittest.TestCase):
    def test_probe_requires_typed_400_and_message(self):
        payload = {
            "error": {
                "type": "invalid_request",
                "message": "prompt 32768 + completion 1 exceeds context 32768",
            }
        }
        self.assertTrue(MODULE.probe_passes(400, payload, "exceeds context 32768"))
        self.assertFalse(MODULE.probe_passes(413, payload, "exceeds context 32768"))
        self.assertFalse(MODULE.probe_passes(400, payload, "another message"))

    def test_evaluation_body_is_deterministic_and_greedy(self):
        body = MODULE.evaluation_body(19, 3)
        self.assertEqual(body["input_ids"][:3], [1000, 1001, 1002])
        self.assertEqual(body["input_ids"][17:], [1000, 1001])
        self.assertEqual(body["max_new_tokens"], 3)
        self.assertEqual(body["temperature"], 0)
        self.assertTrue(body["ignore_eos"])
        self.assertFalse(body["stream"])


if __name__ == "__main__":
    unittest.main()
