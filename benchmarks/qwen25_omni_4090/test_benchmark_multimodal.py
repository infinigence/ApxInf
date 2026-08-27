import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("benchmark_multimodal.py")
SPEC = importlib.util.spec_from_file_location("benchmark_multimodal", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class BenchmarkMultimodalTest(unittest.TestCase):
    def test_image_request_is_fail_closed_data_url(self):
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / "case.png"
            media.write_bytes(b"png")
            body = MODULE.request_body("image", media, "prompt")
        part = body["messages"][0]["content"][0]
        self.assertEqual(part["type"], "image_url")
        self.assertTrue(part["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(body["temperature"], 0)
        self.assertFalse(body["stream"])

    def test_reference_prefers_candidate_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            path.write_text(
                '{"candidate_binary_sha256":"abc","cases":['
                '{"case_id":"case","baseline":{"output_tokens":[1]},'
                '"candidate":{"output_tokens":[2]}}]}',
                encoding="utf-8",
            )
            binary_hash, tokens = MODULE.reference_tokens(path)
        self.assertEqual(binary_hash, "abc")
        self.assertEqual(tokens, {"case": [2]})


if __name__ == "__main__":
    unittest.main()
