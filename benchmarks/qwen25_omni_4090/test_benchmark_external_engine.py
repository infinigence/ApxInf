import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

TEXT_SCRIPT = SCRIPT_DIR / "benchmark_vllm_omni.py"
TEXT_SPEC = importlib.util.spec_from_file_location("benchmark_vllm_omni", TEXT_SCRIPT)
TEXT_MODULE = importlib.util.module_from_spec(TEXT_SPEC)
assert TEXT_SPEC.loader is not None
sys.modules[TEXT_SPEC.name] = TEXT_MODULE
TEXT_SPEC.loader.exec_module(TEXT_MODULE)

MM_SCRIPT = SCRIPT_DIR / "benchmark_vllm_omni_multimodal.py"
MM_SPEC = importlib.util.spec_from_file_location(
    "benchmark_vllm_omni_multimodal", MM_SCRIPT
)
MM_MODULE = importlib.util.module_from_spec(MM_SPEC)
assert MM_SPEC.loader is not None
MM_SPEC.loader.exec_module(MM_MODULE)

RECOVERY_SCRIPT = SCRIPT_DIR / "benchmark_processor_recovery.py"
RECOVERY_SPEC = importlib.util.spec_from_file_location(
    "benchmark_processor_recovery", RECOVERY_SCRIPT
)
RECOVERY_MODULE = importlib.util.module_from_spec(RECOVERY_SPEC)
assert RECOVERY_SPEC.loader is not None
RECOVERY_SPEC.loader.exec_module(RECOVERY_MODULE)


class ExternalEngineBenchmarkTest(unittest.TestCase):
    def test_processor_recovery_requires_typed_422(self):
        payload = {
            "error": {"type": "unprocessable_media", "message": "bad image"}
        }
        self.assertTrue(RECOVERY_MODULE.invalid_media_passes(422, payload))
        self.assertFalse(RECOVERY_MODULE.invalid_media_passes(400, payload))
        self.assertFalse(RECOVERY_MODULE.invalid_media_passes(422, {}))

    def test_engine_version_can_project_server_info(self):
        original = TEXT_MODULE.get_json
        TEXT_MODULE.get_json = lambda url, timeout: {
            "version": "0.5.17",
            "large_internal_state": [1, 2, 3],
        }
        try:
            path, version = TEXT_MODULE.engine_version(
                "http://127.0.0.1:8004/",
                "server_info",
                "version",
                "SGLang",
                1.0,
            )
            self.assertEqual(path, "/server_info")
            self.assertEqual(version, {"version": "0.5.17"})
            with self.assertRaises(RuntimeError):
                TEXT_MODULE.engine_version(
                    "http://127.0.0.1:8004",
                    "/server_info",
                    "missing",
                    "SGLang",
                    1.0,
                )
        finally:
            TEXT_MODULE.get_json = original

    def test_exact_text_prompt_rejects_template_underflow(self):
        self.assertEqual(
            TEXT_MODULE.exact_text_prompt(21),
            "x ",
        )
        with self.assertRaises(ValueError):
            TEXT_MODULE.exact_text_prompt(20)

    def test_audio_request_supports_vllm_and_sglang_schemas(self):
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / "case.wav"
            media.write_bytes(b"wav")
            vllm = MM_MODULE.request_body(
                "model", "audio", media, "prompt", 16, "input_audio"
            )
            sglang = MM_MODULE.request_body(
                "model", "audio", media, "prompt", 16, "audio_url"
            )

        vllm_part = vllm["messages"][0]["content"][0]
        self.assertEqual(vllm_part["type"], "input_audio")
        self.assertEqual(vllm_part["input_audio"]["format"], "wav")

        sglang_part = sglang["messages"][0]["content"][0]
        self.assertEqual(sglang_part["type"], "audio_url")
        self.assertTrue(
            sglang_part["audio_url"]["url"].startswith(
                "data:audio/wav;base64,"
            )
        )


if __name__ == "__main__":
    unittest.main()
