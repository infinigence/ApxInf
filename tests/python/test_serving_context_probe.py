"""Check the local context probe without a model or a GPU."""

import copy
import json
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks" / "serving"))
import context_probe as probe


class FixtureTransport:
    def __init__(self, context=2048, mode="pass"):
        self.mode = mode
        self.paths = []
        self.generated_body = None
        self.counted_body = None
        self.ready_reads = 0
        self.ready = {"ready": True, "profile": "apxinf-serial-v0.1", "model": "local-fixture",
                      "worker": {"model_revision": "a" * 64, "capability_revision": "b" * 64,
                                 "worker_epoch": "fixture-epoch", "model_manifest": {"provider": "fixture"},
                                 "capabilities": {"max_context": context, "max_output_tokens": 128},
                                 "runtime": {"python": "fixture"}}}

    def __call__(self, base_url, path, body=None):
        self.paths.append(path)
        if path == "/readyz":
            self.ready_reads += 1
            ready = copy.deepcopy(self.ready)
            if self.mode == "identity_change" and self.ready_reads > 1:
                ready["worker"]["model_revision"] = "c" * 64
            if self.mode == "profile_change" and self.ready_reads > 1:
                ready["profile"] = "apxinf-native-bounded-thinking-v0.1"
            return 200, ready
        if path == "/v1/messages/count_tokens":
            self.counted_body = body
            if self.mode == "count_error":
                return 400, {"error": "count failed"}
            count = 10 + len(body["messages"][0]["content"].split())
            if count > self.ready["worker"]["capabilities"]["max_context"]:
                return 400, {"error": {"code": "context_limit"}}
            return 200, {"input_tokens": count}
        if path == "/v1/messages":
            self.generated_body = body
            if self.mode == "http_error":
                return 503, {"error": "worker unavailable"}
            text = body["messages"][0]["content"]
            facts = {
                name: re.search(rf"{name}=([0-9a-f]{{16}})", text).group(1)
                for name in ("start", "middle", "end")}
            if self.mode == "wrong_fact":
                facts["middle"] = "0" * 16
            if self.mode == "missing_fact":
                del facts["end"]
            count = 10 + len(text.split())
            if self.mode == "count_mismatch":
                count += 1
            reason = "max_tokens" if self.mode == "truncated" else "end_turn"
            model = "fallback-fixture" if self.mode == "fallback_model" else "local-fixture"
            answer = json.dumps(facts)
            if self.mode == "fenced_json":
                answer = "```json\n" + answer + "\n```"
            return 200, {"model": model, "stop_reason": reason,
                         "usage": {"input_tokens": count},
                         "content": [{"type": "text", "text": answer}]}
        raise AssertionError(path)


def no_metrics(_):
    return {"text": None, "values": {}, "error": "fixture"}


class ContextProbeTests(unittest.TestCase):
    def test_correct_facts_in_a_code_fence_remain_a_format_failure(self):
        case = probe.run_case("http://127.0.0.1:8000", 500, 96,
                              FixtureTransport(mode="fenced_json"), no_metrics)
        self.assertEqual(case["status"], "failed")
        self.assertEqual(case["reason"],
                         "The response is not a plain JSON object with the required string facts.")
        text = case["raw_response"]["content"][0]["text"]
        for value in case["nonce_facts"].values():
            self.assertIn(value, text)

    def test_declared_text_profiles_use_the_same_context_checks(self):
        for profile in ("apxinf-serial-v0.1", "apxinf-native-sampled-v0.1",
                        "apxinf-native-thinking-v0.1", "apxinf-native-bounded-thinking-v0.1"):
            with self.subTest(profile=profile):
                transport = FixtureTransport()
                transport.ready["profile"] = profile
                case = probe.run_case("http://127.0.0.1:8000", 500, 96, transport, no_metrics)
                self.assertEqual(case["status"], "passed", case["reason"])
        transport = FixtureTransport()
        transport.ready["profile"] = "unknown-profile"
        case = probe.run_case("http://127.0.0.1:8000", 500, 96, transport, no_metrics)
        self.assertEqual(case["status"], "failed")
        self.assertNotIn("/v1/messages", transport.paths)

    def test_calibration_uses_counted_template_and_same_generation_body(self):
        transport = FixtureTransport()
        case = probe.run_case("http://127.0.0.1:8000", 500, 96, transport, no_metrics)
        self.assertEqual(case["status"], "passed", case["reason"])
        self.assertGreaterEqual(case["actual_input_tokens"], 475)
        self.assertLessEqual(case["actual_input_tokens"], 500)
        self.assertEqual(transport.counted_body, transport.generated_body)
        self.assertLessEqual(case["count_calls"], 15)
        self.assertEqual(case["response_input_tokens"], case["actual_input_tokens"])
        self.assertEqual(case["nonce_facts"], case["observed_facts"])

    def test_target_overflow_does_not_generate(self):
        transport = FixtureTransport(context=540)
        case = probe.run_case("http://127.0.0.1:8000", 500, 96, transport, no_metrics)
        self.assertEqual(case["status"], "unsupported")
        self.assertNotIn("/v1/messages", transport.paths)
        self.assertNotIn("/v1/messages/count_tokens", transport.paths)

    def test_small_calibration_does_not_claim_target(self):
        transport = FixtureTransport(context=200000)
        case = probe.run_case("http://127.0.0.1:8000", 120000, 96, transport, no_metrics)
        self.assertEqual(case["status"], "unsupported")
        self.assertLess(case["actual_input_tokens"], 114000)
        self.assertNotIn("/v1/messages", transport.paths)

    def test_wrong_missing_and_mismatched_usage_fail(self):
        for mode in ("wrong_fact", "missing_fact", "count_mismatch", "truncated", "fallback_model"):
            with self.subTest(mode=mode):
                case = probe.run_case("http://127.0.0.1:8000", 500, 96,
                                      FixtureTransport(mode=mode), no_metrics)
                self.assertEqual(case["status"], "failed")
                self.assertIsNotNone(case["reason"])

    def test_errors_and_identity_change_fail(self):
        for mode in ("count_error", "http_error", "identity_change", "profile_change"):
            with self.subTest(mode=mode):
                case = probe.run_case("http://127.0.0.1:8000", 500, 96,
                                      FixtureTransport(mode=mode), no_metrics)
                self.assertEqual(case["status"], "failed")
                self.assertIsNotNone(case["reason"])

    def test_run_writes_report(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(base_url="http://127.0.0.1:8000", input_tokens=[500],
                                   max_tokens=96, output_dir=Path(directory), metal_lock=None)
            with patch.object(probe, "request_json", FixtureTransport()), patch.object(
                probe, "metrics_snapshot", no_metrics
            ):
                self.assertTrue(probe.run(args))
            report = json.loads((Path(directory) / "report.json").read_text())
            self.assertEqual(report["cases"][0]["status"], "passed")


if __name__ == "__main__":
    unittest.main()
