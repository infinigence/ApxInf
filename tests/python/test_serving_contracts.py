"""Original cross-language conformance cases for the serial worker profile."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "python/apxinf/apxinf/serving/contracts.py"
SPEC = importlib.util.spec_from_file_location("serving_contracts_under_test", MODULE)
assert SPEC is not None and SPEC.loader is not None
contracts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contracts)
CORPUS = json.loads((ROOT / "tests/fixtures/serving/serial-v0.1.json").read_text())


def fixture(case: dict, corpus: dict = CORPUS) -> dict:
    value = copy.deepcopy(corpus["templates"][case["template"]])
    for pointer, replacement in case.get("set", {}).items():
        keys = pointer.split("/")[1:]
        target = value
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = copy.deepcopy(replacement)
    for key in case.get("remove", []):
        del value[key]
    return value


class ServingContractTests(unittest.TestCase):
    def test_shared_native_sampling_profile(self):
        corpus = json.loads((ROOT / "tests/fixtures/serving/native-sampling-v0.1.json").read_text())
        self.check_sampled_profile(corpus)

    def test_shared_native_thinking_profile(self):
        corpus = json.loads((ROOT / "tests/fixtures/serving/native-thinking-v0.1.json").read_text())
        self.check_sampled_profile(corpus)

    def check_sampled_profile(self, corpus):
        for case in corpus["frame_cases"]:
            with self.subTest(case["name"]):
                frame = fixture(case, corpus)
                if case.get("refresh_identity"):
                    frame["model_revision"] = contracts.canonical_identity_digest("apxinf-model-identity-v1", frame["model_manifest"])
                    frame["capability_revision"] = contracts.canonical_identity_digest("apxinf-capability-v1", frame["capabilities"])
                before = copy.deepcopy(frame)
                if case["valid"]:
                    contracts.validate_frame(frame)
                    self.assertEqual(contracts.decode_frame(contracts.encode_frame(frame)), frame)
                else:
                    with self.assertRaises(contracts.ContractError):
                        contracts.validate_frame(frame)
                if "envelope_valid" in case:
                    if case["envelope_valid"]:
                        contracts.validate_command_envelope(frame)
                    else:
                        with self.assertRaises(contracts.ContractError):
                            contracts.validate_command_envelope(frame)
                self.assertEqual(frame, before, "validator changed frame")
        for vector in corpus["identity_digests"]:
            value = corpus["templates"][vector["template"]]
            for key in vector["pointer"].split("/")[1:]:
                value = value[key]
            self.assertEqual(contracts.canonical_identity_digest(vector["domain"], value), vector["digest"])
        frame = fixture({"template": "submit", "set": {"/sampling": {
            "mode": "random", "temperature": "number-literal", "top_p": 1, "seed": 0,
        }}}, corpus)
        for literal in corpus["invalid_number_literals"]:
            with self.subTest(literal):
                raw = json.dumps(frame).replace('"number-literal"', literal)
                with self.assertRaises(contracts.ContractError):
                    contracts.decode_frame(raw)

    def test_shared_native_profile(self):
        corpus = json.loads((ROOT / "tests/fixtures/serving/native-v0.1.json").read_text())
        base = copy.deepcopy(CORPUS["templates"]["ready"])
        base["protocol"] = "apxinf-worker/2.1"
        base["runtime"] = corpus["runtime"]
        base["capabilities"] = corpus["capabilities"]
        base["model_manifest"]["runtime"] = corpus["runtime"]
        base["model_revision"] = contracts.canonical_identity_digest("apxinf-model-identity-v1", base["model_manifest"])
        base["capability_revision"] = contracts.canonical_identity_digest("apxinf-capability-v1", base["capabilities"])
        for case in corpus["cases"]:
            with self.subTest(case["name"]):
                frame = copy.deepcopy(base)
                for pointer, replacement in case["set"].items():
                    keys = pointer.split("/")[1:]
                    target = frame
                    for key in keys[:-1]:
                        target = target[key]
                    target[keys[-1]] = replacement
                frame["capability_revision"] = contracts.canonical_identity_digest("apxinf-capability-v1", frame["capabilities"])
                if case["valid"]:
                    contracts.validate_frame(frame)
                else:
                    with self.assertRaises(contracts.ContractError):
                        contracts.validate_frame(frame)

    def test_shared_document_cases(self):
        for case in CORPUS["document_cases"]:
            with self.subTest(case["name"]):
                if case["valid"]:
                    self.assertEqual(contracts.parse_document(case["raw"]), case["value"])
                else:
                    with self.assertRaises(contracts.ContractError):
                        contracts.parse_document(case["raw"])

    def test_shared_wire_cases(self):
        for case in CORPUS["frame_cases"]:
            with self.subTest(case["name"]):
                raw = case.get("raw")
                if raw is None:
                    raw = json.dumps(fixture(case), ensure_ascii=False, separators=(",", ":"))
                if case["valid"]:
                    frame = contracts.decode_frame(raw)
                    self.assertEqual(contracts.decode_frame(contracts.encode_frame(frame)), frame)
                else:
                    with self.assertRaises(contracts.ContractError):
                        contracts.decode_frame(raw)

    def test_shared_event_transcripts(self):
        for case in CORPUS["stream_cases"]:
            with self.subTest(case["name"]):
                tracker = contracts.AttemptTracker(CORPUS["templates"]["submit"])
                failure = None
                for index, selected in enumerate(case["events"]):
                    before = copy.deepcopy(tracker.__dict__)
                    try:
                        tracker.observe(fixture(selected))
                    except contracts.ContractError:
                        self.assertEqual(tracker.__dict__, before, "invalid event changed state")
                        failure = index
                        break
                self.assertEqual(failure, case["failure_at"])
                self.assertEqual(tracker.done, case["done"])

    def test_shared_digest_vectors(self):
        for vector in CORPUS["token_digests"]:
            self.assertEqual(contracts.token_prefix_digest(vector["tokens"]), vector["digest"])
        for vector in CORPUS["identity_digests"]:
            self.assertEqual(contracts.canonical_identity_digest(vector["domain"], vector["value"]), vector["digest"])

    def test_envelope_distinguishes_content_from_routing(self):
        command = copy.deepcopy(CORPUS["templates"]["submit"])
        command["max_tokens"] = True
        contracts.validate_command_envelope(command)
        with self.assertRaises(contracts.ContractError):
            contracts.validate_frame(command)
        command["capacity_lease_ids"] = ["untrusted"]
        with self.assertRaises(contracts.ContractError):
            contracts.validate_command_envelope(command)

    def test_bounded_document_and_nested_duplicate_keys(self):
        for raw in (b'{"parameters":{"a":1,"a":2}}', b'[' * 17 + b'0' + b']' * 17,
                    b'{"x":NaN}', b'{"x":9007199254740992}', b' ' * (contracts.MAX_FRAME_BYTES + 1)):
            with self.subTest(raw[:50]):
                with self.assertRaises(contracts.ContractError):
                    contracts.parse_document(raw)
        self.assertEqual(contracts.parse_document(b'{\n"x":1\n}'), {"x": 1})

    def test_zero_generation_finishes_without_token_events(self):
        submit = copy.deepcopy(CORPUS["templates"]["submit"])
        submit["max_tokens"] = 0
        tracker = contracts.AttemptTracker(submit)
        tracker.observe(CORPUS["templates"]["accepted"])
        terminal = fixture({"template": "terminal", "set": {
            "/event_seq": 1, "/usage/output_tokens": 0, "/state_result/history_count": 2,
            "/state_result/consumed_position": 0, "/metrics/ttft_ns": 0,
        }})
        tracker.observe(terminal)
        tracker.observe(fixture({"template": "cleanup", "set": {"/event_seq": 2}}))
        self.assertTrue(tracker.done)


def peer() -> None:
    """Decode Rust frames and return Python encodings for the Rust interop test."""
    for line in sys.stdin.buffer:
        frame = contracts.decode_frame(line)
        sys.stdout.buffer.write(contracts.encode_frame(frame))
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    if sys.argv[1:] == ["--peer"]:
        peer()
    else:
        unittest.main()
