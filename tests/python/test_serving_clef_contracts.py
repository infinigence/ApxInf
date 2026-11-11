from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python" / "apxinf" / "apxinf" / "serving"))
import clef_contracts as c
import contracts

CORPUS = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "serving" / "clef-v0.1.json").read_text())


def expand(value, seen=()):
    if type(value) is dict and set(value) == {"$ref"}:
        reference = value["$ref"]
        if reference in seen:
            raise ValueError("cyclic corpus reference")
        group, name = reference.split(".")
        return expand(CORPUS[group][name], (*seen, reference))
    if type(value) is dict:
        return {key: expand(child, seen) for key, child in value.items()}
    if type(value) is list:
        return [expand(child, seen) for child in value]
    return value


def materialize(case, group="objects"):
    if "raw" in case:
        return contracts.parse_document(case["raw"])
    value = expand(CORPUS[group][case["template"]])
    for pointer in case.get("remove", []):
        parts = pointer[1:].split("/")
        parent = value
        for part in parts[:-1]:
            parent = parent[int(part)] if type(parent) is list else parent[part]
        del parent[int(parts[-1]) if type(parent) is list else parts[-1]]
    for pointer, replacement in case.get("set", {}).items():
        parts = pointer[1:].split("/")
        parent = value
        for part in parts[:-1]:
            parent = parent[int(part)] if type(parent) is list else parent[part]
        parent[int(parts[-1]) if type(parent) is list else parts[-1]] = expand(replacement)
    return value


class ClefContractTests(unittest.TestCase):
    def test_requests(self):
        for case in CORPUS["requests"]:
            with self.subTest(case=case["name"]):
                try:
                    value = case["raw"] if "raw" in case else materialize(case)
                    parsed = c.parse_decision_request(value)
                except contracts.ContractError:
                    self.assertFalse(case["valid"])
                else:
                    self.assertTrue(case["valid"])
                    if "canonical" in case:
                        self.assertEqual(parsed.record.document(), expand(case["canonical"]))

    def test_encodings(self):
        record = expand(CORPUS["objects"]["record"])
        for case in CORPUS["encodings"]:
            with self.subTest(case=case["name"]):
                try:
                    c.validate_encoding(materialize(case), record)
                except contracts.ContractError:
                    self.assertFalse(case["valid"])
                else:
                    self.assertTrue(case["valid"])

    def test_probabilities(self):
        record = c.validate_record(expand(CORPUS["objects"]["record"]))
        encoding = c.validate_encoding(expand(CORPUS["objects"]["encoding"]), record)
        for case in CORPUS["probabilities"]:
            with self.subTest(case=case["name"]):
                try:
                    value = materialize(case)
                    response = c.project_response("local-clef", record, encoding, value)
                except contracts.ContractError:
                    self.assertFalse(case["valid"])
                else:
                    self.assertTrue(case["valid"])
                    if "response" in case:
                        self.assertEqual(response, expand(case["response"]))
                    if "choice" in case:
                        self.assertEqual(response["answers"]["color"]["choice"], case["choice"])

    def test_digests(self):
        for case in CORPUS["digests"]:
            with self.subTest(case=case["name"]):
                self.assertEqual(contracts.canonical_identity_digest(case["domain"], expand(case["document"])), case["expected"])

    def test_frames(self):
        for case in CORPUS["frame_checks"]:
            with self.subTest(case=case["name"]):
                try:
                    frame = materialize(case, "frames")
                    contracts.validate_frame(frame)
                    self.assertEqual(contracts.decode_frame(contracts.encode_frame(frame)), frame)
                except contracts.ContractError:
                    self.assertFalse(case["valid"])
                else:
                    self.assertTrue(case["valid"])

    def test_streams(self):
        for case in CORPUS["streams"]:
            with self.subTest(case=case["name"]):
                tracker = c.DecisionTracker()
                valid = True
                for event in case["events"]:
                    frame = materialize({"template": event}, "frames") if type(event) is str else materialize(event, "frames")
                    previous = copy.deepcopy(tracker.__dict__)
                    try:
                        tracker.observe(frame)
                    except contracts.ContractError:
                        self.assertEqual(tracker.__dict__, previous)
                        valid = False
                        break
                if "before_reap" in case:
                    self.assertEqual(tracker.public_success, case["before_reap"]["public_success"])
                    self.assertEqual(tracker.resources_released, case["before_reap"]["resources_released"])
                tracker.cancelled = case.get("cancelled", False)
                if "reap" in case:
                    valid = tracker.reaped(**case["reap"]) and valid
                self.assertEqual(valid, case["valid"])
                self.assertEqual(tracker.public_success, case["public_success"])
                self.assertEqual(tracker.resources_released, case["resources_released"])

    def test_request_limits_and_immutable_values(self):
        request = expand(CORPUS["objects"]["request"])
        for mutation in ({"state": "x" * 65537}, {"model": "x" * 129}, {"questions": {}}):
            with self.assertRaises(c.ContractError):
                c.parse_decision_request({**request, **mutation})
        parsed = c.parse_decision_request(request)
        request["questions"]["color"]["criteria"]["red"] = "mutated"
        self.assertEqual(dict(parsed.record.questions[0].criteria)["red"], "A red sample")

    def test_frame_kind_must_be_a_string(self):
        frame = expand(CORPUS["frames"]["accepted"])
        for kind in ({}, [], None, True):
            with self.subTest(kind=kind), self.assertRaises(c.ContractError):
                c.validate_frame({**frame, "kind": kind})


if __name__ == "__main__":
    unittest.main()
