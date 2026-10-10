"""Handwritten offline coverage evidence and corruption regressions."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("memory_coverage_under_test", ROOT / "benchmarks/serving/memory_coverage.py")
coverage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(coverage)


class Artifact:
    """Construct original scalar observations without a model or runtime."""
    def __init__(self, directory, context=64, max_output=32):
        self.directory = Path(directory)
        self.records = []
        self.epoch = 0
        self.peak = 100
        self.seed = [3, 5]
        self.profile = {"provider": "mlx-lm", "precision": "bundle", "max_context": context,
                        "max_output_tokens": max_output, "prefill_step_size": 2, "output_batch_tokens": 1,
                        "memory_limit_bytes": 4096, "synchronization": "runtime_owned_streams", "host_pressure_policy": "macos"}
        manifest = {"runtime": {"python": "test", "mlx": "test", "mlx_lm": "test"},
                    "execution": {key: self.profile[key] for key in ("provider", "precision", "prefill_step_size",
                                                                    "output_batch_tokens", "memory_limit_bytes")},
                    "adapter_sha256": "a" * 64, "artifacts": []}
        self.report = {"format": "apxinf-memory-calibration-v1", "artifact_kind": "raw_observations",
                       "run_id": "abcdabcd-1234-4567-8123-abcdefabcdef", "tool_sha256": "b" * 64,
                       "tool_files": {name: "b" * 64 for name in ("memory_calibration.py", "host_memory.py", "metal_matrix.py")},
                       "requested_profile": self.profile, "hardware": {"model": "cpu-test", "memory_bytes": 8192, "errors": {}},
                       "identity": {"model_revision": coverage.contracts.canonical_identity_digest("apxinf-model-identity-v1", manifest),
                                    "manifest": manifest, "runtime": manifest["runtime"], "effective_context": context,
                                    "vocab_size": 100, "eos_token_ids": [2]},
                       "input_source": {"kind": "synthetic_tiled_prepared_tokens", "seed_tokens": self.seed,
                                        "seed_digest": coverage.contracts.token_prefix_digest(self.seed)},
                       "plan": [], "cases": [], "run_samples": [0, 1], "status": "completed",
                       "load_started": True, "run_cleanup": {"settled": True, "error": None},
                       "recovery": {"errors": [], "snapshot_changed": False}}
        self.process = {"pid": 1234, "returncode": 0, "reaped": True, "status": "completed", "error": None}
        self.sample(None, "before_load", memory=False)
        self.sample(None, "loaded")

    def sample(self, name, phase, count=0, position=None, layers=None, memory=True, pressure="normal"):
        self.records.append({"record_index": len(self.records), "case_id": name, "phase": phase,
                             "monotonic_ns": 1000 + len(self.records), "output_tokens": count,
                             "reported_prompt_position": position,
                             "memory": {"peak_epoch": self.epoch, "runtime_consumed_position": 999 if layers else 0,
                                        "allocator": {"active_bytes": 64, "cache_bytes": 8, "peak_bytes": self.peak},
                                        "cache_payload_bytes": (None if any(layer["nbytes"] is None for layer in layers or [])
                                                                else sum(layer["nbytes"] for layer in layers or [])),
                                        "layers": layers or []} if memory else None,
                             "memory_error": (None if memory or phase == "before_load" else "Observation unavailable."),
                             "host": {"pressure": {"state": pressure,
                                                     "dispatch_value": {"normal": 1, "warning": 2, "critical": 4}.get(pressure),
                                                     "error": "Sensor unavailable." if pressure == "unknown" else None}}})

    def case(self, name="case-a", length=4, output=3, actual=None, eos=False, prefill=None, stop=None,
             status="completed", interrupted=False, unknown_offset=False):
        index = len(self.report["cases"])
        planned = {"case_id": name, "input_tokens": length, "max_tokens": output,
                   "stop_after_prefill": prefill, "stop_after_output": stop, "repetition": 0}
        if actual is None:
            actual = 0 if prefill is not None else stop if stop is not None else output
        result = {"case_id": name, "sequence_index": index, "first_request_in_process": index == 0,
                  "status": status, "reason": "length", "error": None, "error_phase": None,
                  "actual_output_tokens": actual, "emitted_eos": eos,
                  "stop_condition_reached": True if prefill is not None or stop is not None else None,
                  "input_digest": coverage.contracts.token_prefix_digest([self.seed[i % 2] for i in range(length)]),
                  "output_digest": coverage.contracts.token_prefix_digest([2 if eos and i == actual - 1 else 7 for i in range(actual)]),
                  "cleanup": {"generator_closed": True, "settled": True, "error": None}}
        first = len(self.records)
        if status == "skipped":
            result.update(reason="earlier_case_or_startup_stopped", input_digest=None, output_digest=None,
                          actual_output_tokens=0, stop_condition_reached=None,
                          cleanup={"generator_closed": None, "settled": None, "error": None})
        else:
            self.sample(name, "before_request")
            layers = []
            if output:
                self.epoch += 1
                self.peak = 200 + index * 20
                layers = [{"type": "KVCache", "offset": None if unknown_offset else length + actual + 1, "nbytes": 48},
                          {"type": "ArraysCache", "offset": None, "nbytes": None}]
                self.sample(name, "prefill_progress", position=0, layers=layers)
                if prefill != 0:
                    self.sample(name, "prefill_progress", position=prefill if prefill is not None else length, layers=layers)
                if actual:
                    self.sample(name, "first_output", count=1, layers=layers)
            if not interrupted:
                self.sample(name, "terminal", count=actual, layers=layers,
                            pressure="warning" if status == "blocked" else "normal")
                self.sample(name, "settled", count=actual)
            if output == 0:
                result["reason"] = "zero_output"
            elif prefill is not None:
                result["reason"] = "prefill_stop"
            elif eos:
                result["reason"] = "eos"
            elif stop is not None:
                result["reason"] = "output_stop"
            if status in ("blocked", "failed"):
                result.update(reason="host_pressure" if status == "blocked" else "execution_error",
                              error="The original test interrupted execution.")
            if interrupted:
                result.update(status="failed", reason="probe_interrupted", actual_output_tokens=None,
                              output_digest=None, cleanup={"generator_closed": None, "settled": None, "error": "Unconfirmed."})
        result["samples"] = {"first_record": first if len(self.records) > first else None,
                             "record_count": len(self.records) - first}
        self.report["plan"].append(planned)
        self.report["cases"].append(result)
        return result

    def write(self):
        payload = b"".join((json.dumps(sample, separators=(",", ":")) + "\n").encode() for sample in self.records)
        self.report["journal"] = {"file": "samples.jsonl", "record_count": len(self.records),
                                  "sha256": hashlib.sha256(payload).hexdigest()}
        if any(case["status"] != "completed" for case in self.report["cases"]):
            self.report["status"] = "blocked" if any(case["status"] == "blocked" for case in self.report["cases"]) else "failed"
            self.process.update(status=self.report["status"], returncode=1)
        self.report["process"] = copy.deepcopy(self.process)
        (self.directory / "calibration.json").write_text(json.dumps(self.report))
        (self.directory / "process.json").write_text(json.dumps(self.process))
        (self.directory / "samples.jsonl").write_bytes(payload)

    def analyze(self):
        self.write()
        return coverage.analyze(self.directory)

    def freeze_inputs(self, file_sources=False):
        messages = [{"role": "user", "content": "Compare the two small arrays. 输入。"}]
        document = {"format": "apxinf-calibration-inputs-v1", "profile": copy.deepcopy(self.profile),
                    "plan": copy.deepcopy(self.report["plan"]), "seed_messages": messages}
        for field, content in (("plan_source", document["plan"]), ("seed_source", messages)):
            payload = (json.dumps(content, indent=2).encode() if file_sources else
                       json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())
            document[field] = {"kind": "file" if file_sources else "default",
                               "path": str(self.directory / (field + "-removed.json")) if file_sources else None,
                               "sha256": hashlib.sha256(payload).hexdigest()}
        self.report["measurement_inputs"] = {"file": "measurement-inputs.json", "sha256": None,
                                              **{field: copy.deepcopy(document[field])
                                                 for field in ("plan_source", "seed_source", "seed_messages")}}
        if self.report["input_source"] is not None:
            self.report["input_source"].update(messages=copy.deepcopy(messages),
                                                template_options={"enable_thinking": False})
        self.replace_frozen(document)
        return document

    def replace_frozen(self, document, payload=None):
        if payload is None:
            payload = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        (self.directory / "measurement-inputs.json").write_bytes(payload)
        self.report["measurement_inputs"]["sha256"] = hashlib.sha256(payload).hexdigest()


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.artifact = Artifact(self.directory)

    def valid(self, report):
        self.assertEqual(report["validation"]["status"], "valid", report["validation"]["errors"])
        self.assertFalse(report["admission_approved"])
        self.assertFalse(report["full_memory_domain_established"])

    def invalid(self, report, text):
        self.assertEqual(report["validation"]["status"], "invalid")
        self.assertIn(text, " ".join(report["validation"]["errors"]))
        self.assertEqual(report["cases"], [])
        self.assertIsNone(report["shape_domain"])

    def test_completed_shape_keeps_physical_offsets_separate_from_runtime_position(self):
        self.artifact.case()
        report = self.artifact.analyze()
        self.valid(report)
        case = report["cases"][0]
        self.assertEqual(report["summary"], {"planned": 1, "executed": 1, "generation_observed": 1, "reached": 1})
        self.assertEqual(case["max_observed_cache_offset"], 8)
        self.assertNotEqual(case["max_observed_cache_offset"], 999)
        self.assertTrue(case["cache_layers"][1]["unknown_offset_observed"])
        self.assertIsNone(case["cache_layers"][1]["max_offset"])
        self.assertEqual(case["owned_peak_bytes"], 200)

    def test_unknown_offsets_do_not_become_output_positions(self):
        self.artifact.case(unknown_offset=True)
        report = self.artifact.analyze()
        self.valid(report)
        self.assertIsNone(report["cases"][0]["max_observed_cache_offset"])
        self.assertTrue(report["cases"][0]["reached"])

    def test_early_eos_does_not_reach_allowance_but_final_eos_does(self):
        self.artifact.case("early", output=4, actual=1, eos=True)
        self.artifact.case("last", output=4, actual=4, eos=True)
        report = self.artifact.analyze()
        self.valid(report)
        self.assertFalse(report["cases"][0]["reached"])
        self.assertIn("early_eos", report["cases"][0]["unreached_reasons"])
        self.assertTrue(report["cases"][1]["reached"])
        self.assertEqual(report["shape_domain"]["reached"]["shape_count"], 1)

    def test_stop_targets_do_not_count_as_full_allowance_shapes(self):
        self.artifact.case("prefill", prefill=0)
        self.artifact.case("output", output=8, stop=2)
        report = self.artifact.analyze()
        self.valid(report)
        self.assertEqual(report["summary"]["reached"], 2)
        self.assertEqual(report["cases"][0]["max_reported_prompt_position"], 0)
        self.assertEqual(report["shape_domain"]["planned"]["shape_count"], 0)
        self.assertEqual(report["shape_domain"]["reached"]["shape_count"], 0)

    def test_stop_flag_without_matching_checkpoint_does_not_reach_target(self):
        self.artifact.case(prefill=3)
        for sample in self.artifact.records:
            if sample["phase"] == "prefill_progress":
                sample["reported_prompt_position"] = 0
        report = self.artifact.analyze()
        self.valid(report)
        self.assertFalse(report["cases"][0]["reached"])

    def test_eos_cannot_claim_a_stop_that_its_result_did_not_take(self):
        result = self.artifact.case(output=8, actual=1, eos=True, stop=2)
        result["stop_condition_reached"] = True
        self.invalid(self.artifact.analyze(), "stop flag contradicts")

    def test_startup_pressure_or_missing_load_state_cannot_establish_reached(self):
        self.artifact.case()
        original = copy.deepcopy(self.artifact.records)
        for index in (0, 1):
            self.artifact.records = copy.deepcopy(original)
            self.artifact.records[index]["host"]["pressure"].update(state="warning", dispatch_value=2)
            self.invalid(self.artifact.analyze(), "before-load" if index == 0 else "startup")
        self.artifact.records = original
        self.artifact.report["load_started"] = False
        self.invalid(self.artifact.analyze(), "load_started")

    def test_pressure_evidence_must_match_the_state_and_error(self):
        self.artifact.case()
        corruptions = [
            {"state": "normal", "dispatch_value": 4, "error": None},
            {"state": "normal", "dispatch_value": None, "error": "Sensor unavailable."},
            {"state": "normal", "dispatch_value": 1, "error": "Sensor unavailable."},
            {"state": "normal", "dispatch_value": True, "error": None},
            {"state": "normal"},
            {"state": "unknown", "dispatch_value": 1, "error": "Sensor unavailable."},
            {"state": "unknown", "dispatch_value": None, "error": None},
            {"state": "unknown", "dispatch_value": 2**32, "error": "Invalid value."},
        ]
        for pressure in corruptions:
            with self.subTest(pressure=pressure):
                # Settlement is valid after blocked pressure, so tuple validation must reject this sample.
                self.artifact.records[-1]["host"]["pressure"] = pressure
                self.invalid(self.artifact.analyze(), "pressure")

    def test_unknown_pressure_retains_blocked_evidence_without_reached_claims(self):
        self.artifact.case(status="blocked", actual=1)
        for raw in (None, 0, 3, 2**32 - 1):
            with self.subTest(raw=raw):
                self.artifact.records[-2]["host"]["pressure"] = {
                    "state": "unknown", "dispatch_value": raw, "error": "Sensor unavailable."}
                report = self.artifact.analyze()
                self.valid(report)
                self.assertEqual(report["summary"]["reached"], 0)

    def test_blocked_case_cannot_hide_contradictory_unknown_pressure(self):
        self.artifact.case(status="blocked", actual=1)
        for raw, error in ((1, "Read failed."), (None, None), (None, ""), (True, "Read failed."), (-1, "Invalid value.")):
            with self.subTest(raw=raw, error=error):
                self.artifact.records[-2]["host"]["pressure"] = {
                    "state": "unknown", "dispatch_value": raw, "error": error}
                self.invalid(self.artifact.analyze(), "pressure")

    def test_legacy_profile_cannot_disable_or_omit_host_pressure_policy(self):
        self.artifact.case()
        self.assertNotIn("measurement_inputs", self.artifact.report)
        for policy in ("disabled", None):
            with self.subTest(policy=policy):
                if policy is None:
                    self.artifact.profile.pop("host_pressure_policy", None)
                else:
                    self.artifact.profile["host_pressure_policy"] = policy
                self.invalid(self.artifact.analyze(), "pressure policy")

    def test_offline_timestamps_keep_unsigned_64_bit_precision(self):
        self.artifact.case()
        for start in (2**53, 2**64 - len(self.artifact.records)):
            with self.subTest(start=start):
                for index, sample in enumerate(self.artifact.records):
                    sample["monotonic_ns"] = start + index
                report = self.artifact.analyze()
                self.valid(report)
                self.assertEqual(report["summary"]["reached"], 1)
                payload = (self.directory / "samples.jsonl").read_text().splitlines()
                self.assertEqual(json.loads(payload[-1])["monotonic_ns"], start + len(payload) - 1)

    def test_offline_timestamps_reject_values_outside_unsigned_64_bit_range(self):
        self.artifact.case()
        for value in (-1, 2**64, True, 1.5):
            with self.subTest(value=value):
                self.artifact.records[0]["monotonic_ns"] = value
                self.invalid(self.artifact.analyze(), "sample time")

    def test_terminal_and_settled_order_is_not_interchangeable(self):
        self.artifact.case()
        self.artifact.records[-2]["phase"] = "settled"
        self.artifact.records[-2]["memory"]["layers"] = []
        self.artifact.records[-2]["memory"]["cache_payload_bytes"] = 0
        self.artifact.records[-1]["phase"] = "terminal"
        self.invalid(self.artifact.analyze(), "phases violate execution order")

    def test_completed_generation_cannot_infer_execution_from_settled_epoch(self):
        result = self.artifact.case()
        records = [sample for sample in self.artifact.records if sample["phase"] not in ("prefill_progress", "first_output")]
        for index, sample in enumerate(records):
            sample["record_index"] = index
            if sample["phase"] == "terminal":
                sample["memory"]["peak_epoch"] = 0
        result["samples"]["record_count"] = 3
        self.artifact.records = records
        self.invalid(self.artifact.analyze(), "initial progress")

    def test_failed_observation_can_first_reveal_epoch_after_actual_settlement(self):
        result = self.artifact.case(status="failed", actual=0)
        records = [sample for sample in self.artifact.records if sample["phase"] not in ("prefill_progress", "terminal")]
        for index, sample in enumerate(records):
            sample["record_index"] = index
        result["samples"]["record_count"] = 2
        self.artifact.records = records
        report = self.artifact.analyze()
        self.valid(report)
        self.assertTrue(report["cases"][0]["generation_observed"])
        self.assertFalse(report["cases"][0]["reached"])

    def test_completed_case_requires_first_output_observation(self):
        result = self.artifact.case()
        self.artifact.records = [sample for sample in self.artifact.records if sample["phase"] != "first_output"]
        for index, sample in enumerate(self.artifact.records):
            sample["record_index"] = index
        result["samples"]["record_count"] -= 1
        self.invalid(self.artifact.analyze(), "first-output observations")

    def test_zero_output_and_next_before_request_do_not_own_inherited_peak(self):
        self.artifact.case("generation")
        self.artifact.peak = 800
        self.artifact.case("zero", output=0)
        self.artifact.case("next")
        report = self.artifact.analyze()
        self.valid(report)
        zero = report["cases"][1]
        self.assertTrue(zero["executed"] and zero["reached"])
        self.assertFalse(zero["generation_observed"])
        self.assertIsNone(zero["owned_peak_bytes"])
        group = report["peak_epochs"][1]
        self.assertEqual((group["run_id"], group["peak_epoch"]), (report["run_id"], 1))
        self.assertEqual((group["owned_peak_bytes"], group["observed_peak_bytes"]), (200, 800))
        self.assertEqual(len(group["inherited_record_indices"]), 4)
        self.assertEqual(report["cases"][2]["owned_peak_bytes"], 240)

    def test_blocked_and_interrupted_cases_keep_partial_counts_without_reached_claims(self):
        for interrupted in (False, True):
            with self.subTest(interrupted=interrupted):
                self.artifact = Artifact(self.directory)
                self.artifact.case("first", output=8, actual=1, status="blocked", interrupted=interrupted)
                self.artifact.case("skipped", status="skipped")
                report = self.artifact.analyze()
                self.valid(report)
                self.assertEqual(report["summary"]["executed"], 1)
                self.assertEqual(report["summary"]["reached"], 0)
                self.assertEqual(report["cases"][0]["actual_output_tokens"], None if interrupted else 1)
                self.assertEqual(report["cases"][0]["observed_output_tokens_lower_bound"], 1)

    def test_prior_success_remains_reached_when_a_later_case_fails(self):
        self.artifact.case("first")
        self.artifact.case("failed", status="failed", interrupted=True)
        self.artifact.case("skipped", status="skipped")
        report = self.artifact.analyze()
        self.valid(report)
        self.assertEqual(report["summary"]["reached"], 1)
        self.assertTrue(report["cases"][0]["reached"])

    def test_execution_after_failure_is_inconsistent_with_serial_calibration(self):
        self.artifact.case("failed", status="failed")
        self.artifact.case("later")
        self.invalid(self.artifact.analyze(), "after the matrix stops")

    def test_no_child_startup_failure_remains_valid_without_runtime_claims(self):
        self.artifact.case(status="skipped")
        self.artifact.records = []
        self.artifact.report.update(identity=None, input_source=None, run_samples=[])
        self.artifact.write()
        process = dict(self.artifact.process, pid=None, returncode=None)
        self.artifact.report["process"] = process
        (self.directory / "calibration.json").write_text(json.dumps(self.artifact.report))
        (self.directory / "process.json").write_text(json.dumps(process))
        report = coverage.analyze(self.directory)
        self.valid(report)
        self.assertEqual(report["summary"]["reached"], 0)

    def test_startup_pressure_block_records_no_generation_coverage(self):
        self.artifact.case(status="skipped")
        self.artifact.report["identity"] = None
        self.artifact.report["input_source"] = None
        self.artifact.records = self.artifact.records[:1]
        self.artifact.report["run_samples"] = [0]
        report = self.artifact.analyze()
        self.valid(report)
        self.assertEqual(report["summary"], {"planned": 1, "executed": 0, "generation_observed": 0, "reached": 0})
        self.assertEqual(report["peak_epochs"], [])

    def test_unreaped_parent_blocks_reached_even_when_probe_reports_completion(self):
        self.artifact.case()
        self.artifact.write()
        parent = dict(self.artifact.process, reaped=False, returncode=None, status="failed")
        (self.directory / "process.json").write_text(json.dumps(parent))
        report = coverage.analyze(self.directory)
        self.valid(report)
        self.assertFalse(report["cases"][0]["reached"])
        self.assertIn("process_reaping_unconfirmed", report["cases"][0]["unreached_reasons"])

    def test_default_short_allowances_leave_service_domain_missing(self):
        self.artifact = Artifact(self.directory, context=16384, max_output=2048)
        self.artifact.case("one", output=1)
        self.artifact.case("longer", output=32)
        self.artifact.case("zero", output=0)
        report = self.artifact.analyze()
        self.valid(report)
        domain = report["shape_domain"]
        self.assertEqual(domain["planned"]["missing_output_allowance_ranges"], [[2, 31], [33, 2048]])
        self.assertEqual(domain["reached"]["shape_count"], 3)
        self.assertEqual(domain["legal_shape_count"], sum(16384 - output for output in range(2049)))
        self.assertEqual(domain["reached"]["missing_shape_count"], domain["legal_shape_count"] - 3)

    def test_even_exhaustive_tiny_shape_enumeration_cannot_approve_memory(self):
        self.artifact = Artifact(self.directory, context=2, max_output=1)
        self.artifact.case("zero-one", length=1, output=0)
        self.artifact.case("zero-two", length=2, output=0)
        self.artifact.case("one", length=1, output=1)
        report = self.artifact.analyze()
        self.valid(report)
        self.assertTrue(report["shape_domain"]["reached"]["exact_shape_domain_exhausted"])
        self.assertEqual(report["shape_domain"]["reached"]["missing_output_allowance_ranges"], [])

    def test_repetitions_keep_separate_epochs_but_one_shape(self):
        self.artifact.case("first")
        self.artifact.case("repeat")
        self.artifact.report["plan"][1]["repetition"] = 1
        report = self.artifact.analyze()
        self.valid(report)
        self.assertEqual(report["summary"]["reached"], 2)
        self.assertEqual(report["shape_domain"]["reached"]["shape_count"], 1)
        self.assertEqual([case["peak_epoch"] for case in report["cases"]], [1, 2])

    def test_journal_byte_corruption_is_rejected_before_deriving_cases(self):
        self.artifact.case()
        self.artifact.write()
        with (self.directory / "samples.jsonl").open("ab") as destination:
            destination.write(b" ")
        self.invalid(coverage.analyze(self.directory), "Journal digest differs")

    def test_partial_final_line_is_rejected_even_with_matching_digest(self):
        self.artifact.case()
        self.artifact.write()
        payload = (self.directory / "samples.jsonl").read_bytes()[:-1]
        (self.directory / "samples.jsonl").write_bytes(payload)
        self.artifact.report["journal"]["sha256"] = hashlib.sha256(payload).hexdigest()
        (self.directory / "calibration.json").write_text(json.dumps(self.artifact.report))
        self.invalid(coverage.analyze(self.directory), "incomplete line")

    def test_duplicate_keys_and_nonfinite_json_numbers_are_rejected(self):
        self.artifact.case()
        for ending in ['"run_id":"other"', '"extra":NaN', '"extra":1e999']:
            with self.subTest(ending=ending):
                self.artifact.write()
                path = self.directory / "calibration.json"
                path.write_text(path.read_text()[:-1] + "," + ending + "}")
                report = coverage.analyze(self.directory)
                self.assertEqual(report["validation"]["status"], "invalid")

    def test_manifest_input_digest_and_process_mismatch_are_rejected(self):
        self.artifact.case()
        mutations = [
            (lambda: self.artifact.report["identity"].update(model_revision="f" * 64), "Model revision"),
            (lambda: self.artifact.report["cases"][0].update(input_digest="f" * 64), "Case input digest"),
        ]
        original = copy.deepcopy(self.artifact.report)
        for mutate, expected in mutations:
            self.artifact.report = copy.deepcopy(original)
            mutate()
            self.invalid(self.artifact.analyze(), expected)
        self.artifact.report = original
        self.artifact.write()
        parent = dict(self.artifact.process, pid=9999)
        (self.directory / "process.json").write_text(json.dumps(parent))
        self.invalid(coverage.analyze(self.directory), "process records differ")

    def test_case_range_index_and_unknown_reference_are_rejected(self):
        self.artifact.case()
        records = copy.deepcopy(self.artifact.records)
        for field, value, expected in [("record_index", 88, "indices differ"),
                                       ("case_id", "unknown", "unknown case"),
                                       ("monotonic_ns", 0, "timestamps decrease")]:
            self.artifact.records = copy.deepcopy(records)
            self.artifact.records[2][field] = value
            self.invalid(self.artifact.analyze(), expected)
        self.artifact.records = records
        self.artifact.report["cases"][0]["samples"]["first_record"] = 3
        self.invalid(self.artifact.analyze(), "sample range differs")

    def test_peak_reuse_skip_and_decrease_are_rejected(self):
        self.artifact.case("first")
        self.artifact.case("second")
        original = copy.deepcopy(self.artifact.records)
        for epoch, expected in [(1, "reuse another operation"), (3, "epochs skip")]:
            self.artifact.records = copy.deepcopy(original)
            for sample in self.artifact.records:
                if sample["case_id"] == "second" and sample["phase"] != "before_request":
                    sample["memory"]["peak_epoch"] = epoch
            self.invalid(self.artifact.analyze(), expected)
        self.artifact.records = original
        self.artifact.records[4]["memory"]["allocator"]["peak_bytes"] = 1
        self.invalid(self.artifact.analyze(), "counter decreases")

    def test_before_request_and_zero_output_cannot_create_peak_epoch(self):
        self.artifact.case(output=0)
        for sample in self.artifact.records:
            if sample["case_id"] is not None:
                sample["memory"]["peak_epoch"] = 1
        self.invalid(self.artifact.analyze(), "non-generation sample")

    def test_boolean_cache_metadata_and_output_counts_are_not_integers(self):
        self.artifact.case()
        original = copy.deepcopy(self.artifact.records)
        self.artifact.records[3]["memory"]["layers"][0]["offset"] = True
        self.invalid(self.artifact.analyze(), "Invalid integer: offset")
        self.artifact.records = original
        self.artifact.report["cases"][0]["actual_output_tokens"] = True
        self.invalid(self.artifact.analyze(), "Invalid integer: actual outputs")

    def test_source_recovery_error_prevents_derived_claims(self):
        self.artifact.case()
        self.artifact.report["recovery"]["errors"] = ["An incomplete record remained."]
        self.invalid(self.artifact.analyze(), "recovery reports errors")

    def test_legacy_artifact_explicitly_lacks_frozen_input_validation(self):
        self.artifact.case()
        (self.directory / "measurement-inputs.json").write_bytes(b"This unreferenced file is not JSON.")
        for present in (False, True):
            with self.subTest(null_metadata=present):
                if present:
                    self.artifact.report["measurement_inputs"] = None
                report = self.artifact.analyze()
                self.valid(report)
                self.assertEqual(report["frozen_input_validation"], "unavailable")
                self.assertNotIn("measurement-inputs.json", report["sources"])
                self.assertTrue(any("held-out" in text for text in report["validation"]["limitations"]))

    def test_frozen_default_inputs_are_verified_with_actual_file_digest(self):
        self.artifact.case()
        self.artifact.freeze_inputs()
        report = self.artifact.analyze()
        self.valid(report)
        self.assertEqual(report["frozen_input_validation"], "verified")
        payload = (self.directory / "measurement-inputs.json").read_bytes()
        self.assertEqual(report["sources"]["measurement-inputs.json"],
                         {"file": "measurement-inputs.json", "size_bytes": len(payload),
                          "sha256": hashlib.sha256(payload).hexdigest()})
        self.assertEqual(report["summary"]["reached"], 1)

    def test_file_source_provenance_uses_raw_hash_without_reading_removed_paths(self):
        self.artifact.case()
        frozen = self.artifact.freeze_inputs(file_sources=True)
        for field, content in (("plan_source", frozen["plan"]), ("seed_source", frozen["seed_messages"])):
            self.assertFalse(Path(frozen[field]["path"]).exists())
            self.assertNotEqual(frozen[field]["sha256"], hashlib.sha256(coverage.canonical_json(content)).hexdigest())
        report = self.artifact.analyze()
        self.valid(report)
        self.assertEqual(report["frozen_input_validation"], "verified")

    def test_declared_frozen_file_is_required(self):
        self.artifact.case()
        self.artifact.freeze_inputs()
        (self.directory / "measurement-inputs.json").unlink()
        report = self.artifact.analyze()
        self.invalid(report, "FileNotFoundError")
        self.assertEqual(report["frozen_input_validation"], "invalid")

    def test_frozen_filename_cannot_redirect_the_reader(self):
        self.artifact.case()
        self.artifact.freeze_inputs()
        self.artifact.report["measurement_inputs"]["file"] = "../outside.json"
        report = self.artifact.analyze()
        self.invalid(report, "frozen input filename")
        self.assertNotIn("../outside.json", report["sources"])

    def test_wrong_frozen_digest_rejects_otherwise_valid_inputs(self):
        self.artifact.case()
        self.artifact.freeze_inputs()
        self.artifact.report["measurement_inputs"]["sha256"] = "0" * 64
        self.invalid(self.artifact.analyze(), "Frozen input digest differs")

    def test_resigned_frozen_content_must_match_calibration_metadata(self):
        self.artifact.case()
        original = self.artifact.freeze_inputs(file_sources=True)
        changes = (("profile", "output_batch_tokens", True), ("plan", "max_tokens", 3.0),
                   ("seed_messages", "content", "A replacement input."),
                   ("plan_source", "sha256", "c" * 64), ("seed_source", "sha256", "d" * 64))
        for field, key, value in changes:
            with self.subTest(field=field):
                changed = copy.deepcopy(original)
                target = changed[field][0] if type(changed[field]) is list else changed[field]
                target[key] = value
                self.artifact.replace_frozen(changed)
                self.invalid(self.artifact.analyze(), f"Frozen {field} differs")

    def test_actual_prepared_messages_must_match_the_frozen_seed(self):
        self.artifact.case()
        self.artifact.freeze_inputs()
        self.artifact.report["input_source"]["messages"][0]["content"] = "A different prepared prompt."
        self.invalid(self.artifact.analyze(), "Prepared messages differ")

    def test_actual_template_options_reject_boolean_integer_equivalence(self):
        self.artifact.case()
        self.artifact.freeze_inputs()
        for template in ({"enable_thinking": 0}, {"enable_thinking": True},
                         {"enable_thinking": False, "extra": False}, None):
            with self.subTest(template=template):
                self.artifact.report["input_source"]["template_options"] = template
                self.invalid(self.artifact.analyze(), "Prepared template options differ")

    def test_default_provenance_requires_digest_of_frozen_content(self):
        self.artifact.case()
        frozen = self.artifact.freeze_inputs()
        frozen["seed_source"]["sha256"] = "a" * 64
        self.artifact.report["measurement_inputs"]["seed_source"] = copy.deepcopy(frozen["seed_source"])
        self.artifact.replace_frozen(frozen)
        self.invalid(self.artifact.analyze(), "Default source provenance differs")

    def test_file_source_requires_absolute_path_and_exact_provenance_fields(self):
        self.artifact.case()
        original = self.artifact.freeze_inputs(file_sources=True)
        for change in ({"path": "relative.json"}, {"kind": "remote"}, {"sha256": "A" * 64}, {"extra": True}):
            with self.subTest(change=change):
                frozen = copy.deepcopy(original)
                frozen["seed_source"].update(change)
                self.artifact.report["measurement_inputs"]["seed_source"] = copy.deepcopy(frozen["seed_source"])
                self.artifact.replace_frozen(frozen)
                report = self.artifact.analyze()
                self.assertEqual(report["validation"]["status"], "invalid", change)
                self.assertEqual(report["frozen_input_validation"], "invalid")

    def test_new_frozen_plan_schema_is_narrower_than_legacy(self):
        mutations = ({"extra": 1}, {"case_id": "x" * 129}, {"case_id": "含空格 case"},
                     {"input_tokens": True}, {"max_tokens": 3.0}, {"repetition": 4},
                     {"input_tokens": 64}, {"stop_after_prefill": 1, "stop_after_output": 1},
                     {"max_tokens": 0, "stop_after_prefill": 0}, {"stop_after_output": 4})
        for change in mutations:
            with self.subTest(change=change):
                artifact = Artifact(self.directory)
                artifact.case(status="skipped")
                artifact.report["plan"][0].update(change)
                artifact.report["cases"][0]["case_id"] = artifact.report["plan"][0]["case_id"]
                artifact.freeze_inputs(file_sources=True)
                self.invalid(artifact.analyze(), "ValueError")
        artifact = Artifact(self.directory)
        artifact.case(status="skipped")
        del artifact.report["plan"][0]["stop_after_output"]
        artifact.freeze_inputs(file_sources=True)
        self.invalid(artifact.analyze(), "six plan fields")

    def test_frozen_profile_cannot_add_fields_or_disable_pressure_checks(self):
        for change in ({"extra": 1}, {"host_pressure_policy": "disabled"}):
            with self.subTest(change=change):
                artifact = Artifact(self.directory)
                artifact.case()
                artifact.profile.update(change)
                artifact.freeze_inputs()
                self.invalid(artifact.analyze(), "pressure policy" if "host_pressure_policy" in change else "frozen profile fields or policy")

    def test_blocked_before_load_can_verify_frozen_inputs_without_claiming_execution(self):
        self.artifact.case(status="skipped")
        self.artifact.report.update(identity=None, input_source=None, run_samples=[0], load_started=False)
        self.artifact.records = self.artifact.records[:1]
        self.artifact.records[0]["host"]["pressure"].update(state="warning", dispatch_value=2)
        self.artifact.freeze_inputs()
        self.artifact.write()
        self.artifact.report["status"] = self.artifact.report["process"]["status"] = "blocked"
        self.artifact.process["status"] = "blocked"
        (self.directory / "calibration.json").write_text(json.dumps(self.artifact.report))
        (self.directory / "process.json").write_text(json.dumps(self.artifact.process))
        report = coverage.analyze(self.directory)
        self.valid(report)
        self.assertEqual(report["frozen_input_validation"], "verified")
        self.assertEqual(report["summary"], {"planned": 1, "executed": 0, "generation_observed": 0, "reached": 0})

    def test_seed_validator_applies_even_without_prepared_input(self):
        self.artifact.case(status="skipped")
        self.artifact.report.update(identity=None, input_source=None)
        frozen = self.artifact.freeze_inputs(file_sources=True)
        frozen["seed_messages"] = [{"role": "invented", "content": "Invalid seed."}]
        self.artifact.report["measurement_inputs"]["seed_messages"] = copy.deepcopy(frozen["seed_messages"])
        self.artifact.replace_frozen(frozen)
        self.invalid(self.artifact.analyze(), "ContractError: unsupported enum value")

    def test_seed_messages_retain_the_public_frame_byte_limit(self):
        self.artifact.case(status="skipped")
        self.artifact.report.update(identity=None, input_source=None)
        frozen = self.artifact.freeze_inputs(file_sources=True)
        frozen["seed_messages"] = [{"role": "user", "content": "s" * 1_048_576}]
        self.artifact.report["measurement_inputs"]["seed_messages"] = copy.deepcopy(frozen["seed_messages"])
        self.artifact.replace_frozen(frozen)
        self.invalid(self.artifact.analyze(), "ContractError: frame exceeds byte limit")

    def test_frozen_document_and_metadata_require_exact_fields(self):
        for target, expected in (("document", "Unsupported frozen input format"),
                                 ("metadata", "metadata fields differ")):
            for remove in (False, True):
                with self.subTest(target=target, remove=remove):
                    artifact = Artifact(self.directory)
                    artifact.case()
                    frozen = artifact.freeze_inputs()
                    value = frozen if target == "document" else artifact.report["measurement_inputs"]
                    if remove:
                        del value["seed_source"]
                    else:
                        value["extra"] = None
                    artifact.replace_frozen(frozen)
                    self.invalid(artifact.analyze(), expected)

    def test_frozen_duplicate_keys_and_nonfinite_numbers_are_rejected_after_resigning(self):
        self.artifact.case()
        frozen = self.artifact.freeze_inputs()
        base = (self.directory / "measurement-inputs.json").read_bytes()
        for payload, expected in ((b'{"format":"duplicate",' + base[1:], "Duplicate JSON key"),
                                  (base.replace(b'"repetition":0', b'"repetition":NaN'), "Non-finite"),
                                  (base.replace(b'"repetition":0', b'"repetition":1e999'), "finite range")):
            with self.subTest(expected=expected):
                self.artifact.replace_frozen(frozen, payload)
                self.invalid(self.artifact.analyze(), expected)

    def test_frozen_snapshot_rejects_non_utf8_encoding(self):
        self.artifact.case()
        frozen = self.artifact.freeze_inputs()
        self.artifact.replace_frozen(frozen, json.dumps(frozen).encode("utf-16"))
        self.invalid(self.artifact.analyze(), "UnicodeDecodeError")

    def test_frozen_snapshot_uses_four_mib_limit_even_with_matching_digest(self):
        self.artifact.case()
        frozen = self.artifact.freeze_inputs()
        base = (self.directory / "measurement-inputs.json").read_bytes()
        self.artifact.replace_frozen(frozen, base + b" " * (4 * 1024**2 + 1 - len(base)))
        self.invalid(self.artifact.analyze(), "byte limit: measurement-inputs.json")

    def test_skipped_zero_output_is_not_an_observed_allowance(self):
        self.artifact.case(output=0, status="skipped")
        report = self.artifact.analyze()
        self.valid(report)
        self.assertFalse(report["cases"][0]["full_allowance_observed"])

    def test_reaped_parent_cannot_retain_process_group_members(self):
        self.artifact.case()
        self.artifact.process["cleanup"] = {"returncode": 0, "remaining_members": [99]}
        self.invalid(self.artifact.analyze(), "remaining members")

    def test_cli_exclusive_output_preserves_all_source_bytes(self):
        self.artifact.case()
        self.artifact.write()
        before = {path.name: path.read_bytes() for path in self.directory.iterdir()}
        destination = self.directory / "coverage.json"
        self.assertEqual(coverage.main([str(self.directory), "--output", str(destination)]), 0)
        self.valid(json.loads(destination.read_text()))
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(coverage.main([str(self.directory), "--output", str(destination)]), 1)
        for name, payload in before.items():
            self.assertEqual((self.directory / name).read_bytes(), payload)
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(coverage.main([str(self.directory)]), 0)
        self.valid(json.loads(stream.getvalue()))

    def test_missing_source_returns_invalid_json_and_nonzero_exit(self):
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(coverage.main([str(self.directory)]), 1)
        self.invalid(json.loads(stream.getvalue()), "FileNotFoundError")


if __name__ == "__main__":
    unittest.main()
