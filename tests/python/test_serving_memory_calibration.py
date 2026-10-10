"""Original CPU tests for calibration accounting. No model or GPU is used."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "benchmarks/serving"
sys.path.insert(0, str(BENCHMARKS))
try:
    spec = importlib.util.spec_from_file_location("memory_calibration_under_test", BENCHMARKS / "memory_calibration.py")
    calibration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calibration)
finally:
    sys.path.pop(0)


def planned_case(name="original-case", *, length=6, output=3, prefill=None, emitted=None):
    return {"case_id": name, "input_tokens": length, "max_tokens": output,
            "stop_after_prefill": prefill, "stop_after_output": emitted, "repetition": 0}


def host(state="normal", error=None):
    return {"pressure": {"state": state, "dispatch_value": {"normal": 1, "warning": 2, "critical": 4}.get(state),
                         "error": error}, "process_peak_rss_bytes": 4096, "swap": {"used_bytes": 128}}


class OriginalIterator:
    def __init__(self, runtime, command, progress):
        self.runtime = runtime
        self.command = command
        self.progress = progress
        self.started = False
        self.closed = False
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        runtime = self.runtime
        if self.closed:
            raise AssertionError("A closed iterator must not execute.")
        if not self.started:
            self.started = True
            runtime.peak_epoch += 1
            length = len(self.command["token_ids"])
            for consumed in sorted({0, min(2, length), length}):
                runtime.events.append(("progress", consumed))
                runtime.consumed_position = consumed
                self.progress(consumed)
        if runtime.fail_after is not None and self.index == runtime.fail_after:
            raise runtime.execution_error
        tokens = runtime.tokens
        if self.index >= self.command["max_tokens"] or (tokens is not None and self.index >= len(tokens)):
            raise StopIteration
        token = tokens[self.index] if tokens is not None else 10 + self.index
        self.index += 1
        runtime.emitted = self.index
        runtime.consumed_position = len(self.command["token_ids"]) + self.index
        runtime.events.append(("yield", token))
        return token

    def close(self):
        self.runtime.events.append(("close",))
        if self.runtime.close_error is not None:
            raise self.runtime.close_error
        self.closed = True


class OriginalRuntime:
    def __init__(self):
        self.max_context = 64
        self.vocab_size = 128
        self.eos_token_ids = [2]
        self.manifest = {"model": "original-calibration-cpu-fixture"}
        self.runtime = {"python": "cpu-test", "mlx": "not-loaded", "mlx_lm": "not-loaded"}
        self.seed = [3, 4]
        self.events = []
        self.tokens = None
        self.iterator = None
        self.emitted = 0
        self.consumed_position = 0
        self.peak_epoch = 0
        self.is_settled = False
        self.fail_after = None
        self.execution_error = RuntimeError("Original generation failure.")
        self.close_error = None
        self.settle_error = None
        self.finish_error = None
        self.decode_error = None
        self.inspect_hook = None

    def prepare(self, command):
        self.events.append(("prepare", command))
        return self.seed

    def generate(self, command, progress):
        self.events.append(("generate", command))
        self.emitted = 0
        self.is_settled = False
        self.iterator = OriginalIterator(self, command, progress)
        return self.iterator

    def decode(self, token, eos=False):
        self.events.append(("decode", token, eos))
        if self.decode_error is not None:
            raise self.decode_error
        return "" if eos else str(token)

    def finish_text(self):
        self.events.append(("finish_text",))
        if self.finish_error is not None:
            raise self.finish_error
        return ""

    def inspect_memory(self):
        self.events.append(("inspect",))
        if self.inspect_hook is not None:
            self.inspect_hook(self)
        return {"allocator": {"active_bytes": 1024, "cache_bytes": 0, "peak_bytes": 2048},
                "peak_epoch": self.peak_epoch, "runtime_consumed_position": self.consumed_position,
                "cache_payload_bytes": 0 if self.is_settled else 64, "layers": []}

    def settle(self):
        self.events.append(("settle",))
        if self.iterator is not None and not self.iterator.closed:
            raise AssertionError("Settlement preceded generator closure.")
        if self.settle_error is not None:
            raise self.settle_error
        self.is_settled = True


class PlanTests(unittest.TestCase):
    def test_explicit_plan_keeps_order_and_per_case_context_allowance(self):
        plan = [planned_case("fresh-long", length=14336, output=2048),
                planned_case("edge", length=16383, output=1),
                planned_case("zero", length=16384, output=0),
                planned_case("late-prefill", length=8192, output=2048, prefill=4096),
                planned_case("late-output", length=14336, output=2048, emitted=1024)]
        plan[-1]["repetition"] = 3
        validated = calibration.validate_plan(plan, 16384, 2048)
        self.assertEqual(validated, plan)
        validated[0]["max_tokens"] = 1
        self.assertEqual(plan[0]["max_tokens"], 2048)

    def test_explicit_plan_rejects_ambiguous_schema_and_ids(self):
        cases = [None, {}, [], [None], [planned_case(), planned_case()],
                 [planned_case(str(index)) for index in range(257)]]
        for name in ("", "-starts-with-dash", "has space", "line\n", "é", "a" * 129, 3):
            cases.append([planned_case(name)])
        for field in planned_case():
            incomplete = planned_case()
            incomplete.pop(field)
            cases.append([incomplete])
        cases.append([{**planned_case(), "provider": "other"}])
        for plan in cases:
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                calibration.validate_plan(plan, 16384, 2048)
        self.assertEqual(len(calibration.validate_plan(
            [planned_case(str(index)) for index in range(256)], 16384, 2048)), 256)

    def test_explicit_plan_checks_strict_integers_and_stop_boundaries(self):
        invalid = {
            "input_tokens": [True, 6.0, 0, 16385], "max_tokens": [False, 3.0, -1, 2049],
            "repetition": [True, 0.0, -1, 4], "stop_after_prefill": [False, 0.0, -1, 7],
            "stop_after_output": [True, 1.0, 0, 4],
        }
        for field, values in invalid.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    calibration.validate_plan([{**planned_case(), field: value}], 16384, 2048)
        for case in (planned_case(length=16383, output=2), planned_case(prefill=1, emitted=1),
                     planned_case(output=0, prefill=0), planned_case(output=0, emitted=1)):
            with self.subTest(case=case), self.assertRaises(ValueError):
                calibration.validate_plan([case], 16384, 2048)
        for boundary in (0, 6):
            calibration.validate_plan([planned_case(prefill=boundary)], 16384, 2048)

    def test_default_plan_covers_chunk_cache_and_context_boundaries(self):
        plan = calibration.make_plan(16384, 256, 32, 2)
        self.assertEqual(len({case["case_id"] for case in plan}), len(plan))
        self.assertEqual({case["repetition"] for case in plan}, {0, 1})
        lengths = {case["input_tokens"] for case in plan}
        self.assertTrue({1, 255, 256, 257, 2048, 8192, 16352} <= lengths)
        self.assertTrue(any(case["input_tokens"] + case["max_tokens"] == 16384 for case in plan))
        for case in plan:
            self.assertGreater(case["input_tokens"], 0)
            self.assertLessEqual(case["input_tokens"] + case["max_tokens"], 16384)
        self.assertEqual(sum(case["max_tokens"] == 0 for case in plan), 2)
        self.assertEqual(sum(case["stop_after_prefill"] is not None for case in plan), 2)
        self.assertEqual(sum(case["stop_after_output"] is not None for case in plan), 2)

    def test_smallest_context_uses_a_zero_prefill_stop_without_exceeding_context(self):
        plan = calibration.make_plan(2, 8192, 1, 1)
        self.assertTrue(all(case["input_tokens"] + case["max_tokens"] <= 2 for case in plan))
        stopped = next(case for case in plan if case["stop_after_prefill"] is not None)
        self.assertEqual((stopped["input_tokens"], stopped["stop_after_prefill"]), (1, 0))

    def test_plan_rejects_invalid_ranges_and_duplicate_lengths(self):
        invalid = [
            (1, 16, 1, 1, None), (131073, 16, 1, 1, None),
            (64, 0, 1, 1, None), (64, 8193, 1, 1, None),
            (64, 16, 0, 1, None), (64, 16, 64, 1, None),
            (64, 16, 4, 0, None), (64, 16, 4, 5, None),
            (64, 16, 4, 1, []), (64, 16, 4, 1, [2, 2]),
            (64, 16, 4, 1, [61]), (64, 16, 4, 1, [True]),
            (64, 16, 4, 1, list(range(1, 18))), (True, 16, 1, 1, None),
        ]
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                calibration.make_plan(*arguments)


class InputValidationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.model = self.directory / "model"
        self.model.mkdir()

    def args(self, *options):
        return calibration.parser().parse_args([
            "--model", str(self.model), "--output-dir", str(self.directory / "output"), *options])

    def test_bounded_json_preserves_exact_byte_digest(self):
        payload = b' [ { "role" : "user", "content": "original" } ]\n'
        path = self.directory / "messages.json"
        path.write_bytes(payload)
        value, digest = calibration.read_json_file(path, len(payload))
        self.assertEqual(value, [{"role": "user", "content": "original"}])
        self.assertEqual(digest, hashlib.sha256(payload).hexdigest())
        with self.assertRaisesRegex(ValueError, "byte limit"):
            calibration.read_json_file(path, len(payload) - 1)

    def test_json_rejects_duplicate_nested_keys_and_non_json_numbers(self):
        path = self.directory / "input.json"
        invalid = [b'{"a":1,"a":2}', b'[{"outer":{"same":1,"same":2}}]',
                   b'[NaN]', b'[Infinity]', b'[-Infinity]', b'[1e999]', b'["\xff"]', b'[] trailing']
        for payload in invalid:
            with self.subTest(payload=payload):
                path.write_bytes(payload)
                with self.assertRaises(ValueError):
                    calibration.read_json_file(path, 1024)

    def test_seed_uses_public_prepare_message_schema_and_frame_limit(self):
        messages = [{"role": "assistant", "content": "", "tool_calls": [
            {"id": "original-call", "type": "function", "function": {"name": "test", "arguments": {}}}]},
            {"role": "tool", "content": "Original response.", "tool_call_id": "original-call"}]
        self.assertEqual(calibration.validate_seed_messages(messages), messages)
        invalid = [[], {"messages": calibration.SEED_MESSAGES},
                   [{"role": "user", "content": 1}], [{"role": "user", "content": "x", "tools": []}],
                   [{"role": "user", "content": "x", "enable_thinking": True}],
                   [{"role": "user", "content": "x", "provider": "other"}],
                   [{"role": "user", "content": "x" * calibration.INPUT_FILE_LIMIT}]]
        for messages in invalid:
            with self.subTest(size=len(str(messages))), self.assertRaises(ValueError):
                calibration.validate_seed_messages(messages)

    def test_default_cli_matches_the_original_plan_and_seed(self):
        args = self.args()
        plan = calibration.validate_arguments(args)
        self.assertEqual(plan, calibration.make_plan(16384, 256, 32, 2))
        self.assertEqual(args.prepared_seed_messages, calibration.SEED_MESSAGES)
        self.assertEqual(args.plan_source["kind"], "default")
        self.assertIsNone(args.seed_source["path"])
        self.assertFalse(args.output_dir.exists())

    def test_explicit_plan_conflicts_even_with_explicit_default_options(self):
        path = self.directory / "plan.json"
        path.write_text(json.dumps([planned_case()]))
        for flags in (("--max-tokens", "32"), ("--repetitions", "2"), ("--input-lengths", "6")):
            with self.subTest(flags=flags), self.assertRaisesRegex(ValueError, "conflicts"):
                calibration.validate_arguments(self.args("--plan-file", str(path), *flags))

    def test_external_seed_records_original_source_content_and_digest(self):
        path = self.directory / "seed.json"
        payload = '[ {"role":"user","content":"Original 种子 input."} ]\n'.encode()
        path.write_bytes(payload)
        args = self.args("--seed-messages", str(path))
        calibration.validate_arguments(args)
        self.assertEqual(args.seed_source, {"kind": "file", "path": str(path.resolve()),
                                           "sha256": hashlib.sha256(payload).hexdigest()})
        self.assertEqual(args.prepared_seed_messages, json.loads(payload))


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "samples.jsonl"
        self.journal = calibration.Journal(self.path)
        self.addCleanup(self.journal.close)
        self.runtime = OriginalRuntime()
        self.persisted = []

    def records(self):
        return [json.loads(line) for line in self.path.read_text().splitlines()]

    def measure(self, case=None, probe=lambda: host()):
        case = case or planned_case()
        result = calibration.empty_case(case, 0)
        calibration.measure_case(self.runtime, case, result, [3, 4], self.journal, probe)
        return result

    def calibrate(self, plan=None, probe=lambda: host(), factory=None, context=64, **options):
        return calibration.calibrate(
            model_path=Path("/original-cpu-model"), requested_profile={"max_context": context},
            plan=plan or [planned_case()], runtime_factory=factory or (lambda: self.runtime),
            journal=self.journal, persist=lambda value: self.persisted.append(json.loads(json.dumps(value))),
            host_probe=probe, hardware={"model": "CPU-test"}, **options,
        )

    def test_custom_messages_reach_prepare_and_report_with_fixed_options(self):
        messages = [{"role": "system", "content": "Original CPU calibration input."},
                    {"role": "user", "content": "Continue the numbered sequence."}]
        report = self.calibrate(seed_messages=messages)
        prepared = next(event[1] for event in self.runtime.events if event[0] == "prepare")
        self.assertEqual(prepared, {"messages": messages, "tools": [], "template_options": {"enable_thinking": False}})
        self.assertEqual(report["input_source"]["messages"], messages)
        self.assertEqual(report["status"], "completed")
        prepared["messages"][0]["content"] = "Changed after preparation."
        self.assertEqual(report["input_source"]["messages"][0]["content"], messages[0]["content"])

    def test_before_load_block_preserves_frozen_input_metadata(self):
        metadata = {"file": "measurement-inputs.json", "sha256": "a" * 64,
                    "seed_messages": [{"role": "user", "content": "Original blocked input."}]}
        report = self.calibrate(probe=lambda: host("warning"), measurement_inputs=metadata,
                                seed_messages=metadata["seed_messages"],
                                factory=lambda: self.fail("No runtime may load under pressure."))
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["load_started"])
        self.assertEqual(report["measurement_inputs"], metadata)
        self.assertEqual(self.persisted[0]["measurement_inputs"], metadata)
        metadata["seed_messages"][0]["content"] = "Later mutation."
        self.assertNotEqual(report["measurement_inputs"], metadata)

    def test_journal_flushes_exclusively_and_hashes_exact_records(self):
        self.journal.append(phase="original", memory=None)
        self.assertEqual(self.records()[0]["record_index"], 0)
        with self.assertRaises(FileExistsError):
            calibration.Journal(self.path)
        with self.assertRaises(ValueError):
            self.journal.append(phase="invalid", value=float("nan"))
        description = self.journal.description()
        self.assertEqual(description["record_count"], 1)
        self.assertEqual(description["sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())

    def test_zero_output_does_not_call_generate_or_reset_peak(self):
        result = self.measure(planned_case(output=0))
        self.assertEqual((result["status"], result["reason"], result["actual_output_tokens"]),
                         ("completed", "zero_output", 0))
        self.assertFalse(any(event[0] == "generate" for event in self.runtime.events))
        self.assertTrue(all(record["memory"]["peak_epoch"] == 0 for record in self.records()))
        self.assertEqual([record["phase"] for record in self.records()], ["before_request", "terminal", "settled"])
        self.assertTrue(result["cleanup"]["settled"])

    def test_prefill_stop_closes_before_settlement_and_has_no_outputs(self):
        result = self.measure(planned_case(prefill=2))
        self.assertEqual((result["status"], result["reason"]), ("completed", "prefill_stop"))
        self.assertTrue(result["stop_condition_reached"])
        self.assertEqual(result["actual_output_tokens"], 0)
        self.assertFalse(any(event[0] == "yield" for event in self.runtime.events))
        self.assertLess(self.runtime.events.index(("close",)), self.runtime.events.index(("settle",)))
        self.assertEqual([record["reported_prompt_position"] for record in self.records()
                          if record["phase"] == "prefill_progress"], [0, 2])

    def test_output_stop_counts_actual_yields_and_retains_partial_digest(self):
        result = self.measure(planned_case(output=8, emitted=3))
        self.assertEqual((result["status"], result["reason"], result["actual_output_tokens"]),
                         ("completed", "output_stop", 3))
        self.assertTrue(result["stop_condition_reached"])
        self.assertEqual(result["output_digest"], calibration.token_digest([10, 11, 12]))
        self.assertEqual([event[1] for event in self.runtime.events if event[0] == "yield"], [10, 11, 12])

    def test_early_eos_is_counted_and_can_prevent_a_requested_stop(self):
        self.runtime.tokens = [10, 2, 12]
        result = self.measure(planned_case(output=8, emitted=4))
        self.assertEqual((result["status"], result["reason"], result["actual_output_tokens"]),
                         ("completed", "eos", 2))
        self.assertTrue(result["emitted_eos"])
        self.assertFalse(result["stop_condition_reached"])
        self.assertIn(("decode", 2, True), self.runtime.events)
        self.assertEqual(result["output_digest"], calibration.token_digest([10, 2]))

    def test_decode_failure_does_not_erase_an_already_yielded_eos(self):
        self.runtime.tokens = [2]
        self.runtime.decode_error = RuntimeError("original decode failed")
        result = self.measure()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["actual_output_tokens"], 1)
        self.assertEqual(result["output_digest"], calibration.token_digest([2]))
        self.assertTrue(result["emitted_eos"])
        self.assertIn("original decode failed", result["error"])
        self.assertTrue(result["cleanup"]["settled"])

    def test_short_non_eos_generator_fails_and_settles(self):
        self.runtime.tokens = [10]
        result = self.measure(planned_case(output=3))
        self.assertEqual(result["status"], "failed")
        self.assertIn("without EOS", result["error"])
        self.assertEqual(result["actual_output_tokens"], 1)
        self.assertTrue(result["cleanup"]["generator_closed"])
        self.assertTrue(result["cleanup"]["settled"])

    def test_preload_pressure_preserves_plan_without_loading(self):
        plan = [planned_case("first"), planned_case("second")]
        def forbidden_load():
            raise AssertionError("Pressure must be checked before loading.")
        report = self.calibrate(plan, probe=lambda: host("unknown", "original sensor read failed"), factory=forbidden_load)
        self.assertEqual((report["status"], report["error_phase"], report["load_started"]),
                         ("blocked", "before_load", False))
        self.assertIsNone(report["identity"])
        self.assertEqual(len(report["cases"]), len(plan))
        self.assertTrue(all(case["status"] == "skipped" for case in report["cases"]))
        self.assertEqual(self.records()[0]["host"]["pressure"]["error"], "original sensor read failed")
        self.assertIsNone(self.records()[0]["memory"])
        self.assertEqual(self.persisted[0]["plan"], plan)

    def test_postload_pressure_records_identity_and_cleans_loaded_runtime(self):
        readings = iter([host(), host("critical")])
        report = self.calibrate(probe=lambda: next(readings, host()))
        self.assertEqual((report["status"], report["error_phase"]), ("blocked", "loaded"))
        self.assertTrue(report["load_started"])
        self.assertIsNotNone(report["identity"])
        self.assertTrue(report["run_cleanup"]["settled"])
        self.assertFalse(any(event[0] in ("prepare", "generate") for event in self.runtime.events))
        self.assertEqual(self.records()[-1]["host"]["pressure"]["state"], "critical")

    def test_loading_failure_keeps_planned_cases_and_does_not_invent_cleanup(self):
        def fail_loading():
            raise RuntimeError("original load failed")
        report = self.calibrate([planned_case("first"), planned_case("later")], factory=fail_loading)
        self.assertEqual((report["status"], report["error_phase"]), ("failed", "load"))
        self.assertTrue(report["load_started"])
        self.assertIsNone(report["identity"])
        self.assertIsNone(report["run_cleanup"]["settled"])
        self.assertEqual([case["status"] for case in report["cases"]], ["skipped", "skipped"])
        self.assertEqual(self.records()[0]["phase"], "before_load")

    def test_invalid_prepared_seed_never_reaches_generation(self):
        self.runtime.seed = [True]
        report = self.calibrate()
        self.assertEqual((report["status"], report["error_phase"]), ("failed", "prepare_seed"))
        self.assertFalse(any(event[0] == "generate" for event in self.runtime.events))
        self.assertEqual(report["cases"][0]["status"], "skipped")
        self.assertTrue(report["run_cleanup"]["settled"])

    def test_cleanup_checkpoint_pressure_blocks_remaining_cases_after_safe_settlement(self):
        for boundary in ["terminal", "settled"]:
            with self.subTest(boundary=boundary):
                self.runtime = OriginalRuntime()
                def probe():
                    iterator = self.runtime.iterator
                    closed = iterator is not None and iterator.closed
                    blocked = closed and (self.runtime.is_settled if boundary == "settled"
                                          else not self.runtime.is_settled)
                    return host("critical" if blocked else "normal")
                report = self.calibrate([planned_case("first-" + boundary), planned_case("later-" + boundary)], probe=probe)
                first, later = report["cases"]
                self.assertEqual((report["status"], first["status"], later["status"]),
                                 ("blocked", "blocked", "skipped"))
                self.assertEqual(first["error_phase"], boundary)
                self.assertTrue(first["cleanup"]["settled"])
                blocked = [record for record in self.records()
                           if record["case_id"] == first["case_id"] and record["phase"] == boundary]
                self.assertEqual(blocked[-1]["host"]["pressure"]["state"], "critical")

    def test_each_yield_checks_pressure_and_preserves_unsampled_failure(self):
        def probe():
            active = self.runtime.iterator is not None and not self.runtime.iterator.closed
            return host("warning" if active and self.runtime.emitted == 3 else "normal")
        report = self.calibrate([planned_case("first", output=20), planned_case("later")], probe=probe)
        first, later = report["cases"]
        self.assertEqual((first["status"], first["actual_output_tokens"]), ("blocked", 3))
        self.assertTrue(first["cleanup"]["settled"])
        self.assertEqual(later["status"], "skipped")
        failed_read = next(record for record in self.records() if record["host"]["pressure"]["state"] == "warning")
        self.assertEqual((failed_read["phase"], failed_read["output_tokens"]), ("output_checkpoint", 3))
        self.assertIsNone(failed_read["memory"])
        self.assertTrue(failed_read["memory_error"])
        self.assertEqual(first["output_digest"], calibration.token_digest([10, 11, 12]))

    def test_pressure_failure_at_sample_limit_preserves_state_without_exceeding_cap(self):
        # Before request, three progress reports, and first output fill five records.
        def probe():
            active = self.runtime.iterator is not None and not self.runtime.iterator.closed
            return host("warning" if active and self.runtime.emitted == 2 else "normal")
        with patch.object(calibration, "SAMPLE_LIMIT", 5):
            result = self.measure(planned_case(output=3), probe=probe)
        self.assertLessEqual(len(self.records()), 5)
        self.assertEqual(result["actual_output_tokens"], 2)
        self.assertIn("warning", result["error"])
        self.assertIn("sample_limit", result["error"])
        self.assertTrue(result["cleanup"]["settled"])

    def test_progress_sample_limit_stops_execution_but_still_settles(self):
        with patch.object(calibration, "SAMPLE_LIMIT", 2):
            result = self.measure()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["actual_output_tokens"], 0)
        self.assertIn("sample limit", result["error"])
        self.assertLessEqual(len(self.records()), 2)
        self.assertTrue(result["cleanup"]["settled"])

    def test_close_failure_preserves_outputs_and_prevents_settlement_and_later_cases(self):
        self.runtime.close_error = RuntimeError("original close failed")
        report = self.calibrate([planned_case("first"), planned_case("later")])
        first, later = report["cases"]
        self.assertEqual(first["status"], "failed")
        self.assertEqual(first["actual_output_tokens"], 3)
        self.assertFalse(first["cleanup"]["generator_closed"])
        self.assertIsNot(first["cleanup"]["settled"], True)
        self.assertIn("original close failed", first["cleanup"]["error"])
        self.assertEqual(later["status"], "skipped")
        self.assertFalse(any(event[0] == "settle" for event in self.runtime.events))

    def test_settle_failure_does_not_claim_cleanup_or_execute_later_cases(self):
        self.runtime.settle_error = RuntimeError("original settlement failed")
        report = self.calibrate([planned_case("first"), planned_case("later")])
        first, later = report["cases"]
        self.assertEqual(first["status"], "failed")
        self.assertTrue(first["cleanup"]["generator_closed"])
        self.assertFalse(first["cleanup"]["settled"])
        self.assertIn("original settlement failed", first["cleanup"]["error"])
        self.assertFalse(report["run_cleanup"]["settled"])
        self.assertEqual(later["status"], "skipped")
        self.assertFalse(any(record["phase"] == "settled" for record in self.records()))

    def test_terminal_observation_failure_still_attempts_real_settlement(self):
        def fail_after_close(runtime):
            if runtime.iterator is not None and runtime.iterator.closed and not runtime.is_settled:
                raise RuntimeError("original terminal observation failed")
        self.runtime.inspect_hook = fail_after_close
        result = self.measure()
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["cleanup"]["settled"])
        self.assertTrue(self.runtime.is_settled)
        self.assertIn("original terminal observation failed", result["cleanup"]["error"])

    def test_multiple_cleanup_errors_do_not_overwrite_original_failure_evidence(self):
        self.runtime.fail_after = 1
        self.runtime.finish_error = RuntimeError("original finalization failed")
        self.runtime.settle_error = RuntimeError("original settlement failed")
        def fail_after_close(runtime):
            if runtime.iterator is not None and runtime.iterator.closed:
                raise RuntimeError("original memory inspection failed")
        self.runtime.inspect_hook = fail_after_close
        result = self.measure()
        self.assertEqual(result["status"], "failed")
        self.assertIn("Original generation failure", result["error"])
        for message in ["original finalization failed", "original memory inspection failed", "original settlement failed"]:
            self.assertIn(message, result["cleanup"]["error"])
        self.assertFalse(result["cleanup"]["settled"])
        self.assertTrue(result["cleanup"]["generator_closed"])

    def test_settled_observation_error_does_not_deny_completed_resource_settlement(self):
        def fail_after_settlement(runtime):
            if runtime.is_settled:
                raise RuntimeError("original settled observation failed")
        self.runtime.inspect_hook = fail_after_settlement
        result = self.measure()
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["cleanup"]["settled"])
        self.assertIn("original settled observation failed", result["cleanup"]["error"])

    def test_interruption_preserves_partial_samples_digests_and_all_case_slots(self):
        self.runtime.fail_after = 2
        self.runtime.execution_error = KeyboardInterrupt("original interruption")
        report = self.calibrate([planned_case("first", output=8), planned_case("later")])
        first, later = report["cases"]
        self.assertEqual((report["status"], first["status"], later["status"]), ("failed", "failed", "skipped"))
        self.assertEqual(first["actual_output_tokens"], 2)
        self.assertIn("KeyboardInterrupt", first["error"])
        self.assertEqual(first["output_digest"], calibration.token_digest([10, 11]))
        self.assertTrue(first["cleanup"]["settled"])
        self.assertEqual(report["journal"]["sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(report["journal"]["record_count"], len(self.records()))
        self.assertEqual(self.persisted[-1], report)
        self.assertEqual(len(self.persisted[0]["cases"]), 2)
        records = self.records()
        begin = first["samples"]["first_record"]
        end = begin + first["samples"]["record_count"]
        self.assertTrue(all(record["case_id"] == "first" for record in records[begin:end]))

    def test_success_keeps_independent_case_records_and_reset_epochs(self):
        report = self.calibrate([planned_case("first"), planned_case("zero", output=0), planned_case("last")])
        self.assertEqual(report["status"], "completed")
        self.assertEqual([case["status"] for case in report["cases"]], ["completed"] * 3)
        self.assertEqual([case["first_request_in_process"] for case in report["cases"]], [True, False, False])
        self.assertEqual(report["input_source"]["kind"], "synthetic_tiled_prepared_tokens")
        self.assertFalse(report["input_source"]["quality_evidence"])
        for case in report["cases"]:
            samples = [record for record in self.records() if record["case_id"] == case["case_id"]]
            self.assertEqual(len(samples), case["samples"]["record_count"])
        self.assertEqual(self.runtime.peak_epoch, 2)


if __name__ == "__main__":
    unittest.main()
