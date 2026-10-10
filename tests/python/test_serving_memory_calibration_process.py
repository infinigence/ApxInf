"""CPU checks for calibration process failure and recovery. No model is loaded."""

import contextlib
import copy
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "benchmarks/serving"
spec = importlib.util.spec_from_file_location(
    "memory_calibration_process_test", DIRECTORY / "memory_calibration.py")
calibration = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(DIRECTORY))
try:
    spec.loader.exec_module(calibration)
finally:
    sys.path.pop(0)


def encoded_sample(index, case_id, output=0, phase="terminal"):
    sample = {"record_index": index, "case_id": case_id, "phase": phase,
              "monotonic_ns": index + 1, "output_tokens": output,
              "reported_prompt_position": None, "memory": None, "memory_error": None,
              "host": {"pressure": {"state": "normal", "dispatch_value": 1, "error": None}}}
    return (json.dumps(sample, sort_keys=True, separators=(",", ":")) + "\n").encode()


class ChildStub:
    """Represent a child without starting an interpreter or touching the GPU."""
    pid = 43121

    def __init__(self, on_wait):
        self.returncode = None
        self.on_wait = on_wait

    def wait(self, timeout):
        self.on_wait(self, timeout)
        return self.returncode


class CalibrationProcessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.model = self.directory / "model"
        self.model.mkdir()
        self.lock = self.directory / "metal.lock"
        self.lock.touch()
        self.output = self.directory / "output"
        self.args = calibration.parser().parse_args([
            "--model", str(self.model), "--output-dir", str(self.output),
            "--python", sys.executable, "--metal-lock", str(self.lock),
            "--max-context", "32", "--max-output-tokens", "4", "--max-tokens", "4",
            "--prefill-step-size", "4", "--input-lengths", "8", "--repetitions", "1",
            "--timeout", "0.25",
        ])
        self.plan = calibration.make_plan(32, 4, 4, 1, [8])
        self.fallback = calibration.initial_report(self.model, calibration.profile(self.args), self.plan)

    def read(self, filename):
        return json.loads((self.output / filename).read_text())

    def persist(self, report, payload):
        self.output.mkdir(exist_ok=True)
        (self.output / "samples.jsonl").write_bytes(payload)
        calibration.save(self.output / "calibration.json", report)

    def process(self, **changes):
        result = {"pid": ChildStub.pid, "returncode": 0, "reaped": True,
                  "status": "running", "error": None}
        result.update(changes)
        return result

    def complete_report(self):
        metadata = self.read("calibration.json").get("measurement_inputs") if (self.output / "calibration.json").exists() else None
        report = calibration.initial_report(self.model, calibration.profile(self.args), self.plan,
                                            measurement_inputs=metadata)
        payload = b""
        for index, result in enumerate(report["cases"]):
            output = self.plan[index]["max_tokens"]
            payload += encoded_sample(index, result["case_id"], output)
            result.update(status="completed", reason="fixture_complete", actual_output_tokens=output,
                          input_digest="b" * 64, output_digest="a" * 64,
                          samples={"first_record": index, "record_count": 1},
                          cleanup={"generator_closed": True, "settled": True, "error": None})
        report["status"] = "completed"
        report["load_started"] = True
        report["identity"] = {
            "model_revision": "c" * 64, "manifest": {"fixture": "original CPU observations"},
            "runtime": {"fixture": "CPU only"}, "effective_context": 32,
            "vocab_size": 64, "eos_token_ids": [63],
        }
        report["input_source"] = {
            "kind": "synthetic_tiled_prepared_tokens",
            "messages": metadata["seed_messages"] if metadata else calibration.SEED_MESSAGES,
            "template_options": calibration.TEMPLATE_OPTIONS,
            "seed_tokens": [1, 2], "seed_digest": "d" * 64, "quality_evidence": False,
        }
        report["run_cleanup"]["settled"] = True
        report["journal"] = {"file": "samples.jsonl", "record_count": len(report["cases"]),
                             "sha256": hashlib.sha256(payload).hexdigest()}
        self.persist(report, payload)
        return report, payload

    def invoke(self, on_wait=None, cleanup=None, spawn_error=None, on_start=None):
        child = ChildStub(on_wait or (lambda selected, timeout: setattr(selected, "returncode", 0)))

        def stop(selected):
            self.assertIs(selected, child)
            if cleanup is not None:
                return cleanup(selected)
            selected.returncode = 0 if selected.returncode is None else selected.returncode
            return {"pgid": selected.pid, "returncode": selected.returncode, "remaining_members": []}

        matrix = types.SimpleNamespace(stop_owned_group=Mock(side_effect=stop))

        def start(command, **options):
            if spawn_error is not None:
                raise spawn_error
            self.assertTrue(options["start_new_session"])
            self.assertEqual(len(options["pass_fds"]), 1)
            descriptor = options["pass_fds"][0]
            actual, expected = os.fstat(descriptor), self.lock.stat()
            self.assertEqual((actual.st_dev, actual.st_ino), (expected.st_dev, expected.st_ino))
            self.assertEqual(command[command.index("--lock-fd") + 1], str(descriptor))
            with self.lock.open("rb") as rival:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(rival, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if on_start is not None:
                on_start(command, options)
            return child

        with patch.dict(sys.modules, {"metal_matrix": matrix}), \
                patch.object(calibration.subprocess, "Popen", side_effect=start) as popen, \
                patch.object(calibration, "create_runtime", side_effect=AssertionError("No model may load.")), \
                contextlib.redirect_stdout(io.StringIO()):
            status = calibration.run(self.args)
        # The parent must release its own descriptor after finalization.
        with self.lock.open("rb") as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(owner, fcntl.LOCK_UN)
        return status, child, matrix.stop_owned_group, popen

    def use_explicit_inputs(self):
        self.plan = [{"case_id": "fresh-long-original", "input_tokens": 24, "max_tokens": 8,
                      "stop_after_prefill": None, "stop_after_output": None, "repetition": 0},
                     {"case_id": "edge-original", "input_tokens": 31, "max_tokens": 1,
                      "stop_after_prefill": None, "stop_after_output": None, "repetition": 3}]
        self.messages = [{"role": "user", "content": "Continue this original CPU sequence."}]
        self.plan_path, self.seed_path = self.directory / "source-plan.json", self.directory / "source-seed.json"
        self.plan_payload = (json.dumps(self.plan, indent=2) + "\n").encode()
        self.seed_payload = (json.dumps(self.messages, indent=2) + "\n").encode()
        self.plan_path.write_bytes(self.plan_payload)
        self.seed_path.write_bytes(self.seed_payload)
        self.args = calibration.parser().parse_args([
            "--model", str(self.model), "--output-dir", str(self.output), "--python", sys.executable,
            "--metal-lock", str(self.lock), "--max-context", "32", "--max-output-tokens", "8",
            "--prefill-step-size", "4", "--plan-file", str(self.plan_path),
            "--seed-messages", str(self.seed_path), "--timeout", "0.25"])

    def test_parent_freezes_inputs_before_launch_and_child_ignores_mutated_sources(self):
        self.use_explicit_inputs()
        checked = []

        def started(command, options):
            for forbidden in ("--plan-file", "--seed-messages", "--max-tokens", "--input-lengths", "--repetitions",
                              str(self.plan_path), str(self.seed_path)):
                self.assertNotIn(forbidden, command)
            document = self.read(calibration.FROZEN_INPUT_FILE)
            self.assertEqual(document["plan"], self.plan)
            self.assertEqual(document["seed_messages"], self.messages)
            self.plan_path.write_text("[]")
            self.seed_path.unlink()
            child_args = calibration.parser().parse_args(command[3:])
            self.assertEqual(calibration.validate_arguments(child_args), self.plan)
            self.assertEqual(child_args.prepared_seed_messages, self.messages)
            self.assertEqual(child_args.measurement_inputs, self.read("calibration.json")["measurement_inputs"])
            checked.append(True)

        def waited(child, timeout):
            self.complete_report()
            child.returncode = 0

        status, _, _, _ = self.invoke(on_start=started, on_wait=waited)
        self.assertEqual(checked, [True])
        self.assertEqual(status, 0)

    def test_custom_spawn_failure_retains_plan_seed_and_exact_provenance(self):
        self.use_explicit_inputs()
        status, _, _, _ = self.invoke(spawn_error=OSError("Original launch failure."))
        self.assertEqual(status, 1)
        report = self.read("calibration.json")
        self.assertEqual(report["plan"], self.plan)
        self.assertEqual([item["case_id"] for item in report["cases"]], [item["case_id"] for item in self.plan])
        self.assertTrue(all(item["status"] == "skipped" for item in report["cases"]))
        self.assertFalse(report["load_started"])
        metadata = report["measurement_inputs"]
        self.assertEqual(metadata["seed_messages"], self.messages)
        self.assertEqual(metadata["plan_source"]["sha256"], hashlib.sha256(self.plan_payload).hexdigest())
        self.assertEqual(metadata["seed_source"]["sha256"], hashlib.sha256(self.seed_payload).hexdigest())
        self.assertEqual(metadata["sha256"], hashlib.sha256((self.output / calibration.FROZEN_INPUT_FILE).read_bytes()).hexdigest())

    def test_child_rejects_snapshot_hash_profile_and_source_inconsistency(self):
        self.use_explicit_inputs()
        plan = calibration.validate_arguments(self.args)
        self.output.mkdir()
        metadata = calibration.freeze_inputs(self.args, plan)
        path = self.output / calibration.FROZEN_INPUT_FILE
        original = path.read_bytes()
        child_flags = ["--model", str(self.model), "--output-dir", str(self.output), "--probe",
                       "--max-context", "32", "--max-output-tokens", "8", "--prefill-step-size", "4"]

        def validate(digest):
            args = calibration.parser().parse_args([*child_flags, "--frozen-inputs-sha256", digest])
            return calibration.validate_arguments(args)

        self.assertEqual(validate(metadata["sha256"]), plan)
        path.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "digest differs"):
            validate(metadata["sha256"])
        for field, value in (("profile", {**calibration.profile(self.args), "output_batch_tokens": True}),
                             ("seed_source", {"kind": "file", "path": "relative.json", "sha256": "a" * 64}),
                             ("plan", [{**plan[0], "input_tokens": True}])):
            with self.subTest(field=field):
                document = json.loads(original)
                document[field] = value
                payload = calibration.json_bytes(document)
                path.write_bytes(payload)
                with self.assertRaises(ValueError):
                    validate(hashlib.sha256(payload).hexdigest())
        path.write_bytes(original)
        with self.assertRaises(FileExistsError):
            calibration.freeze_inputs(self.args, plan)
        self.assertEqual(path.read_bytes(), original)

    def test_invalid_external_input_precedes_lock_and_output_creation(self):
        self.use_explicit_inputs()
        self.seed_path.write_text('[{"role":"user","content":"x","content":"y"}]')
        with patch.object(calibration.os, "open", side_effect=AssertionError("Lock access preceded input validation.")), \
                patch.object(calibration.subprocess, "Popen") as start:
            with self.assertRaisesRegex(ValueError, "repeats"):
                calibration.run(self.args)
        start.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_parent_rejects_changed_frozen_report_fields_and_prepared_input(self):
        self.use_explicit_inputs()
        plan = calibration.validate_arguments(self.args)
        self.output.mkdir()
        metadata = calibration.freeze_inputs(self.args, plan)
        fallback = calibration.initial_report(self.model, calibration.profile(self.args), plan, measurement_inputs=metadata)
        changes = [lambda report: report["plan"][0].update(max_tokens=7),
                   lambda report: report["requested_profile"].update(output_batch_tokens=True),
                   lambda report: report["measurement_inputs"].update(sha256="b" * 64),
                   lambda report: report["input_source"].update(messages=calibration.SEED_MESSAGES),
                   lambda report: report["input_source"].update(template_options={"enable_thinking": 0})]
        for index, change in enumerate(changes):
            with self.subTest(change=index):
                calibration.save(self.output / "calibration.json", fallback)
                report, payload = self.complete_report()
                change(report)
                calibration.save(self.output / "calibration.json", report)
                self.assertEqual(calibration.finalize_process(self.output, self.process(), fallback), 1)
                self.assertEqual(self.read("process.json")["status"], "failed")
                self.assertIn("differ", self.read("process.json")["error"])
                self.assertEqual((self.output / "samples.jsonl").read_bytes(), payload)

    def test_timeout_remains_failed_when_probe_exits_zero_during_cleanup(self):
        def waited(child, timeout):
            self.complete_report()
            self.assertEqual(timeout, 0.25)
            raise subprocess.TimeoutExpired("original-probe", timeout)

        status, child, stop, _ = self.invoke(on_wait=waited)
        self.assertEqual(status, 1)
        self.assertEqual(child.returncode, 0)
        stop.assert_called_once_with(child)
        report, process = self.read("calibration.json"), self.read("process.json")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(process["status"], "failed")
        self.assertIn("timeout", process["error"])
        self.assertTrue(process["reaped"])
        self.assertTrue(all(case["status"] == "completed" for case in report["cases"]))

    def test_timeout_recovers_flushed_inflight_samples_without_output_claim(self):
        stored = {}

        def waited(child, timeout):
            report = self.read("calibration.json")
            first, active = report["cases"][:2]
            prefix = encoded_sample(0, first["case_id"], 1)
            first.update(status="completed", reason="length", actual_output_tokens=1,
                         output_digest="b" * 64, samples={"first_record": 0, "record_count": 1},
                         cleanup={"generator_closed": True, "settled": True, "error": None})
            active["status"] = "running"
            report["journal"] = {"file": "samples.jsonl", "record_count": 1,
                                 "sha256": hashlib.sha256(prefix).hexdigest()}
            stored["payload"] = prefix + encoded_sample(1, active["case_id"], 0, "before_request")
            stored["payload"] += encoded_sample(2, active["case_id"], 3, "output_checkpoint")
            self.persist(report, stored["payload"])
            raise subprocess.TimeoutExpired("original-probe", timeout)

        def killed(child):
            child.returncode = -9
            return {"pgid": child.pid, "returncode": -9, "remaining_members": []}

        status, _, _, _ = self.invoke(on_wait=waited, cleanup=killed)
        self.assertEqual(status, 1)
        report = self.read("calibration.json")
        self.assertEqual(report["journal"]["record_count"], 3)
        self.assertEqual(report["journal"]["sha256"], hashlib.sha256(stored["payload"]).hexdigest())
        self.assertEqual((self.output / "samples.jsonl").read_bytes(), stored["payload"])
        self.assertEqual(report["cases"][0]["status"], "completed")
        self.assertEqual(report["cases"][0]["actual_output_tokens"], 1)
        active = report["cases"][1]
        self.assertEqual(active["status"], "failed")
        self.assertEqual(active["samples"], {"first_record": 1, "record_count": 2})
        self.assertIsNone(active["actual_output_tokens"])
        self.assertIsNone(active["output_digest"])
        self.assertIsNone(active["cleanup"]["settled"])
        self.assertTrue(all(case["status"] == "skipped" for case in report["cases"][2:]))

    def test_spawn_failure_keeps_the_complete_plan_and_parent_error(self):
        status, _, stop, popen = self.invoke(spawn_error=OSError("Cannot execute the probe."))
        self.assertEqual(status, 1)
        popen.assert_called_once()
        stop.assert_not_called()
        report, process = self.read("calibration.json"), self.read("process.json")
        self.assertEqual(report["plan"], self.plan)
        self.assertEqual(len(report["cases"]), len(self.plan))
        self.assertTrue(all(case["status"] == "skipped" for case in report["cases"]))
        self.assertEqual(report["status"], "failed")
        self.assertIn("Cannot execute", report["error"])
        self.assertIsNone(process["pid"])
        self.assertIsNone(process["returncode"])
        self.assertTrue(process["reaped"])
        self.assertEqual(report["journal"]["record_count"], 0)

    def test_cleanup_exception_keeps_parent_failure_without_touching_child_report(self):
        stored = {}

        def waited(child, timeout):
            self.complete_report()
            stored["report"] = (self.output / "calibration.json").read_bytes()
            child.returncode = 0

        def failed_cleanup(child):
            raise RuntimeError("Process group inspection failed.")

        status, _, _, _ = self.invoke(on_wait=waited, cleanup=failed_cleanup)
        self.assertEqual(status, 1)
        self.assertEqual((self.output / "calibration.json").read_bytes(), stored["report"])
        process = self.read("process.json")
        self.assertEqual(process["status"], "failed")
        self.assertFalse(process["reaped"])
        self.assertIn("inspection failed", process["cleanup_error"])
        self.assertIn("inspection failed", process["error"])

    def test_unconfirmed_reaping_never_recovers_or_rewrites_child_files(self):
        self.output.mkdir()
        path = self.output / "calibration.json"
        path.write_bytes(b'{"status":"running","writer":"probe"}\n')
        journal = self.output / "samples.jsonl"
        journal.write_bytes(b'{"incomplete":')
        before = path.read_bytes(), journal.read_bytes()
        process = self.process(reaped=False, returncode=None)
        with patch.object(calibration, "recover_journal", side_effect=AssertionError("Probe still owns the files.")):
            status = calibration.finalize_process(self.output, process, self.fallback)
        self.assertEqual(status, 1)
        self.assertEqual((path.read_bytes(), journal.read_bytes()), before)
        self.assertEqual(self.read("process.json")["status"], "failed")
        self.assertFalse(self.read("process.json")["reaped"])

    def test_invalid_report_shape_keeps_raw_bytes_and_parent_reaping_outcome(self):
        self.output.mkdir()
        damaged = copy.deepcopy(self.fallback)
        damaged["cases"][0]["samples"] = []
        missing_cases = copy.deepcopy(self.fallback)
        missing_cases["cases"] = []
        payloads = [b"[]\n", b"{}\n", b"null\n", b'{"incomplete":',
                    json.dumps(damaged).encode(), json.dumps(missing_cases).encode()]
        journal = encoded_sample(0, self.plan[0]["case_id"], 1, "first_output")
        for raw in payloads:
            with self.subTest(raw=raw[:64]):
                (self.output / "calibration.json").write_bytes(raw)
                (self.output / "samples.jsonl").write_bytes(journal)
                calibration.save(self.output / "process.json", self.process(status="running", reaped=False, returncode=None))
                process = self.process(returncode=-9, error="Parent terminated the probe after timeout.")
                self.assertEqual(calibration.finalize_process(self.output, process, self.fallback), 1)
                saved, report = self.read("process.json"), self.read("calibration.json")
                self.assertEqual(saved["status"], "failed")
                self.assertTrue(saved["reaped"])
                self.assertEqual(saved["returncode"], -9)
                self.assertEqual(saved["error"], "Parent terminated the probe after timeout.")
                self.assertEqual(report["process"], saved)
                self.assertEqual(report["plan"], self.plan)
                self.assertEqual(len(report["cases"]), len(self.plan))
                self.assertEqual(report["journal"]["record_count"], 1)
                self.assertIsNone(report["cases"][0]["actual_output_tokens"])
                self.assertIsNone(report["identity"])
                preserved = saved["invalid_report"]
                self.assertIsNone(preserved["error"])
                self.assertEqual((self.output / preserved["file"]).read_bytes(), raw)
                self.assertEqual((self.output / "samples.jsonl").read_bytes(), journal)

    def test_invalid_child_report_does_not_leave_running_parent_record(self):
        def waited(child, timeout):
            (self.output / "calibration.json").write_bytes(b"[]\n")
            child.returncode = 0

        status, _, _, _ = self.invoke(on_wait=waited)
        self.assertEqual(status, 1)
        process, report = self.read("process.json"), self.read("calibration.json")
        self.assertTrue(process["reaped"])
        self.assertEqual(process["returncode"], 0)
        self.assertEqual(process["status"], "failed")
        self.assertEqual(report["measurement_inputs"]["seed_messages"], calibration.SEED_MESSAGES)
        self.assertEqual(report["plan"], self.plan)

    def test_report_preservation_error_does_not_erase_parent_outcome(self):
        self.output.mkdir()
        (self.output / "calibration.json").write_bytes(b"[]\n")
        with patch.object(Path, "rename", side_effect=OSError("Original preservation failure.")):
            self.assertEqual(calibration.finalize_process(self.output, self.process(), self.fallback), 1)
        process = self.read("process.json")
        self.assertTrue(process["reaped"])
        self.assertEqual(process["status"], "failed")
        self.assertIsNone(process["invalid_report"]["file"])
        self.assertIn("Original preservation failure", process["invalid_report"]["error"])
        self.assertEqual(self.read("calibration.json")["plan"], self.plan)

    def test_parent_outcome_survives_final_report_write_failure(self):
        self.complete_report()
        original_save = calibration.save

        def fail_report(path, value):
            if path.name == "calibration.json":
                raise OSError("Original report write failure.")
            return original_save(path, value)

        with patch.object(calibration, "save", side_effect=fail_report):
            self.assertEqual(calibration.finalize_process(self.output, self.process(), self.fallback), 1)
        process = self.read("process.json")
        self.assertTrue(process["reaped"])
        self.assertEqual(process["returncode"], 0)
        self.assertEqual(process["status"], "failed")
        self.assertIn("Original report write failure", process["error"])

    def test_observation_timestamp_preserves_u64_and_rejects_out_of_range_clock(self):
        journal = calibration.Journal(self.directory / "clock-samples.jsonl")
        self.addCleanup(journal.close)
        host = lambda: {"pressure": {"state": "normal", "dispatch_value": 1, "error": None}}
        for value in (0, 2**53, 2**64 - 1):
            with patch.object(calibration.time, "monotonic_ns", return_value=value):
                calibration.observe(journal, None, host, "before_load")
        original = journal.path.read_bytes()
        self.assertEqual([json.loads(line)["monotonic_ns"] for line in original.splitlines()], [0, 2**53, 2**64 - 1])
        for value in (-1, 2**64, True):
            with self.subTest(value=value), patch.object(calibration.time, "monotonic_ns", return_value=value):
                with self.assertRaisesRegex(ValueError, "timestamp"):
                    calibration.observe(journal, None, host, "before_load")
                self.assertEqual(journal.path.read_bytes(), original)

    def test_bad_or_incomplete_journal_is_preserved_and_fails(self):
        case_id = self.plan[0]["case_id"]
        tails = [b'{"record_index":1', b'{invalid json}\n', b'[]\n',
                 encoded_sample(9, case_id), encoded_sample(1, "unknown-case")]
        for tail in tails:
            with self.subTest(tail=tail):
                report = calibration.initial_report(self.model, calibration.profile(self.args), self.plan)
                report["cases"][0]["status"] = "running"
                payload = encoded_sample(0, case_id, 1, "first_output") + tail
                self.persist(report, payload)
                status = calibration.finalize_process(self.output, self.process(returncode=-9), self.fallback)
                self.assertEqual(status, 1)
                recovered = self.read("calibration.json")
                self.assertEqual(recovered["status"], "failed")
                self.assertTrue(recovered["recovery"]["errors"])
                self.assertEqual(recovered["journal"]["record_count"], 1)
                self.assertEqual(recovered["journal"]["sha256"], hashlib.sha256(payload).hexdigest())
                self.assertEqual((self.output / "samples.jsonl").read_bytes(), payload)
                self.assertIsNone(recovered["cases"][0]["actual_output_tokens"])

    def test_completed_report_with_changed_journal_fails(self):
        report, payload = self.complete_report()
        changed = payload.replace(b'"output_tokens":1', b'"output_tokens":2', 1)
        self.assertNotEqual(changed, payload)
        (self.output / "samples.jsonl").write_bytes(changed)
        status = calibration.finalize_process(self.output, self.process(), self.fallback)
        self.assertEqual(status, 1)
        final = self.read("calibration.json")
        self.assertEqual(final["status"], "failed")
        self.assertIn("does not match", " ".join(final["recovery"]["errors"]))
        self.assertEqual((self.output / "samples.jsonl").read_bytes(), changed)

    def test_confirmed_completion_retains_raw_artifact_without_budget_approval(self):
        def waited(child, timeout):
            self.complete_report()
            child.returncode = 0

        status, _, _, _ = self.invoke(on_wait=waited)
        self.assertEqual(status, 0)
        report = self.read("calibration.json")
        self.assertEqual(report["artifact_kind"], "raw_observations")
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["recovery"]["errors"], [])
        self.assertTrue(report["process"]["reaped"])
        self.assertNotIn("approved_envelope", report)

    def test_completed_report_requires_final_cases_identity_and_settlement(self):
        def running_case(report):
            report["cases"][0]["status"] = "running"

        def failed_case(report):
            report["cases"][0]["status"] = "failed"

        def unsettled_case(report):
            report["cases"][0]["cleanup"]["settled"] = None

        def unclosed_generator(report):
            report["cases"][0]["cleanup"]["generator_closed"] = False

        def missing_identity(report):
            report["identity"] = None

        def missing_input_source(report):
            report["input_source"] = None

        def unsettled_run(report):
            report["run_cleanup"]["settled"] = False

        for damage in (running_case, failed_case, unsettled_case, unclosed_generator,
                       missing_identity, missing_input_source, unsettled_run):
            with self.subTest(condition=damage.__name__):
                report, payload = self.complete_report()
                damage(report)
                calibration.save(self.output / "calibration.json", report)
                status = calibration.finalize_process(self.output, self.process(), self.fallback)
                self.assertEqual(status, 1)
                self.assertEqual(self.read("calibration.json")["status"], "failed")
                self.assertEqual(self.read("process.json")["status"], "failed")
                self.assertEqual((self.output / "samples.jsonl").read_bytes(), payload)

    def test_existing_lock_prevents_probe_and_artifact_creation(self):
        with self.lock.open("rb") as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(calibration.subprocess, "Popen") as popen:
                with self.assertRaises(BlockingIOError):
                    calibration.run(self.args)
                popen.assert_not_called()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
