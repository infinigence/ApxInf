#!/usr/bin/env python3
"""Record serial MLX memory observations without approving an admission budget."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

import host_memory

ROOT = Path(__file__).resolve().parents[2]
WORKER_DIRECTORY = ROOT / "python/apxinf/apxinf/serving"
SEED_MESSAGES = [{"role": "user", "content": "Calibrate memory for this local model."}]
TEMPLATE_OPTIONS = {"enable_thinking": False}
SAMPLE_LIMIT = 4096
INPUT_FILE_LIMIT = 1024**2
FROZEN_INPUT_LIMIT = 4 * 1024**2
FROZEN_INPUT_FILE = "measurement-inputs.json"
MAX_TIMESTAMP_NS = 2**64 - 1
CASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class PressureBlocked(Exception):
    pass


class PlannedStop(Exception):
    pass


def integer(value, label, lower=0, upper=9007199254740991):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"The {label} is outside its integer range.")
    return value


def make_plan(max_context, prefill_step_size, max_tokens, repetitions, input_lengths=None):
    integer(max_context, "context limit", 2, 131072)
    integer(prefill_step_size, "prefill chunk", 1, 8192)
    integer(max_tokens, "output allowance", 1, min(65536, max_context - 1))
    integer(repetitions, "repetition count", 1, 4)
    largest = max_context - max_tokens
    lengths = input_lengths if input_lengths is not None else sorted({
        1, min(largest, max(1, prefill_step_size - 1)), min(largest, prefill_step_size),
        min(largest, prefill_step_size + 1), min(largest, 255), min(largest, 256),
        min(largest, 257), min(largest, 2048), min(largest, 8192), largest,
    })
    if not lengths or len(lengths) > 16 or len(set(lengths)) != len(lengths):
        raise ValueError("Supply one to sixteen unique input lengths.")
    for length in lengths:
        integer(length, "input length", 1, largest)
    result = []

    def add(kind, length, output, repeat, prefill=None, emitted=None):
        result.append({"case_id": f"{kind}-p{length}-o{output}-r{repeat}",
                       "input_tokens": length, "max_tokens": output,
                       "stop_after_prefill": prefill, "stop_after_output": emitted,
                       "repetition": repeat})

    for repeat in range(repetitions):
        for length in lengths:
            for output in sorted({1, max_tokens}):
                add("complete", length, output, repeat)
        add("zero", min(lengths), 0, repeat)
        prefill_length = min(largest, max(2, prefill_step_size + 2))
        add("prefill-stop", prefill_length, max_tokens, repeat,
            prefill=min(prefill_step_size, prefill_length - 1))
        add("output-stop", min(32, largest), max_tokens, repeat, emitted=min(4, max_tokens))
    return result


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def read_json_file(path, limit):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"The input repeats a JSON key: {key}.")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"The input contains a non-finite number: {value}.")

    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("The input contains an overflowing JSON number.")
        return parsed

    with Path(path).open("rb") as source:
        payload = source.read(limit + 1)
    if len(payload) > limit:
        raise ValueError("The input file exceeds its byte limit.")
    value = json.loads(payload.decode("utf-8"), object_pairs_hook=object_pairs,
                       parse_constant=invalid_constant, parse_float=finite_float)
    return value, hashlib.sha256(payload).hexdigest()


def validate_plan(plan, max_context, max_output_tokens):
    integer(max_context, "context limit", 2, 131072)
    integer(max_output_tokens, "output limit", 1, 65536)
    if type(plan) is not list or not 1 <= len(plan) <= 256:
        raise ValueError("The explicit plan requires one to 256 cases.")
    fields = {"case_id", "input_tokens", "max_tokens", "stop_after_prefill", "stop_after_output", "repetition"}
    identifiers = set()
    for case in plan:
        if type(case) is not dict or set(case) != fields:
            raise ValueError("A case must contain exactly the six plan fields.")
        name = case["case_id"]
        if type(name) is not str or CASE_ID.fullmatch(name) is None or name in identifiers:
            raise ValueError("Case IDs must be valid and unique.")
        identifiers.add(name)
        length = integer(case["input_tokens"], "input length", 1, max_context)
        output = integer(case["max_tokens"], "case output allowance", 0, max_output_tokens)
        if length + output > max_context:
            raise ValueError("The case input and output allowance exceed context.")
        integer(case["repetition"], "repetition index", 0, 3)
        prefill_stop, output_stop = case["stop_after_prefill"], case["stop_after_output"]
        if prefill_stop is not None and output_stop is not None:
            raise ValueError("A case cannot specify two stop boundaries.")
        if output == 0 and (prefill_stop is not None or output_stop is not None):
            raise ValueError("A zero-output case cannot stop generation.")
        if prefill_stop is not None:
            integer(prefill_stop, "prefill stop", 0, length)
        if output_stop is not None:
            integer(output_stop, "output stop", 1, output)
    return copy.deepcopy(plan)


def validate_seed_messages(messages):
    sys.path.insert(0, str(WORKER_DIRECTORY))
    try:
        import contracts
        frame = {"protocol": contracts.PROTOCOL, "kind": "prepare_input",
                 "worker_epoch": "00000000-0000-4000-8000-000000000001",
                 "command_id": "00000000-0000-4000-8000-000000000002", "model_revision": "0" * 64,
                 "messages": messages, "tools": [], "template_options": TEMPLATE_OPTIONS}
        contracts.encode_frame(frame)
    finally:
        sys.path.pop(0)
    return copy.deepcopy(messages)


def input_source_record(path, value, digest=None):
    return {"kind": "file" if path is not None else "default", "path": str(path) if path is not None else None,
            "sha256": digest if digest is not None else hashlib.sha256(json_bytes(value)).hexdigest()}


def frozen_metadata(document, digest):
    return {"file": FROZEN_INPUT_FILE, "sha256": digest,
            "plan_source": copy.deepcopy(document["plan_source"]),
            "seed_source": copy.deepcopy(document["seed_source"]),
            "seed_messages": copy.deepcopy(document["seed_messages"])}


def freeze_inputs(args, plan):
    document = {"format": "apxinf-calibration-inputs-v1", "profile": profile(args), "plan": plan,
                "seed_messages": args.prepared_seed_messages,
                "plan_source": args.plan_source, "seed_source": args.seed_source}
    payload = json_bytes(document)
    if len(payload) > FROZEN_INPUT_LIMIT:
        raise ValueError("The frozen inputs exceed their byte limit.")
    with (args.output_dir / FROZEN_INPUT_FILE).open("xb") as destination:
        destination.write(payload)
    return frozen_metadata(document, hashlib.sha256(payload).hexdigest())


def read_frozen_inputs(args):
    expected = args.frozen_inputs_sha256
    if type(expected) is not str or SHA256.fullmatch(expected) is None:
        raise ValueError("The probe requires a valid frozen-input digest.")
    document, actual = read_json_file(args.output_dir / FROZEN_INPUT_FILE, FROZEN_INPUT_LIMIT)
    if actual != expected:
        raise ValueError("The frozen-input digest differs from the parent's snapshot.")
    fields = {"format", "profile", "plan", "seed_messages", "plan_source", "seed_source"}
    if type(document) is not dict or set(document) != fields or document["format"] != "apxinf-calibration-inputs-v1":
        raise ValueError("The frozen inputs have an unsupported format.")
    if json_bytes(document["profile"]) != json_bytes(profile(args)):
        raise ValueError("The frozen profile differs from the probe's profile.")
    for field in ("plan_source", "seed_source"):
        source = document[field]
        if (type(source) is not dict or set(source) != {"kind", "path", "sha256"}
                or source["kind"] not in ("file", "default") or type(source["sha256"]) is not str
                or SHA256.fullmatch(source["sha256"]) is None):
            raise ValueError("Frozen source provenance is invalid.")
        if source["kind"] == "default":
            content = document["plan"] if field == "plan_source" else document["seed_messages"]
            if source["path"] is not None or source["sha256"] != hashlib.sha256(json_bytes(content)).hexdigest():
                raise ValueError("Default source provenance differs from its content.")
        elif type(source["path"]) is not str or not Path(source["path"]).is_absolute():
            raise ValueError("A frozen source path must be absolute.")
    plan = validate_plan(document["plan"], args.max_context, args.max_output_tokens)
    args.prepared_seed_messages = validate_seed_messages(document["seed_messages"])
    args.measurement_inputs = frozen_metadata(document, actual)
    return plan


class Journal:
    """Retain JSON scalars on disk without keeping model or cache references."""
    def __init__(self, path):
        self.path = Path(path)
        self.file = self.path.open("xb")
        self.count = 0
        self.digest = hashlib.sha256()

    def append(self, **sample):
        record = {"record_index": self.count, **sample}
        payload = (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        self.file.write(payload)
        self.file.flush()
        self.digest.update(payload)
        self.count += 1
        return record["record_index"]

    def description(self):
        return {"file": self.path.name, "record_count": self.count, "sha256": self.digest.hexdigest()}

    def close(self):
        self.file.close()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def token_digest(tokens):
    # Match the existing first-party wire digest rather than inventing another token encoding.
    sys.path.insert(0, str(WORKER_DIRECTORY))
    try:
        import contracts
        return contracts.token_prefix_digest(tokens)
    finally:
        sys.path.pop(0)


def runtime_identity(runtime):
    sys.path.insert(0, str(WORKER_DIRECTORY))
    try:
        import contracts
        revision = contracts.canonical_identity_digest("apxinf-model-identity-v1", runtime.manifest)
    finally:
        sys.path.pop(0)
    return {"model_revision": revision, "manifest": runtime.manifest, "runtime": runtime.runtime,
            "effective_context": runtime.max_context, "vocab_size": runtime.vocab_size,
            "eos_token_ids": list(runtime.eos_token_ids)}


def empty_case(case, index):
    return {"case_id": case["case_id"], "sequence_index": index,
            "first_request_in_process": index == 0, "status": "skipped", "reason": "not_run",
            "error": None, "error_phase": None, "input_digest": None,
            "output_digest": None, "actual_output_tokens": 0, "emitted_eos": False,
            "stop_condition_reached": None, "samples": {"first_record": None, "record_count": 0},
            "cleanup": {"generator_closed": None, "settled": None, "error": None}}


def observation_time_ns():
    return integer(time.monotonic_ns(), "observation timestamp", upper=MAX_TIMESTAMP_NS)


def observe(journal, runtime, host_probe, phase, *, case_id=None, output_tokens=0, prompt_position=None):
    memory = runtime.inspect_memory() if runtime is not None else None
    host = host_probe()
    index = journal.append(case_id=case_id, phase=phase, monotonic_ns=observation_time_ns(),
                           output_tokens=output_tokens, reported_prompt_position=prompt_position,
                           memory=memory, memory_error=None, host=host)
    return index, host


def require_normal(host, phase):
    if host.get("pressure", {}).get("state") != "normal":
        state = host.get("pressure", {}).get("state", "unknown")
        raise PressureBlocked(f"Host pressure blocks {phase}: {state}.")


def measure_case(runtime, case, result, seed_tokens, journal, host_probe):
    first_record = journal.count
    phase = "before_request"
    generator = None
    mutated = False
    emitted = []
    closed = True
    input_tokens = [seed_tokens[index % len(seed_tokens)] for index in range(case["input_tokens"])]
    result["input_digest"] = token_digest(input_tokens)
    result["status"] = "failed"
    result["reason"] = None

    def sample(name, prompt_position=None, check_pressure=True):
        nonlocal phase
        phase = name
        if journal.count - first_record >= SAMPLE_LIMIT:
            raise RuntimeError("The per-case sample limit was reached.")
        _, host = observe(journal, runtime, host_probe, name, case_id=case["case_id"],
                          output_tokens=len(emitted), prompt_position=prompt_position)
        if check_pressure:
            require_normal(host, name)

    def progress(consumed):
        sample("prefill_progress", consumed)
        threshold = case["stop_after_prefill"]
        if threshold is not None and consumed >= threshold:
            result["stop_condition_reached"] = True
            raise PlannedStop("prefill_stop")

    def cleanup_error(message):
        previous = result["cleanup"]["error"]
        result["cleanup"]["error"] = f"{previous}; {message}" if previous else message

    def cleanup_sample(name):
        try:
            sample(name)
        except PressureBlocked as error:
            if result["status"] == "completed":
                result.update(status="blocked", reason="host_pressure", error_phase=name,
                              error=f"{type(error).__name__}: {error}")
        except BaseException as error:
            cleanup_error(f"{name} observation failed: {type(error).__name__}: {error}")

    try:
        sample("before_request")
        if case["max_tokens"]:
            generator = runtime.generate({"token_ids": input_tokens, "max_tokens": case["max_tokens"]}, progress)
            closed = False
            mutated = True
            for token in generator:
                integer(token, "output token", 0, runtime.vocab_size - 1)
                emitted.append(token)
                eos = token in runtime.eos_token_ids
                result["emitted_eos"] = eos
                runtime.decode(token, eos=eos)
                if len(emitted) == 1 or len(emitted) % 16 == 0:
                    sample("first_output" if len(emitted) == 1 else "output_checkpoint")
                else:
                    phase = "output_checkpoint"
                    host = host_probe()
                    if host.get("pressure", {}).get("state") != "normal":
                        # Preserve the exact failing read before any subsequent recovery sample.
                        if journal.count - first_record < SAMPLE_LIMIT:
                            journal.append(case_id=case["case_id"], phase=phase, monotonic_ns=observation_time_ns(),
                                           output_tokens=len(emitted), reported_prompt_position=None,
                                           memory=None, memory_error="Pressure blocked an unsampled output yield.", host=host)
                            require_normal(host, phase)
                        require_normal(host, f"{phase} (sample_limit)")
                if eos:
                    result["reason"] = "eos"
                    break
                if case["stop_after_output"] is not None and len(emitted) >= case["stop_after_output"]:
                    result["stop_condition_reached"] = True
                    result["reason"] = "output_stop"
                    break
            if result["reason"] is None:
                if len(emitted) != case["max_tokens"]:
                    raise RuntimeError("Generation ended before its output allowance without EOS.")
                result["reason"] = "length"
        else:
            result["reason"] = "zero_output"
        if case["stop_after_prefill"] is not None or case["stop_after_output"] is not None:
            result["stop_condition_reached"] = result["stop_condition_reached"] is True
        result["status"] = "completed"
    except PlannedStop as stopped:
        result["status"], result["reason"] = "completed", str(stopped)
    except BaseException as error:
        result["status"] = "blocked" if isinstance(error, PressureBlocked) else "failed"
        result["reason"] = "host_pressure" if isinstance(error, PressureBlocked) else "execution_error"
        result["error_phase"] = phase
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        result["actual_output_tokens"] = len(emitted)
        result["output_digest"] = token_digest(emitted)
        if generator is not None:
            try:
                generator.close()
                closed = True
            except BaseException as error:
                cleanup_error(f"Generator closure failed: {type(error).__name__}: {error}")
        result["cleanup"]["generator_closed"] = closed
        if closed:
            try:
                # Finalize text as the worker does before terminal memory and settlement.
                if mutated:
                    runtime.finish_text()
            except BaseException as error:
                cleanup_error(f"Text finalization failed: {type(error).__name__}: {error}")
            cleanup_sample("terminal")
            try:
                if mutated:
                    runtime.settle()
                result["cleanup"]["settled"] = True
            except BaseException as error:
                result["cleanup"]["settled"] = False
                cleanup_error(f"Settlement failed: {type(error).__name__}: {error}")
            if result["cleanup"]["settled"]:
                cleanup_sample("settled")
        if result["cleanup"]["error"] is not None:
            result["status"] = "failed"
            result["reason"] = "cleanup_error"
        result["samples"] = {"first_record": first_record, "record_count": journal.count - first_record}


def initial_report(model_path, requested_profile, plan, hardware=None, measurement_inputs=None):
    return {"format": "apxinf-memory-calibration-v1", "artifact_kind": "raw_observations",
              "run_id": str(uuid.uuid4()), "created_at": datetime.now(timezone.utc).isoformat(),
              "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "tool_files": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                             for name in ("memory_calibration.py", "host_memory.py", "metal_matrix.py")},
              "model_path": str(model_path), "requested_profile": requested_profile,
              "hardware": hardware, "identity": None, "input_source": None, "plan": copy.deepcopy(plan),
              "measurement_inputs": copy.deepcopy(measurement_inputs),
              "cases": [empty_case(case, index) for index, case in enumerate(plan)],
              "run_samples": [], "status": "running", "error": None, "error_phase": None,
              "load_started": False, "run_cleanup": {"settled": None, "error": None},
              "process": {"pid": os.getpid(), "returncode": None, "reaped": False},
              "recovery": None,
              "journal": {"file": "samples.jsonl", "record_count": 0, "sha256": hashlib.sha256().hexdigest()}}


def calibrate(*, model_path, requested_profile, plan, runtime_factory, journal,
              persist, host_probe=host_memory.snapshot, hardware=None, seed_messages=None, measurement_inputs=None):
    messages = validate_seed_messages(SEED_MESSAGES if seed_messages is None else seed_messages)
    report = initial_report(model_path, requested_profile, plan, hardware, measurement_inputs)
    runtime = None
    phase = "before_load"

    def checkpoint():
        report["journal"] = journal.description()
        persist(report)

    checkpoint()
    try:
        index, host = observe(journal, None, host_probe, phase)
        report["run_samples"].append(index)
        checkpoint()
        require_normal(host, phase)
        phase = "load"
        report["load_started"] = True
        checkpoint()
        runtime = runtime_factory()
        report["identity"] = runtime_identity(runtime)
        phase = "loaded"
        index, host = observe(journal, runtime, host_probe, phase)
        report["run_samples"].append(index)
        checkpoint()
        require_normal(host, phase)
        if runtime.max_context != requested_profile["max_context"]:
            raise RuntimeError("The loaded model context differs from the planned profile.")
        phase = "prepare_seed"
        seed = runtime.prepare({"messages": copy.deepcopy(messages), "tools": [], "template_options": copy.deepcopy(TEMPLATE_OPTIONS)})
        if not seed or len(seed) > 131072:
            raise ValueError("The prepared seed has an invalid token count.")
        for token in seed:
            integer(token, "seed token", 0, runtime.vocab_size - 1)
        report["input_source"] = {"kind": "synthetic_tiled_prepared_tokens", "messages": messages,
                                  "template_options": copy.deepcopy(TEMPLATE_OPTIONS), "seed_tokens": seed,
                                  "seed_digest": token_digest(seed), "quality_evidence": False}
        checkpoint()
        for case, result in zip(plan, report["cases"]):
            phase = case["case_id"]
            result.update(status="running", reason=None)
            checkpoint()
            measure_case(runtime, case, result, seed, journal, host_probe)
            checkpoint()
            if result["status"] in ("failed", "blocked"):
                report["status"] = result["status"]
                report["error"] = result["error"] or result["cleanup"]["error"]
                report["error_phase"] = f"{case['case_id']}:{result['error_phase'] or result['reason']}"
                break
        else:
            report["status"] = "completed"
    except BaseException as error:
        report["status"] = "blocked" if isinstance(error, PressureBlocked) else "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        report["error_phase"] = phase
    finally:
        if runtime is not None:
            # A case with uncertain cleanup retains its state until this isolated process exits.
            uncertain = any(case["status"] == "failed" and case["cleanup"]["settled"] is not True
                            for case in report["cases"])
            settled_case = any(case["cleanup"]["settled"] is True for case in report["cases"])
            if settled_case and not uncertain:
                report["run_cleanup"]["settled"] = True
            elif not uncertain:
                try:
                    runtime.settle()
                    report["run_cleanup"]["settled"] = True
                except BaseException as error:
                    report["status"] = "failed"
                    report["run_cleanup"]["settled"] = False
                    report["run_cleanup"]["error"] = f"{type(error).__name__}: {error}"
            else:
                report["run_cleanup"]["settled"] = False
        for result in report["cases"]:
            if result["status"] == "skipped":
                result["reason"] = "earlier_case_or_startup_stopped"
        checkpoint()
    return report


def profile(args):
    return {"provider": "mlx-lm", "precision": "bundle", "max_context": args.max_context,
            "max_output_tokens": args.max_output_tokens, "prefill_step_size": args.prefill_step_size,
            "output_batch_tokens": args.output_batch_tokens, "memory_limit_bytes": args.memory_budget_bytes,
            "synchronization": "runtime_owned_streams", "host_pressure_policy": "macos"}


def create_runtime(args):
    sys.path.insert(0, str(WORKER_DIRECTORY))
    try:
        from text_worker import MLXRuntime
        return MLXRuntime(str(args.model), args.max_context, args.prefill_step_size,
                          args.output_batch_tokens, args.memory_budget_bytes)
    finally:
        sys.path.pop(0)


def validate_arguments(args):
    args.model = args.model.resolve(strict=True)
    if not args.model.is_dir():
        raise ValueError("The model must be a directory.")
    args.output_dir = args.output_dir.resolve()
    if args.output_dir == ROOT or ROOT in args.output_dir.parents:
        raise ValueError("Store raw calibration artifacts outside the checkout.")
    integer(args.max_output_tokens, "output limit", 1, 65536)
    integer(args.max_context, "context limit", 2, 131072)
    integer(args.prefill_step_size, "prefill chunk", 1, 8192)
    integer(args.output_batch_tokens, "output batch", 1, 256)
    integer(args.memory_budget_bytes, "memory guideline", 1)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("The process timeout must be finite and positive.")
    planner_options = (args.max_tokens, args.repetitions, args.input_lengths)
    if args.probe:
        if args.plan_file is not None or args.seed_messages is not None or any(value is not None for value in planner_options):
            raise ValueError("The probe accepts only frozen plans and seed messages.")
        return read_frozen_inputs(args)
    if args.frozen_inputs_sha256 is not None:
        raise ValueError("A frozen-input digest belongs to an isolated probe only.")
    if args.plan_file is not None:
        if any(value is not None for value in planner_options):
            raise ValueError("An explicit plan conflicts with generated-plan options.")
        path = args.plan_file.resolve(strict=True)
        supplied, digest = read_json_file(path, INPUT_FILE_LIMIT)
        plan = validate_plan(supplied, args.max_context, args.max_output_tokens)
        args.plan_source = input_source_record(path, supplied, digest)
    else:
        max_tokens = 32 if args.max_tokens is None else args.max_tokens
        repetitions = 2 if args.repetitions is None else args.repetitions
        integer(max_tokens, "case output allowance", 1, args.max_output_tokens)
        plan = make_plan(args.max_context, args.prefill_step_size, max_tokens, repetitions, args.input_lengths)
        args.plan_source = input_source_record(None, plan)
    if args.seed_messages is not None:
        path = args.seed_messages.resolve(strict=True)
        messages, digest = read_json_file(path, INPUT_FILE_LIMIT)
        args.seed_source = input_source_record(path, messages, digest)
    else:
        messages = SEED_MESSAGES
        args.seed_source = input_source_record(None, messages)
    args.prepared_seed_messages = validate_seed_messages(messages)
    return plan


def probe(args, plan):
    descriptor = args.lock_fd
    expected = args.metal_lock.stat()
    actual = os.fstat(descriptor)
    if (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino):
        raise RuntimeError("The inherited descriptor does not identify the Metal lock.")
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    journal = Journal(args.output_dir / "samples.jsonl")
    try:
        report = calibrate(model_path=args.model, requested_profile=profile(args), plan=plan,
                           runtime_factory=lambda: create_runtime(args), journal=journal,
                           persist=lambda value: save(args.output_dir / "calibration.json", value),
                           hardware=host_memory.hardware_snapshot(), seed_messages=args.prepared_seed_messages,
                           measurement_inputs=args.measurement_inputs)
        return 0 if report["status"] == "completed" else 1
    finally:
        journal.close()
        # The inherited lock stays open until this process exits and reclaims its model.


def recover_journal(directory, report):
    """Recover flushed records after process exit without fabricating completed work."""
    path = directory / "samples.jsonl"
    if not path.exists():
        path.touch(exist_ok=False)
    previous = report["journal"]
    results = {case["case_id"]: case for case in report["cases"]}
    ranges = {case_id: [] for case_id in results}
    run_samples = []
    count = 0
    errors = []
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
        source.seek(0)
        while payload := source.readline(4 * 1024**2 + 1):
            try:
                if len(payload) > 4 * 1024**2 or not payload.endswith(b"\n"):
                    raise ValueError("The journal has an oversized or incomplete record.")
                sample = json.loads(payload)
                if type(sample.get("record_index")) is not int or sample["record_index"] != count:
                    raise ValueError("Journal indices do not form a complete sequence.")
                case_id = sample.get("case_id")
                if case_id is None:
                    run_samples.append(count)
                elif case_id in ranges:
                    if len(ranges[case_id]) >= SAMPLE_LIMIT:
                        raise ValueError("A case exceeds its journal sample limit.")
                    ranges[case_id].append(count)
                else:
                    raise ValueError("A journal record has an unknown case ID.")
                count += 1
            except (ValueError, TypeError, AttributeError) as error:
                errors.append(f"Record {count}: {type(error).__name__}: {error}")
                break
    description = {"file": path.name, "record_count": count, "sha256": digest}
    if report["status"] == "completed" and previous != description:
        errors.append("The completed report does not match its journal.")
    for case_id, result in results.items():
        indices = ranges[case_id]
        if indices and indices[-1] - indices[0] + 1 != len(indices):
            errors.append(f"The records for {case_id} are not contiguous.")
        old_count = result["samples"]["record_count"]
        result["samples"] = {"first_record": indices[0] if indices else None, "record_count": len(indices)}
        interrupted = result["status"] == "running" or (indices and result["status"] == "skipped")
        if old_count != len(indices) and result["status"] not in ("running", "skipped"):
            errors.append(f"The final sample count differs for {case_id}.")
            interrupted = True
        if interrupted:
            if report["status"] == "completed":
                errors.append(f"The completed report contains an interrupted case: {case_id}.")
            result.update(status="failed", reason="probe_interrupted", error_phase="process_exit",
                          error="The probe exited without a final case result.",
                          actual_output_tokens=None, output_digest=None)
            result["cleanup"] = {"generator_closed": None, "settled": None,
                                 "error": "Request settlement was not confirmed before process exit."}
        elif result["status"] == "skipped":
            result["reason"] = "earlier_case_or_startup_stopped"
    report["run_samples"] = run_samples
    report["journal"] = description
    report["recovery"] = {"errors": errors, "snapshot_changed": previous != description}
    return errors


def completion_errors(report):
    """Reject a successful outcome when its required evidence is incomplete."""
    errors = []
    if report["identity"] is None or report["input_source"] is None or not report["load_started"]:
        errors.append("A completed run requires loaded identity and input evidence.")
    if report["run_cleanup"]["settled"] is not True or report["run_cleanup"]["error"] is not None:
        errors.append("A completed run requires successful final settlement.")
    plan, cases = report["plan"], report["cases"]
    if not plan or [item["case_id"] for item in plan] != [item["case_id"] for item in cases]:
        errors.append("The planned and completed case sequences differ.")
    for index, (planned, case) in enumerate(zip(plan, cases)):
        cleanup = case["cleanup"]
        count = case["actual_output_tokens"]
        if (case["status"] != "completed" or case["sequence_index"] != index
                or cleanup["generator_closed"] is not True or cleanup["settled"] is not True
                or cleanup["error"] is not None or case["error"] is not None
                or type(count) is not int or not 0 <= count <= planned["max_tokens"]
                or case["input_digest"] is None or case["output_digest"] is None):
            errors.append(f"Case {planned['case_id']} lacks complete execution and settlement evidence.")
    return errors


def validate_probe_report_structure(report, fallback):
    """Check recovery containers before they can replace the parent's complete plan."""
    def fields(value, required, label):
        if type(value) is not dict or not set(required).issubset(value):
            raise ValueError(f"The probe report has invalid {label} fields.")

    fields(report, fallback, "top-level")
    if report["status"] not in ("running", "completed", "failed", "blocked"):
        raise ValueError("The probe report has an invalid run status.")
    for name in ("error", "error_phase"):
        if report[name] is not None and type(report[name]) is not str:
            raise ValueError(f"The probe report has an invalid {name}.")
    fields(report["journal"], ("file", "record_count", "sha256"), "journal")
    fields(report["run_cleanup"], ("settled", "error"), "run cleanup")
    if type(report["plan"]) is not list or type(report["cases"]) is not list:
        raise ValueError("The probe report requires plan and case arrays.")
    expected_ids = [case["case_id"] for case in fallback["plan"]]
    for planned in report["plan"]:
        fields(planned, ("case_id", "max_tokens"), "plan")
    if [case["case_id"] for case in report["plan"]] != expected_ids:
        raise ValueError("The probe plan differs from the complete parent case sequence.")
    if len(report["cases"]) != len(expected_ids):
        raise ValueError("The probe report lacks the complete parent case sequence.")
    for index, case in enumerate(report["cases"]):
        fields(case, fallback["cases"][index], "case")
        fields(case["samples"], ("first_record", "record_count"), "case samples")
        fields(case["cleanup"], ("generator_closed", "settled", "error"), "case cleanup")
        if case["case_id"] != expected_ids[index] or case["status"] not in ("running", "completed", "failed", "blocked", "skipped"):
            raise ValueError("The probe report has an invalid case identity or status.")


def preserve_invalid_report(path):
    destination = path.with_name(f"calibration.invalid-{uuid.uuid4()}.json")
    try:
        path.rename(destination)
        return {"file": destination.name, "error": None}
    except OSError as error:
        return {"file": None, "error": f"Cannot preserve the invalid probe report: {error}"}


def finalize_process(directory, process, fallback):
    """Write the parent outcome and merge probe observations only after reaping."""
    path = directory / "calibration.json"
    process["status"] = "failed"
    if not process["reaped"]:
        process["error"] = process.get("error") or "Probe reaping was not confirmed."
        # A living probe can still write its report. Only the parent writes process.json.
        save(directory / "process.json", process)
        return 1
    try:
        report = json.loads(path.read_text()) if path.exists() else copy.deepcopy(fallback)
        validate_probe_report_structure(report, fallback)
    except (ValueError, OSError, TypeError, KeyError, RecursionError) as error:
        report = copy.deepcopy(fallback)
        process["error"] = process.get("error") or f"The probe report is invalid: {type(error).__name__}: {error}"
        process["invalid_report"] = preserve_invalid_report(path)
    try:
        errors = recover_journal(directory, report)
        if fallback.get("measurement_inputs") is not None:
            for field in ("plan", "requested_profile", "measurement_inputs"):
                if json_bytes(report.get(field)) != json_bytes(fallback[field]):
                    errors.append(f"The probe report differs from frozen parent input: {field}.")
            source = report.get("input_source")
            if source is not None:
                if type(source) is not dict:
                    raise ValueError("The prepared input evidence must be an object.")
                expected_seed = fallback["measurement_inputs"]["seed_messages"]
                if json_bytes(source.get("messages")) != json_bytes(expected_seed):
                    errors.append("The prepared messages differ from the frozen seed.")
                if json_bytes(source.get("template_options")) != json_bytes(TEMPLATE_OPTIONS):
                    errors.append("The prepared template options differ from the fixed options.")
        if report["status"] == "completed":
            errors.extend(completion_errors(report))
    except (OSError, ValueError, TypeError, KeyError) as error:
        errors = [f"Journal recovery failed: {type(error).__name__}: {error}"]
        report["recovery"] = {"errors": errors, "snapshot_changed": None}
    if errors:
        process["error"] = process.get("error") or "; ".join(errors)
    parent_failed = bool(process.get("error")) or process["pid"] is None
    if parent_failed or report["status"] not in ("completed", "blocked"):
        report["status"] = "failed"
    elif process["returncode"] != 0 and report["status"] != "blocked":
        report["status"] = "failed"
    if report["status"] == "failed":
        report["error"] = report["error"] or process.get("error") or "The probe did not complete successfully."
        report["error_phase"] = report["error_phase"] or "process_exit"
    process["status"] = report["status"]
    report["process"] = process
    try:
        save(path, report)
    except (OSError, ValueError, TypeError, RecursionError) as error:
        process["status"] = "failed"
        process["report_write_error"] = f"Cannot save the final calibration report: {error}"
        process["error"] = process.get("error") or process["report_write_error"]
    save(directory / "process.json", process)
    return 0 if process["status"] == "completed" else 1


def run(args):
    plan = validate_arguments(args)
    if args.probe:
        return probe(args, plan)
    if args.lock_fd is not None:
        raise ValueError("A lock descriptor belongs to an isolated probe only.")
    args.python = args.python.resolve(strict=True)
    descriptor = os.open(args.metal_lock, os.O_RDONLY)
    child = None
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        save(args.output_dir / "plan.json", {"profile": profile(args), "cases": plan})
        measurement_inputs = freeze_inputs(args, plan)
        fallback = initial_report(args.model, profile(args), plan, measurement_inputs=measurement_inputs)
        fallback["process"]["pid"] = None
        save(args.output_dir / "calibration.json", fallback)
        command = [str(args.python), "-u", str(Path(__file__).resolve()), "--probe", "--lock-fd", str(descriptor),
                   "--model", str(args.model), "--output-dir", str(args.output_dir),
                   "--metal-lock", str(args.metal_lock.resolve()), "--max-context", str(args.max_context),
                   "--max-output-tokens", str(args.max_output_tokens),
                   "--prefill-step-size", str(args.prefill_step_size), "--output-batch-tokens", str(args.output_batch_tokens),
                   "--memory-budget-bytes", str(args.memory_budget_bytes),
                   "--frozen-inputs-sha256", measurement_inputs["sha256"]]
        process = {"command": command, "pid": None, "returncode": None, "reaped": False,
                   "status": "running", "error": None}
        save(args.output_dir / "process.json", process)
        try:
            with (args.output_dir / "probe.log").open("xb") as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                         pass_fds=(descriptor,), start_new_session=True)
                process["pid"] = child.pid
                save(args.output_dir / "process.json", process)
                try:
                    child.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    process["error"] = "The isolated probe exceeded its process timeout."
        except BaseException as error:
            process["error"] = f"{type(error).__name__}: {error}"
        finally:
            if child is not None:
                try:
                    from metal_matrix import stop_owned_group
                    process["cleanup"] = stop_owned_group(child)
                    process["returncode"] = child.returncode
                    process["reaped"] = child.returncode is not None and not process["cleanup"]["remaining_members"]
                except BaseException as error:
                    process["cleanup_error"] = f"{type(error).__name__}: {error}"
                    process["error"] = process.get("error") or process["cleanup_error"]
            else:
                process["reaped"] = True  # No child exists to reclaim.
            status = finalize_process(args.output_dir, process, fallback)
        print(json.dumps({"output_dir": str(args.output_dir), "status": process["status"],
                          "returncode": process["returncode"], "reaped": process["reaped"]}))
        return status
    finally:
        os.close(descriptor)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--python", type=Path, default=ROOT / ".apxinf/toolchains/mlx-lm-0.31.3-copies/bin/python")
    result.add_argument("--metal-lock", type=Path, default=Path.home() / ".cache/enginetailor/metal-measure.lock")
    result.add_argument("--max-context", type=int, default=16384)
    result.add_argument("--max-output-tokens", type=int, default=2048)
    result.add_argument("--max-tokens", type=int, help="Generated-plan output allowance. Default: 32.")
    result.add_argument("--prefill-step-size", type=int, default=256)
    result.add_argument("--output-batch-tokens", type=int, default=1)
    result.add_argument("--memory-budget-bytes", type=int, default=10 * 1024**3)
    result.add_argument("--repetitions", type=int, help="Generated-plan repetitions. Default: 2.")
    result.add_argument("--input-lengths", type=lambda text: [int(value) for value in text.split(",")])
    result.add_argument("--plan-file", type=Path)
    result.add_argument("--seed-messages", type=Path)
    result.add_argument("--timeout", type=float, default=1800)
    result.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    result.add_argument("--frozen-inputs-sha256", help=argparse.SUPPRESS)
    return result


if __name__ == "__main__":
    def interrupt(_signal, _frame):
        raise KeyboardInterrupt("The calibration process received a termination signal.")
    signal.signal(signal.SIGTERM, interrupt)
    raise SystemExit(run(parser().parse_args()))
