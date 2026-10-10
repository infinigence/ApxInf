#!/usr/bin/env python3
"""Check offline calibration evidence without importing a model runtime."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import uuid


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "apxinf_coverage_contracts", ROOT / "python/apxinf/apxinf/serving/contracts.py")
contracts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contracts)
DIGEST = re.compile(r"[0-9a-f]{64}\Z")
CASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
MAX_DOCUMENT = 16 * 1024**2
FROZEN_INPUT_FILE = "measurement-inputs.json"
MAX_FROZEN_INPUT = 4 * 1024**2
MAX_JOURNAL = 64 * 1024**2
MAX_LINE = 4 * 1024**2
MAX_RECORDS = 65536
MAX_TIMESTAMP_NS = 2**64 - 1
PHASES = {"before_load", "loaded", "before_request", "prefill_progress",
          "first_output", "output_checkpoint", "terminal", "settled"}
LIMITATIONS = [
    "Coverage does not approve an admission budget or establish a memory upper bound.",
    "Validation checks internal consistency, not source authenticity or local model files.",
    "Synchronized observations can change allocation overlap and execution timing.",
    "Output digests cannot be reconstructed without all output token IDs.",
    "Source identity lacks an explicit OS build, device identity, environment policy, and sampling-policy identity.",
    "Exact shapes do not cover all inputs, schedules, or host conditions.",
    "Logical cache payload, allocator counters, RSS, and system swap remain separate measurements.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, label, lower=0, upper=contracts.MAX_SAFE_INTEGER):
    require(type(value) is int and lower <= value <= upper, f"Invalid integer: {label}.")
    return value


def digest(value, label):
    require(type(value) is str and DIGEST.fullmatch(value) is not None, f"Invalid digest: {label}.")


def object_value(value, label):
    require(type(value) is dict, f"Expected an object: {label}.")
    return value


def boolean(value, label, nullable=False):
    require(type(value) is bool or (nullable and value is None), f"Invalid Boolean: {label}.")


def pairs_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}.")
        result[key] = value
    return result


def parse_json(payload):
    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON number: {value}.")
    def finite_float(value):
        parsed = float(value)
        require(math.isfinite(parsed), "A JSON number exceeds the finite range.")
        return parsed
    return json.loads(payload, object_pairs_hook=pairs_object, parse_constant=invalid_constant,
                      parse_float=finite_float)


def read_source(directory, name, limit, sources):
    with (directory / name).open("rb") as source:
        payload = source.read(limit + 1)
    require(len(payload) <= limit, f"Source exceeds the byte limit: {name}.")
    sources[name] = {"file": name, "size_bytes": len(payload),
                     "sha256": hashlib.sha256(payload).hexdigest()}
    return payload


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def validate_frozen_inputs(directory, report, sources):
    metadata = report.get("measurement_inputs")
    if metadata is None:
        return "unavailable"
    object_value(metadata, "measurement inputs")
    require(set(metadata) == {"file", "sha256", "plan_source", "seed_source", "seed_messages"},
            "Frozen input metadata fields differ.")
    require(metadata["file"] == FROZEN_INPUT_FILE, "Invalid frozen input filename.")
    digest(metadata["sha256"], "frozen inputs")
    payload = read_source(directory, FROZEN_INPUT_FILE, MAX_FROZEN_INPUT, sources)
    require(hashlib.sha256(payload).hexdigest() == metadata["sha256"], "Frozen input digest differs.")
    frozen = object_value(parse_json(payload.decode("utf-8")), "frozen inputs")
    require(set(frozen) == {"format", "profile", "plan", "seed_messages", "plan_source", "seed_source"}
            and frozen["format"] == "apxinf-calibration-inputs-v1", "Unsupported frozen input format.")
    expected_fields = {"profile": report["requested_profile"], "plan": report["plan"],
                       "seed_messages": metadata["seed_messages"], "plan_source": metadata["plan_source"],
                       "seed_source": metadata["seed_source"]}
    for field, expected in expected_fields.items():
        require(canonical_json(frozen[field]) == canonical_json(expected), f"Frozen {field} differs from calibration metadata.")
    profile = object_value(frozen["profile"], "frozen profile")
    require(set(profile) == {"provider", "precision", "max_context", "max_output_tokens", "prefill_step_size",
                            "output_batch_tokens", "memory_limit_bytes", "synchronization", "host_pressure_policy"}
            and profile["host_pressure_policy"] == "macos", "Unsupported frozen profile fields or policy.")
    for field, content in (("plan_source", frozen["plan"]), ("seed_source", frozen["seed_messages"])):
        source = object_value(frozen[field], field)
        require(set(source) == {"kind", "path", "sha256"} and source["kind"] in ("default", "file"),
                "Invalid frozen source provenance.")
        digest(source["sha256"], field)
        if source["kind"] == "default":
            require(source["path"] is None and source["sha256"] == hashlib.sha256(canonical_json(content)).hexdigest(),
                    "Default source provenance differs from its content.")
        else:
            require(type(source["path"]) is str and Path(source["path"]).is_absolute(),
                    "A frozen source path must be absolute.")
    template = {"enable_thinking": False}
    contracts.encode_frame({"protocol": contracts.PROTOCOL, "kind": "prepare_input",
                            "worker_epoch": "00000000-0000-4000-8000-000000000001",
                            "command_id": "00000000-0000-4000-8000-000000000002", "model_revision": "0" * 64,
                            "messages": frozen["seed_messages"], "tools": [], "template_options": template})
    if report.get("input_source") is not None:
        source = object_value(report["input_source"], "input source")
        require(canonical_json(source.get("messages")) == canonical_json(frozen["seed_messages"]),
                "Prepared messages differ from the frozen seed messages.")
        require(canonical_json(source.get("template_options")) == canonical_json(template),
                "Prepared template options differ from the frozen policy.")
    return "verified"


def validate_memory(memory):
    object_value(memory, "memory")
    require(all(field in memory for field in ("allocator", "peak_epoch", "runtime_consumed_position", "cache_payload_bytes", "layers")),
            "Memory observation fields are missing.")
    integer(memory.get("peak_epoch"), "peak epoch", upper=256)
    integer(memory.get("runtime_consumed_position"), "runtime position")
    allocator = object_value(memory.get("allocator"), "allocator")
    for field in ("active_bytes", "cache_bytes", "peak_bytes"):
        integer(allocator.get(field), field)
    layers = memory.get("layers")
    require(type(layers) is list and len(layers) <= 4096, "Invalid cache layers.")
    for layer in layers:
        object_value(layer, "cache layer")
        require(type(layer.get("type")) is str and bool(layer["type"]), "Invalid cache layer type.")
        for field in ("offset", "nbytes"):
            require(field in layer, f"Missing cache property: {field}.")
            if layer[field] is not None:
                integer(layer[field], field)
    sizes = [layer["nbytes"] for layer in layers]
    expected = None if any(value is None for value in sizes) else sum(sizes)
    require(memory.get("cache_payload_bytes") == expected, "The cache payload differs from its layer observations.")
    if memory.get("cache_payload_bytes") is not None:
        integer(memory["cache_payload_bytes"], "cache payload")


def validate_pressure(pressure):
    object_value(pressure, "host pressure")
    require(all(field in pressure for field in ("state", "dispatch_value", "error")),
            "Host pressure evidence fields are missing.")
    states = {"normal": 1, "warning": 2, "critical": 4}
    state, raw, error = pressure["state"], pressure["dispatch_value"], pressure["error"]
    require(type(state) is str and state in (*states, "unknown"), "Invalid pressure state.")
    if raw is not None:
        integer(raw, "pressure dispatch value", upper=2**32 - 1)
    if state == "unknown":
        require(type(error) is str and bool(error) and raw not in states.values(),
                "Unknown pressure contradicts its dispatch value or error.")
    else:
        require(raw == states[state] and error is None,
                "Known pressure contradicts its dispatch value or error.")


def validate_header(report, process):
    object_value(report, "calibration")
    object_value(process, "process")
    require(report.get("format") == "apxinf-memory-calibration-v1", "Unsupported calibration format.")
    require(report.get("artifact_kind") == "raw_observations", "Expected raw calibration observations.")
    run_id = report.get("run_id")
    require(type(run_id) is str and str(uuid.UUID(run_id)) == run_id, "Invalid run identity.")
    require(report.get("status") in ("completed", "failed", "blocked", "running"), "Invalid run status.")
    digest(report.get("tool_sha256"), "tool")
    files = object_value(report.get("tool_files"), "tool files")
    for name in ("memory_calibration.py", "host_memory.py", "metal_matrix.py"):
        digest(files.get(name), name)
    require(files["memory_calibration.py"] == report["tool_sha256"], "Tool digests differ.")
    profile = object_value(report.get("requested_profile"), "requested profile")
    integer(profile.get("max_context"), "context", 2, 131072)
    integer(profile.get("max_output_tokens"), "service output allowance", 1, 65536)
    integer(profile.get("prefill_step_size"), "prefill step", 1, 8192)
    integer(profile.get("output_batch_tokens"), "output batch", 1, 256)
    integer(profile.get("memory_limit_bytes"), "memory limit", 1)
    require(profile.get("provider") == "mlx-lm" and profile.get("precision") == "bundle",
            "Unsupported execution profile.")
    require(profile.get("synchronization") == "runtime_owned_streams", "Unsupported synchronization policy.")
    require(profile.get("host_pressure_policy") == "macos", "Unsupported host pressure policy.")
    boolean(process.get("reaped"), "process reaped")
    require("pid" in process and "returncode" in process, "Process identity fields are missing.")
    if process["pid"] is not None:
        integer(process["pid"], "process ID", 1, 2**31 - 1)
    if process["reaped"]:
        if process["pid"] is None:
            require(process["returncode"] is None, "An absent child has a process return code.")
        else:
            integer(process.get("returncode"), "process return code", -(2**31), 2**31 - 1)
        require(report.get("process") == process, "Parent and calibration process records differ.")
        require(process.get("status") == report["status"], "Parent and calibration statuses differ.")
    if process.get("cleanup") is not None:
        cleanup = object_value(process["cleanup"], "process cleanup")
        require(type(cleanup.get("remaining_members")) is list, "Process cleanup lacks remaining member evidence.")
        require(not process["reaped"] or not cleanup["remaining_members"], "Reaped process still has remaining members.")
        require(cleanup.get("returncode") == process["returncode"], "Process cleanup return codes differ.")
    recovery = report.get("recovery")
    if recovery is not None:
        object_value(recovery, "recovery")
        require(recovery.get("errors") == [], "Source journal recovery reports errors.")
    identity = report.get("identity")
    if identity is not None:
        object_value(identity, "identity")
        manifest = object_value(identity.get("manifest"), "manifest")
        revision = contracts.canonical_identity_digest("apxinf-model-identity-v1", manifest)
        require(identity.get("model_revision") == revision, "Model revision does not match its manifest.")
        require(identity.get("runtime") == manifest.get("runtime"), "Runtime identity fields differ.")
        execution = object_value(manifest.get("execution"), "manifest execution")
        for field in ("provider", "precision", "prefill_step_size", "output_batch_tokens", "memory_limit_bytes"):
            require(execution.get(field) == profile[field], f"Execution identity differs: {field}.")
        require(identity.get("effective_context") == profile["max_context"], "Context identity differs from the profile.")
        integer(identity.get("vocab_size"), "vocabulary size", 1, 2**32)
        require(type(identity.get("eos_token_ids")) is list, "Invalid EOS identity.")
        for token in identity["eos_token_ids"]:
            integer(token, "EOS token", upper=identity["vocab_size"] - 1)
    source = report.get("input_source")
    if source is not None:
        object_value(source, "input source")
        require(identity is not None, "Input evidence requires loaded identity.")
        require(source.get("kind") == "synthetic_tiled_prepared_tokens", "Unsupported input source.")
        seed = source.get("seed_tokens")
        require(type(seed) is list and 1 <= len(seed) <= 131072, "Invalid seed tokens.")
        for token in seed:
            integer(token, "seed token", upper=identity["vocab_size"] - 1)
        require(source.get("seed_digest") == contracts.token_prefix_digest(seed), "Seed digest differs.")
    return profile


def validate_cases(report, profile, frozen=False):
    plan, cases = report.get("plan"), report.get("cases")
    require(type(plan) is list and 1 <= len(plan) <= 256, "Invalid case plan.")
    require(type(cases) is list and len(cases) == len(plan), "Plan and case counts differ.")
    identifiers = set()
    stopped = False
    for index, (planned, case) in enumerate(zip(plan, cases)):
        object_value(planned, "planned case")
        object_value(case, "case result")
        name = planned.get("case_id")
        require(type(name) is str and 0 < len(name) <= 256 and name not in identifiers, "Invalid or duplicate case identity.")
        if frozen:
            require(set(planned) == {"case_id", "input_tokens", "max_tokens", "stop_after_prefill", "stop_after_output", "repetition"},
                    "A frozen case must contain exactly the six plan fields.")
            require(CASE_ID.fullmatch(name) is not None, "Invalid frozen case identity.")
        identifiers.add(name)
        require(case.get("case_id") == name, "Plan and case identities differ.")
        require(type(case.get("sequence_index")) is int and case["sequence_index"] == index, "Case sequence differs.")
        require(type(case.get("first_request_in_process")) is bool and case["first_request_in_process"] == (index == 0),
                "First-request labels differ from case order.")
        length = integer(planned.get("input_tokens"), "input length", 1, profile["max_context"])
        allowance = integer(planned.get("max_tokens"), "case output allowance", upper=profile["max_output_tokens"])
        require(length + allowance <= profile["max_context"], "A planned shape exceeds context.")
        integer(planned.get("repetition"), "repetition", upper=3)
        prefill, output = planned.get("stop_after_prefill"), planned.get("stop_after_output")
        require(prefill is None or output is None, "A case has two stop boundaries.")
        if prefill is not None:
            integer(prefill, "prefill stop", upper=length)
            require(allowance > 0, "A zero-output case cannot stop generation.")
        if output is not None:
            integer(output, "output stop", 1, allowance)
        require(case.get("status") in ("completed", "blocked", "failed", "skipped", "running"), "Invalid case status.")
        require(not stopped or case["status"] == "skipped", "A later case executes after the matrix stops.")
        stopped = case["status"] != "completed"
        count = case.get("actual_output_tokens")
        if count is not None:
            integer(count, "actual outputs", upper=allowance)
        boolean(case.get("emitted_eos"), "emitted EOS")
        require(not case["emitted_eos"] or count is None or count > 0, "EOS requires an output token.")
        boolean(case.get("stop_condition_reached"), "stop reached", nullable=True)
        cleanup = object_value(case.get("cleanup"), "case cleanup")
        boolean(cleanup.get("generator_closed"), "generator closed", nullable=True)
        boolean(cleanup.get("settled"), "case settled", nullable=True)
        ranges = object_value(case.get("samples"), "case sample range")
        integer(ranges.get("record_count"), "case sample count", upper=4096)
        if ranges.get("first_record") is not None:
            integer(ranges["first_record"], "first record", upper=MAX_RECORDS)
        for field in ("input_digest", "output_digest"):
            if case.get(field) is not None:
                digest(case[field], field)
        if case.get("input_digest") is not None:
            source = report.get("input_source")
            require(source is not None, "A case input digest requires seed evidence.")
            seed = source["seed_tokens"]
            tokens = [seed[position % len(seed)] for position in range(length)]
            require(case["input_digest"] == contracts.token_prefix_digest(tokens), "Case input digest differs from its planned input.")
        if case.get("output_digest") is not None:
            require(count is not None, "An unknown output count cannot have a final digest.")
            if count == 0:
                require(case["output_digest"] == contracts.token_prefix_digest([]), "The empty output digest differs.")
        if case["status"] == "completed":
            require(count is not None and case.get("input_digest") is not None and case.get("output_digest") is not None,
                    "A completed case lacks token evidence.")
            require(cleanup.get("generator_closed") is True and cleanup.get("settled") is True
                    and cleanup.get("error") is None and case.get("error") is None,
                    "A completed case lacks successful cleanup.")
            reason = case.get("reason")
            require(reason in ("length", "eos", "zero_output", "prefill_stop", "output_stop"),
                    "A completed case has an invalid reason.")
            if prefill is None and output is None:
                require(case["stop_condition_reached"] is None, "An unstopped case has a stop flag.")
            else:
                require(case["stop_condition_reached"] is (reason in ("prefill_stop", "output_stop")),
                        "The completed stop flag contradicts its reason.")
            if reason == "length":
                require(count == allowance and not case["emitted_eos"], "Length completion has inconsistent outputs.")
            elif reason == "eos":
                require(case["emitted_eos"] and count > 0, "EOS completion lacks EOS evidence.")
            elif reason == "zero_output":
                require(allowance == 0 and count == 0 and not case["emitted_eos"], "Invalid zero-output completion.")
            elif reason == "prefill_stop":
                require(prefill is not None and case["stop_condition_reached"] is True and count == 0,
                        "Prefill stop lacks matching stop evidence.")
            else:
                require(output is not None and case["stop_condition_reached"] is True and count == output,
                        "Output stop lacks matching stop evidence.")
        if case["status"] == "skipped":
            require(count == 0 and ranges["record_count"] == 0 and case.get("input_digest") is None,
                    "A skipped case contains execution evidence.")
    return {entry["case_id"]: entry for entry in plan}


def validate_journal(report, payload, plan):
    description = object_value(report.get("journal"), "journal description")
    require(description.get("file") == "samples.jsonl", "Unsupported journal filename.")
    require(description.get("sha256") == hashlib.sha256(payload).hexdigest(), "Journal digest differs.")
    records, ranges, run_samples = [], {name: [] for name in plan}, []
    previous_time, previous_case_index = -1, -1
    order = {name: index for index, name in enumerate(plan)}
    for index, line in enumerate(payload.splitlines(keepends=True)):
        require(index < MAX_RECORDS, "The journal exceeds its record limit.")
        require(len(line) <= MAX_LINE and line.endswith(b"\n"), "The journal has an oversized or incomplete line.")
        sample = object_value(parse_json(line), "journal sample")
        require(type(sample.get("record_index")) is int and sample["record_index"] == index, "Journal indices differ.")
        timestamp = integer(sample.get("monotonic_ns"), "sample time", upper=MAX_TIMESTAMP_NS)
        require(timestamp >= previous_time, "Journal timestamps decrease.")
        previous_time = timestamp
        phase, name = sample.get("phase"), sample.get("case_id")
        require(phase in PHASES, "Unknown journal phase.")
        count = integer(sample.get("output_tokens"), "observed outputs")
        if name is None:
            require(phase in ("before_load", "loaded") and previous_case_index == -1 and count == 0,
                    "Invalid startup sample.")
            require(index == (0 if phase == "before_load" else 1), "Startup samples are missing or duplicated.")
            run_samples.append(index)
        else:
            require(type(name) is str and name in plan, "A sample has an unknown case identity.")
            require(phase not in ("before_load", "loaded"), "A case contains a startup sample.")
            require(order[name] >= previous_case_index, "Case samples violate execution order.")
            previous_case_index = order[name]
            require(count <= plan[name]["max_tokens"], "Observed outputs exceed the allowance.")
            indices = ranges[name]
            if indices:
                require(records[indices[-1]]["output_tokens"] <= count, "Observed output counts decrease.")
            indices.append(index)
            require(len(indices) <= 4096, "A case exceeds its sample limit.")
        position = sample.get("reported_prompt_position")
        if phase == "prefill_progress":
            integer(position, "reported prompt position", upper=plan[name]["input_tokens"])
            require(count == 0, "Prefill progress follows output yields.")
        else:
            require(position is None, "A non-prefill sample contains a prompt position.")
        if phase == "before_request":
            require(count == 0, "A before-request sample contains outputs.")
        if phase == "first_output":
            require(count == 1, "A first-output sample has a different count.")
        if phase == "output_checkpoint":
            require(count > 0, "An output checkpoint has no outputs.")
        memory = sample.get("memory")
        if memory is not None:
            validate_memory(memory)
            require(sample.get("memory_error") is None, "Memory values also report an observation error.")
        elif phase != "before_load":
            require(type(sample.get("memory_error")) is str and bool(sample["memory_error"]),
                    "A missing memory observation requires its error.")
        if phase == "before_load":
            require(memory is None, "A before-load sample contains runtime memory.")
        host = object_value(sample.get("host"), "host observation")
        validate_pressure(host.get("pressure"))
        records.append(sample)
    require(type(description.get("record_count")) is int and description["record_count"] == len(records),
            "Journal record count differs.")
    require(type(report.get("run_samples")) is list and all(type(value) is int for value in report["run_samples"])
            and report["run_samples"] == run_samples, "Run sample references differ.")
    for case in report["cases"]:
        indices = ranges[case["case_id"]]
        claimed = case["samples"]
        require(claimed["record_count"] == len(indices), "Case sample count differs.")
        if indices:
            require(claimed["first_record"] == indices[0] and indices[-1] - indices[0] + 1 == len(indices),
                    "Case sample range differs.")
        else:
            require(claimed.get("first_record") is None or claimed["first_record"] <= len(records),
                    "An empty case range points beyond the journal.")
        count = case.get("actual_output_tokens")
        if count is not None:
            require(all(records[index]["output_tokens"] <= count for index in indices),
                    "A sample exceeds the case's actual output count.")
            require(all(records[index]["output_tokens"] == count for index in indices
                        if records[index]["phase"] in ("terminal", "settled")),
                    "A final observation differs from the actual output count.")
    return records, ranges


def group_epochs(run_id, records, plan):
    groups, owners, previous_epoch = [], {}, None
    for sample in records:
        memory = sample["memory"]
        if memory is None:
            continue
        epoch, phase, name = memory["peak_epoch"], sample["phase"], sample["case_id"]
        if previous_epoch is None or epoch != previous_epoch:
            require(epoch == (0 if previous_epoch is None else previous_epoch + 1), "Peak epochs skip or decrease.")
            if epoch == 0:
                require(name is None and phase == "loaded", "The loading epoch lacks its loaded observation.")
                owner = None
            else:
                require(name in plan and plan[name]["max_tokens"] > 0 and phase != "before_request",
                        "A non-generation sample starts a new peak epoch.")
                require(name not in owners, "A case owns more than one peak epoch.")
                owner = name
                owners[name] = epoch
            groups.append({"run_id": run_id, "peak_epoch": epoch, "owner_case_id": owner,
                           "record_indices": [], "inherited_record_indices": [],
                           "observed_peak_bytes": None, "owned_peak_bytes": None})
            previous_epoch = epoch
        group = groups[-1]
        peak = memory["allocator"]["peak_bytes"]
        require(group["observed_peak_bytes"] is None or peak >= group["observed_peak_bytes"],
                "A peak counter decreases within its epoch.")
        group["observed_peak_bytes"] = peak
        group["record_indices"].append(sample["record_index"])
        owned = ((epoch == 0 and name is None and phase == "loaded") or
                 (epoch > 0 and name == group["owner_case_id"] and phase != "before_request"))
        if phase in ("prefill_progress", "first_output", "output_checkpoint"):
            require(owned and epoch > 0, "Generation observations reuse another operation's peak epoch.")
        if owned:
            group["owned_peak_bytes"] = peak if group["owned_peak_bytes"] is None else max(group["owned_peak_bytes"], peak)
        else:
            group["inherited_record_indices"].append(sample["record_index"])
    return groups, owners


def validate_execution(report, records, ranges, owners):
    boolean(report.get("load_started"), "load started")
    startup = {sample["phase"]: sample for sample in records if sample["case_id"] is None}
    if report.get("identity") is not None or "loaded" in startup:
        require(report["load_started"], "Loaded evidence contradicts load_started.")
    if "before_load" in startup and startup["before_load"]["host"]["pressure"]["state"] != "normal":
        require(not report["load_started"] and report.get("identity") is None and len(records) == 1,
                "Execution follows a blocked before-load decision.")
    if any(ranges[case["case_id"]] or case["input_digest"] is not None for case in report["cases"]):
        require(report["load_started"] and report.get("identity") is not None and report.get("input_source") is not None,
                "Executed cases lack loaded identity or input evidence.")
        require(set(startup) == {"before_load", "loaded"}
                and all(sample["host"]["pressure"]["state"] == "normal" for sample in startup.values()),
                "Executed cases lack normal startup observations.")
    ranks = {"before_request": 0, "prefill_progress": 1, "first_output": 2,
             "output_checkpoint": 3, "terminal": 4, "settled": 5}
    for planned, case in zip(report["plan"], report["cases"]):
        samples = [records[index] for index in ranges[case["case_id"]]]
        previous_rank, previous_position, blocked = -1, -1, False
        seen = set()
        for sample in samples:
            phase = sample["phase"]
            require(ranks[phase] >= previous_rank, "Case phases violate execution order.")
            previous_rank = ranks[phase]
            if phase in ("before_request", "first_output", "terminal", "settled"):
                require(phase not in seen, "A case repeats a single-occurrence phase.")
            seen.add(phase)
            if phase == "prefill_progress":
                position = sample["reported_prompt_position"]
                require(position >= previous_position, "Reported prompt positions decrease.")
                previous_position = position
            if phase in ("prefill_progress", "first_output", "output_checkpoint"):
                require(not blocked, "Generation continues after a blocked pressure observation.")
            blocked = blocked or sample["host"]["pressure"]["state"] != "normal"
            if phase == "settled" and sample["memory"] is not None:
                require(sample["memory"]["layers"] == [], "Settled observations retain cache layers.")
        if case["status"] == "completed":
            require({"before_request", "terminal", "settled"} <= seen,
                    "A completed case lacks required lifecycle observations.")
            require(not blocked, "A completed case contains blocked pressure observations.")
            if planned["max_tokens"] > 0:
                progress = [sample for sample in samples if sample["phase"] == "prefill_progress"]
                epoch = owners.get(case["case_id"])
                require(progress and progress[0]["reported_prompt_position"] == 0 and epoch is not None,
                        "A completed generation lacks its initial progress observation.")
                require(all(sample["memory"] is not None and sample["memory"]["peak_epoch"] == epoch
                            for sample in samples if sample["phase"] != "before_request"),
                        "Completed generation observations disagree on their epoch.")
                if case["actual_output_tokens"] > 0:
                    require("first_output" in seen and previous_position == planned["input_tokens"],
                            "Completed outputs lack prefill or first-output observations.")


def layer_summary(samples):
    layers = {}
    for sample in samples:
        memory = sample["memory"]
        if memory is None:
            continue
        for index, layer in enumerate(memory["layers"]):
            if index not in layers:
                layers[index] = {"index": index, "type": layer["type"], "max_offset": None,
                                 "max_nbytes": None, "unknown_offset_observed": False,
                                 "unknown_nbytes_observed": False}
            current = layers[index]
            require(current["type"] == layer["type"], "A cache layer changes type within one generation.")
            for field in ("offset", "nbytes"):
                value = layer[field]
                if value is None:
                    current[f"unknown_{field}_observed"] = True
                else:
                    previous = current[f"max_{field}"]
                    current[f"max_{field}"] = value if previous is None else max(previous, value)
    return list(layers.values())


def reduce_case(planned, result, samples, process, epochs, owners):
    name, allowance = planned["case_id"], planned["max_tokens"]
    epoch = owners.get(name)
    owned = [sample for sample in samples if sample["memory"] is not None
             and sample["memory"]["peak_epoch"] == epoch and sample["phase"] != "before_request"]
    layers = layer_summary(owned)
    offsets = [layer["max_offset"] for layer in layers if layer["max_offset"] is not None]
    count = result["actual_output_tokens"]
    reasons = []
    if result["status"] != "completed":
        reasons.append(f"case_{result['status']}")
    if not process["reaped"]:
        reasons.append("process_reaping_unconfirmed")
    cleanup = result["cleanup"]
    if cleanup["generator_closed"] is not True or cleanup["settled"] is not True or cleanup.get("error") is not None:
        reasons.append("cleanup_unconfirmed")
    phases = {sample["phase"] for sample in samples}
    if not {"terminal", "settled"} <= phases:
        reasons.append("final_observations_missing")
    if allowance > 0 and epoch is None:
        reasons.append("generation_epoch_unobserved")
    if any(sample["host"]["pressure"]["state"] != "normal" or sample["memory"] is None for sample in samples):
        reasons.append("observation_blocked_or_missing")
    prefill, output = planned["stop_after_prefill"], planned["stop_after_output"]
    progress = [sample["reported_prompt_position"] for sample in samples if sample["phase"] == "prefill_progress"]
    full_allowance = bool(samples) and count is not None and count == allowance
    if prefill is not None:
        boundary = result["stop_condition_reached"] is True and any(value >= prefill for value in progress)
    elif output is not None:
        boundary = result["stop_condition_reached"] is True and count is not None and count >= output
    else:
        boundary = full_allowance
    if not boundary:
        reasons.append("early_eos" if result["emitted_eos"] and not full_allowance else "target_not_reached")
    return {"case_id": name, "plan": planned, "planned": True,
            "executed": bool(samples) or result["input_digest"] is not None,
            "generation_observed": epoch is not None, "reached": not reasons,
            "unreached_reasons": reasons, "status": result["status"], "reason": result["reason"],
            "actual_output_tokens": count, "observed_output_tokens_lower_bound": max(
                (sample["output_tokens"] for sample in samples), default=0),
            "emitted_eos": result["emitted_eos"], "full_allowance_observed": full_allowance,
            "stop_condition_reached": result["stop_condition_reached"],
            "max_reported_prompt_position": max(progress, default=None),
            "cleanup": cleanup, "process_reaped": process["reaped"],
            "peak_epoch": epoch, "owned_peak_bytes": epochs[epoch]["owned_peak_bytes"] if epoch is not None else None,
            "cache_layers": layers, "max_observed_cache_offset": max(offsets, default=None)}


def missing_intervals(first, last, points):
    result, next_value = [], first
    for value in sorted(set(points)):
        if value > next_value:
            result.append([next_value, value - 1])
        next_value = value + 1
    if next_value <= last:
        result.append([next_value, last])
    return result


def describe_shapes(shapes, context, max_output):
    shapes = sorted(set(shapes))
    by_output = {}
    for length, allowance in shapes:
        by_output.setdefault(allowance, []).append(length)
    total = (max_output + 1) * context - max_output * (max_output + 1) // 2
    return {"shapes": [{"input_tokens": length, "max_tokens": allowance} for length, allowance in shapes],
            "shape_count": len(shapes), "missing_shape_count": total - len(shapes),
            "output_allowances": sorted(by_output),
            "missing_output_allowance_ranges": missing_intervals(0, max_output, by_output),
            "missing_input_ranges_by_allowance": [
                {"max_tokens": allowance, "input_ranges": missing_intervals(1, context - allowance, lengths)}
                for allowance, lengths in sorted(by_output.items())],
            "exact_shape_domain_exhausted": len(shapes) == total}


def analyze(directory):
    """Return a conservative report. Invalid evidence has no derived coverage."""
    directory = Path(directory)
    output = {"format": "apxinf-memory-coverage-v1", "artifact_kind": "coverage_evidence",
              "admission_approved": False, "full_memory_domain_established": False,
              "frozen_input_validation": "unavailable",
              "sources": {}, "validation": {"status": "invalid", "errors": [], "limitations": list(LIMITATIONS)},
              "run_id": None, "identity": None, "requested_profile": None, "hardware": None,
              "process": None, "cases": [], "peak_epochs": [], "shape_domain": None, "summary": None}
    try:
        report = parse_json(read_source(directory, "calibration.json", MAX_DOCUMENT, output["sources"]))
        declared_frozen = type(report) is dict and report.get("measurement_inputs") is not None
        if declared_frozen:
            output["frozen_input_validation"] = "invalid"
        process = parse_json(read_source(directory, "process.json", MAX_DOCUMENT, output["sources"]))
        payload = read_source(directory, "samples.jsonl", MAX_JOURNAL, output["sources"])
        profile = validate_header(report, process)
        plan = validate_cases(report, profile, frozen=declared_frozen)
        output["frozen_input_validation"] = validate_frozen_inputs(directory, report, output["sources"])
        if not declared_frozen:
            output["validation"]["limitations"].append(
                "Frozen input validation is unavailable and cannot establish provenance for a held-out experiment.")
        records, ranges = validate_journal(report, payload, plan)
        epochs, owners = group_epochs(report["run_id"], records, plan)
        validate_execution(report, records, ranges, owners)
        if process["pid"] is None:
            require(not records and report.get("identity") is None
                    and all(case["status"] == "skipped" for case in report["cases"]),
                    "An absent child cannot produce runtime evidence.")
        if report["status"] == "completed":
            require(not process["reaped"] or process["returncode"] == 0,
                    "A completed, reaped run requires successful process exit.")
            require(report.get("identity") is not None and report.get("input_source") is not None,
                    "A completed run requires loaded identity and input evidence.")
            cleanup = object_value(report.get("run_cleanup"), "run cleanup")
            require(cleanup.get("settled") is True and cleanup.get("error") is None,
                    "A completed run requires successful final settlement.")
            require(all(case["status"] == "completed" for case in report["cases"]),
                    "A completed run contains unfinished cases.")
        cases = [reduce_case(planned, result, [records[index] for index in ranges[result["case_id"]]],
                             process, epochs, owners) for planned, result in zip(report["plan"], report["cases"])]
        context = profile["max_context"]
        max_output = min(profile["max_output_tokens"], context - 1)
        planned_shapes, reached_shapes = [], []
        for case in cases:
            entry = case["plan"]
            if entry["stop_after_prefill"] is None and entry["stop_after_output"] is None:
                shape = (entry["input_tokens"], entry["max_tokens"])
                planned_shapes.append(shape)
                if case["reached"]:
                    reached_shapes.append(shape)
        domain = {"max_context": context, "max_output_tokens": max_output,
                  "legal_shape_count": (max_output + 1) * context - max_output * (max_output + 1) // 2,
                  "planned": describe_shapes(planned_shapes, context, max_output),
                  "reached": describe_shapes(reached_shapes, context, max_output)}
        output.update(run_id=report["run_id"], identity=report.get("identity"), requested_profile=profile,
                      hardware=report.get("hardware"), process=process, cases=cases, peak_epochs=epochs,
                      shape_domain=domain,
                      summary={field: sum(case[field] for case in cases)
                               for field in ("planned", "executed", "generation_observed", "reached")})
        output["validation"]["status"] = "valid"
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError) as error:
        output["validation"]["errors"].append(f"{type(error).__name__}: {error}")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = analyze(args.directory)
    payload = json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    if args.output is None:
        sys.stdout.write(payload)
    else:
        try:
            with args.output.open("x", encoding="utf-8") as destination:
                destination.write(payload)
        except OSError as error:
            print(f"Cannot create coverage report: {error}", file=sys.stderr)
            return 1
    return 0 if report["validation"]["status"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
