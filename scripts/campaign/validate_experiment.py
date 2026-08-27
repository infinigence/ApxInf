#!/usr/bin/env python3
"""Validate a member3 paired A/B experiment manifest offline."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
PLACEHOLDER_RE = re.compile(r"^(?:TODO|TBD|<[^>]+>|)$", re.IGNORECASE)
REQUIRED_TOP = (
    "schema", "status", "experiment_id", "owner", "branch", "base_good_sha",
    "commit_sha", "model_revision", "contract_sha256", "input_manifest_sha256",
    "hypothesis", "primary_variable", "baseline", "candidate", "environment",
    "measurement", "evidence", "conclusion", "rollback",
)

RELIABILITY_FIELDS = (
    "no_unexpected_oom",
    "no_nan",
    "no_fallback",
    "no_xid",
)
RECOVERY_FIELDS = ("service_healthy_after_failure", "recovery_pass")


def missing(value: Any, allow_placeholder: bool) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        if allow_placeholder:
            return False
        return (not value.strip()) or bool(PLACEHOLDER_RE.match(value.strip()))
    return False


def cv(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    if mean == 0:
        return math.inf if any(values) else 0.0
    return statistics.stdev(values) / mean


def validate(data: dict[str, Any], mode: str) -> list[str]:
    errors: list[str] = []
    allow_placeholder = mode == "template"
    for key in REQUIRED_TOP:
        if key not in data:
            errors.append(f"missing top-level field: {key}")
    if data.get("schema") != "apxinf.member3.experiment.v1":
        errors.append("schema must be apxinf.member3.experiment.v1")
    if data.get("owner") != "member3":
        errors.append("owner must be member3")
    if data.get("status") not in {"planned", "active", "review", "accepted", "rejected", "inconclusive"}:
        errors.append("status is invalid")
    for key in ("experiment_id", "branch", "hypothesis", "primary_variable"):
        if missing(data.get(key), allow_placeholder):
            errors.append(f"{key} is required")
    for key in ("commit_sha", "base_good_sha"):
        if key in data and not allow_placeholder and not COMMIT_RE.fullmatch(str(data[key])):
            errors.append(f"{key} must be a full 40-character commit SHA")
    for key in ("contract_sha256", "input_manifest_sha256"):
        if key in data and not allow_placeholder and not SHA256_RE.fullmatch(str(data[key])):
            errors.append(f"{key} must be a 64-character SHA256")
    if missing(data.get("model_revision"), allow_placeholder):
        errors.append("model_revision is required")
    elif not allow_placeholder and not COMMIT_RE.fullmatch(str(data.get("model_revision"))):
        errors.append("model_revision must be a full 40-character revision")
    for side in ("baseline", "candidate"):
        value = data.get(side)
        if not isinstance(value, dict):
            errors.append(f"{side} must be an object")
            continue
        for key in ("feature", "command"):
            if missing(value.get(key), allow_placeholder):
                errors.append(f"{side}.{key} is required")
    baseline = data.get("baseline") or {}
    candidate = data.get("candidate") or {}
    if baseline.get("feature") != "off":
        errors.append("baseline.feature must be off")
    if candidate.get("feature") != "on":
        errors.append("candidate.feature must be on")
    if baseline.get("command") == candidate.get("command") and not allow_placeholder:
        errors.append("baseline and candidate commands must differ")
    if baseline.get("config") != candidate.get("config") and not allow_placeholder:
        errors.append("baseline and candidate fixed config must match")
    measurement = data.get("measurement") or {}
    warmup = measurement.get("warmup_repeats")
    repeats = measurement.get("measured_repeats")
    if not isinstance(warmup, int) or warmup < 1:
        errors.append("measurement.warmup_repeats must be >= 1")
    if not isinstance(repeats, int) or repeats < 5:
        errors.append("measurement.measured_repeats must be >= 5")
    if not isinstance(measurement.get("timeout_s"), (int, float)) or measurement.get("timeout_s", 0) <= 0:
        errors.append("measurement.timeout_s must be positive")
    if missing(measurement.get("clock_policy"), allow_placeholder):
        errors.append("measurement.clock_policy is required")
    environment = data.get("environment") or {}
    for key in (
        "gpu_uuid", "gpu_model", "driver_version", "cuda_version",
        "apxinf_cuda_arch", "replay_lane", "evidence_scope",
    ):
        if missing(environment.get(key), allow_placeholder):
            errors.append(f"environment.{key} is required")
    evidence = data.get("evidence") or {}
    for section in ("correctness", "reliability", "recovery", "latency", "memory"):
        if not isinstance(evidence.get(section), dict):
            errors.append(f"evidence.{section} must be an object")
    for key in ("raw_artifact_path", "raw_artifact_sha256"):
        if missing(evidence.get(key), allow_placeholder):
            errors.append(f"evidence.{key} is required")
    rollback = data.get("rollback") or {}
    if not isinstance(rollback, dict) or missing(rollback.get("sha"), allow_placeholder) or missing(rollback.get("command"), allow_placeholder):
        errors.append("rollback.sha and rollback.command are required")
    if mode == "ready":
        if not COMMIT_RE.fullmatch(str(rollback.get("sha", ""))):
            errors.append("rollback.sha must be a full 40-character commit SHA")
        gate_failures: list[str] = []
        correctness = evidence.get("correctness") or {}
        for key in ("baseline_pass", "candidate_pass"):
            if correctness.get(key) is not True:
                gate_failures.append(f"evidence.correctness.{key}")
        reliability = evidence.get("reliability") or {}
        for key in RELIABILITY_FIELDS:
            if reliability.get(key) is not True:
                gate_failures.append(f"evidence.reliability.{key}")
        recovery = evidence.get("recovery") or {}
        for key in RECOVERY_FIELDS:
            if recovery.get(key) is not True:
                gate_failures.append(f"evidence.recovery.{key}")
        if not SHA256_RE.fullmatch(str(evidence.get("raw_artifact_sha256", ""))):
            errors.append("evidence.raw_artifact_sha256 must be a SHA256")
        for label, side in (("baseline", baseline), ("candidate", candidate)):
            latency = ((evidence.get("latency") or {}).get(label) or {})
            for metric in ("ttft_ms", "tpot_ms"):
                values = latency.get(metric)
                if not isinstance(values, list) or len(values) < repeats:
                    gate_failures.append(f"evidence.latency.{label}.{metric}.measured_repeats")
                elif any(not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in values):
                    gate_failures.append(f"evidence.latency.{label}.{metric}.non_finite")
                elif (value_cv := cv([float(v) for v in values])) is None or value_cv > 0.10:
                    gate_failures.append(f"evidence.latency.{label}.{metric}.cv")
        conclusion = data.get("conclusion") or {}
        decision = conclusion.get("decision")
        if decision not in {"accepted", "rejected", "inconclusive"}:
            errors.append("conclusion.decision must be accepted, rejected, or inconclusive")
        if missing(conclusion.get("reason"), False):
            errors.append("conclusion.reason is required")
        if gate_failures and decision != "rejected":
            errors.append("failed gates require conclusion.decision=rejected: " + ", ".join(gate_failures))
        if not gate_failures and decision == "rejected" and missing(conclusion.get("reason"), False):
            errors.append("a rejected decision without failed gates needs conclusion.reason")
        if decision == "accepted" and data.get("status") != "accepted":
            errors.append("an accepted conclusion requires status=accepted")
        if decision == "accepted" and environment.get("replay_lane") != "GPU0":
            errors.append("an accepted conclusion requires environment.replay_lane=GPU0")
        if decision == "accepted" and "4090" not in str(environment.get("gpu_model", "")):
            errors.append("an accepted conclusion requires an RTX 4090 GPU model")
        if decision == "accepted" and environment.get("evidence_scope") != "official":
            errors.append("an accepted conclusion requires environment.evidence_scope=official")
        if decision == "accepted" and environment.get("apxinf_cuda_arch") != "sm_89":
            errors.append("an accepted conclusion requires APXINF_CUDA_ARCH=sm_89")
        if decision == "rejected" and data.get("status") != "rejected":
            errors.append("a rejected conclusion requires status=rejected")
        if decision == "inconclusive" and data.get("status") != "inconclusive":
            errors.append("an inconclusive conclusion requires status=inconclusive")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--mode", choices=("template", "ready"), default="template")
    parser.add_argument("--json", action="store_true", help="emit machine-readable result")
    args = parser.parse_args()
    try:
        data = json.loads(args.manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"validate_experiment: error: {error}", file=sys.stderr)
        return 2
    if not isinstance(data, dict):
        errors = ["manifest root must be an object"]
    else:
        errors = validate(data, args.mode)
    result = {"manifest": args.manifest.as_posix(), "mode": args.mode, "valid": not errors, "errors": errors}
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("VALID" if not errors else "INVALID")
        for error in errors:
            print(f"- {error}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
