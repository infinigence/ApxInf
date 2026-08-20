#!/usr/bin/env python3
"""Strictly audit a 10x10 PI0.5 LIBERO campaign and its BF16 parity gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib


TASK_COUNT = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-jsonl", required=True, type=pathlib.Path)
    parser.add_argument("--summary-json", required=True, type=pathlib.Path)
    parser.add_argument("--calibration", required=True, type=pathlib.Path)
    parser.add_argument("--parity-calibration", required=True, type=pathlib.Path)
    parser.add_argument("--parity-json", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--trials-per-task", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--require-zero-technical-errors", action="store_true")
    return parser.parse_args()


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_nonnegative(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def main() -> None:
    args = parse_args()
    if args.trials_per_task <= 0:
        raise ValueError("--trials-per-task must be positive")

    expected_keys = {
        (task_id, trial_id)
        for task_id in range(TASK_COUNT)
        for trial_id in range(args.trials_per_task)
    }
    failures: list[str] = []
    completed: dict[tuple[int, int], dict] = {}
    technical_errors = []
    unknown_statuses = []
    nonempty_lines = 0
    with args.results_jsonl.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            nonempty_lines += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                failures.append(f"results line {line_number} is invalid JSON: {error}")
                continue
            status = row.get("status")
            if status == "technical_error":
                technical_errors.append(row)
                continue
            if status != "completed":
                unknown_statuses.append(row)
                continue
            try:
                key = (int(row["task_id"]), int(row["trial_id"]))
            except (KeyError, TypeError, ValueError) as error:
                failures.append(f"results line {line_number} has an invalid key: {error}")
                continue
            if key in completed:
                failures.append(f"duplicate completed result {key}")
            completed[key] = row

    actual_keys = set(completed)
    missing_keys = sorted(expected_keys - actual_keys)
    extra_keys = sorted(actual_keys - expected_keys)
    if missing_keys:
        failures.append(f"missing completed keys: {missing_keys}")
    if extra_keys:
        failures.append(f"unexpected completed keys: {extra_keys}")
    if unknown_statuses:
        failures.append(f"unknown result statuses: {len(unknown_statuses)}")
    if args.require_zero_technical_errors and technical_errors:
        failures.append(f"technical errors present: {len(technical_errors)}")

    per_task = {}
    total_replans = 0
    total_action_steps = 0
    total_preprocess_seconds = 0.0
    total_inference_seconds = 0.0
    total_elapsed_seconds = 0.0
    for task_id in range(TASK_COUNT):
        rows = [completed[key] for key in sorted(completed) if key[0] == task_id]
        successes = sum(bool(row.get("success")) for row in rows)
        per_task[str(task_id)] = {
            "completed": len(rows),
            "successes": successes,
            "success_rate": successes / len(rows) if rows else None,
        }
        for row in rows:
            key = (int(row["task_id"]), int(row["trial_id"]))
            if row.get("suite") != "libero_10":
                failures.append(f"{key}: suite is not libero_10")
            if row.get("seed") != args.seed:
                failures.append(f"{key}: seed is {row.get('seed')}, expected {args.seed}")
            if row.get("attempt") != 1 and args.require_zero_technical_errors:
                failures.append(f"{key}: attempt is {row.get('attempt')}, expected 1")
            token_count = row.get("token_count")
            if not isinstance(token_count, int) or not 0 < token_count <= 200:
                failures.append(f"{key}: invalid token_count {token_count}")
            action_steps = row.get("action_steps")
            replans = row.get("replans")
            if not isinstance(action_steps, int) or not 0 < action_steps <= 520:
                failures.append(f"{key}: invalid action_steps {action_steps}")
                continue
            if not isinstance(replans, int) or replans != (action_steps + 4) // 5:
                failures.append(
                    f"{key}: replans {replans} disagree with action_steps {action_steps}"
                )
                continue
            for field in ("preprocess_seconds", "inference_seconds", "elapsed_seconds"):
                if not finite_nonnegative(row.get(field)):
                    failures.append(f"{key}: invalid {field} {row.get(field)}")
            checksum = row.get("first_normalized_action_abs_checksum")
            if not finite_nonnegative(checksum) or checksum == 0:
                failures.append(f"{key}: invalid first action checksum {checksum}")
            total_action_steps += action_steps
            total_replans += replans
            total_preprocess_seconds += float(row.get("preprocess_seconds", 0.0))
            total_inference_seconds += float(row.get("inference_seconds", 0.0))
            total_elapsed_seconds += float(row.get("elapsed_seconds", 0.0))

    successes = sum(bool(row.get("success")) for row in completed.values())
    evaluator_summary = json.loads(args.summary_json.read_text())
    expected_summary = {
        "schema": "apxinf.pi05.libero-eval.v1",
        "suite": "libero_10",
        "expected_runs": len(expected_keys),
        "completed_runs": len(completed),
        "missing_runs": [],
        "successes": successes,
        "success_rate": successes / len(completed) if completed else None,
        "per_task": per_task,
    }
    for field, expected in expected_summary.items():
        if evaluator_summary.get(field) != expected:
            failures.append(
                f"summary field {field!r} is {evaluator_summary.get(field)!r}, "
                f"expected {expected!r}"
            )

    calibration_hash = sha256(args.calibration)
    parity_calibration_hash = sha256(args.parity_calibration)
    if calibration_hash != parity_calibration_hash:
        failures.append("rollout calibration is not byte-identical to parity calibration")
    calibration = json.loads(args.calibration.read_text())
    if calibration.get("bootstrap_margin") != 2.35:
        failures.append(
            f"rollout calibration margin is {calibration.get('bootstrap_margin')}, expected 2.35"
        )

    parity = json.loads(args.parity_json.read_text())
    parity_results = parity.get("results", [])
    if parity.get("schema") != "apxinf.pi05.libero-calibration-sweep.v1":
        failures.append(f"unexpected parity schema {parity.get('schema')!r}")
    if parity.get("min_cosine") != 0.997 or parity.get("max_relative_l2") != 0.10:
        failures.append("parity thresholds are not cosine >= 0.997 and relative L2 <= 0.10")
    if len(parity_results) != 1:
        failures.append(f"expected one frozen parity profile, got {len(parity_results)}")
    parity_fixtures = parity_results[0].get("fixtures", []) if parity_results else []
    if len(parity_fixtures) != TASK_COUNT:
        failures.append(f"expected {TASK_COUNT} parity fixtures, got {len(parity_fixtures)}")
    if parity_results and not parity_results[0].get("passed"):
        failures.append("frozen calibration did not pass its aggregate parity gate")
    if any(not fixture.get("passed") for fixture in parity_fixtures):
        failures.append("at least one BF16 fixture failed its parity gate")

    mean_preprocess_ms = (
        1000.0 * total_preprocess_seconds / total_replans if total_replans else None
    )
    mean_inference_ms = (
        1000.0 * total_inference_seconds / total_replans if total_replans else None
    )
    audit = {
        "schema": "apxinf.pi05.libero-integrity-audit.v1",
        "passed": not failures,
        "failures": failures,
        "requirements": {
            "task_count": TASK_COUNT,
            "trials_per_task": args.trials_per_task,
            "expected_completed_runs": len(expected_keys),
            "require_zero_technical_errors": args.require_zero_technical_errors,
            "seed": args.seed,
            "minimum_bf16_cosine": parity.get("min_cosine"),
            "maximum_bf16_relative_l2": parity.get("max_relative_l2"),
        },
        "evidence": {
            "results_jsonl": str(args.results_jsonl),
            "results_sha256": sha256(args.results_jsonl),
            "summary_json": str(args.summary_json),
            "summary_sha256": sha256(args.summary_json),
            "calibration": str(args.calibration),
            "calibration_sha256": calibration_hash,
            "parity_calibration": str(args.parity_calibration),
            "parity_calibration_sha256": parity_calibration_hash,
            "parity_json": str(args.parity_json),
            "parity_json_sha256": sha256(args.parity_json),
        },
        "campaign": {
            "nonempty_ledger_lines": nonempty_lines,
            "completed_runs": len(completed),
            "technical_errors": len(technical_errors),
            "unknown_statuses": len(unknown_statuses),
            "missing_keys": missing_keys,
            "extra_keys": extra_keys,
            "successes": successes,
            "success_rate": successes / len(completed) if completed else None,
            "per_task": per_task,
            "total_action_steps": total_action_steps,
            "total_replans": total_replans,
            "mean_preprocess_ms_per_replan": mean_preprocess_ms,
            "mean_inference_ms_per_replan": mean_inference_ms,
            "episode_elapsed_seconds_sum": total_elapsed_seconds,
        },
        "parity": {
            "fixture_count": len(parity_fixtures),
            "minimum_cosine": (
                min(row["cosine"] for row in parity_fixtures) if parity_fixtures else None
            ),
            "maximum_relative_l2": (
                max(row["relative_l2"] for row in parity_fixtures)
                if parity_fixtures
                else None
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(audit, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
