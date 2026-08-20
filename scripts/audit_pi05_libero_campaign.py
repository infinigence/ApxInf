#!/usr/bin/env python3
"""Audit one precision-tagged ApxInf PI0.5 LIBERO campaign ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-jsonl", required=True, type=pathlib.Path)
    parser.add_argument("--summary-json", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--precision", required=True, choices=("fp8", "bf16", "int8"))
    parser.add_argument("--task-ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--trials-per-task", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--image-input", choices=("nhwc", "nchw", "patches"), default="nhwc"
    )
    parser.add_argument("--minimum-successes", type=int)
    return parser.parse_args()


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_nonnegative(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def main() -> None:
    args = parse_args()
    task_ids = [int(value) for value in args.task_ids.split(",") if value.strip()]
    if sorted(set(task_ids)) != sorted(task_ids) or any(
        not 0 <= value < 10 for value in task_ids
    ):
        raise ValueError("--task-ids must contain unique IDs in 0..9")
    if args.trials_per_task <= 0:
        raise ValueError("--trials-per-task must be positive")

    expected_keys = {
        (task_id, trial_id)
        for task_id in task_ids
        for trial_id in range(args.trials_per_task)
    }
    if args.minimum_successes is not None and not (
        0 <= args.minimum_successes <= len(expected_keys)
    ):
        raise ValueError(f"--minimum-successes must be in 0..={len(expected_keys)}")
    failures: list[str] = []
    completed: dict[tuple[int, int], dict] = {}
    technical_errors: list[dict] = []
    unknown_statuses: list[dict] = []
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
            if row.get("precision") != args.precision:
                failures.append(
                    f"results line {line_number} precision is {row.get('precision')!r}, "
                    f"expected {args.precision!r}"
                )
            status = row.get("status")
            if status == "technical_error":
                technical_errors.append(row)
                continue
            if status != "completed":
                unknown_statuses.append(row)
                continue
            task_id = row.get("task_id")
            trial_id = row.get("trial_id")
            if (
                not isinstance(task_id, int)
                or isinstance(task_id, bool)
                or not isinstance(trial_id, int)
                or isinstance(trial_id, bool)
            ):
                failures.append(f"results line {line_number} has an invalid key")
                continue
            key = (task_id, trial_id)
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
    if technical_errors:
        failures.append(f"technical errors present: {len(technical_errors)}")
    if unknown_statuses:
        failures.append(f"unknown result statuses: {len(unknown_statuses)}")

    per_task = {}
    total_replans = 0
    total_action_steps = 0
    total_preprocess_seconds = 0.0
    total_inference_seconds = 0.0
    total_elapsed_seconds = 0.0
    for task_id in task_ids:
        rows = [completed[key] for key in sorted(completed) if key[0] == task_id]
        successes = sum(row.get("success") is True for row in rows)
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
                failures.append(
                    f"{key}: seed is {row.get('seed')}, expected {args.seed}"
                )
            if row.get("image_input") != args.image_input:
                failures.append(
                    f"{key}: image_input is {row.get('image_input')!r}, "
                    f"expected {args.image_input!r}"
                )
            if row.get("attempt") != 1:
                failures.append(f"{key}: attempt is {row.get('attempt')}, expected 1")
            if not isinstance(row.get("success"), bool):
                failures.append(f"{key}: success is not boolean")
            token_count = row.get("token_count")
            if (
                not isinstance(token_count, int)
                or isinstance(token_count, bool)
                or not 0 < token_count <= 200
            ):
                failures.append(f"{key}: invalid token_count {token_count}")
            action_steps = row.get("action_steps")
            replans = row.get("replans")
            if (
                not isinstance(action_steps, int)
                or isinstance(action_steps, bool)
                or not 0 < action_steps <= 520
            ):
                failures.append(f"{key}: invalid action_steps {action_steps}")
                continue
            if (
                not isinstance(replans, int)
                or isinstance(replans, bool)
                or replans != (action_steps + 4) // 5
            ):
                failures.append(
                    f"{key}: replans {replans} disagree with action_steps {action_steps}"
                )
                continue
            timings = {}
            for field in (
                "preprocess_seconds",
                "inference_seconds",
                "elapsed_seconds",
            ):
                value = row.get(field)
                if not finite_nonnegative(value):
                    failures.append(f"{key}: invalid {field} {value}")
                else:
                    timings[field] = float(value)
            checksum = row.get("first_normalized_action_abs_checksum")
            if not finite_nonnegative(checksum) or checksum == 0:
                failures.append(f"{key}: invalid first action checksum {checksum}")
            total_action_steps += action_steps
            total_replans += replans
            total_preprocess_seconds += timings.get("preprocess_seconds", 0.0)
            total_inference_seconds += timings.get("inference_seconds", 0.0)
            total_elapsed_seconds += timings.get("elapsed_seconds", 0.0)

    successes = sum(row.get("success") is True for row in completed.values())
    if args.minimum_successes is not None and successes < args.minimum_successes:
        failures.append(
            f"successes {successes} are below required minimum {args.minimum_successes}"
        )

    evaluator_summary = json.loads(args.summary_json.read_text())
    expected_summary = {
        "schema": "apxinf.pi05.libero-eval.v1",
        "suite": "libero_10",
        "precision": args.precision,
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

    audit = {
        "schema": "apxinf.pi05.libero-campaign-audit.v1",
        "passed": not failures,
        "failures": failures,
        "requirements": {
            "precision": args.precision,
            "task_ids": task_ids,
            "trials_per_task": args.trials_per_task,
            "expected_completed_runs": len(expected_keys),
            "require_zero_technical_errors": True,
            "seed": args.seed,
            "image_input": args.image_input,
            "minimum_successes": args.minimum_successes,
        },
        "evidence": {
            "results_jsonl": str(args.results_jsonl),
            "results_sha256": sha256(args.results_jsonl),
            "summary_json": str(args.summary_json),
            "summary_sha256": sha256(args.summary_json),
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
            "mean_preprocess_ms_per_replan": (
                1000.0 * total_preprocess_seconds / total_replans
                if total_replans
                else None
            ),
            "mean_inference_ms_per_replan": (
                1000.0 * total_inference_seconds / total_replans
                if total_replans
                else None
            ),
            "episode_elapsed_seconds_sum": total_elapsed_seconds,
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
