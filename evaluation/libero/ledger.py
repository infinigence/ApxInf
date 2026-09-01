"""Resumable result ledger and aggregate LIBERO accuracy summary."""

from __future__ import annotations

import json
import os
import pathlib
import time
from typing import Tuple


LedgerKey = Tuple[str, int, int]


def completed_runs(path: pathlib.Path, precision: str) -> dict[LedgerKey, dict]:
    """Load completed rows, rejecting mixed precision and duplicate episodes."""
    result: dict[LedgerKey, dict] = {}
    if not path.exists():
        return result
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        item_precision = item.get("precision", "fp8")
        if item_precision != precision:
            raise ValueError(
                f"ledger precision is {item_precision!r}, requested {precision!r} "
                f"at line {line_number}"
            )
        if item.get("status") != "completed":
            continue
        key: LedgerKey = (
            str(item["suite"]),
            int(item["task_id"]),
            int(item["trial_id"]),
        )
        if key in result:
            raise ValueError(f"duplicate completed run {key} at line {line_number}")
        result[key] = item
    return result


def append_record(path: pathlib.Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _aggregate_timing(rows: list[dict]) -> dict:
    completed = [row for row in rows if row.get("status") == "completed"]
    total_replans = sum(int(row.get("replans", 0)) for row in completed)
    segments = (
        "preprocess_seconds",
        "model_seconds",
        "server_processor_seconds",
        "websocket_transport_seconds",
        "inference_seconds",
    )
    totals = {
        name: sum(float(row.get(name, 0.0)) for row in completed) for name in segments
    }
    per_call_ms = {
        name.replace("_seconds", "_ms"): (
            value / total_replans * 1000.0 if total_replans else None
        )
        for name, value in totals.items()
    }
    return {
        "episodes": len(completed),
        "total_inference_calls": total_replans,
        "total_seconds": {name: round(value, 6) for name, value in totals.items()},
        "per_call_ms": {
            name: round(value, 4) if value is not None else None
            for name, value in per_call_ms.items()
        },
    }


def write_summary(
    path: pathlib.Path,
    ledger: dict[LedgerKey, dict],
    expected_keys: set[LedgerKey],
    precision: str,
    transport: str,
) -> None:
    """Atomically write success rates over completed episodes, not missing ones."""
    per_suite: dict[str, dict] = {}
    for suite in sorted({key[0] for key in expected_keys}):
        per_task = {}
        task_ids = {task for name, task, _ in expected_keys if name == suite}
        for task_id in sorted(task_ids):
            rows = [
                row
                for (name, task, _), row in ledger.items()
                if name == suite and task == task_id
            ]
            successes = sum(bool(row["success"]) for row in rows)
            per_task[str(task_id)] = {
                "completed": len(rows),
                "successes": successes,
                "success_rate": successes / len(rows) if rows else None,
            }
        suite_rows = [row for (name, _, _), row in ledger.items() if name == suite]
        suite_successes = sum(bool(row["success"]) for row in suite_rows)
        per_suite[suite] = {
            "completed": len(suite_rows),
            "successes": suite_successes,
            "success_rate": suite_successes / len(suite_rows) if suite_rows else None,
            "per_task": per_task,
        }
    rows = list(ledger.values())
    successes = sum(bool(row["success"]) for row in rows)
    document = {
        "schema": "apxinf.libero-eval.v2",
        "suites": sorted({key[0] for key in expected_keys}),
        "transport": transport,
        "precision": precision,
        "expected_runs": len(expected_keys),
        "completed_runs": len(rows),
        "missing_runs": [
            {"suite": suite, "task_id": task, "trial_id": trial}
            for suite, task, trial in sorted(expected_keys - set(ledger))
        ],
        "successes": successes,
        "success_rate": successes / len(rows) if rows else None,
        "per_suite": per_suite,
        "timing": _aggregate_timing(rows),
        "updated_unix_seconds": time.time(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
