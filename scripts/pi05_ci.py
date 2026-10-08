#!/usr/bin/env python3
"""Run a frozen PI05 bank or verify that both board reports cover its matrix."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

from compare_pi05_openpi import compare_results, load_suite, sha256, write_json


SCHEMA = "apxinf.pi05.ci.v1"
PRECISIONS = {"thor": ("bf16", "fp8"), "orin": ("bf16", "int8")}
LIMIT_KEYS = {"min_cosine", "max_relative_l2", "max_abs", "zero_max_abs"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def asset(root: Path, entry: dict) -> Path:
    path = (root / entry["path"]).resolve()
    if sha256(path) != entry["sha256"]:
        raise ValueError(f"asset digest differs: {entry['path']}")
    return path


def load_bank(path: Path, digest: str) -> dict:
    if sha256(path) != digest:
        raise ValueError("bank digest differs from approved digest")
    bank = read_json(path)
    if bank.get("schema") != SCHEMA or not bank.get("cells"):
        raise ValueError("invalid or empty bank")
    names = [cell["id"] for cell in bank["cells"]]
    if len(set(names)) != len(names):
        raise ValueError("duplicate cell")
    for board, precisions in PRECISIONS.items():
        for precision in precisions:
            for views in (1, 2, 3):
                if not any((cell["hardware"], cell["precision"], cell["views"]) ==
                           (board, precision, views) for cell in bank["cells"]):
                    raise ValueError(f"missing required cell: {board}/{precision}/{views}view")
    for cell in bank["cells"]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", cell["id"]):
            raise ValueError("cell id must be safe for artifact filenames")
        if cell["precision"] not in PRECISIONS[cell["hardware"]] or cell["views"] not in (1, 2, 3):
            raise ValueError("unsupported cell")
        for name in ("reference_limits", "baseline_limits"):
            limits = cell[name]
            if set(limits) != LIMIT_KEYS or not all(math.isfinite(v) for v in limits.values()):
                raise ValueError("limits must be explicit and finite")
            if not -1 <= limits["min_cosine"] <= 1 or any(limits[k] < 0 for k in LIMIT_KEYS - {"min_cosine"}):
                raise ValueError("invalid limits")
        if cell["samples"] < 1 or not isinstance(cell["samples"], int):
            raise ValueError("samples must be a positive integer")
        if not cell.get("budget_reason"):
            raise ValueError("record the evidence/approval for each cell's budgets")
    return bank


def freeze(args: argparse.Namespace) -> int:
    """Hash an explicitly reviewed definition; never refresh CI references implicitly."""
    definition = read_json(args.definition)
    cells = []
    for profile in definition["profiles"]:
        for policy in definition["policies"]:
            for views in profile["views"]:
                cell = {**profile, **policy, "profile": profile["id"], "views": views}
                cell["id"] = f"{profile['id']}-{policy['hardware']}-{policy['precision']}-{views}view"
                cells.append(cell)
    bank = {"schema": SCHEMA, "cells": cells}
    for cell in bank["cells"]:
        for key in ("suite", "checkpoint", "config", "reference", "baseline", "calibration"):
            if cell.get(key) is not None:
                path = (args.definition.parent / cell[key].format(**cell)).resolve()
                cell[key] = {"path": str(path), "sha256": sha256(path)}
    write_json(args.bank, bank)
    digest = sha256(args.bank)
    load_bank(args.bank, digest)
    print(f"Review {args.bank}; approved digest: {digest}")
    return 0


def performance(rows: list[dict], cell: dict) -> dict:
    import numpy as np

    samples = [sample for row in rows for sample in row["latency_ms"]]
    if not samples or not all(math.isfinite(sample) and sample > 0 for sample in samples):
        raise ValueError("invalid latency samples")
    if not all(math.isfinite(row["first_call_ms"]) and row["first_call_ms"] > 0 for row in rows):
        raise ValueError("invalid first-call latency")
    measured = {"p50_ms": float(np.percentile(samples, 50)),
                "p95_ms": float(np.percentile(samples, 95)),
                "first_call_ms": max(row["first_call_ms"] for row in rows)}
    budgets = cell.get("performance_limits")
    if budgets is None:
        return {"state": "uncalibrated", **measured}
    if set(budgets) != set(measured) or not all(math.isfinite(v) and v > 0 for v in budgets.values()):
        raise ValueError("invalid performance budgets")
    # A pooled distribution alone can hide a regression of a slow case.
    per_case = [{"name": row["name"], "p50_ms": float(np.percentile(row["latency_ms"], 50)),
                 "p95_ms": float(np.percentile(row["latency_ms"], 95)),
                 "first_call_ms": row["first_call_ms"]} for row in rows]
    passed = all(row[key] <= budgets[key] for row in per_case for key in budgets)
    return {"state": "pass" if passed else "fail", **measured, "cases": per_case}


def evaluate(cell: dict, suite: Path, reference: dict, baseline: dict,
             actual: dict, revision: str) -> dict:
    if len(load_suite(suite)["image_keys"]) != cell["views"]:
        raise ValueError("suite/cell view count differs")
    if any(len(row["latency_ms"]) != cell["samples"] for row in actual["cases"]):
        raise ValueError("latency sample count differs from approved protocol")
    if actual.get("revision") != revision:
        raise ValueError("candidate revision receipt differs")
    expected_calibration = cell["calibration"]["sha256"] if cell.get("calibration") else None
    for record in (actual, baseline):
        if record.get("engine") != "apxinf":
            raise ValueError("unexpected implementation engine")
        if (record.get("precision"), record.get("hardware"), record.get("calibration_sha256")) != (
                cell["precision"], cell["hardware"], expected_calibration):
            raise ValueError("baseline/candidate precision, hardware or calibration differs")
        if record.get("checkpoint_sha256") != cell["checkpoint"]["sha256"]:
            raise ValueError("checkpoint receipt mismatch")
    if reference.get("engine") != "openpi" or not reference.get("revision"):
        raise ValueError("official reference provenance missing")
    for record in (actual, reference, baseline):
        if record.get("config_sha256") != cell["config"]["sha256"]:
            raise ValueError("configuration receipt differs")
    if not re.fullmatch(r"[0-9a-f]{40}", baseline.get("revision", "")):
        raise ValueError("approved baseline revision missing")
    official = compare_results(suite, reference, actual, cell["reference_limits"])
    regression = compare_results(suite, baseline, actual, cell["baseline_limits"])
    perf = performance(actual["cases"], cell)
    repeat_ok = all(row["repeat_max_abs"] <= cell["baseline_limits"]["max_abs"]
                    for row in actual["cases"])
    repeat_ok &= actual["revisit_max_abs"] <= cell["baseline_limits"]["max_abs"]
    return dict(reference=official, baseline=regression, performance=perf,
          repeat_passed=repeat_ok,
          passed=all(row["passed"] for row in official + regression) and repeat_ok
          and perf["state"] == "pass")


def run(args: argparse.Namespace) -> int:
    bank = load_bank(args.bank, args.bank_sha256)
    root = args.bank.parent
    cells = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for cell in bank["cells"]:
        if cell["hardware"] != args.hardware:
            continue
        result = {"id": cell["id"], "passed": False}
        try:
            suite = asset(root, cell["suite"]).parent
            manifest = read_json(suite / "manifest.json")
            if len(manifest["image_keys"]) != cell["views"]:
                raise ValueError("manifest/cell views differ")
            checkpoint = asset(root, cell["checkpoint"]).parent
            asset(root, cell["config"])
            if sha256(checkpoint / "config.json") != cell["config"]["sha256"]:
                raise ValueError("checkpoint configuration differs")
            reference = read_json(asset(root, cell["reference"]))
            baseline = read_json(asset(root, cell["baseline"]))
            calibration = asset(root, cell["calibration"]) if cell.get("calibration") else None
            output = args.output_dir / f"{cell['id']}.json"
            command = [str(args.python), str(Path(__file__).with_name("compare_pi05_openpi.py")),
                       "apxinf", "--suite-dir", str(suite), "--checkpoint-dir", str(checkpoint),
                       "--precision", cell["precision"], "--hardware", args.hardware,
                       "--revision", args.revision, "--repeats", str(cell["samples"]),
                       "--output", str(output), "--force"]
            if calibration:
                command.extend(("--calibration", str(calibration)))
            subprocess.run(command, check=True)
            actual = read_json(output)
            result.update(evaluate(cell, suite, reference, baseline, actual, args.revision))
            evidence = args.output_dir / "evidence" / cell["id"]
            evidence.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(suite / "manifest.json", evidence / "manifest.json")
            for entry in load_suite(suite)["cases"]:
                destination = evidence / entry["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(suite / entry["path"], destination)
            for name in ("reference", "baseline"):
                shutil.copyfile(asset(root, cell[name]), evidence / f"{name}.json")
        except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
            result["error"] = str(error)
        cells.append(result)
        print(f"{cell['id']}: {'PASS' if result['passed'] else 'FAIL'}", flush=True)
    report = {"schema": SCHEMA, "bank_sha256": args.bank_sha256, "revision": args.revision,
              "hardware": args.hardware, "passed": bool(cells) and all(c["passed"] for c in cells),
              "cells": cells}
    write_json(args.output_dir / "summary.json", report, force=True)
    return 0 if report["passed"] else 1


def aggregate(args: argparse.Namespace) -> int:
    bank = load_bank(args.bank, args.bank_sha256)
    if len(args.report) != len(PRECISIONS):
        raise ValueError("both board reports are required")
    seen = set()
    passed = True
    for path in args.report:
        report = read_json(path)
        board = report["hardware"]
        if board not in PRECISIONS or board in seen:
            raise ValueError("duplicate or unsupported board report")
        seen.add(board)
        if (report["schema"], report["revision"], report["bank_sha256"]) != (
                SCHEMA, args.revision, args.bank_sha256):
            raise ValueError("report revision/bank/schema differs")
        expected = {cell["id"] for cell in bank["cells"] if cell["hardware"] == board}
        if len(report["cells"]) != len(expected) or {cell["id"] for cell in report["cells"]} != expected:
            raise ValueError("board report has missing or unexpected cells")
        passed &= report["passed"] is True and all(cell["passed"] is True for cell in report["cells"])
        for cell in bank["cells"]:
            if cell["hardware"] != board:
                continue
            evidence = path.parent / "evidence" / cell["id"]
            if sha256(evidence / "manifest.json") != cell["suite"]["sha256"]:
                raise ValueError("artifact suite digest differs")
            for name in ("reference", "baseline"):
                if sha256(evidence / f"{name}.json") != cell[name]["sha256"]:
                    raise ValueError("artifact golden output digest differs")
            result = evaluate(cell, evidence, read_json(evidence / "reference.json"),
                              read_json(evidence / "baseline.json"),
                              read_json(path.parent / f"{cell['id']}.json"), args.revision)
            passed &= result["passed"]
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "aggregate"))
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--bank-sha256")
    parser.add_argument("--revision")
    parser.add_argument("--definition", type=Path)
    parser.add_argument("--hardware", choices=tuple(PRECISIONS))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.command == "freeze":
        if args.definition is None:
            parser.error("freeze requires definition")
        return freeze(args)
    if args.bank_sha256 is None or args.revision is None:
        parser.error("run/aggregate require bank-sha256 and revision")
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("revision must be a full commit SHA")
    if args.command == "run" and (args.hardware is None or args.output_dir is None):
        parser.error("run requires hardware and output-dir")
    return run(args) if args.command == "run" else aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())
