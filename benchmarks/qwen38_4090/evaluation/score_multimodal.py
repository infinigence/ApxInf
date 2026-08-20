#!/usr/bin/env python3
"""Assign the frozen multimodal capability badge; no leaderboard points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "apxinf.qwen38_27b.multimodal_report.v1":
        raise ValueError(f"{path}: unsupported report schema")
    return report


def valid_pass(report: dict[str, Any], expected_split: str, expected_total: int) -> bool:
    return all(
        (
            report.get("split") == expected_split,
            report.get("capability_declared") is True,
            report.get("fallback_active") is not True,
            report.get("cases_total") == expected_total,
            report.get("cases_passed") == expected_total,
            report.get("request_success_rate") == 1.0,
            report.get("service_healthy_after_run") is True,
        )
    )


def score(public: dict[str, Any], hidden: dict[str, Any] | None) -> dict[str, Any]:
    evidence = public.get("evidence", {})
    if public.get("capability_declared") is False:
        badge = "declared-unsupported" if public.get("fail_closed") is True else "invalid-unsupported-path"
    elif valid_pass(public, "public", 4):
        badge = "multimodal-public-pass"
        if hidden is not None:
            if hidden.get("evidence", {}).get("contract_sha256") != evidence.get("contract_sha256"):
                raise ValueError("public and hidden reports use different contracts")
            if valid_pass(hidden, "hidden", 8):
                badge = "multimodal-ready"
    else:
        badge = "multimodal-not-passed"
    return {
        "schema": "apxinf.qwen38_27b.multimodal_badge.v1",
        "implementation": public.get("implementation"),
        "badge": badge,
        "leaderboard_points": 0.0,
        "public": {
            "cases_passed": public.get("cases_passed"),
            "cases_total": public.get("cases_total"),
            "request_success_rate": public.get("request_success_rate"),
            "fail_closed": public.get("fail_closed"),
        },
        "hidden": None
        if hidden is None
        else {
            "cases_passed": hidden.get("cases_passed"),
            "cases_total": hidden.get("cases_total"),
            "request_success_rate": hidden.get("request_success_rate"),
        },
        "contract_sha256": evidence.get("contract_sha256"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-report", type=Path, required=True)
    parser.add_argument("--hidden-report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = score(
        load_report(args.public_report),
        load_report(args.hidden_report) if args.hidden_report else None,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
