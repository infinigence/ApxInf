#!/usr/bin/env python3
"""Shared validation and scoring primitives for the multimodal bonus."""

from __future__ import annotations

import math
from typing import Any


REPORT_SCHEMA = "apxinf.qwen38_27b.multimodal_report.v1"


def _rate(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number in [0, 1]")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be a finite number in [0, 1]")
    return number


def _sha256(value: Any, field: str, *, required: bool) -> str | None:
    if value is None and not required:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def report_summary(
    report: dict[str, Any],
    *,
    report_sha256: str | None = None,
) -> dict[str, Any]:
    """Reduce a platform-generated report to the fields trusted by the scorer."""
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError("unsupported multimodal report schema")
    evidence = report.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("multimodal report has no evidence object")
    return {
        "split": report.get("split"),
        "capability_declared": report.get("capability_declared"),
        "fallback_active": report.get("fallback_active"),
        "fail_closed": report.get("fail_closed"),
        "cases_passed": report.get("cases_passed"),
        "cases_total": report.get("cases_total"),
        "request_success_rate": report.get("request_success_rate"),
        "service_healthy_after_run": report.get("service_healthy_after_run"),
        "contract_sha256": evidence.get("contract_sha256"),
        "manifest_sha256": evidence.get("manifest_sha256"),
        "report_sha256": report_sha256,
    }


def _score_split(
    summary: dict[str, Any],
    *,
    split: str,
    expected_total: int,
    correctness_weight: float,
    integration_weight: float,
    require_report_hash: bool,
) -> dict[str, Any]:
    prefix = f"multimodal.{split}"
    if summary.get("split") != split:
        raise ValueError(f"{prefix}.split must equal {split!r}")
    passed = summary.get("cases_passed")
    total = summary.get("cases_total")
    for name, value in (("cases_passed", passed), ("cases_total", total)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{prefix}.{name} must be a non-negative integer")
    if total != expected_total:
        raise ValueError(f"{prefix}.cases_total must equal frozen total {expected_total}")
    if passed > total:
        raise ValueError(f"{prefix}.cases_passed cannot exceed cases_total")
    success_rate = _rate(summary.get("request_success_rate"), f"{prefix}.request_success_rate")
    _sha256(summary.get("contract_sha256"), f"{prefix}.contract_sha256", required=True)
    _sha256(summary.get("manifest_sha256"), f"{prefix}.manifest_sha256", required=True)
    _sha256(
        summary.get("report_sha256"),
        f"{prefix}.report_sha256",
        required=require_report_hash,
    )

    declared = summary.get("capability_declared")
    fallback_active = summary.get("fallback_active")
    service_healthy = summary.get("service_healthy_after_run")
    valid_capability_path = declared is True and fallback_active is False
    correctness_points = correctness_weight * passed / expected_total if valid_capability_path else 0.0
    integration_pass = all(
        (
            valid_capability_path,
            passed == expected_total,
            success_rate == 1.0,
            service_healthy is True,
        )
    )
    reasons: list[str] = []
    if declared is not True:
        reasons.append("capability_not_declared_true")
    if fallback_active is not False:
        reasons.append("fallback_active_or_unproven")
    if passed != expected_total:
        reasons.append("not_all_cases_passed")
    if success_rate != 1.0:
        reasons.append("request_success_rate_not_one")
    if service_healthy is not True:
        reasons.append("service_not_healthy_after_run")
    return {
        "split": split,
        "cases_passed": passed,
        "cases_total": total,
        "case_pass_rate": passed / expected_total,
        "request_success_rate": success_rate,
        "correctness_points": correctness_points,
        "integration_points": integration_weight if integration_pass else 0.0,
        "points": correctness_points + (integration_weight if integration_pass else 0.0),
        "integration_pass": integration_pass,
        "reasons": reasons,
    }


def score_multimodal(
    config: dict[str, Any],
    multimodal: dict[str, Any] | None,
    *,
    require_report_hash: bool,
) -> dict[str, Any]:
    """Score the public and optional hidden image suites from one frozen config."""
    if multimodal is None:
        return {
            "badge": "not-submitted",
            "public": None,
            "hidden": None,
            "points": 0.0,
            "reasons": ["multimodal_evidence_missing"],
        }
    if not isinstance(multimodal, dict) or not isinstance(multimodal.get("public"), dict):
        raise ValueError("multimodal.public must be an object when multimodal evidence is supplied")

    public_summary = multimodal["public"]
    hidden_summary = multimodal.get("hidden")
    if hidden_summary is not None and not isinstance(hidden_summary, dict):
        raise ValueError("multimodal.hidden must be an object or null")
    public = _score_split(
        public_summary,
        split="public",
        expected_total=int(config["public_case_count"]),
        correctness_weight=float(config["public_correctness_weight"]),
        integration_weight=float(config["public_integration_weight"]),
        require_report_hash=require_report_hash,
    )
    hidden = None
    if hidden_summary is not None:
        if hidden_summary.get("contract_sha256") != public_summary.get("contract_sha256"):
            raise ValueError("public and hidden multimodal evidence use different contracts")
        for field in ("capability_declared", "fallback_active"):
            if hidden_summary.get(field) != public_summary.get(field):
                raise ValueError(
                    f"public and hidden multimodal evidence disagree on {field}"
                )
        hidden = _score_split(
            hidden_summary,
            split="hidden",
            expected_total=int(config["hidden_case_count"]),
            correctness_weight=float(config["hidden_correctness_weight"]),
            integration_weight=float(config["hidden_integration_weight"]),
            require_report_hash=require_report_hash,
        )

    if public_summary.get("capability_declared") is False:
        badge = (
            "declared-unsupported"
            if public_summary.get("fail_closed") is True
            else "invalid-unsupported-path"
        )
    elif public["integration_pass"]:
        badge = "multimodal-public-pass"
        if hidden is not None and hidden["integration_pass"]:
            badge = "multimodal-ready"
    else:
        badge = "multimodal-not-passed"
    points = public["points"] + (hidden["points"] if hidden is not None else 0.0)
    maximum = float(config["weight"])
    if points > maximum + 1e-9:
        raise ValueError("multimodal scoring weights exceed the frozen bonus maximum")
    reasons = [f"public:{reason}" for reason in public["reasons"]]
    if hidden is None:
        reasons.append("hidden:evidence_missing")
    else:
        reasons.extend(f"hidden:{reason}" for reason in hidden["reasons"])
    return {
        "badge": badge,
        "public": public,
        "hidden": hidden,
        "points": points,
        "reasons": reasons,
    }
