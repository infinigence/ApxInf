#!/usr/bin/env python3
"""Score a Qwen3.8-27B cohort using the frozen RTX 4090 course contract."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SUBMISSION_SCHEMA = "apxinf.qwen38_27b.leaderboard_submission.v1"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return value


def _rate(value: Any, field: str, *, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number in [0, 1]")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be a finite number in [0, 1]")
    return number


def _positive(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0.0 else None


def _nonnegative(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0.0 else None


def _count_rate(
    values: dict[str, Any],
    prefix: str,
    expected_total: int,
    *,
    allow_none: bool,
) -> float | None:
    passed = values.get(f"{prefix}_passed")
    total = values.get(f"{prefix}_total")
    if passed is None and total is None and allow_none:
        return None
    for name, value in (("passed", passed), ("total", total)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"correctness.{prefix}_{name} must be a non-negative integer")
    if total != expected_total:
        raise ValueError(
            f"correctness.{prefix}_total must equal frozen total {expected_total}, got {total}"
        )
    if passed > total:
        raise ValueError(f"correctness.{prefix}_passed cannot exceed total")
    return passed / total


def _validate_top_level(submission: dict[str, Any]) -> None:
    required = ("implementation", "correctness", "cells", "context", "reliability")
    if submission.get("schema") != SUBMISSION_SCHEMA:
        raise ValueError(f"schema must equal {SUBMISSION_SCHEMA!r}")
    missing = [field for field in required if not isinstance(submission.get(field), dict)]
    if missing:
        raise ValueError(f"missing object field(s): {', '.join(missing)}")
    implementation = submission["implementation"]
    for field in ("name", "revision", "backend"):
        if not isinstance(implementation.get(field), str) or not implementation[field].strip():
            raise ValueError(f"implementation.{field} must be a non-empty string")


def _implementation_key(submission: dict[str, Any]) -> tuple[str, str, str]:
    implementation = submission["implementation"]
    return (
        implementation["name"],
        implementation["revision"],
        implementation["backend"],
    )


def _latency_evidence(
    definition: dict[str, Any],
    cells: dict[str, Any],
    metric: str,
    measurement: dict[str, Any],
) -> dict[str, Any]:
    cell_id = definition["id"]
    observed_cell = cells.get(cell_id)
    reasons: list[str] = []
    observed = None
    cv = None
    if not isinstance(observed_cell, dict):
        reasons.append("missing_cell")
    else:
        if observed_cell.get("actual_prompt_tokens") != definition["prompt_tokens"]:
            reasons.append("prompt_token_count_mismatch")
        if observed_cell.get("completion_tokens") != definition["output_tokens"]:
            reasons.append("output_token_count_mismatch")
        success_rate = _rate(
            observed_cell.get("success_rate"),
            f"cells.{cell_id}.success_rate",
            allow_none=True,
        )
        if success_rate != 1.0:
            reasons.append("success_rate_not_one")
        measured_repeats = observed_cell.get("measured_repeats")
        if (
            not isinstance(measured_repeats, int)
            or isinstance(measured_repeats, bool)
            or measured_repeats
            < int(measurement["minimum_measured_repeats_per_latency_cell"])
        ):
            reasons.append("insufficient_measured_repeats")
        warmup_repeats = observed_cell.get("warmup_repeats")
        if (
            not isinstance(warmup_repeats, int)
            or isinstance(warmup_repeats, bool)
            or warmup_repeats < int(measurement["required_warmup_repeats"])
        ):
            reasons.append("insufficient_warmup_repeats")
        observed = _positive(observed_cell.get(metric))
        if observed is None:
            reasons.append(f"missing_or_invalid_{metric}")
        maximum_cv = measurement.get("maximum_latency_cv")
        if maximum_cv is not None:
            cv = _nonnegative(observed_cell.get(metric.replace("_s", "_cv")))
            if cv is None:
                reasons.append(f"missing_or_invalid_{metric.replace('_s', '_cv')}")
            elif cv > float(maximum_cv):
                reasons.append(f"{metric.replace('_s', '_cv')}_above_limit")
    return {
        "id": cell_id,
        "metric": metric,
        "eligible": not reasons,
        "observed": observed,
        "cv": cv,
        "weight": float(definition["weight"]),
        "reasons": reasons,
    }


def _correctness_and_reliability(
    contract: dict[str, Any],
    submission: dict[str, Any],
    profile_name: str,
) -> dict[str, Any]:
    profile = contract["score_profiles"][profile_name]
    workload = contract["correctness_workload"]
    correctness = submission["correctness"]
    protocol_pass = correctness.get("protocol_pass") is True
    public_rate = _count_rate(
        correctness,
        "public_cases",
        int(workload["public_functional_suite"]["case_count"]),
        allow_none=False,
    )
    hidden_rate = _count_rate(
        correctness,
        "hidden_cases",
        int(workload["hidden_functional_suite"]["case_count"]),
        allow_none=True,
    )
    public_trajectory_rate = _count_rate(
        correctness,
        "public_trajectory_tokens",
        int(workload["public_token_trajectory_suite"]["total_tokens"]),
        allow_none=False,
    )
    hidden_trajectory_rate = _count_rate(
        correctness,
        "hidden_trajectory_tokens",
        int(workload["hidden_token_trajectory_suite"]["total_tokens"]),
        allow_none=True,
    )

    weights = profile["correctness"]
    correctness_points = {
        "protocol": float(weights["protocol"]) * float(protocol_pass),
        "public_cases": float(weights["public_cases"]) * float(public_rate or 0.0),
        "hidden_cases": float(weights["hidden_cases"]) * float(hidden_rate or 0.0),
        "public_token_trajectory": float(weights["public_token_trajectory"])
        * float(public_trajectory_rate or 0.0),
        "hidden_token_trajectory": float(weights["hidden_token_trajectory"])
        * float(hidden_trajectory_rate or 0.0),
    }

    reliability = submission["reliability"]
    reliability_contract = contract["reliability_scoring"]
    request_success_rate = _rate(
        reliability.get("request_success_rate"), "reliability.request_success_rate"
    )
    boolean_checks: dict[str, bool] = {}
    for field in reliability_contract["boolean_checks"]:
        if not isinstance(reliability.get(field), bool):
            raise ValueError(f"reliability.{field} must be boolean")
        boolean_checks[field] = reliability[field]
    boolean_weight_total = float(reliability_contract["weight"]) - float(
        reliability_contract["request_success_rate_weight"]
    )
    boolean_weight = boolean_weight_total / len(boolean_checks)
    reliability_points = float(
        reliability_contract["request_success_rate_weight"]
    ) * float(request_success_rate) + boolean_weight * sum(boolean_checks.values())

    eligibility = profile["eligibility"]
    failures: list[str] = []
    if not protocol_pass:
        failures.append("protocol_failed")
    if public_rate is None or public_rate < float(eligibility["public_pass_rate_min"]):
        failures.append("public_pass_rate_below_threshold")
    hidden_min = eligibility["hidden_pass_rate_min"]
    if hidden_min is not None and (hidden_rate is None or hidden_rate < float(hidden_min)):
        failures.append("hidden_pass_rate_below_threshold_or_missing")
    if public_trajectory_rate is None or public_trajectory_rate < float(
        eligibility["public_token_trajectory_rate_min"]
    ):
        failures.append("public_token_trajectory_rate_below_threshold")
    hidden_trajectory_min = eligibility["hidden_token_trajectory_rate_min"]
    if hidden_trajectory_min is not None and (
        hidden_trajectory_rate is None
        or hidden_trajectory_rate < float(hidden_trajectory_min)
    ):
        failures.append("hidden_token_trajectory_rate_below_threshold_or_missing")
    if request_success_rate < float(eligibility["request_success_rate_min"]):
        failures.append("request_success_rate_below_threshold")
    failures.extend(
        f"reliability_check_failed:{field}"
        for field, passed in boolean_checks.items()
        if not passed
    )
    return {
        "eligible": not failures,
        "eligibility_failures": failures,
        "correctness_points": correctness_points,
        "correctness_rates": {
            "public_cases": public_rate,
            "hidden_cases": hidden_rate,
            "public_token_trajectory": public_trajectory_rate,
            "hidden_token_trajectory": hidden_trajectory_rate,
        },
        "protocol_pass": protocol_pass,
        "reliability_points": reliability_points,
        "request_success_rate": request_success_rate,
        "boolean_checks": boolean_checks,
    }


def _context_bonus(contract: dict[str, Any], submission: dict[str, Any]) -> dict[str, Any]:
    context = submission["context"]
    bonus = contract["context_bonus"]
    max_prompt = context.get("max_verified_prompt_tokens")
    output_tokens = context.get("verified_output_tokens")
    case_count = context.get("verified_cases_at_max_context")
    pass_rate = _rate(
        context.get("pass_rate_at_max_context"),
        "context.pass_rate_at_max_context",
        allow_none=True,
    )
    if not isinstance(max_prompt, int) or isinstance(max_prompt, bool) or max_prompt < 0:
        raise ValueError("context.max_verified_prompt_tokens must be a non-negative integer")
    if not isinstance(output_tokens, int) or isinstance(output_tokens, bool) or output_tokens < 0:
        raise ValueError("context.verified_output_tokens must be a non-negative integer")
    if not isinstance(case_count, int) or isinstance(case_count, bool) or case_count < 0:
        raise ValueError("context.verified_cases_at_max_context must be a non-negative integer")

    start = int(bonus["bonus_start_prompt_tokens"])
    maximum = int(bonus["full_bonus_prompt_tokens"])
    reasons: list[str] = []
    points = 0.0
    if max_prompt > start:
        if max_prompt > maximum:
            reasons.append("claimed_prompt_exceeds_native_scoring_limit")
        if output_tokens != int(bonus["required_output_tokens"]):
            reasons.append("output_token_count_mismatch")
        if case_count < int(bonus["required_cases_at_max_context"]):
            reasons.append("insufficient_cases_at_max_context")
        if pass_rate != float(bonus["required_pass_rate_at_max_context"]):
            reasons.append("max_context_pass_rate_not_one")
        if context.get("service_healthy_after_failure") is not True:
            reasons.append("service_not_healthy_after_failure")
        first_failed = context.get("first_failed_prompt_tokens")
        if max_prompt < maximum and (
            not isinstance(first_failed, int)
            or isinstance(first_failed, bool)
            or first_failed <= max_prompt
        ):
            reasons.append("missing_or_invalid_first_failed_prompt_tokens")
        if not reasons:
            progress = math.log2(max_prompt / start) / math.log2(maximum / start)
            points = float(bonus["weight"]) * min(1.0, max(0.0, progress))
    return {
        "max_verified_prompt_tokens": max_prompt,
        "verified_output_tokens": output_tokens,
        "verified_cases_at_max_context": case_count,
        "pass_rate_at_max_context": pass_rate,
        "points": points,
        "reasons": reasons,
    }


def _multi_evidence(
    contract: dict[str, Any],
    submission: dict[str, Any],
    definition: dict[str, Any],
    measurement: dict[str, Any],
) -> dict[str, Any]:
    multi = submission.get("multi_request")
    observed_cell = None
    if isinstance(multi, dict) and isinstance(multi.get("cells"), dict):
        observed_cell = multi["cells"].get(definition["id"])
    reasons: list[str] = []
    goodput = None
    if not isinstance(observed_cell, dict):
        reasons.append("missing_cell")
    else:
        exact_fields = {
            "concurrency": definition["concurrency"],
            "total_requests": definition["total_requests"],
            "actual_prompt_tokens": definition["prompt_tokens"],
            "completion_tokens": definition["output_tokens"],
        }
        for field, expected in exact_fields.items():
            if observed_cell.get(field) != expected:
                reasons.append(f"{field}_mismatch")
        for field, expected in (("success_rate", 1.0), ("correctness_rate", 1.0)):
            value = _rate(
                observed_cell.get(field),
                f"multi_request.cells.{definition['id']}.{field}",
                allow_none=True,
            )
            if value != expected:
                reasons.append(f"{field}_not_one")
        repeats = observed_cell.get("measured_repeats")
        if (
            not isinstance(repeats, int)
            or isinstance(repeats, bool)
            or repeats < int(measurement["minimum_measured_repeats_per_latency_cell"])
        ):
            reasons.append("insufficient_measured_repeats")
        warmups = observed_cell.get("warmup_repeats")
        if (
            not isinstance(warmups, int)
            or isinstance(warmups, bool)
            or warmups < int(measurement["required_warmup_repeats"])
        ):
            reasons.append("insufficient_warmup_repeats")
        maximum_cv = measurement.get("maximum_latency_cv")
        if maximum_cv is not None:
            goodput_cv = _nonnegative(observed_cell.get("goodput_cv"))
            if goodput_cv is None:
                reasons.append("missing_or_invalid_goodput_cv")
            elif goodput_cv > float(maximum_cv):
                reasons.append("goodput_cv_above_limit")
        goodput = _positive(observed_cell.get("goodput_tokens_per_s"))
        if goodput is None:
            reasons.append("missing_or_invalid_goodput_tokens_per_s")
        fairness = _rate(
            observed_cell.get("jain_fairness_index"),
            f"multi_request.cells.{definition['id']}.jain_fairness_index",
            allow_none=True,
        )
        validity = contract["multi_request_bonus"]["validity"]
        if fairness is None or fairness < float(validity["minimum_jain_fairness_index"]):
            reasons.append("jain_fairness_below_limit")
        if observed_cell.get("no_fallback") is not True:
            reasons.append("fallback_or_missing_no_fallback_proof")
        if observed_cell.get("service_healthy_after_run") is not True:
            reasons.append("service_not_healthy_after_run")

        base_cell = submission["cells"].get(definition["base_cell_id"])
        single_ttft = _positive(base_cell.get("ttft_s")) if isinstance(base_cell, dict) else None
        single_tpot = _positive(base_cell.get("tpot_s")) if isinstance(base_cell, dict) else None
        p95_ttft = _positive(observed_cell.get("p95_ttft_s"))
        p95_tpot = _positive(observed_cell.get("p95_tpot_s"))
        if single_ttft is None or p95_ttft is None:
            reasons.append("missing_single_or_p95_ttft")
        elif p95_ttft > (
            float(definition["concurrency"])
            * float(validity["maximum_p95_ttft_multiple_of_own_single_request"])
            * single_ttft
        ):
            reasons.append("p95_ttft_tail_guard_failed")
        if single_tpot is None or p95_tpot is None:
            reasons.append("missing_single_or_p95_tpot")
        elif p95_tpot > (
            float(validity["maximum_p95_tpot_multiple_of_own_single_request"])
            * single_tpot
        ):
            reasons.append("p95_tpot_tail_guard_failed")
    return {
        "id": definition["id"],
        "eligible": not reasons,
        "goodput_tokens_per_s": goodput,
        "support_weight": float(definition["support_weight"]),
        "goodput_weight": float(definition["goodput_weight"]),
        "reasons": reasons,
    }


def score_cohort(
    contract: dict[str, Any],
    submissions: list[dict[str, Any]],
    profile_name: str,
    *,
    controls: list[dict[str, Any]] | None = None,
    require_official_control: bool = False,
) -> dict[str, Any]:
    profiles = contract.get("score_profiles", {})
    if profile_name not in profiles:
        raise ValueError(f"unknown score profile {profile_name!r}; choose from {sorted(profiles)}")
    if not submissions:
        raise ValueError("at least one submission is required")
    controls = controls or []
    all_entries = [(False, item) for item in submissions] + [(True, item) for item in controls]
    seen: set[tuple[str, str, str]] = set()
    prepared: list[dict[str, Any]] = []
    for is_control, submission in all_entries:
        _validate_top_level(submission)
        key = _implementation_key(submission)
        if key in seen:
            raise ValueError(f"duplicate implementation identity in cohort: {key}")
        seen.add(key)
        gate = _correctness_and_reliability(contract, submission, profile_name)
        prepared.append(
            {"is_control": is_control, "submission": submission, "key": key, "gate": gate}
        )

    required_backend = contract["performance_scoring"]["required_official_control_backend"]
    official_control_present = any(
        item["is_control"]
        and item["submission"]["implementation"]["backend"] == required_backend
        and item["gate"]["eligible"]
        for item in prepared
    )
    if require_official_control and not official_control_present:
        raise ValueError(
            f"official cohort requires an eligible --control-submission with backend {required_backend!r}"
        )

    profile = contract["score_profiles"][profile_name]
    performance = contract["performance_scoring"]
    references: dict[str, dict[str, float | None]] = {"ttft_s": {}, "tpot_s": {}}
    evidence: dict[tuple[str, str, str], dict[str, dict[str, dict[str, Any]]]] = {}
    for item in prepared:
        by_metric: dict[str, dict[str, dict[str, Any]]] = {"ttft_s": {}, "tpot_s": {}}
        for metric, definitions in (
            ("ttft_s", performance["ttft_cells"]),
            ("tpot_s", performance["tpot_cells"]),
        ):
            for definition in definitions:
                by_metric[metric][definition["id"]] = _latency_evidence(
                    definition,
                    item["submission"]["cells"],
                    metric,
                    profile["measurement"],
                )
        evidence[item["key"]] = by_metric

    for metric, definitions in (
        ("ttft_s", performance["ttft_cells"]),
        ("tpot_s", performance["tpot_cells"]),
    ):
        for definition in definitions:
            candidates = [
                evidence[item["key"]][metric][definition["id"]]["observed"]
                for item in prepared
                if item["gate"]["eligible"]
                and evidence[item["key"]][metric][definition["id"]]["eligible"]
            ]
            references[metric][definition["id"]] = min(candidates) if candidates else None

    multi_definitions = contract["multi_request_bonus"]["cells"]
    multi_evidence: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    multi_references: dict[str, float | None] = {}
    for item in prepared:
        multi_evidence[item["key"]] = {
            definition["id"]: _multi_evidence(
                contract, item["submission"], definition, profile["measurement"]
            )
            for definition in multi_definitions
        }
    for definition in multi_definitions:
        candidates = [
            multi_evidence[item["key"]][definition["id"]]["goodput_tokens_per_s"]
            for item in prepared
            if item["gate"]["eligible"]
            and multi_evidence[item["key"]][definition["id"]]["eligible"]
        ]
        multi_references[definition["id"]] = max(candidates) if candidates else None

    scores: list[dict[str, Any]] = []
    control_scores: list[dict[str, Any]] = []
    for item in prepared:
        submission = item["submission"]
        gate = item["gate"]
        latency_details: dict[str, list[dict[str, Any]]] = {"ttft": [], "tpot": []}
        latency_points = {"ttft": 0.0, "tpot": 0.0}
        for label, metric, definitions in (
            ("ttft", "ttft_s", performance["ttft_cells"]),
            ("tpot", "tpot_s", performance["tpot_cells"]),
        ):
            for definition in definitions:
                detail = dict(evidence[item["key"]][metric][definition["id"]])
                reference = references[metric][definition["id"]]
                points = 0.0
                if detail["eligible"] and reference is not None and detail["observed"] is not None:
                    points = float(definition["weight"]) * min(
                        1.0, float(reference) / float(detail["observed"])
                    )
                elif reference is None:
                    detail["reasons"] = [*detail["reasons"], "no_valid_cohort_reference"]
                detail["best_valid_median_seconds"] = reference
                detail["points"] = points
                latency_points[label] += points
                latency_details[label].append(detail)

        context = _context_bonus(contract, submission)
        multi_details: list[dict[str, Any]] = []
        multi_points = 0.0
        for definition in multi_definitions:
            detail = dict(multi_evidence[item["key"]][definition["id"]])
            reference = multi_references[definition["id"]]
            points = 0.0
            if detail["eligible"] and reference is not None and detail["goodput_tokens_per_s"]:
                points = float(definition["support_weight"]) + float(
                    definition["goodput_weight"]
                ) * min(1.0, float(detail["goodput_tokens_per_s"]) / float(reference))
            elif reference is None:
                detail["reasons"] = [*detail["reasons"], "no_valid_cohort_reference"]
            detail["best_valid_goodput_tokens_per_s"] = reference
            detail["points"] = points
            multi_points += points
            multi_details.append(detail)

        section_scores = {
            "correctness": sum(gate["correctness_points"].values()),
            "ttft": latency_points["ttft"],
            "tpot": latency_points["tpot"],
            "reliability": gate["reliability_points"],
            "context_bonus": context["points"],
            "multi_request_bonus": multi_points,
        }
        base_score = sum(
            section_scores[name] for name in ("correctness", "ttft", "tpot", "reliability")
        )
        bonus_score = section_scores["context_bonus"] + section_scores["multi_request_bonus"]
        diagnostic_score = base_score + bonus_score
        eligible = bool(gate["eligible"])
        rendered_score = {
            "schema": "apxinf.qwen38_27b.leaderboard_score.v1",
            "implementation": submission["implementation"],
            "is_control": item["is_control"],
            "profile": profile_name,
            "eligible": eligible,
            "base_score": base_score if eligible else None,
            "bonus_score": bonus_score if eligible else None,
            "leaderboard_score": diagnostic_score if eligible else None,
            "automated_course_points": min(80.0, 0.8 * diagnostic_score)
            if eligible
            else None,
            "diagnostic_score": diagnostic_score,
            "section_scores": section_scores,
            "correctness_details": {
                "points": gate["correctness_points"],
                "rates": gate["correctness_rates"],
            },
            "latency_details": latency_details,
            "context_bonus_details": context,
            "multi_request_bonus_details": multi_details,
            "reliability_details": {
                "request_success_rate": gate["request_success_rate"],
                "boolean_checks": gate["boolean_checks"],
                "points": gate["reliability_points"],
            },
            "eligibility_failures": gate["eligibility_failures"],
        }
        if item["is_control"]:
            control_scores.append(rendered_score)
        else:
            scores.append(rendered_score)

    provisional_reasons: list[str] = []
    if profile_name == "public_calibration":
        provisional_reasons.append("public_profile_reweights_visible_correctness")
    if not official_control_present:
        provisional_reasons.append("eligible_vllm_control_missing")
    if profile_name == "public_calibration" and len(submissions) == 1:
        provisional_reasons.append("single_submission_cohort")
    return {
        "schema": "apxinf.qwen38_27b.leaderboard_cohort_score.v1",
        "profile": profile_name,
        "candidate_count": len(submissions),
        "control_count": len(controls),
        "official_control_present": official_control_present,
        "provisional": bool(provisional_reasons),
        "provisional_reasons": provisional_reasons,
        "performance_references": references,
        "multi_request_references": multi_references,
        "scores": scores,
        "control_scores": control_scores,
    }


def score_submission(
    contract: dict[str, Any],
    submission: dict[str, Any],
    profile_name: str,
    *,
    cohort_submissions: list[dict[str, Any]] | None = None,
    controls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compatibility helper; official scoring should call score_cohort."""
    cohort = cohort_submissions or [submission]
    result = score_cohort(contract, cohort, profile_name, controls=controls)
    key = _implementation_key(submission)
    return next(score for score in result["scores"] if _implementation_key(score) == key)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--contract", type=Path, default=here / "contract-v1.json")
    parser.add_argument(
        "--submission",
        type=Path,
        action="append",
        required=True,
        help="Candidate submission; repeat for every PR in the cohort.",
    )
    parser.add_argument(
        "--control-submission",
        type=Path,
        action="append",
        default=[],
        help="Teacher-run control such as vLLM; repeat when needed.",
    )
    parser.add_argument(
        "--profile",
        choices=("public_calibration", "midterm_leaderboard"),
        default="public_calibration",
    )
    parser.add_argument(
        "--require-official-control",
        action="store_true",
        help="Fail unless an eligible vLLM control is present.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = score_cohort(
        _load_json(args.contract),
        [_load_json(path) for path in args.submission],
        args.profile,
        controls=[_load_json(path) for path in args.control_submission],
        require_official_control=(
            args.require_official_control or args.profile == "midterm_leaderboard"
        ),
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
