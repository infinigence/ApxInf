#!/usr/bin/env python3
"""Score the frozen multimodal bonus and assign capability badges."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from multimodal_scoring import report_summary, score_multimodal


def load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "apxinf.qwen38_27b.multimodal_report.v1":
        raise ValueError(f"{path}: unsupported report schema")
    return report


def score(
    public: dict[str, Any],
    hidden: dict[str, Any] | None,
    contract: dict[str, Any] | None = None,
    expected_multimodal_contract_sha256: str | None = None,
) -> dict[str, Any]:
    if contract is None:
        contract = json.loads(
            Path(__file__).with_name("contract-v1.json").read_text(encoding="utf-8")
        )
    public_summary = report_summary(public)
    hidden_summary = report_summary(hidden) if hidden is not None else None
    if (
        expected_multimodal_contract_sha256 is not None
        and public_summary["contract_sha256"] != expected_multimodal_contract_sha256
    ):
        raise ValueError("public report belongs to a different multimodal contract")
    if (
        expected_multimodal_contract_sha256 is not None
        and hidden_summary is not None
        and hidden_summary["contract_sha256"] != expected_multimodal_contract_sha256
    ):
        raise ValueError("hidden report belongs to a different multimodal contract")
    details = score_multimodal(
        contract["multimodal_bonus"],
        {"public": public_summary, "hidden": hidden_summary},
        require_report_hash=False,
    )
    return {
        "schema": "apxinf.qwen38_27b.multimodal_badge.v1",
        "implementation": public.get("implementation"),
        "badge": details["badge"],
        "leaderboard_points": details["points"],
        "public": details["public"],
        "hidden": details["hidden"],
        "reasons": details["reasons"],
        "contract_sha256": public_summary["contract_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-report", type=Path, required=True)
    parser.add_argument("--hidden-report", type=Path)
    parser.add_argument(
        "--leaderboard-contract",
        type=Path,
        default=Path(__file__).with_name("contract-v1.json"),
    )
    parser.add_argument(
        "--multimodal-contract",
        type=Path,
        default=Path(__file__).with_name("multimodal-contract-v1.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    leaderboard_contract = json.loads(
        args.leaderboard_contract.read_text(encoding="utf-8")
    )
    multimodal_contract = json.loads(args.multimodal_contract.read_text(encoding="utf-8"))
    leaderboard_hash = hashlib.sha256(args.leaderboard_contract.read_bytes()).hexdigest()
    if multimodal_contract.get("overlay_for", {}).get("contract_sha256") != leaderboard_hash:
        raise ValueError("multimodal contract does not overlay the selected leaderboard contract")
    result = score(
        load_report(args.public_report),
        load_report(args.hidden_report) if args.hidden_report else None,
        leaderboard_contract,
        hashlib.sha256(args.multimodal_contract.read_bytes()).hexdigest(),
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
