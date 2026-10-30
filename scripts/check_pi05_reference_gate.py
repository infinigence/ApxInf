#!/usr/bin/env python3
"""Validate the PI0.5 cosine policy, ratchet it against base, or check a GPU report."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs/pi05/reference-gate.json"
MANIFEST = ROOT / "crates/apxinf-model/tests/fixtures/pi05_gpu/manifest.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate(policy, manifest_bytes):
    require(policy["schema"] == 1, "unsupported policy schema")
    require(policy["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest(),
            "corpus digest changed without updating policy")
    names = [c["name"] for c in json.loads(manifest_bytes)["cases"]]
    require(names and len(set(names)) == len(names), "empty or duplicate cases")
    for key in ("thor/bf16", "thor/fp8_static", "orin/bf16"):
        require(policy["lanes"][key]["required"] is True, f"missing required lane: {key}")
    require(policy["lanes"]["orin/int8_dynamic"]["required"] is False,
            "Orin INT8 must remain advisory")
    for key, lane in policy["lanes"].items():
        require(type(lane["required"]) is bool, f"invalid required flag: {key}")
        require(set(lane["cases"]) == set(names), f"incomplete corpus: {key}")
        require(finite(lane["eager_graph_max_abs"]) and lane["eager_graph_max_abs"] >= 0,
                f"invalid graph tolerance: {key}")
        for name, case in lane["cases"].items():
            limit = case["min_cosine"]
            require(finite(limit) and -1 <= limit <= 1, f"invalid cosine: {key}/{name}")


def ratchet(policy, base):
    # Freezing corpus identity prevents substituting easier inputs under old case names.
    require(policy["manifest_sha256"] == base["manifest_sha256"],
            "existing corpus identity is immutable; add a separate corpus to expand coverage")
    for key, previous in base["lanes"].items():
        require(key in policy["lanes"], f"removed lane: {key}")
        lane = policy["lanes"][key]
        require(not previous["required"] or lane["required"], f"demoted lane: {key}")
        require(lane["eager_graph_max_abs"] <= previous["eager_graph_max_abs"],
                f"relaxed graph consistency: {key}")
        for name, old in previous["cases"].items():
            require(name in lane["cases"], f"removed case: {key}/{name}")
            require(lane["cases"][name]["min_cosine"] >= old["min_cosine"],
                    f"lowered cosine threshold: {key}/{name}")


def check_report(policy, key, report):
    lane = policy["lanes"][key]
    require(report["manifest_sha256"] == policy["manifest_sha256"], "report corpus mismatch")
    require(report["variant"] == key.split("/")[1], "report precision mismatch")
    require(report["mode"] in ("measurement_only", "check"), "not a GPU report")
    rows = report["cases"]
    require(len(rows) == len(lane["cases"]) and {r["name"] for r in rows} == set(lane["cases"]),
            "missing, extra or duplicate report cases")
    failures = []
    for row in rows:
        name = row["name"]
        for mode in ("eager_vs_cpu", "graph_vs_cpu", "eager_vs_graph"):
            require(all(finite(row[mode][m]) for m in ("cosine", "relative_l2", "max_abs")),
                    f"non-finite metrics: {name}/{mode}")
        floor = lane["cases"][name]["min_cosine"]
        for mode in ("eager_vs_cpu", "graph_vs_cpu"):
            if row[mode]["cosine"] < floor:
                failures.append(f"{name}/{mode}: cosine {row[mode]['cosine']} < {floor}")
        if row["repeat_bitwise_equal"] is not True:
            failures.append(f"{name}: unstable graph replay")
        if row["eager_vs_graph"]["max_abs"] > lane["eager_graph_max_abs"]:
            failures.append(f"{name}: eager/graph mismatch")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, default=POLICY)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--lane")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text())
    validate(policy, args.manifest.read_bytes())
    if args.base:
        ratchet(policy, json.loads(args.base.read_text()))
    require(bool(args.lane) == bool(args.report), "--lane and --report must be paired")
    if args.report:
        failures = check_report(policy, args.lane, json.loads(args.report.read_text()))
        for failure in failures:
            print(failure)
        required = policy["lanes"][args.lane]["required"]
        print(json.dumps({"lane": args.lane, "required": required, "passed": not failures}))
        if required and failures:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
