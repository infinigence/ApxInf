#!/usr/bin/env python3
"""Single entry point for the Qwen3.8 RTX 4090 assignment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import py_compile
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
PUBLIC_DATA = HERE / ".cache" / "public"
RUNS = HERE / "runs"
MODULES = (
    "fetch_public_corpus.py",
    "generate_evaluation_cases.py",
    "multimodal_scoring.py",
    "run_evaluation.py",
    "score_multimodal.py",
    "score_submission.py",
    "test.py",
)
PUBLIC_FILES = {
    ".gitignore",
    "contract-v1.json",
    "fetch_public_corpus.py",
    "generate_evaluation_cases.py",
    "multimodal-contract-v1.json",
    "multimodal_scoring.py",
    "run_evaluation.py",
    "score_multimodal.py",
    "score_submission.py",
    "submission-schema-v1.json",
    "test.py",
}


def command(*argv: str, cwd: Path | None = None) -> None:
    print("+", " ".join(argv), flush=True)
    subprocess.run(argv, cwd=cwd, check=True)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_runner():
    return load_module("assignment_runner", HERE / "run_evaluation.py")


class FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        del token_ids, skip_special_tokens
        return "42"


class FakeSampler:
    def window(self, start, end):
        del start, end
        return {"sample_count": 0}


class ProtocolHandler(BaseHTTPRequestHandler):
    mode = "happy"

    def log_message(self, format, *args):
        del format, args

    def do_GET(self):
        if self.path != "/health":
            self.send_error(404)
            return
        payload = json.dumps({"status": "ok", "fallback_active": False}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        if self.path != "/v1/evaluations/generate":
            self.send_error(404)
            return
        size = int(self.headers.get("Content-Length", "0"))
        json.loads(self.rfile.read(size))
        second_index = 2 if self.mode == "index_gap" else 1
        second_request_id = "req-b" if self.mode == "crossed_request" else "req-a"
        events = [
            {"type": "token", "request_id": "req-a", "index": 0, "token_id": 7},
            {"type": "token", "request_id": second_request_id, "index": second_index, "token_id": 8},
            {
                "type": "done",
                "request_id": "req-a",
                "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
            },
        ]
        body = "".join(
            f"data: {json.dumps(event, separators=(',', ':'))}\n\n" for event in events
        ) + "data: [DONE]\n\n"
        payload = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def protocol_checks() -> None:
    runner = load_runner()
    server = ThreadingHTTPServer(("127.0.0.1", 0), ProtocolHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        case = {
            "id": "fake-case",
            "suite": "functional",
            "category": "exact_retrieval",
            "roles": [],
            "input_ids": list(range(8)),
            "input_ids_sha256": "0" * 64,
            "max_new_tokens": 2,
            "temperature": 0.0,
            "ignore_eos": True,
            "validation": "normalized_exact",
            "expected": "42",
        }

        def request(mode: str):
            ProtocolHandler.mode = mode
            return runner.request_evaluation_api(
                case, base_url, 5.0, FakeTokenizer(), FakeSampler()
            )

        happy = request("happy")
        assert happy["success"] and happy["functional_pass"]
        assert happy["output_ids"] == [7, 8]
        assert happy["usage"]["total_tokens"] == 10

        gap = request("index_gap")
        assert not gap["success"] and "token index gap" in gap["error"]

        crossed = request("crossed_request")
        assert not crossed["success"] and "request_id changed" in crossed["error"]

        assert runner.token_edit_distance([1, 2, 3], [1, 9, 3]) == 1
        assert runner.token_edit_distance([1, 2, 3], [1, 2]) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def expect_value_error(action, message: str) -> None:
    try:
        action()
    except ValueError as error:
        assert message in str(error), (message, str(error))
    else:
        raise AssertionError(f"expected ValueError containing {message!r}")


def multimodal_report(
    split: str,
    total: int,
    implementation: dict[str, str],
    contract_sha256: str,
    *,
    passed: int | None = None,
    declared: bool = True,
    fallback: bool = False,
    healthy: bool = True,
    fail_closed: bool | None = None,
) -> dict:
    passed = total if passed is None else passed
    return {
        "schema": "apxinf.qwen38_27b.multimodal_report.v1",
        "split": split,
        "implementation": implementation,
        "capability_declared": declared,
        "fallback_active": fallback,
        "fail_closed": fail_closed,
        "cases_passed": passed,
        "cases_total": total,
        "request_success_rate": 1.0 if declared and not fallback else 0.0,
        "service_healthy_after_run": healthy,
        "evidence": {
            "contract_sha256": contract_sha256,
            "manifest_sha256": "2" * 64,
        },
    }


def multimodal_summary(split: str, total: int, passed: int) -> dict:
    return {
        "split": split,
        "capability_declared": True,
        "fallback_active": False,
        "fail_closed": None,
        "cases_passed": passed,
        "cases_total": total,
        "request_success_rate": 1.0,
        "service_healthy_after_run": True,
        "contract_sha256": "1" * 64,
        "manifest_sha256": "2" * 64,
        "report_sha256": "3" * 64,
    }


def full_submission(contract: dict, multimodal: dict | None) -> dict:
    cells = {}
    for definition in contract["performance_scoring"]["ttft_cells"]:
        cells[definition["id"]] = {
            "actual_prompt_tokens": definition["prompt_tokens"],
            "completion_tokens": definition["output_tokens"],
            "success_rate": 1.0,
            "measured_repeats": 1,
            "warmup_repeats": 0,
            "ttft_cv": 0.0,
            "tpot_cv": 0.0,
            "ttft_s": 1.0,
            "tpot_s": 0.1,
            "e2e_s": 13.7,
            "peak_vram_mib": 20000.0,
        }
    multi_cells = {}
    for definition in contract["multi_request_bonus"]["cells"]:
        multi_cells[definition["id"]] = {
            "concurrency": definition["concurrency"],
            "total_requests": definition["total_requests"],
            "actual_prompt_tokens": definition["prompt_tokens"],
            "completion_tokens": definition["output_tokens"],
            "success_rate": 1.0,
            "correctness_rate": 1.0,
            "measured_repeats": 1,
            "warmup_repeats": 0,
            "goodput_tokens_per_s": 100.0,
            "p95_ttft_s": 1.0,
            "p95_tpot_s": 0.1,
            "jain_fairness_index": 1.0,
            "no_fallback": True,
            "service_healthy_after_run": True,
        }
    submission = {
        "schema": "apxinf.qwen38_27b.leaderboard_submission.v1",
        "implementation": {"name": "candidate", "revision": "a" * 40, "backend": "apxinf"},
        "correctness": {
            "protocol_pass": True,
            "public_cases_passed": 6,
            "public_cases_total": 6,
            "hidden_cases_passed": None,
            "hidden_cases_total": None,
            "public_trajectory_tokens_passed": 256,
            "public_trajectory_tokens_total": 256,
            "hidden_trajectory_tokens_passed": None,
            "hidden_trajectory_tokens_total": None,
        },
        "cells": cells,
        "context": {
            "max_verified_prompt_tokens": 262016,
            "verified_output_tokens": 128,
            "verified_cases_at_max_context": 6,
            "pass_rate_at_max_context": 1.0,
            "first_failed_prompt_tokens": None,
            "failure_mode": None,
            "service_healthy_after_failure": True,
        },
        "multi_request": {"cells": multi_cells},
        "reliability": {
            "request_success_rate": 1.0,
            "no_unexpected_oom": True,
            "no_nan": True,
            "no_fallback": True,
            "no_xid": True,
            "service_healthy_after_failure": True,
        },
    }
    if multimodal is not None:
        submission["multimodal"] = multimodal
    return submission


def multimodal_checks(contract: dict) -> None:
    shared = load_module("multimodal_scoring", HERE / "multimodal_scoring.py")
    standalone = load_module("assignment_multimodal_scorer", HERE / "score_multimodal.py")
    leaderboard = load_module("assignment_leaderboard_scorer", HERE / "score_submission.py")
    runner = load_runner()
    config = contract["multimodal_bonus"]

    partial_public = {
        "public": multimodal_summary("public", 4, 2),
        "hidden": None,
    }
    assert shared.score_multimodal(config, partial_public, require_report_hash=True)["points"] == 1.0
    partial_private = {
        "public": multimodal_summary("public", 4, 4),
        "hidden": multimodal_summary("hidden", 8, 4),
    }
    assert shared.score_multimodal(config, partial_private, require_report_hash=True)["points"] == 6.0

    full = {
        "public": multimodal_summary("public", 4, 4),
        "hidden": multimodal_summary("hidden", 8, 8),
    }
    full_score = shared.score_multimodal(config, full, require_report_hash=True)
    assert full_score["points"] == 10.0 and full_score["badge"] == "multimodal-ready"

    false_claim = multimodal_summary("public", 4, 0)
    assert shared.score_multimodal(
        config, {"public": false_claim}, require_report_hash=True
    )["points"] == 0.0
    fallback = multimodal_summary("public", 4, 4)
    fallback["fallback_active"] = True
    assert shared.score_multimodal(
        config, {"public": fallback}, require_report_hash=True
    )["points"] == 0.0
    unhealthy = multimodal_summary("public", 4, 4)
    unhealthy["service_healthy_after_run"] = False
    unhealthy_score = shared.score_multimodal(
        config, {"public": unhealthy}, require_report_hash=True
    )
    assert unhealthy_score["points"] == 2.0
    assert "public:service_not_healthy_after_run" in unhealthy_score["reasons"]
    assert shared.score_multimodal(config, None, require_report_hash=True)["points"] == 0.0

    bad_hash = multimodal_summary("public", 4, 4)
    bad_hash["report_sha256"] = "not-a-digest"
    expect_value_error(
        lambda: shared.score_multimodal(
            config, {"public": bad_hash}, require_report_hash=True
        ),
        "report_sha256",
    )

    implementation = {"name": "candidate", "revision": "a" * 40, "backend": "apxinf"}
    multimodal_contract_path = HERE / "multimodal-contract-v1.json"
    multimodal_contract_hash = hashlib.sha256(multimodal_contract_path.read_bytes()).hexdigest()
    raw_report = multimodal_report(
        "public", 4, implementation, multimodal_contract_hash
    )
    with tempfile.TemporaryDirectory() as raw:
        report_path = Path(raw) / "public.json"
        report_path.write_text(json.dumps(raw_report, sort_keys=True), encoding="utf-8")
        evidence, paths = runner.load_multimodal_evidence(
            report_path,
            None,
            multimodal_contract_path,
            HERE / "contract-v1.json",
            implementation,
        )
        assert paths == {"public": report_path}
        assert evidence["public"]["report_sha256"] == hashlib.sha256(
            report_path.read_bytes()
        ).hexdigest()
        expect_value_error(
            lambda: runner.load_multimodal_evidence(
                report_path,
                None,
                multimodal_contract_path,
                HERE / "contract-v1.json",
                {**implementation, "name": "different"},
            ),
            "implementation identity",
        )
        raw_report["evidence"]["contract_sha256"] = "f" * 64
        report_path.write_text(json.dumps(raw_report, sort_keys=True), encoding="utf-8")
        expect_value_error(
            lambda: runner.load_multimodal_evidence(
                report_path,
                None,
                multimodal_contract_path,
                HERE / "contract-v1.json",
                implementation,
            ),
            "different multimodal contract",
        )

    public_report = multimodal_report(
        "public", 4, implementation, multimodal_contract_hash
    )
    private_report = multimodal_report(
        "hidden", 8, implementation, multimodal_contract_hash
    )
    assert standalone.score(public_report, private_report)["leaderboard_points"] == 10.0

    complete = leaderboard.score_cohort(
        contract, [full_submission(contract, full)], "public_calibration"
    )["scores"][0]
    assert complete["eligible"]
    assert complete["section_scores"]["multimodal_bonus"] == 10.0
    assert complete["base_score"] == 100.0
    assert complete["bonus_score"] == 30.0
    assert complete["leaderboard_score"] == 130.0

    text_only = leaderboard.score_cohort(
        contract, [full_submission(contract, None)], "public_calibration"
    )["scores"][0]
    assert text_only["eligible"]
    assert text_only["section_scores"]["multimodal_bonus"] == 0.0
    assert text_only["leaderboard_score"] == 120.0


def git_revision() -> str:
    return subprocess.run(
        ["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def check() -> None:
    for name in MODULES:
        py_compile.compile(str(HERE / name), doraise=True)

    contract = json.loads((HERE / "contract-v1.json").read_text(encoding="utf-8"))
    assert contract["status"] == "released"
    assert contract["hardware"]["gpu"] == "NVIDIA GeForce RTX 4090"
    assert contract["hardware"]["count"] == 1
    assert contract["model"]["revision"] == "63768c10df38c0395e12ef49edac1bd539eaeeea"
    assert contract["multimodal_bonus"]["weight"] == 10.0
    assert contract["score_totals"]["maximum_leaderboard_score"] == 130.0

    tracked = subprocess.run(
        ["git", "-C", str(REPOSITORY), "ls-files", str(HERE.relative_to(REPOSITORY))],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    actual = {Path(path).name for path in tracked}
    if actual != PUBLIC_FILES:
        raise RuntimeError(
            "assignment file boundary mismatch: "
            f"missing={sorted(PUBLIC_FILES - actual)}, extra={sorted(actual - PUBLIC_FILES)}"
        )

    readmes = (REPOSITORY / "README.md", REPOSITORY / "README_EN.md")
    for readme in readmes:
        if not readme.is_file():
            raise RuntimeError(f"missing assignment README: {readme.name}")
    documentation = "\n".join(
        readme.read_text(encoding="utf-8").lower() for readme in readmes
    )
    forbidden = (
        "teach" + "er",
        "\u6559\u5e08",
        "teach" + "ing_guide",
        "slide_outline",
        "release_manifest",
    )
    leaked = [word for word in forbidden if word in documentation]
    if leaked:
        raise RuntimeError(f"internal terminology in assignment READMEs: {leaked}")
    for required in (
        "63768c10df38c0395e12ef49edac1bd539eaeeea",
        "maximum is\n130",
        "排行榜总分上限为 130",
        "python3 benchmarks/qwen38_4090/evaluation/test.py check",
    ):
        if required not in documentation:
            raise RuntimeError(f"assignment READMEs are missing {required!r}")

    protocol_checks()
    multimodal_checks(contract)
    for name in (
        "fetch_public_corpus.py",
        "generate_evaluation_cases.py",
        "run_evaluation.py",
        "score_multimodal.py",
        "score_submission.py",
    ):
        command(sys.executable, str(HERE / name), "--help")
    command("cargo", "check", "--workspace", "--locked", cwd=REPOSITORY)
    print("assignment checks passed")


def prepare(model_dir: Path, output_dir: Path) -> None:
    corpus = HERE / ".cache" / "pg24264.txt"
    command(sys.executable, str(HERE / "fetch_public_corpus.py"), "--output", str(corpus))
    command(
        sys.executable,
        str(HERE / "generate_evaluation_cases.py"),
        "--model-dir",
        str(model_dir),
        "--corpus",
        str(corpus),
        "--output-dir",
        str(output_dir),
        "--suite",
        "public",
    )


def run(model_dir: Path, base_url: str, data_dir: Path, output_dir: Path) -> None:
    if not (data_dir / "manifest.json").is_file():
        prepare(model_dir, data_dir)
    command(
        sys.executable,
        str(HERE / "run_evaluation.py"),
        "--dataset",
        str(data_dir),
        "--model-dir",
        str(model_dir),
        "--base-url",
        base_url,
        "--implementation-name",
        "apxinf-student",
        "--implementation-revision",
        git_revision(),
        "--backend",
        "apxinf",
        "--profile",
        "public_calibration",
        "--output-dir",
        str(output_dir),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="check the assignment package and Rust workspace")

    prepare_parser = subparsers.add_parser("prepare", help="prepare public test data")
    prepare_parser.add_argument("--model-dir", type=Path, required=True)
    prepare_parser.add_argument("--data-dir", type=Path, default=PUBLIC_DATA)

    run_parser = subparsers.add_parser("run", help="run public tests against a service")
    run_parser.add_argument("--model-dir", type=Path, required=True)
    run_parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    run_parser.add_argument("--data-dir", type=Path, default=PUBLIC_DATA)
    run_parser.add_argument("--output-dir", type=Path, default=RUNS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "check":
        check()
    elif args.command == "prepare":
        prepare(args.model_dir, args.data_dir)
    elif args.command == "run":
        run(args.model_dir, args.base_url, args.data_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
