#!/usr/bin/env python3
"""Single entry point for the Qwen3.8 RTX 4090 assignment."""

from __future__ import annotations

import argparse
import importlib.util
import json
import py_compile
import subprocess
import sys
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
    "run_evaluation.py",
    "test.py",
)
PUBLIC_FILES = {
    ".gitignore",
    "ASSIGNMENT.md",
    "contract-v1.json",
    "fetch_public_corpus.py",
    "generate_evaluation_cases.py",
    "run_evaluation.py",
    "submission-schema-v1.json",
    "test.py",
}


def command(*argv: str, cwd: Path | None = None) -> None:
    print("+", " ".join(argv), flush=True)
    subprocess.run(argv, cwd=cwd, check=True)


def load_runner():
    spec = importlib.util.spec_from_file_location("assignment_runner", HERE / "run_evaluation.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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

    assignment = (HERE / "ASSIGNMENT.md").read_text(encoding="utf-8").lower()
    forbidden = ("teacher", "教师", "teaching_guide", "slide_outline", "release_manifest")
    leaked = [word for word in forbidden if word in assignment]
    if leaked:
        raise RuntimeError(f"internal terminology in assignment: {leaked}")

    protocol_checks()
    for name in ("fetch_public_corpus.py", "generate_evaluation_cases.py", "run_evaluation.py"):
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
