#!/usr/bin/env python3
"""Start ApxInf's pinned DM05-libero HTTP service."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PYTHON_PACKAGE = REPO / "python" / "apxinf"
if str(PYTHON_PACKAGE) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGE))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16",), default="bf16")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7891)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    from apxinf import AutoPolicy
    from apxinf.serving import serve_dm05_http

    policy = AutoPolicy.from_pretrained(
        args.model_dir,
        device=args.device,
        precision=args.precision,
        default_seed=args.seed,
    )
    print(f"apxinf dm05 ready metadata={dict(policy.metadata)}", flush=True)
    try:
        serve_dm05_http(policy, args.host, args.port)
    finally:
        policy.close()


if __name__ == "__main__":
    main()
