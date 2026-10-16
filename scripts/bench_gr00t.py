#!/usr/bin/env python3
"""Benchmark GR00T model-core inference with constructed in-memory inputs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from _benchmark import provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--views", type=int, choices=(1, 2), default=2)
    parser.add_argument("--precision", choices=("bf16", "fp8", "int8"), required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--tactics", type=Path)
    parser.add_argument(
        "--autotune",
        action="store_true",
        help="tune missing GEMM tactics and persist them at --tactics",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("devlocal/gr00t-n1d7/results/gr00t-bench.json"),
    )
    parser.add_argument(
        "--binary",
        type=Path,
        help="prebuilt gr00t_bench binary; otherwise cargo run --release is used",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.samples <= 0:
        raise SystemExit("--warmup must be non-negative and --samples must be positive")
    args.out = args.out.resolve()
    args.calibration = args.calibration.resolve() if args.calibration else None
    args.tactics = args.tactics.resolve() if args.tactics else None
    args.binary = args.binary.resolve() if args.binary else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    executable = (
        [str(args.binary)]
        if args.binary is not None
        else [
            "cargo",
            "run",
            "--release",
            "-p",
            "apxinf-model",
            "--features",
            "cuda",
            "--example",
            "gr00t_bench",
            "--",
        ]
    )
    command = [
        *executable,
        str(args.model_dir.resolve()),
        str(args.views),
        args.precision,
        str(args.device),
        str(args.warmup),
        str(args.samples),
        str(args.calibration) if args.calibration is not None else "-",
        str(args.tactics) if args.tactics is not None else "-",
        str(args.out),
    ]
    if args.autotune:
        if args.tactics is None:
            raise SystemExit("--autotune requires an explicit --tactics output path")
        args.tactics.parent.mkdir(parents=True, exist_ok=True)
        command.append("--autotune")
    subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)
    report = json.loads(args.out.read_text())
    binary = args.binary or Path(os.environ.get('CARGO_TARGET_DIR', Path(__file__).resolve().parents[1] / 'target')) / 'release/examples/gr00t_bench'
    report['provenance'] = provenance(Path(__file__).resolve().parents[1], args.model_dir,
                                      binary=binary if binary.is_file() else None, tactics=args.tactics,
                                      calibration=args.calibration)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
