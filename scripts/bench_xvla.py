#!/usr/bin/env python3
"""Benchmark the X-VLA policy on the local GPU.

One entry point: load ``AutoPolicy`` from a real X-VLA checkpoint and time
end-to-end ``policy.infer`` on a fixed two-view observation.

Two measurements per run:

* ``steady`` — P50/P95 over ``--samples`` inferences after ``--warmup``
  warm-up calls. The runtime selects ``PreferGraph`` (CUDA graph capture on
  the first inference, replay after), so steady-state samples time the graph
  replay path that serving uses.
* ``cold`` — wall time of the very first ``policy.infer`` in a fresh process,
  including model load, eager warm-up and CUDA graph capture.

``--runs`` repeats the steady protocol; each run reuses the same policy
(warm-up only on the first run is skipped by re-warming ``--warmup//2``).
The per-sample ``model_ms``/``total_ms`` split from the policy timing dict is
reported for the last sample of each run.

The Python surface always selects ``PreferGraph``; ``ExecutionPolicy::Eager``
is a runtime-internal path (used for the pre-capture warm-up) and cannot be
forced from this script.

Examples:

    python scripts/bench_xvla.py --model-dir /path/to/X-VLA-Libero
    python scripts/bench_xvla.py --model-dir /path/to/X-VLA-Libero \
        --warmup 10 --samples 30 --runs 3 --out bench-xvla.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]

_DEFAULT_PROMPT = "put both moka pots on the stove"


def _stats(samples_ms):
    ordered = sorted(samples_ms)
    n = len(ordered)
    return {
        "p50": ordered[int(0.50 * (n - 1))],
        "p95": ordered[int(0.95 * (n - 1))],
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "std": statistics.pstdev(ordered) if n > 1 else 0.0,
        "samples": n,
    }


def _time_loop(fn, warmup, samples):
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(samples):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1000.0)
    return out


def _git_commit():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=_REPO_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _observation():
    return {
        "image0": np.zeros((224, 224, 3), np.uint8),
        "image1": np.zeros((224, 224, 3), np.uint8),
        "proprio": np.zeros(20, np.float32),
        "language_instruction": _DEFAULT_PROMPT,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="X-VLA checkpoint directory")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", help="write the report as JSON to this path")
    args = parser.parse_args()

    from apxinf import AutoPolicy

    load_start = time.perf_counter()
    policy = AutoPolicy.from_pretrained(args.model_dir)
    load_s = time.perf_counter() - load_start
    metadata = dict(policy.metadata)
    print(f"model_type={metadata.get('model_type')} load_s={load_s:.2f}")

    observation = _observation()

    # Cold first call: eager warm-up + CUDA graph capture.
    cold_start = time.perf_counter()
    cold_result = policy.infer(observation)
    cold_ms = (time.perf_counter() - cold_start) * 1000.0
    actions = np.asarray(cold_result["actions"])
    print(f"cold_first_infer_ms={cold_ms:.2f} actions={actions.shape}")
    print(f"cold timing={cold_result.get('timing')}")

    report = {
        "model": "xvla",
        "checkpoint": str(args.model_dir),
        "metadata": {k: str(v) for k, v in metadata.items()},
        "commit": _git_commit(),
        "load_s": load_s,
        "cold_first_infer_ms": cold_ms,
        "cold_timing": cold_result.get("timing"),
        "protocol": {
            "warmup": args.warmup,
            "samples": args.samples,
            "runs": args.runs,
            "batch": 1,
            "views": 2,
            "image": "224x224x3 uint8 zeros",
            "prompt": _DEFAULT_PROMPT,
            "mode": "PreferGraph (default; CUDA graph replay after capture)",
        },
        "runs": [],
    }

    infer = lambda: policy.infer(observation)  # noqa: E731
    for run in range(args.runs):
        samples_ms = _time_loop(infer, args.warmup, args.samples)
        stats = _stats(samples_ms)
        last = infer()
        report["runs"].append(
            {
                "run": run + 1,
                "steady_ms": stats,
                "last_timing": last.get("timing"),
            }
        )
        print(
            f"run {run + 1}: p50={stats['p50']:.2f}ms p95={stats['p95']:.2f}ms "
            f"mean={stats['mean']:.2f}ms min={stats['min']:.2f}ms max={stats['max']:.2f}ms"
        )

    all_p50 = [r["steady_ms"]["p50"] for r in report["runs"]]
    report["steady_p50_ms_median"] = statistics.median(all_p50)
    print(f"steady p50 median across runs: {report['steady_p50_ms_median']:.2f} ms")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
        print(f"report written to {args.out}")


if __name__ == "__main__":
    main()
