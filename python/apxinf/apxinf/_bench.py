"""The ``apxinf.pi05.latency.v2`` report's arithmetic, in one place.

The schema is a contract between two scripts: ``scripts/bench_pi05.py`` (which
reports CUDA numbers) and the per-backend benchmarks that report the same layer
boundaries elsewhere.  ``bench_pi05.py`` is upstream's file: importing its
private ``_stats`` would make an upstream script depend on this package's
arithmetic in the other direction, so the canonical implementation lives here
and the two are held together *by value* — a MUSA latency and a CUDA latency
are not the same measurement otherwise.

Nothing here touches a device or a model: it is percentile bookkeeping, the
layer grammar, and the provenance strings a report carries.
"""

from __future__ import annotations

import os
import statistics
import subprocess
import time
from pathlib import Path


ALL_LAYERS = ("l0", "l1", "l2", "l3")
IN_PROCESS = ("l0", "l1", "l2")


def stats(samples_ms: list[float]) -> dict:
    """Percentiles of one layer's timed calls, nearest-rank.

    ``p95`` is the order statistic at ``int(0.95 * (n - 1))`` rather than an
    interpolation, so every number in the report is one that was actually
    measured -- with thirty samples there is nothing to interpolate between.
    """

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


def time_loop(fn, warmup: int, samples: int) -> tuple[list[float], list]:
    """Time ``samples`` calls of ``fn`` in milliseconds, after ``warmup`` of them.

    Also returns what the calls produced.  A benchmark that scores the actions
    has to score one of the calls it timed: a separate scored invocation is a
    second configuration, and its number and its p50 would describe two runs.
    """

    for _ in range(warmup):
        fn()
    timings, results = [], []
    for _ in range(samples):
        started = time.perf_counter()
        results.append(fn())
        timings.append((time.perf_counter() - started) * 1000.0)
    return timings, results


def parse_layers(spec: str, allowed: tuple[str, ...] = ALL_LAYERS) -> list[str]:
    """``l0,l1,l2,l3`` or ``all``, in canonical (innermost-first) order.

    ``all`` expands to ``allowed`` rather than to a fixed list: a report says
    which layers it covered, and a shorter set is a scope, not a typo.
    """

    if spec == "all":
        return list(allowed)
    picked = [item.strip().lower() for item in spec.split(",") if item.strip()]
    unknown = [item for item in picked if item not in allowed]
    if unknown:
        raise SystemExit(
            f"unknown --layer value(s): {', '.join(unknown)} "
            f"(choose from {','.join(allowed)},all)"
        )
    if not picked:
        raise SystemExit(f"--layer names nothing (choose from {','.join(allowed)},all)")
    return [layer for layer in allowed if layer in picked]


def git_commit(root: Path) -> str:
    """The revision a number was taken at, marked when the tree was not clean."""

    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, stderr=subprocess.DEVNULL
        )
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], cwd=root, stderr=subprocess.DEVNULL
        )
        return rev.decode().strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def env_snapshot(names) -> dict:
    """Which switches were set, and to what -- an unset switch is a decision too."""

    return {name: os.environ.get(name) for name in names}
