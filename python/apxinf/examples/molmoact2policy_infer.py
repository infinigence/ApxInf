#!/usr/bin/env python3
"""Minimal ``MolmoAct2Policy`` example: load the LIBERO checkpoint, run chunks through the CUDA graph.

Shows the concrete entry point and the graph lifecycle:

* the first ``infer`` call *prepares* the runner: it records the whole-model plan for this prompt/image
  structure and, for a graph runner, captures one CUDA graph (an ``eager`` runner prepares the same plan
  without capturing); this is paid once and reported as the first call below;
* later calls only update the inputs in place and *replay* the graph;
* a request whose structure differs (other prompt length / image layout) transparently builds a new plan
  and re-captures (``execution="require-graph"`` makes a failed capture an error instead of a prepared-eager run;
  ``execution="eager"`` never captures and enqueues the same traversal from the host on reused executions);
* every returned ``actions`` array is an independent copy: it stays valid across later calls.

Requires the ``apxinf_py`` CUDA binding (``maturin develop`` of crates/apxinf-py), the original
``allenai/MolmoAct2-LIBERO`` checkpoint directory, and ``torch``/``torchvision`` for the host resize.

    python examples/molmoact2policy_infer.py --model-dir /path/to/MolmoAct2-LIBERO

The synthetic frames make the actions meaningless; this only exercises the interface.
"""

from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np

from _common import synthetic_observation  # noqa: E402 (path shim in _common)

from apxinf.policies import MolmoAct2Policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=pathlib.Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--execution",
        choices=("prefer-graph", "require-graph", "eager"),
        default=None,
        help="default: prefer-graph (graph, with an explained prepared-eager fallback if capture fails); eager = prepared eager, no graph",
    )
    parser.add_argument("--tactics", type=pathlib.Path, help="GEMM/attention recipe directory (read; written too when --autotune is given; see doc/molmoact2.md)")
    parser.add_argument("--autotune", action="store_true", help="tune missing tactics and write them to --tactics")
    parser.add_argument("--chunks", type=int, default=5, help="number of repeated chunks to time")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = MolmoAct2Policy.from_pretrained(
        args.model_dir,
        device=args.device,
        execution=args.execution,
        tactics=args.tactics,
        autotune=args.autotune,
    )
    try:
        print("metadata:", policy.metadata)
        print("execution before the first call:", policy.execution_mode)

        observation = synthetic_observation(
            image_keys=policy.image_keys,
            state_key=policy.state_key,
            prompt_key="task",
            state_dim=8,
            prompt="put the white mug on the plate",
        )

        started = time.perf_counter()
        first = policy.infer(observation)  # prepare (+ capture for a graph runner) + first run
        print(f"first call (prepare, plus capture for a graph runner, and run): {(time.perf_counter() - started) * 1e3:.0f} ms")
        print("execution after the first call:", first["execution_mode"])

        times = []
        results = []
        for _ in range(args.chunks):
            started = time.perf_counter()
            results.append(policy.infer(observation))
            times.append((time.perf_counter() - started) * 1e3)
        print(f"steady chunks (ms, incl. resize/tokenizer/upload/download): {[round(t, 1) for t in times]}")

        actions = first["actions"]
        print(f"actions: shape={actions.shape} dtype={actions.dtype}  (physical 7-dim, horizon 10)")
        print(f"timing:  {results[-1]['timing']}")
        print(f"first action vector: {actions[0]}")

        # Output ownership: an array returned earlier is a private copy, later replays never touch it.
        snapshot = first["actions"].copy()
        policy.infer(observation)
        print("earlier result unchanged by later replays:", bool(np.array_equal(first["actions"], snapshot)))
        # Provided noise (official full FP32 [1, 10, 32]) is bound per call and reproduces exactly;
        # leaving it out draws fresh device noise on every replay.
        noise = np.random.default_rng(0).standard_normal((1, 10, 32)).astype(np.float32)
        a = policy.infer(observation, noise=noise)["actions"]
        b = policy.infer(observation, noise=noise)["actions"]
        print("same provided noise -> identical actions:", bool(np.array_equal(a, b)))
    finally:
        policy.close()


if __name__ == "__main__":
    main()
