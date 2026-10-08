#!/usr/bin/env python3
"""SmolVLA-specific LIBERO accuracy evaluation (in-process)."""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
import traceback
from typing import Any, Optional, Tuple

import numpy as np

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_APXINF_PKG = _REPO_ROOT / "python" / "apxinf"
for path in (_SCRIPT_DIR, _APXINF_PKG):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from apxinf.policies.impls.smolvla import SmolVlaPolicy
from eval_libero import (
    MAX_STEPS,
    REPLAN_STEPS,
    append_record,
    completed_runs,
    libero_state,
    load_libero_init_states,
    make_env,
    resolve_suites,
    resolve_task_ids,
    run_episode,
    state_finger_joints,
    write_summary,
)

_DEFAULT_RESULTS = (
    _REPO_ROOT
    / "devlocal"
    / "smolvla-integration"
    / "results"
    / "libero-results.jsonl"
)
_DEFAULT_SUMMARY = (
    _REPO_ROOT
    / "devlocal"
    / "smolvla-integration"
    / "results"
    / "libero-summary.json"
)


class SmolVlaBackend:
    """In-process adapter from the shared LIBERO rollout to SmolVlaPolicy."""

    def __init__(self, args: argparse.Namespace) -> None:
        self._policy = SmolVlaPolicy.from_pretrained(
            args.model_dir,
            device=args.device,
            seed=args.seed,
            num_views=2,
            tactics=args.tactics,
            autotune=args.autotune,
            model_variant=args.model_variant,
            image_keys=("observation.images.camera1", "observation.images.camera2"),
            state_key="observation.state",
            prompt_key="task",
        )
        self.metadata = dict(self._policy.metadata)
        self._finger_joints = state_finger_joints(self.metadata)

    @property
    def finger_joints(self) -> Any:
        return self._finger_joints

    def state_from_observation(self, observation) -> Any:
        return libero_state(observation, finger_joints=self._finger_joints)

    def infer(
        self,
        base: np.ndarray,
        wrist: np.ndarray,
        state: Any,
        prompt: str,
        noise: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        result = self._policy.infer(
            {
                "observation.images.camera1": base,
                "observation.images.camera2": wrist,
                "observation.state": state,
                "task": prompt,
            },
            noise=noise,
        )
        timing = result.get("timing", {}) or {}
        model_ms = float(timing.get("model_ms", 0.0))
        total_ms = float(timing.get("total_ms", model_ms))
        return (
            np.asarray(result["actions"], dtype=np.float32),
            np.asarray(result["normalized_actions"], dtype=np.float32),
            {
                "model_seconds": model_ms / 1000.0,
                "server_processor_seconds": max(
                    0.0, total_ms - model_ms
                ) / 1000.0,
            },
        )

    def close(self) -> None:
        self._policy.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=pathlib.Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--model-variant",
        choices=("bf16", "fp16"),
        default="bf16",
    )
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--trials-per-task", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tactics", type=pathlib.Path)
    parser.add_argument(
        "--autotune",
        action="store_true",
        help="tune missing exact GEMM keys before the first rollout",
    )
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--replan-steps", type=int, default=REPLAN_STEPS)
    parser.add_argument("--settle-gripper", type=float, default=-1.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--results-jsonl", type=pathlib.Path, default=_DEFAULT_RESULTS)
    parser.add_argument("--summary-json", type=pathlib.Path, default=_DEFAULT_SUMMARY)
    args = parser.parse_args()
    if args.trials_per_task <= 0 or args.trials_per_task > 50:
        parser.error("--trials-per-task must be in 1..=50")
    if args.max_steps <= 0 or args.replan_steps <= 0:
        parser.error("--max-steps and --replan-steps must be positive")
    if args.replan_steps > 50:
        parser.error("--replan-steps must be <= 50 (SmolVLA action horizon)")
    return args


def main() -> None:
    args = parse_args()
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    suites: dict[str, object] = {}
    task_ids_by_suite: dict[str, list[int]] = {}
    for suite_name in resolve_suites(args.suite):
        suite = benchmark_dict[suite_name]()
        suites[suite_name] = suite
        task_ids_by_suite[suite_name] = resolve_task_ids(
            args.tasks, suite.n_tasks, suite_name
        )

    expected_keys = {
        (suite_name, task_id, trial_id)
        for suite_name, task_ids in task_ids_by_suite.items()
        for task_id in task_ids
        for trial_id in range(args.trials_per_task)
    }
    ledger = completed_runs(
        args.results_jsonl,
        args.model_variant,
        max_steps=args.max_steps,
        replan_steps=args.replan_steps,
    )
    unexpected = set(ledger) - expected_keys
    if unexpected:
        raise ValueError(
            f"ledger contains runs outside requested scope: {sorted(unexpected)}"
        )
    write_summary(
        args.summary_json,
        ledger,
        expected_keys,
        args.model_variant,
        "smolvla_in_process",
        max_steps=args.max_steps,
        replan_steps=args.replan_steps,
    )

    backend = SmolVlaBackend(args)
    print(f"backend=smolvla_in_process metadata={backend.metadata}", flush=True)
    try:
        for suite_name, suite in suites.items():
            for task_id in task_ids_by_suite[suite_name]:
                task = suite.get_task(task_id)
                prompt = str(task.language)
                pending = [
                    trial_id
                    for trial_id in range(args.trials_per_task)
                    if (suite_name, task_id, trial_id) not in ledger
                ]
                if not pending:
                    print(f"{suite_name} task {task_id}: already complete", flush=True)
                    continue
                print(
                    f"{suite_name} task {task_id}: pending trials {pending}", flush=True
                )
                initial_states = load_libero_init_states(suite, task_id)
                env = make_env(task, args.seed)
                try:
                    for trial_id in pending:
                        for attempt in range(1, args.max_attempts + 1):
                            try:
                                record = run_episode(
                                    env,
                                    initial_states[trial_id],
                                    suite_name,
                                    task_id,
                                    trial_id,
                                    prompt,
                                    backend,
                                    "smolvla_in_process",
                                    args.seed,
                                    False,
                                    0.5,
                                    args.replan_steps,
                                    args.settle_gripper,
                                    backend.finger_joints,
                                    args.max_steps,
                                )
                                record["attempt"] = attempt
                                record["precision"] = args.model_variant
                                append_record(args.results_jsonl, record)
                                ledger[(suite_name, task_id, trial_id)] = record
                                write_summary(
                                    args.summary_json,
                                    ledger,
                                    expected_keys,
                                    args.model_variant,
                                    "smolvla_in_process",
                                    max_steps=args.max_steps,
                                    replan_steps=args.replan_steps,
                                )
                                print(
                                    f"{suite_name} task={task_id} trial={trial_id} "
                                    f"success={record['success']} "
                                    f"steps={record['action_steps']} "
                                    f"replans={record['replans']} "
                                    f"completed={len(ledger)}/{len(expected_keys)}",
                                    flush=True,
                                )
                                break
                            except Exception as error:
                                failure = {
                                    "status": "technical_error",
                                    "suite": suite_name,
                                    "task_id": task_id,
                                    "trial_id": trial_id,
                                    "attempt": attempt,
                                    "precision": args.model_variant,
                                    "transport": "smolvla_in_process",
                                    "error": repr(error),
                                    "traceback": traceback.format_exc(),
                                    "time_unix_seconds": time.time(),
                                }
                                append_record(args.results_jsonl, failure)
                                print(
                                    f"{suite_name} task={task_id} trial={trial_id} "
                                    f"attempt={attempt} ERROR: {error}",
                                    file=sys.stderr,
                                    flush=True,
                                )
                                if attempt == args.max_attempts:
                                    raise
                finally:
                    env.close()
    finally:
        backend.close()

    write_summary(
        args.summary_json,
        ledger,
        expected_keys,
        args.model_variant,
        "smolvla_in_process",
        max_steps=args.max_steps,
        replan_steps=args.replan_steps,
    )
    missing = expected_keys - set(ledger)
    if missing:
        raise RuntimeError(f"evaluation incomplete; missing {sorted(missing)}")
    successes = sum(bool(record["success"]) for record in ledger.values())
    print(
        f"SmolVLA LIBERO complete: {successes}/{len(expected_keys)} successes",
        flush=True,
    )


if __name__ == "__main__":
    main()
