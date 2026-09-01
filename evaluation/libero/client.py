"""Client-only, resumable LIBERO accuracy evaluation over websocket."""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
import traceback
from typing import Optional, Sequence

from .contract import LiberoWebsocketClient
from .ledger import LedgerKey, append_record, completed_runs, write_summary
from .rollout import make_env, run_episode


ALL_SUITES = (
    "libero_10",
    "libero_90",
    "libero_spatial",
    "libero_object",
    "libero_goal",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=("fp8", "bf16", "int8"), required=True)
    parser.add_argument("--suite", default="libero_10", choices=(*ALL_SUITES, "all"))
    parser.add_argument(
        "--tasks",
        default="all",
        help="'all' (default) or a comma list of task ids, applied within each suite",
    )
    parser.add_argument("--trials-per-task", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--results-jsonl", required=True, type=pathlib.Path)
    parser.add_argument("--summary-json", required=True, type=pathlib.Path)
    parser.add_argument("--host", default="127.0.0.1", help="policy server host")
    parser.add_argument("--port", type=int, default=8000, help="policy server port")
    args = parser.parse_args(argv)
    if args.trials_per_task <= 0 or args.trials_per_task > 50:
        parser.error("--trials-per-task must be in 1..=50")
    return args


def resolve_suites(name: str) -> list[str]:
    return list(ALL_SUITES) if name == "all" else [name]


def resolve_task_ids(spec: str, n_tasks: int, suite: str) -> list[int]:
    if spec.strip() == "all":
        return list(range(n_tasks))
    task_ids = [int(value) for value in spec.split(",") if value.strip()]
    if sorted(set(task_ids)) != sorted(task_ids):
        raise ValueError(f"--tasks has duplicate ids: {task_ids}")
    out_of_range = [task for task in task_ids if not 0 <= task < n_tasks]
    if out_of_range:
        raise ValueError(
            f"--tasks {out_of_range} out of range for suite {suite!r} "
            f"(0..{n_tasks - 1})"
        )
    return task_ids


def run_evaluation(args: argparse.Namespace) -> None:
    transport = "openpi_websocket"
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    suites: dict[str, object] = {}
    task_ids_by_suite: dict[str, list[int]] = {}
    for name in resolve_suites(args.suite):
        suite = benchmark_dict[name]()
        suites[name] = suite
        task_ids_by_suite[name] = resolve_task_ids(args.tasks, suite.n_tasks, name)

    expected_keys: set[LedgerKey] = {
        (name, task_id, trial_id)
        for name, task_ids in task_ids_by_suite.items()
        for task_id in task_ids
        for trial_id in range(args.trials_per_task)
    }
    ledger = completed_runs(args.results_jsonl, args.precision)
    unexpected = set(ledger) - expected_keys
    if unexpected:
        raise ValueError(
            f"ledger contains runs outside requested scope: {sorted(unexpected)}"
        )
    write_summary(args.summary_json, ledger, expected_keys, args.precision, transport)

    client = LiberoWebsocketClient(args.host, args.port, args.precision)
    print(f"client=websocket server_metadata={client.metadata}", flush=True)
    try:
        for name, suite in suites.items():
            for task_id in task_ids_by_suite[name]:
                task = suite.get_task(task_id)
                prompt = str(task.language)
                pending = [
                    trial_id
                    for trial_id in range(args.trials_per_task)
                    if (name, task_id, trial_id) not in ledger
                ]
                if not pending:
                    print(f"{name} task {task_id}: already complete", flush=True)
                    continue
                print(f"{name} task {task_id}: pending trials {pending}", flush=True)
                initial_states = suite.get_task_init_states(task_id)
                env = make_env(task, args.seed)
                try:
                    for trial_id in pending:
                        for attempt in range(1, args.max_attempts + 1):
                            try:
                                record = run_episode(
                                    env, initial_states[trial_id], name, task_id,
                                    trial_id, prompt, client, transport, args.seed,
                                )
                                record.update(attempt=attempt, precision=args.precision)
                                append_record(args.results_jsonl, record)
                                ledger[(name, task_id, trial_id)] = record
                                write_summary(
                                    args.summary_json, ledger, expected_keys,
                                    args.precision, transport,
                                )
                                print(
                                    f"{name} task={task_id} trial={trial_id} "
                                    f"success={record['success']} "
                                    f"steps={record['action_steps']} "
                                    f"replans={record['replans']} "
                                    f"completed={len(ledger)}/{len(expected_keys)}",
                                    flush=True,
                                )
                                break
                            except Exception as error:
                                append_record(
                                    args.results_jsonl,
                                    {
                                        "status": "technical_error",
                                        "suite": name,
                                        "task_id": task_id,
                                        "trial_id": trial_id,
                                        "attempt": attempt,
                                        "precision": args.precision,
                                        "transport": transport,
                                        "error": repr(error),
                                        "traceback": traceback.format_exc(),
                                        "time_unix_seconds": time.time(),
                                    },
                                )
                                print(
                                    f"{name} task={task_id} trial={trial_id} "
                                    f"attempt={attempt} ERROR: {error}",
                                    file=sys.stderr,
                                    flush=True,
                                )
                                if attempt == args.max_attempts:
                                    raise
                finally:
                    env.close()
    finally:
        client.close()

    write_summary(args.summary_json, ledger, expected_keys, args.precision, transport)
    missing = expected_keys - set(ledger)
    if missing:
        raise RuntimeError(f"evaluation incomplete; missing {sorted(missing)}")
    successes = sum(bool(record["success"]) for record in ledger.values())
    print(
        f"LIBERO [{','.join(suites)}] complete: "
        f"{successes}/{len(expected_keys)} successes",
        flush=True,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_evaluation(parse_args(argv))


if __name__ == "__main__":
    main()
