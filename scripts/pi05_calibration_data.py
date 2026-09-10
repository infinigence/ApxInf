"""Data-source adapters for PI0.5 calibration Observations.

The calibration module consumes ApxInf Observations.  This module is the
optional outer seam that translates a *source* into that contract: NPZ files and
JSONL manifests, both readable anywhere with numpy and PIL, plus a direct
in-process capture from native LIBERO initial states.

The LIBERO path (``--libero-suite``) and ``apxinf-robo capture-libero``
**overlap deliberately**.  The downstream command writes an NPZ directory that
``load_npz_observations`` then reads, which makes the calibration input a
reviewable, re-runnable artifact and keeps MuJoCo out of a robot deployment's
engine.  This copy exists so ApxInf can recalibrate FP8 against its own
published LIBERO protocol without checking out a repository that takes ApxInf as
a submodule — a kernel change should not require the downstream repo to
regress.  Both go through ``scripts/libero_observation.py``, so a change to
camera orientation or state layout moves them together; see that module on
keeping it and its apxinf-robo mirror in step.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import pathlib
from typing import Any

import numpy as np
from PIL import Image

if __package__:
    from .libero_observation import make_env, to_apxinf_observation
else:
    from libero_observation import make_env, to_apxinf_observation


def _decode_npz_value(value):
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    return np.ascontiguousarray(array)


def load_npz_observations(
    paths: Sequence[pathlib.Path],
) -> tuple[Mapping[str, object], ...]:
    observations = []
    for path in paths:
        with np.load(path, allow_pickle=False) as sample:
            observations.append(
                {name: _decode_npz_value(sample[name]) for name in sample.files}
            )
    return tuple(observations)


def _load_rgb(path: pathlib.Path, *, field: str, line_number: int) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except (OSError, ValueError) as error:
        raise ValueError(
            f"manifest line {line_number}: cannot load {field} image {path}"
        ) from error


def load_observation_manifest(
    path: pathlib.Path,
    *,
    image_keys: Sequence[str],
    prompt_key: str,
    state_key: str,
) -> tuple[Mapping[str, object], ...]:
    """Load JSONL rows whose fields mirror the public Observation contract.

    Image fields contain paths relative to the manifest (or absolute paths),
    the prompt is a string, and optional state is an inline numeric array.
    """
    observations: list[Mapping[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"manifest line {line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(f"manifest line {line_number}: expected a JSON object")

            missing = [key for key in (*image_keys, prompt_key) if key not in row]
            if missing:
                raise ValueError(
                    f"manifest line {line_number}: missing Observation field(s): {missing}"
                )
            prompt = row[prompt_key]
            if not isinstance(prompt, str):
                raise ValueError(
                    f"manifest line {line_number}: {prompt_key} must be a string"
                )

            observation: dict[str, Any] = {prompt_key: prompt}
            for key in image_keys:
                image_value = row[key]
                if not isinstance(image_value, str):
                    raise ValueError(
                        f"manifest line {line_number}: {key} must be an image path"
                    )
                image_path = pathlib.Path(image_value).expanduser()
                if not image_path.is_absolute():
                    image_path = path.parent / image_path
                observation[key] = _load_rgb(
                    image_path, field=key, line_number=line_number
                )

            if state_key in row:
                try:
                    state = np.asarray(row[state_key], dtype=np.float32)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"manifest line {line_number}: {state_key} must be numeric"
                    ) from error
                if state.ndim != 1 or not np.all(np.isfinite(state)):
                    raise ValueError(
                        f"manifest line {line_number}: {state_key} must be a finite 1D array"
                    )
                observation[state_key] = np.ascontiguousarray(state)
            observations.append(observation)

    if not observations:
        raise ValueError(f"calibration manifest has no observations: {path}")
    return tuple(observations)


def task_stratified_indices(
    task_indices: Sequence[object], *, sample_count: int, seed: int
) -> list[int]:
    """Choose deterministic, balanced frame indices across every task group."""
    if sample_count < 1:
        raise ValueError("calibration sample count must be positive")
    if sample_count > len(task_indices):
        raise ValueError(
            f"requested {sample_count} calibration samples from {len(task_indices)} frames"
        )
    groups: dict[object, list[int]] = {}
    for index, raw_task in enumerate(task_indices):
        task = raw_task.item() if hasattr(raw_task, "item") else raw_task
        groups.setdefault(task, []).append(index)
    if sample_count < len(groups):
        raise ValueError(
            f"--samples={sample_count} cannot cover all {len(groups)} tasks; "
            "increase --samples"
        )

    rng = np.random.default_rng(seed)
    tasks = sorted(groups, key=lambda value: (type(value).__name__, repr(value)))
    queues = {task: list(rng.permutation(groups[task])) for task in tasks}
    selected: list[int] = []
    while len(selected) < sample_count:
        progressed = False
        for task in tasks:
            if queues[task] and len(selected) < sample_count:
                selected.append(int(queues[task].pop()))
                progressed = True
        if not progressed:
            break
    return selected


def _load_libero_suite(suite_name: str):
    try:
        from libero.libero import benchmark
    except ImportError as error:
        raise ImportError(
            "native LIBERO observations require the LIBERO and MuJoCo evaluation "
            "dependencies; install LIBERO as described in README.md"
        ) from error
    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        raise ValueError(f"unknown LIBERO suite: {suite_name}")
    return benchmark_dict[suite_name]()


def load_libero_observations(
    suite_name: str,
    *,
    image_keys: Sequence[str],
    sample_count: int | None,
    seed: int,
    prompt_key: str,
    state_key: str,
    progress: Callable[[str], None] | None = None,
) -> tuple[Mapping[str, object], ...]:
    """Capture task-balanced observations from native LIBERO initial states.

    Mirrors ``apxinf-robo capture-libero``, which writes the same frames to NPZ
    for :func:`load_npz_observations` to read back; see the module docstring on
    why both exist.
    """
    if len(image_keys) != 2:
        raise ValueError(
            "native LIBERO calibration requires exactly two configured image views"
        )
    progress = progress or (lambda _message: None)
    suite = _load_libero_suite(suite_name)
    states_by_task = [
        suite.get_task_init_states(task_id) for task_id in range(suite.n_tasks)
    ]
    task_indices = [
        task_id
        for task_id, initial_states in enumerate(states_by_task)
        for _ in range(len(initial_states))
    ]
    selected_count = sample_count if sample_count is not None else suite.n_tasks
    flat_indices = task_stratified_indices(
        task_indices, sample_count=selected_count, seed=seed
    )
    offsets = []
    offset = 0
    for initial_states in states_by_task:
        offsets.append(offset)
        offset += len(initial_states)
    selected_by_task: dict[int, list[int]] = {}
    for flat_index in flat_indices:
        task_id = task_indices[flat_index]
        selected_by_task.setdefault(task_id, []).append(flat_index - offsets[task_id])

    observations = []
    dummy_action = [0.0] * 6 + [-1.0]
    for task_id in sorted(selected_by_task):
        task = suite.get_task(task_id)
        prompt = str(task.language)
        progress(
            f"Capturing {suite_name} task {task_id} "
            f"({len(selected_by_task[task_id])} observation(s))..."
        )
        env = make_env(task, seed)
        try:
            for initial_state_index in selected_by_task[task_id]:
                env.reset()
                raw_observation = env.set_init_state(
                    states_by_task[task_id][initial_state_index]
                )
                for _ in range(10):
                    raw_observation, _, _, _ = env.step(dummy_action)
                observations.append(
                    to_apxinf_observation(
                        raw_observation,
                        prompt=prompt,
                        image_keys=(image_keys[0], image_keys[1]),
                        prompt_key=prompt_key,
                        state_key=state_key,
                    )
                )
        finally:
            env.close()
    return tuple(observations)
