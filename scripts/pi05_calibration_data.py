"""Data-source adapters for PI0.5 calibration Observations.

The calibration module consumes ApxInf Observations.  This module is the
optional outer seam that translates *storage formats* into that contract: NPZ
files and JSONL manifests, both readable anywhere with numpy and PIL.

Capturing observations from a live simulator is a different job with a different
dependency set, and it belongs to whoever owns the environment.  ``apxinf-robo
capture-libero`` writes an NPZ directory this module then reads, which keeps
MuJoCo out of the engine and makes the calibration input a reviewable artifact
rather than a side effect of a rollout.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import pathlib
from typing import Any

import numpy as np
from PIL import Image


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
