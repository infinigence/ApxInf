"""Shared numeric encodings for GR00T MUSA fixtures and reference scripts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


FIXTURE_SCHEMA = "apxinf.gr00t-n1.7.preprocessed-fixture.v1"
STORAGE = {"bfloat16": "<u2", "uint32": "<u4", "uint8": "u1"}
TENSOR_DTYPES = {
    "pixel_values": "bfloat16", "image_grid_thw": "uint32", "token_ids": "uint32",
    "attention_mask": "uint8", "state": "bfloat16", "noise": "bfloat16",
}


def decode_float_array(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind == "u" and values.dtype.itemsize == 2:
        values = (values.astype(np.uint32) << 16).view(np.float32)
    elif values.dtype.kind == "f":
        with np.errstate(over="ignore", invalid="ignore"):
            values = values.astype(np.float32)
    else:
        raise ValueError(f"{name} must contain floating values or raw BF16 uint16 bits, got {values.dtype}")
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain nonempty, finite float32 values")
    return np.ascontiguousarray(values)


def normalize_noise(values: np.ndarray) -> np.ndarray:
    values = decode_float_array(values, "noise")
    if values.shape == (40, 132):
        values = values[None]
    if values.shape != (1, 40, 132):
        raise ValueError(f"noise must have shape [40,132] or [1,40,132], got {values.shape}")
    return values


def fixture_bf16_bits(values: np.ndarray) -> np.ndarray:
    values = decode_float_array(values, "fixture BF16 input")
    bits = values.view(np.uint32)
    if np.any(bits & 0xFFFF):
        raise ValueError("fixture values must be exactly representable as BF16")
    return np.ascontiguousarray((bits >> 16).astype(np.uint16))


def normalize_actions(values: np.ndarray, name: str = "normalized_actions") -> np.ndarray:
    values = decode_float_array(values, name)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2:
        raise ValueError(f"{name} must have shape [horizon,dim] or [1,horizon,dim], got {values.shape}")
    return values


def load_noise(path: Path) -> np.ndarray:
    path = path.expanduser().resolve()
    if path.suffix == ".npy":
        return normalize_noise(np.load(path, allow_pickle=False))
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "noise" not in archive:
                raise ValueError(f"{path} has no 'noise' array")
            return normalize_noise(archive["noise"])
    raise ValueError(f"{path} must be .npy or .npz")


def load_metadata(path: Path) -> dict[str, str]:
    """Return the optional model/schema metadata of a JSON archive."""
    path = path.expanduser().resolve()
    if path.suffix != ".json":
        return {}
    document = json.loads(path.read_text())
    return {
        key: document[key]
        for key in ("model", "schema")
        if isinstance(document.get(key), str)
    }


def checked_shape(shape: object, name: str) -> tuple[int, ...]:
    if not isinstance(shape, list) or not shape or any(
        type(dimension) is not int or dimension <= 0 for dimension in shape
    ):
        raise ValueError(f"{name} shape must be a nonempty list of positive integers")
    return tuple(shape)


def load_archive(
    path: Path,
    *,
    expect_model: str | None = None,
    expect_schema_prefix: str | None = None,
) -> dict[str, np.ndarray]:
    path = path.expanduser().resolve()
    if path.suffix == ".npy":
        return {"normalized_actions": normalize_actions(np.load(path, allow_pickle=False))}
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "normalized_actions" not in archive:
                raise ValueError(f"{path} is missing normalized_actions; decoded actions are not a substitute")
            result = {"normalized_actions": normalize_actions(archive["normalized_actions"])}
            if "actions" in archive:
                result["actions"] = normalize_actions(archive["actions"], "actions")
            if "noise" in archive:
                result["noise"] = normalize_noise(archive["noise"])
            return result
    if path.suffix == ".json":
        document = json.loads(path.read_text())
        try:
            output = document["output"]
            shape = checked_shape(output["shape"], "output")
            values = np.asarray(output["values"])
        except (KeyError, TypeError) as error:
            raise ValueError(f"{path} requires output.values and output.shape") from error
        if values.dtype.kind not in "fiu" or values.size != math.prod(shape):
            raise ValueError(f"{path} output.values must be numeric and match output.shape")
        with np.errstate(over="ignore", invalid="ignore"):
            values = values.astype(np.float32).reshape(shape)
        result = {"normalized_actions": normalize_actions(values)}
        if "noise" in document:
            noise = document["noise"]
            if not isinstance(noise, dict):
                raise ValueError(f"{path} noise must be an object with shape and values")
            try:
                noise_shape = checked_shape(noise["shape"], "noise")
                noise_values = np.asarray(noise["values"])
            except (KeyError, TypeError) as error:
                raise ValueError(f"{path} noise requires shape and values") from error
            if noise_values.dtype.kind not in "fiu" or noise_values.size != math.prod(noise_shape):
                raise ValueError(f"{path} noise.values must be numeric and match noise.shape")
            with np.errstate(over="ignore", invalid="ignore"):
                noise_values = noise_values.astype(np.float32).reshape(noise_shape)
            result["noise"] = normalize_noise(noise_values)
        model = document.get("model")
        if expect_model is not None and model is not None:
            if not isinstance(model, str) or model != expect_model:
                raise ValueError(
                    f"{path} model {model!r} does not match expected {expect_model!r}"
                )
        schema = document.get("schema")
        if expect_schema_prefix is not None and schema is not None:
            if not isinstance(schema, str) or not schema.startswith(expect_schema_prefix):
                raise ValueError(
                    f"{path} schema {schema!r} does not match prefix {expect_schema_prefix!r}"
                )
        return result
    raise ValueError(f"{path} must be .npy, .npz or .json")


def load_fixture(root: Path) -> tuple[dict, dict[str, np.ndarray]]:
    root = root.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != FIXTURE_SCHEMA:
        raise ValueError("unsupported fixture schema")
    arrays = {}
    for name, dtype in TENSOR_DTYPES.items():
        entry = manifest["tensors"][name]
        if entry["dtype"] != dtype:
            raise ValueError(f"{name} must have dtype {dtype}")
        shape = checked_shape(entry["shape"], name)
        path = (root / entry["file"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"{name} tensor escapes fixture directory")
        storage = np.dtype(STORAGE[dtype])
        expected_bytes = math.prod(shape) * storage.itemsize
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"{name} byte size must be {expected_bytes} for shape {shape}")
        values = np.fromfile(path, dtype=storage).reshape(shape)
        arrays[name] = decode_float_array(values, name) if dtype == "bfloat16" else np.ascontiguousarray(values)
    tokens, mask, grid = arrays["token_ids"], arrays["attention_mask"], arrays["image_grid_thw"]
    if tokens.ndim != 1 or mask.shape != tokens.shape or not np.isin(mask, (0, 1)).all():
        raise ValueError("token_ids and attention_mask must be equal-length vectors with a boolean mask")
    if grid.ndim != 2 or grid.shape[1] != 3 or not (grid > 0).all():
        raise ValueError("image_grid_thw must have shape [images,3] with positive entries")
    if arrays["pixel_values"].ndim != 2:
        raise ValueError("pixel_values must be rank 2")
    if arrays["state"].ndim not in (2, 3) or (arrays["state"].ndim == 3 and arrays["state"].shape[0] != 1):
        raise ValueError("state must have shape [history,dim] or [1,history,dim]")
    arrays["noise"] = normalize_noise(arrays["noise"])
    if type(manifest.get("embodiment_id")) is not int or manifest["embodiment_id"] < 0:
        raise ValueError("embodiment_id must be a non-negative integer")
    if not isinstance(manifest.get("fixture"), str) or not manifest["fixture"]:
        raise ValueError("fixture must be a nonempty name")
    return manifest, arrays
