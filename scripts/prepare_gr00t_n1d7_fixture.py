#!/usr/bin/env python3
"""Convert an official GR00T N1.7 processor dump into a compact fixture.

The input NPZ is expected to contain the tensors emitted by the NVIDIA
processor at the model boundary.  Floating-point arrays are written as exact
little-endian BF16 payloads so the Rust benchmark consumes the same values
without JSON rounding or an additional Python dependency.

This tool does not run preprocessing and therefore does not claim raw-image
latency.  A processor timing report can be attached to the manifest with
``--processor-report``; the Rust benchmark then reports both components while
keeping their timing boundaries explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "apxinf.gr00t-n1.7.preprocessed-fixture.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="official processor NPZ")
    parser.add_argument("--metadata", type=Path, help="optional source metadata JSON")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--fixture",
        default="nvidia-libero-episode0-step0-real-v1",
        help="stable fixture identifier",
    )
    parser.add_argument(
        "--processor-report",
        type=Path,
        help="optional JSON containing latency_ms.samples or latency_ms.median",
    )
    parser.add_argument("--action-horizon", type=int, default=40)
    parser.add_argument("--action-dim", type=int, default=132)
    parser.add_argument(
        "--noise-kind",
        choices=("fixed-zero", "seeded-normal"),
        default="fixed-zero",
        help="description recorded for initial_noise; values come from the NPZ when present",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        help="seed recorded for seeded-normal initial_noise",
    )
    return parser.parse_args()


def require_array(archive: Any, name: str) -> np.ndarray:
    if name not in archive:
        raise KeyError(f"processor dump is missing {name!r}")
    value = np.asarray(archive[name])
    if value.size == 0:
        raise ValueError(f"processor tensor {name!r} is empty")
    return value


def integral_u32(value: np.ndarray, name: str) -> np.ndarray:
    numeric = np.asarray(value, dtype=np.float64)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{name} contains a non-integral or non-finite value")
    if numeric.min() < 0 or numeric.max() > np.iinfo(np.uint32).max:
        raise ValueError(f"{name} is outside the u32 range")
    return numeric.astype("<u4")


def f32_to_bf16_bits(value: np.ndarray, name: str) -> np.ndarray:
    numeric = np.asarray(value, dtype="<f4")
    if not np.isfinite(numeric).all():
        raise ValueError(f"{name} contains NaN or infinity")
    bits = numeric.view("<u4")
    # Round-to-nearest-even.  Official dumps are normally expanded from BF16,
    # so their low 16 bits are already zero; the general conversion keeps the
    # fixture writer correct for an ordinary FP32 dump as well.
    rounding_bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding_bias) >> np.uint32(16)).astype("<u2")


def write_array(
    output_dir: Path,
    file_name: str,
    value: np.ndarray,
    logical_dtype: str,
) -> dict[str, Any]:
    path = output_dir / file_name
    value.tofile(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "file": file_name,
        "shape": list(value.shape),
        "dtype": logical_dtype,
        "bytes": path.stat().st_size,
        "sha256": digest,
    }


def load_optional_json(path: Path | None) -> Any:
    if path is None:
        return None
    return json.loads(path.read_text())


def main() -> None:
    args = parse_args()
    if args.action_horizon <= 0 or args.action_dim <= 0:
        raise ValueError("action horizon and dimension must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.input, allow_pickle=False) as archive:
        input_ids = integral_u32(require_array(archive, "input_ids"), "input_ids")
        attention = integral_u32(
            require_array(archive, "attention_mask"), "attention_mask"
        )
        if attention.max() > 1:
            raise ValueError("attention_mask must contain only zero or one")
        attention = attention.astype("u1")
        grids = integral_u32(
            require_array(archive, "image_grid_thw"), "image_grid_thw"
        )
        pixels_source = require_array(archive, "pixel_values")
        state_source = require_array(archive, "state")
        embodiment = integral_u32(
            require_array(archive, "embodiment_id"), "embodiment_id"
        )
        noise_source = (
            np.asarray(archive["initial_noise"])
            if "initial_noise" in archive
            else None
        )

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"input_ids must be [1,tokens], got {input_ids.shape}")
    if attention.shape != input_ids.shape:
        raise ValueError("attention_mask shape does not match input_ids")
    if grids.ndim != 2 or grids.shape[1] != 3:
        raise ValueError(f"image_grid_thw must be [images,3], got {grids.shape}")
    if pixels_source.ndim != 2:
        raise ValueError(f"pixel_values must be rank two, got {pixels_source.shape}")
    if state_source.ndim != 3 or state_source.shape[0] != 1:
        raise ValueError(f"state must be [1,history,width], got {state_source.shape}")
    if embodiment.size != 1:
        raise ValueError(f"embodiment_id must have one element, got {embodiment.shape}")
    expected_noise_shape = (1, args.action_horizon, args.action_dim)
    if noise_source is not None and noise_source.shape != expected_noise_shape:
        raise ValueError(
            f"initial_noise must be {expected_noise_shape}, got {noise_source.shape}"
        )
    if args.noise_kind == "seeded-normal" and noise_source is None:
        raise ValueError("seeded-normal requires initial_noise in the input NPZ")
    if args.noise_kind == "seeded-normal" and args.noise_seed is None:
        raise ValueError("seeded-normal requires --noise-seed")
    if args.noise_kind == "fixed-zero" and noise_source is not None:
        if np.any(np.asarray(noise_source, dtype=np.float32) != 0):
            raise ValueError("fixed-zero metadata disagrees with non-zero initial_noise")

    pixels = f32_to_bf16_bits(pixels_source, "pixel_values")
    state = f32_to_bf16_bits(state_source, "state")
    noise = (
        np.zeros(expected_noise_shape, dtype="<u2")
        if noise_source is None
        else f32_to_bf16_bits(noise_source, "initial_noise")
    )

    tensors = {
        "pixel_values": write_array(
            args.output_dir, "pixel_values.bf16", pixels, "bfloat16"
        ),
        "image_grid_thw": write_array(
            args.output_dir, "image_grid_thw.u32", grids, "uint32"
        ),
        "token_ids": write_array(
            args.output_dir, "token_ids.u32", input_ids, "uint32"
        ),
        "attention_mask": write_array(
            args.output_dir, "attention_mask.u8", attention, "uint8"
        ),
        "state": write_array(args.output_dir, "state.bf16", state, "bfloat16"),
        "noise": write_array(args.output_dir, "initial_noise.bf16", noise, "bfloat16"),
    }
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "fixture": args.fixture,
        "source_npz": str(args.input),
        "source": load_optional_json(args.metadata),
        "embodiment_id": int(embodiment.reshape(-1)[0]),
        "tensors": tensors,
        "noise": {
            "kind": args.noise_kind,
            "seed": args.noise_seed,
            "source": (
                "input NPZ" if noise_source is not None else "fixture-writer default"
            ),
            "reason": "deterministic cross-runtime comparison",
        },
    }
    processor_report = load_optional_json(args.processor_report)
    if processor_report is not None:
        manifest["processor_report"] = processor_report

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), **manifest}, indent=2))


if __name__ == "__main__":
    main()
