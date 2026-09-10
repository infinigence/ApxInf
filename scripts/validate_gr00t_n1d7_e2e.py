#!/usr/bin/env python3
"""Validate the raw-observation-to-action GR00T N1.7 pipeline.

The script deliberately keeps model-family preprocessing outside ApxInf's
common Rust interfaces. It runs the pinned NVIDIA processor on one or two raw
LIBERO RGB camera streams, a language prompt and named robot-state values,
converts that exact processor output into the repository's versioned fixture
format, and invokes the persistent-shape ApxInf BF16 CUDA Graph benchmark.

The report uses the same processor/Model-Core component boundary as NVIDIA.
Its primary E2E statistic adds samples with the same iteration index and then
reports their percentiles, matching NVIDIA's pinned benchmark calculation; an
independent-median sum is retained as an explicitly labelled legacy field.
Neither value is presented as one contiguous wall-clock measurement. Optional
NVIDIA reference generation and elementwise comparison close the numerical
correctness loop.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor


EXPECTED_SOURCE_REVISION = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
LIBERO_STATE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--backbone", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--wrist-image", type=Path)
    parser.add_argument(
        "--views",
        type=int,
        choices=(1, 2),
        default=2,
        help=(
            "number of physical LIBERO camera streams; one view keeps only "
            "the checkpoint's first video modality, matching FlashRT's "
            "public N1.7 one-view fixture rule"
        ),
    )
    parser.add_argument(
        "--fixture",
        help="versioned fixture name (derived from --views when omitted)",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--state",
        action="append",
        default=[],
        metavar="KEY=VALUE[,VALUE...]",
        help=(
            "raw state vector; widths are read from checkpoint statistics and "
            "omitted LIBERO keys default to zero"
        ),
    )
    parser.add_argument(
        "--run-nvidia-reference",
        action="store_true",
        help="also run the pinned NVIDIA model and compare final actions",
    )
    return parser.parse_args()


def source_revision(source_dir: Path) -> str:
    return subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={source_dir}",
            "-C",
            str(source_dir),
            "rev-parse",
            "HEAD",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def state_widths(checkpoint: Path) -> dict[str, int]:
    statistics_path = checkpoint / "statistics.json"
    statistics = json.loads(statistics_path.read_text())
    libero_state = statistics["libero_sim"]["state"]
    widths: dict[str, int] = {}
    for key in LIBERO_STATE_KEYS:
        width = len(libero_state[key]["q01"])
        if width <= 0:
            raise ValueError(f"checkpoint reports an empty state group for {key!r}")
        widths[key] = width
    return widths


def parse_state(
    entries: list[str], widths: dict[str, int]
) -> dict[str, list[float]]:
    values = {key: [0.0] * width for key, width in widths.items()}
    for entry in entries:
        key, separator, raw_values = entry.partition("=")
        if not separator or key not in values:
            raise ValueError(
                f"invalid --state {entry!r}; expected one of "
                + ", ".join(f"{name}=VALUE[,VALUE...]" for name in LIBERO_STATE_KEYS)
            )
        parsed = [float(value) for value in raw_values.split(",")]
        if len(parsed) != widths[key]:
            raise ValueError(
                f"state {key!r} requires {widths[key]} value(s), got {len(parsed)}"
            )
        if not all(np.isfinite(value) for value in parsed):
            raise ValueError(f"state {key!r} must be finite")
        values[key] = parsed
    return values


def summarize(samples: list[float]) -> dict[str, Any]:
    if not samples:
        raise ValueError("latency sample list must not be empty")
    ordered = sorted(samples)
    p90_index = min(len(ordered) - 1, int(np.ceil(0.9 * len(ordered))) - 1)
    p95_index = min(len(ordered) - 1, int(np.ceil(0.95 * len(ordered))) - 1)
    return {
        "samples": samples,
        "minimum": min(samples),
        "median": statistics.median(samples),
        "p90": ordered[p90_index],
        "p95": ordered[p95_index],
        "mean": statistics.fmean(samples),
        "maximum": max(samples),
    }


def recursive_bf16(value: Any) -> Any:
    if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
        return value.to(dtype=torch.bfloat16)
    if isinstance(value, dict) or hasattr(value, "items"):
        return {key: recursive_bf16(item) for key, item in value.items()}
    if isinstance(value, list):
        return [recursive_bf16(item) for item in value]
    return value


def numpy_tensor(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"processor output {name!r} is not a torch.Tensor")
    if torch.is_floating_point(value):
        return value.detach().float().cpu().contiguous().numpy()
    return value.detach().cpu().contiguous().numpy()


def assert_deterministic(first: dict[str, Any], second: dict[str, Any]) -> None:
    for name in sorted(first):
        left = first[name]
        right = second[name]
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            if not torch.equal(left, right):
                raise RuntimeError(
                    f"processor output {name!r} changed across identical eval calls"
                )


def run_checked(command: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(command, check=True, env=env)


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    revision = source_revision(args.source_dir)
    if revision != EXPECTED_SOURCE_REVISION:
        raise RuntimeError(
            f"Isaac-GR00T source revision is {revision}, expected "
            f"{EXPECTED_SOURCE_REVISION}"
        )
    for path in (args.checkpoint, args.backbone, args.source_dir):
        if not path.is_dir():
            raise FileNotFoundError(path)
    input_files = [args.image, args.engine]
    if args.views == 2:
        if args.wrist_image is None:
            raise ValueError("--wrist-image is required when --views=2")
        input_files.append(args.wrist_image)
    elif args.wrist_image is not None:
        input_files.append(args.wrist_image)
    for path in input_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    sys.path.insert(0, str(args.source_dir.resolve()))
    # Importing the package registers the custom processor with Transformers.
    import gr00t.model  # noqa: F401
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import MessageType, VLAStepData

    processor_dir = (
        args.checkpoint / "processor"
        if (args.checkpoint / "processor").is_dir()
        and not (args.checkpoint / "processor_config.json").exists()
        else args.checkpoint
    )
    processor = AutoProcessor.from_pretrained(
        processor_dir,
        model_name=str(args.backbone.resolve()),
        local_files_only=True,
        trust_remote_code=True,
        transformers_loading_kwargs={
            "local_files_only": True,
            "trust_remote_code": True,
        },
    )
    processor.eval()
    embodiment = EmbodimentTag.resolve("libero_sim")
    configs = copy.deepcopy(processor.get_modality_configs()[embodiment.value])
    configured_video_keys = list(configs["video"].modality_keys)
    if len(configured_video_keys) < args.views:
        raise RuntimeError(
            f"checkpoint exposes only {len(configured_video_keys)} video "
            f"modalities, cannot select {args.views}"
        )
    # FlashRT's public capture_aux_multi.py applies the same first-N truncation
    # before preprocessing. Keeping that rule here makes the one-view Model
    # Core row directly shape-aligned with its published LIBERO benchmark.
    selected_video_keys = configured_video_keys[: args.views]
    configs["video"].modality_keys = selected_video_keys
    language_key = configs["language"].modality_keys[0]
    state_values = parse_state(args.state, state_widths(args.checkpoint))
    image_paths = {"image": args.image}
    if args.wrist_image is not None:
        image_paths["wrist_image"] = args.wrist_image
    missing_images = [key for key in selected_video_keys if key not in image_paths]
    if missing_images:
        raise ValueError(
            "no raw image path was supplied for selected video modalities "
            f"{missing_images}"
        )
    raw_images = {
        key: np.asarray(Image.open(image_paths[key]).convert("RGB"), dtype=np.uint8)[
            None
        ]
        for key in selected_video_keys
    }
    raw_states = {
        key: np.asarray([state_values[key]], dtype=np.float32)
        for key in LIBERO_STATE_KEYS
    }
    step = VLAStepData(
        images=raw_images,
        states=raw_states,
        actions={},
        text=args.prompt,
        embodiment=embodiment,
    )
    messages = [{"type": MessageType.EPISODE_STEP.value, "content": step}]

    def process() -> dict[str, Any]:
        processed = processor(messages)
        return recursive_bf16(processor.collator([processed])["inputs"])

    for _ in range(args.warmup):
        process()
    processor_samples: list[float] = []
    inputs: dict[str, Any] | None = None
    for _ in range(args.iterations):
        start = time.perf_counter()
        inputs = process()
        processor_samples.append((time.perf_counter() - start) * 1_000.0)
    assert inputs is not None
    assert_deterministic(inputs, process())

    required = {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
        "state",
        "embodiment_id",
    }
    missing = sorted(required.difference(inputs))
    if missing:
        raise RuntimeError(f"processor output is missing {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    processor_npz = args.output_dir / "processor-output.npz"
    np.savez(
        processor_npz,
        **{name: numpy_tensor(inputs[name], name) for name in sorted(required)},
    )
    processor_latency = summarize(processor_samples)
    processor_report = args.output_dir / "processor-benchmark.json"
    processor_report.write_text(
        json.dumps(
            {
                "schema": "apxinf.gr00t-n1.7.processor-benchmark.v1",
                "source_revision": revision,
                "checkpoint": str(args.checkpoint.resolve()),
                "raw_input": {
                    "configured_video_keys": configured_video_keys,
                    "selected_video_keys": selected_video_keys,
                    "physical_camera_count": len(raw_images),
                    "images": {
                        key: str(image_paths[key].resolve()) for key in raw_images
                    },
                    "image_sha256": {
                        key: sha256(image_paths[key]) for key in raw_images
                    },
                    "prompt": args.prompt,
                    "state": state_values,
                },
                "model_boundary": {
                    "input_ids_shape": list(inputs["input_ids"].shape),
                    "pixel_values_shape": list(inputs["pixel_values"].shape),
                    "image_grid_thw": numpy_tensor(
                        inputs["image_grid_thw"], "image_grid_thw"
                    )
                    .astype(np.int64)
                    .tolist(),
                    "state_shape": list(inputs["state"].shape),
                },
                "timing_boundary": "VLAStepData through processor and collator",
                "latency_ms": processor_latency,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    script_dir = Path(__file__).resolve().parent
    fixture_dir = args.output_dir / "fixture"
    fixture_name = args.fixture or f"raw-libero-{args.views}-view-e2e-v1"
    run_checked(
        [
            sys.executable,
            str(script_dir / "prepare_gr00t_n1d7_fixture.py"),
            "--input",
            str(processor_npz),
            "--output-dir",
            str(fixture_dir),
            "--fixture",
            fixture_name,
            "--processor-report",
            str(processor_report),
        ]
    )

    apxinf_report = args.output_dir / "apxinf-action.json"
    run_checked(
        [
            str(args.engine),
            str(args.checkpoint),
            str(args.backbone),
            str(fixture_dir),
            str(args.device),
            str(args.warmup),
            str(args.iterations),
            "graph",
            str(apxinf_report),
        ]
    )

    report: dict[str, Any] = {
        "schema": "apxinf.gr00t-n1.7.raw-to-action-validation.v1",
        "source_revision": revision,
        "checkpoint": str(args.checkpoint.resolve()),
        "processor_input_npz_sha256": sha256(processor_npz),
        "processor_latency_ms": processor_latency,
        "apxinf_report": str(apxinf_report),
        "timing_note": (
            "processor and Model Core are measured separately; "
            "e2e_component_sample_sum_latency_ms pairs their samples before "
            "summarizing, matching NVIDIA's statistic. "
            "e2e_component_median_sum_ms retains the independently summed "
            "medians for backward compatibility. Neither is a contiguous timer"
        ),
    }
    apxinf = json.loads(apxinf_report.read_text())
    model_core_samples = [
        float(value) for value in apxinf["model_core_latency_ms"].get("samples", [])
    ]
    if model_core_samples:
        model_core_latency = summarize(model_core_samples)
    else:
        model_core_latency = dict(apxinf["model_core_latency_ms"])
    report["model_core_latency_ms"] = model_core_latency
    report["e2e_component_median_sum_ms"] = (
        processor_latency["median"]
        + model_core_latency["median"]
    )
    if model_core_samples:
        if len(model_core_samples) != len(processor_samples):
            raise RuntimeError(
                "processor and Model Core sample counts differ: "
                f"{len(processor_samples)} versus {len(model_core_samples)}"
            )
        report["e2e_component_sample_sum_latency_ms"] = summarize(
            [
                processor_ms + model_core_ms
                for processor_ms, model_core_ms in zip(
                    processor_samples, model_core_samples, strict=True
                )
            ]
        )

    if args.run_nvidia_reference:
        reference = args.output_dir / "nvidia-reference.npz"
        parity = args.output_dir / "parity.json"
        reference_env = os.environ.copy()
        reference_env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
        run_checked(
            [
                sys.executable,
                str(script_dir / "gr00t_n1d7_reference_dump.py"),
                "--checkpoint",
                str(args.checkpoint),
                "--backbone",
                str(args.backbone),
                "--source-dir",
                str(args.source_dir),
                "--input-npz",
                str(processor_npz),
                "--output",
                str(reference),
                "--device",
                f"cuda:{args.device}",
                "--fixture",
                fixture_name,
            ],
            env=reference_env,
        )
        comparison = subprocess.run(
            [
                sys.executable,
                str(script_dir / "compare_gr00t_n1d7.py"),
                "--reference",
                str(reference),
                "--apxinf",
                str(apxinf_report),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        parity.write_text(comparison.stdout)
        report["nvidia_reference"] = str(reference)
        report["parity_report"] = str(parity)
        report["parity"] = json.loads(parity.read_text())

    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
