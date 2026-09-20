#!/usr/bin/env python3
"""Build a self-describing π0-FAST static-FP8 profile from business Observations."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
import pathlib
import platform
import sys
from typing import Callable, Optional

import numpy as np


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_APXINF_PKG = _REPO_ROOT / "python" / "apxinf"
if _APXINF_PKG.is_dir() and str(_APXINF_PKG) not in sys.path:
    sys.path.insert(0, str(_APXINF_PKG))

SCHEMA = "apxinf.pi0fast.fp8-calibration.v1"


def _progress(message: str) -> None:
    print(f"[calibration] {message}", file=sys.stderr, flush=True)


if __package__:
    from .pi0fast_calibration_data import (
        load_libero_observations,
        load_npz_observations,
        load_observation_manifest,
        task_stratified_indices,
    )
    from .pi0fast_calibration_profile import (
        calibration_data_identity,
        checkpoint_identity,
        observation_identity,
        source_revision,
        write_profile,
    )
else:
    from pi0fast_calibration_data import (
        load_libero_observations,
        load_npz_observations,
        load_observation_manifest,
        task_stratified_indices,
    )
    from pi0fast_calibration_profile import (
        calibration_data_identity,
        checkpoint_identity,
        observation_identity,
        source_revision,
        write_profile,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        usage=(
            "%(prog)s --model-dir MODEL_DIR "
            "(--libero-suite libero_10 | --manifest OBSERVATIONS.jsonl "
            "| --input-dir DIR | SOURCE) [--output PATH]"
        ),
        description=(
            "Generate a checkpoint-bound π0-FAST FP8 profile from representative "
            "business Observations. π0-FAST decodes action tokens with a greedy "
            "argmax, so a profile is the only thing standing between E4M3 weights "
            "and a token stream that never terminates: a uniform scale is known "
            "not to preserve it. Native LIBERO, manifest, and NPZ sources are "
            "supported; --libero-suite drives the simulator here and needs LIBERO "
            "installed, while capturing frames elsewhere and passing --input-dir "
            "an NPZ directory is the right seam for a deployment that owns its "
            "own environment."
        ),
    )
    parser.add_argument("--model-dir", required=True, type=pathlib.Path)
    parser.add_argument("--checkpoint", type=pathlib.Path)
    parser.add_argument(
        "--input",
        action="append",
        type=pathlib.Path,
        default=[],
        help="Observation NPZ containing configured image keys, prompt, and state",
    )
    parser.add_argument(
        "--input-dir",
        type=pathlib.Path,
        help="directory of replayable Observation *.npz files",
    )
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        help="JSONL Observation manifest; image fields are paths relative to this file",
    )
    parser.add_argument(
        "--libero-suite",
        choices=(
            "libero_10",
            "libero_90",
            "libero_spatial",
            "libero_object",
            "libero_goal",
        ),
        help="capture native simulator observations from this LIBERO task suite",
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="task-balanced LIBERO sample count (default: one initial state per task)",
    )
    parser.add_argument(
        "--zero-fixture",
        action="store_true",
        help="explicitly labeled non-production synthetic Observation",
    )
    parser.add_argument("--data-id", help="stable representative-dataset identifier")
    parser.add_argument(
        "--source-revision",
        help="source commit/version (required when calibration runs outside a Git checkout)",
    )
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-key", action="append", default=[])
    parser.add_argument("--num-views", type=int)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--state-key", default="observation/state")
    parser.add_argument("--tokenizer-path", type=pathlib.Path)
    parser.add_argument("--fast-tokenizer-path", type=pathlib.Path)
    parser.add_argument("--action-horizon", type=int)
    parser.add_argument("--margin", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def validate_args(args):
    if not args.model_dir.is_dir():
        raise ValueError(f"model directory does not exist: {args.model_dir}")
    modes = sum(
        bool(value)
        for value in (args.input, args.input_dir, args.manifest, args.libero_suite)
    )
    if args.zero_fixture:
        if modes:
            raise ValueError("--zero-fixture cannot be combined with an observation source")
    elif modes != 1:
        raise ValueError(
            "pass exactly one calibration source: --manifest, --libero-suite, --input-dir, "
            "or one or more --input files (or --zero-fixture)"
        )
    if not np.isfinite(args.margin) or args.margin < 1.0:
        raise ValueError("--margin must be finite and >= 1")
    if args.samples is not None and args.samples < 1:
        raise ValueError("--samples must be positive")
    if args.samples is not None and args.libero_suite is None:
        raise ValueError("--samples applies only to --libero-suite")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.num_views is not None and args.num_views < 1:
        raise ValueError("--num-views must be positive")
    if args.num_views is not None and not args.image_key:
        raise ValueError("--num-views requires one --image-key per view")
    if args.image_key and args.num_views is not None and len(args.image_key) != args.num_views:
        raise ValueError("--num-views must equal the number of --image-key values")
    if args.action_horizon is not None and args.action_horizon < 1:
        raise ValueError("--action-horizon must be positive")
    if args.data_id is not None and not args.data_id.strip():
        raise ValueError("--data-id must not be empty")
    if (
        (args.input or args.input_dir or args.manifest or args.libero_suite)
        and args.data_id is not None
        and args.data_id.startswith("synthetic:")
    ):
        raise ValueError("representative --input data cannot use a synthetic: --data-id")
    missing = [path for path in args.input if not path.is_file()]
    if missing:
        raise ValueError(f"calibration input does not exist: {missing[0]}")
    if args.input_dir is not None:
        if not args.input_dir.is_dir():
            raise ValueError(f"calibration input directory does not exist: {args.input_dir}")
        if not any(args.input_dir.glob("*.npz")):
            raise ValueError(f"calibration input directory has no *.npz files: {args.input_dir}")
    if args.manifest is not None and not args.manifest.is_file():
        raise ValueError(f"calibration manifest does not exist: {args.manifest}")
    checkpoint = args.checkpoint or args.model_dir / "model.safetensors"
    if not checkpoint.exists():
        raise ValueError(f"checkpoint does not exist: {checkpoint}")
    output = args.output or args.model_dir / "calibration.json"
    if output.exists() and not args.force:
        raise ValueError(f"output already exists (pass --force to replace it): {output}")
    return output, checkpoint


def state_finger_joints(policy) -> int:
    """How many of LIBERO's mirrored finger joints this checkpoint's state carries.

    The width is a checkpoint property — the width of its own normalization
    statistics — so it is read from the policy rather than guessed. 7 keeps the
    collapsed openpi convention, 8 keeps both joints (LeRobot's LIBERO
    environment, which π0-FAST was trained on).
    """
    width = int(policy.state_mean.size)
    if width == 7:
        return 1
    if width == 8:
        return 2
    raise ValueError(
        f"π0-FAST state statistics are {width}-wide; LIBERO calibration "
        "supports the 7- and 8-wide conventions only"
    )


def resolve_observations(args, policy):
    """Resolve one CLI source into replayable public Observations plus identity."""
    if args.zero_fixture:
        image_size = int(policy.model.image_size)
        observation = {
            key: np.zeros((image_size, image_size, 3), np.uint8) for key in policy.image_keys
        }
        observation[policy.prompt_key] = "synthetic calibration fixture"
        observation[policy.state_key] = np.zeros(policy.state_mean.size, np.float32)
        return (observation,), "synthetic:zero-observation-v1"

    if args.libero_suite is not None:
        observations = load_libero_observations(
            args.libero_suite,
            image_keys=policy.image_keys,
            sample_count=args.samples,
            seed=args.seed,
            prompt_key=policy.prompt_key,
            state_key=policy.state_key,
            finger_joints=state_finger_joints(policy),
            progress=_progress,
        )
        return observations, observation_identity(observations)

    if args.manifest is not None:
        observations = load_observation_manifest(
            args.manifest,
            image_keys=policy.image_keys,
            prompt_key=policy.prompt_key,
            state_key=policy.state_key,
        )
        return observations, observation_identity(observations)

    paths = tuple(args.input)
    if args.input_dir is not None:
        paths = tuple(sorted(args.input_dir.glob("*.npz")))
    observations = load_npz_observations(paths)
    return observations, calibration_data_identity(paths, args.data_id)


def load_observations(args, policy) -> Iterable[Mapping[str, object]]:
    """Compatibility iterator over the source resolved by the calibration job."""
    observations, _ = resolve_observations(args, policy)
    yield from observations


@dataclass(frozen=True)
class CalibrationJobResult:
    """Artifact produced by one completed π0-FAST calibration job."""

    document: Mapping[str, object] | None
    output: pathlib.Path | None


class Pi0FastCalibrationJob:
    """Turn model-native Observations into one π0-FAST calibration profile."""

    def __init__(
        self,
        args,
        *,
        policy,
        output: pathlib.Path,
        checkpoint: pathlib.Path,
        progress: Optional[Callable[[str], None]] = None,
    ):
        self.args = args
        self.policy = policy
        self.output = output
        self.checkpoint = checkpoint
        self.progress = progress or (lambda _message: None)

    def run(
        self,
        observations: Iterable[Mapping[str, object]],
        *,
        data_identity: str,
        bootstrap: bool = False,
    ) -> CalibrationJobResult:
        args = self.args
        from apxinf.calibration import CalibrationRunner

        self.progress("Resolving the FP8 calibration execution plan...")
        plan = self.policy.calibration_plan()
        self.progress(
            "Hashing the checkpoint for profile identity "
            "(this reads all weight files)..."
        )
        checkpoint = checkpoint_identity(self.checkpoint)
        self.progress("Checkpoint identity complete.")
        runner = CalibrationRunner(
            self.policy,
            plan,
            checkpoint=checkpoint,
            data_identity=data_identity,
            source_revision=source_revision(args.source_revision),
            device={"requested": args.device, "host": platform.platform()},
            margin=args.margin,
            seed=args.seed,
            bootstrap=bootstrap,
        )
        sample_count = len(observations) if isinstance(observations, Sequence) else None
        count = f" over {sample_count} observation(s)" if sample_count is not None else ""
        self.progress(f"Running eager BF16 calibration{count}...")
        document = runner.run(observations)
        self.progress("Calibration sweep complete.")

        if document is None:
            return CalibrationJobResult(document=None, output=None)
        self.progress(f"Writing calibration profile to {self.output}...")
        write_profile(self.output, document, force=args.force)
        self.progress("Calibration profile written.")
        return CalibrationJobResult(document=document, output=self.output)


def _load_policy(args, checkpoint: pathlib.Path, policy_factory=None):
    if policy_factory is None:
        from apxinf import Pi0FastPolicy

        policy_factory = Pi0FastPolicy.from_pretrained
    policy_options = {
        "checkpoint": checkpoint,
        "device": args.device,
        # A profile is a statement about BF16 activations; the capture has to
        # come from the BF16 runtime, so calibration never runs the FP8 path.
        "precision": "bf16",
        "seed": args.seed,
        "prompt_key": args.prompt_key,
        "state_key": args.state_key,
    }
    if args.image_key:
        policy_options["image_keys"] = tuple(args.image_key)
        policy_options["num_views"] = args.num_views or len(args.image_key)
    elif args.num_views is not None:
        policy_options["num_views"] = args.num_views
    if args.tokenizer_path is not None:
        policy_options["tokenizer_path"] = args.tokenizer_path
    if args.fast_tokenizer_path is not None:
        policy_options["fast_tokenizer_path"] = args.fast_tokenizer_path
    if args.action_horizon is not None:
        policy_options["action_horizon"] = args.action_horizon
    return policy_factory(args.model_dir, **policy_options)


def run_from_args(args, *, policy_factory=None) -> CalibrationJobResult:
    """CLI adapter: resolve one storage source, then cross the job seam."""
    output, checkpoint = validate_args(args)
    _progress(f"Loading the BF16 model from {checkpoint}...")
    policy = _load_policy(args, checkpoint, policy_factory)
    _progress("BF16 model loaded.")
    try:
        _progress("Loading calibration observations...")
        observations, inferred_identity = resolve_observations(args, policy)
        _progress(f"Loaded {len(observations)} observation(s).")
        return Pi0FastCalibrationJob(
            args,
            policy=policy,
            output=output,
            checkpoint=checkpoint,
            progress=_progress,
        ).run(
            observations,
            data_identity=args.data_id or inferred_identity,
            bootstrap=args.zero_fixture,
        )
    finally:
        policy.close()


def main(argv=None):
    args = parse_args(argv)
    result = run_from_args(args)
    document = result.document
    if document is None:
        print("dynamic activation FP8 is calibration-free; no profile was generated")
        return
    print(
        f"wrote {len(document['scales'])} activation scales from "
        f"{document['calibration_data']['sample_count']} sample(s): {result.output}"
    )
    if args.zero_fixture:
        print("warning: synthetic profile is non-production; calibrate representative data")


if __name__ == "__main__":
    main()
