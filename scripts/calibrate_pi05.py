#!/usr/bin/env python3
"""Build a self-describing PI0.5 static-FP8 profile from business Observations."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys
from typing import Callable, Optional

import numpy as np


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_APXINF_PKG = _REPO_ROOT / "python" / "apxinf"
if _APXINF_PKG.is_dir() and str(_APXINF_PKG) not in sys.path:
    sys.path.insert(0, str(_APXINF_PKG))

SCHEMA = "apxinf.pi05.fp8-calibration.v1"


def _progress(message: str) -> None:
    print(f"[calibration] {message}", file=sys.stderr, flush=True)


if __package__:
    from .pi05_calibration_data import (
        load_npz_observations,
        load_observation_manifest,
    )
else:
    from pi05_calibration_data import (
        load_npz_observations,
        load_observation_manifest,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        usage=(
            "%(prog)s --model-dir MODEL_DIR "
            "(--manifest OBSERVATIONS.jsonl | --input-dir DIR | SOURCE) "
            "[--output PATH]"
        ),
        description=(
            "Generate a checkpoint-bound PI0.5 FP8 profile from representative "
            "business Observations. Manifest and NPZ sources are supported. "
            "Capturing observations out of a live simulator belongs to whoever "
            "owns the environment: 'apxinf-robo capture-libero' writes an NPZ "
            "directory that --input-dir reads."
        ),
    )
    parser.add_argument("--model-dir", required=True, type=pathlib.Path)
    parser.add_argument("--checkpoint", type=pathlib.Path)
    parser.add_argument(
        "--input",
        action="append",
        type=pathlib.Path,
        default=[],
        help="Observation NPZ containing configured image keys, prompt, and optional state",
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
    parser.add_argument("--discrete-state", action="store_true")
    parser.add_argument("--state-norm-key", default="state")
    parser.add_argument("--tokenizer-path", type=pathlib.Path)
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
        for value in (
            args.input,
            args.input_dir,
            args.manifest,
            args.zero_fixture,
        )
    )
    if modes != 1:
        raise ValueError(
            "pass exactly one calibration source: --manifest, --input-dir, "
            "one or more --input files, or --zero-fixture"
        )
    if not np.isfinite(args.margin) or args.margin < 1.0:
        raise ValueError("--margin must be finite and >= 1")
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
        (args.input or args.input_dir or args.manifest)
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


def _observation_identity(observations: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for observation in observations:
        for name in sorted(observation):
            digest.update(name.encode())
            digest.update(b"\0")
            value = observation[name]
            if isinstance(value, str):
                digest.update(b"str\0")
                digest.update(value.encode())
            else:
                array = np.ascontiguousarray(np.asarray(value))
                digest.update(array.dtype.str.encode())
                digest.update(json.dumps(array.shape).encode())
                digest.update(array.tobytes())
            digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def resolve_observations(args, policy):
    """Resolve one CLI source into replayable public Observations plus identity."""
    if args.zero_fixture:
        observation = {
            key: np.zeros((policy.model.image_size, policy.model.image_size, 3), np.uint8)
            for key in policy.image_keys
        }
        observation[policy.prompt_key] = "synthetic calibration fixture"
        if policy.discrete_state:
            tokenize = policy.input_pipeline["tokenize"]
            state_normalizer = getattr(tokenize, "state_normalizer", None)
            state_width = getattr(state_normalizer, "width", policy.action_dim)
            observation[policy.state_key] = np.zeros(state_width, np.float32)
        return (observation,), "synthetic:zero-observation-v1"

    if args.manifest is not None:
        observations = load_observation_manifest(
            args.manifest,
            image_keys=policy.image_keys,
            prompt_key=policy.prompt_key,
            state_key=policy.state_key,
        )
        return observations, _observation_identity(observations)

    paths = tuple(args.input)
    if args.input_dir is not None:
        paths = tuple(sorted(args.input_dir.glob("*.npz")))
    observations = load_npz_observations(paths)
    return observations, calibration_data_identity(paths, args.data_id)


def load_observations(args, policy) -> Iterable[Mapping[str, object]]:
    """Compatibility iterator over the source resolved by the calibration job."""
    observations, _ = resolve_observations(args, policy)
    yield from observations


def deterministic_noise(policy, seed: int, sample_index: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([seed, sample_index]))
    return np.ascontiguousarray(
        rng.standard_normal((policy.model.action_horizon, policy.model.action_dim)),
        dtype=np.float32,
    )


def _hash_files(paths: Iterable[pathlib.Path], root: pathlib.Path) -> str:
    digest = hashlib.sha256()
    canonical = []
    for path in paths:
        try:
            relative = path.relative_to(root).as_posix()
            encoded = relative.encode("utf-8", errors="strict")
        except (UnicodeEncodeError, ValueError) as error:
            raise ValueError(f"checkpoint path is not canonical UTF-8: {path}") from error
        canonical.append((encoded, path))
    for relative, path in sorted(canonical, key=lambda item: item[0]):
        digest.update(relative)
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_index_files(index_path: pathlib.Path) -> list[pathlib.Path]:
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"checkpoint index has no weight_map: {index_path}")
    names = set()
    for value in weight_map.values():
        if not isinstance(value, str):
            raise ValueError(f"checkpoint index has a non-string shard: {index_path}")
        relative = pathlib.PurePosixPath(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"checkpoint index has an unsafe shard path: {value}")
        names.add(relative)
    return [index_path.parent / name for name in names]


def checkpoint_identity(checkpoint: pathlib.Path) -> str:
    if checkpoint.is_dir():
        root = checkpoint
        index = checkpoint / "model.safetensors.index.json"
        model = checkpoint / "model.safetensors"
        if index.is_file():
            files = _checkpoint_index_files(index)
        elif model.is_file():
            files = [model]
        else:
            files = list(checkpoint.rglob("*.safetensors"))
    elif checkpoint.name.endswith(".index.json"):
        files = _checkpoint_index_files(checkpoint)
        root = checkpoint.parent
    else:
        files = [checkpoint]
        root = checkpoint.parent
    if not files or any(not path.is_file() for path in files):
        raise ValueError(f"cannot resolve checkpoint files from {checkpoint}")
    return "sha256:" + _hash_files(files, root)


def calibration_data_identity(paths: Iterable[pathlib.Path], explicit) -> str:
    if explicit:
        return explicit
    paths = [path.resolve() for path in paths]
    if not paths:
        return "synthetic:zero-observation-v1"
    common = pathlib.Path(os.path.commonpath(paths))
    if common.is_file():
        common = common.parent
    return "sha256:" + _hash_files(paths, common)


def source_revision(explicit=None) -> str:
    if explicit is not None:
        if not explicit.strip() or explicit == "unknown":
            raise ValueError("--source-revision must identify a real commit or release")
        return explicit
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
        ).strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet"], cwd=_REPO_ROOT, check=False
        ).returncode != 0
        return revision + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(
            "cannot determine source revision; pass --source-revision explicitly"
        ) from error


def write_profile(output: pathlib.Path, document, *, force: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if force else "x"
    try:
        with output.open(mode) as handle:
            json.dump(document, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise ValueError(
            f"output already exists (pass --force to replace it): {output}"
        ) from error


@dataclass(frozen=True)
class CalibrationJobResult:
    """Artifact produced by one completed PI0.5 calibration job."""

    document: Mapping[str, object] | None
    output: pathlib.Path | None


class Pi05CalibrationJob:
    """Turn model-native Observations into one PI0.5 calibration profile."""

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
        if bootstrap:
            plan = replace(
                plan,
                minimum_amax={
                    "vision.patch_input": 1.0 / args.margin,
                    "action.input": 5.0 / args.margin,
                },
            )
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
        from apxinf import Pi05Policy

        policy_factory = Pi05Policy.from_pretrained
    policy_options = {
        "checkpoint": checkpoint,
        "device": args.device,
        "precision": "bf16",
        "seed": args.seed,
        "prompt_key": args.prompt_key,
        "state_key": args.state_key,
        "discrete_state": args.discrete_state,
        "state_norm_key": args.state_norm_key,
    }
    if args.image_key:
        policy_options["image_keys"] = tuple(args.image_key)
        policy_options["num_views"] = args.num_views or len(args.image_key)
    if args.tokenizer_path is not None:
        policy_options["tokenizer_path"] = args.tokenizer_path
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
        return Pi05CalibrationJob(
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
