#!/usr/bin/env python3
"""Collect GR00T BF16 activation maxima from LeRobot data or explicit fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/apxinf"))
sys.path.insert(0, str(ROOT / "scripts" / "gr00t"))

from apxinf.calibration import (  # noqa: E402
    Fp8ExecutionPlan,
    QuantizationSpec,
    QuantizedOperator,
    build_calibration_document,
)
from apxinf.policies.impls.gr00t import Gr00tPolicy  # noqa: E402
from formats import decode_float_array, load_fixture  # noqa: E402
from calibration_data import SCHEMA, file_hash  # noqa: E402


DEFAULT_CHECKPOINT = ROOT.parent / "models" / "gr00t" / "libero_10"
DEFAULT_TACTICS = (
    ROOT
    / "devlocal"
    / "s5000-gr00t"
    / "run-20261112-installed-env"
    / "results"
    / "s5000-tactics.json"
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="musa:1")
    parser.add_argument("--tactics", type=Path, default=DEFAULT_TACTICS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--prepared-data", type=Path)
    parser.add_argument("--processor-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gr00t-source", type=Path, default=ROOT.parent / "Isaac-GR00T")
    parser.add_argument("--episodes-per-task", type=int, default=3)
    parser.add_argument("--frames-per-episode", type=int, default=3)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--margin", type=float, default=1.2)
    parser.add_argument("fixtures", nargs="*", type=Path)
    args = parser.parse_args(argv)
    if sum(bool(source) for source in (args.dataset, args.prepared_data, args.fixtures)) != 1:
        parser.error("choose exactly one: --dataset, --prepared-data, or explicit fixtures")
    if args.seed < 0 or min(args.episodes_per_task, args.frames_per_episode) < 1:
        parser.error("seed must be non-negative and sample counts positive")
    if not np.isfinite(args.margin) or args.margin < 1:
        parser.error("margin must be finite and >= 1")
    if args.output.exists() or args.output.with_suffix(".samples.jsonl").exists():
        parser.error("output already exists; choose a new output path")
    return args


def fixture_identity(fixtures: list[Path]) -> str:
    digest = hashlib.sha256()
    for fixture in fixtures:
        manifest = (fixture / "manifest.json").read_bytes()
        digest.update(manifest)
        for entry in json.loads(manifest)["tensors"].values():
            digest.update(file_hash(fixture / entry["file"]).encode())
    return "sha256:" + digest.hexdigest()


def main() -> None:
    args = parse_args()
    fixtures = [path.expanduser().resolve() for path in args.fixtures]
    checkpoint = args.checkpoint.expanduser().resolve()
    backbone = checkpoint / "assets" / "cosmos"
    checkpoint_identity = Gr00tPolicy.checkpoint_identity(checkpoint, backbone)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prepared = args.prepared_data
    if args.dataset:
        prepared = args.output.with_suffix(".inputs")
        subprocess.run([
            str(args.processor_python), str(Path(__file__).with_name("calibration_data.py")),
            "--dataset", str(args.dataset), "--checkpoint", str(checkpoint),
            "--gr00t-source", str(args.gr00t_source), "--output", str(prepared),
            "--episodes-per-task", str(args.episodes_per_task),
            "--frames-per-episode", str(args.frames_per_episode), "--seed", str(args.seed),
        ], check=True)
    manifest = None
    if prepared:
        manifest_path = prepared / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if (manifest["schema"] != SCHEMA or manifest["checkpoint"] != checkpoint_identity
                or manifest["seed"] != args.seed or manifest["kind"] != "representative"
                or not manifest["samples"]):
            raise ValueError("prepared observations do not match schema/checkpoint/seed/source")
        samples = manifest["samples"]
        for sample in samples:
            if file_hash(prepared / sample["file"]) != sample["sha256"]:
                raise ValueError("prepared observation checksum mismatch")
        data_identity = file_hash(manifest_path)
    else:
        samples = [{"fixture": str(fixture)} for fixture in fixtures]
        data_identity = fixture_identity(fixtures)

    import apxinf_py

    model = apxinf_py.ModelRunner.load(
        "gr00t",
        str(checkpoint),
        device=args.device,
        precision="bf16",
        assets={"backbone": str(backbone)},
        tactics=str(args.tactics.expanduser().resolve()),
        execution="eager",
    )
    sites = model._calibration_plan()
    plan = QuantizationSpec(model_family="gr00t").plan_for(
        Fp8ExecutionPlan(
            operators=tuple(
                QuantizedOperator(stable_id=site.removesuffix(".input"), kind="linear")
                for site in sites
            )
        )
    )

    records: dict[str, float] = {}
    sample_path = args.output.with_suffix(".samples.jsonl")
    for sample_index, sample in enumerate(samples):
        if prepared:
            with np.load(prepared / sample["file"], allow_pickle=False) as archive:
                arrays = dict(archive)
            embodiment_id = int(arrays["embodiment_id"])
        else:
            fixture_manifest, arrays = load_fixture(Path(sample["fixture"]))
            embodiment_id = fixture_manifest["embodiment_id"]
        current = model._calibrate_preprocessed(
            decode_float_array(arrays["pixel_values"], "pixel_values"),
            arrays["image_grid_thw"],
            arrays["token_ids"],
            arrays["attention_mask"],
            decode_float_array(arrays["state"], "state"),
            embodiment_id,
            decode_float_array(arrays["noise"], "noise"),
        )
        merge_amax(records, current, sites)
        with sample_path.open("a") as stream:
            stream.write(json.dumps({"sample_index": sample_index, **sample,
                                     "amax": dict(current)}, sort_keys=True) + "\n")
        print(f"sample={sample_index + 1}/{len(samples)} sites={len(current)}", flush=True)

    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    document = build_calibration_document(
        records,
        plan=plan,
        checkpoint=checkpoint_identity,
        data_identity=data_identity,
        source_revision=revision,
        device={"requested": args.device, "host": platform.node()},
        margin=args.margin,
        seed=args.seed,
        bootstrap=False,
        sample_count=len(samples),
    )
    if not prepared:
        document["calibration_data"].update(kind="fixed-input-benchmark", production=False)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    args.output.with_suffix(".provenance.json").write_text(json.dumps({
        "input_manifest": str(prepared / "manifest.json") if prepared else None,
        "profile_sha256": file_hash(args.output), "samples_sha256": file_hash(sample_path),
        "collector_sha256": file_hash(Path(__file__)),
        "data_tool_sha256": file_hash(Path(__file__).with_name("calibration_data.py")),
        "tactics": str(args.tactics), "tactics_sha256": file_hash(args.tactics),
        "quality_scope": "representative calibration, not held-out task-success validation",
    }, indent=2) + "\n")

    print(f"output={args.output}")
    print("selected scales:")
    for name, entry in document["scales"].items():
        if ".fc2.input" in name or name.endswith(".down.input"):
            print(f"  {name} amax={entry['amax']:.8g} scale={entry['scale']:.8g}")


def merge_amax(records, current, sites):
    if set(current) != set(sites):
        raise ValueError("sample calibration site coverage mismatch")
    if any(not np.isfinite(value) or value < 0 for value in current.values()):
        raise ValueError("sample contains invalid amax")
    for site, maximum in current.items():
        records[site] = max(records.get(site, 0.0), float(maximum))


if __name__ == "__main__":
    main()
