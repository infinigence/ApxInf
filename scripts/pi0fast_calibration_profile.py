"""Profile identity and serialization for the π0-FAST calibration driver.

A π0-FAST FP8 profile binds itself to a checkpoint's byte content and to
the data it was measured on, and the Rust loader reproduces that binding
exactly or its identity check would be meaningless. The helpers here hash
files, name a dataset, resolve a source revision, and write the document
atomically.

The Rust side (``apxinf_model::pi0fast::checkpoint_identity``) reproduces
:func:`checkpoint_identity` byte for byte; ``tests/fixtures/checkpoint_identity``
pins both languages against the same expected digest.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import os
import pathlib
import subprocess
import sys

import numpy as np


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def observation_identity(observations: Sequence[Mapping[str, object]]) -> str:
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
