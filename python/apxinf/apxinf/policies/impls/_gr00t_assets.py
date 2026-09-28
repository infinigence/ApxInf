"""Local, relocatable Cosmos processor assets for GR00T checkpoints.

The v1 bundle contains configuration and processor resources only. Existing
explicit Cosmos snapshots without a manifest remain a supported legacy input.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Mapping


DEFAULT_SUBDIR = "assets/cosmos"
MANIFEST = "apxinf_assets.json"
SCHEMA = "apxinf.gr00t-assets.v1"

RESOURCE_NAMES = frozenset(
    {
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "special_tokens_map.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "processor_config.json",
        "chat_template.json",
        "chat_template.jinja",
        "generation_config.json",
    }
)
_REQUIRED_NAMES = frozenset(
    {"config.json", "tokenizer_config.json", "preprocessor_config.json"}
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _validate_resource_name(relative: str) -> None:
    if not isinstance(relative, str):
        raise ValueError("GR00T asset paths must be strings")
    parts = relative.split("/")
    if (
        not relative
        or "\\" in relative
        or "\0" in relative
        or any(part in ("", ".", "..") for part in parts)
    ):
        raise ValueError(f"invalid GR00T asset path: {relative!r}")
    try:
        relative.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"invalid UTF-8 GR00T asset path: {relative!r}") from error
    if relative not in RESOURCE_NAMES and not (
        len(parts) == 2
        and parts[0] == "chat_templates"
        and parts[1].endswith(".jinja")
    ):
        raise ValueError(f"unsupported GR00T asset path: {relative!r}")


def _validate_required_files(files: Mapping[str, str]) -> None:
    missing = _REQUIRED_NAMES - files.keys()
    if missing:
        raise ValueError(f"missing required GR00T assets: {sorted(missing)}")
    if "tokenizer.json" not in files and not {"vocab.json", "merges.txt"} <= files.keys():
        raise ValueError("GR00T assets require tokenizer.json or both vocab.json and merges.txt")


def _resource_names(root: Path) -> set[str]:
    names = {
        name
        for name in RESOURCE_NAMES
        if (root / name).exists() or (root / name).is_symlink()
    }
    templates = root / "chat_templates"
    if templates.exists() or templates.is_symlink():
        if not templates.is_dir():
            raise ValueError(f"GR00T chat_templates is not a directory: {templates}")
        names.update(
            f"chat_templates/{path.name}"
            for path in templates.iterdir()
            if path.name.endswith(".jinja")
        )
    return names


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate GR00T asset manifest key: {key!r}")
        result[key] = value
    return result


def verify_asset_manifest(root) -> dict[str, str]:
    """Verify a v1 asset bundle and return its relative-path/hash mapping."""
    root = Path(root)
    manifest = root / MANIFEST
    if manifest.is_symlink():
        raise ValueError(f"GR00T asset manifest must not be a symlink: {manifest}")
    if not manifest.is_file():
        raise FileNotFoundError(f"GR00T asset manifest is missing: {manifest}")
    with manifest.open(encoding="utf-8") as stream:
        document = json.load(stream, object_pairs_hook=_unique_json_object)
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError(f"unsupported GR00T asset manifest schema; expected {SCHEMA}")
    if document.keys() != {"schema", "files"}:
        raise ValueError("GR00T asset manifest accepts only schema and files fields")
    files = document.get("files")
    if not isinstance(files, dict):
        raise ValueError("GR00T asset manifest files must be an object")
    for relative, expected in files.items():
        _validate_resource_name(relative)
        if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
            raise ValueError(f"invalid lowercase SHA256 for GR00T asset {relative!r}")
    _validate_required_files(files)
    if (root / "chat_templates").is_symlink():
        raise ValueError("GR00T asset chat_templates directory must not be a symlink")
    extra = _resource_names(root) - files.keys()
    if extra:
        raise ValueError(f"unlisted GR00T asset files: {sorted(extra)}")
    for relative, expected in files.items():
        path = root / relative
        if path.is_symlink():
            raise ValueError(f"GR00T asset must not be a symlink: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"GR00T asset file is missing: {relative}")
        if _file_sha256(path) != expected:
            raise ValueError(f"GR00T asset SHA256 mismatch: {relative}")
    return dict(files)


def resolve_assets(model_dir, backbone=None) -> Path:
    """Resolve the prepared default bundle or a compatible explicit snapshot."""
    root = Path(backbone) if backbone is not None else Path(model_dir) / DEFAULT_SUBDIR
    manifest = root / MANIFEST
    if manifest.exists() or manifest.is_symlink():
        verify_asset_manifest(root)
    elif backbone is None:
        raise FileNotFoundError(
            f"GR00T local assets require {manifest}. Prepare them with "
            "Gr00tPolicy.prepare_assets(MODEL, COSMOS), "
            "or pass backbone= for an existing local Cosmos snapshot."
        )
    return root


def asset_identity(root) -> str:
    """Hash verified resource content independently of its directory location."""
    files = verify_asset_manifest(root)
    digest = hashlib.sha256(SCHEMA.encode("utf-8") + b"\0")
    for relative in sorted(files, key=lambda name: name.encode("utf-8")):
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(files[relative].encode("ascii") + b"\0")
    return f"sha256:{digest.hexdigest()}"


def prepare_assets(model_dir, source) -> Path:
    """Copy local processor/config resources into a new v1 checkpoint bundle.

    Source snapshot symlinks are followed and materialized as ordinary files.
    No weights are copied and an existing destination is never overwritten.
    """
    model_dir, source = Path(model_dir), Path(source)
    if not model_dir.is_dir():
        raise NotADirectoryError(f"GR00T checkpoint directory does not exist: {model_dir}")
    if not source.is_dir():
        raise NotADirectoryError(f"Cosmos source directory does not exist: {source}")
    destination = model_dir / DEFAULT_SUBDIR
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"GR00T asset destination already exists: {destination}")
    names = _resource_names(source)
    _validate_required_files(dict.fromkeys(names, ""))
    for relative in names:
        _validate_resource_name(relative)
        if not (source / relative).is_file():
            raise FileNotFoundError(f"Cosmos source resource is not a file: {relative}")
    destination.mkdir(parents=True, exist_ok=False)
    try:
        files = {}
        for relative in sorted(names, key=lambda name: name.encode("utf-8")):
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, target)
            files[relative] = _file_sha256(target)
        with (destination / MANIFEST).open("x", encoding="utf-8") as stream:
            json.dump({"schema": SCHEMA, "files": files}, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        verify_asset_manifest(destination)
    except BaseException:
        shutil.rmtree(destination)
        raise
    return destination
