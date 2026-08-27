#!/usr/bin/env python3
"""Build and validate a portable Qwen3.5 selective-oracle job bundle.

Manifest-only mode never loads checkpoint weights and never invents golden
values. An explicit runner may populate only the artifacts declared by the
bundle; hashes are recorded after all files and metadata validate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import struct
from typing import Any, Iterable, Sequence


MODEL_VOCAB_SIZE = 248_320
EOS_TOKEN_IDS = [248_046, 248_044]
EXPECTED_REVISION = "63768c10df38c0395e12ef49edac1bd539eaeeea"
DEFAULT_INPUT_IDS = list(range(1, 9))
ALLOWED_STAGES = {
    "embedding",
    "layer_hidden",
    "gdn_state",
    "kv_state",
    "logits",
    "tokens",
}
LAYER_STAGES = {"layer_hidden", "gdn_state", "kv_state"}
CONTROL_FILE_NAMES = (
    "input-manifest.json",
    "selection.json",
    "generation.json",
    "golden-schema.json",
    "artifact-manifest.json",
)


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"failed to read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON root in {path} must be an object")
    return value


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get("text_config")
    if nested is None:
        return config
    if not isinstance(nested, dict):
        raise ValueError("config.json text_config must be an object")
    return nested


def _required_positive_int(config: dict[str, Any], name: str) -> int:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"config field {name} must be a positive integer")
    return value


def _normalize_input(path: Path | None) -> dict[str, Any]:
    raw = (
        {"schema": "apxinf.oracle-input.v1", "input_ids": DEFAULT_INPUT_IDS}
        if path is None
        else read_json(path)
    )
    input_ids = raw.get("input_ids")
    if not isinstance(input_ids, list) or not input_ids:
        raise ValueError("input manifest input_ids must be a non-empty array")
    normalized_ids = []
    for index, token in enumerate(input_ids):
        if isinstance(token, bool) or not isinstance(token, int):
            raise ValueError(f"input token at index {index} must be an integer")
        if token < 0 or token >= MODEL_VOCAB_SIZE:
            raise ValueError(
                f"input token at index {index} must be in [0,{MODEL_VOCAB_SIZE})"
            )
        normalized_ids.append(token)
    return {
        "schema": "apxinf.oracle-input.v1",
        "input_ids": normalized_ids,
    }


def _artifact(
    file_name: str,
    schema_ref: str,
    dtype: str,
    shape: list[Any],
    **metadata: Any,
) -> dict[str, Any]:
    return {
        "file": file_name,
        "schema_ref": schema_ref,
        "dtype": dtype,
        "shape": shape,
        "status": "pending",
        "sha256": None,
        **metadata,
    }


def _build_artifacts(
    layers: list[int], stages: list[str], model: dict[str, Any]
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    if "embedding" in stages:
        artifacts.append(
            _artifact(
                "embedding.f32.bin",
                "embedding",
                "F32",
                ["prompt_tokens", model["hidden_size"]],
            )
        )
    for layer in layers:
        prefix = f"layer-{layer:03d}"
        if "layer_hidden" in stages:
            artifacts.append(
                _artifact(
                    f"{prefix}-hidden.f32.bin",
                    "layer_hidden",
                    "F32",
                    ["trajectory_tokens", model["hidden_size"]],
                    layer=layer,
                )
            )
        layer_type = model["layer_types"][layer]
        if "gdn_state" in stages and layer_type == "linear_attention":
            artifacts.append(
                _artifact(
                    f"{prefix}-gdn-state.f32.bin",
                    "gdn_state",
                    "F32",
                    ["runner_resolved"],
                    layer=layer,
                    required_dimensions={
                        name: model[name]
                        for name in (
                            "linear_conv_kernel_dim",
                            "linear_key_head_dim",
                            "linear_num_key_heads",
                            "linear_num_value_heads",
                            "linear_value_head_dim",
                        )
                        if name in model
                    },
                )
            )
        if "kv_state" in stages and layer_type == "full_attention":
            for component in ("key", "value"):
                artifacts.append(
                    _artifact(
                        f"{prefix}-kv-{component}.f32.bin",
                        f"kv_{component}",
                        "F32",
                        [
                            1,
                            model["num_key_value_heads"],
                            "trajectory_tokens",
                            model["head_dim"],
                        ],
                        layer=layer,
                    )
                )
    if "logits" in stages:
        artifacts.append(
            _artifact(
                "logits.f32.bin",
                "logits",
                "F32",
                ["completion_tokens", MODEL_VOCAB_SIZE],
            )
        )
    if "tokens" in stages:
        artifacts.append(
            _artifact(
                "output-tokens.json",
                "tokens",
                "json",
                ["completion_tokens"],
            )
        )
    return sorted(artifacts, key=lambda item: item["file"])


def build_job(
    model_dir: Path | str,
    revision: str,
    layers: Iterable[int],
    stages: Iterable[str],
    input_manifest: Path | str | None,
    max_new_tokens: int,
) -> dict[str, Any]:
    model_path = Path(model_dir).resolve()
    if not model_path.is_dir():
        raise ValueError(f"model directory does not exist: {model_path}")
    if not revision or not revision.strip():
        raise ValueError("model revision must not be empty")
    if revision.strip() != EXPECTED_REVISION:
        raise ValueError(
            f"model revision must be frozen revision {EXPECTED_REVISION}, got {revision.strip()}"
        )
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise ValueError("max_new_tokens must be a positive integer")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")

    config_path = model_path / "config.json"
    generation_path = model_path / "generation_config.json"
    config = read_json(config_path)
    if config.get("model_type") != "qwen3_5":
        raise ValueError("config.json model_type must be qwen3_5")
    text = _text_config(config)
    generation_config = read_json(generation_path)
    vocab_size = _required_positive_int(text, "vocab_size")
    if vocab_size != MODEL_VOCAB_SIZE:
        raise ValueError(
            f"model config vocab_size must be {MODEL_VOCAB_SIZE}, got {vocab_size}"
        )
    eos_ids = generation_config.get("eos_token_id")
    if eos_ids != EOS_TOKEN_IDS:
        raise ValueError(
            f"generation_config eos_token_id must be {EOS_TOKEN_IDS}, got {eos_ids}"
        )

    number_of_layers = _required_positive_int(text, "num_hidden_layers")
    normalized_layers = sorted(set(layers))
    if any(
        isinstance(layer, bool)
        or not isinstance(layer, int)
        or layer < 0
        or layer >= number_of_layers
        for layer in normalized_layers
    ):
        raise ValueError(
            f"layer selection must be integers in [0,{number_of_layers})"
        )
    normalized_stages = sorted(set(stages))
    if normalized_layers and not normalized_stages:
        normalized_stages = ["layer_hidden"]
    invalid_stages = sorted(set(normalized_stages) - ALLOWED_STAGES)
    if invalid_stages:
        raise ValueError(f"unknown oracle stage: {', '.join(invalid_stages)}")
    if not normalized_layers and not normalized_stages:
        raise ValueError("oracle selection must include layers or stages")
    if set(normalized_stages) & LAYER_STAGES and not normalized_layers:
        raise ValueError("layer-specific oracle stages require at least one layer")

    input_path = Path(input_manifest) if input_manifest is not None else None
    normalized_input = _normalize_input(input_path)
    input_bytes = canonical_bytes(normalized_input)
    model = {
        "repo_id": "cyankiwi/Qwen3.8-27B-AWQ-INT4",
        "revision": revision.strip(),
        "model_dir": str(model_path),
        "config_sha256": sha256_file(config_path),
        "generation_config_sha256": sha256_file(generation_path),
        "vocab_size": vocab_size,
        "hidden_size": _required_positive_int(text, "hidden_size"),
        "num_hidden_layers": number_of_layers,
        "num_attention_heads": _required_positive_int(text, "num_attention_heads"),
        "num_key_value_heads": _required_positive_int(text, "num_key_value_heads"),
        "head_dim": _required_positive_int(text, "head_dim"),
    }
    layer_types = text.get("layer_types")
    if (
        not isinstance(layer_types, list)
        or len(layer_types) != number_of_layers
        or any(
            layer_type not in {"linear_attention", "full_attention"}
            for layer_type in layer_types
        )
    ):
        raise ValueError(
            "config field layer_types must describe every layer as linear_attention or full_attention"
        )
    model["layer_types"] = layer_types
    if "gdn_state" in normalized_stages and not any(
        layer_types[layer] == "linear_attention" for layer in normalized_layers
    ):
        raise ValueError("gdn_state selection contains no linear_attention layer")
    if "kv_state" in normalized_stages and not any(
        layer_types[layer] == "full_attention" for layer in normalized_layers
    ):
        raise ValueError("kv_state selection contains no full_attention layer")
    for name in (
        "linear_conv_kernel_dim",
        "linear_key_head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_value_head_dim",
    ):
        value = text.get(name)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            model[name] = value

    source_path = Path(__file__).resolve()
    contract_path = source_path.parents[2] / "benchmarks/qwen38_4090/evaluation/contract-v1.json"
    contract_sha256 = sha256_file(contract_path) if contract_path.is_file() else None
    selection = {"layers": normalized_layers, "stages": normalized_stages}
    generation = {
        "temperature": 0,
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "ignore_eos": False,
        "eos_token_ids": EOS_TOKEN_IDS,
        "trajectory": "record every generated token through EOS or budget",
    }
    job = {
        "schema": "apxinf.oracle-job.v1",
        "generator": {
            "entrypoint": "tools/oracle/generate_golden.py",
            "sha256": sha256_file(source_path),
        },
        "contract_sha256": contract_sha256,
        "model": model,
        "input": {**normalized_input, "sha256": sha256_bytes(input_bytes)},
        "selection": selection,
        "generation": generation,
    }
    job["artifacts"] = _build_artifacts(normalized_layers, normalized_stages, model)
    identity = {
        "generator_sha256": job["generator"]["sha256"],
        "contract_sha256": job["contract_sha256"],
        "model_revision": model["revision"],
        "config_sha256": model["config_sha256"],
        "generation_config_sha256": model["generation_config_sha256"],
        "input_manifest_sha256": job["input"]["sha256"],
        "selection": selection,
        "generation": generation,
        "schema": job["schema"],
    }
    job["artifact_identity_sha256"] = sha256_bytes(canonical_bytes(identity))
    return job


def _schema_document(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "apxinf.oracle-golden-schema.v1",
        "status": "declarations_only",
        "numeric_endianness": "little",
        "artifacts": [
            {
                key: value
                for key, value in artifact.items()
                if key not in {"status", "sha256"}
            }
            for artifact in job["artifacts"]
        ],
        "tokens_schema": {
            "schema": "apxinf.oracle-tokens.v1",
            "required": ["output_token_ids", "decoded_text"],
            "output_token_ids": {
                "type": "array[integer]",
                "range": [0, MODEL_VOCAB_SIZE - 1],
            },
            "decoded_text": {"type": "string", "special_tokens_skipped": True},
        },
        "artifact_report_schema": {
            "schema": "apxinf.oracle-artifact-report.v1",
            "generation_required": ["completion_tokens", "stop_reason"],
            "logits_generation_required": ["top1_top2_margin"],
            "gdn_record_required": ["required_dimensions"],
        },
        "comparison": {
            "tokens": "exact",
            "hidden_state_logits": {"absolute_tolerance": 0.01, "relative_tolerance": 0.01},
            "top1": "exact with top1/top2 margin recorded by runner",
        },
    }


def _write_canonical(path: Path, value: object) -> None:
    path.write_bytes(canonical_bytes(value))


def write_manifest_bundle(job: dict[str, Any], output_dir: Path | str) -> Path:
    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"output directory must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    artifact_dir = root / "artifacts"
    artifact_dir.mkdir()

    input_document = {
        "schema": job["input"]["schema"],
        "input_ids": job["input"]["input_ids"],
    }
    artifact_manifest = {
        "schema": "apxinf.oracle-artifact-manifest.v1",
        "status": "pending",
        "artifact_identity_sha256": job["artifact_identity_sha256"],
        "artifacts": job["artifacts"],
        "runner_report_sha256": None,
    }
    documents = {
        "input-manifest.json": input_document,
        "selection.json": {
            "schema": "apxinf.oracle-selection.v1",
            **job["selection"],
        },
        "generation.json": {
            "schema": "apxinf.oracle-generation.v1",
            **job["generation"],
        },
        "golden-schema.json": _schema_document(job),
        "artifact-manifest.json": artifact_manifest,
    }
    for name, document in documents.items():
        _write_canonical(root / name, document)
    control_hashes = {name: sha256_file(root / name) for name in CONTROL_FILE_NAMES}
    job_manifest = {
        "schema": "apxinf.oracle-job-manifest.v1",
        "status": "manifest_only",
        "artifact_identity_sha256": job["artifact_identity_sha256"],
        "generator": job["generator"],
        "contract_sha256": job["contract_sha256"],
        "model": job["model"],
        "input_manifest_sha256": job["input"]["sha256"],
        "selection": job["selection"],
        "control_files": control_hashes,
        "artifact_directory": "artifacts",
    }
    manifest_path = root / "job-manifest.json"
    _write_canonical(manifest_path, job_manifest)
    return manifest_path


def _validate_shape(
    actual: object,
    expected: list[Any],
    symbols: dict[str, int],
    file_name: str,
) -> list[int]:
    if not isinstance(actual, list) or not actual:
        raise ValueError(f"artifact {file_name} shape must be a non-empty array")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in actual
    ):
        raise ValueError(f"artifact {file_name} shape must contain positive integers")
    if expected != ["runner_resolved"] and len(actual) != len(expected):
        raise ValueError(
            f"artifact {file_name} rank mismatch: expected {len(expected)}, got {len(actual)}"
        )
    for expected_dimension, actual_dimension in zip(expected, actual):
        if isinstance(expected_dimension, int) and expected_dimension != actual_dimension:
            raise ValueError(
                f"artifact {file_name} shape mismatch: expected dimension {expected_dimension}, got {actual_dimension}"
            )
        if isinstance(expected_dimension, str):
            expected_value = symbols.get(expected_dimension)
            if expected_value is not None and expected_value != actual_dimension:
                raise ValueError(
                    f"artifact {file_name} shape mismatch: {expected_dimension} expected {expected_value}, got {actual_dimension}"
                )
    return actual


def _validate_f32_artifact(path: Path, shape: list[int], file_name: str) -> None:
    elements = 1
    for dimension in shape:
        elements *= dimension
    expected_bytes = elements * 4
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"artifact {file_name} byte length does not match F32 shape")
    with path.open("rb") as handle:
        for index in range(elements):
            value = struct.unpack("<f", handle.read(4))[0]
            if not math.isfinite(value):
                raise ValueError(f"artifact {file_name} contains non-finite value at {index}")


def _validate_token_artifact(
    path: Path,
    shape: list[int],
    completion_tokens: int,
    stop_reason: str,
) -> None:
    value = read_json(path)
    if value.get("schema") != "apxinf.oracle-tokens.v1":
        raise ValueError("output-tokens.json has an invalid schema")
    token_ids = value.get("output_token_ids")
    if not isinstance(token_ids, list):
        raise ValueError("output-tokens.json output_token_ids must be an array")
    if not token_ids or len(token_ids) != completion_tokens or shape != [len(token_ids)]:
        raise ValueError("output-tokens.json token count does not match metadata")
    if any(
        isinstance(token, bool)
        or not isinstance(token, int)
        or token < 0
        or token >= MODEL_VOCAB_SIZE
        for token in token_ids
    ):
        raise ValueError("output-tokens.json contains an invalid model token")
    if not isinstance(value.get("decoded_text"), str):
        raise ValueError("output-tokens.json decoded_text must be a string")
    eos_positions = [index for index, token in enumerate(token_ids) if token in EOS_TOKEN_IDS]
    if stop_reason == "eos":
        if not eos_positions or eos_positions[0] != len(token_ids) - 1:
            raise ValueError("EOS stop must end with exactly one EOS token")
    elif eos_positions:
        if eos_positions[0] != len(token_ids) - 1:
            raise ValueError("output-tokens.json contains tokens after EOS")


def _validate_report_generation(
    report: dict[str, Any], max_new_tokens: int, requires_margin: bool
) -> tuple[int, str]:
    report_generation = report.get("generation")
    if not isinstance(report_generation, dict):
        raise ValueError("runner artifact report generation metadata is required")
    completion_tokens = report_generation.get("completion_tokens")
    stop_reason = report_generation.get("stop_reason")
    if (
        isinstance(completion_tokens, bool)
        or not isinstance(completion_tokens, int)
        or completion_tokens <= 0
        or completion_tokens > max_new_tokens
    ):
        raise ValueError("runner completion_tokens must be within the requested budget")
    if stop_reason not in {"eos", "budget"}:
        raise ValueError("runner stop_reason must be eos or budget")
    if stop_reason == "budget" and completion_tokens != max_new_tokens:
        raise ValueError("output tokens must consume the budget or end with EOS")
    if requires_margin:
        margins = report_generation.get("top1_top2_margin")
        if (
            not isinstance(margins, list)
            or len(margins) != completion_tokens
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
                for value in margins
            )
        ):
            raise ValueError(
                "runner logits report requires one finite nonnegative top1/top2 margin per completion token"
            )
    return completion_tokens, stop_reason


def run_runner(
    bundle_dir: Path | str, runner: Sequence[str], runner_args: Sequence[str]
) -> None:
    root = Path(bundle_dir).resolve()
    if not runner:
        raise ValueError("runner command must not be empty")
    job_path = root / "job-manifest.json"
    job = read_json(job_path)
    artifact_manifest_path = root / "artifact-manifest.json"
    pending = read_json(artifact_manifest_path)
    if job.get("status") != "manifest_only" or pending.get("status") != "pending":
        raise ValueError("oracle bundle is not in manifest-only pending state")
    model = job.get("model")
    if not isinstance(model, dict):
        raise ValueError("job manifest model metadata is invalid")
    model_dir = Path(str(model.get("model_dir", "")))
    metadata_files = {
        "config.json": model.get("config_sha256"),
        "generation_config.json": model.get("generation_config_sha256"),
    }
    for name, expected_sha in metadata_files.items():
        path = model_dir / name
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha:
            raise ValueError(f"model metadata SHA256 mismatch for {name}")
    contract_path = Path(__file__).resolve().parents[2] / "benchmarks/qwen38_4090/evaluation/contract-v1.json"
    if (
        contract_path.is_symlink()
        or not contract_path.is_file()
        or sha256_file(contract_path) != job.get("contract_sha256")
    ):
        raise ValueError("evaluation contract SHA256 mismatch")
    control_hashes = job.get("control_files")
    if not isinstance(control_hashes, dict) or set(control_hashes) != set(
        CONTROL_FILE_NAMES
    ):
        raise ValueError("job manifest control file set is invalid")
    for name, expected_sha in control_hashes.items():
        control_path = root / name
        if control_path.is_symlink() or not control_path.is_file():
            raise ValueError(f"control file {name} must be a regular file")
        if sha256_file(control_path) != expected_sha:
            raise ValueError(f"control file {name} SHA256 does not match job manifest")
    job_manifest_sha = sha256_file(job_path)
    artifact_dir = root / str(job.get("artifact_directory", "artifacts"))
    if artifact_dir.is_symlink() or not artifact_dir.is_dir() or any(artifact_dir.iterdir()):
        raise ValueError("oracle artifact directory must exist and be empty before runner execution")

    environment = os.environ.copy()
    environment["APXINF_ORACLE_JOB_MANIFEST"] = str(job_path)
    environment["APXINF_ORACLE_OUTPUT_DIR"] = str(artifact_dir)
    completed = subprocess.run(
        [*runner, *runner_args],
        cwd=root,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"oracle runner exited with status {completed.returncode}")

    if artifact_dir.is_symlink() or not artifact_dir.is_dir():
        raise ValueError("runner replaced artifact directory")

    if sha256_file(job_path) != job_manifest_sha:
        raise ValueError("runner modified job-manifest.json")
    for name, expected_sha in control_hashes.items():
        control_path = root / name
        if control_path.is_symlink() or not control_path.is_file():
            raise ValueError(f"runner replaced control file {name}")
        if sha256_file(control_path) != expected_sha:
            raise ValueError(f"runner modified control file {name}")

    report_path = artifact_dir / "artifact-report.json"
    expected = {item["file"]: item for item in pending.get("artifacts", [])}
    entries = list(artifact_dir.iterdir())
    invalid_entries = sorted(
        path.name for path in entries if path.is_symlink() or not path.is_file()
    )
    if invalid_entries:
        raise ValueError(
            f"runner artifact directory contains non-regular entries: {invalid_entries}"
        )
    actual_names = {path.name for path in entries}
    expected_names = set(expected) | {"artifact-report.json"}
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing or extra:
        raise ValueError(f"runner artifact set mismatch: missing={missing}, extra={extra}")

    report = read_json(report_path)
    if report.get("schema") != "apxinf.oracle-artifact-report.v1":
        raise ValueError("runner artifact report has an invalid schema")
    records = report.get("artifacts")
    if not isinstance(records, list):
        raise ValueError("runner artifact report artifacts must be an array")
    max_new_tokens = int(read_json(root / "generation.json")["max_new_tokens"])
    completion_tokens, stop_reason = _validate_report_generation(
        report,
        max_new_tokens,
        any(item.get("schema_ref") == "logits" for item in pending.get("artifacts", [])),
    )
    by_name: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("file"), str):
            raise ValueError("runner artifact report contains an invalid record")
        name = record["file"]
        if name in by_name:
            raise ValueError(f"runner artifact report duplicates {name}")
        by_name[name] = record
    if set(by_name) != set(expected):
        raise ValueError("runner artifact report file set does not match declaration")

    completed_artifacts = []
    generation = read_json(root / "generation.json")
    prompt_tokens = len(read_json(root / "input-manifest.json")["input_ids"])
    symbols = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "trajectory_tokens": prompt_tokens + completion_tokens,
    }
    for name, declaration in sorted(expected.items()):
        record = by_name[name]
        if record.get("schema_ref") != declaration["schema_ref"]:
            raise ValueError(f"artifact {name} schema_ref does not match declaration")
        if record.get("dtype") != declaration["dtype"]:
            raise ValueError(f"artifact {name} dtype does not match declaration")
        if declaration["schema_ref"] == "gdn_state" and record.get(
            "required_dimensions"
        ) != declaration.get("required_dimensions"):
            raise ValueError(f"artifact {name} GDN dimensions do not match declaration")
        shape = _validate_shape(record.get("shape"), declaration["shape"], symbols, name)
        artifact_path = artifact_dir / name
        actual_sha = sha256_file(artifact_path)
        if record.get("sha256") != actual_sha:
            raise ValueError(f"artifact {name} SHA256 does not match report")
        if declaration["dtype"] == "json":
            _validate_token_artifact(
                artifact_path, shape, completion_tokens, stop_reason
            )
        else:
            _validate_f32_artifact(artifact_path, shape, name)
        completed_artifacts.append(
            {
                **declaration,
                "status": "complete",
                "shape": shape,
                "sha256": actual_sha,
                "bytes": artifact_path.stat().st_size,
            }
        )

    complete_manifest = {
        **pending,
        "status": "complete",
        "artifacts": completed_artifacts,
        "runner_report_sha256": sha256_file(report_path),
    }
    _write_canonical(artifact_manifest_path, complete_manifest)
    job["status"] = "complete"
    job["control_files"]["artifact-manifest.json"] = sha256_file(
        artifact_manifest_path
    )
    job["artifact_report_sha256"] = sha256_file(report_path)
    _write_canonical(job_path, job)


def _parse_layers(values: Sequence[str] | None) -> list[int]:
    layers: list[int] = []
    for value in values or []:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start_text, end_text = part.split("-", 1)
                start, end = int(start_text), int(end_text)
                if end < start:
                    raise ValueError(f"invalid descending layer range {part}")
                layers.extend(range(start, end + 1))
            else:
                layers.append(int(part))
    return layers


def _parse_stages(values: Sequence[str] | None) -> list[str]:
    return [part.strip() for value in values or [] for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--layers", action="append")
    parser.add_argument("--stages", action="append")
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--runner")
    parser.add_argument("--runner-arg", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        job = build_job(
            arguments.model_dir,
            arguments.revision,
            _parse_layers(arguments.layers),
            _parse_stages(arguments.stages),
            arguments.input_manifest,
            arguments.max_new_tokens,
        )
        write_manifest_bundle(job, arguments.output_dir)
        if arguments.runner:
            run_runner(arguments.output_dir, [arguments.runner], arguments.runner_arg)
    except (ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
