#!/usr/bin/env python3
"""One-shot, offline MLX-LM generation worker with a JSON-lines contract."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import stat
import sys
import time
from typing import NoReturn


REQUEST_FORMAT = "apxinf-mlx-generation-request-v1"
RECEIPT_FORMAT = "apxinf-mlx-generation-receipt-v1"
ERROR_FORMAT = "apxinf-mlx-generation-error-v1"
GREEDY_STRATEGY = "mlx-generate-step-argmax-v1"
MAX_REQUEST_BYTES = 1024 * 1024
MAX_CONFIG_BYTES = 2 * 1024 * 1024
MAX_OUTPUT_BYTES = 4 * 1024 * 1024
MAX_PROMPT_TOKENS = 131_072
MAX_GENERATED_TOKENS = 65_536
MAX_TOKEN_ID = 2**31 - 1
PINNED_PYTHON_VERSION = "3.14.3"
PINNED_PACKAGE_VERSIONS = {
    "huggingface-hub": "1.28.0",
    "mlx": "0.32.1",
    "mlx-lm": "0.31.3",
    "mlx-metal": "0.32.1",
    "numpy": "2.5.2",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "transformers": "5.15.1",
}


class WorkerError(ValueError):
    """A deterministic worker contract failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise WorkerError("invalid_arguments", message)


def _reject_constant(value: str) -> NoReturn:
    raise WorkerError("invalid_json", f"non-finite JSON number is not allowed: {value}")


def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise WorkerError("invalid_json", f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json(payload: bytes, label: str) -> dict[str, object]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise WorkerError("invalid_json", f"{label} must be UTF-8") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise WorkerError(
            "invalid_json",
            f"invalid {label} JSON at line {error.lineno} column {error.colno}",
        ) from error
    if type(value) is not dict:
        raise WorkerError("invalid_json", f"{label} root must be an object")
    return value


def _read_request() -> dict[str, object]:
    payload = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if len(payload) > MAX_REQUEST_BYTES:
        raise WorkerError(
            "invalid_request", f"request exceeds {MAX_REQUEST_BYTES} bytes"
        )
    if payload.endswith(b"\n"):
        payload = payload[:-1]
    if not payload:
        raise WorkerError("invalid_request", "request must not be empty")
    if b"\n" in payload or b"\r" in payload:
        raise WorkerError("invalid_request", "request must contain exactly one line")
    return _parse_json(payload, "request")


def _token_id(value: object, label: str) -> int:
    if type(value) is not int or not 0 <= value <= MAX_TOKEN_ID:
        raise WorkerError(
            "invalid_request", f"{label} must be an integer in [0, {MAX_TOKEN_ID}]"
        )
    return value


def _validate_request(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise WorkerError("invalid_request", "request must be an object")
    required = {"format", "prompt_token_ids", "max_tokens"}
    allowed = required | {"eos_token_id", "stop_on_eos"}
    observed = set(value)
    if not required <= observed or not observed <= allowed:
        raise WorkerError(
            "invalid_request",
            "request keys mismatch "
            f"(missing={sorted(required - observed)}, unknown={sorted(observed - allowed)})",
        )
    if value["format"] != REQUEST_FORMAT:
        raise WorkerError(
            "invalid_request", f"request.format must be {REQUEST_FORMAT!r}"
        )
    prompt = value["prompt_token_ids"]
    if type(prompt) is not list or not prompt:
        raise WorkerError(
            "invalid_request", "prompt_token_ids must be a non-empty array"
        )
    if len(prompt) > MAX_PROMPT_TOKENS:
        raise WorkerError(
            "invalid_request",
            f"prompt_token_ids exceeds {MAX_PROMPT_TOKENS} entries",
        )
    prompt_ids = [
        _token_id(token, f"prompt_token_ids[{index}]")
        for index, token in enumerate(prompt)
    ]
    max_tokens = value["max_tokens"]
    if (
        type(max_tokens) is not int
        or max_tokens < 0
        or max_tokens > MAX_GENERATED_TOKENS
    ):
        raise WorkerError(
            "invalid_request",
            f"max_tokens must be an integer in [0, {MAX_GENERATED_TOKENS}]",
        )
    stop_on_eos = value.get("stop_on_eos", True)
    if type(stop_on_eos) is not bool:
        raise WorkerError("invalid_request", "stop_on_eos must be a boolean")
    eos_token_id = value.get("eos_token_id")
    if "eos_token_id" in value:
        eos_token_id = _token_id(eos_token_id, "eos_token_id")
    return {
        "prompt_token_ids": prompt_ids,
        "max_tokens": max_tokens,
        "stop_on_eos": stop_on_eos,
        "eos_token_id": eos_token_id,
    }


def _read_model_config(model_dir_argument: str) -> tuple[Path, dict[str, object], str]:
    model_dir = Path(model_dir_argument)
    if not model_dir.is_absolute():
        raise WorkerError("invalid_model", "--model-dir must be an absolute path")
    try:
        directory_info = model_dir.lstat()
    except OSError as error:
        raise WorkerError(
            "invalid_model", f"cannot inspect model directory: {error}"
        ) from error
    if stat.S_ISLNK(directory_info.st_mode) or not stat.S_ISDIR(directory_info.st_mode):
        raise WorkerError(
            "invalid_model", "--model-dir must be an existing non-symlink directory"
        )
    model_dir = model_dir.resolve(strict=True)
    config_path = model_dir / "config.json"
    try:
        config_info = config_path.lstat()
    except OSError as error:
        raise WorkerError(
            "invalid_model", f"cannot inspect config.json: {error}"
        ) from error
    if stat.S_ISLNK(config_info.st_mode) or not stat.S_ISREG(config_info.st_mode):
        raise WorkerError(
            "invalid_model", "config.json must be a regular non-symlink file"
        )
    if config_info.st_size > MAX_CONFIG_BYTES:
        raise WorkerError(
            "invalid_model", f"config.json exceeds {MAX_CONFIG_BYTES} bytes"
        )
    try:
        payload = config_path.read_bytes()
    except OSError as error:
        raise WorkerError(
            "invalid_model", f"cannot read config.json: {error}"
        ) from error
    if len(payload) != config_info.st_size:
        raise WorkerError("invalid_model", "config.json changed while it was read")
    config = _parse_json(payload, "config.json")
    model_type = config.get("model_type")
    if (
        type(model_type) is not str
        or not model_type
        or model_type != model_type.strip()
    ):
        raise WorkerError(
            "invalid_model", "config.json model_type must be a non-empty string"
        )
    if config.get("model_file") is not None or config.get("auto_map") is not None:
        raise WorkerError("invalid_model", "model configuration requests remote code")
    return model_dir, config, hashlib.sha256(payload).hexdigest()


def _configured_quantization(config: dict[str, object]) -> object:
    for key in ("quantization", "quantization_config"):
        if key in config:
            return config[key]
    text_config = config.get("text_config")
    if type(text_config) is dict:
        for key in ("quantization", "quantization_config"):
            if key in text_config:
                return text_config[key]
    return None


def _normalise_eos_ids(value: object) -> list[int]:
    if value is None:
        return []
    values = value if type(value) in (list, tuple, set, frozenset) else [value]
    result: list[int] = []
    for index, token in enumerate(values):
        if type(token) is not int or not 0 <= token <= MAX_TOKEN_ID:
            raise WorkerError(
                "invalid_model", f"effective eos token {index} is not a valid token id"
            )
        if token not in result:
            result.append(token)
    return sorted(result)


def _config_eos_ids(config: dict[str, object]) -> list[int]:
    if "eos_token_id" in config:
        return _normalise_eos_ids(config["eos_token_id"])
    text_config = config.get("text_config")
    if type(text_config) is dict and "eos_token_id" in text_config:
        return _normalise_eos_ids(text_config["eos_token_id"])
    return []


def _package_version(distribution: str) -> str:
    try:
        version = metadata.version(distribution)
    except metadata.PackageNotFoundError as error:
        raise WorkerError(
            "dependency_unavailable", f"package metadata not found for {distribution}"
        ) from error
    if not version:
        raise WorkerError(
            "dependency_unavailable", f"package version is empty for {distribution}"
        )
    return version


def _pinned_toolchain_versions() -> dict[str, str]:
    python_version = platform.python_version()
    if python_version != PINNED_PYTHON_VERSION:
        raise WorkerError(
            "unsupported_toolchain",
            f"Python {python_version} does not match pinned {PINNED_PYTHON_VERSION}",
        )
    observed = {
        package: _package_version(package) for package in PINNED_PACKAGE_VERSIONS
    }
    mismatches = {
        package: {"expected": expected, "observed": observed[package]}
        for package, expected in PINNED_PACKAGE_VERSIONS.items()
        if observed[package] != expected
    }
    if mismatches:
        details = ", ".join(
            f"{package}={values['observed']} (expected {values['expected']})"
            for package, values in sorted(mismatches.items())
        )
        raise WorkerError(
            "unsupported_toolchain", f"package version mismatch: {details}"
        )
    return observed


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise WorkerError(
            "invalid_runtime", f"cannot hash runtime file {path}: {error}"
        ) from error
    return digest.hexdigest()


def _token_ids_sha256(token_ids: list[int]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _generate_receipt(
    request: dict[str, object],
    model_dir: Path,
    config: dict[str, object],
    config_sha256: str,
) -> dict[str, object]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    versions = _pinned_toolchain_versions()
    try:
        python_executable = Path(sys.executable).resolve(strict=True)
        runner = Path(__file__).resolve(strict=True)
    except OSError as error:
        raise WorkerError(
            "invalid_runtime", f"cannot resolve runtime identity: {error}"
        ) from error
    runtime_identity = {
        "python_version": platform.python_version(),
        "python_executable": str(python_executable),
        "python_executable_sha256": _file_sha256(python_executable),
        "runner": str(runner),
        "runner_sha256": _file_sha256(runner),
    }

    try:
        import mlx.core as mx
        import mlx_lm
        from mlx_lm.generate import generate_step
    except (ImportError, ModuleNotFoundError) as error:
        raise WorkerError(
            "dependency_unavailable", f"cannot import MLX runtime: {error}"
        ) from error

    mx.reset_peak_memory()
    load_start = time.perf_counter_ns()
    try:
        loaded = mlx_lm.load(
            str(model_dir),
            tokenizer_config={
                "local_files_only": True,
                "trust_remote_code": False,
            },
            lazy=False,
            return_config=True,
        )
    except Exception as error:
        raise WorkerError(
            "model_load_failed", f"MLX-LM model load failed: {error}"
        ) from error
    load_end = time.perf_counter_ns()
    if type(loaded) is not tuple or len(loaded) != 3:
        raise WorkerError(
            "model_load_failed", "MLX-LM load returned an unexpected value"
        )
    model, tokenizer, loaded_config = loaded
    if type(loaded_config) is not dict:
        raise WorkerError(
            "model_load_failed", "MLX-LM load did not return model config"
        )
    if loaded_config.get("model_type") != config["model_type"]:
        raise WorkerError(
            "model_load_failed", "loaded model_type differs from config.json"
        )

    requested_eos = request["eos_token_id"]
    if requested_eos is not None:
        eos_ids = [requested_eos]
    else:
        eos_ids = _normalise_eos_ids(getattr(tokenizer, "eos_token_ids", None))
        if not eos_ids:
            eos_ids = _config_eos_ids(config)
    if request["stop_on_eos"] and request["max_tokens"] > 0 and not eos_ids:
        raise WorkerError(
            "invalid_model", "stop_on_eos requires an effective EOS token id"
        )

    generated_ids: list[int] = []
    generation_start = time.perf_counter_ns()
    first_token_at: int | None = None
    stop_reason = "length"
    if request["max_tokens"] > 0:
        try:
            prompt = mx.array(request["prompt_token_ids"])

            # Bind the production strategy explicitly. The pipeline's prompt
            # chunking and one-token-ahead cache schedule are part of this
            # versioned contract and are covered by the model-level parity gate.
            def greedy_argmax(logprobs):
                return mx.argmax(logprobs, axis=-1)

            steps = generate_step(
                prompt,
                model,
                max_tokens=request["max_tokens"],
                sampler=greedy_argmax,
            )
            for raw_token, _logprobs in steps:
                observed_at = time.perf_counter_ns()
                if type(raw_token) is not int:
                    try:
                        raw_token = raw_token.item()
                    except (AttributeError, TypeError, ValueError) as error:
                        raise WorkerError(
                            "generation_failed", "MLX-LM produced a non-integer token"
                        ) from error
                token = _token_id(raw_token, "generated token")
                if first_token_at is None:
                    first_token_at = observed_at
                generated_ids.append(token)
                if request["stop_on_eos"] and token in eos_ids:
                    stop_reason = "eos"
                    break
                if len(generated_ids) == request["max_tokens"]:
                    break
            close_steps = getattr(steps, "close", None)
            if close_steps is not None:
                close_steps()
        except WorkerError:
            raise
        except Exception as error:
            raise WorkerError(
                "generation_failed", f"MLX-LM generation failed: {error}"
            ) from error
    generation_end = time.perf_counter_ns()
    if request["max_tokens"] > 0 and (not generated_ids or first_token_at is None):
        raise WorkerError("generation_failed", "MLX-LM produced no tokens")
    if stop_reason == "length" and len(generated_ids) != request["max_tokens"]:
        raise WorkerError(
            "generation_failed", "MLX-LM ended before max_tokens without EOS"
        )

    decode_tokens = max(0, len(generated_ids) - 1)
    decode_ns = max(1, generation_end - first_token_at) if decode_tokens else 0
    tpot_ms = decode_ns / decode_tokens / 1_000_000 if decode_tokens else 0.0
    tps = decode_tokens * 1_000_000_000 / decode_ns if decode_tokens else 0.0
    try:
        mx.synchronize()
        peak_memory = int(mx.get_peak_memory())
    except (RuntimeError, TypeError, ValueError, OverflowError) as error:
        raise WorkerError(
            "generation_failed", "MLX returned an invalid peak memory measurement"
        ) from error
    if peak_memory < 0:
        raise WorkerError(
            "generation_failed", "MLX returned a negative peak memory measurement"
        )

    return {
        "format": RECEIPT_FORMAT,
        "request": {
            "prompt_token_count": len(request["prompt_token_ids"]),
            "prompt_token_ids_sha256": _token_ids_sha256(request["prompt_token_ids"]),
            "max_tokens": request["max_tokens"],
            "stop_on_eos": request["stop_on_eos"],
            "greedy_strategy": GREEDY_STRATEGY,
            "requested_eos_token_id": requested_eos,
            "effective_eos_token_ids": eos_ids,
        },
        "model": {
            "model_dir": str(model_dir),
            "model_type": config["model_type"],
            "quantization": _configured_quantization(config),
            "config_sha256": config_sha256,
        },
        "packages": versions,
        "runtime": {
            "offline": True,
            "trust_remote_code": False,
            **runtime_identity,
        },
        "metrics": {
            "load_ms": (load_end - load_start) / 1_000_000,
            "ttft_ms": (
                (first_token_at - generation_start) / 1_000_000
                if first_token_at is not None
                else 0.0
            ),
            "tpot_ms": tpot_ms,
            "tps": tps,
            "timed_decode_tokens": decode_tokens,
            "mlx_peak_memory_bytes": peak_memory,
        },
        "generation": {
            "generated_token_ids": generated_ids,
            "generated_token_count": len(generated_ids),
            "stop_reason": stop_reason,
        },
    }


def _json_line(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as error:
        raise WorkerError(
            "internal_error", f"cannot serialize JSON output: {error}"
        ) from error


def _write_receipt(receipt: object) -> None:
    payload = _json_line(receipt)
    if len(payload) > MAX_OUTPUT_BYTES:
        raise WorkerError(
            "output_too_large", f"receipt exceeds {MAX_OUTPUT_BYTES} bytes"
        )
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def _write_error(error: WorkerError) -> None:
    message = " ".join(str(error).split())[:1024] or "unknown worker error"
    payload = _json_line(
        {
            "format": ERROR_FORMAT,
            "error": {"code": error.code, "message": message},
        }
    )
    sys.stderr.buffer.write(payload)
    sys.stderr.buffer.flush()


def _parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = JsonArgumentParser(add_help=False)
    parser.add_argument("--model-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        arguments = _parse_arguments(sys.argv[1:] if argv is None else argv)
        request = _validate_request(_read_request())
        model_dir, config, config_sha256 = _read_model_config(arguments.model_dir)
        with open(os.devnull, "w", encoding="utf-8") as dependency_output:
            with (
                contextlib.redirect_stdout(dependency_output),
                contextlib.redirect_stderr(dependency_output),
            ):
                receipt = _generate_receipt(request, model_dir, config, config_sha256)
        _write_receipt(receipt)
        return 0
    except WorkerError as error:
        _write_error(error)
        return 2
    except Exception as error:
        _write_error(
            WorkerError("internal_error", f"unexpected worker failure: {error}")
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
