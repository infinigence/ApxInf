#!/usr/bin/env python3
"""Enforce the public cuda-new layering rules without requiring a CUDA build."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OPS_ROOT = REPO_ROOT / "crates/apxinf-cuda-new/src/ops"
MODEL_ROOT = REPO_ROOT / "crates/apxinf-model/src"
ADAPTER_ROOT = REPO_ROOT / "crates/apxinf-cuda-new/native/adapters"

PUBLIC_ITEM = re.compile(
    r"^\s*pub(?:\([^)]*\))?\s+"
    r"(?:async\s+|unsafe\s+|const\s+)?"
    r"(fn|struct|enum|trait|type|const|static|mod)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)"
)
FORBIDDEN_L3_NAME = re.compile(
    r"(?:^|_)(?:bf16|f16|f32|fp8|w8a8|nvfp4|e4m3|packed4|packed8|"
    r"cached_1024|cached_1536|interleaved|aot|m41)(?:_|$)"
)

# Existing debt is explicit and reviewable. New entries must not be added merely
# to make CI green: move dtype/layout/shape/implementation details into a Spec,
# candidate config, or supports() condition. Removing an entry requires deleting
# it here as well, keeping the allowlist minimal.
ALLOWED_L3_NAME_DEBT: set[str] = set()


def rust_sources(root: Path):
    return sorted(root.rglob("*.rs"))


def public_items() -> list[tuple[str, int, str, str]]:
    result: list[tuple[str, int, str, str]] = []
    for path in rust_sources(OPS_ROOT):
        relative = path.relative_to(OPS_ROOT).as_posix()
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if "pub(crate)" in line or "pub(super)" in line:
                continue
            match = PUBLIC_ITEM.match(line)
            if match:
                result.append((relative, line_number, match.group(1), match.group(2)))
    return result


def matching_lines(root: Path, suffixes: tuple[str, ...], pattern: re.Pattern[str]):
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if pattern.search(line):
                yield path.relative_to(REPO_ROOT).as_posix(), line_number, line.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the mechanically discovered public src/ops API",
    )
    args = parser.parse_args()

    items = public_items()
    if args.list:
        for path, line, kind, name in items:
            print(f"{path}:{line}:{kind}:{name}")
        return 0

    errors: list[str] = []
    forbidden = {
        f"{path}:{name}"
        for path, _line, _kind, name in items
        if FORBIDDEN_L3_NAME.search(name)
    }
    for item in sorted(forbidden - ALLOWED_L3_NAME_DEBT):
        errors.append(f"new public L3 dtype/layout/shape/provider name: {item}")
    for item in sorted(ALLOWED_L3_NAME_DEBT - forbidden):
        errors.append(f"stale public L3 name allowlist entry: {item}")

    model_ffi = re.compile(r"(?:apxinf_cuda_new::)?ffi::abi|apxinf_cuda_new::raw")
    for path, line, source in matching_lines(MODEL_ROOT, (".rs",), model_ffi):
        errors.append(f"model layer calls cuda-new FFI directly: {path}:{line}: {source}")

    adapter_kernel = re.compile(r"\b__global__\b")
    for path, line, source in matching_lines(
        ADAPTER_ROOT, (".cu", ".cuh", ".cpp", ".h", ".hpp"), adapter_kernel
    ):
        errors.append(f"adapter defines a CUDA kernel: {path}:{line}: {source}")

    if errors:
        print("cuda-new architecture checks failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(
        "cuda-new architecture checks passed "
        f"({len(items)} public src/ops items, {len(forbidden)} allowlisted name debts)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
