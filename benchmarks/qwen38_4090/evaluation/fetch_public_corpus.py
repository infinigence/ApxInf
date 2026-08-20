#!/usr/bin/env python3
"""Fetch and verify the public long-document corpus declared by the contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return value


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def fetch_verified_corpus(contract: dict[str, Any], output: Path) -> dict[str, Any]:
    source = contract["data_generation"]["corpus"]
    request = urllib.request.Request(
        source["source_url"],
        headers={"User-Agent": "ApxInf-course-evaluator/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()

    actual_sha256 = sha256_bytes(payload)
    actual_bytes = len(payload)
    if actual_sha256 != source["sha256"]:
        raise ValueError(
            "corpus SHA-256 mismatch: "
            f"expected {source['sha256']}, got {actual_sha256}"
        )
    if actual_bytes != int(source["bytes"]):
        raise ValueError(
            f"corpus size mismatch: expected {source['bytes']}, got {actual_bytes}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return {
        "schema": "apxinf.qwen38_27b.corpus_fetch.v1",
        "corpus_id": source["id"],
        "source_url": source["source_url"],
        "output": str(output),
        "bytes": actual_bytes,
        "sha256": actual_sha256,
        "verified": True,
    }


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=here / "contract-v1.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=here / ".cache" / "pg24264.txt",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = fetch_verified_corpus(load_json(args.contract), args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
