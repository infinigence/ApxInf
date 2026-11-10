#!/usr/bin/env python3
"""Fetch the pinned Qwen3.8-27B-AWQ-INT4 metadata bundle (no weights).

The leaderboard contract pins one model repository and one revision.  Everything
in this bundle is configuration, tokenizer, and the safetensors *index* — the
tensor-name/shape/shard map — so the whole download is ~12.5 MiB with zero
weight values.  That is enough to:

  * load the tokenizer (`AutoTokenizer` resolves `Qwen2Tokenizer` from
    `tokenizer_config.json`, so a local `transformers` that does not know the
    `qwen3_5` architecture still tokenizes correctly);
  * run `test.py prepare` and generate the public / hidden-proxy case suites;
  * build the W4 dispatch manifest, dtype classification, shape inventory and
    the VRAM ledger from `model.safetensors.index.json`;
  * read the real `eos_token_id` list the service must honour.

It is deliberately *not* enough to run the model.  Weights are never fetched by
this script and must never be committed.

Following the same policy the contract states for the public corpus — fetch and
verify at use time, do not vendor third-party payloads in the repository — every
file is checked against the SHA256 recorded below before it is accepted.

Usage:

    python3 scripts/fetch_model_metadata.py
    python3 scripts/fetch_model_metadata.py --output-dir /path/to/dir
    python3 scripts/fetch_model_metadata.py --check-only
    python3 scripts/fetch_model_metadata.py --force

Then point the evaluation tooling at the same directory:

    python3 benchmarks/qwen38_4090/evaluation/test.py prepare \
      --model-dir ../apxinf-private-models/63768c10df38c0395e12ef49edac1bd539eaeeea

`vocab.json` and `merges.txt` are intentionally absent: `tokenizer.json` already
carries the full vocabulary and merge table, and the case generator's
`tokenizer_fingerprint()` only identifies `tokenizer.json`,
`tokenizer_config.json`, `special_tokens_map.json` and `config.json`.  Dropping
the two redundant files saves 9.6 MiB and provably changes nothing: the public
`cases.jsonl` generated from this bundle is byte-identical to the one generated
from the full checkpoint directory.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parent.parent

# Pinned by benchmarks/qwen38_4090/evaluation/contract-v1.json -> model.
REPO_ID = "cyankiwi/Qwen3.8-27B-AWQ-INT4"
REVISION = "63768c10df38c0395e12ef49edac1bd539eaeeea"

# name -> (size in bytes, sha256).  Both are verified; a size-only match is not
# accepted.  Regenerate with `sha256sum` against the pinned revision only.
MANIFEST: dict[str, tuple[int, str]] = {
    "config.json": (
        20927,
        "fece2915d4c8ad4c10877622f04ea5e01cd3ae38768ce5c1edb700dd1de290f6",
    ),
    "generation_config.json": (
        202,
        "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e",
    ),
    "model.safetensors.index.json": (
        241486,
        "82b1bf79f5b61333e83da17ec3bf89c9f178e29395a14c6b3ce3bbc474e1ead8",
    ),
    "tokenizer.json": (
        12809320,
        "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
    ),
    "tokenizer_config.json": (
        17928,
        "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
    ),
    "chat_template.jinja": (
        8952,
        "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
    ),
    "preprocessor_config.json": (
        390,
        "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
    ),
    "video_preprocessor_config.json": (
        385,
        "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
    ),
}

DEFAULT_OUTPUT_DIR = REPOSITORY.parent / "apxinf-private-models" / REVISION

CHUNK_BYTES = 1 << 20
ATTEMPTS = 3
TIMEOUT_SECONDS = 120
USER_AGENT = "apxinf-fetch-model-metadata/1"


def file_digest(path: Path) -> tuple[int, str]:
    """Return (size, sha256) for an existing file, streaming it once."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def verify(path: Path, expected_size: int, expected_sha256: str) -> str | None:
    """Return None when the file matches, else a human-readable reason."""
    if not path.is_file():
        return "missing"
    size, sha256 = file_digest(path)
    if size != expected_size:
        return f"size {size} != expected {expected_size}"
    if sha256 != expected_sha256:
        return f"sha256 {sha256} != expected {expected_sha256}"
    return None


def download(name: str, destination: Path, expected_size: int, expected_sha256: str) -> None:
    """Download one file to `destination`, verifying before it is published.

    The bytes land in a sibling `.part` file and are only renamed after the
    digest matches, so an interrupted run can never leave a truncated file that
    a later `--check-only` would have to distinguish from a real one.
    """
    url = f"https://huggingface.co/{REPO_ID}/resolve/{REVISION}/{name}"
    partial = destination.with_suffix(destination.suffix + ".part")
    last_error: str | None = None

    for attempt in range(1, ATTEMPTS + 1):
        digest = hashlib.sha256()
        size = 0
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                with partial.open("wb") as handle:
                    while chunk := response.read(CHUNK_BYTES):
                        handle.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            last_error = f"{type(error).__name__}: {error}"
            partial.unlink(missing_ok=True)
            if attempt < ATTEMPTS:
                print(f"    attempt {attempt}/{ATTEMPTS} failed: {last_error}")
                continue
            raise SystemExit(f"{name}: download failed after {ATTEMPTS} attempts: {last_error}")

        if size != expected_size or digest.hexdigest() != expected_sha256:
            partial.unlink(missing_ok=True)
            raise SystemExit(
                f"{name}: integrity check failed (got {size} bytes / "
                f"{digest.hexdigest()}, expected {expected_size} bytes / {expected_sha256}). "
                "The pinned revision must not change; do not edit MANIFEST to make this pass."
            )

        partial.replace(destination)
        return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"where to place the bundle (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify an existing bundle and exit non-zero on any mismatch; download nothing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download every file even when the local copy already verifies",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir.expanduser().resolve()
    try:
        output_dir.relative_to(REPOSITORY)
    except ValueError:
        pass
    else:
        raise SystemExit(
            f"refusing to write inside the repository: {output_dir}\n"
            "This bundle is third-party data and must stay untracked. Pick a path "
            f"outside {REPOSITORY}, for example {DEFAULT_OUTPUT_DIR}."
        )

    total = sum(size for size, _ in MANIFEST.values())
    print(f"repo     {REPO_ID}")
    print(f"revision {REVISION}")
    print(f"target   {output_dir}")
    print(f"bundle   {len(MANIFEST)} files, {total} bytes ({total / (1 << 20):.1f} MiB), no weights")
    print()

    if not args.check_only:
        output_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    fetched = 0
    for name, (expected_size, expected_sha256) in MANIFEST.items():
        destination = output_dir / name
        reason = verify(destination, expected_size, expected_sha256)

        if args.check_only:
            if reason is None:
                print(f"  ok       {name}")
            else:
                print(f"  FAILED   {name}: {reason}")
                failures.append(name)
            continue

        if reason is None and not args.force:
            print(f"  cached   {name}")
            continue

        print(f"  fetching {name} ({expected_size} bytes)")
        download(name, destination, expected_size, expected_sha256)
        fetched += 1

    print()
    if args.check_only:
        if failures:
            print(f"{len(failures)} file(s) failed verification: {', '.join(failures)}")
            print("Run without --check-only to (re)download them.")
            return 1
        print(f"all {len(MANIFEST)} files verified")
        return 0

    print(f"bundle ready in {output_dir} ({fetched} downloaded, {len(MANIFEST) - fetched} cached)")
    print()
    print("Next:")
    print(f"  python3 benchmarks/qwen38_4090/evaluation/test.py prepare --model-dir {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
