"""Shared fixtures for the ``apxinf`` processor-library tests.

The processor tests are pure numpy/PIL and run offline. Two things are optional:

* a SentencePiece model — set ``APXINF_TOKENIZER`` (or drop a
  ``tokenizer.model`` under ``APXINF_PI05_MODEL_DIR``) to run tokenizer encode
  tests; they skip otherwise.
* the ``apxinf_py`` CUDA binding + a checkpoint — only the real-model layering
  test needs these and skips cleanly without them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make ``import apxinf`` work from a source checkout without installation.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))


def _model_dir() -> Path | None:
    value = os.environ.get("APXINF_PI05_MODEL_DIR")
    return Path(value) if value else None


@pytest.fixture(scope="session")
def tokenizer_path() -> str:
    explicit = os.environ.get("APXINF_TOKENIZER")
    if explicit and Path(explicit).is_file():
        return explicit
    model_dir = _model_dir()
    if model_dir is not None:
        for name in ("tokenizer.model", "paligemma_tokenizer.model"):
            candidate = model_dir / name
            if candidate.is_file():
                return str(candidate)
    pytest.skip("no SentencePiece tokenizer (set APXINF_TOKENIZER or APXINF_PI05_MODEL_DIR)")


@pytest.fixture(scope="session")
def model_dir() -> Path:
    directory = _model_dir()
    if directory is None or not directory.is_dir():
        pytest.skip("APXINF_PI05_MODEL_DIR not set; skipping checkpoint-backed test")
    return directory
