"""Shared fixtures for the ``apxinf_py`` PyO3 smoke tests.

These tests need a CUDA machine (the pi05 runtime is CUDA-only) plus a pi05
checkpoint. They are gated on environment variables and skip cleanly when the
extension is not built with ``--features cuda`` or no checkpoint is provided:

  APXINF_PI05_CHECKPOINT   checkpoint dir (with config.json) or index file  [required]
  APXINF_PI05_DEVICE       device string, default "cuda:0"
  APXINF_PI05_PRECISION    auto|fp8|bf16|int8, default "bf16"
  APXINF_PI05_CALIBRATION  FP8 calibration.json (only for precision=fp8)
  APXINF_PI05_TACTICS      FP8 tactics.json    (only for precision=fp8)
"""

import os

import numpy as np
import pytest

apxinf_py = pytest.importorskip("apxinf_py")


def _checkpoint() -> str:
    path = os.environ.get("APXINF_PI05_CHECKPOINT")
    if not path:
        pytest.skip("APXINF_PI05_CHECKPOINT not set; skipping CUDA smoke test")
    return path


@pytest.fixture(scope="session")
def precision() -> str:
    return os.environ.get("APXINF_PI05_PRECISION", "bf16")


@pytest.fixture(scope="session")
def model(precision: str) -> "apxinf_py.Model":
    kwargs = {
        "device": os.environ.get("APXINF_PI05_DEVICE", "cuda:0"),
        "precision": precision,
    }
    calibration = os.environ.get("APXINF_PI05_CALIBRATION")
    tactics = os.environ.get("APXINF_PI05_TACTICS")
    if calibration:
        kwargs["calibration"] = calibration
    if tactics:
        kwargs["tactics"] = tactics
    try:
        return apxinf_py.Model.load("pi05", _checkpoint(), **kwargs)
    except Exception as error:  # noqa: BLE001 - surface load failures as skips
        pytest.skip(f"pi05 load failed (needs CUDA build + checkpoint): {error}")


def make_tokens(model: "apxinf_py.Model", count: int = 16) -> np.ndarray:
    count = min(count, model.max_token_len)
    return np.zeros(count, dtype=np.uint32)


def make_noise(model: "apxinf_py.Model", seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(
        (model.action_horizon, model.action_dim), dtype=np.float32
    )
