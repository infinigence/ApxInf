"""Golden tests for Normalizer / Unnormalizer and state discretization."""

from __future__ import annotations

import numpy as np
import pytest

from apxinf.processors import Normalizer, Unnormalizer, discretize_state

LIBERO_DIM = 7
EPS = 1e-6


def ref_unnormalize(normalized: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    # Exact formula from scripts/pi05_openpi_websocket_server.py.
    return (
        (normalized + 1.0) * (q99 - q01 + np.float32(1.0e-6)) / 2.0 + q01
    ).astype(np.float32)


def ref_digitize(state: np.ndarray) -> np.ndarray:
    # NumPy digitize(state, linspace(-1, 1, 257)[:-1]) - 1, with saturation.
    edges = np.linspace(-1.0, 1.0, 257)[:-1]
    return np.clip(np.digitize(state, edges) - 1, 0, 255).astype(np.uint8)


@pytest.fixture
def quantiles():
    rng = np.random.default_rng(0)
    q01 = rng.uniform(-2.0, -0.5, size=LIBERO_DIM).astype(np.float32)
    q99 = rng.uniform(0.5, 2.0, size=LIBERO_DIM).astype(np.float32)
    return q01, q99


def test_unnormalize_matches_reference(quantiles):
    q01, q99 = quantiles
    rng = np.random.default_rng(1)
    normalized = rng.uniform(-1.0, 1.0, size=(10, LIBERO_DIM)).astype(np.float32)
    got = Unnormalizer(q01=q01, q99=q99)(normalized)
    want = ref_unnormalize(normalized, q01, q99)
    np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-6)


def test_normalize_is_left_inverse_of_unnormalize(quantiles):
    q01, q99 = quantiles
    rng = np.random.default_rng(2)
    normalized = rng.uniform(-1.0, 1.0, size=(10, LIBERO_DIM)).astype(np.float32)
    physical = Unnormalizer(q01=q01, q99=q99)(normalized)
    recovered = Normalizer(q01=q01, q99=q99)(physical)
    np.testing.assert_allclose(recovered, normalized, rtol=0.0, atol=1e-4)


def test_dims_trims_stats():
    q01 = np.full(32, -1.0, dtype=np.float32)
    q99 = np.full(32, 1.0, dtype=np.float32)
    un = Unnormalizer(q01=q01, q99=q99, dims=LIBERO_DIM)
    assert un.width == LIBERO_DIM
    out = un(np.zeros((3, LIBERO_DIM), dtype=np.float32))
    assert out.shape == (3, LIBERO_DIM)


def test_mean_std_roundtrip():
    rng = np.random.default_rng(3)
    mean = rng.normal(size=LIBERO_DIM).astype(np.float32)
    std = rng.uniform(0.5, 2.0, size=LIBERO_DIM).astype(np.float32)
    x = rng.normal(size=(5, LIBERO_DIM)).astype(np.float32)
    norm = Normalizer(mean=mean, std=std, mode="mean_std")(x)
    back = Unnormalizer(mean=mean, std=std, mode="mean_std")(norm)
    np.testing.assert_allclose(back, x, rtol=0.0, atol=1e-4)


def test_wrong_last_dim_raises(quantiles):
    q01, q99 = quantiles
    with pytest.raises(ValueError):
        Unnormalizer(q01=q01, q99=q99)(np.zeros((4, LIBERO_DIM + 1), dtype=np.float32))


def test_discretize_matches_numpy_digitize():
    rng = np.random.default_rng(4)
    state = rng.uniform(-1.5, 1.5, size=64).astype(np.float32)
    np.testing.assert_array_equal(discretize_state(state), ref_digitize(state))


def test_discretize_saturates_out_of_range():
    state = np.array([-5.0, -1.0, 0.0, 0.99999, 1.0, 5.0], dtype=np.float32)
    out = discretize_state(state)
    assert out[0] == 0 and out[-1] == 255
    assert out.dtype == np.uint8
