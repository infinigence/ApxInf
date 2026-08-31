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
    # Verbatim from openpi models/tokenizer.py. Note there is NO lower clamp:
    # digitize returns 0 for v < -1, so the bin is -1 and openpi puts that
    # straight into the prompt. An earlier version of this helper clipped to
    # [0, 255], which made the test agree with a wrong implementation.
    edges = np.linspace(-1.0, 1.0, 257)[:-1]
    return (np.digitize(state, edges) - 1).astype(np.int16)


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
    # Narrower than the stats is a genuine mismatch on both steps...
    with pytest.raises(ValueError):
        Unnormalizer(q01=q01, q99=q99)(np.zeros((4, LIBERO_DIM - 1), dtype=np.float32))
    with pytest.raises(ValueError):
        Normalizer(q01=q01, q99=q99)(np.zeros((4, LIBERO_DIM - 1), dtype=np.float32))
    # ...and so is a *wider* array on the input side, where openpi's Normalize
    # would broadcast-fail too. Only Unnormalizer widens; see below.
    with pytest.raises(ValueError):
        Normalizer(q01=q01, q99=q99)(np.zeros((4, LIBERO_DIM + 1), dtype=np.float32))


def test_unnormalize_passes_a_wider_array_s_tail_through(quantiles):
    """openpi ``Unnormalize._unnormalize_quantile``'s narrow-stats branch.

    A checkpoint's ``norm_stats`` is the robot's width (16 for a Unitree G1)
    while the model emits its padded width (32), so without this the G1 path
    cannot serve a real ``norm_stats.json`` at all.
    """
    q01, q99 = quantiles
    pad = 4
    rng = np.random.default_rng(5)
    wide = rng.uniform(-1.0, 1.0, size=(3, LIBERO_DIM + pad)).astype(np.float32)

    got = Unnormalizer(q01=q01, q99=q99)(wide)

    assert got.shape == wide.shape
    # The head is unnormalized exactly as if the tail were not there...
    np.testing.assert_array_equal(got[:, :LIBERO_DIM], Unnormalizer(q01=q01, q99=q99)(wide[:, :LIBERO_DIM]))
    # ...and the tail is copied verbatim, not scaled by some implied identity.
    np.testing.assert_array_equal(got[:, LIBERO_DIM:], wide[:, LIBERO_DIM:])


def test_discretize_matches_numpy_digitize():
    rng = np.random.default_rng(4)
    state = rng.uniform(-1.5, 1.5, size=64).astype(np.float32)
    np.testing.assert_array_equal(discretize_state(state), ref_digitize(state))


def test_discretize_underflows_signed_and_saturates_high():
    # Under -1 openpi emits -1, not 0: that token is in the training distribution.
    state = np.array([-5.0, -1.0000001, -1.0, 0.0, 0.99999, 1.0, 5.0], dtype=np.float64)
    out = discretize_state(state)
    np.testing.assert_array_equal(out, ref_digitize(state))
    np.testing.assert_array_equal(out, [-1, -1, 0, 128, 255, 255, 255])
    assert out.dtype == np.int16


def test_discretize_matches_digitize_on_bin_edges():
    edges = np.linspace(-1.0, 1.0, 257)[:-1]
    for probe in (edges, edges - 1e-9, edges + 1e-9):
        np.testing.assert_array_equal(discretize_state(probe), ref_digitize(probe))


def test_discretize_matches_digitize_one_ulp_from_every_edge():
    """The ±1e-9 probe above is ~10 million ulps wide and misses the real trap.

    ``floor((v + 1.0) * 128.0)`` -- the obvious way to write this -- is not equal
    to ``digitize``. For a value one ulp below an edge in ``[-1, -0.5)``, adding
    ``1.0`` moves it into a binade with twice the ulp, the sum rounds *up* onto
    the edge, and the index comes out one bin high. One bin is a different prompt
    string, hence different token ids, hence a different rollout.
    """
    edges = np.linspace(-1.0, 1.0, 257)[:-1]
    for probe in (np.nextafter(edges, -np.inf), np.nextafter(edges, np.inf)):
        np.testing.assert_array_equal(discretize_state(probe), ref_digitize(probe))

    # The one f64 value that the arithmetic form gets wrong, pinned by name.
    trap = np.nextafter(-0.4921875, -np.inf)
    assert np.floor((trap + 1.0) * 128.0) == 65, "the trap this test guards"
    assert discretize_state([trap])[0] == 64
    np.testing.assert_array_equal(discretize_state([trap]), ref_digitize(np.array([trap])))
