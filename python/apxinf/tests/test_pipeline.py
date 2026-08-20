"""Pipeline behavior: chaining, whole-step replacement, per-step override."""

from __future__ import annotations

import numpy as np
import pytest

from apxinf.processors import GaussianNoise, ParseImage, Pipeline, ProcessorStep, ResizeWithPad


class AddN(ProcessorStep):
    PARAMS = ("n",)

    def __init__(self, n: int):
        self.n = n

    def __call__(self, x):
        return x + self.n


class MulN(ProcessorStep):
    PARAMS = ("n",)

    def __init__(self, n: int):
        self.n = n

    def __call__(self, x):
        return x * self.n


def test_pipeline_chains_left_to_right():
    pipe = Pipeline([("add", AddN(2)), ("mul", MulN(3))])
    assert pipe(1) == (1 + 2) * 3
    assert pipe.names == ["add", "mul"]


def test_pipeline_replace_swaps_whole_step():
    pipe = Pipeline([("add", AddN(2)), ("mul", MulN(3))])
    swapped = pipe.replace("mul", AddN(10))
    assert swapped(1) == (1 + 2) + 10
    assert pipe(1) == (1 + 2) * 3  # original untouched


def test_pipeline_override_touches_one_step_only():
    add = AddN(2)
    pipe = Pipeline([("add", add), ("mul", MulN(3))])
    overridden = pipe.override("add", n=5)
    assert overridden(1) == (1 + 5) * 3
    assert pipe(1) == (1 + 2) * 3  # original pipeline unchanged
    assert add.n == 2  # original step object unchanged


def test_pipeline_rejects_unknown_override_param():
    pipe = Pipeline([("add", AddN(2))])
    with pytest.raises(KeyError):
        pipe.override("add", bogus=1)


def test_pipeline_rejects_duplicate_names():
    with pytest.raises(ValueError):
        Pipeline([("s", AddN(1)), ("s", AddN(2))])


def test_image_pipeline_parse_then_resize():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(3, 100, 80), dtype=np.uint8)  # CHW
    pipe = Pipeline([("parse", ParseImage()), ("resize", ResizeWithPad(224))])
    out = pipe(image)
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8


def test_noise_override_seed_is_reproducible_and_isolated():
    base = GaussianNoise(4, 8, seed=0)
    a = base()
    reseeded = base.with_overrides(seed=123)
    b1 = reseeded()
    b2 = GaussianNoise(4, 8, seed=123)()
    np.testing.assert_array_equal(b1, b2)
    assert not np.array_equal(a, b1)


# --- insert / remove / reorder (copy-on-write) -----------------------------


def test_insert_before_and_after():
    pipe = Pipeline([("add", AddN(2)), ("mul", MulN(3))])
    before = pipe.insert_before("mul", ("pre", AddN(10)))
    assert before.names == ["add", "pre", "mul"]
    assert before(1) == ((1 + 2) + 10) * 3
    after = pipe.insert_after("add", ("mid", MulN(2)))
    assert after.names == ["add", "mid", "mul"]
    assert after(1) == ((1 + 2) * 2) * 3
    assert pipe.names == ["add", "mul"]  # original untouched


def test_remove_step():
    pipe = Pipeline([("add", AddN(2)), ("mul", MulN(3))])
    removed = pipe.remove("mul")
    assert removed.names == ["add"]
    assert removed(1) == 1 + 2
    assert pipe.names == ["add", "mul"]  # original untouched


def test_reorder_is_permutation_only():
    pipe = Pipeline([("add", AddN(2)), ("mul", MulN(3))])
    swapped = pipe.reorder(["mul", "add"])
    assert swapped.names == ["mul", "add"]
    assert swapped(1) == (1 * 3) + 2
    assert pipe(1) == (1 + 2) * 3  # original untouched
    with pytest.raises(ValueError):
        pipe.reorder(["add"])  # not a permutation
    with pytest.raises(ValueError):
        pipe.reorder(["add", "mul", "add"])  # duplicate / wrong set


def test_insert_rejects_duplicate_name():
    pipe = Pipeline([("add", AddN(2)), ("mul", MulN(3))])
    with pytest.raises(ValueError):
        pipe.insert_after("add", ("mul", AddN(1)))  # name already present


def test_remove_unknown_name_raises():
    pipe = Pipeline([("add", AddN(2))])
    with pytest.raises(KeyError):
        pipe.remove("nope")
