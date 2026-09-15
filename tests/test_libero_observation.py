"""Golden values for the LIBERO observation conversion.

These assertions are **literals**, not re-derivations, because this conversion is
mirrored in ``apxinf_robo.envs.libero`` and a silent divergence between the two
copies yields wrong success rates on both sides with no error. A re-derived
expectation (``base[::-1, ::-1]``) passes no matter what the function does; a
literal does not. apxinf-robo's ``tests/test_libero_observation.py`` pins the
same values.
"""

import pickle
import sys
from types import SimpleNamespace

import numpy as np

from scripts.libero_observation import (
    libero_gr00t_action,
    libero_gr00t_state,
    libero_images,
    libero_state,
    load_libero_init_states,
)


def test_libero_images_rotate_each_frame_by_180_degrees():
    # 2x2 RGB, one channel value per pixel, so the rotation is readable at a glance:
    #   [[1, 2],          [[4, 3],
    #    [3, 4]]    ->     [2, 1]]
    base = np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]], dtype=np.uint8)
    wrist = base + 10

    images = libero_images(base, wrist)

    assert images.shape == (2, 2, 2, 3)
    np.testing.assert_array_equal(
        images[0],
        np.array([[[4, 4, 4], [3, 3, 3]], [[2, 2, 2], [1, 1, 1]]], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        images[1],
        np.array([[[14, 14, 14], [13, 13, 13]], [[12, 12, 12], [11, 11, 11]]], np.uint8),
    )
    assert images[0].flags["C_CONTIGUOUS"], "the engine reads these as contiguous NHWC"


def test_libero_state_collapses_mirrored_gripper_joints():
    observation = {
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.array([0.04, -0.04]),
    }

    state = libero_state(observation)

    np.testing.assert_array_equal(
        state,
        np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.04], dtype=np.float32),
    )
    assert state.dtype == np.float32


def test_libero_gr00t_state_preserves_named_two_joint_contract():
    observation = {
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.array([0.04, -0.04]),
    }

    state = libero_gr00t_state(observation)

    assert list(state) == ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    np.testing.assert_array_equal(state["x"], np.array([0.1], dtype=np.float32))
    np.testing.assert_array_equal(state["y"], np.array([0.2], dtype=np.float32))
    np.testing.assert_array_equal(state["z"], np.array([0.3], dtype=np.float32))
    np.testing.assert_array_equal(state["roll"], np.array([0.0], dtype=np.float32))
    np.testing.assert_array_equal(state["pitch"], np.array([0.0], dtype=np.float32))
    np.testing.assert_array_equal(state["yaw"], np.array([0.0], dtype=np.float32))
    np.testing.assert_array_equal(state["gripper"], np.array([0.04, -0.04], dtype=np.float32))


def test_libero_gr00t_action_matches_nvidia_environment_convention():
    decoded = np.array(
        [[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.0], [0, 0, 0, 0, 0, 0, 1.0]],
        dtype=np.float32,
    )

    actual = libero_gr00t_action(decoded)

    np.testing.assert_array_equal(actual[:, :6], decoded[:, :6])
    np.testing.assert_array_equal(actual[:, -1], np.array([1.0, -1.0], dtype=np.float32))
    np.testing.assert_array_equal(decoded[:, -1], np.array([0.0, 1.0], dtype=np.float32))


def test_libero_state_converts_the_quaternion_to_an_axis_angle():
    # Identity quaternion hides the conversion entirely: a 90-degree rotation about
    # +z must come back as (0, 0, pi/2), which pins both the axis and the scale.
    half = np.sqrt(0.5)
    observation = {
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([0.0, 0.0, half, half]),
        "robot0_gripper_qpos": np.array([0.02, -0.02]),
    }

    state = libero_state(observation)

    np.testing.assert_allclose(
        state,
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2, 0.02], dtype=np.float32),
        atol=1e-6,
    )


def test_a_gripper_that_is_not_two_mirrored_joints_is_rejected():
    # Silently accepting a 1- or 3-value gripper would shift every later state
    # component by one and score plausibly wrong.
    observation = {
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.array([0.04]),
    }

    try:
        libero_state(observation)
    except ValueError as error:
        assert "2 values" in str(error)
    else:
        raise AssertionError("expected a ValueError for a 1-value gripper")


def test_libero_init_states_retry_the_pytorch_26_default_for_trusted_fixture(
    monkeypatch, tmp_path
):
    expected = np.array([[1.0, 2.0]], dtype=np.float32)
    init_root = tmp_path / "init_files"
    init_file = init_root / "suite" / "task.init"
    init_file.parent.mkdir(parents=True)
    init_file.write_bytes(b"trusted fixture placeholder")

    class Suite:
        def get_task_init_states(self, _task_id):
            raise pickle.UnpicklingError("Weights only load failed")

        def get_task(self, _task_id):
            return SimpleNamespace(problem_folder="suite", init_states_file="task.init")

    calls = []

    def torch_load(path, **kwargs):
        calls.append((path, kwargs))
        return expected

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=torch_load))
    monkeypatch.setitem(
        sys.modules,
        "libero.libero",
        SimpleNamespace(get_libero_path=lambda key: str(init_root)),
    )

    actual = load_libero_init_states(Suite(), 0)

    np.testing.assert_array_equal(actual, expected)
    assert calls == [(init_file, {"weights_only": False})]
