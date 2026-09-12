"""Golden values for the LIBERO observation conversion.

These assertions are **literals**, not re-derivations, because this conversion is
mirrored in ``apxinf_robo.envs.libero`` and a silent divergence between the two
copies yields wrong success rates on both sides with no error. A re-derived
expectation (``base[::-1, ::-1]``) passes no matter what the function does; a
literal does not. apxinf-robo's ``tests/test_libero_observation.py`` pins the
same values.
"""

import numpy as np
import pytest

from scripts.eval_libero import state_finger_joints
from scripts.libero_observation import libero_images, libero_state


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


def test_libero_state_keeps_both_finger_joints_on_request():
    """LeRobot's own LIBERO env feeds both mirrored joints (8-dim state)."""
    observation = {
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.array([0.04, -0.04]),
    }

    state = libero_state(observation, finger_joints=2)

    np.testing.assert_array_equal(
        state,
        np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.04, -0.04], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="finger_joints must be 1 or 2"):
        libero_state(observation, finger_joints=3)


def test_state_finger_joints_follows_the_policys_declared_state_width():
    assert state_finger_joints({"state_dim": 8}) == 2
    assert state_finger_joints({"state_dim": 7}) == 1
    assert state_finger_joints({}) == 1
    with pytest.raises(ValueError, match="can only build"):
        state_finger_joints({"state_dim": 32})


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
