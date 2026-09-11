import numpy as np
import pytest

from scripts.eval_libero import state_finger_joints
from scripts.libero_observation import libero_images, libero_state


def test_libero_images_only_reorients_raw_frames():
    base = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    wrist = base + 1

    images = libero_images(base, wrist)

    assert images.shape == (2, 4, 5, 3)
    np.testing.assert_array_equal(images[0], base[::-1, ::-1])
    np.testing.assert_array_equal(images[1], wrist[::-1, ::-1])


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
