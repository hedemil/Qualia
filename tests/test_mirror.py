"""Tests for mirror augmentation kinematic correctness.

These tests verify that mirroring produces physically valid trajectories:
- Left arm movements become right arm movements (and vice versa)
- Rotation-direction-dependent joints are sign-flipped
- Images are horizontally flipped
- Camera pairs are swapped
- Applying mirror twice returns the original data
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from augmentations.mirror import MirrorAugmentation


@pytest.fixture
def aloha_mirror():
    """Mirror augmentation configured for ALOHA (14-DOF bimanual)."""
    return MirrorAugmentation(
        arm_size=7,
        sign_flip_within_arm=[0, 3, 5],  # waist, forearm_roll, wrist_rotate
        camera_swap_pairs=[
            ("observation.images.cam_left_wrist", "observation.images.cam_right_wrist"),
        ],
    )


@pytest.fixture
def sample_frame():
    """A sample frame with known joint values for testing."""
    # Left arm: joints 0-6, Right arm: joints 7-13
    # Using distinct values so swaps are easy to verify
    left_arm = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
    right_arm = np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], dtype=np.float32)
    state = np.concatenate([left_arm, right_arm])
    action = state * 0.1  # Different from state but same structure

    # Create a simple test image (8x8 RGB with left/right halves distinct)
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[:, :4, :] = 100  # Left half = gray
    img[:, 4:, :] = 200  # Right half = lighter

    return {
        "observation.state": state.copy(),
        "action": action.copy(),
        "observation.images.cam_high": img.copy(),
        "observation.images.cam_left_wrist": (img * 0.5).astype(np.uint8),
        "observation.images.cam_right_wrist": (img * 0.8).astype(np.uint8),
        "next.done": np.array([False]),
        "task": "Pick up the cup",
    }


@pytest.fixture
def metadata():
    return {
        "camera_keys": [
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ]
    }


class TestArmSwap:
    """Test that left/right arm joints are correctly swapped."""

    def test_left_becomes_right(self, aloha_mirror, sample_frame, metadata):
        """Left arm joint values should appear in right arm positions after mirroring."""
        original_left = sample_frame["observation.state"][:7].copy()
        result = aloha_mirror.apply_frame(sample_frame, metadata)
        mirrored_right = result["observation.state"][7:14]

        # After swap, original left values are in right positions
        # But sign-flipped joints (0, 3, 5) will have negated values
        for i in range(7):
            if i in [0, 3, 5]:
                np.testing.assert_almost_equal(mirrored_right[i], -original_left[i])
            else:
                np.testing.assert_almost_equal(mirrored_right[i], original_left[i])

    def test_right_becomes_left(self, aloha_mirror, sample_frame, metadata):
        """Right arm joint values should appear in left arm positions after mirroring."""
        original_right = sample_frame["observation.state"][7:14].copy()
        result = aloha_mirror.apply_frame(sample_frame, metadata)
        mirrored_left = result["observation.state"][:7]

        for i in range(7):
            if i in [0, 3, 5]:
                np.testing.assert_almost_equal(mirrored_left[i], -original_right[i])
            else:
                np.testing.assert_almost_equal(mirrored_left[i], original_right[i])

    def test_action_swapped_same_as_state(self, aloha_mirror, sample_frame, metadata):
        """Action vector should be swapped with the same logic as state."""
        original_action = sample_frame["action"].copy()
        result = aloha_mirror.apply_frame(sample_frame, metadata)

        # Verify left->right swap happened on action too
        for i in range(7):
            if i in [0, 3, 5]:
                np.testing.assert_almost_equal(
                    result["action"][7 + i], -original_action[i]
                )
            else:
                np.testing.assert_almost_equal(
                    result["action"][7 + i], original_action[i]
                )


class TestSignFlip:
    """Test that rotation-direction-dependent joints are sign-flipped."""

    def test_waist_sign_flipped(self, aloha_mirror, sample_frame, metadata):
        """Waist joints (index 0, 7) should be sign-flipped after swap."""
        result = aloha_mirror.apply_frame(sample_frame, metadata)
        # Original left waist (index 0) = 0.1, goes to right waist (index 7), negated
        np.testing.assert_almost_equal(result["observation.state"][7], -0.1)
        # Original right waist (index 7) = 1.1, goes to left waist (index 0), negated
        np.testing.assert_almost_equal(result["observation.state"][0], -1.1)

    def test_forearm_roll_sign_flipped(self, aloha_mirror, sample_frame, metadata):
        """Forearm roll joints (index 3, 10) should be sign-flipped."""
        result = aloha_mirror.apply_frame(sample_frame, metadata)
        # Original left forearm_roll (index 3) = 0.4, goes to right (index 10), negated
        np.testing.assert_almost_equal(result["observation.state"][10], -0.4)

    def test_wrist_rotate_sign_flipped(self, aloha_mirror, sample_frame, metadata):
        """Wrist rotate joints (index 5, 12) should be sign-flipped."""
        result = aloha_mirror.apply_frame(sample_frame, metadata)
        # Original left wrist_rotate (index 5) = 0.6, goes to right (index 12), negated
        np.testing.assert_almost_equal(result["observation.state"][12], -0.6)

    def test_non_flipped_joints_unchanged(self, aloha_mirror, sample_frame, metadata):
        """Joints NOT in sign_flip list should be swapped but not negated."""
        result = aloha_mirror.apply_frame(sample_frame, metadata)
        # Index 1 (shoulder) not in flip list: left shoulder (0.2) goes to right (index 8)
        np.testing.assert_almost_equal(result["observation.state"][8], 0.2)
        # Index 2 (elbow) not in flip list: left elbow (0.3) goes to right (index 9)
        np.testing.assert_almost_equal(result["observation.state"][9], 0.3)


class TestImageFlip:
    """Test that camera images are horizontally flipped."""

    def test_image_horizontally_flipped(self, aloha_mirror, sample_frame, metadata):
        """Left half of image should become right half after flipping."""
        original = sample_frame["observation.images.cam_high"].copy()
        result = aloha_mirror.apply_frame(sample_frame, metadata)
        flipped = result["observation.images.cam_high"]

        # Original left half (cols 0-3) was 100, should now be in right half (cols 4-7)
        np.testing.assert_array_equal(flipped[:, 4:, :], original[:, :4, :])
        # Original right half (cols 4-7) was 200, should now be in left half (cols 0-3)
        np.testing.assert_array_equal(flipped[:, :4, :], original[:, 4:, :])

    def test_camera_pairs_swapped(self, aloha_mirror, sample_frame, metadata):
        """Left wrist camera data should be swapped with right wrist camera data."""
        original_left = sample_frame["observation.images.cam_left_wrist"].copy()
        original_right = sample_frame["observation.images.cam_right_wrist"].copy()
        result = aloha_mirror.apply_frame(sample_frame, metadata)

        # After swap: left wrist gets flipped right data, right wrist gets flipped left data
        # The flip happens first, then the swap
        np.testing.assert_array_equal(
            result["observation.images.cam_left_wrist"],
            np.fliplr(original_right),
        )
        np.testing.assert_array_equal(
            result["observation.images.cam_right_wrist"],
            np.fliplr(original_left),
        )


class TestDoubleFlipIdentity:
    """Test that applying mirror twice returns the original data."""

    def test_double_mirror_state_identity(self, aloha_mirror, sample_frame, metadata):
        """mirror(mirror(state)) should equal original state."""
        original = sample_frame["observation.state"].copy()
        once = aloha_mirror.apply_frame(sample_frame, metadata)
        twice = aloha_mirror.apply_frame(once, metadata)
        np.testing.assert_array_almost_equal(twice["observation.state"], original)

    def test_double_mirror_action_identity(self, aloha_mirror, sample_frame, metadata):
        """mirror(mirror(action)) should equal original action."""
        original = sample_frame["action"].copy()
        once = aloha_mirror.apply_frame(sample_frame, metadata)
        twice = aloha_mirror.apply_frame(once, metadata)
        np.testing.assert_array_almost_equal(twice["action"], original)

    def test_double_mirror_image_identity(self, aloha_mirror, sample_frame, metadata):
        """mirror(mirror(image)) should equal original image."""
        original = sample_frame["observation.images.cam_high"].copy()
        once = aloha_mirror.apply_frame(sample_frame, metadata)
        twice = aloha_mirror.apply_frame(once, metadata)
        np.testing.assert_array_equal(twice["observation.images.cam_high"], original)


class TestTaskPreserved:
    """Test that non-visual/non-kinematic data is preserved."""

    def test_task_unchanged(self, aloha_mirror, sample_frame, metadata):
        result = aloha_mirror.apply_frame(sample_frame, metadata)
        assert result["task"] == "Pick up the cup"

    def test_done_flag_unchanged(self, aloha_mirror, sample_frame, metadata):
        result = aloha_mirror.apply_frame(sample_frame, metadata)
        np.testing.assert_array_equal(result["next.done"], sample_frame["next.done"])


class TestRobotPresets:
    """Test that different robot configs produce different behavior."""

    def test_so100_arm_size_6(self):
        """SO-100 has 6-DOF arms, not 7."""
        mirror = MirrorAugmentation(
            arm_size=6,
            sign_flip_within_arm=[0, 4],
            camera_swap_pairs=[],
        )
        state = np.arange(12, dtype=np.float32)  # 6 + 6
        frame = {
            "observation.state": state.copy(),
            "action": state.copy(),
            "task": "test",
        }
        result = mirror.apply_frame(frame, {"camera_keys": []})
        # Left (0-5) should swap with right (6-11)
        np.testing.assert_almost_equal(result["observation.state"][0], -6.0)  # sign-flipped
        np.testing.assert_almost_equal(result["observation.state"][1], 7.0)   # not flipped
        np.testing.assert_almost_equal(result["observation.state"][6], -0.0)  # sign-flipped
        np.testing.assert_almost_equal(result["observation.state"][7], 1.0)   # not flipped
