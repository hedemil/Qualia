"""Tests for action/state noise augmentation."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from augmentations.action_noise import ActionNoiseAugmentation


@pytest.fixture
def noise_aug():
    return ActionNoiseAugmentation(noise_std=0.01)


@pytest.fixture
def sample_frame():
    return {
        "observation.state": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "action": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        "observation.images.cam_high": np.zeros((8, 8, 3), dtype=np.uint8),
        "task": "test",
    }


@pytest.fixture
def metadata():
    return {"camera_keys": ["observation.images.cam_high"]}


class TestActionNoise:

    def test_action_modified(self, noise_aug, sample_frame, metadata):
        """Action should be modified by noise."""
        original = sample_frame["action"].copy()
        result = noise_aug.apply_frame(sample_frame, metadata)
        assert not np.array_equal(result["action"], original)

    def test_state_modified(self, noise_aug, sample_frame, metadata):
        """Observation state should also be modified by noise."""
        original = sample_frame["observation.state"].copy()
        result = noise_aug.apply_frame(sample_frame, metadata)
        assert not np.array_equal(result["observation.state"], original)

    def test_noise_magnitude_bounded(self, noise_aug, sample_frame, metadata):
        """Noise should be small relative to signal (within ~4 sigma)."""
        original_action = sample_frame["action"].copy()
        result = noise_aug.apply_frame(sample_frame, metadata)
        diff = np.abs(result["action"] - original_action)
        # With noise_std=0.01, values should be within ~0.04 (4 sigma)
        assert np.all(diff < 0.1)

    def test_dtype_preserved(self, noise_aug, sample_frame, metadata):
        """Output dtype should match input dtype."""
        result = noise_aug.apply_frame(sample_frame, metadata)
        assert result["action"].dtype == np.float32
        assert result["observation.state"].dtype == np.float32

    def test_shape_preserved(self, noise_aug, sample_frame, metadata):
        """Output shape should match input shape."""
        result = noise_aug.apply_frame(sample_frame, metadata)
        assert result["action"].shape == (4,)
        assert result["observation.state"].shape == (4,)

    def test_images_unchanged(self, noise_aug, sample_frame, metadata):
        """Action noise must NOT modify images."""
        original = sample_frame["observation.images.cam_high"].copy()
        result = noise_aug.apply_frame(sample_frame, metadata)
        np.testing.assert_array_equal(result["observation.images.cam_high"], original)

    def test_statistical_properties(self, noise_aug, metadata):
        """Over many samples, noise should be zero-mean with correct std."""
        base_action = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        diffs = []
        for _ in range(10000):
            frame = {
                "action": base_action.copy(),
                "observation.state": base_action.copy(),
                "observation.images.cam_high": np.zeros((1, 1, 3), dtype=np.uint8),
                "task": "test",
            }
            result = noise_aug.apply_frame(frame, metadata)
            diffs.append(result["action"] - base_action)

        diffs = np.array(diffs)
        # Mean should be approximately 0
        np.testing.assert_almost_equal(diffs.mean(), 0.0, decimal=2)
        # Std should be approximately noise_std
        np.testing.assert_almost_equal(diffs.std(), 0.01, decimal=2)
