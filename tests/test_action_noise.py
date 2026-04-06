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

    def test_clipping_enforces_bounds(self, metadata):
        """Noisy values must be clipped to dataset min/max when stats are provided."""
        aug = ActionNoiseAugmentation(noise_std=1.0)  # Large noise to force clipping
        stats = {
            "action": {
                "min": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                "max": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                "std": np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
            },
            "observation.state": {
                "min": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                "max": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                "std": np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
            },
        }
        aug.prepare([], stats=stats)
        frame = {
            "action": np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
            "observation.state": np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
            "observation.images.cam_high": np.zeros((8, 8, 3), dtype=np.uint8),
            "task": "test",
        }
        for _ in range(100):
            result = aug.apply_frame(frame, metadata)
            assert np.all(result["action"] >= 0.0)
            assert np.all(result["action"] <= 1.0)
            assert np.all(result["observation.state"] >= 0.0)
            assert np.all(result["observation.state"] <= 1.0)

    def test_no_clipping_without_stats(self, metadata):
        """Without stats, values can exceed any range (no clipping applied)."""
        aug = ActionNoiseAugmentation(noise_std=1.0)
        # No prepare() call — stats remain empty
        frame = {
            "action": np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
            "observation.state": np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
            "observation.images.cam_high": np.zeros((8, 8, 3), dtype=np.uint8),
            "task": "test",
        }
        saw_out_of_range = False
        for _ in range(100):
            result = aug.apply_frame(frame, metadata)
            if np.any(result["action"] < 0.0) or np.any(result["action"] > 1.0):
                saw_out_of_range = True
                break
        assert saw_out_of_range, "With noise_std=1.0 and no clipping, values should exceed [0,1]"

    def test_statistical_properties_without_stats(self, noise_aug, metadata):
        """Without stats, noise should be uniform across dims with std = noise_std."""
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
        np.testing.assert_almost_equal(diffs.mean(), 0.0, decimal=2)
        np.testing.assert_almost_equal(diffs.std(), 0.01, decimal=2)

    def test_statistical_properties_with_stats(self, metadata):
        """With stats, noise std per dimension should scale by dimension std."""
        aug = ActionNoiseAugmentation(noise_std=0.01)
        dim_stds = np.array([0.1, 1.0, 0.5, 2.0], dtype=np.float32)
        stats = {
            "action": {
                "min": np.array([-10, -10, -10, -10], dtype=np.float32),
                "max": np.array([10, 10, 10, 10], dtype=np.float32),
                "std": dim_stds,
            },
        }
        aug.prepare([], stats=stats)

        base_action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        diffs = []
        for _ in range(10000):
            frame = {"action": base_action.copy(), "task": "test"}
            result = aug.apply_frame(frame, metadata)
            diffs.append(result["action"] - base_action)

        diffs = np.array(diffs)
        # Each dimension's noise std should be noise_std * dim_std
        for d in range(4):
            expected = 0.01 * dim_stds[d]
            actual = diffs[:, d].std()
            np.testing.assert_almost_equal(actual, expected, decimal=3,
                err_msg=f"Dim {d}: expected std {expected}, got {actual}")
