"""Tests for visual augmentation correctness."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from augmentations.visual import VisualAugmentation


@pytest.fixture
def visual_aug():
    return VisualAugmentation(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        blur_max_kernel=5,
        blur_probability=1.0,  # Always blur for deterministic testing
    )


@pytest.fixture
def sample_frame():
    img = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
    return {
        "observation.images.cam_high": img.copy(),
        "observation.state": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "action": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "task": "Pick up the cup",
    }


@pytest.fixture
def metadata():
    return {"camera_keys": ["observation.images.cam_high"]}


class TestVisualAugmentation:

    def test_image_modified(self, visual_aug, sample_frame, metadata):
        """Augmented image should differ from original."""
        original = sample_frame["observation.images.cam_high"].copy()
        result = visual_aug.apply_frame(sample_frame, metadata)
        assert not np.array_equal(result["observation.images.cam_high"], original)

    def test_image_shape_preserved(self, visual_aug, sample_frame, metadata):
        """Image shape and dtype should be preserved."""
        result = visual_aug.apply_frame(sample_frame, metadata)
        assert result["observation.images.cam_high"].shape == (64, 64, 3)
        assert result["observation.images.cam_high"].dtype == np.uint8

    def test_image_values_clipped(self, visual_aug, sample_frame, metadata):
        """All pixel values should be in [0, 255]."""
        result = visual_aug.apply_frame(sample_frame, metadata)
        assert result["observation.images.cam_high"].min() >= 0
        assert result["observation.images.cam_high"].max() <= 255

    def test_state_unchanged(self, visual_aug, sample_frame, metadata):
        """Visual augmentation must NOT modify state vectors."""
        original_state = sample_frame["observation.state"].copy()
        result = visual_aug.apply_frame(sample_frame, metadata)
        np.testing.assert_array_equal(result["observation.state"], original_state)

    def test_action_unchanged(self, visual_aug, sample_frame, metadata):
        """Visual augmentation must NOT modify action vectors."""
        original_action = sample_frame["action"].copy()
        result = visual_aug.apply_frame(sample_frame, metadata)
        np.testing.assert_array_equal(result["action"], original_action)

    def test_episode_consistency(self, visual_aug, sample_frame, metadata):
        """Same jitter params within an episode (no on_episode_start between calls)."""
        result1 = visual_aug.apply_frame(sample_frame, metadata)
        result2 = visual_aug.apply_frame(sample_frame, metadata)
        np.testing.assert_array_equal(
            result1["observation.images.cam_high"],
            result2["observation.images.cam_high"],
        )

    def test_episode_variation(self, visual_aug, sample_frame, metadata):
        """Different episodes should (usually) produce different jitter."""
        np.random.seed(42)
        visual_aug.on_episode_start()
        result1 = visual_aug.apply_frame(sample_frame, metadata)

        np.random.seed(123)
        visual_aug.on_episode_start()
        result2 = visual_aug.apply_frame(sample_frame, metadata)

        # With different seeds, jitter should differ
        assert not np.array_equal(
            result1["observation.images.cam_high"],
            result2["observation.images.cam_high"],
        )
