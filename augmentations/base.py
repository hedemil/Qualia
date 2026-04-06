"""Base class for dataset augmentations."""

from abc import ABC, abstractmethod


class Augmentation(ABC):
    """Abstract base class for frame-level augmentations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name for this augmentation."""

    @abstractmethod
    def apply_frame(self, frame_dict: dict, metadata: dict) -> dict:
        """Apply augmentation to a single frame.

        Args:
            frame_dict: Dict with numpy arrays for each feature + 'task' string.
                        Images are (H, W, C) uint8 numpy arrays.
            metadata: Dict with 'camera_keys' list and any other context.

        Returns:
            Modified frame dict (should be a copy, not in-place).
        """

    def prepare(self, tasks: list[str], robot_cfg: dict | None = None):
        """Prepare augmentation with dataset-specific metadata.

        Args:
            tasks: List of all unique task strings in the dataset.
            robot_cfg: Dictionary with robot-specific settings from config.py.
        """
        pass

    def on_episode_start(self):
        """Optional hook called at the start of each augmented episode."""
        pass
