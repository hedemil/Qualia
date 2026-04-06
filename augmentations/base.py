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
