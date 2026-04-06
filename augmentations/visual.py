"""Visual augmentations: color jitter applied consistently per episode."""

import numpy as np

from augmentations import register
from augmentations.base import Augmentation


@register("visual")
class VisualAugmentation(Augmentation):
    """Apply random color jitter (brightness, contrast, saturation) to camera images.

    Jitter parameters are sampled once per episode (via set_episode_seed)
    to ensure temporal consistency (no flickering).
    """

    def __init__(
        self,
        brightness: float = 0.3,
        contrast: float = 0.3,
        saturation: float = 0.3,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self._resample_params()

    @property
    def name(self) -> str:
        return "visual"

    def _resample_params(self):
        """Sample new jitter parameters."""
        self._brightness_factor = 1.0 + np.random.uniform(-self.brightness, self.brightness)
        self._contrast_factor = 1.0 + np.random.uniform(-self.contrast, self.contrast)
        self._saturation_factor = 1.0 + np.random.uniform(-self.saturation, self.saturation)

    def _apply_jitter(self, img: np.ndarray) -> np.ndarray:
        """Apply brightness, contrast, and saturation jitter to an (H,W,C) uint8 image."""
        img = img.astype(np.float32)

        # Brightness
        img = img * self._brightness_factor

        # Contrast: blend toward mean gray
        gray_mean = img.mean()
        img = gray_mean + (img - gray_mean) * self._contrast_factor

        # Saturation: blend toward grayscale
        gray = np.mean(img, axis=2, keepdims=True)
        img = gray + (img - gray) * self._saturation_factor

        return np.clip(img, 0, 255).astype(np.uint8)

    def apply_frame(self, frame_dict: dict, metadata: dict) -> dict:
        result = dict(frame_dict)
        for key in metadata["camera_keys"]:
            if key in result:
                result[key] = self._apply_jitter(result[key])
        return result

    def on_episode_start(self):
        """Call at the start of each augmented episode to resample jitter params."""
        self._resample_params()
