"""Visual augmentations: color jitter and Gaussian blur, applied consistently per episode.

Physical safety: Visual augmentations modify only camera images, NOT action or state
vectors. This is safe because the robot's physical commands remain unchanged — only
the visual input varies, teaching VLA models to be robust to lighting conditions.
Parameters are sampled once per episode so all frames in an episode share the same
transform, preserving temporal consistency (no flickering artifacts).
"""

import cv2
import numpy as np

from augmentations import register
from augmentations.base import Augmentation
import config


@register("visual")
class VisualAugmentation(Augmentation):
    """Apply random color jitter (brightness, contrast, saturation) and Gaussian blur.

    All parameters are sampled once per episode (via on_episode_start)
    to ensure temporal consistency (no flickering).
    """

    def __init__(
        self,
        brightness: float = config.VISUAL_BRIGHTNESS,
        contrast: float = config.VISUAL_CONTRAST,
        saturation: float = config.VISUAL_SATURATION,
        blur_max_kernel: int = config.VISUAL_BLUR_MAX_KERNEL,
        blur_probability: float = config.VISUAL_BLUR_PROBABILITY,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.blur_max_kernel = blur_max_kernel
        self.blur_probability = blur_probability
        self._resample_params()

    @property
    def name(self) -> str:
        return "visual"

    def _resample_params(self):
        """Sample new jitter and blur parameters for this episode."""
        self._brightness_factor = 1.0 + np.random.uniform(-self.brightness, self.brightness)
        self._contrast_factor = 1.0 + np.random.uniform(-self.contrast, self.contrast)
        self._saturation_factor = 1.0 + np.random.uniform(-self.saturation, self.saturation)
        # Blur: random odd kernel size, or 0 (no blur)
        if np.random.random() < self.blur_probability:
            k = np.random.choice(range(3, self.blur_max_kernel + 1, 2))
            self._blur_kernel = int(k)
        else:
            self._blur_kernel = 0

    def _apply_jitter(self, img: np.ndarray) -> np.ndarray:
        """Apply brightness, contrast, saturation jitter and blur to (H,W,C) uint8 image."""
        img = img.astype(np.float32)

        # Brightness
        img = img * self._brightness_factor

        # Contrast: blend toward mean gray
        gray_mean = img.mean()
        img = gray_mean + (img - gray_mean) * self._contrast_factor

        # Saturation: blend toward grayscale
        gray = np.mean(img, axis=2, keepdims=True)
        img = gray + (img - gray) * self._saturation_factor

        img = np.clip(img, 0, 255).astype(np.uint8)

        # Gaussian blur
        if self._blur_kernel > 0:
            img = cv2.GaussianBlur(img, (self._blur_kernel, self._blur_kernel), 0)

        return img

    def apply_frame(self, frame_dict: dict, metadata: dict) -> dict:
        result = dict(frame_dict)
        for key in metadata["camera_keys"]:
            if key in result:
                result[key] = self._apply_jitter(result[key])
        return result

    def on_episode_start(self):
        """Resample jitter and blur params at the start of each augmented episode."""
        self._resample_params()
