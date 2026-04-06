"""Action noise augmentation: add Gaussian noise to action vectors."""

import numpy as np

from augmentations import register
from augmentations.base import Augmentation


@register("action_noise")
class ActionNoiseAugmentation(Augmentation):
    """Add Gaussian noise to action vectors for regularization."""

    def __init__(self, noise_std: float = 0.01):
        self.noise_std = noise_std

    @property
    def name(self) -> str:
        return "action_noise"

    def apply_frame(self, frame_dict: dict, metadata: dict) -> dict:
        result = dict(frame_dict)
        if "action" in result:
            noise = np.random.normal(0, self.noise_std, result["action"].shape)
            result["action"] = (result["action"] + noise).astype(result["action"].dtype)
        return result
