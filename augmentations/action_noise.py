"""Action and state noise augmentation: add Gaussian noise to action and observation.state vectors."""

import numpy as np

from augmentations import register
from augmentations.base import Augmentation
import config


@register("action_noise")
class ActionNoiseAugmentation(Augmentation):
    """Add Gaussian noise to action and observation.state vectors for regularization."""

    def __init__(self, noise_std: float = config.ACTION_NOISE_STD):
        self.noise_std = noise_std

    @property
    def name(self) -> str:
        return "action_noise"

    def apply_frame(self, frame_dict: dict, metadata: dict) -> dict:
        result = dict(frame_dict)
        for key in ["action", "observation.state"]:
            if key in result:
                noise = np.random.normal(0, self.noise_std, result[key].shape)
                result[key] = (result[key] + noise).astype(result[key].dtype)
        return result
