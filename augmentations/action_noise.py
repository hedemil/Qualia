"""Action and state noise augmentation: add Gaussian noise to action and observation.state vectors.

Noise is scaled per dimension: each joint's noise std = noise_std * dimension_std,
so joints with small ranges get small noise and joints with large ranges get
proportionally larger noise. When dataset stats are available (via prepare()),
noisy values are clipped to the observed [min, max] range to prevent physically
impossible states. The augmented stats.json is automatically recomputed by the
pipeline, so normalization layers in downstream VLA training will use correct
statistics.
"""

import numpy as np

from augmentations import register
from augmentations.base import Augmentation
import config


@register("action_noise")
class ActionNoiseAugmentation(Augmentation):
    """Add per-dimension scaled Gaussian noise to action and observation.state vectors."""

    def __init__(self, noise_std: float = config.ACTION_NOISE_STD):
        self.noise_std = noise_std
        self.stats = {}

    @property
    def name(self) -> str:
        return "action_noise"

    def prepare(self, tasks, robot_cfg=None, stats=None):
        self.stats = stats or {}

    def apply_frame(self, frame_dict: dict, metadata: dict) -> dict:
        result = dict(frame_dict)
        for key in ["action", "observation.state"]:
            if key in result:
                if key in self.stats:
                    per_dim_std = self.stats[key]["std"]
                    noise = np.random.normal(0, self.noise_std * per_dim_std, result[key].shape)
                else:
                    noise = np.random.normal(0, self.noise_std, result[key].shape)
                dtype = result[key].dtype
                noisy = (result[key] + noise).astype(dtype)
                if key in self.stats:
                    noisy = np.clip(noisy, self.stats[key]["min"], self.stats[key]["max"]).astype(dtype)
                result[key] = noisy
        return result
