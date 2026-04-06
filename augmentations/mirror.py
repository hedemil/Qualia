"""Episode mirroring augmentation: horizontal flip + left/right arm swap."""

import numpy as np

from augmentations import register
from augmentations.base import Augmentation


@register("mirror")
class MirrorAugmentation(Augmentation):
    """Horizontally flip images and swap left/right arm joints.

    For bimanual robots (e.g. ALOHA), this swaps the left and right arm
    joint values and flips the sign of joints whose positive direction
    reverses under mirroring (waist, forearm_roll, wrist_rotate).
    Also swaps left/right camera views if present.
    """

    def __init__(
        self,
        arm_size: int = 7,
        sign_flip_within_arm: list[int] | None = None,
        camera_swap_pairs: list[tuple[str, str]] | None = None,
    ):
        """
        Args:
            arm_size: Number of joints per arm (default 7 for ALOHA).
            sign_flip_within_arm: Indices within each arm to sign-flip.
                Default for ALOHA: [0, 3, 5] (waist, forearm_roll, wrist_rotate).
            camera_swap_pairs: Pairs of camera keys to swap.
                Default: [("observation.images.cam_left_wrist", "observation.images.cam_right_wrist")].
        """
        self.arm_size = arm_size
        self.sign_flip_within_arm = sign_flip_within_arm or [0, 3, 5]
        self.camera_swap_pairs = camera_swap_pairs or [
            ("observation.images.cam_left_wrist", "observation.images.cam_right_wrist"),
        ]

        # Precompute absolute sign-flip indices for the full joint vector
        self._sign_flip_indices = []
        for offset in [0, arm_size]:
            for idx in self.sign_flip_within_arm:
                self._sign_flip_indices.append(offset + idx)

    @property
    def name(self) -> str:
        return "mirror"

    def _swap_arms(self, arr: np.ndarray) -> np.ndarray:
        """Swap left and right arm joints, then sign-flip mirrored joints."""
        out = arr.copy()
        left = arr[: self.arm_size].copy()
        right = arr[self.arm_size : 2 * self.arm_size].copy()
        out[: self.arm_size] = right
        out[self.arm_size : 2 * self.arm_size] = left
        out[self._sign_flip_indices] *= -1
        return out

    def apply_frame(self, frame_dict: dict, metadata: dict) -> dict:
        result = dict(frame_dict)

        # Flip all camera images horizontally
        for key in metadata["camera_keys"]:
            if key in result:
                result[key] = np.ascontiguousarray(np.fliplr(result[key]))

        # Swap left/right camera pairs
        for cam_a, cam_b in self.camera_swap_pairs:
            if cam_a in result and cam_b in result:
                result[cam_a], result[cam_b] = result[cam_b], result[cam_a]

        # Swap and sign-flip joints in state and action
        for key in ["observation.state", "action"]:
            if key in result:
                result[key] = self._swap_arms(result[key])

        return result
