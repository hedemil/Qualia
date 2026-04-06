"""Episode mirroring augmentation: horizontal flip + left/right arm swap.

Physical safety: Mirroring is physically consistent because it jointly transforms
BOTH the visual observations AND the action/state vectors. Images are flipped,
left/right arm joints are swapped, and joints whose rotation direction reverses
under mirroring (waist, forearm_roll, wrist_rotate) are sign-flipped. This ensures
the augmented data represents a valid physical trajectory, not a corrupted one.
Camera pairs (e.g. left_wrist ↔ right_wrist) are also swapped to maintain consistency.
"""

import numpy as np

from augmentations import register
from augmentations.base import Augmentation
import config


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
        arm_size: int | None = None,
        sign_flip_within_arm: list[int] | None = None,
        camera_swap_pairs: list[tuple[str, str]] | None = None,
    ):
        # Use provided values, or fallback to config defaults if prepare isn't called
        self.arm_size = arm_size if arm_size is not None else config.MIRROR_ARM_SIZE
        self.sign_flip_within_arm = sign_flip_within_arm or config.MIRROR_SIGN_FLIP_WITHIN_ARM
        self.camera_swap_pairs = camera_swap_pairs or config.MIRROR_CAMERA_SWAP_PAIRS
        self._recompute_indices()

    def _recompute_indices(self):
        """Precompute absolute sign-flip indices for the full joint vector."""
        self._sign_flip_indices = []
        for offset in [0, self.arm_size]:
            for idx in self.sign_flip_within_arm:
                self._sign_flip_indices.append(offset + idx)

    def prepare(self, tasks: list[str], robot_cfg: dict | None = None):
        """Update settings based on robot metadata."""
        if not robot_cfg:
            return

        cfg = robot_cfg.get("mirror", {})
        if not cfg:
            return

        print(f"  Reconfiguring mirror for robot-specific settings")
        if "arm_size" in cfg:
            self.arm_size = cfg["arm_size"]
        if "sign_flip_within_arm" in cfg:
            self.sign_flip_within_arm = cfg["sign_flip_within_arm"]
        if "camera_swap_pairs" in cfg:
            self.camera_swap_pairs = cfg["camera_swap_pairs"]

        self._recompute_indices()

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
