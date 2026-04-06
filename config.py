"""Central configuration for all tunable parameters.

Edit this file to change augmentation parameters, LLM settings, prompts, etc.
"""

# =============================================================================
# LLM Configuration (for instruction variation augmentation)
# =============================================================================

LLM_MODEL = "claude-haiku-4-5-20251001"
LLM_MAX_TOKENS = 1024

# Prompt template for generating task paraphrases.
# {num_paraphrases} and {task} will be substituted at runtime.
LLM_PARAPHRASE_PROMPT = (
    "Generate exactly {num_paraphrases} different paraphrases of this robot task instruction. "
    "Each paraphrase should use different words but preserve the exact same meaning and action. "
    "Keep them concise (one sentence each). "
    "Return ONLY the paraphrases, one per line, no numbering or bullets.\n\n"
    "Original: {task}"
)

# =============================================================================
# Visual Augmentation Defaults
# =============================================================================

VISUAL_BRIGHTNESS = 0.3
VISUAL_CONTRAST = 0.3
VISUAL_SATURATION = 0.3
VISUAL_BLUR_MAX_KERNEL = 5
VISUAL_BLUR_PROBABILITY = 0.5

# =============================================================================
# Action/State Noise Defaults
# =============================================================================

ACTION_NOISE_STD = 0.01

# =============================================================================
# Mirror Augmentation Defaults (for ALOHA bimanual robots)
# =============================================================================

MIRROR_ARM_SIZE = 7
# Indices within each arm to sign-flip under mirroring
# For ALOHA: waist (0), forearm_roll (3), wrist_rotate (5)
MIRROR_SIGN_FLIP_WITHIN_ARM = [0, 3, 5]
# Camera pairs to swap under mirroring
MIRROR_CAMERA_SWAP_PAIRS = [
    ("observation.images.cam_left_wrist", "observation.images.cam_right_wrist"),
]

# =============================================================================
# Instruction Variation Defaults
# =============================================================================

INSTRUCTION_NUM_PARAPHRASES = 5
