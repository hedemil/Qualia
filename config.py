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
# Robot-Specific Configurations (Presets)
# =============================================================================

ROBOT_CONFIGS = {
    "aloha": {
        "mirror": {
            "arm_size": 7,
            "sign_flip_within_arm": [0, 3, 5],
            "camera_swap_pairs": [
                ("observation.images.cam_left_wrist", "observation.images.cam_right_wrist"),
            ],
        }
    },
    # Placeholder for other common robots
    "so_100": {
        "mirror": {
            "arm_size": 6,
            "sign_flip_within_arm": [1, 2],
            "camera_swap_pairs": [],
        }
    },
}

# Default fallbacks if robot_type is unknown or metadata is missing
MIRROR_ARM_SIZE = 7
MIRROR_SIGN_FLIP_WITHIN_ARM = [0, 3, 5]
MIRROR_CAMERA_SWAP_PAIRS = [
    ("observation.images.cam_left_wrist", "observation.images.cam_right_wrist"),
]

# =============================================================================
# Instruction Variation Defaults
# =============================================================================

INSTRUCTION_NUM_PARAPHRASES = 5
