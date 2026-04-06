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
#
# Prompt design informed by "Enhancing Linguistic Generalization of VLA:
# Fine-Tuning OpenVLA via Synthetic Instruction Augmentation" (Shin, 2025,
# arxiv:2603.16044), which showed that structured linguistic diversity
# (imperative/goal-oriented/conditional forms, abstraction levels, synonym
# variation) improves VLA generalization over simple paraphrasing.
LLM_PARAPHRASE_PROMPT = (
    "You are a linguistic expert specializing in robotic task annotation. "
    "Generate exactly {num_paraphrases} diverse natural language instructions "
    "that are semantically equivalent to the original robot task instruction below.\n\n"
    "Requirements:\n"
    "- Ensure linguistic variety: use different sentence structures "
    "(imperative, goal-oriented, and conditional forms)\n"
    "- Vary the level of abstraction: include instructions ranging from "
    "low-level motor descriptions to high-level intent\n"
    "- Use vocabulary diversity: use synonyms for objects (e.g., 'item', 'target', "
    "'utensil') and actions (e.g., 'grasp', 'pick up', 'relocate')\n"
    "- Each instruction must preserve the exact same physical action and goal\n"
    "- Keep them concise (one sentence each)\n\n"
    "Return ONLY the {num_paraphrases} instructions, one per line, no numbering or bullets.\n\n"
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
    "mobile_aloha": {
        "mirror": {
            "arm_size": 7,
            "sign_flip_within_arm": [0, 3, 5],
            "camera_swap_pairs": [
                ("observation.images.cam_left_wrist", "observation.images.cam_right_wrist"),
            ],
            # Note: Mobile ALOHA also has a base, but usually base movements are mirrored 
            # by sign-flipping the rotation component in the state vector if present.
        }
    },
    "so100": {
        "mirror": {
            "arm_size": 6,
            "sign_flip_within_arm": [0, 4],  # shoulder_pan (0), wrist_roll (4)
            "camera_swap_pairs": [],
        }
    },
    "koch": {
        "mirror": {
            "arm_size": 6,
            "sign_flip_within_arm": [0, 4],
            "camera_swap_pairs": [],
        }
    },
    "umi": {
        "mirror": {
            "arm_size": 7,
            "sign_flip_within_arm": [0, 2, 4, 6], # typical for Franka-style 7DOF
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
