# LeRobot Dataset Augmentation Tool

A CLI tool that augments [LeRobot v3](https://github.com/huggingface/lerobot) robotics datasets and uploads the results to HuggingFace Hub.

## What it does

Takes a source LeRobot dataset, applies one or more augmentations, and produces a new dataset containing both the original and augmented episodes. The augmented dataset is automatically uploaded to HuggingFace Hub with a visualizer link.

### Available Augmentations

| Augmentation | Name | Description |
|---|---|---|
| Episode Mirroring | `mirror` | Horizontally flips camera images and swaps left/right arm joints. For bimanual robots (ALOHA), this doubles the dataset with physically meaningful variation. |
| Visual Jitter | `visual` | Applies random color jitter (brightness, contrast, saturation) and Gaussian blur. Parameters are sampled once per episode for temporal consistency. |
| Action/State Noise | `action_noise` | Adds Gaussian noise to both `action` and `observation.state` vectors for regularization. |
| Instruction Variation | `instruction` | Uses Claude API to generate paraphrased task descriptions (e.g., "Pick up the cup" becomes "Grasp the mug"). Requires `ANTHROPIC_API_KEY`. |

Augmentations can be combined: `--augmentations mirror,visual,action_noise,instruction`

## Setup

```bash
conda create -n qualia python=3.11 -y
conda activate qualia
pip install -r requirements.txt
```

### Environment variables

```bash
cp .env.example .env
# Edit .env with your keys:
#   HF_TOKEN — HuggingFace token with write access
#   ANTHROPIC_API_KEY — required only for instruction augmentation
```

Then log in to HuggingFace:
```bash
huggingface-cli login
```

## Usage

```bash
# Identity copy (no augmentation) — useful for testing the pipeline
python augment.py --source lerobot/aloha_static_cups_open \
                  --target <your-username>/aloha_cups_copy \
                  --dry-run

# Mirror augmentation on all episodes
python augment.py --source lerobot/aloha_static_cups_open \
                  --target <your-username>/aloha_cups_mirrored \
                  --augmentations mirror

# Multiple augmentations on specific episodes
python augment.py --source lerobot/aloha_static_cups_open \
                  --target <your-username>/aloha_cups_augmented \
                  --augmentations mirror,visual,action_noise \
                  --episodes 0,1,2

# With instruction variation (requires ANTHROPIC_API_KEY)
python augment.py --source lerobot/aloha_static_cups_open \
                  --target <your-username>/aloha_cups_full \
                  --augmentations mirror,visual,action_noise,instruction

# All options
python augment.py --help
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--source` | `lerobot/aloha_static_cups_open` | Source dataset repo_id |
| `--target` | (required) | Target dataset repo_id |
| `--augmentations` | (none) | Comma-separated augmentation names |
| `--episodes` | all | Comma-separated episode indices |
| `--dry-run` | false | Skip upload, keep local only |
| `--private` | false | Upload as private dataset |

### Output

The tool prints a visualizer link on completion:
```
https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Fhedemil%2Faloha_cups_augmented%2Fepisode_0
```

## Configuration

All tunable parameters (LLM model, prompts, augmentation defaults) are centralized in **`config.py`**. Edit this file to customize behavior without modifying augmentation code.

Key settings:
- `LLM_MODEL` — which Claude model to use for instruction variation
- `LLM_PARAPHRASE_PROMPT` — the prompt template for generating paraphrases
- `VISUAL_*` — brightness, contrast, saturation, blur parameters
- `ACTION_NOISE_STD` — noise standard deviation
- `MIRROR_*` — arm size, sign-flip indices, camera swap pairs

## Example Result

Dataset with mirror augmentation: [hedemil/aloha_cups_augmented](https://huggingface.co/datasets/hedemil/aloha_cups_augmented)

## How it works

1. Downloads the source dataset using the LeRobotDataset API
2. Creates a new dataset with the same schema (features, fps, robot type)
3. For each episode: copies the original, then produces one augmented copy per augmentation
4. Stats are computed incrementally per episode via `save_episode()`
5. Calls `finalize()` to close parquet writers
6. Pushes to HuggingFace Hub

### Mirror augmentation details

For ALOHA's 14-DOF bimanual setup:
- Camera images are horizontally flipped
- Left arm joints (indices 0-6) are swapped with right arm joints (indices 7-13)
- Joints whose positive direction reverses under mirroring (waist, forearm_roll, wrist_rotate) get sign-flipped
- Left/right wrist cameras are swapped

### Instruction variation details

- Calls Claude API once per unique task at startup to generate paraphrases
- Paraphrases are cached — no repeated API calls during frame processing
- Each augmented episode gets a randomly selected paraphrase
- Prompt and model are configurable in `config.py`

## Performance

- ~14 frames/sec processing speed (dominated by video decode + AV1 re-encode)
- 1 episode (~400 frames) takes ~35 seconds for identity copy
- With mirror augmentation: ~70 seconds per episode (2 copies)
- Full 50-episode dataset with mirror: ~1 hour

## AI Agent Usage

This project was built using Claude Code (Claude Opus 4.6) as an AI coding agent. The workflow:

1. **Planning phase**: Used Claude's plan mode to explore the LeRobot v3 dataset format, inspect the API signatures (`LeRobotDataset.create`, `add_frame`, `save_episode`, `finalize`), and design the architecture
2. **Iterative development**: Built the pipeline skeleton first (identity copy), validated it works end-to-end, then added augmentations one at a time
3. **Debugging**: Claude inspected actual frame data shapes and types from the dataset to understand the exact conversion needed (e.g., `(C,H,W)` float32 tensors to `(H,W,C)` uint8 numpy arrays)
4. **API exploration**: Used Python introspection (`inspect.getsource`, `inspect.signature`) to understand undocumented API behavior (e.g., `DEFAULT_FEATURES` that are auto-managed, `validate_frame` requirements)

The agent handled everything from initial research through implementation, testing, and deployment. Each phase was validated before moving to the next.

## Project Structure

```
augment.py              # CLI entry point
pipeline.py             # Core download-transform-upload loop
config.py               # Central configuration (models, prompts, parameters)
utils.py                # Helpers (visualizer URL, feature extraction)
augmentations/
    __init__.py          # Augmentation registry
    base.py              # Abstract base class
    mirror.py            # Episode mirroring (flip + arm swap)
    visual.py            # Color jitter + Gaussian blur
    action_noise.py      # Gaussian noise on action + state
    instruction.py       # LLM-based task rewriting
.env.example             # Template for API keys
requirements.txt
```
