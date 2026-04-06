# LeRobot Dataset Augmentation Tool

A CLI tool that augments [LeRobot v3](https://github.com/huggingface/lerobot) robotics datasets and uploads the results to HuggingFace Hub.

## What it does

Takes a source LeRobot dataset, applies one or more augmentations, and produces a new dataset containing both the original and augmented episodes. The augmented dataset is automatically uploaded to HuggingFace Hub with a visualizer link.

### Available Augmentations

| Augmentation | Description |
|---|---|
| `mirror` | Horizontally flips camera images and swaps left/right arm joints. For bimanual robots (ALOHA), this doubles the dataset with physically meaningful variation. |
| `visual` | Applies random color jitter (brightness, contrast, saturation) to camera images. Parameters are sampled once per episode for temporal consistency. |
| `action_noise` | Adds Gaussian noise to action vectors for regularization. |

Augmentations can be combined: `--augmentations mirror,visual,action_noise`

## Setup

```bash
conda create -n qualia python=3.11 -y
conda activate qualia
pip install -r requirements.txt
huggingface-cli login  # Need write access token
```

## Usage

```bash
# Identity copy (no augmentation) - useful for testing the pipeline
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

## Example Result

Dataset with mirror augmentation: [hedemil/aloha_cups_augmented](https://huggingface.co/datasets/hedemil/aloha_cups_augmented)

## How it works

1. Downloads the source dataset using the LeRobotDataset API
2. Creates a new dataset with the same schema (features, fps, robot type)
3. For each episode: copies the original, then produces one augmented copy per augmentation
4. Calls `finalize()` to close parquet writers and compute stats
5. Pushes to HuggingFace Hub

### Mirror augmentation details

For ALOHA's 14-DOF bimanual setup:
- Camera images are horizontally flipped
- Left arm joints (indices 0-6) are swapped with right arm joints (indices 7-13)
- Joints whose positive direction reverses under mirroring (waist, forearm_roll, wrist_rotate) get sign-flipped
- Left/right wrist cameras are swapped

## Performance

- ~14 frames/sec processing speed (dominated by video decode + AV1 re-encode)
- 1 episode (~400 frames) takes ~35 seconds for identity copy
- With mirror augmentation: ~70 seconds per episode (2 copies)
- Full 50-episode dataset with mirror: ~1 hour

## AI Agent Usage

This project was built entirely using Claude Code (Claude Opus 4.6) as an AI coding agent. The workflow:

1. **Planning phase**: Used Claude's plan mode to explore the LeRobot v3 dataset format, inspect the API signatures (`LeRobotDataset.create`, `add_frame`, `save_episode`, `finalize`), and design the architecture
2. **Iterative development**: Built the pipeline skeleton first (identity copy), validated it works end-to-end, then added augmentations one at a time
3. **Debugging**: Claude inspected actual frame data shapes and types from the dataset to understand the exact conversion needed (e.g., `(C,H,W)` float32 tensors to `(H,W,C)` uint8 numpy arrays)
4. **API exploration**: Used Python introspection (`inspect.getsource`, `inspect.signature`) to understand undocumented API behavior (e.g., `DEFAULT_FEATURES` that are auto-managed, `validate_frame` requirements)

The agent handled everything from initial research through implementation, testing, and deployment. Each phase was validated before moving to the next.

## Project Structure

```
augment.py              # CLI entry point
pipeline.py             # Core download-transform-upload loop
utils.py                # Helpers (visualizer URL, feature extraction)
augmentations/
    __init__.py          # Augmentation registry
    base.py              # Abstract base class
    mirror.py            # Episode mirroring
    visual.py            # Color jitter
    action_noise.py      # Gaussian action noise
requirements.txt
README.md
```
