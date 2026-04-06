# LeRobot Dataset Augmentation Tool

A CLI tool that augments [LeRobot v3](https://github.com/huggingface/lerobot) robotics datasets and uploads the results to HuggingFace Hub.

## What it does

Takes a source LeRobot dataset, applies one or more augmentations, and produces a new dataset containing both the original and augmented episodes. The augmented dataset is automatically uploaded to HuggingFace Hub with a visualizer link.

### Available Augmentations

| Augmentation | Name | Description |
|---|---|---|
| Episode Mirroring | `mirror` | Horizontally flips camera images and swaps left/right arm joints. Supports dynamic detection for **ALOHA, Mobile ALOHA, SO-100, Koch, and UMI**. |
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

### Option 1: YAML Config (recommended)

```bash
# Use a preset config — no Python code changes needed
python augment.py --config configs/heavy_visual_noise.yaml --target hedemil/my_dataset

# Override specific settings via CLI
python augment.py --config configs/default.yaml --target hedemil/my_dataset --episodes 0,1,2
```

Example YAML config (`configs/heavy_visual_noise.yaml`):
```yaml
source: lerobot/aloha_static_cups_open

augmentations:
  - name: visual
    params:
      brightness: 0.5
      contrast: 0.5
      saturation: 0.5
      blur_max_kernel: 7
      blur_probability: 0.8

  - name: action_noise
    params:
      noise_std: 0.02
```

### Option 2: CLI args

```bash
# Mirror augmentation on all episodes
python augment.py --source lerobot/aloha_static_cups_open \
                  --target hedemil/aloha_cups_mirrored \
                  --augmentations mirror

# Multiple augmentations on specific episodes
python augment.py --source lerobot/aloha_static_cups_open \
                  --target hedemil/aloha_cups_augmented \
                  --augmentations mirror,visual,action_noise \
                  --episodes 0,1,2

# Dry run (skip upload, local only)
python augment.py --target hedemil/test --augmentations mirror --episodes 0 --dry-run
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--config` | (none) | Path to YAML config file |
| `--source` | `lerobot/aloha_static_cups_open` | Source dataset repo_id |
| `--target` | (required) | Target dataset repo_id |
| `--augmentations` | (none) | Comma-separated augmentation names |
| `--episodes` | all | Comma-separated episode indices |
| `--dry-run` | false | Skip upload, keep local only |
| `--private` | false | Upload as private dataset |
| `--encoder-threads` | auto | Threads for AV1 video encoding (higher = faster) |

CLI args override YAML values when both are provided.

### Output

The tool prints a visualizer link on completion:
```
https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Fhedemil%2Faloha_cups_augmented%2Fepisode_0
```

## Configuration

### YAML Configs

Pre-built configs in `configs/`:
- `default.yaml` — mirror + visual + action noise with standard parameters
- `heavy_visual_noise.yaml` — aggressive visual augmentation (brightness 0.5, blur 0.8 probability)
- `full_pipeline.yaml` — all augmentations including LLM instruction variation
- `mirror_only.yaml` — just mirroring (fastest, doubles dataset)
- `instructions.yaml` — instruction variation only (requires `ANTHROPIC_API_KEY`)

Create your own by copying and editing any config. No Python code changes needed.

### Python Config

All default parameters and robot presets are in **`config.py`**:
- `ROBOT_CONFIGS` — Presets for **ALOHA, Mobile ALOHA, SO-100, Koch, and UMI**
- `LLM_MODEL` / `LLM_PARAPHRASE_PROMPT` — LLM settings for instruction variation
- `VISUAL_*` — brightness, contrast, saturation, blur defaults
- `ACTION_NOISE_STD` — noise standard deviation

The tool auto-detects `robot_type` from dataset metadata and applies the correct preset.

## Design Decisions

### Memory Management

The pipeline processes data **episode-by-episode**, never loading the entire dataset into RAM. Each frame is decoded from video on the fly, transformed, written via `add_frame()`, and flushed to disk on `save_episode()`. Memory usage stays proportional to a single frame (~3.5MB for 4x 480x640 images), not the dataset size. This scales to datasets of any size.

### Physical Integrity

Robotics augmentations must preserve physical consistency — corrupted data produces dangerous policies:

- **Visual augmentations** modify only camera images, never action/state vectors. This is safe because the robot's physical commands remain unchanged — only the visual input varies.
- **Mirror augmentation** jointly transforms both visual observations AND action/state vectors. Images are flipped, left/right arm joints are swapped, and rotation-direction-dependent joints are sign-flipped. This ensures augmented trajectories are physically valid.
- **Action/state noise** uses small `noise_std` (default 0.01 rad) relative to typical joint ranges. Validate appropriateness for your robot's joint limits.
- **Stats recalculation**: `save_episode()` incrementally recomputes `stats.json` (mean/std/min/max) for the augmented dataset, so normalization layers in downstream VLA training use correct statistics and don't crash or diverge.

### Robot-Specific Presets

The mirror augmentation requires knowing which joints to swap and sign-flip, which varies by robot morphology. Rather than hardcoding for one robot, `config.py` contains presets for 5 robot types that are auto-selected based on dataset metadata.

### API Resilience

The instruction variation augmentation uses exponential backoff retry (3 attempts) for transient API errors (rate limits, 502s, timeouts). On persistent failure, it **falls back to the original task text** rather than crashing — losing instruction variation on one task is acceptable, losing an hour of video encoding is not.

### Video Encoding Performance

AV1 re-encoding is the primary bottleneck (~14 fps). The tool exposes `--encoder-threads N` to increase threads for AV1 encoding and utilize multiple CPU cores. The LeRobot API's `save_episode()` already uses `ProcessPoolExecutor` for parallel video encoding internally.

### Testing

29 unit tests verify augmentation correctness, especially the kinematic math:

```bash
pytest tests/ -v
```

Key test categories:
- **Mirror kinematic tests**: arm swap, per-joint sign flip, double-flip identity (`mirror(mirror(x)) == x`)
- **Visual safety tests**: images modified but state/action vectors untouched
- **Noise statistical tests**: zero-mean, correct std, dtype preservation

## How it works

1. Downloads the source dataset using the LeRobotDataset API
2. Inspects `robot_type` metadata to select the correct augmentation parameters
3. Creates a new dataset with the same schema (features, fps, robot type)
4. For each episode: copies the original, then produces one augmented copy per augmentation
5. Stats are computed incrementally per episode via `save_episode()`
6. Calls `finalize()` to close parquet writers and pushes to HuggingFace Hub
7. Prints the HuggingFace visualizer URL

### Mirror augmentation details

Mirroring handles both camera and joint transformations:
- **Image Flipping:** All camera images are horizontally flipped
- **Camera Swapping:** Left/Right camera pairs (e.g., wrist cams) are swapped
- **Arm Swapping:** Left and right arm joint values are swapped
- **Sign Flipping:** Joints whose physical direction reverses under mirroring (e.g., waist rotation) are sign-flipped

**Presets:**
- **ALOHA:** 14-DOF (7x2), flips waist, forearm roll, and wrist rotate
- **SO-100 / Koch:** 6-DOF, flips shoulder pan and wrist roll
- **UMI:** 7-DOF arm configuration

### Instruction variation details

Informed by [**"Enhancing Linguistic Generalization of VLA: Fine-Tuning OpenVLA via Synthetic Instruction Augmentation"**](https://arxiv.org/abs/2603.16044) (Shin, 2025), which demonstrated that structured linguistic diversity improves VLA generalization. The paper showed that generating paraphrases with explicit variation along three axes — **sentence structure** (imperative/goal-oriented/conditional), **abstraction level** (motor-level vs. intent-level), and **vocabulary** (synonym substitution) — then randomly pairing them with trajectories during training, decouples linguistic patterns from rigid task labels and improves 5-bin tolerance accuracy.

Our implementation applies this approach:
- Calls Claude API once per unique task at startup with a structured prompt (see `config.py`) requesting variation across all three axes
- Paraphrases are cached — no repeated API calls during frame processing
- Each augmented episode gets a randomly selected paraphrase, matching the paper's random-pairing strategy
- Prompt template and model are configurable in `config.py`

## Performance

- ~14 frames/sec processing speed (dominated by video decode + AV1 re-encode)
- 1 episode (~400 frames) takes ~35 seconds for identity copy
- With mirror augmentation: ~70 seconds per episode (2 copies)
- Full 50-episode dataset with mirror: ~1 hour

## Example Result

Dataset with mirror augmentation: [hedemil/aloha_cups_augmented](https://huggingface.co/datasets/hedemil/aloha_cups_augmented)

## Agentic Build Log

This project was built using **Claude Code (Claude Opus 4.6)** as an AI coding agent. Below is a detailed log of the agentic workflow — not just "I used Claude," but how AI was orchestrated at each step.

### Phase 1: API Research & Architecture (Plan Mode)

**Problem:** The LeRobot v3 dataset format is not well-documented outside the source code. I needed to understand the exact API contracts before writing any pipeline code.

**Agent workflow:**
- Launched an **Explore agent** to research the LeRobot v3 format: parquet structure, meta/ folder layout, video storage, and the `LeRobotDataset` API
- Launched a **Plan agent** with the research results to design the pipeline architecture
- The Plan agent recommended using the LeRobotDataset API exclusively (vs. raw parquet manipulation), which I agreed with

**Key finding:** The agent discovered that `finalize()` is critical — without it, parquet files are corrupt (missing footer metadata). This would have been a painful debugging session without the research phase.

### Phase 2: Pipeline Skeleton (Identity Copy)

**Problem:** Before adding augmentations, I needed to validate the full download → transform → upload loop.

**Agent workflow:**
- Used `inspect.getsource(LeRobotDataset.add_frame)` and `inspect.signature(LeRobotDataset.create)` to understand the exact API contracts — the agent introspected the installed library code directly
- Discovered that `__getitem__` returns `(C,H,W)` float32 tensors in [0,1], but `add_frame()` expects `(H,W,C)` uint8 numpy arrays. The agent found this by running `ds[0]` and printing shapes/dtypes
- Hit a `ValueError: shape '()' does not have expected shape '(1,)'` on `next.done` — the agent diagnosed this was a scalar vs 1D tensor mismatch and added reshape logic using the feature schema
- Successfully ran identity copy on 1 episode (400 frames at ~14 fps), pushed to Hub, verified on visualizer

### Phase 3: Augmentations (Iterative)

**Agent workflow:**
- Built each augmentation one at a time, testing after each: mirror → visual → action_noise → instruction
- For mirror augmentation, the agent needed to understand ALOHA's joint naming convention (`left_waist`, `left_shoulder`, ..., `right_gripper`) from the dataset's `features.names` metadata to determine which indices to swap and sign-flip

### Phase 4: Config System & Polish

**Problem:** The user identified hardcoded parameters — needed a dynamic config system.

**Agent workflow:**
- Implemented `--config` flag with YAML loading, merging with CLI args
- Created example configs for different use cases (`heavy_visual_noise.yaml`, `full_pipeline.yaml`, etc.)
- Added robot-specific presets in `config.py` for 5 robot types, auto-detected from dataset metadata
- Added physical safety comments to each augmentation explaining why it's safe for VLA training

### Phase 5: Review Response

**Problem:** Self-review identified 3 gaps.

**Agent workflow:**
- **Tests:** Created `tests/` with 29 pytest tests. The critical ones: `TestDoubleFlipIdentity` asserts that `mirror(mirror(x)) == x` for state, action, and images — if the sign-flip indices are wrong, this fails. `TestSignFlip` verifies each specific joint (waist, forearm_roll, wrist_rotate) is negated. These catch the exact "poisoned dataset" scenario where a wrong sign on wrist rotation would produce physically impossible trajectories.
- **API resilience:** Added exponential backoff retry (3 attempts, 2s/4s/8s) for transient Claude API errors. On persistent failure, falls back to original task text instead of crashing. This prevents losing an hour of video encoding because of a 502 on episode 49.
- **Video encoding parallelism:** Exposed `--encoder-threads` flag that passes through to the LeRobot API's `ProcessPoolExecutor`-based video encoder. Rather than fighting the API's single-writer design with our own multiprocessing, we leverage the parallelism it already provides.

### Tools Used

- **Claude Code (Opus 4.6):** Architecture design, code generation, debugging, API research
- **Python introspection:** `inspect.getsource()`, `inspect.signature()` to reverse-engineer the LeRobot API
- **Iterative testing:** Each augmentation tested on 1-2 episodes before committing

## Project Structure

```
augment.py              # CLI entry point (YAML + CLI args)
pipeline.py             # Core download-transform-upload loop
config.py               # Central configuration (models, prompts, robot presets)
utils.py                # Helpers (visualizer URL, feature extraction)
augmentations/
    __init__.py          # Augmentation registry
    base.py              # Abstract base class
    mirror.py            # Episode mirroring (flip + arm swap)
    visual.py            # Color jitter + Gaussian blur
    action_noise.py      # Gaussian noise on action + state
    instruction.py       # LLM-based task rewriting
configs/
    default.yaml         # Standard augmentation config
    heavy_visual_noise.yaml  # Aggressive visual noise
    full_pipeline.yaml   # All augmentations
    mirror_only.yaml     # Just mirroring
    instructions.yaml    # Instruction variation only
tests/
    test_mirror.py       # Kinematic correctness tests (15 tests)
    test_visual.py       # Visual safety tests (7 tests)
    test_action_noise.py # Noise statistical tests (7 tests)
.env.example             # Template for API keys
requirements.txt
```

## Future Work & Scalability

Given more time or compute resources, the following augmentations — inspired by recent VLA literature — would significantly expand this tool's capabilities:

**Failure Scenario Injection.** Reward distillation and state-aware training require negative examples, not just successful demonstrations. An augmentation that intentionally truncates successful trajectories (e.g., dropping the last 20% of frames) or splices in repeated "stall" actions would create synthetic failure/retry data. This teaches VLA models to recognize and recover from failure states rather than only imitating success.

**VLM-Powered Hierarchical Annotations.** Our current `instruction.py` paraphrases existing task descriptions using an LLM. A more powerful approach would integrate a Vision-Language Model (e.g., Claude Sonnet) to actually *watch* the video frames and generate rich scene captions, visual grounding annotations (bounding boxes for grasp points), and step-by-step sub-task decompositions. This would produce the kind of hierarchical instruction data that methods like InstructVLA and LLaRA leverage for improved spatial reasoning.

**Consistent Visual Inpainting.** Our `visual.py` applies color jitter and blur — simple pixel-level transforms. Domain randomization could be taken much further with diffusion-based inpainting that swaps the entire background (e.g., moving the ALOHA robot from a lab to a kitchen) while maintaining temporal consistency across video frames. Epipolar-motion-aware approaches like ERMV ensure the inpainted background respects the camera geometry, preventing artifacts that would confuse a VLA model.

**Cross-Embodiment Transfer.** Given the robot-specific presets already in `config.py`, a natural extension is augmenting datasets to simulate different embodiments — retargeting joint trajectories from a 7-DOF arm to a 6-DOF arm, or synthesizing camera views for robots with different mounting positions.

## References

- Shin, D. (2025). *Enhancing Linguistic Generalization of VLA: Fine-Tuning OpenVLA via Synthetic Instruction Augmentation*. arXiv:2603.16044. [[paper]](https://arxiv.org/abs/2603.16044) — Informed our instruction variation prompt design and random-pairing strategy.
- Yu, Z. et al. (2025). *A Survey on Efficient Vision-Language-Action Models*. arXiv:2510.24795. [[paper]](https://arxiv.org/abs/2510.24795) — Survey of VLA efficiency techniques including data augmentation strategies that inspired the future work directions above.
