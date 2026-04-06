"""Core pipeline: download source dataset, apply augmentations, upload to Hub.

Memory management: The pipeline processes data episode-by-episode, never loading
the entire dataset into RAM. Each episode's frames are read one at a time from
the source (video-decoded on the fly), transformed, written to the target via
add_frame(), then flushed to disk on save_episode(). This keeps memory usage
proportional to a single frame (~3.5MB for 4x 480x640 images), not the dataset size.

Physical integrity: Stats (mean/std/min/max for all features) are recomputed
incrementally by save_episode() for each episode written. The final stats.json
in the augmented dataset reflects the augmented data, so normalization layers
in downstream VLA training will use correct statistics.
"""

import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME

import config
from utils import get_camera_keys, get_non_default_feature_keys, get_visualizer_url


def frame_to_add_dict(frame: dict, camera_keys: list[str], feature_keys: list[str], features: dict) -> dict:
    """Convert a frame from __getitem__ format to add_frame format.

    - Images: (C,H,W) float32 [0,1] tensor -> (H,W,C) uint8 numpy
    - Scalars/vectors: tensor -> numpy
    - task: kept as string
    - Default features (episode_index, frame_index, index, task_index) are excluded.
    """
    out = {}
    for key in feature_keys:
        val = frame[key]
        if key in camera_keys:
            # (C, H, W) float32 [0,1] -> (H, W, C) uint8
            out[key] = (val.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        elif isinstance(val, torch.Tensor):
            expected_shape = features.get(key, {}).get("shape")
            arr = val.numpy()
            # Reshape scalar tensors to match expected shape (e.g. () -> (1,))
            if expected_shape and arr.shape != tuple(expected_shape):
                arr = arr.reshape(expected_shape)
            out[key] = arr
        else:
            out[key] = val
    out["task"] = frame["task"]
    return out


def build_features_for_create(src_features: dict) -> dict:
    """Build the features dict for LeRobotDataset.create from source features.

    Strips out default auto-managed features (episode_index, frame_index, index, task_index, timestamp)
    since create() adds them automatically.
    """
    default = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    return {k: v for k, v in src_features.items() if k not in default}


def get_episode_frame_range(src: LeRobotDataset, episode_idx: int) -> tuple[int, int]:
    """Get the (from_index, to_index) global frame range for an episode."""
    ep = src.meta.episodes[episode_idx]
    return ep["dataset_from_index"], ep["dataset_to_index"]


def run_pipeline(
    source: str,
    target: str,
    augmentations: list | None = None,
    episodes: list[int] | None = None,
    dry_run: bool = False,
    private: bool = False,
):
    """Run the full augmentation pipeline.

    Args:
        source: Source dataset repo_id (e.g. 'lerobot/aloha_static_cups_open')
        target: Target dataset repo_id (e.g. 'hedemil/aloha_cups_augmented')
        augmentations: List of augmentation instances to apply. None = identity copy.
        episodes: Optional list of episode indices to process. None = all.
        dry_run: If True, skip push_to_hub.
        private: If True, upload as private dataset.
    """
    if augmentations is None:
        augmentations = []

    print(f"Loading source dataset: {source}")
    src = LeRobotDataset(source)

    total_episodes = src.meta.episodes.num_rows
    episode_indices = episodes if episodes is not None else list(range(total_episodes))
    print(f"Processing {len(episode_indices)} episodes (of {total_episodes} total)")

    camera_keys = get_camera_keys(src.meta.features)
    feature_keys = get_non_default_feature_keys(src.meta.features)
    create_features = build_features_for_create(src.meta.features)

    print(f"Camera keys: {camera_keys}")
    print(f"Creating target dataset: {target}")

    # Clean up any existing local cache for target to avoid conflicts
    target_root = HF_LEROBOT_HOME / target
    if target_root.exists():
        shutil.rmtree(target_root)

    dst = LeRobotDataset.create(
        repo_id=target,
        fps=src.meta.fps,
        features=create_features,
        robot_type=src.meta.info.get("robot_type"),
    )

    aug_names = [a.name for a in augmentations] if augmentations else ["identity"]
    print(f"Augmentations: {aug_names}")

    # Detect robot type and fetch its configuration
    robot_type = src.meta.info.get("robot_type", "unknown")
    robot_cfg = config.ROBOT_CONFIGS.get(robot_type, {})
    if robot_cfg:
        print(f"Detected robot: {robot_type} (using preset)")
    else:
        print(f"Detected robot: {robot_type} (no preset found, using defaults)")

    # Prepare augmentations that need upfront data (e.g. instruction variation needs task list)
    all_tasks = list(src.meta.tasks.index)  # task strings
    for aug in augmentations:
        if hasattr(aug, "prepare"):
            print(f"Preparing augmentation: {aug.name}")
            aug.prepare(all_tasks, robot_cfg=robot_cfg)

    for ep_idx in tqdm(episode_indices, desc="Episodes"):
        from_idx, to_idx = get_episode_frame_range(src, ep_idx)
        num_frames = to_idx - from_idx

        # First pass: original (identity) copy
        for global_idx in tqdm(range(from_idx, to_idx), desc=f"Ep {ep_idx} (original)", leave=False):
            frame = src[global_idx]
            frame_dict = frame_to_add_dict(frame, camera_keys, feature_keys, src.meta.features)
            dst.add_frame(frame_dict)
        dst.save_episode()

        # Additional passes: one per augmentation
        for aug in augmentations:
            if hasattr(aug, "on_episode_start"):
                aug.on_episode_start()
            for global_idx in tqdm(
                range(from_idx, to_idx), desc=f"Ep {ep_idx} ({aug.name})", leave=False
            ):
                frame = src[global_idx]
                frame_dict = frame_to_add_dict(frame, camera_keys, feature_keys, src.meta.features)
                frame_dict = aug.apply_frame(frame_dict, {"camera_keys": camera_keys})
                dst.add_frame(frame_dict)
            dst.save_episode()

    print("Finalizing dataset...")
    dst.finalize()

    if dry_run:
        print(f"Dry run — skipping upload. Local dataset at: {dst.root}")
    else:
        print(f"Pushing to Hub: {target}")
        dst.push_to_hub(private=private)
        print(f"\nDone! Visualize at:\n  {get_visualizer_url(target)}")

    return dst
