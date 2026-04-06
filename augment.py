#!/usr/bin/env python3
"""CLI entry point for LeRobot dataset augmentation.

Supports two modes:
  1. CLI args:   python augment.py --target user/ds --augmentations mirror,visual
  2. YAML config: python augment.py --config configs/heavy_visual_noise.yaml --target user/ds

CLI args override YAML values when both are provided.
"""

import argparse
import os
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi


def load_env():
    """Load .env file if it exists (simple key=value parser, no dependency needed)."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


load_env()

from augmentations import get_augmentation, REGISTRY
from pipeline import run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Augment a LeRobot v3 dataset and upload to HuggingFace Hub.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file (e.g. configs/heavy_visual_noise.yaml)",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Source dataset repo_id (default: lerobot/aloha_static_cups_open)",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Target dataset repo_id (e.g. hedemil/aloha_cups_augmented)",
    )
    parser.add_argument(
        "--augmentations",
        default=None,
        help="Comma-separated list of augmentations to apply (e.g. mirror,visual,action_noise)",
    )
    parser.add_argument(
        "--episodes",
        default=None,
        help="Comma-separated list of episode indices to process (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip upload to Hub (local dataset only)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Upload as private dataset",
    )
    return parser.parse_args()


def load_yaml_config(config_path: str) -> dict:
    """Load and validate a YAML config file."""
    path = Path(config_path)
    if not path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    print(f"Loaded config: {config_path}")
    return cfg or {}


def build_augmentations_from_yaml(aug_configs: list[dict]) -> list:
    """Build augmentation instances from YAML config entries.

    Each entry has 'name' and optional 'params' dict.
    Example YAML:
        augmentations:
          - name: visual
            params:
              brightness: 0.5
              contrast: 0.5
    """
    aug_list = []
    for entry in aug_configs:
        name = entry["name"]
        params = entry.get("params", {})
        aug_list.append(get_augmentation(name, **params))
    return aug_list


def build_augmentations_from_cli(aug_string: str) -> list:
    """Build augmentation instances from comma-separated CLI string (default params)."""
    aug_list = []
    for name in aug_string.split(","):
        name = name.strip()
        if name:
            aug_list.append(get_augmentation(name))
    return aug_list


def check_hf_auth():
    """Verify HuggingFace authentication."""
    try:
        info = HfApi().whoami()
        print(f"Authenticated as: {info['name']}")
    except Exception as e:
        print(f"Error: Not authenticated with HuggingFace Hub: {e}")
        print("Run: huggingface-cli login")
        sys.exit(1)


def main():
    args = parse_args()

    # Load YAML config if provided
    yaml_cfg = {}
    if args.config:
        yaml_cfg = load_yaml_config(args.config)

    # Resolve settings: CLI args override YAML values
    source = args.source or yaml_cfg.get("source", "lerobot/aloha_static_cups_open")
    target = args.target or yaml_cfg.get("target")

    if not target:
        print("Error: --target is required (either via CLI or YAML config)")
        sys.exit(1)

    if not args.dry_run:
        check_hf_auth()

    # Build augmentations: CLI string takes precedence over YAML list
    if args.augmentations:
        aug_list = build_augmentations_from_cli(args.augmentations)
    elif "augmentations" in yaml_cfg:
        aug_list = build_augmentations_from_yaml(yaml_cfg["augmentations"])
    else:
        aug_list = []

    # Parse episodes: CLI takes precedence over YAML
    episode_list = None
    if args.episodes:
        episode_list = [int(x.strip()) for x in args.episodes.split(",")]
    elif "episodes" in yaml_cfg:
        episode_list = yaml_cfg["episodes"]

    run_pipeline(
        source=source,
        target=target,
        augmentations=aug_list,
        episodes=episode_list,
        dry_run=args.dry_run,
        private=args.private,
    )


if __name__ == "__main__":
    main()
