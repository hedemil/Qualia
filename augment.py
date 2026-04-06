#!/usr/bin/env python3
"""CLI entry point for LeRobot dataset augmentation."""

import argparse
import os
import sys
from pathlib import Path

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
        "--source",
        default="lerobot/aloha_static_cups_open",
        help="Source dataset repo_id (default: lerobot/aloha_static_cups_open)",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Target dataset repo_id (e.g. hedemil/aloha_cups_augmented)",
    )
    parser.add_argument(
        "--augmentations",
        default="",
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

    if not args.dry_run:
        check_hf_auth()

    # Parse augmentations
    aug_list = []
    if args.augmentations:
        for name in args.augmentations.split(","):
            name = name.strip()
            if name:
                aug_list.append(get_augmentation(name))

    # Parse episodes
    episode_list = None
    if args.episodes:
        episode_list = [int(x.strip()) for x in args.episodes.split(",")]

    run_pipeline(
        source=args.source,
        target=args.target,
        augmentations=aug_list,
        episodes=episode_list,
        dry_run=args.dry_run,
        private=args.private,
    )


if __name__ == "__main__":
    main()
