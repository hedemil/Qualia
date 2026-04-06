"""Utility helpers for the augmentation tool."""

from urllib.parse import quote


def get_visualizer_url(repo_id: str, episode: int = 0) -> str:
    """Construct a HuggingFace LeRobot visualizer URL."""
    encoded = quote(f"/{repo_id}/episode_{episode}", safe="")
    return f"https://huggingface.co/spaces/lerobot/visualize_dataset?path={encoded}"


def get_camera_keys(features: dict) -> list[str]:
    """Extract camera/image feature keys from the dataset features dict."""
    return [k for k, v in features.items() if v.get("dtype") in ("video", "image")]


def get_non_default_feature_keys(features: dict) -> list[str]:
    """Get feature keys that need to be passed to add_frame (excluding auto-managed ones)."""
    default = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    return [k for k in features if k not in default]
