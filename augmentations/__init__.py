"""Augmentation registry."""

from augmentations.base import Augmentation

REGISTRY: dict[str, type[Augmentation]] = {}


def register(name: str):
    """Decorator to register an augmentation class."""
    def decorator(cls):
        REGISTRY[name] = cls
        return cls
    return decorator


def get_augmentation(name: str, **kwargs) -> Augmentation:
    """Instantiate an augmentation by name."""
    if name not in REGISTRY:
        raise ValueError(f"Unknown augmentation '{name}'. Available: {list(REGISTRY.keys())}")
    return REGISTRY[name](**kwargs)


# Import augmentation modules to trigger registration
from augmentations import mirror  # noqa: F401, E402
from augmentations import visual  # noqa: F401, E402
from augmentations import action_noise  # noqa: F401, E402
