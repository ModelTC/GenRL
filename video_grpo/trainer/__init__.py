from video_grpo.trainer.wan_trainer import WanTrainer


def get_trainer(name: str = "wan"):
    """Get trainer class by name.

    Args:
        name: Name of the trainer (e.g., 'wan').

    Returns:
        Trainer class.
    """
    name = name.lower()
    if name == "wan":
        return WanTrainer
    raise ValueError(f"Unknown trainer: {name}")


__all__ = ["get_trainer"]
