from genrl.trainer.wan_trainer import WanTrainer


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
    msg = f"Unknown trainer: {name}"
    raise ValueError(msg)


__all__ = ["get_trainer"]
