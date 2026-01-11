from video_grpo.trainer import wan_trainer


def get_trainer(name: str = "wan"):
    name = name.lower()
    if name == "wan":
        return wan_trainer.run
    raise ValueError(f"Unknown trainer: {name}")


__all__ = ["get_trainer"]
