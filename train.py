import argparse
from video_grpo.config import load_config
from video_grpo.trainer import get_trainer  # type: ignore


def main():
    """Entry point: load config, log parameters, and run selected trainer.

    Args:
        None directly; CLI args are parsed within.

    Returns:
        None. Side effects: starts training.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    trainer_fn = get_trainer(getattr(cfg, "trainer", "wan"))
    trainer_fn(cfg)


if __name__ == "__main__":
    main()
