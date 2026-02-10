import argparse

from genrl.config import load_config
from genrl.trainer import get_trainer  # type: ignore


def main():
    """Entry point: load config, log parameters, and run selected trainer.

    Args:
        None directly; CLI args are parsed within.

    Returns:
        None. Side effects: starts training.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    cfg = load_config(args.config)
    trainer_cls = get_trainer(getattr(cfg, "trainer", "wan"))
    trainer = trainer_cls(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
