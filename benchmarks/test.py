import argparse
import sys
import torch
from pathlib import Path

import pytorch_lightning as pl

from src.data_with_augmentations import SmallSSDDataModule
from src.models import FullySupervised


def parse_args(args):
    """Parse the arguments."""
    parser = argparse.ArgumentParser(
        description="Testing script for a pytorch lightning model."
    )

    parser.add_argument(
        "--model",
        help="Chooses model architecture",
        type=str,
        default="FRCNN",
        choices=["FRCNN", "RetinaNet", "SSD"],
    )
    parser.add_argument(
        "--version",
        help="lightning_logs version folder of the checkpoint",
        type=str,
        default="version_0",
    )

    return parser.parse_args(args)


def get_checkpoint(version: str) -> Path:
    return list(Path(f"lightning_logs/{version}/checkpoints").glob("*.ckpt"))[0]


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    pl.seed_everything(42)

    checkpoint_path = get_checkpoint(args.version)

    model = FullySupervised.load_from_checkpoint(checkpoint_path, model_base=args.model)
    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
    )
    datamodule = SmallSSDDataModule(num_workers=0)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
