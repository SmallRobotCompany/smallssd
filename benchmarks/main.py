import argparse
import sys
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.data_with_augmentations import SmallSSDDataModule
from src.models import FullySupervised
from src.constants import Metrics


def parse_args(args):
    """Parse the arguments."""
    parser = argparse.ArgumentParser(
        description="Simple training script for training a pytorch lightning model."
    )

    parser.add_argument(
        "--model",
        help="Chooses model architecture",
        type=str,
        default="FRCNN",
        choices=["FRCNN", "RetinaNet", "SSD", "YOLO"],
    )
    parser.add_argument(
        "--workers", help="Number of dataloader workers", type=int, default="1"
    )

    return parser.parse_args(args)


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    pl.seed_everything(42)

    model = FullySupervised(model_base=args.model)

    trainer = pl.Trainer(
        callbacks=[EarlyStopping(monitor=Metrics.MAP, mode="max", patience=10)],
        gpus=torch.cuda.device_count(),
    )

    datamodule = SmallSSDDataModule(args.workers)
    trainer.fit(model, datamodule=datamodule)

    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
