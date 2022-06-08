import argparse
import sys
import torch
from torch.utils.data import DataLoader
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.models import FullySupervised
from src.data_with_augmentations import SmallSSDDataModule
from src.data_with_pseudo_labels import PseudoLabelledData, update_datamodule
from src.constants import Metrics

from smallssd.config import DATAFOLDER_PATH


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

    parser.add_argument(
        "--workers", help="Number of dataloader workers", type=int, default="1"
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

    psuedo_labels = PseudoLabelledData(root=DATAFOLDER_PATH)
    psuedo_labels_dl = DataLoader(psuedo_labels, batch_size=1, shuffle=False)

    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        callbacks=[EarlyStopping(monitor=Metrics.MAP, mode="max", patience=10)],
    )

    predictions = trainer.predict(model, psuedo_labels_dl)
    psuedo_labels.add_targets(predictions)
    datamodule = update_datamodule(SmallSSDDataModule(args.workers), psuedo_labels)

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
