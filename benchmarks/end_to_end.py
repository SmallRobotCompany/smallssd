import argparse
import sys
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.data_with_augmentations import SmallSSDDataModule
from src.data_with_pseudo_labels import PseudoLabelledData, update_datamodule
from src.models import FullySupervised
from src.constants import Metrics

from smallssd.config import DATAFOLDER_PATH


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
        choices=["FRCNN", "RetinaNet", "SSD"],
    )
    parser.add_argument(
        "--workers", help="Number of dataloader workers", type=int, default="1"
    )
    parser.add_argument("--seed", help="Seed", type=int, default="42")

    return parser.parse_args(args)


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    pl.seed_everything(args.seed)

    # first, train a fully supervised model
    model = FullySupervised(model_base=args.model)

    trainer = pl.Trainer(
        callbacks=[EarlyStopping(monitor=Metrics.MAP, mode="max", patience=10)],
        gpus=torch.cuda.device_count(),
    )

    datamodule = SmallSSDDataModule(args.workers)
    trainer.fit(model, datamodule=datamodule)

    trainer.test(model, datamodule=datamodule)

    # then, use this model to generate predictions
    psuedo_labels = PseudoLabelledData(root=DATAFOLDER_PATH, teacher_model=model.model)

    teacher_student_trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        callbacks=[EarlyStopping(monitor=Metrics.MAP, mode="max", patience=10)],
    )

    # finally, use these new predictions to continue training the model
    datamodule = update_datamodule(SmallSSDDataModule(args.workers), psuedo_labels)

    teacher_student_trainer.fit(model, datamodule=datamodule)
    teacher_student_trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
