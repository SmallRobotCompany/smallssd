from pathlib import Path
from random import choice, shuffle
import numpy as np

import torch
from torch import nn

from smallssd.data import UnlabelledData
from smallssd.keys import LabelKeys, CLASSNAME_TO_IDX
from .data_with_augmentations import (
    AugmentedDataset,
    SmallSSDDataModule,
    train_val_augmentations,
)
from .config import MAX_PSUEDO_LABELLED_IMAGES

from typing import Callable, Dict, Optional, Tuple


class PseudoLabelledData(UnlabelledData):
    def __init__(
        self,
        root: Path,
        teacher_model: nn.Module,
        max_unlabelled_images: Optional[int] = None,
        max_unlabelled_images_per_epoch: Optional[int] = MAX_PSUEDO_LABELLED_IMAGES,
        augmentations: Callable = train_val_augmentations(),
    ) -> None:
        super().__init__(root, transforms=None)

        self.augmentations = augmentations

        self.teacher = teacher_model
        self.teacher.eval()

        if max_unlabelled_images is not None:
            shuffle(self.image_paths)
            self.image_paths = self.image_paths[:max_unlabelled_images]
        if max_unlabelled_images_per_epoch is None:
            max_unlabelled_images_per_epoch = len(self.image_paths)
            self.mapper = None
        else:
            self.mapper = list(
                range(int(len(self.image_paths) / max_unlabelled_images_per_epoch))
            )
        self.max_unlabelled_images_per_epoch = max_unlabelled_images_per_epoch

    def __len__(self) -> int:
        return self.max_unlabelled_images_per_epoch

    @staticmethod
    def _create_mask(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:

        return ((labels == CLASSNAME_TO_IDX["wheat"]) & (scores >= 0.2)) | (
            (labels == CLASSNAME_TO_IDX["weed"]) & (scores >= 0.3)
        )

    def add_targets(self, img: torch.Tensor, model: nn.Module) -> None:
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        with torch.no_grad():
            model.eval()
            target = model([img])[0]
        boxes_np = target[LabelKeys.BOXES].cpu().numpy()
        labels_np = target[LabelKeys.LABELS].cpu().numpy()
        scores_np = target["scores"].cpu().numpy()

        mask = self._create_mask(labels_np, scores_np)
        boxes_masked = boxes_np[mask]
        if len(boxes_masked) == 0:
            return None
        else:
            return {LabelKeys.BOXES: boxes_np, LabelKeys.LABELS: labels_np[mask]}

    def __getitem__(self, idx: int):
        if self.mapper is not None:
            idx += self.max_unlabelled_images_per_epoch * choice(self.mapper)
        img = super().__getitem__(idx)

        image_dict = {
            "image": img.permute(1, 2, 0).numpy(),
        }
        image_dict.update(self.add_targets(img, self.teacher))
        image_dict_processed = self.augmentations(**image_dict)

        image = image_dict_processed["image"]
        target = {
            LabelKeys.BOXES: torch.as_tensor(
                image_dict_processed["bboxes"], dtype=torch.float32
            ),
            LabelKeys.LABELS: torch.as_tensor(
                image_dict_processed["labels"], dtype=torch.int64
            ),
        }
        return image, target


class PseudoAndRealLabels:
    def __init__(
        self, real_labels: AugmentedDataset, psuedo_labels: PseudoLabelledData
    ) -> None:

        self.real_labels = real_labels
        self.psuedo_labels = psuedo_labels

        self.collate_fn = real_labels.collate_fn

    def __len__(self) -> int:
        return len(self.real_labels) + len(self.psuedo_labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if idx < len(self.real_labels):
            return self.real_labels[idx]
        else:
            return self.psuedo_labels[idx - len(self.real_labels)]


def update_datamodule(
    module: SmallSSDDataModule, psuedo_labels: PseudoLabelledData
) -> SmallSSDDataModule:

    assert psuedo_labels.inference_run
    new_train_ds = PseudoAndRealLabels(module.train_ds, psuedo_labels)
    module.train_ds = new_train_ds
    return module
