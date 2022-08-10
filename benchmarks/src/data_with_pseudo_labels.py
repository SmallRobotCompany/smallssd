from pathlib import Path
from random import shuffle
from itertools import chain
import pandas as pd
import numpy as np

import torch
from torchvision.transforms.functional import get_image_size

from smallssd.data import UnlabelledData
from smallssd.keys import LabelKeys, CLASSNAME_TO_IDX
from .data_with_augmentations import (
    AugmentedDataset,
    SmallSSDDataModule,
    train_val_augmentations,
)
from .config import MAX_PSUEDO_LABELLED_IMAGES

from typing import Callable, Dict, List, Optional, Tuple


class PseudoLabelledData(UnlabelledData):
    def __init__(
        self,
        root: Path,
        max_unlabelled_images: Optional[int] = MAX_PSUEDO_LABELLED_IMAGES,
        augmentations: Callable = train_val_augmentations(),
    ) -> None:
        super().__init__(root, transforms=None)

        self.augmentations = augmentations

        if max_unlabelled_images is not None:
            shuffle(self.image_paths)
            self.image_paths = self.image_paths[:max_unlabelled_images]

        self.targets: pd.DataFrame = None
        self.inference_run = False

    @staticmethod
    def _create_mask(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:

        return ((labels == CLASSNAME_TO_IDX["wheat"]) & (scores >= 0.2)) | (
            (labels == CLASSNAME_TO_IDX["weed"]) & (scores >= 0.3)
        )

    def add_targets(self, predictions: List[Dict]) -> None:
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        flat_targets = list(chain.from_iterable(predictions))
        assert len(flat_targets) == len(self)
        boxes, labels = [], []

        image_indices_without_labels = []
        for idx, target in enumerate(flat_targets):
            height, width = get_image_size(self[idx])
            boxes_np = target[LabelKeys.BOXES].cpu().numpy()

            boxes_np[:, 0:2] = np.clip(boxes_np[:, 0:2], a_min=0)
            boxes_np[:, 3] = np.clip(boxes_np[:, 3], a_max=width)
            boxes_np[:, 4] = np.clip(boxes_np[:, 4], a_max=height)

            labels_np = target[LabelKeys.LABELS].cpu().numpy()
            scores_np = target["scores"].cpu().numpy()

            mask = self._create_mask(labels_np, scores_np)
            boxes_masked = boxes_np[mask]
            if len(boxes_masked) == 0:
                image_indices_without_labels.append(idx)
            else:
                boxes.append(boxes_masked)
                labels.append(labels_np[mask])

        assert len(boxes) == (len(self) - len(image_indices_without_labels))
        self.image_paths = [
            im
            for idx, im in enumerate(self.image_paths)
            if idx not in image_indices_without_labels
        ]
        self.targets = pd.DataFrame({LabelKeys.BOXES: boxes, LabelKeys.LABELS: labels})
        self.inference_run = True

    def __getitem__(self, idx: int):
        img = super().__getitem__(idx)
        if self.inference_run:
            row = self.targets.iloc[idx]
            image_dict = {
                "image": img.permute(1, 2, 0).numpy(),
                "bboxes": row[LabelKeys.BOXES],
                "labels": row[LabelKeys.LABELS],
            }
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
        return img


class PseudoAndRealLabels:
    def __init__(
        self, real_labels: AugmentedDataset, psuedo_labels: PseudoLabelledData
    ) -> None:
        assert psuedo_labels.inference_run

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
