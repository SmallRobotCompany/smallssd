from pathlib import Path
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from random import shuffle

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from smallssd.data import LabelledData
from smallssd.config import DATAFOLDER_PATH
from smallssd.keys import LabelKeys

from .config import BATCH_SIZE

from typing import Callable, List, Tuple

SHARING_STRATEGY = "file_system"

torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)


def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)


def train_val_augmentations() -> Callable:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["labels"], min_visibility=0.3
        ),
    )


def test_augmentations() -> Callable:
    return A.Compose(
        [
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )


class AugmentedDataset(LabelledData):
    def __init__(
        self,
        root: Path = DATAFOLDER_PATH,
        augmentations: Callable = train_val_augmentations(),
        eval: bool = False,
    ) -> None:
        super().__init__(root=root, transforms=None, eval=eval)

        self.augmentations = augmentations

    def __getitem__(self, idx: int):
        img, annos = super().__getitem__(idx)
        image_dict = {
            "image": img.permute(1, 2, 0).numpy(),
            "bboxes": annos[LabelKeys.BOXES],
            "labels": annos[LabelKeys.LABELS],
        }
        if self.eval:
            image_dict_processed = self.augmentations(**image_dict)
            filtered_boxes, filtered_labels = (
                image_dict_processed["bboxes"],
                image_dict_processed["labels"],
            )
        else:
            num_targets = 0
            while num_targets == 0:
                image_dict_processed = self.augmentations(**image_dict)
                filtered_boxes, filtered_labels = self.filter_targets(
                    image_dict_processed["bboxes"], image_dict_processed["labels"]
                )
                num_targets = len(filtered_boxes)

        image = image_dict_processed["image"]
        target = {
            LabelKeys.BOXES: torch.as_tensor(filtered_boxes, dtype=torch.float32),
            LabelKeys.LABELS: torch.as_tensor(filtered_labels, dtype=torch.int64),
        }
        return image, target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

    @staticmethod
    def has_length_zero_side_rounded(bbox: Tuple[float, float, float, float]) -> bool:
        """
        Remove bounding boxes of size <1px
        """
        xmin, ymin, xmax, ymax = bbox
        if (round(xmin) == round(xmax)) or (round(ymin) == round(ymax)):
            return True
        return False

    @classmethod
    def filter_targets(
        cls, bboxes: List[Tuple], labels: List[torch.Tensor]
    ) -> Tuple[List[Tuple], List[torch.Tensor]]:
        output_boxes, output_labels = [], []
        for bbox, label in zip(bboxes, labels):
            if not cls.has_length_zero_side_rounded(bbox):
                output_boxes.append(bbox)
                output_labels.append(label)
        return output_boxes, output_labels

    @classmethod
    def split(
        cls, root: Path, augmentations: Callable, eval: bool, ratio: float = 0.2
    ) -> Tuple["AugmentedDataset", "AugmentedDataset"]:
        ds1 = cls(root=root, augmentations=augmentations, eval=eval)
        ds2 = cls(root=root, augmentations=augmentations, eval=eval)

        image_paths = ds1.image_paths
        shuffle(image_paths)
        ds2_size = int(len(image_paths) * ratio)

        ds2_paths = image_paths[:ds2_size]
        ds1_paths = image_paths[ds2_size:]

        ds1.image_paths = ds1_paths
        ds2.image_paths = ds2_paths

        return ds1, ds2


class SmallSSDDataModule(LightningDataModule):
    def __init__(self, num_workers: int = 0):
        super().__init__()
        self.num_workers = num_workers
        self.train_ds, self.val_ds, self.test_ds = self.make_datasets()

    def make_datasets(
        self,
    ) -> Tuple[AugmentedDataset, AugmentedDataset, AugmentedDataset]:
        train_ds, val_ds = AugmentedDataset.split(
            root=DATAFOLDER_PATH, augmentations=train_val_augmentations(), eval=False
        )
        return (
            train_ds,
            val_ds,
            AugmentedDataset(
                root=DATAFOLDER_PATH, eval=True, augmentations=test_augmentations()
            ),
        )

    @staticmethod
    def make_dataloader(dataset: AugmentedDataset, num_workers: int, shuffle: bool):
        return DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            collate_fn=dataset.collate_fn,
            num_workers=num_workers,
            # https://github.com/pytorch/pytorch/issues/11201#issuecomment-895047235
            worker_init_fn=set_worker_sharing_strategy,
        )

    def train_dataloader(self):
        return self.make_dataloader(self.train_ds, self.num_workers, shuffle=True)

    def val_dataloader(self):
        return self.make_dataloader(self.val_ds, self.num_workers, shuffle=False)

    def test_dataloader(self):
        return self.make_dataloader(self.test_ds, self.num_workers, shuffle=False)
