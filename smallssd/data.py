import torch
from pathlib import Path
import json
from PIL import Image

from torchvision.transforms import functional as F

from .config import (
    DATAFOLDER_PATH,
    EVAL_DATAFOLDER_NAME,
    IMAGES,
    TRAINING_DATAFOLDER_NAME,
    UNLABELLED_DATAFOLDER_NAME,
    WEED_ANNOTATIONS,
    CROP_ANNOTATIONS,
)
from .keys import LabelKeys, RawLabelKeys, RawBoxKeys, CLASSNAME_TO_IDX
from .utils import download_and_extract_archive

from typing import Callable, Dict, List, Tuple, Optional


class BaseDataset:
    def __init__(
        self,
        root: Path,
        datafolder_name: str,
        transforms: Optional[Callable] = None,
        download: bool = True,
    ):
        self.root = root
        self.datafolder = root / datafolder_name
        if not self.datafolder.exists():
            if download:
                download_and_extract_archive(root, datafolder_name)
            else:
                raise FileNotFoundError(
                    f"{datafolder_name} does not exist in {root}, "
                    f"it can be downloaded by setting download=True"
                )

        self.image_paths = list((self.datafolder / IMAGES).glob("*.jpg"))
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        raise NotImplementedError


class LabelledData(BaseDataset):
    r"""
    A dataset object for the labelled data collected by the
    Small Robot Company.

    This class returns images and bounding box annotations (in the format
    expected by torchvision object detection models). Specfically, a tuple containing
    the image and a dictionary of annotations is returned.

    The dictionary contains:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in
            [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and
            0 <= y1 < y2 <= H.
        - labels (Int64Tensor[N]): the class label for each ground-
            truth box
    """

    def __init__(
        self,
        root: Path = DATAFOLDER_PATH,
        transforms: Optional[Callable] = None,
        eval: bool = False,
        download: bool = True,
    ):
        super().__init__(
            root=root,
            datafolder_name=EVAL_DATAFOLDER_NAME if eval else TRAINING_DATAFOLDER_NAME,
            transforms=transforms,
            download=download,
        )
        self.eval = eval

    @staticmethod
    def load_json(json_path: Path) -> Dict[str, torch.Tensor]:
        r"""
        During training, the model expects both the input
        tensors, as well as a targets (list of dictionary),
        containing:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in
            [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and
            0 <= y1 < y2 <= H.
        - labels (Int64Tensor[N]): the class label for each ground-
            truth box
        """
        with json_path.open("r") as f:
            labels_dict = json.load(f)

        labels_list = labels_dict[RawLabelKeys.LABELS]
        if len(labels_list) == 0:
            return {"boxes": torch.tensor([]).float(), "labels": torch.tensor([]).int()}
        boxes, labels = [], []
        for label in labels_list:
            boxes.append(
                [
                    label[RawBoxKeys.X1],
                    label[RawBoxKeys.Y1],
                    label[RawBoxKeys.X2],
                    label[RawBoxKeys.Y2],
                ]
            )
            labels.append(CLASSNAME_TO_IDX[label[RawBoxKeys.CLASSNAME]])

        return {
            LabelKeys.BOXES: torch.tensor(boxes).float(),
            LabelKeys.LABELS: torch.tensor(labels).int(),
        }

    @staticmethod
    def combine_annotations(
        annos: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        output_dict = {}
        for key in annos[0].keys():
            output_dict[key] = torch.cat([anno[key] for anno in annos], dim=0)
        return output_dict

    @staticmethod
    def annotation_paths_from_image_path(
        datafolder_path: Path, image_path: Path
    ) -> Tuple[Path, Path]:
        image_name = image_path.stem
        return (
            datafolder_path / CROP_ANNOTATIONS / f"{image_name}.json",
            datafolder_path / WEED_ANNOTATIONS / f"{image_name}.json",
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        image_path = self.image_paths[idx]
        annotations_paths = self.annotation_paths_from_image_path(
            self.datafolder, image_path
        )
        annos = self.combine_annotations(
            [self.load_json(annotation_path) for annotation_path in annotations_paths]
        )

        img = F.pil_to_tensor(Image.open(image_path)).float() / 255
        if self.transforms is not None:
            img, annos = self.transforms(img, annos)

        return img, annos


class UnlabelledData(BaseDataset):
    r"""
    A dataset object for the unlabelled data collected by the
    Small Robot Company.

    This class returns unlabelled images as tensors.
    """

    def __init__(
        self,
        root: Path = DATAFOLDER_PATH,
        transforms: Optional[Callable] = None,
        download: bool = True,
    ):
        super().__init__(
            root=root,
            datafolder_name=UNLABELLED_DATAFOLDER_NAME,
            transforms=transforms,
            download=download,
        )

    def __getitem__(self, idx: int) -> torch.Tensor:
        image_path = self.image_paths[idx]
        img = F.pil_to_tensor(Image.open(image_path)).float() / 255
        if self.transforms is not None:
            img = self.transforms(img)
        return img
