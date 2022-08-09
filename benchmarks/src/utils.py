import importlib
import torch
from tqdm import tqdm
from urllib.request import urlopen, Request

from typing import Any

from .config import NUM_OUTPUT_CLASSES


def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """
    https://github.com/kedro-org/kedro/blob/main/kedro/utils.py

    Extract an object from a given path.
    Args:
        obj_path: Path to an object to be extracted, including the object name.
        default_obj_path: Default object path.
    Returns:
        Extracted object.
    Raises:
        AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object `{obj_name}` cannot be loaded from `{obj_path}`.")
    return getattr(module_obj, obj_name)


def load_torchvision_model(
    base_name: str = "torchvision.models.detection.fasterrcnn_resnet50_fpn",
    head_name: str = "torchvision.models.detection.faster_rcnn.FastRCNNPredictor",
    pretrained: bool = True,
) -> torch.nn.Module:
    model = load_obj(base_name)
    model = model(pretrained=pretrained)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    head = load_obj(head_name)

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = head(in_features, NUM_OUTPUT_CLASSES)

    return model


def download_from_url(url: str, filename: str, chunk_size: int = 1024) -> None:
    with open(filename, "wb") as fh:
        with urlopen(Request(url)) as response:
            with tqdm(total=response.length) as pbar:
                for chunk in iter(lambda: response.read(chunk_size), ""):
                    if not chunk:
                        break
                    pbar.update(chunk_size)
                    fh.write(chunk)
