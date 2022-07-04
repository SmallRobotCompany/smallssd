import torch
from torch import nn
from pathlib import Path

from torchvision.models.detection.retinanet import RetinaNetHead, retinanet_resnet50_fpn

from smallssd import data
from smallssd.keys import LabelKeys, CLASSNAME_TO_IDX

from typing import Any, Optional


def test_labelled_dataset():
    d = data.LabelledData(Path(__file__).parent / "test_data", eval=False)
    assert len(d) == 1

    x, y = d[0]

    assert isinstance(x, torch.Tensor)

    assert len(y) == 2
    assert y[LabelKeys.BOXES].shape[1] == 4
    assert y[LabelKeys.BOXES].shape[0] == y[LabelKeys.LABELS].shape[0]


def test_unlabelled_data():
    d = data.UnlabelledData(Path(__file__).parent / "test_data")
    assert len(d) == 1
    assert isinstance(d[0], torch.Tensor)


def test_data_downloads_and_unpacks_correctly(tmp_path):
    d = data.LabelledData(root=tmp_path, eval=True, download=True)
    assert len(d) == 156


def test_data_with_model():
    def _initialize_retinanet(
        num_classes: int = len(CLASSNAME_TO_IDX) + 1,
        pretrained: bool = False,
        pretrained_backbone: bool = True,
        trainable_backbone_layers: Optional[int] = 5,
        **kwargs: Any,
    ) -> nn.Module:
        model = retinanet_resnet50_fpn(
            pretrained=pretrained,
            pretrained_backbone=pretrained_backbone,
            trainable_backbone_layers=trainable_backbone_layers,
            **kwargs,
        )

        model.head = RetinaNetHead(
            in_channels=model.backbone.out_channels,
            num_anchors=model.head.classification_head.num_anchors,
            num_classes=num_classes,
            **kwargs,
        )
        return model

    model = _initialize_retinanet()
    d = data.LabelledData(Path(__file__).parent / "test_data", eval=False)
    x, y = d[0]

    _ = model([x], [y])
