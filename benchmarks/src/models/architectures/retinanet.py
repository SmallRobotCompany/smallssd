from torch import nn

from typing import Any

from torchvision.models.detection.retinanet import RetinaNetHead, retinanet_resnet50_fpn

from typing import Optional, Tuple

from ...config import NUM_OUTPUT_CLASSES, DEFAULT_LEARNING_RATE


def initialize_retinanet(
    num_classes: int = NUM_OUTPUT_CLASSES,
    pretrained: bool = False,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: Optional[int] = 5,
    **kwargs: Any,
) -> Tuple[nn.Module, float]:
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
    return model, DEFAULT_LEARNING_RATE
