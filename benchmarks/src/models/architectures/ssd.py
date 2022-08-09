from torch import nn

from typing import Any

from torchvision.models.detection.ssd import ssd300_vgg16

from typing import Optional, Tuple

from ...config import NUM_OUTPUT_CLASSES, DEFAULT_LEARNING_RATE


def initialize_ssd(
    num_classes: int = NUM_OUTPUT_CLASSES,
    pretrained: bool = False,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: Optional[int] = 5,
    **kwargs: Any,
) -> Tuple[nn.Module, float]:
    return (
        ssd300_vgg16(
            pretrained=pretrained,
            trainable_backbone_layers=trainable_backbone_layers,
            num_classes=num_classes,
            pretrained_backbone=pretrained_backbone,
            **kwargs,
        ),
        DEFAULT_LEARNING_RATE,
    )
