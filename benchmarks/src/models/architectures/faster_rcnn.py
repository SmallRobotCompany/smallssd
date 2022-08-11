from torch import nn

from typing import Optional

from ...config import NUM_OUTPUT_CLASSES, DEFAULT_LEARNING_RATE

from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
    fasterrcnn_resnet50_fpn,
)

from typing import Tuple


def initialize_fasterrcnn(
    num_classes: int = NUM_OUTPUT_CLASSES,
    pretrained: bool = False,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: Optional[int] = 5,
) -> Tuple[nn.Module, float]:
    """
    Args:
        num_classes: number of detection classes (including background)
        pretrained: if true, returns a model pre-trained on COCO train2017
        pretrained_backbone: if true, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers: number of trainable resnet layers starting from final block
    """
    model = fasterrcnn_resnet50_fpn(
        pretrained=pretrained,
        pretrained_backbone=pretrained_backbone,
        trainable_backbone_layers=trainable_backbone_layers,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model, DEFAULT_LEARNING_RATE
