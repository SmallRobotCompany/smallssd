import torch
from pytorch_lightning import LightningModule
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from .iou import evaluate_iou
from ..constants import Metrics
from ..config import TEST_MAP_KWARGS

from .architectures import STR2FUNC

from typing import Dict, List


class DetectionBase(LightningModule):
    def __init__(self, model_base: str, learning_rate: float = 1e-5, **kwargs) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.model: torch.nn.Module = STR2FUNC[model_base](**kwargs)

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, _):
        images, targets = batch
        # fasterrcnn takes only images for eval() mode
        with torch.no_grad():
            self.model.eval()
            outs = self.model(images)
        iou = torch.stack([evaluate_iou(t, o) for t, o in zip(targets, outs)]).mean()
        return {Metrics.IOU: iou, Metrics.PREDICTIONS: outs, Metrics.TARGETS: targets}

    def test_epoch_end(self, outs):
        metric = MeanAveragePrecision(**TEST_MAP_KWARGS)
        for output in outs:
            metric.update(
                self._to_cpu(output[Metrics.PREDICTIONS]),
                self._to_cpu(output[Metrics.TARGETS]),
            )
        computed_metrics = metric.compute()
        print(f"mAP: {computed_metrics['map']}, \n {computed_metrics}")


class FullySupervised(DetectionBase):
    def training_step(self, batch, _):
        images, targets = batch
        # fasterrcnn takes both images and targets for training, returns
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log("training_loss", loss)
        return {"loss": loss, "log": loss_dict}

    def validation_step(self, batch, _):
        images, targets = batch
        # fasterrcnn takes only images for eval() mode
        with torch.no_grad():
            self.model.eval()
            outs = self.model(images)
        iou = torch.stack([evaluate_iou(t, o) for t, o in zip(targets, outs)]).mean()
        return {Metrics.IOU: iou, Metrics.PREDICTIONS: outs, Metrics.TARGETS: targets}

    def validation_epoch_end(self, outs):
        avg_iou = torch.stack([o[Metrics.IOU] for o in outs]).mean()
        self.log(Metrics.AVG_IOU, avg_iou)

        metric = MeanAveragePrecision()
        for output in outs:
            metric.update(
                self._to_cpu(output[Metrics.PREDICTIONS]),
                self._to_cpu(output[Metrics.TARGETS]),
            )
        computed_metrics = metric.compute()
        self.log(Metrics.MAP, computed_metrics["map"])
        return {
            Metrics.AVG_IOU: avg_iou,
            "log": {Metrics.AVG_IOU: avg_iou, Metrics.MAP: computed_metrics["map"]},
        }

    @staticmethod
    def _to_cpu(
        targets: List[Dict[str, torch.Tensor]]
    ) -> List[Dict[str, torch.Tensor]]:
        return [{key: val.detach().cpu() for key, val in d.items()} for d in targets]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=2, verbose=True, mode="max"
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,  # Changed scheduler to lr_scheduler
            "monitor": Metrics.MAP,
        }
