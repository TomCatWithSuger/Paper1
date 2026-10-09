"""多分类任务的 Lightning 训练入口。"""

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src.models.modules.generative import _map_legacy_state_dict


class ClassificationLitModule(LightningModule):
    """管理可配置类别数的多分类训练与指标。"""

    def __init__(self, method, optimizer, scheduler=None, compile=False, num_classes=2):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes 必须至少为 2")
        self.save_hyperparameters(logger=False, ignore=["method", "optimizer", "scheduler"])
        self.method = method
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
        self.compile_model = compile
        for stage in ("train", "val", "test"):
            setattr(self, f"{stage}_acc", Accuracy(task="multiclass", num_classes=num_classes))
            setattr(self, f"{stage}_loss", MeanMetric())
        self.val_acc_best = MaxMetric()

    @property
    def criterion(self):
        return self.method.criterion

    def forward(self, x):
        return self.method(x)

    def on_train_start(self):
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(self, batch):
        return self.method.compute_loss(*batch)

    def _log_step(self, batch, stage):
        loss, preds, targets = self.model_step(batch)
        loss_metric = getattr(self, f"{stage}_loss")
        acc_metric = getattr(self, f"{stage}_acc")
        loss_metric(loss)
        acc_metric(preds, targets)
        self.log(f"{stage}/loss", loss_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/acc", acc_metric, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._log_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._log_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._log_step(batch, "test")

    def on_train_epoch_end(self):
        pass

    def on_test_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        self.val_acc_best(self.val_acc.compute())
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def setup(self, stage):
        if self.compile_model and stage == "fit":
            self.method.network = torch.compile(self.method.network)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return super().load_state_dict(
            _map_legacy_state_dict(self.method, state_dict), strict=strict, assign=assign
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = self.optimizer_factory(params=self.parameters())
        if self.scheduler_factory is not None:
            scheduler = self.scheduler_factory(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
