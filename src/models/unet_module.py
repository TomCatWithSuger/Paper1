from collections.abc import Callable, Mapping
from typing import Any, Protocol, Tuple, cast

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import LRSchedulerTypeUnion, OptimizerLRScheduler
from torch.optim import Optimizer
from torchmetrics import MeanMetric


class UNetNetwork(Protocol):
    """The network operations required by `UNetLitModule`."""

    def __call__(self, waveforms: torch.Tensor) -> torch.Tensor: ...


class UNetLitModule(LightningModule):
    """A Lightning module for masked waveform reconstruction with a one-dimensional U-Net."""

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: Callable[..., Optimizer],
        scheduler: Callable[..., LRSchedulerTypeUnion] | None,
        compile: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(
            logger=False,
            ignore=["net", "optimizer", "scheduler"],
        )

        self.net: UNetNetwork = cast(UNetNetwork, net)
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
        self.compile_model = compile

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        return self.net(waveforms)

    def on_train_start(self) -> None:
        self.val_loss.reset()

    def model_step(
        self, batch: Mapping[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        waveforms = cast(torch.Tensor, batch["waveforms"])
        attention_mask = cast(torch.Tensor, batch["attention_mask"])
        reconstruction = self.forward(waveforms)

        squared_error = (reconstruction - waveforms).pow(2)
        valid_mask = attention_mask.to(dtype=squared_error.dtype)
        loss = (squared_error * valid_mask).sum() / valid_mask.sum().clamp_min(1)
        return loss, reconstruction

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        loss, _ = self.model_step(batch)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> None:
        loss, _ = self.model_step(batch)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Mapping[str, Any], batch_idx: int) -> None:
        loss, _ = self.model_step(batch)
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        waveforms = cast(torch.Tensor, batch["waveforms"])
        return self.forward(waveforms)

    def setup(self, stage: str) -> None:
        if self.compile_model and stage == "fit":
            self.net = cast(UNetNetwork, torch.compile(self.net))

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
