from collections.abc import Callable
from typing import Any, Mapping, Protocol, cast

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import LRSchedulerTypeUnion, OptimizerLRScheduler
from torch.optim import Optimizer
from torchmetrics import MeanMetric


class DiTNetwork(Protocol):
    """The network operations required by `DiTLitModule`."""

    def __call__(self, waveforms: torch.Tensor, times: torch.Tensor) -> torch.Tensor: ...

    def interpolate(
        self, data: torch.Tensor, noise: torch.Tensor, times: torch.Tensor
    ) -> torch.Tensor: ...


class DiTLitModule(LightningModule):
    """A Lightning module for diffusion transformer training."""

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

        self.net: DiTNetwork = cast(DiTNetwork, net)
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
        self.compile_model = compile

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, waveforms: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        return self.net(waveforms, times)

    def on_train_start(self) -> None:
        self.val_loss.reset()

    def model_step(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        waveforms = batch["waveforms"].unsqueeze(-1)
        times = torch.rand(waveforms.size(0), device=waveforms.device, dtype=waveforms.dtype)
        noise = torch.randn_like(waveforms)
        path_samples = self.net.interpolate(waveforms, noise, times)
        target_velocity = waveforms - noise
        predicted_velocity = self.forward(path_samples, times)
        loss = torch.nn.functional.mse_loss(predicted_velocity, target_velocity)
        return loss, predicted_velocity, target_velocity, path_samples

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        loss, _, _, _ = self.model_step(batch)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> None:
        loss, _, _, _ = self.model_step(batch)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Mapping[str, Any], batch_idx: int) -> None:
        loss, _, _, _ = self.model_step(batch)
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        _, predicted_velocity, _, _ = self.model_step(batch)
        return predicted_velocity

    def setup(self, stage: str) -> None:
        if self.compile_model and stage == "fit":
            self.net = cast(DiTNetwork, torch.compile(self.net))

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
