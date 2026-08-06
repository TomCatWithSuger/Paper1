from collections.abc import Callable
from typing import Tuple

import torch
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.utilities.types import LRSchedulerTypeUnion, OptimizerLRScheduler
from torch.optim import Optimizer
from torchmetrics import MeanMetric


class AELitModule(LightningModule):
    """Example of a `LightningModule` for autoencoder training.

    A `LightningModule` implements the training, validation, test, prediction, and optimizer
    configuration steps required by Lightning.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: Callable[..., Optimizer],
        scheduler: Callable[..., LRSchedulerTypeUnion] | None,
        compile: bool = False,
    ) -> None:
        """Initialize an `AELitModule`.

        :param net: The autoencoder to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param compile: Whether to compile the model with `torch.compile`.
        """
        super().__init__()

        self.save_hyperparameters(
            logger=False,
            ignore=["net", "optimizer", "scheduler"],
        )

        self.net = net
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
        self.compile_model = compile

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of input images.
        :return: A tensor containing reconstructed images.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        self.val_loss.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch containing input images and labels. Labels are not used by the AE.
        :return: A tuple containing the reconstruction loss and reconstructed images.
        """
        x, _ = batch
        reconstruction = self.forward(x)
        loss = F.mse_loss(reconstruction, x, reduction="sum") / x.size(0)
        return loss, reconstruction

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch containing input images and labels.
        :param batch_idx: The index of the current batch.
        :return: The reconstruction loss.
        """
        loss, _ = self.model_step(batch)

        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch containing input images and labels.
        :param batch_idx: The index of the current batch.
        """
        loss, _ = self.model_step(batch)

        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch containing input images and labels.
        :param batch_idx: The index of the current batch.
        """
        loss, _ = self.model_step(batch)

        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def predict_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        """Reconstruct a batch of input images.

        :param batch: A batch containing input images and labels.
        :param batch_idx: The index of the current batch.
        :param dataloader_idx: The index of the current dataloader.
        :return: A tensor containing reconstructed images.
        """
        x, _ = batch
        return self.forward(x)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of each stage.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.compile_model and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Choose the optimizer and optional learning-rate scheduler.

        :return: A dictionary containing the configured optimizer and optional scheduler.
        """
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
