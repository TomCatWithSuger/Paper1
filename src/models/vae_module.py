from collections.abc import Callable
from typing import Tuple

import torch
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.utilities.types import LRSchedulerTypeUnion, OptimizerLRScheduler
from torch.optim import Optimizer
from torchmetrics import MeanMetric


class VAELitModule(LightningModule):
    """Example of a `LightningModule` for variational autoencoder training.

    A `LightningModule` implements the training, validation, test, prediction, and optimizer
    configuration steps required by Lightning.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: Callable[..., Optimizer],
        scheduler: Callable[..., LRSchedulerTypeUnion] | None,
        beta: float = 1.0,
        compile: bool = False,
    ) -> None:
        """Initialize a `VAELitModule`.

        :param net: The variational autoencoder to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param beta: The weight applied to the KL-divergence loss.
        :param compile: Whether to compile the model with `torch.compile`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False,
            ignore=["net", "optimizer", "scheduler"],
        )

        self.net = net
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
        self.beta = beta
        self.compile_model = compile

        # for averaging losses across batches
        self.train_loss = MeanMetric()
        self.train_reconstruction_loss = MeanMetric()
        self.train_kl_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_reconstruction_loss = MeanMetric()
        self.val_kl_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.test_reconstruction_loss = MeanMetric()
        self.test_kl_loss = MeanMetric()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of input images.
        :return: A tuple containing the reconstruction, mean, and log-variance tensors.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        self.val_loss.reset()
        self.val_reconstruction_loss.reset()
        self.val_kl_loss.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch containing input images and labels. Labels are not used by the VAE.
        :return: A tuple containing the total loss, reconstruction loss, KL loss, and
            reconstruction.
        """
        x, _ = batch
        reconstruction, mu, logvar = self.forward(x)
        batch_size = x.size(0)

        reconstruction_loss = F.mse_loss(reconstruction, x, reduction="sum") / batch_size
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        loss = reconstruction_loss + self.beta * kl_loss
        return loss, reconstruction_loss, kl_loss, reconstruction

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch containing input images and labels.
        :param batch_idx: The index of the current batch.
        :return: The total VAE loss.
        """
        loss, reconstruction_loss, kl_loss, _ = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_reconstruction_loss(reconstruction_loss)
        self.train_kl_loss(kl_loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train/reconstruction_loss",
            self.train_reconstruction_loss,
            on_step=False,
            on_epoch=True,
        )
        self.log("train/kl_loss", self.train_kl_loss, on_step=False, on_epoch=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch containing input images and labels.
        :param batch_idx: The index of the current batch.
        """
        loss, reconstruction_loss, kl_loss, _ = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_reconstruction_loss(reconstruction_loss)
        self.val_kl_loss(kl_loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/reconstruction_loss",
            self.val_reconstruction_loss,
            on_step=False,
            on_epoch=True,
        )
        self.log("val/kl_loss", self.val_kl_loss, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch containing input images and labels.
        :param batch_idx: The index of the current batch.
        """
        loss, reconstruction_loss, kl_loss, _ = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_reconstruction_loss(reconstruction_loss)
        self.test_kl_loss(kl_loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "test/reconstruction_loss",
            self.test_reconstruction_loss,
            on_step=False,
            on_epoch=True,
        )
        self.log("test/kl_loss", self.test_kl_loss, on_step=False, on_epoch=True)

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
        reconstruction, _, _ = self.forward(x)
        return reconstruction

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
