from collections.abc import Callable
from typing import Protocol, Tuple, cast

import torch
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.utilities.types import LRSchedulerTypeUnion, OptimizerLRScheduler
from torch.optim import Optimizer
from torchmetrics import MeanMetric


class DDPMNetwork(Protocol):
    """The network operations required by `DDPMLitModule`."""

    timesteps: int

    def __call__(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor: ...

    def q_sample(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


class DDPMLitModule(LightningModule):
    """Example of a `LightningModule` for denoising diffusion model training.

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
        """Initialize a `DDPMLitModule`.

        :param net: The denoising diffusion model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param compile: Whether to compile the model with `torch.compile`.
        """
        super().__init__()

        self.save_hyperparameters(
            logger=False,
            ignore=["net", "optimizer", "scheduler"],
        )

        self.net: DDPMNetwork = cast(DDPMNetwork, net)
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
        self.compile_model = compile

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor containing noisy images.
        :param timesteps: A tensor containing diffusion timesteps.
        :return: A tensor containing predicted noise.
        """
        return self.net(x, timesteps)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        self.val_loss.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch containing input images and labels. Labels are not used by the DDPM.
        :return: A tuple containing the loss, predicted noise, target noise, and noisy images.
        """
        x, _ = batch
        timesteps = torch.randint(
            0,
            self.net.timesteps,
            (x.size(0),),
            device=x.device,
            dtype=torch.long,
        )
        noise = torch.randn_like(x)
        noisy_images = self.net.q_sample(x, timesteps, noise)
        predicted_noise = self.forward(noisy_images, timesteps)
        loss = F.mse_loss(predicted_noise, noise)
        return loss, predicted_noise, noise, noisy_images

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch containing input images and labels.
        :param batch_idx: The index of the current batch.
        :return: The noise prediction loss.
        """
        loss, _, _, _ = self.model_step(batch)

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
        loss, _, _, _ = self.model_step(batch)

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
        loss, _, _, _ = self.model_step(batch)

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
        """Predict noise for a randomly noised batch of images.

        :param batch: A batch containing input images and labels.
        :param batch_idx: The index of the current batch.
        :param dataloader_idx: The index of the current dataloader.
        :return: A tensor containing predicted noise.
        """
        _, predicted_noise, _, _ = self.model_step(batch)
        return predicted_noise

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of each stage.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.compile_model and stage == "fit":
            self.net = cast(DDPMNetwork, torch.compile(self.net))

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
