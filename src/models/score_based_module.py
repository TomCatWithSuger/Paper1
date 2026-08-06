from collections.abc import Callable
from math import log
from typing import Protocol, Tuple, cast

import torch
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.utilities.types import LRSchedulerTypeUnion, OptimizerLRScheduler
from torch.optim import Optimizer
from torchmetrics import MeanMetric


class ScoreNetwork(Protocol):
    """The network operations required by `ScoreBasedLitModule`."""

    sigma_min: float
    sigma_max: float

    def __call__(self, x: torch.Tensor, noise_levels: torch.Tensor) -> torch.Tensor: ...

    def perturb(
        self,
        data: torch.Tensor,
        noise_levels: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


class ScoreBasedLitModule(LightningModule):
    """Example of a `LightningModule` for score-based model training.

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
        """Initialize a `ScoreBasedLitModule`.

        :param net: The noise-conditional score model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param compile: Whether to compile the model with `torch.compile`.
        """
        super().__init__()

        self.save_hyperparameters(
            logger=False,
            ignore=["net", "optimizer", "scheduler"],
        )

        self.net: ScoreNetwork = cast(ScoreNetwork, net)
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
        self.compile_model = compile

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, x: torch.Tensor, noise_levels: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor containing noisy samples.
        :param noise_levels: A tensor containing the corresponding noise levels.
        :return: A tensor containing predicted scores.
        """
        return self.net(x, noise_levels)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        self.val_loss.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch containing input images and labels. Labels are not used by the model.
        :return: A tuple containing the loss, predicted score, target score, and noisy samples.
        """
        data, _ = batch
        random_levels = torch.rand(data.size(0), device=data.device, dtype=data.dtype)
        log_noise_levels = log(self.net.sigma_min) + random_levels * (
            log(self.net.sigma_max) - log(self.net.sigma_min)
        )
        noise_levels = log_noise_levels.exp()
        expanded_noise_levels = noise_levels.view(noise_levels.size(0), *((1,) * (data.ndim - 1)))

        noise = torch.randn_like(data)
        noisy_samples = self.net.perturb(data, noise_levels, noise)
        target_score = -noise / expanded_noise_levels
        predicted_score = self.forward(noisy_samples, noise_levels)
        loss = F.mse_loss(
            expanded_noise_levels * predicted_score,
            expanded_noise_levels * target_score,
        )
        return loss, predicted_score, target_score, noisy_samples

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch containing input images and labels.
        :param batch_idx: The index of the current batch.
        :return: The denoising score matching loss.
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
        """Predict scores for randomly perturbed samples.

        :param batch: A batch containing input images and labels.
        :param batch_idx: The index of the current batch.
        :param dataloader_idx: The index of the current dataloader.
        :return: A tensor containing predicted scores.
        """
        _, predicted_score, _, _ = self.model_step(batch)
        return predicted_score

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of each stage.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.compile_model and stage == "fit":
            self.net = cast(ScoreNetwork, torch.compile(self.net))

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
