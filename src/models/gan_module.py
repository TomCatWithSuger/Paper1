from collections.abc import Callable
from typing import Protocol, Tuple, cast

import torch
from lightning import LightningModule
from torch.optim import Optimizer
from torchmetrics import MeanMetric


class GANNetwork(Protocol):
    """The network operations required by `GANLitModule`."""

    generator: torch.nn.Module
    discriminator: torch.nn.Module

    def __call__(self, latent: torch.Tensor) -> torch.Tensor: ...

    def generate(
        self, num_samples: int, device: torch.device | str
    ) -> torch.Tensor: ...

    def discriminate(self, images: torch.Tensor) -> torch.Tensor: ...


class GANLitModule(LightningModule):
    """Example of a `LightningModule` for generative adversarial network training.

    A GAN uses manual optimization because its generator and discriminator have separate
    optimization objectives.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        generator_optimizer: Callable[..., Optimizer],
        discriminator_optimizer: Callable[..., Optimizer],
        compile: bool = False,
    ) -> None:
        """Initialize a `GANLitModule`.

        :param net: The generative adversarial network to train.
        :param generator_optimizer: The optimizer to use for the generator.
        :param discriminator_optimizer: The optimizer to use for the discriminator.
        :param compile: Whether to compile the model with `torch.compile`.
        """
        super().__init__()

        self.save_hyperparameters(
            logger=False,
            ignore=["net", "generator_optimizer", "discriminator_optimizer"],
        )

        self.net: GANNetwork = cast(GANNetwork, net)
        self.generator_optimizer_factory = generator_optimizer
        self.discriminator_optimizer_factory = discriminator_optimizer
        self.compile_model = compile
        self.automatic_optimization = False

        self.criterion = torch.nn.BCEWithLogitsLoss()

        self.train_generator_loss = MeanMetric()
        self.train_discriminator_loss = MeanMetric()
        self.val_generator_loss = MeanMetric()
        self.val_discriminator_loss = MeanMetric()
        self.test_generator_loss = MeanMetric()
        self.test_discriminator_loss = MeanMetric()

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the generator.

        :param latent: A tensor containing latent vectors.
        :return: A tensor containing generated samples.
        """
        return self.net(latent)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        self.val_generator_loss.reset()
        self.val_discriminator_loss.reset()

    def generator_loss(self, generated_images: torch.Tensor) -> torch.Tensor:
        """Calculate the generator adversarial loss.

        :param generated_images: A tensor containing generated samples.
        :return: The generator loss.
        """
        generated_logits = self.net.discriminate(generated_images)
        return self.criterion(generated_logits, torch.ones_like(generated_logits))

    def discriminator_loss(
        self, real_images: torch.Tensor, generated_images: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the discriminator adversarial loss.

        :param real_images: A tensor containing real samples.
        :param generated_images: A tensor containing generated samples.
        :return: The discriminator loss.
        """
        real_logits = self.net.discriminate(real_images)
        generated_logits = self.net.discriminate(generated_images)
        real_loss = self.criterion(real_logits, torch.ones_like(real_logits))
        generated_loss = self.criterion(generated_logits, torch.zeros_like(generated_logits))
        return 0.5 * (real_loss + generated_loss)

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch containing real images and labels. Labels are not used by the GAN.
        :return: A tuple containing generator loss, discriminator loss, and generated images.
        """
        real_images, _ = batch
        generated_images = self.net.generate(real_images.size(0), real_images.device)
        generator_loss = self.generator_loss(generated_images)
        discriminator_loss = self.discriminator_loss(real_images, generated_images.detach())
        return generator_loss, discriminator_loss, generated_images

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single manual-optimization GAN training step.

        :param batch: A batch containing real images and labels.
        :param batch_idx: The index of the current batch.
        :return: The sum of generator and discriminator losses.
        """
        real_images, _ = batch
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            raise RuntimeError("GAN training requires two optimizers")
        generator_optimizer, discriminator_optimizer = optimizers

        self.toggle_optimizer(discriminator_optimizer)
        discriminator_optimizer.zero_grad()
        generated_images = self.net.generate(real_images.size(0), real_images.device)
        discriminator_loss = self.discriminator_loss(real_images, generated_images.detach())
        self.manual_backward(discriminator_loss)
        discriminator_optimizer.step()
        self.untoggle_optimizer(discriminator_optimizer)

        self.toggle_optimizer(generator_optimizer)
        generator_optimizer.zero_grad()
        generated_images = self.net.generate(real_images.size(0), real_images.device)
        generator_loss = self.generator_loss(generated_images)
        self.manual_backward(generator_loss)
        generator_optimizer.step()
        self.untoggle_optimizer(generator_optimizer)

        self.train_generator_loss(generator_loss)
        self.train_discriminator_loss(discriminator_loss)
        self.log(
            "train/generator_loss",
            self.train_generator_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/discriminator_loss",
            self.train_discriminator_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return (generator_loss + discriminator_loss).detach()

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch containing real images and labels.
        :param batch_idx: The index of the current batch.
        """
        generator_loss, discriminator_loss, _ = self.model_step(batch)

        self.val_generator_loss(generator_loss)
        self.val_discriminator_loss(discriminator_loss)
        self.log(
            "val/generator_loss",
            self.val_generator_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/discriminator_loss",
            self.val_discriminator_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch containing real images and labels.
        :param batch_idx: The index of the current batch.
        """
        generator_loss, discriminator_loss, _ = self.model_step(batch)

        self.test_generator_loss(generator_loss)
        self.test_discriminator_loss(discriminator_loss)
        self.log(
            "test/generator_loss",
            self.test_generator_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/discriminator_loss",
            self.test_discriminator_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def predict_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        """Generate the same number of samples as the input batch size.

        :param batch: A batch containing input images and labels.
        :param batch_idx: The index of the current batch.
        :param dataloader_idx: The index of the current dataloader.
        :return: A tensor containing generated samples.
        """
        images, _ = batch
        return self.net.generate(images.size(0), images.device)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of each stage.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.compile_model and stage == "fit":
            self.net = cast(GANNetwork, torch.compile(self.net))

    def configure_optimizers(self) -> list[Optimizer]:
        """Configure separate optimizers for the generator and discriminator.

        :return: A list containing the generator and discriminator optimizers.
        """
        generator_optimizer = self.generator_optimizer_factory(
            params=self.net.generator.parameters()
        )
        discriminator_optimizer = self.discriminator_optimizer_factory(
            params=self.net.discriminator.parameters()
        )
        return [generator_optimizer, discriminator_optimizer]
