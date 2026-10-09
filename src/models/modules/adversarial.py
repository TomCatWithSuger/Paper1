"""对抗生成的 Lightning 训练入口。"""

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric

from src.models.modules.generative import _map_legacy_state_dict


class GANLitModule(LightningModule):
    """管理 GAN 双优化器手动训练与分项指标记录。"""

    def __init__(self, method, generator_optimizer, discriminator_optimizer, compile=False):
        super().__init__()
        self.save_hyperparameters(
            logger=False, ignore=["method", "generator_optimizer", "discriminator_optimizer"]
        )
        self.method = method
        self.generator_optimizer_factory = generator_optimizer
        self.discriminator_optimizer_factory = discriminator_optimizer
        self.compile_model = compile
        self.automatic_optimization = False
        for stage in ("train", "val", "test"):
            setattr(self, f"{stage}_generator_loss", MeanMetric())
            setattr(self, f"{stage}_discriminator_loss", MeanMetric())

    @property
    def criterion(self):
        return self.method.criterion

    def forward(self, latent):
        return self.method(latent)

    def on_train_start(self):
        self.val_generator_loss.reset()
        self.val_discriminator_loss.reset()

    def generator_loss(self, generated_images):
        """委托 Method 计算生成器损失。"""
        return self.method.generator_loss(generated_images)

    def discriminator_loss(self, real_images, generated_images):
        """委托 Method 计算判别器损失。"""
        return self.method.discriminator_loss(real_images, generated_images)

    def model_step(self, batch):
        return self.method.compute_loss(batch[0])

    def _log_losses(self, stage, generator_loss, discriminator_loss):
        for name, value in (
            ("generator_loss", generator_loss),
            ("discriminator_loss", discriminator_loss),
        ):
            metric = getattr(self, f"{stage}_{name}")
            metric(value)
            self.log(f"{stage}/{name}", metric, on_step=False, on_epoch=True, prog_bar=True)

    def training_step(self, batch, batch_idx):
        real, _ = batch
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            raise RuntimeError("GAN training requires two optimizers")
        generator_optimizer, discriminator_optimizer = optimizers
        self.toggle_optimizer(discriminator_optimizer)
        discriminator_optimizer.zero_grad()
        generated = self.method.generate(real.size(0), real.device)
        discriminator_loss = self.discriminator_loss(real, generated.detach())
        self.manual_backward(discriminator_loss)
        discriminator_optimizer.step()
        self.untoggle_optimizer(discriminator_optimizer)
        self.toggle_optimizer(generator_optimizer)
        generator_optimizer.zero_grad()
        generated = self.method.generate(real.size(0), real.device)
        generator_loss = self.generator_loss(generated)
        self.manual_backward(generator_loss)
        generator_optimizer.step()
        self.untoggle_optimizer(generator_optimizer)
        self._log_losses("train", generator_loss, discriminator_loss)
        return (generator_loss + discriminator_loss).detach()

    def validation_step(self, batch, batch_idx):
        generator_loss, discriminator_loss, _ = self.model_step(batch)
        self._log_losses("val", generator_loss, discriminator_loss)

    def test_step(self, batch, batch_idx):
        generator_loss, discriminator_loss, _ = self.model_step(batch)
        self._log_losses("test", generator_loss, discriminator_loss)

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        pass

    def on_test_epoch_end(self):
        pass

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.method.generate(batch[0].size(0), batch[0].device)

    def setup(self, stage):
        if self.compile_model and stage == "fit":
            self.method.network = torch.compile(self.method.network)

    def configure_optimizers(self):
        return [
            self.generator_optimizer_factory(params=self.method.network.generator.parameters()),
            self.discriminator_optimizer_factory(params=self.method.network.discriminator.parameters()),
        ]

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return super().load_state_dict(
            _map_legacy_state_dict(self.method, state_dict), strict=strict, assign=assign
        )
