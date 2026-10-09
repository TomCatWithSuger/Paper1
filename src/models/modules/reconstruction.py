"""自动编码器和波形重构的 Lightning 训练入口。"""

from torchmetrics import MeanMetric

from src.models.modules.generative import GenerativeModule


class AELitModule(GenerativeModule):
    """为确定性自动编码器提供重构预测入口。"""

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch[0])


class VAELitModule(GenerativeModule):
    """记录 VAE 总损失、重构损失和 KL 损失。"""

    def __init__(self, method, optimizer, scheduler, compile=False):
        super().__init__(method, optimizer, scheduler, compile)
        for stage in ("train", "val", "test"):
            setattr(self, f"{stage}_reconstruction_loss", MeanMetric())
            setattr(self, f"{stage}_kl_loss", MeanMetric())

    def on_train_start(self):
        super().on_train_start()
        self.val_reconstruction_loss.reset()
        self.val_kl_loss.reset()

    def _log_step(self, batch, stage):
        loss, reconstruction_loss, kl_loss, _ = self.model_step(batch)
        for name, value in (
            ("loss", loss),
            ("reconstruction_loss", reconstruction_loss),
            ("kl_loss", kl_loss),
        ):
            metric = getattr(self, f"{stage}_{name}")
            metric(value)
            self.log(
                f"{stage}/{name}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=name == "loss",
            )
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch[0])[0]


class UNetLitModule(GenerativeModule):
    """为掩码波形重构适配字典形式的批次。"""

    def model_step(self, batch):
        return self.method.compute_loss(batch["waveforms"], batch["attention_mask"])

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch["waveforms"])
