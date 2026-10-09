"""扩散、流匹配和 DiT 的 Lightning 训练入口。"""

from collections import OrderedDict
from typing import Any, cast

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torchmetrics import MeanMetric


def _map_legacy_state_dict(method, state_dict):
    network_keys = set(method.network.state_dict())
    method_keys = set(method.state_dict())
    mapped: Any = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("net."):
            suffix = key[len("net.") :]
            destination = (
                "method.network." + suffix
                if suffix in network_keys or suffix not in method_keys
                else "method." + suffix
            )
        else:
            destination = key
        if destination in mapped:
            raise ValueError(f"检查点键冲突: {destination}")
        mapped[destination] = value
    if hasattr(state_dict, "_metadata"):
        metadata = OrderedDict()
        for key, value in getattr(state_dict, "_metadata").items():
            destinations = [key]
            if key == "net":
                destinations = ["method", "method.network"]
            elif key.startswith("net."):
                destinations = ["method.network" + key[len("net") :]]
            elif key == "criterion":
                destinations = ["method.criterion"]
            for destination in destinations:
                if destination in metadata:
                    raise ValueError(f"检查点元数据键冲突: {destination}")
                metadata[destination] = value
        mapped._metadata = metadata
    return mapped


class GenerativeModule(LightningModule):
    """提供单优化器和标量主损失的通用训练生命周期。"""

    def __init__(self, method, optimizer, scheduler, compile=False):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["method", "optimizer", "scheduler"])
        self.method = method
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
        self.compile_model = compile
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, *args, **kwargs):
        return self.method(*args, **kwargs)

    def model_step(self, batch):
        return self.method.compute_loss(batch[0])

    def on_train_start(self):
        self.val_loss.reset()

    def _log_step(self, batch, stage):
        loss = self.model_step(batch)[0]
        metric = getattr(self, f"{stage}_loss")
        metric(loss)
        self.log(f"{stage}/loss", metric, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._log_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._log_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._log_step(batch, "test")

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        pass

    def on_test_epoch_end(self):
        pass

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.model_step(batch)[1]

    def setup(self, stage):
        if self.compile_model and stage == "fit":
            self.method.network = cast(torch.nn.Module, torch.compile(self.method.network))

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

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return super().load_state_dict(
            _map_legacy_state_dict(self.method, state_dict), strict=strict, assign=assign
        )


class DDPMLitModule(GenerativeModule):
    """为 DDPM 配置提供通用生成训练生命周期。"""

    pass


class FlowMatchingLitModule(GenerativeModule):
    """为 Flow Matching 配置提供通用生成训练生命周期。"""

    pass


class ScoreBasedLitModule(GenerativeModule):
    """为分数生成配置提供通用生成训练生命周期。"""

    pass


class DiTLitModule(GenerativeModule):
    """为波形 DiT 适配字典形式的训练批次。"""

    def model_step(self, batch):
        return self.method.compute_loss(batch["waveforms"].unsqueeze(-1), time_first=True)
