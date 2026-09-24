"""条件语音流实验共享的 Lightning 基础设施。"""

# ====================
# 1. Imports
# ====================

from collections.abc import Callable, Mapping
from typing import Any, Protocol, cast

import torch
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.utilities.types import LRSchedulerTypeUnion, OptimizerLRScheduler
from torch.optim import Optimizer
from torchmetrics import MeanMetric

# ====================
# 2. Definitions
# ====================


class VoiceConditionEncoder(Protocol):
    """构造逐帧语音条件所需的接口。"""

    def __call__(
        self,
        content_features: torch.Tensor,
        content_lengths: torch.Tensor,
        speaker_features: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor: ...


# ====================
# 3. Base Module
# ====================


class VoiceFlowLitModuleBase(LightningModule):
    """提供条件语音流实验共用的数据、指标和优化器流程。"""

    def __init__(
        self,
        net: torch.nn.Module,
        condition_encoder: torch.nn.Module,
        optimizer: Callable[..., Optimizer],
        scheduler: Callable[..., LRSchedulerTypeUnion] | None,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            logger=False,
            ignore=["net", "condition_encoder", "optimizer", "scheduler"],
        )

        self.net = net
        self.condition_encoder = condition_encoder
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
        self.compile_model = compile

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    @staticmethod
    def _tensor(
        batch: Mapping[str, Any],
        primary_key: str,
        fallback_key: str | None = None,
    ) -> torch.Tensor:
        """读取张量，同时兼容显式字段名和 VCTK 回退字段名。"""
        value = batch.get(primary_key)
        if value is None and fallback_key is not None:
            value = batch.get(fallback_key)
        if not isinstance(value, torch.Tensor):
            raise KeyError(f"batch tensor not found: {primary_key}")
        return value

    def _training_data(
        self, batch: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """提取目标 Mel 及对应的离线条件特征。"""
        target_mels = self._tensor(batch, "target_mel_spectrograms", "mel_spectrograms")
        target_mask = self._tensor(batch, "target_mel_attention_mask", "mel_attention_mask")
        content_features = self._tensor(batch, "target_content_features", "content_features")
        content_lengths = self._tensor(
            batch, "target_content_feature_lengths", "content_feature_lengths"
        )
        speaker_features = self._tensor(batch, "target_speaker_features", "speaker_features")
        return target_mels, target_mask, content_features, content_lengths, speaker_features

    def _inference_data(
        self, batch: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """提取源内容特征、参考说话人向量及生成掩码。"""
        source_mask = self._tensor(batch, "source_mel_attention_mask", "mel_attention_mask")
        content_features = self._tensor(batch, "source_content_features", "content_features")
        content_lengths = self._tensor(
            batch, "source_content_feature_lengths", "content_feature_lengths"
        )
        speaker_features = self._tensor(batch, "reference_speaker_features", "speaker_features")
        return source_mask, content_features, content_lengths, speaker_features

    def encode_condition(
        self,
        content_features: torch.Tensor,
        content_lengths: torch.Tensor,
        speaker_features: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """融合离线内容特征和说话人特征。"""
        encoder = cast(VoiceConditionEncoder, self.condition_encoder)
        return encoder(
            content_features=content_features,
            content_lengths=content_lengths,
            speaker_features=speaker_features,
            target_mask=target_mask,
        )

    @staticmethod
    def _masked_mse(
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """计算有效 Mel 帧上的均方误差。"""
        valid = mask.unsqueeze(1).to(dtype=prediction.dtype)
        squared_error = F.mse_loss(prediction, target, reduction="none") * valid
        denominator = valid.sum() * prediction.size(1)
        return squared_error.sum() / denominator.clamp_min(1.0)

    def model_step(
        self, batch: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """由具体实验实现单批次目标。"""
        raise NotImplementedError

    def on_train_start(self) -> None:
        """避免训练前 sanity check 污染验证指标。"""
        self.val_loss.reset()

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        """执行一次训练步骤。"""
        loss, _, _, _ = self.model_step(batch)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> None:
        """执行一次验证步骤。"""
        loss, _, _, _ = self.model_step(batch)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Mapping[str, Any], batch_idx: int) -> None:
        """执行一次测试步骤。"""
        loss, _, _, _ = self.model_step(batch)
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def setup(self, stage: str) -> None:
        """按配置编译网络和条件编码器。"""
        if self.compile_model and stage == "fit":
            self.net = torch.compile(self.net)
            self.condition_encoder = torch.compile(self.condition_encoder)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """配置通用优化器和可选的 epoch 级调度器。"""
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
