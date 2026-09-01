"""条件 Flow Matching 基线的 Lightning 训练组织模块。

该模块连接 VCTK 批数据、离线 MeanVC 特征和速度场网络。训练采用干净语音自重构：同一 段语音同时提供内容、说话人身份和目标 Mel。预测时则组合源语音内容与独立的目标说话人
参考向量，形成零样本语音转换接口。
"""

from collections.abc import Callable, Mapping
from typing import Any, Protocol, cast

import torch
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.utilities.types import LRSchedulerTypeUnion, OptimizerLRScheduler
from torch.optim import Optimizer
from torchmetrics import MeanMetric


class ConditionalFlowMatchingNetwork(Protocol):
    """Flow Matching 速度场主干需要满足的结构接口。

    Protocol 使 Lightning 模块不依赖当前 Transformer 的具体实现。后续可以替换为 U-Net、 DiT 或 Mean Flow 网络，同时保持相同的公共操作。
    """

    def __call__(
        self,
        noisy_mels: torch.Tensor,
        times: torch.Tensor,
        condition: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    def interpolate(
        self,
        data: torch.Tensor,
        noise: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor: ...

    def sample(
        self,
        condition: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        integration_steps: int | None = None,
    ) -> torch.Tensor: ...


class VoiceConditionEncoder(Protocol):
    """构造统一语音条件 ``h`` 所需的结构接口。

    Lightning 模块只依赖融合后的条件，不关心内部的内容、说话人以及未来韵律或音色编码器 如何设计。
    """

    def __call__(
        self,
        content_features: torch.Tensor,
        content_lengths: torch.Tensor,
        speaker_features: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor: ...


class ConditionalFlowMatchingLitModule(LightningModule):
    """训练和评估标准条件 Flow Matching 语音转换基线。

    训练遵循 ``x_0 ~ N(0, I)``、``x_1 = 干净 Mel``、``x_t = (1-t)x_0 + tx_1``，目标
    速度为 ``v = x_1 - x_0``。模型最小化 ``v_theta(x_t, t, h)`` 与 ``v`` 之间的带掩码
    MSE。本基线不使用 MeanVoiceFlow 的伪源语音、零输入重构或条件扩散输入目标。
    """

    def __init__(
        self,
        net: torch.nn.Module,
        condition_encoder: torch.nn.Module,
        optimizer: Callable[..., Optimizer],
        scheduler: Callable[..., LRSchedulerTypeUnion] | None,
        compile: bool = False,
    ) -> None:
        """初始化训练组件和 epoch 级损失指标。

        :param net: 实现 ``model(x_t, t, h)`` 的速度场主干。
        :param condition_encoder: 分离并融合内容与说话人信息的模块。
        :param optimizer: 使用本模块全部参数实例化的优化器工厂。
        :param scheduler: 可选的学习率调度器工厂。
        :param compile: 是否在训练前编译条件编码器和速度场网络。
        """
        super().__init__()
        self.save_hyperparameters(
            logger=False,
            ignore=["net", "condition_encoder", "optimizer", "scheduler"],
        )

        self.net: ConditionalFlowMatchingNetwork = cast(ConditionalFlowMatchingNetwork, net)
        self.condition_encoder: VoiceConditionEncoder = cast(
            VoiceConditionEncoder, condition_encoder
        )
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
        self.compile_model = compile

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(
        self,
        noisy_mels: torch.Tensor,
        times: torch.Tensor,
        condition: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """将 ``v_theta(x_t, t, h)`` 的预测委托给选定的速度场主干。

        :param noisy_mels: 插值状态 ``x_t``。
        :param times: 每个样本对应的连续流时间。
        :param condition: 统一的逐帧条件 ``h``。
        :param target_mask: 标识有效目标帧的掩码。
        :return: 与 ``noisy_mels`` 形状相同的预测速度。
        """
        return self.net(
            noisy_mels=noisy_mels,
            times=times,
            condition=condition,
            target_mask=target_mask,
        )

    @staticmethod
    def _tensor(
        batch: Mapping[str, Any],
        primary_key: str,
        fallback_key: str | None = None,
    ) -> torch.Tensor:
        """读取张量，同时兼容显式字段名和阶段一 VCTK 字段名。

        优先使用 ``target_mel_spectrograms`` 等显式字段。回退字段用于兼容当前输出
        ``mel_spectrograms`` 的 VCTK 数据模块。
        """
        value = batch.get(primary_key)
        if value is None and fallback_key is not None:
            value = batch.get(fallback_key)
        if not isinstance(value, torch.Tensor):
            raise KeyError(f"batch tensor not found: {primary_key}")
        return value

    def _training_data(
        self, batch: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """提取自重构目标及对应的离线条件特征。"""
        target_mels = self._tensor(batch, "target_mel_spectrograms", "mel_spectrograms")
        target_mask = self._tensor(batch, "target_mel_attention_mask", "mel_attention_mask")
        content_features = self._tensor(batch, "target_content_features", "content_features")
        content_lengths = self._tensor(
            batch, "target_content_feature_lengths", "content_feature_lengths"
        )
        speaker_features = self._tensor(batch, "target_speaker_features", "speaker_features")
        return (
            target_mels,
            target_mask,
            content_features,
            content_lengths,
            speaker_features,
        )

    def _inference_data(
        self, batch: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """提取源内容特征、参考说话人向量及生成长度。"""
        source_mask = self._tensor(batch, "source_mel_attention_mask", "mel_attention_mask")
        content_features = self._tensor(batch, "source_content_features", "content_features")
        content_lengths = self._tensor(
            batch, "source_content_feature_lengths", "content_feature_lengths"
        )
        speaker_features = self._tensor(batch, "reference_speaker_features", "speaker_features")
        return (
            source_mask,
            content_features,
            content_lengths,
            speaker_features,
        )

    def encode_condition(
        self,
        content_features: torch.Tensor,
        content_lengths: torch.Tensor,
        speaker_features: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """融合离线 MeanVC 特征以构造条件 ``h``。

        :return: 形状为 ``[batch, condition_dim, frames]`` 的条件张量。
        """
        return self.condition_encoder(
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
        """计算有效 Mel 通道和帧上的 MSE，并排除填充区域。

        分母为有效帧数乘以 Mel 通道数，使损失尺度不受序列长度和填充比例影响。
        """
        valid = mask.unsqueeze(1).to(dtype=prediction.dtype)
        squared_error = F.mse_loss(prediction, target, reduction="none") * valid
        denominator = valid.sum() * prediction.size(1)
        return squared_error.sum() / denominator.clamp_min(1.0)

    def on_train_start(self) -> None:
        """在第一个训练 epoch 开始前重置验证损失指标。"""
        self.val_loss.reset()

    def model_step(
        self, batch: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """构造一次标准条件 Flow Matching 优化步骤。

        同一段干净目标语音同时输入内容编码器和说话人编码器，形成基线自重构条件。每个样本
        独立采样高斯噪声和时间，网络随后回归线性路径上的恒定速度 ``x_1-x_0``。

        :return: 损失、预测速度、目标速度和插值状态 ``x_t``。
        """
        target_mels, target_mask, content_features, content_lengths, speaker_features = (
            self._training_data(batch)
        )
        condition = self.encode_condition(
            content_features=content_features,
            content_lengths=content_lengths,
            speaker_features=speaker_features,
            target_mask=target_mask,
        )
        noise = torch.randn_like(target_mels)
        times = torch.rand(
            target_mels.size(0),
            device=target_mels.device,
            dtype=target_mels.dtype,
        )
        path_samples = self.net.interpolate(target_mels, noise, times)
        # 线性概率路径的导数恒为 dx_t/dt = x_1 - x_0。
        target_velocity = target_mels - noise
        predicted_velocity = self.forward(
            noisy_mels=path_samples,
            times=times,
            condition=condition,
            target_mask=target_mask,
        )
        loss = self._masked_mse(predicted_velocity, target_velocity, target_mask)
        return loss, predicted_velocity, target_velocity, path_samples

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        """执行一次条件 Flow Matching 训练步骤并记录损失。"""
        loss, _, _, _ = self.model_step(batch)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> None:
        """执行一次条件 Flow Matching 验证步骤并记录损失。"""
        loss, _, _, _ = self.model_step(batch)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Mapping[str, Any], batch_idx: int) -> None:
        """执行一次条件 Flow Matching 测试步骤并记录损失。"""
        loss, _, _, _ = self.model_step(batch)
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        """根据源语音内容和参考语音生成目标说话人的 Mel 频谱。

        与训练不同，预测会有意向两个编码分支输入不同语音。输出时长跟随源内容序列。
        """
        source_mask, content_features, content_lengths, speaker_features = self._inference_data(
            batch
        )
        condition = self.encode_condition(
            content_features=content_features,
            content_lengths=content_lengths,
            speaker_features=speaker_features,
            target_mask=source_mask,
        )
        return self.net.sample(
            condition=condition,
            target_mask=source_mask,
        )

    def setup(self, stage: str) -> None:
        """可选地在训练前编译条件编码器和速度场网络。"""
        if self.compile_model and stage == "fit":
            self.net = cast(ConditionalFlowMatchingNetwork, torch.compile(self.net))
            self.condition_encoder = cast(
                VoiceConditionEncoder, torch.compile(self.condition_encoder)
            )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """为条件编码器和速度场网络的全部参数配置优化过程。

        使用学习率调度器时监控 ``val/loss``，因为验证和训练采用相同的标准 Flow Matching
        目标。
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
