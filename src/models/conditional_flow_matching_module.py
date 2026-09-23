"""条件 Flow Matching 基线的 Lightning 训练组织模块。

该模块连接 VCTK 批数据、离线 MeanVC 特征和速度场网络。训练采用干净语音自重构：同一 段语音同时提供内容、说话人身份和目标 Mel。预测时则组合源语音内容与独立的目标说话人
参考向量，形成零样本语音转换接口。
"""

from collections.abc import Callable, Mapping
from typing import Any, Protocol, cast

import torch
from lightning.pytorch.utilities.types import LRSchedulerTypeUnion
from torch.optim import Optimizer

from src.models.voice_flow_module_base import VoiceFlowLitModuleBase


class ConditionalFlowMatchingNetwork(Protocol):
    """Flow Matching 速度场主干需要满足的结构接口。

    Protocol 使 Lightning 模块可以独立选择 U-Net 或 Transformer 网络。
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


class ConditionalFlowMatchingLitModule(VoiceFlowLitModuleBase):
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
        super().__init__(
            net=net,
            condition_encoder=condition_encoder,
            optimizer=optimizer,
            scheduler=scheduler,
            compile=compile,
        )

    @property
    def flow_matching_net(self) -> ConditionalFlowMatchingNetwork:
        """返回标准 Flow Matching 网络接口。"""
        return cast(ConditionalFlowMatchingNetwork, cast(object, self.net))

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
        return self.flow_matching_net(
            noisy_mels=noisy_mels,
            times=times,
            condition=condition,
            target_mask=target_mask,
        )

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
        path_samples = self.flow_matching_net.interpolate(target_mels, noise, times)
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
        return self.flow_matching_net.sample(
            condition=condition,
            target_mask=source_mask,
        )
