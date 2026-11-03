"""模块说明：条件 Flow Matching 的线性路径、训练目标和 Euler 生成。"""

import torch
from torch import nn

from src.losses.masked_regression import masked_mse
from src.models.algorithms.flow import euler_integrate, linear_path
from src.models.methods.base import VoiceMethod


class ConditionalFlowMatching(VoiceMethod):
    """使用条件线性概率路径训练语音速度场。"""

    def __init__(self, network: nn.Module, condition_encoder: nn.Module, integration_steps=50):
        super().__init__(network, condition_encoder)
        if integration_steps <= 0:
            raise ValueError("integration_steps 必须为正数")
        self.integration_steps = integration_steps

    def forward(self, noisy_mels, times, condition, target_mask=None):
        return self.network(noisy_mels, {"time": times}, condition, target_mask)

    @staticmethod
    def interpolate(data, noise, times):
        """在线性条件概率路径上插值噪声与目标 Mel。"""
        return linear_path(data, noise, times)

    def compute_loss(
        self,
        target_mels: torch.Tensor,
        target_mask: torch.Tensor,
        content_features: torch.Tensor,
        content_lengths: torch.Tensor,
        speaker_features: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回掩码速度损失、预测、目标和路径状态。"""
        condition = self.encode_condition(
            content_features, content_lengths, speaker_features, target_mask
        )
        noise = torch.randn(
            target_mels.shape,
            device=target_mels.device,
            dtype=target_mels.dtype,
            generator=generator,
        )
        times = torch.rand(
            target_mels.size(0),
            device=target_mels.device,
            dtype=target_mels.dtype,
            generator=generator,
        )
        path_samples = self.interpolate(target_mels, noise, times)
        target_velocity = target_mels - noise
        predicted_velocity = self(path_samples, times, condition, target_mask)
        if predicted_velocity.shape != target_velocity.shape:
            raise ValueError("速度场输出必须与目标形状一致")
        loss = masked_mse(predicted_velocity, target_velocity, target_mask)
        return loss, predicted_velocity, target_velocity, path_samples

    @torch.no_grad()
    def sample(self, condition, target_mask=None, integration_steps=None):
        """在给定语音条件下通过 Euler 积分生成 Mel。"""
        steps = self.integration_steps if integration_steps is None else integration_steps
        if steps <= 0:
            raise ValueError("integration_steps 必须为正数")
        samples = torch.randn(
            condition.size(0),
            self.network.n_mels,
            condition.size(2),
            device=condition.device,
            dtype=condition.dtype,
        )
        samples = euler_integrate(
            lambda states, times: self(states, times, condition, target_mask),
            samples,
            steps,
        )
        return self.apply_mask(samples, target_mask)
