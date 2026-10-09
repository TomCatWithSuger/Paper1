"""条件 MeanFlow 的反向线性路径与停止梯度 JVP 目标。"""

from collections.abc import Callable
from typing import cast

import torch
from torch import nn

from src.losses.masked_regression import masked_mse
from src.models.methods.base import VoiceMethod


class ConditionalMeanFlow(VoiceMethod):
    """使用停止梯度 JVP 目标学习条件平均速度场。"""

    def __init__(self, network: nn.Module, condition_encoder: nn.Module, equal_time_probability=0.75):
        super().__init__(network, condition_encoder)
        if not 0.0 <= equal_time_probability <= 1.0:
            raise ValueError("equal_time_probability 必须位于 [0, 1]")
        self.equal_time_probability = equal_time_probability

    def forward(self, noisy_mels, start_times, end_times, condition, target_mask=None):
        return self.network(
            noisy_mels, {"time": end_times, "interval": end_times - start_times}, condition, target_mask
        )

    @staticmethod
    def interpolate(data, noise, times):
        """沿目标到噪声的反向线性路径插值。"""
        shape = (times.size(0),) + (1,) * (data.ndim - 1)
        return (1.0 - times.view(shape)) * data + times.view(shape) * noise

    def _sample_time_intervals(self, batch_size, device, dtype, generator=None):
        first = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
        second = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
        start_times = torch.minimum(first, second)
        end_times = torch.maximum(first, second)
        equal_time_mask = (
            torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
            < self.equal_time_probability
        )
        return torch.where(equal_time_mask, end_times, start_times), end_times

    def compute_loss(
        self,
        target_mels: torch.Tensor,
        target_mask: torch.Tensor,
        content_features: torch.Tensor,
        content_lengths: torch.Tensor,
        speaker_features: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回平均速度损失、预测、JVP 目标和路径状态。"""
        condition = self.encode_condition(
            content_features, content_lengths, speaker_features, target_mask
        )
        noise = torch.randn(
            target_mels.shape, device=target_mels.device, dtype=target_mels.dtype, generator=generator
        )
        start_times, end_times = self._sample_time_intervals(
            target_mels.size(0), target_mels.device, target_mels.dtype, generator=generator
        )
        path_samples = self.interpolate(target_mels, noise, end_times)
        instantaneous_velocity = noise - target_mels

        def average_velocity(states, times, interval_starts):
            return self(states, interval_starts, times, condition, target_mask)

        predicted_velocity = average_velocity(path_samples, end_times, start_times)
        with torch.no_grad(), torch.autocast(device_type=target_mels.device.type, enabled=False):
            jvp_fn = cast(Callable[..., tuple[torch.Tensor, torch.Tensor]], getattr(torch.func, "jvp"))
            _, total_derivative = jvp_fn(
                average_velocity,
                (path_samples, end_times, start_times),
                (instantaneous_velocity, torch.ones_like(end_times), torch.zeros_like(start_times)),
                has_aux=False,
            )
        shape = (end_times.size(0),) + (1,) * (target_mels.ndim - 1)
        interval = (end_times - start_times).view(shape)
        target_velocity = (instantaneous_velocity - interval * total_derivative).detach()
        loss = masked_mse(predicted_velocity, target_velocity, target_mask)
        return loss, predicted_velocity, target_velocity, path_samples

    @torch.no_grad()
    def sample(self, condition, target_mask=None, integration_steps=None):
        """使用学习到的平均速度场执行一步生成。"""
        if integration_steps not in (None, 1):
            raise ValueError("MeanFlow 仅支持一步生成")
        samples = torch.randn(
            condition.size(0), self.network.n_mels, condition.size(2),
            device=condition.device, dtype=condition.dtype,
        )
        end_times = torch.ones(samples.size(0), device=samples.device, dtype=samples.dtype)
        start_times = torch.zeros_like(end_times)
        velocity = self(samples, start_times, end_times, condition, target_mask)
        return self.apply_mask(samples - velocity, target_mask)
