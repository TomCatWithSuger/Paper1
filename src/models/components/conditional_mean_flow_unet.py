"""B2 MeanFlow 使用的 FastVoiceGrad 风格条件一维 U-Net。"""

import torch
from torch import nn

from src.models.components.conditional_flow_matching import ContinuousTimeEmbedding
from src.models.components.conditional_flow_matching_unet import (
    ConditionalFlowMatchingUNet,
)


class ConditionalMeanFlowUNet(ConditionalFlowMatchingUNet):
    """在 B1 U-Net 主干上增加 MeanFlow 区间条件。"""

    def __init__(
        self,
        n_mels: int = 80,
        condition_dim: int = 128,
        hidden_channels: int = 512,
        time_embedding_dim: int = 128,
        kernel_size: int = 3,
    ) -> None:
        """初始化与 B1 相同的卷积主干及额外区间嵌入。"""
        super().__init__(
            n_mels=n_mels,
            condition_dim=condition_dim,
            hidden_channels=hidden_channels,
            time_embedding_dim=time_embedding_dim,
            kernel_size=kernel_size,
            integration_steps=1,
        )
        self.interval_embedding = nn.Sequential(
            ContinuousTimeEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, hidden_channels),
            nn.SiLU(),
        )

    def mean_flow(
        self,
        noisy_mels: torch.Tensor,
        start_times: torch.Tensor,
        end_times: torch.Tensor,
        condition: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """预测区间 ``[r, t]`` 上的平均速度 ``u_theta(x_t, r, t, h)``。"""
        if noisy_mels.ndim != 3 or condition.ndim != 3:
            raise ValueError("noisy_mels 和 condition 必须是三维张量")
        if noisy_mels.size(1) != self.n_mels:
            raise ValueError(f"noisy_mels 必须包含 {self.n_mels} 个通道")
        if condition.size(1) != self.condition_dim:
            raise ValueError(f"condition 必须包含 {self.condition_dim} 个通道")
        if noisy_mels.size(0) != condition.size(0):
            raise ValueError("noisy_mels 和 condition 的 batch 大小必须一致")
        batch_size = noisy_mels.size(0)
        if start_times.ndim != 1 or start_times.size(0) != batch_size:
            raise ValueError("start_times 必须具有形状 [batch]")
        if end_times.ndim != 1 or end_times.size(0) != batch_size:
            raise ValueError("end_times 必须具有形状 [batch]")
        if torch.any(start_times > end_times):
            raise ValueError("start_times 不能大于 end_times")

        time_embedding = self.time_embedding(end_times)
        interval_embedding = self.interval_embedding(end_times - start_times)
        return self._forward_with_time_embedding(
            noisy_mels=noisy_mels,
            condition=condition,
            time_embedding=time_embedding + interval_embedding,
            target_mask=target_mask,
        )

    @staticmethod
    def interpolate(
        data: torch.Tensor,
        noise: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        """构造从 ``t=0`` 干净 Mel 到 ``t=1`` 高斯噪声的线性路径。"""
        time_shape = (times.size(0),) + (1,) * (data.ndim - 1)
        return (1.0 - times.view(time_shape)) * data + times.view(time_shape) * noise

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        integration_steps: int | None = None,
    ) -> torch.Tensor:
        """使用 ``u_theta(x_1, 0, 1, h)`` 一步生成目标 Mel。"""
        if integration_steps not in (None, 1):
            raise ValueError("MeanFlow 仅支持一步生成")
        samples = torch.randn(
            condition.size(0),
            self.n_mels,
            condition.size(2),
            device=condition.device,
            dtype=condition.dtype,
        )
        end_times = torch.ones(samples.size(0), device=samples.device, dtype=samples.dtype)
        start_times = torch.zeros_like(end_times)
        average_velocity = self.mean_flow(
            noisy_mels=samples,
            start_times=start_times,
            end_times=end_times,
            condition=condition,
            target_mask=target_mask,
        )
        return self._apply_mask(samples - average_velocity, target_mask)


if __name__ == "__main__":
    _ = ConditionalMeanFlowUNet()
