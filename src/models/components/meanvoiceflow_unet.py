"""B3 MeanVoiceFlow 使用的条件一维 U-Net。"""

# ====================
# 1. 导入
# ====================

import torch
from torch import nn

from src.models.components.flow_matching_backbone import (
    ConditionalUNetBackbone,
    ContinuousTimeEmbedding,
)

# ====================
# 2. 核心模型
# ====================


class MeanVoiceFlowUNet(ConditionalUNetBackbone):
    """在共享 U-Net 主干上加入平均速度区间和源扩散时间条件。"""

    def __init__(
        self,
        n_mels: int = 80,
        condition_dim: int = 128,
        hidden_channels: int = 512,
        time_embedding_dim: int = 128,
        kernel_size: int = 3,
    ) -> None:
        """初始化卷积主干和源扩散时间条件。"""
        super().__init__(
            n_mels=n_mels,
            condition_dim=condition_dim,
            hidden_channels=hidden_channels,
            time_embedding_dim=time_embedding_dim,
            kernel_size=kernel_size,
        )
        self.interval_embedding = nn.Sequential(
            ContinuousTimeEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, hidden_channels),
            nn.SiLU(),
        )
        self.source_diffusion_embedding = nn.Sequential(
            ContinuousTimeEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, hidden_channels),
            nn.SiLU(),
        )

    def mean_voice_flow(
        self,
        noisy_mels: torch.Tensor,
        start_times: torch.Tensor,
        end_times: torch.Tensor,
        source_diffusion_times: torch.Tensor,
        condition: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """预测带源扩散时间条件的区间平均速度。"""
        self._validate_inputs(noisy_mels, condition)
        batch_size = noisy_mels.size(0)
        for name, times in (
            ("start_times", start_times),
            ("end_times", end_times),
            ("source_diffusion_times", source_diffusion_times),
        ):
            if times.ndim != 1 or times.size(0) != batch_size:
                raise ValueError(f"{name} 必须具有形状 [batch]")
        if torch.any(start_times > end_times):
            raise ValueError("start_times 不能大于 end_times")
        if torch.any((source_diffusion_times < 0) | (source_diffusion_times > 1)):
            raise ValueError("source_diffusion_times 必须位于 [0, 1]")

        time_embedding = self.time_embedding(end_times)
        interval_embedding = self.interval_embedding(end_times - start_times)
        source_embedding = self.source_diffusion_embedding(source_diffusion_times)
        return self._forward_with_time_embedding(
            noisy_mels=noisy_mels,
            condition=condition,
            time_embedding=time_embedding + interval_embedding + source_embedding,
            target_mask=target_mask,
        )

    @staticmethod
    def interpolate(
        data: torch.Tensor,
        prior: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        """构造从目标 Mel 到训练先验端点的线性路径。"""
        time_shape = (times.size(0),) + (1,) * (data.ndim - 1)
        return (1.0 - times.view(time_shape)) * data + times.view(time_shape) * prior

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        integration_steps: int | None = None,
        *,
        source_mels: torch.Tensor | None = None,
        source_diffusion_time: float = 0.95,
    ) -> torch.Tensor:
        """从扩散源 Mel 执行一次 MeanVoiceFlow 更新。"""
        if integration_steps not in (None, 1):
            raise ValueError("MeanVoiceFlow 仅支持一步生成")
        if source_mels is None:
            raise ValueError("MeanVoiceFlow 推理需要 source_mels")
        if not 0.0 <= source_diffusion_time <= 1.0:
            raise ValueError("source_diffusion_time 必须位于 [0, 1]")
        self._validate_inputs(source_mels, condition)

        noise = torch.randn_like(source_mels)
        batch_size = source_mels.size(0)
        diffusion_times = torch.full(
            (batch_size,),
            source_diffusion_time,
            device=source_mels.device,
            dtype=source_mels.dtype,
        )
        time_shape = (batch_size,) + (1,) * (source_mels.ndim - 1)
        diffused_source = (
            1.0 - diffusion_times.view(time_shape)
        ) * source_mels + diffusion_times.view(time_shape) * noise
        end_times = torch.ones_like(diffusion_times)
        start_times = torch.zeros_like(diffusion_times)
        average_velocity = self.mean_voice_flow(
            noisy_mels=diffused_source,
            start_times=start_times,
            end_times=end_times,
            source_diffusion_times=diffusion_times,
            condition=condition,
            target_mask=target_mask,
        )
        return self._apply_mask(diffused_source - average_velocity, target_mask)


# ====================
# 3. 入口
# ====================

if __name__ == "__main__":
    _ = MeanVoiceFlowUNet()
