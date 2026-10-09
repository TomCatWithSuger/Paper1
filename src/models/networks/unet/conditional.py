"""由显式时间字段驱动的条件语音 U-Net。"""

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from src.models.components.blocks.gated_conv import GatedConv1d, GatedConvTranspose1d
from src.models.components.embeddings.continuous_time import ContinuousTimeEmbedding


class ConditionalUNetBackbone(nn.Module):
    """条件语音流网络共用的 FastVoiceGrad 风格 U-Net 主干。"""

    def __init__(
        self,
        n_mels: int = 80,
        condition_dim: int = 128,
        hidden_channels: int = 512,
        time_embedding_dim: int = 128,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if n_mels <= 0 or condition_dim <= 0 or hidden_channels <= 0:
            raise ValueError("模型通道数必须为正数")
        if time_embedding_dim < 2:
            raise ValueError("时间嵌入维数必须有效")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size 必须为正奇数")

        self.n_mels = n_mels
        self.condition_dim = condition_dim
        self.hidden_channels = hidden_channels
        self.time_embedding = nn.Sequential(
            ContinuousTimeEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, hidden_channels),
            nn.SiLU(),
        )

        self.input_conv = GatedConv1d(n_mels + condition_dim, hidden_channels, kernel_size)
        self.encoder_block1 = GatedConv1d(hidden_channels, hidden_channels, kernel_size)
        self.downsample1 = GatedConv1d(
            hidden_channels,
            hidden_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.encoder_block2 = GatedConv1d(hidden_channels, hidden_channels, kernel_size)
        self.downsample2 = GatedConv1d(
            hidden_channels,
            hidden_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.bottleneck_block1 = GatedConv1d(hidden_channels, hidden_channels, kernel_size)
        self.bottleneck_block2 = GatedConv1d(hidden_channels, hidden_channels, kernel_size)
        self.upsample1 = GatedConvTranspose1d(hidden_channels)
        self.decoder_block1 = GatedConv1d(hidden_channels, hidden_channels, kernel_size)
        self.upsample2 = GatedConvTranspose1d(hidden_channels)
        self.decoder_block2 = GatedConv1d(hidden_channels, hidden_channels, kernel_size)
        self.output_conv = weight_norm(
            nn.Conv1d(hidden_channels, n_mels, kernel_size=kernel_size, padding=kernel_size // 2)
        )

    def _validate_inputs(
        self,
        noisy_mels: torch.Tensor,
        condition: torch.Tensor,
    ) -> None:
        """校验共享主干输入。"""
        if noisy_mels.ndim != 3 or condition.ndim != 3:
            raise ValueError("noisy_mels 和 condition 必须是三维张量")
        if noisy_mels.size(1) != self.n_mels:
            raise ValueError(f"noisy_mels 必须包含 {self.n_mels} 个通道")
        if condition.size(1) != self.condition_dim:
            raise ValueError(f"condition 必须包含 {self.condition_dim} 个通道")
        if noisy_mels.size(0) != condition.size(0):
            raise ValueError("noisy_mels 和 condition 的 batch 大小必须一致")

    @staticmethod
    def _resize_mask(mask: torch.Tensor | None, frames: int) -> torch.Tensor | None:
        """将有效帧掩码调整到当前分辨率。"""
        if mask is None or mask.size(1) == frames:
            return mask
        resized = F.interpolate(mask.unsqueeze(1).float(), size=frames, mode="nearest")
        return resized.squeeze(1).bool()

    @staticmethod
    def _apply_mask(hidden: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """将填充位置置零。"""
        if mask is None:
            return hidden
        return hidden * mask.unsqueeze(1).to(hidden.dtype)

    @staticmethod
    def _align(hidden: torch.Tensor, frames: int) -> torch.Tensor:
        """对齐上下采样后的时间长度。"""
        if hidden.size(2) == frames:
            return hidden
        return F.interpolate(hidden, size=frames, mode="linear", align_corners=False)

    @staticmethod
    def _add_time(hidden: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        """向所有时间帧注入全局时间特征。"""
        return hidden + time_embedding.unsqueeze(2)

    def _forward_with_time_embedding(
        self,
        noisy_mels: torch.Tensor,
        condition: torch.Tensor,
        time_embedding: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """使用外部构造的时间条件执行共享 U-Net 主干。"""
        self._validate_inputs(noisy_mels, condition)
        frames = noisy_mels.size(2)
        condition = self._align(condition, frames)
        if target_mask is not None:
            valid = target_mask.unsqueeze(1).to(noisy_mels.dtype)
            noisy_mels = noisy_mels * valid
            condition = condition * valid

        hidden = self.input_conv(torch.cat((noisy_mels, condition), dim=1))
        hidden = self._apply_mask(self._add_time(hidden, time_embedding), target_mask)
        skip1 = self.encoder_block1(hidden)
        skip1 = self._apply_mask(self._add_time(skip1, time_embedding), target_mask)

        hidden = self.downsample1(skip1)
        mask1 = self._resize_mask(target_mask, hidden.size(2))
        hidden = self._apply_mask(self._add_time(hidden, time_embedding), mask1)
        skip2 = self.encoder_block2(hidden)
        skip2 = self._apply_mask(self._add_time(skip2, time_embedding), mask1)

        hidden = self.downsample2(skip2)
        mask2 = self._resize_mask(target_mask, hidden.size(2))
        hidden = self._apply_mask(self._add_time(hidden, time_embedding), mask2)
        hidden = self.bottleneck_block1(hidden)
        hidden = self._apply_mask(self._add_time(hidden, time_embedding), mask2)
        hidden = self.bottleneck_block2(hidden)
        hidden = self._apply_mask(self._add_time(hidden, time_embedding), mask2)

        hidden = self._align(self.upsample1(hidden), skip2.size(2))
        hidden = self._apply_mask(self._add_time(hidden, time_embedding), mask1)
        hidden = self.decoder_block1(hidden + skip2)
        hidden = self._apply_mask(self._add_time(hidden, time_embedding), mask1)

        hidden = self._align(self.upsample2(hidden), skip1.size(2))
        hidden = self._apply_mask(self._add_time(hidden, time_embedding), target_mask)
        hidden = self.decoder_block2(hidden + skip1)
        hidden = self._apply_mask(self._add_time(hidden, time_embedding), target_mask)

        velocity = self.output_conv(hidden)
        return self._apply_mask(velocity, target_mask)


class UNet(ConditionalUNetBackbone):
    """融合一个或多个时间字段的条件语音 U-Net。"""

    def __init__(
        self,
        n_mels: int = 80,
        condition_dim: int = 128,
        hidden_channels: int = 512,
        time_embedding_dim: int = 128,
        kernel_size: int = 3,
        time_fields: Sequence[str] = ("time",),
    ) -> None:
        fields = tuple(time_fields)
        if not fields or fields[0] != "time" or len(set(fields)) != len(fields):
            raise ValueError("time_fields 必须以 time 开头且不能重复")
        if any(field not in ("time", "interval", "source_time") for field in fields):
            raise ValueError("未知的时间字段")
        super().__init__(n_mels, condition_dim, hidden_channels, time_embedding_dim, kernel_size)
        self.time_fields = fields
        # 保留旧参数注册顺序，确保初始化随机数和优化器参数顺序不变。
        for field in fields[1:]:
            self.add_module(
                "source_diffusion_embedding" if field == "source_time" else f"{field}_embedding",
                nn.Sequential(
                    ContinuousTimeEmbedding(time_embedding_dim),
                    nn.Linear(time_embedding_dim, hidden_channels),
                    nn.SiLU(),
                ),
            )

    def forward(
        self,
        noisy_mels: torch.Tensor,
        times: Mapping[str, torch.Tensor],
        condition: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if set(times) != set(self.time_fields):
            raise ValueError(f"时间字段必须为 {self.time_fields}")
        self._validate_inputs(noisy_mels, condition)
        for field in self.time_fields:
            value = times[field]
            if value.ndim != 1 or value.size(0) != noisy_mels.size(0):
                raise ValueError(f"{field} 必须具有形状 [batch]")
        embedding = self.time_embedding(times["time"])
        for field in self.time_fields[1:]:
            name = "source_diffusion" if field == "source_time" else field
            layer = getattr(self, f"{name}_embedding")
            embedding = embedding + layer(times[field])
        return self._forward_with_time_embedding(noisy_mels, condition, embedding, target_mask)
