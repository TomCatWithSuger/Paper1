"""只预测条件速度场的 Transformer 网络。"""

from collections.abc import Mapping
from math import log

import torch
import torch.nn.functional as F
from torch import nn

from src.models.components.blocks.modulated_transformer_block import ConditionalTransformerBlock
from src.models.components.embeddings.continuous_time import ContinuousTimeEmbedding


class Transformer(nn.Module):
    """融合 Mel、序列条件和连续时间的 Transformer 场网络。"""

    def __init__(
        self, n_mels=80, hidden_dim=256, condition_dim=128, num_layers=6, num_heads=8,
    ):
        super().__init__()
        if min(n_mels, hidden_dim, condition_dim, num_layers, num_heads) <= 0:
            raise ValueError("模型维度和层数必须为正数")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim 必须能被 num_heads 整除")
        self.n_mels = n_mels
        self.hidden_dim = hidden_dim
        self.time_fields = ("time",)
        self.noisy_projection = nn.Linear(n_mels, hidden_dim)
        self.condition_projection = nn.Linear(condition_dim, hidden_dim)
        self.time_embedding = nn.Sequential(
            ContinuousTimeEmbedding(condition_dim),
            nn.Linear(condition_dim, condition_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([
            ConditionalTransformerBlock(hidden_dim, num_heads, condition_dim)
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, n_mels)

    @staticmethod
    def _position_embedding(sequence_length, embedding_dim, device, dtype):
        positions = torch.arange(sequence_length, device=device, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            -log(10_000) * torch.arange(0, embedding_dim, 2, device=device, dtype=torch.float32)
            / embedding_dim
        )
        embeddings = torch.zeros(sequence_length, embedding_dim, device=device)
        embeddings[:, 0::2] = torch.sin(positions * frequencies)
        embeddings[:, 1::2] = torch.cos(positions * frequencies[: embedding_dim // 2])
        return embeddings.to(dtype=dtype).unsqueeze(0)

    def forward(
        self, noisy_mels: torch.Tensor, times: Mapping[str, torch.Tensor],
        condition: torch.Tensor, target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noisy_mels.ndim != 3 or condition.ndim != 3:
            raise ValueError("状态和条件必须具有形状 [batch, channels, frames]")
        if noisy_mels.size(1) != self.n_mels or condition.size(0) != noisy_mels.size(0):
            raise ValueError("Mel 通道数或条件 batch 大小不匹配")
        if set(times) != set(self.time_fields):
            raise ValueError("时间字段必须为 ('time',)")
        time = times["time"]
        if time.ndim != 1 or time.size(0) != noisy_mels.size(0):
            raise ValueError("time 必须具有形状 [batch]")
        if condition.size(2) != noisy_mels.size(2):
            condition = F.interpolate(
                condition, size=noisy_mels.size(2), mode="linear", align_corners=False
            )
        noisy_hidden = self.noisy_projection(noisy_mels.transpose(1, 2))
        condition_hidden = self.condition_projection(condition.transpose(1, 2))
        position = self._position_embedding(
            noisy_mels.size(2), self.hidden_dim, noisy_mels.device, noisy_hidden.dtype
        )
        hidden = noisy_hidden + condition_hidden + position
        time_condition = self.time_embedding(time)
        padding_mask = None if target_mask is None else ~target_mask
        for block in self.blocks:
            hidden = block(hidden, time_condition, padding_mask)
        velocity = self.output_projection(self.output_norm(hidden)).transpose(1, 2)
        if target_mask is not None:
            velocity = velocity * target_mask.unsqueeze(1).to(velocity.dtype)
        return velocity
