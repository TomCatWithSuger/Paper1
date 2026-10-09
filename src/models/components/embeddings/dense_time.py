"""全连接网络使用的离散时间、连续时间及噪声嵌入。"""

from math import log, pi

import torch
import torch.nn.functional as F
from torch import nn


class _TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        if embedding_dim < 2:
            raise ValueError("embedding_dim must be at least 2")
        self.embedding_dim = embedding_dim


class SinusoidalTimeEmbedding(_TimeEmbedding):
    """将离散时间步编码为正弦特征。"""

    def forward(self, timesteps):
        half_dim = self.embedding_dim // 2
        frequencies = torch.exp(
            -log(10_000)
            * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
            / max(half_dim - 1, 1)
        )
        embeddings = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=1)
        return F.pad(embeddings, (0, 1)) if self.embedding_dim % 2 else embeddings


class ContinuousTimeEmbedding(_TimeEmbedding):
    """将连续时间编码为多频率正弦特征。"""

    def forward(self, times):
        half_dim = self.embedding_dim // 2
        frequencies = torch.exp(
            log(10_000)
            * torch.arange(half_dim, device=times.device, dtype=torch.float32)
            / max(half_dim - 1, 1)
        )
        embeddings = 2 * pi * times.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=1)
        return F.pad(embeddings, (0, 1)) if self.embedding_dim % 2 else embeddings


class NoiseLevelEmbedding(_TimeEmbedding):
    """在对数尺度上编码正噪声等级。"""

    def forward(self, noise_levels):
        half_dim = self.embedding_dim // 2
        frequencies = torch.exp(
            log(10_000)
            * torch.arange(half_dim, device=noise_levels.device, dtype=torch.float32)
            / max(half_dim - 1, 1)
        )
        embeddings = 2 * pi * noise_levels.log().unsqueeze(1) * frequencies.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=1)
        return F.pad(embeddings, (0, 1)) if self.embedding_dim % 2 else embeddings
