"""使用多频率正弦特征表示连续时间。"""

from math import log, pi

import torch
import torch.nn.functional as F
from torch import nn


class ContinuousTimeEmbedding(nn.Module):
    """使用有界导数的多频率正弦特征表示连续时间。"""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim < 2:
            raise ValueError("embedding_dim must be at least 2")
        self.embedding_dim = embedding_dim

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        """返回形状为 ``[batch, embedding_dim]`` 的时间嵌入。"""
        half_dim = self.embedding_dim // 2
        frequencies = torch.exp(
            -log(10_000)
            * torch.arange(half_dim, device=times.device, dtype=torch.float32)
            / max(half_dim - 1, 1)
        )
        embeddings = 2 * pi * times.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=1)
        if self.embedding_dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))
        return embeddings
