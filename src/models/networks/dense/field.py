"""带时间或噪声条件的全连接张量预测网络。"""

from math import prod

import torch
from torch import nn

from src.models.components.embeddings.dense_time import (
    ContinuousTimeEmbedding, NoiseLevelEmbedding, SinusoidalTimeEmbedding,
)


class _DenseField(nn.Module):
    def __init__(
        self, input_shape=(1, 28, 28), hidden_dims=(512, 512, 256),
        time_embedding_dim=64, embedding="continuous", predictor="velocity_predictor",
    ):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one dimension")
        embeddings = {"continuous": ContinuousTimeEmbedding, "discrete": SinusoidalTimeEmbedding}
        self.input_shape = tuple(input_shape)
        self.input_size = prod(self.input_shape)
        self.time_embedding = nn.Sequential(
            embeddings[embedding](time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim), nn.SiLU(),
        )
        layers = []
        size = self.input_size + time_embedding_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(size, hidden), nn.SiLU()])
            size = hidden
        layers.append(nn.Linear(size, self.input_size))
        self.predictor_name = predictor
        self.add_module(predictor, nn.Sequential(*layers))

    def _predict(self, x, times):
        features = torch.cat((x.flatten(start_dim=1), self.time_embedding(times)), dim=1)
        return getattr(self, self.predictor_name)(features).view(x.size(0), *self.input_shape)


class DenseTimeField(_DenseField):
    """根据连续时间条件预测与输入同形状的张量场。"""

    def forward(self, x, times):
        return self._predict(x, times)


class DenseNoiseField(_DenseField):
    """根据离散扩散时间步预测与输入同形状的噪声。"""

    def __init__(self, input_shape=(1, 28, 28), hidden_dims=(512, 512, 256), time_embedding_dim=64):
        super().__init__(input_shape, hidden_dims, time_embedding_dim, "discrete", "noise_predictor")

    def forward(self, x, timesteps):
        return self._predict(x, timesteps)


class DenseScoreField(nn.Module):
    """根据连续噪声等级预测与输入同形状的分数场。"""

    def __init__(self, input_shape=(1, 28, 28), hidden_dims=(512, 512, 256), noise_embedding_dim=64):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one dimension")
        self.input_shape = tuple(input_shape)
        self.input_size = prod(self.input_shape)
        self.noise_embedding = nn.Sequential(
            NoiseLevelEmbedding(noise_embedding_dim),
            nn.Linear(noise_embedding_dim, noise_embedding_dim), nn.SiLU(),
        )
        layers = []
        size = self.input_size + noise_embedding_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(size, hidden), nn.SiLU()])
            size = hidden
        layers.append(nn.Linear(size, self.input_size))
        self.score_predictor = nn.Sequential(*layers)

    def forward(self, x, noise_levels):
        features = torch.cat((x.flatten(start_dim=1), self.noise_embedding(noise_levels)), dim=1)
        return self.score_predictor(features).view(x.size(0), *self.input_shape)
