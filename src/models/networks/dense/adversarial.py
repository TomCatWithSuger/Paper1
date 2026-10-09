"""全连接生成器与判别器。"""

from math import prod

import torch
from torch import nn


class DenseGenerator(nn.Module):
    """将潜变量映射为固定形状样本的全连接生成器。"""

    def __init__(self, latent_dim, output_shape, hidden_dims):
        super().__init__()
        layers = []
        size = latent_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(size, hidden), nn.BatchNorm1d(hidden), nn.LeakyReLU(0.2)])
            size = hidden
        layers.append(nn.Linear(size, prod(output_shape)))
        self.output_shape = tuple(output_shape)
        self.model = nn.Sequential(*layers)

    def forward(self, latent):
        return self.model(latent).view(latent.size(0), *self.output_shape)


class DenseDiscriminator(nn.Module):
    """将固定形状样本映射为单个判别 logit。"""

    def __init__(self, input_shape, hidden_dims, dropout=0.2):
        super().__init__()
        layers = []
        size = prod(input_shape)
        for hidden in hidden_dims:
            layers.extend([nn.Linear(size, hidden), nn.LeakyReLU(0.2), nn.Dropout(dropout)])
            size = hidden
        layers.append(nn.Linear(size, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x.flatten(start_dim=1))


class DenseAdversarialPair(nn.Module):
    """组合共享数据契约的全连接生成器与判别器。"""

    def __init__(
        self, input_shape=(1, 28, 28), latent_dim=100,
        generator_hidden_dims=(256, 512, 1024), discriminator_hidden_dims=(512, 256),
        discriminator_dropout=0.2,
    ):
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if not generator_hidden_dims or not discriminator_hidden_dims:
            raise ValueError("hidden_dims must contain at least one dimension")
        if not 0 <= discriminator_dropout < 1:
            raise ValueError("discriminator_dropout must be in the interval [0, 1)")
        self.input_shape = tuple(input_shape)
        self.latent_dim = latent_dim
        self.generator = DenseGenerator(latent_dim, self.input_shape, generator_hidden_dims)
        self.discriminator = DenseDiscriminator(self.input_shape, discriminator_hidden_dims, discriminator_dropout)

    def forward(self, latent):
        return self.generator(latent)

    def discriminate(self, x):
        return self.discriminator(x)
