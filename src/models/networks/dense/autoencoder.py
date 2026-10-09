"""全连接自编码器及高斯潜变量编码器。"""

from math import prod

import torch
from torch import nn


class DenseAutoencoder(nn.Module):
    """提供确定性全连接编码器和解码器。"""

    def __init__(self, input_shape=(1, 28, 28), hidden_dims=(512, 256), latent_dim=32):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one dimension")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        self.input_shape = tuple(input_shape)
        self.input_size = prod(self.input_shape)
        self.latent_dim = latent_dim
        layers = []
        size = self.input_size
        for hidden in hidden_dims:
            layers.extend([nn.Linear(size, hidden), nn.ReLU()])
            size = hidden
        layers.append(nn.Linear(size, latent_dim))
        self.encoder = nn.Sequential(*layers)
        layers = []
        size = latent_dim
        for hidden in reversed(hidden_dims):
            layers.extend([nn.Linear(size, hidden), nn.ReLU()])
            size = hidden
        layers.append(nn.Linear(size, self.input_size))
        self.decoder = nn.Sequential(*layers)

    def encode(self, x):
        return self.encoder(x.flatten(start_dim=1))

    def decode(self, z):
        return self.decoder(z).view(z.size(0), *self.input_shape)

    def forward(self, x):
        return self.decode(self.encode(x))


class DenseGaussianAutoencoder(nn.Module):
    """输出高斯潜变量参数并提供全连接解码器。"""

    def __init__(self, input_shape=(1, 28, 28), hidden_dims=(512, 256), latent_dim=32):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one dimension")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        self.input_shape = tuple(input_shape)
        self.input_size = prod(self.input_shape)
        self.latent_dim = latent_dim
        layers = []
        size = self.input_size
        for hidden in hidden_dims:
            layers.extend([nn.Linear(size, hidden), nn.ReLU()])
            size = hidden
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(size, latent_dim)
        self.fc_logvar = nn.Linear(size, latent_dim)
        layers = []
        size = latent_dim
        for hidden in reversed(hidden_dims):
            layers.extend([nn.Linear(size, hidden), nn.ReLU()])
            size = hidden
        layers.append(nn.Linear(size, self.input_size))
        self.decoder = nn.Sequential(*layers)

    def encode(self, x):
        features = self.encoder(x.flatten(start_dim=1))
        return self.fc_mu(features), self.fc_logvar(features)

    def decode(self, z):
        return self.decoder(z).view(z.size(0), *self.input_shape)

    def forward(self, x):
        return self.encode(x)
