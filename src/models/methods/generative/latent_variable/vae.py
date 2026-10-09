"""变分自动编码器潜变量生成方法。"""

import torch
import torch.nn.functional as F
from torch import nn


class VariationalAutoencoder(nn.Module):
    """使用高斯潜变量、重参数化和 KL 正则训练生成模型。"""

    def __init__(self, network, beta=1.0):
        super().__init__()
        self.network = network
        self.beta = beta

    @staticmethod
    def reparameterize(mu, logvar):
        """使用重参数化技巧采样高斯潜变量。"""
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.network(x)
        return self.network.decode(self.reparameterize(mu, logvar)), mu, logvar

    def compute_loss(self, x):
        """返回总损失、重构损失、KL 损失和重构结果。"""
        reconstruction, mu, logvar = self(x)
        reconstruction_loss = F.mse_loss(reconstruction, x, reduction="sum") / x.size(0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return reconstruction_loss + self.beta * kl_loss, reconstruction_loss, kl_loss, reconstruction

    def sample(self, num_samples, device):
        """从标准高斯先验采样并解码数据。"""
        z = torch.randn(num_samples, self.network.latent_dim, device=device)
        return self.network.decode(z)
