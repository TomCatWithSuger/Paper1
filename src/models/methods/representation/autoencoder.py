"""普通自动编码器表征学习方法。"""

import torch.nn.functional as F
from torch import nn


class Autoencoder(nn.Module):
    """使用确定性编码与重构目标学习数据表示。"""

    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x):
        return self.network(x)

    def compute_loss(self, x):
        """返回批平均重构损失和重构结果。"""
        reconstruction = self(x)
        loss = F.mse_loss(reconstruction, x, reduction="sum") / x.size(0)
        return loss, reconstruction
