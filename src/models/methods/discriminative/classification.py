"""分类训练方法。"""

import torch
from torch import nn


class Classification(nn.Module):
    """使用交叉熵目标训练多分类网络。"""

    def __init__(self, network):
        super().__init__()
        self.network = network
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.network(x)

    def compute_loss(self, x, y):
        """返回交叉熵损失、预测类别和目标标签。"""
        logits = self(x)
        return self.criterion(logits, y), torch.argmax(logits, dim=1), y
