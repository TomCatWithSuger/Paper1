"""带掩码的波形表征重构方法。"""

from torch import nn


class MaskedReconstruction(nn.Module):
    """仅在有效位置计算波形重构目标。"""

    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, waveforms):
        return self.network(waveforms)

    def compute_loss(self, waveforms, attention_mask):
        """返回有效波形位置上的重构损失和重构结果。"""
        reconstruction = self(waveforms)
        error = (reconstruction - waveforms).pow(2)
        mask = attention_mask.to(dtype=error.dtype)
        return (error * mask).sum() / mask.sum().clamp_min(1), reconstruction
