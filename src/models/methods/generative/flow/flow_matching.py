"""无条件流匹配方法。"""

import torch
import torch.nn.functional as F
from torch import nn


class FlowMatching(nn.Module):
    """使用线性概率路径训练无条件连续速度场。"""

    def __init__(self, network, integration_steps=100):
        super().__init__()
        self.network = network
        if integration_steps <= 0:
            raise ValueError("integration_steps must be positive")
        self.integration_steps = integration_steps

    @staticmethod
    def interpolate(data, noise, times):
        """在线性概率路径上插值噪声与数据。"""
        times = times.view((times.size(0),) + (1,) * (data.ndim - 1))
        return (1.0 - times) * noise + times * data

    def forward(self, data, times):
        return self.network(data, times)

    def compute_loss(self, data, *, time_first=False):
        """返回速度回归损失、预测、目标和路径状态。"""
        if time_first:
            times = torch.rand(data.size(0), device=data.device, dtype=data.dtype)
            noise = torch.randn_like(data)
        else:
            noise = torch.randn_like(data)
            times = torch.rand(data.size(0), device=data.device, dtype=data.dtype)
        path = self.interpolate(data, noise, times)
        target = data - noise
        prediction = self(path, times)
        return F.mse_loss(prediction, target), prediction, target, path

    @torch.no_grad()
    def sample(self, num_samples, device, integration_steps=None):
        """从高斯噪声通过 Euler 积分生成样本。"""
        steps = integration_steps or self.integration_steps
        if steps <= 0:
            raise ValueError("integration_steps must be positive")
        samples = torch.randn(num_samples, *self.network.input_shape, device=device)
        return self._integrate(samples, steps)

    def _integrate(self, samples, steps):
        step_size = 1.0 / steps
        for step in range(steps):
            times = torch.full((samples.size(0),), step / steps, device=samples.device, dtype=samples.dtype)
            samples = samples + step_size * self(samples, times)
        return samples

    @torch.no_grad()
    def sample_waveform(self, num_frames, device, integration_steps=None):
        """从高斯噪声积分生成单条指定长度的波形。"""
        steps = integration_steps or self.integration_steps
        samples = torch.randn(1, num_frames, self.network.in_channels, device=device)
        return self._integrate(samples, steps).squeeze(0)
