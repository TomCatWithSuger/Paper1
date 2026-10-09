"""分数匹配及噪声扰动方法。"""

from math import log

import torch
import torch.nn.functional as F
from torch import nn


class ScoreMatching(nn.Module):
    """实现多噪声尺度分数匹配和退火 Langevin 采样。"""

    def __init__(
        self,
        network,
        sigma_min=0.01,
        sigma_max=1.0,
        num_noise_levels=100,
        sampling_steps_per_level=10,
        sampling_step_size=1e-5,
    ):
        super().__init__()
        self.network = network
        if not 0 < sigma_min < sigma_max:
            raise ValueError("noise levels must satisfy 0 < sigma_min < sigma_max")
        if num_noise_levels <= 0 or sampling_steps_per_level <= 0 or sampling_step_size <= 0:
            raise ValueError("sampling parameters must be positive")
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sampling_steps_per_level = sampling_steps_per_level
        self.sampling_step_size = sampling_step_size
        levels = torch.exp(torch.linspace(log(sigma_max), log(sigma_min), num_noise_levels))
        self.register_buffer("noise_levels", levels)

    @staticmethod
    def _expand_noise_levels(noise_levels, data):
        return noise_levels.view(noise_levels.size(0), *((1,) * (data.ndim - 1)))

    def perturb(self, data, noise_levels, noise=None):
        """按每个样本的噪声等级扰动数据。"""
        if noise is None:
            noise = torch.randn_like(data)
        return data + self._expand_noise_levels(noise_levels, data) * noise

    def forward(self, data, noise_levels):
        return self.network(data, noise_levels)

    def compute_loss(self, data):
        """返回加权分数损失、预测、目标和扰动样本。"""
        random_levels = torch.rand(data.size(0), device=data.device, dtype=data.dtype)
        levels = (
            log(self.sigma_min) + random_levels * (log(self.sigma_max) - log(self.sigma_min))
        ).exp()
        expanded = self._expand_noise_levels(levels, data)
        noise = torch.randn_like(data)
        noisy = self.perturb(data, levels, noise)
        target = -noise / expanded
        prediction = self(noisy, levels)
        loss = F.mse_loss(expanded * prediction, expanded * target)
        return loss, prediction, target, noisy

    @torch.no_grad()
    def sample(self, num_samples, device, steps_per_level=None):
        """使用退火 Langevin 动力学生成样本。"""
        steps = steps_per_level or self.sampling_steps_per_level
        if steps <= 0:
            raise ValueError("steps_per_level must be positive")
        samples = (
            torch.randn(num_samples, *self.network.input_shape, device=device) * self.sigma_max
        )
        for level in self.noise_levels:
            levels = torch.full((num_samples,), level.item(), device=device, dtype=samples.dtype)
            step_size = self.sampling_step_size * (level / self.sigma_min).pow(2)
            for _ in range(steps):
                scores = self(samples, levels)
                samples = samples + step_size * scores
                samples = samples + torch.sqrt(2.0 * step_size) * torch.randn_like(samples)
        return samples
