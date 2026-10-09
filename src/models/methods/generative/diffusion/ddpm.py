"""去噪扩散概率模型方法。"""

import torch
import torch.nn.functional as F
from torch import nn


class DDPM(nn.Module):
    """实现离散扩散加噪、噪声预测和反向采样。"""

    def __init__(self, network, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.network = network
        if timesteps <= 0:
            raise ValueError("timesteps must be positive")
        if not 0 < beta_start < beta_end < 1:
            raise ValueError("beta values must satisfy 0 < beta_start < beta_end < 1")
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        previous = F.pad(alpha_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod))
        self.register_buffer("sqrt_reciprocal_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_variance", betas * (1.0 - previous) / (1.0 - alpha_cumprod)
        )

    @staticmethod
    def _extract(values, timesteps, x):
        return values.gather(0, timesteps).view(timesteps.size(0), *((1,) * (x.ndim - 1)))

    def q_sample(self, x_start, timesteps, noise=None):
        """按累计噪声日程采样前向扩散状态。"""
        if noise is None:
            noise = torch.randn_like(x_start)
        signal = self._extract(self.sqrt_alpha_cumprod, timesteps, x_start)
        scale = self._extract(self.sqrt_one_minus_alpha_cumprod, timesteps, x_start)
        return signal * x_start + scale * noise

    @torch.no_grad()
    def p_sample(self, x, timesteps):
        """执行给定时间步的一次随机反向扩散。"""
        betas = self._extract(self.betas, timesteps, x)
        scale = self._extract(self.sqrt_one_minus_alpha_cumprod, timesteps, x)
        reciprocal = self._extract(self.sqrt_reciprocal_alphas, timesteps, x)
        predicted_noise = self.network(x, timesteps)
        mean = reciprocal * (x - betas * predicted_noise / scale)
        variance = self._extract(self.posterior_variance, timesteps, x)
        noise = torch.randn_like(x)
        mask = (timesteps != 0).float().view(timesteps.size(0), *((1,) * (x.ndim - 1)))
        return mean + mask * torch.sqrt(variance) * noise

    @torch.no_grad()
    def sample(self, num_samples, device):
        """从高斯噪声迭代执行完整反向扩散。"""
        samples = torch.randn(num_samples, *self.network.input_shape, device=device)
        for timestep in reversed(range(self.timesteps)):
            times = torch.full((num_samples,), timestep, device=device, dtype=torch.long)
            samples = self.p_sample(samples, times)
        return samples

    def forward(self, x, timesteps):
        return self.network(x, timesteps)

    def compute_loss(self, x):
        """返回噪声预测损失、预测噪声、真实噪声和带噪样本。"""
        times = torch.randint(0, self.timesteps, (x.size(0),), device=x.device, dtype=torch.long)
        noise = torch.randn_like(x)
        noisy = self.q_sample(x, times, noise)
        prediction = self(noisy, times)
        return F.mse_loss(prediction, noise), prediction, noise, noisy
