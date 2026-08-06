from math import log, prod
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """A sinusoidal embedding for diffusion timesteps."""

    def __init__(self, embedding_dim: int) -> None:
        """Initialize a `SinusoidalTimeEmbedding` module.

        :param embedding_dim: The number of dimensions in each timestep embedding.
        """
        super().__init__()

        if embedding_dim < 2:
            raise ValueError("embedding_dim must be at least 2")

        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Embed a tensor of diffusion timesteps.

        :param timesteps: A one-dimensional tensor containing diffusion timesteps.
        :return: A tensor containing sinusoidal timestep embeddings.
        """
        half_dim = self.embedding_dim // 2
        frequencies = torch.exp(
            -log(10_000)
            * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
            / max(half_dim - 1, 1)
        )
        embeddings = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=1)
        if self.embedding_dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))
        return embeddings


class DenseDDPM(nn.Module):
    """A fully-connected denoising diffusion probabilistic model."""

    def __init__(
        self,
        input_shape: Tuple[int, ...] = (1, 28, 28),
        hidden_dims: Sequence[int] = (512, 512, 256),
        time_embedding_dim: int = 64,
        timesteps: int = 1_000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        """Initialize a `DenseDDPM` module.

        :param input_shape: The shape of each input sample.
        :param hidden_dims: The hidden dimensions used by the noise prediction network.
        :param time_embedding_dim: The number of dimensions in each timestep embedding.
        :param timesteps: The number of diffusion timesteps.
        :param beta_start: The first value in the linear noise schedule.
        :param beta_end: The final value in the linear noise schedule.
        """
        super().__init__()

        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one dimension")
        if timesteps <= 0:
            raise ValueError("timesteps must be positive")
        if not 0 < beta_start < beta_end < 1:
            raise ValueError("beta values must satisfy 0 < beta_start < beta_end < 1")

        self.input_shape = tuple(input_shape)
        self.input_size = prod(self.input_shape)
        self.timesteps = timesteps

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU(),
        )

        layers = []
        in_features = self.input_size + time_embedding_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(in_features, hidden_dim), nn.SiLU()])
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, self.input_size))
        self.noise_predictor = nn.Sequential(*layers)

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_previous = F.pad(alpha_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod))
        self.register_buffer("sqrt_reciprocal_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alpha_cumprod_previous) / (1.0 - alpha_cumprod),
        )

    def _extract(self, values: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return values.gather(0, timesteps).view(timesteps.size(0), *((1,) * len(self.input_shape)))

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Predict the noise contained in a batch of noisy inputs.

        :param x: A tensor containing noisy inputs.
        :param timesteps: A tensor containing the corresponding diffusion timesteps.
        :return: A tensor containing predicted noise.
        """
        flattened = x.flatten(start_dim=1)
        time_embeddings = self.time_embedding(timesteps)
        predicted_noise = self.noise_predictor(torch.cat((flattened, time_embeddings), dim=1))
        return predicted_noise.view(x.size(0), *self.input_shape)

    def q_sample(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the forward diffusion process to clean inputs.

        :param x_start: A tensor containing clean inputs.
        :param timesteps: A tensor containing diffusion timesteps.
        :param noise: Optional Gaussian noise to add to the inputs.
        :return: A tensor containing noisy inputs at the requested timesteps.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        signal_scale = self._extract(self.sqrt_alpha_cumprod, timesteps)
        noise_scale = self._extract(self.sqrt_one_minus_alpha_cumprod, timesteps)
        return signal_scale * x_start + noise_scale * noise

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Perform one reverse diffusion step.

        :param x: A tensor containing noisy inputs.
        :param timesteps: A tensor containing the current diffusion timesteps.
        :return: A tensor containing inputs from the previous diffusion timestep.
        """
        betas = self._extract(self.betas, timesteps)
        noise_scale = self._extract(self.sqrt_one_minus_alpha_cumprod, timesteps)
        reciprocal_alpha = self._extract(self.sqrt_reciprocal_alphas, timesteps)
        predicted_noise = self.forward(x, timesteps)
        model_mean = reciprocal_alpha * (x - betas * predicted_noise / noise_scale)

        posterior_variance = self._extract(self.posterior_variance, timesteps)
        noise = torch.randn_like(x)
        nonzero_mask = (
            (timesteps != 0).float().view(timesteps.size(0), *((1,) * len(self.input_shape)))
        )
        return model_mean + nonzero_mask * torch.sqrt(posterior_variance) * noise

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device | str) -> torch.Tensor:
        """Generate samples by iteratively reversing the diffusion process.

        :param num_samples: The number of samples to generate.
        :param device: The device on which to generate samples.
        :return: A tensor containing generated samples.
        """
        samples = torch.randn(num_samples, *self.input_shape, device=device)
        for timestep in reversed(range(self.timesteps)):
            timesteps = torch.full((num_samples,), timestep, device=device, dtype=torch.long)
            samples = self.p_sample(samples, timesteps)
        return samples


if __name__ == "__main__":
    _ = DenseDDPM()
