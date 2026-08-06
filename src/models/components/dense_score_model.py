from math import log, pi, prod
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class NoiseLevelEmbedding(nn.Module):
    """A sinusoidal embedding for continuous noise levels."""

    def __init__(self, embedding_dim: int) -> None:
        """Initialize a `NoiseLevelEmbedding` module.

        :param embedding_dim: The number of dimensions in each noise-level embedding.
        """
        super().__init__()

        if embedding_dim < 2:
            raise ValueError("embedding_dim must be at least 2")

        self.embedding_dim = embedding_dim

    def forward(self, noise_levels: torch.Tensor) -> torch.Tensor:
        """Embed a tensor of continuous noise levels.

        :param noise_levels: A one-dimensional tensor containing positive noise levels.
        :return: A tensor containing sinusoidal noise-level embeddings.
        """
        half_dim = self.embedding_dim // 2
        frequencies = torch.exp(
            log(10_000)
            * torch.arange(half_dim, device=noise_levels.device, dtype=torch.float32)
            / max(half_dim - 1, 1)
        )
        embeddings = 2 * pi * noise_levels.log().unsqueeze(1) * frequencies.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=1)
        if self.embedding_dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))
        return embeddings


class DenseScoreModel(nn.Module):
    """A fully-connected noise-conditional score model."""

    def __init__(
        self,
        input_shape: Tuple[int, ...] = (1, 28, 28),
        hidden_dims: Sequence[int] = (512, 512, 256),
        noise_embedding_dim: int = 64,
        sigma_min: float = 0.01,
        sigma_max: float = 1.0,
        num_noise_levels: int = 100,
        sampling_steps_per_level: int = 10,
        sampling_step_size: float = 1e-5,
    ) -> None:
        """Initialize a `DenseScoreModel` module.

        :param input_shape: The shape of each input sample.
        :param hidden_dims: The hidden dimensions used by the score prediction network.
        :param noise_embedding_dim: The number of dimensions in each noise-level embedding.
        :param sigma_min: The minimum standard deviation in the noise schedule.
        :param sigma_max: The maximum standard deviation in the noise schedule.
        :param num_noise_levels: The number of noise levels used during sampling.
        :param sampling_steps_per_level: The number of Langevin steps at each noise level.
        :param sampling_step_size: The base step size used by annealed Langevin dynamics.
        """
        super().__init__()

        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one dimension")
        if not 0 < sigma_min < sigma_max:
            raise ValueError("noise levels must satisfy 0 < sigma_min < sigma_max")
        if num_noise_levels <= 0:
            raise ValueError("num_noise_levels must be positive")
        if sampling_steps_per_level <= 0:
            raise ValueError("sampling_steps_per_level must be positive")
        if sampling_step_size <= 0:
            raise ValueError("sampling_step_size must be positive")

        self.input_shape = tuple(input_shape)
        self.input_size = prod(self.input_shape)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sampling_steps_per_level = sampling_steps_per_level
        self.sampling_step_size = sampling_step_size

        self.noise_embedding = nn.Sequential(
            NoiseLevelEmbedding(noise_embedding_dim),
            nn.Linear(noise_embedding_dim, noise_embedding_dim),
            nn.SiLU(),
        )

        layers = []
        in_features = self.input_size + noise_embedding_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(in_features, hidden_dim), nn.SiLU()])
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, self.input_size))
        self.score_predictor = nn.Sequential(*layers)

        noise_levels = torch.exp(
            torch.linspace(log(sigma_max), log(sigma_min), num_noise_levels)
        )
        self.register_buffer("noise_levels", noise_levels)

    def _expand_noise_levels(self, noise_levels: torch.Tensor) -> torch.Tensor:
        return noise_levels.view(
            noise_levels.size(0), *((1,) * len(self.input_shape))
        )

    def forward(self, x: torch.Tensor, noise_levels: torch.Tensor) -> torch.Tensor:
        """Predict the score of noisy samples.

        :param x: A tensor containing noisy samples.
        :param noise_levels: A tensor containing the corresponding noise levels.
        :return: A tensor containing predicted scores.
        """
        flattened = x.flatten(start_dim=1)
        noise_embeddings = self.noise_embedding(noise_levels)
        scores = self.score_predictor(torch.cat((flattened, noise_embeddings), dim=1))
        return scores.view(x.size(0), *self.input_shape)

    def perturb(
        self,
        data: torch.Tensor,
        noise_levels: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perturb data samples with Gaussian noise.

        :param data: A tensor containing data samples.
        :param noise_levels: A tensor containing standard deviations for each sample.
        :param noise: Optional standard Gaussian noise.
        :return: A tensor containing perturbed samples.
        """
        if noise is None:
            noise = torch.randn_like(data)
        return data + self._expand_noise_levels(noise_levels) * noise

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        device: torch.device | str,
        steps_per_level: int | None = None,
    ) -> torch.Tensor:
        """Generate samples using annealed Langevin dynamics.

        :param num_samples: The number of samples to generate.
        :param device: The device on which to generate samples.
        :param steps_per_level: Optional number of Langevin steps at each noise level.
        :return: A tensor containing generated samples.
        """
        steps = steps_per_level or self.sampling_steps_per_level
        if steps <= 0:
            raise ValueError("steps_per_level must be positive")

        samples = torch.randn(num_samples, *self.input_shape, device=device) * self.sigma_max
        for noise_level in self.noise_levels:
            noise_levels = torch.full(
                (num_samples,),
                noise_level.item(),
                device=device,
                dtype=samples.dtype,
            )
            step_size = self.sampling_step_size * (
                noise_level / self.sigma_min
            ).pow(2)
            for _ in range(steps):
                scores = self.forward(samples, noise_levels)
                samples = samples + step_size * scores
                samples = samples + torch.sqrt(2.0 * step_size) * torch.randn_like(samples)
        return samples


if __name__ == "__main__":
    _ = DenseScoreModel()
