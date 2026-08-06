from math import log, pi, prod
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class ContinuousTimeEmbedding(nn.Module):
    """A sinusoidal embedding for continuous flow times."""

    def __init__(self, embedding_dim: int) -> None:
        """Initialize a `ContinuousTimeEmbedding` module.

        :param embedding_dim: The number of dimensions in each time embedding.
        """
        super().__init__()

        if embedding_dim < 2:
            raise ValueError("embedding_dim must be at least 2")

        self.embedding_dim = embedding_dim

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        """Embed a tensor of continuous times.

        :param times: A one-dimensional tensor containing values in the interval `[0, 1]`.
        :return: A tensor containing sinusoidal time embeddings.
        """
        half_dim = self.embedding_dim // 2
        frequencies = torch.exp(
            log(10_000)
            * torch.arange(half_dim, device=times.device, dtype=torch.float32)
            / max(half_dim - 1, 1)
        )
        embeddings = 2 * pi * times.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=1)
        if self.embedding_dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))
        return embeddings


class DenseFlowMatching(nn.Module):
    """A fully-connected vector field for flow matching."""

    def __init__(
        self,
        input_shape: Tuple[int, ...] = (1, 28, 28),
        hidden_dims: Sequence[int] = (512, 512, 256),
        time_embedding_dim: int = 64,
        integration_steps: int = 100,
    ) -> None:
        """Initialize a `DenseFlowMatching` module.

        :param input_shape: The shape of each input sample.
        :param hidden_dims: The hidden dimensions used by the velocity prediction network.
        :param time_embedding_dim: The number of dimensions in each time embedding.
        :param integration_steps: The default number of Euler integration steps used for sampling.
        """
        super().__init__()

        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one dimension")
        if integration_steps <= 0:
            raise ValueError("integration_steps must be positive")

        self.input_shape = tuple(input_shape)
        self.input_size = prod(self.input_shape)
        self.integration_steps = integration_steps

        self.time_embedding = nn.Sequential(
            ContinuousTimeEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU(),
        )

        layers = []
        in_features = self.input_size + time_embedding_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(in_features, hidden_dim), nn.SiLU()])
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, self.input_size))
        self.velocity_predictor = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Predict the velocity of samples along the probability path.

        :param x: A tensor containing interpolated samples.
        :param times: A tensor containing continuous flow times.
        :return: A tensor containing predicted velocities.
        """
        flattened = x.flatten(start_dim=1)
        time_embeddings = self.time_embedding(times)
        velocity = self.velocity_predictor(torch.cat((flattened, time_embeddings), dim=1))
        return velocity.view(x.size(0), *self.input_shape)

    def interpolate(
        self,
        data: torch.Tensor,
        noise: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate linearly from noise to data samples.

        :param data: A tensor containing data samples.
        :param noise: A tensor containing Gaussian noise samples.
        :param times: A tensor containing continuous flow times.
        :return: A tensor containing samples along the linear probability path.
        """
        time_shape = (times.size(0),) + (1,) * len(self.input_shape)
        times = times.view(time_shape)
        return (1.0 - times) * noise + times * data

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        device: torch.device | str,
        integration_steps: int | None = None,
    ) -> torch.Tensor:
        """Generate samples by integrating the learned vector field with Euler's method.

        :param num_samples: The number of samples to generate.
        :param device: The device on which to generate samples.
        :param integration_steps: Optional number of Euler integration steps.
        :return: A tensor containing generated samples.
        """
        steps = integration_steps or self.integration_steps
        if steps <= 0:
            raise ValueError("integration_steps must be positive")

        samples = torch.randn(num_samples, *self.input_shape, device=device)
        step_size = 1.0 / steps
        for step in range(steps):
            times = torch.full(
                (num_samples,),
                step / steps,
                device=device,
                dtype=samples.dtype,
            )
            samples = samples + step_size * self.forward(samples, times)
        return samples


if __name__ == "__main__":
    _ = DenseFlowMatching()
