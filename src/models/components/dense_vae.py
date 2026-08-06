from math import prod
from typing import Sequence, Tuple

import torch
from torch import nn


class DenseVAE(nn.Module):
    """A fully-connected variational autoencoder for image reconstruction."""

    def __init__(
        self,
        input_shape: Tuple[int, ...] = (1, 28, 28),
        hidden_dims: Sequence[int] = (512, 256),
        latent_dim: int = 32,
    ) -> None:
        """Initialize a `DenseVAE` module.

        :param input_shape: The shape of each input sample.
        :param hidden_dims: The hidden dimensions used by the encoder and decoder.
        :param latent_dim: The number of dimensions in the latent space.
        """
        super().__init__()

        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one dimension")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")

        self.input_shape = tuple(input_shape)
        self.input_size = prod(self.input_shape)
        self.latent_dim = latent_dim

        encoder_layers = []
        in_features = self.input_size
        for hidden_dim in hidden_dims:
            encoder_layers.extend([nn.Linear(in_features, hidden_dim), nn.ReLU()])
            in_features = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(in_features, latent_dim)
        self.fc_logvar = nn.Linear(in_features, latent_dim)

        decoder_layers = []
        in_features = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(in_features, hidden_dim), nn.ReLU()])
            in_features = hidden_dim
        decoder_layers.append(nn.Linear(in_features, self.input_size))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode an input tensor into a latent distribution.

        :param x: The input tensor.
        :return: A tuple containing the mean and log-variance tensors.
        """
        features = self.encoder(x.flatten(start_dim=1))
        return self.fc_mu(features), self.fc_logvar(features)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample a latent tensor using the reparameterization trick.

        :param mu: The mean of the latent distribution.
        :param logvar: The log-variance of the latent distribution.
        :return: A sampled latent tensor.
        """
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent tensor into a reconstructed input tensor.

        :param z: The latent tensor.
        :return: A reconstructed input tensor.
        """
        return self.decoder(z).view(z.size(0), *self.input_shape)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single forward pass through the variational autoencoder.

        :param x: The input tensor.
        :return: A tuple containing the reconstruction, mean, and log-variance tensors.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def sample(self, num_samples: int, device: torch.device | str) -> torch.Tensor:
        """Generate samples from the standard normal latent distribution.

        :param num_samples: The number of samples to generate.
        :param device: The device on which to generate samples.
        :return: A tensor containing generated samples.
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)


if __name__ == "__main__":
    _ = DenseVAE()
