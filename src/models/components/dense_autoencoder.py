from math import prod
from typing import Sequence, Tuple

import torch
from torch import nn


class DenseAutoencoder(nn.Module):
    """A fully-connected autoencoder for image reconstruction."""

    def __init__(
        self,
        input_shape: Tuple[int, ...] = (1, 28, 28),
        hidden_dims: Sequence[int] = (512, 256),
        latent_dim: int = 32,
    ) -> None:
        """Initialize a `DenseAutoencoder` module.

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
        encoder_layers.append(nn.Linear(in_features, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_features = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(in_features, hidden_dim), nn.ReLU()])
            in_features = hidden_dim
        decoder_layers.append(nn.Linear(in_features, self.input_size))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an input tensor into a latent tensor.

        :param x: The input tensor.
        :return: The latent tensor.
        """
        return self.encoder(x.flatten(start_dim=1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent tensor into a reconstructed input tensor.

        :param z: The latent tensor.
        :return: A reconstructed input tensor.
        """
        return self.decoder(z).view(z.size(0), *self.input_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the autoencoder.

        :param x: The input tensor.
        :return: A reconstructed input tensor.
        """
        return self.decode(self.encode(x))


if __name__ == "__main__":
    _ = DenseAutoencoder()
