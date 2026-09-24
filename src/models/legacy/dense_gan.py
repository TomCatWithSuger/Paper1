from math import prod
from typing import Sequence, Tuple

import torch
from torch import nn


class DenseGenerator(nn.Module):
    """A fully-connected generator for image synthesis."""

    def __init__(
        self,
        latent_dim: int,
        output_shape: Tuple[int, ...],
        hidden_dims: Sequence[int],
    ) -> None:
        """Initialize a `DenseGenerator` module.

        :param latent_dim: The number of dimensions in each latent vector.
        :param output_shape: The shape of each generated sample.
        :param hidden_dims: The hidden dimensions used by the generator.
        """
        super().__init__()

        output_size = prod(output_shape)
        layers = []
        in_features = latent_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_features, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(0.2),
                ]
            )
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, output_size))

        self.output_shape = tuple(output_shape)
        self.model = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Generate samples from latent vectors.

        :param latent: A tensor containing latent vectors.
        :return: A tensor containing generated samples.
        """
        generated = self.model(latent)
        return generated.view(latent.size(0), *self.output_shape)


class DenseDiscriminator(nn.Module):
    """A fully-connected discriminator for image samples."""

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        hidden_dims: Sequence[int],
        dropout: float = 0.2,
    ) -> None:
        """Initialize a `DenseDiscriminator` module.

        :param input_shape: The shape of each input sample.
        :param hidden_dims: The hidden dimensions used by the discriminator.
        :param dropout: The dropout probability used between hidden layers.
        """
        super().__init__()

        input_size = prod(input_shape)
        layers = []
        in_features = input_size
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_features, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout),
                ]
            )
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict whether input samples are real or generated.

        :param x: A tensor containing input samples.
        :return: A tensor containing discriminator logits.
        """
        return self.model(x.flatten(start_dim=1))


class DenseGAN(nn.Module):
    """A fully-connected generative adversarial network."""

    def __init__(
        self,
        input_shape: Tuple[int, ...] = (1, 28, 28),
        latent_dim: int = 100,
        generator_hidden_dims: Sequence[int] = (256, 512, 1024),
        discriminator_hidden_dims: Sequence[int] = (512, 256),
        discriminator_dropout: float = 0.2,
    ) -> None:
        """Initialize a `DenseGAN` module.

        :param input_shape: The shape of each real or generated sample.
        :param latent_dim: The number of dimensions in each latent vector.
        :param generator_hidden_dims: The hidden dimensions used by the generator.
        :param discriminator_hidden_dims: The hidden dimensions used by the discriminator.
        :param discriminator_dropout: The dropout probability used by the discriminator.
        """
        super().__init__()

        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if not generator_hidden_dims:
            raise ValueError("generator_hidden_dims must contain at least one dimension")
        if not discriminator_hidden_dims:
            raise ValueError("discriminator_hidden_dims must contain at least one dimension")
        if not 0 <= discriminator_dropout < 1:
            raise ValueError("discriminator_dropout must be in the interval [0, 1)")

        self.input_shape = tuple(input_shape)
        self.latent_dim = latent_dim
        self.generator = DenseGenerator(
            latent_dim=latent_dim,
            output_shape=self.input_shape,
            hidden_dims=generator_hidden_dims,
        )
        self.discriminator = DenseDiscriminator(
            input_shape=self.input_shape,
            hidden_dims=discriminator_hidden_dims,
            dropout=discriminator_dropout,
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Generate samples from latent vectors.

        :param latent: A tensor containing latent vectors.
        :return: A tensor containing generated samples.
        """
        return self.generator(latent)

    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        """Compute discriminator logits for input samples.

        :param x: A tensor containing real or generated samples.
        :return: A tensor containing discriminator logits.
        """
        return self.discriminator(x)

    def generate(self, num_samples: int, device: torch.device | str) -> torch.Tensor:
        """Generate samples from standard normal latent vectors.

        :param num_samples: The number of samples to generate.
        :param device: The device on which to generate samples.
        :return: A tensor containing generated samples.
        """
        latent = torch.randn(num_samples, self.latent_dim, device=device)
        return self.forward(latent)


if __name__ == "__main__":
    _ = DenseGAN()
