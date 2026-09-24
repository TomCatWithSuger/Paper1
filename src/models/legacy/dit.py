from math import log, pi

import torch
import torch.nn.functional as F
from torch import nn


class TimestepEmbedding(nn.Module):
    """A sinusoidal embedding for continuous times."""

    def __init__(self, embedding_dim: int) -> None:
        """Initialize the sinusoidal time embedding."""
        super().__init__()

        if embedding_dim < 2:
            raise ValueError("embedding_dim must be at least 2")

        self.embedding_dim = embedding_dim

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        """Compute sinusoidal embeddings for the given timesteps."""
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


class TransformerBlock(nn.Module):
    """A pre-norm transformer block conditioned on an external embedding."""

    def __init__(self, embed_dim: int, num_heads: int, context_dim: int) -> None:
        """Initialize the transformer block with cross-attention conditioning."""
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.norm = nn.LayerNorm(embed_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.condition_proj = nn.Linear(context_dim, embed_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Apply cross-attention and feed-forward with external context conditioning."""
        context = self.condition_proj(context).unsqueeze(1)
        hidden = self.norm(x)
        hidden, _ = self.self_attention(query=hidden, key=context, value=context)
        x = x + hidden
        x = x + self.feed_forward(x)
        return x


class DiT1D(nn.Module):
    """A minimal one-dimensional diffusion transformer for waveform modeling."""

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 64,
        context_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        integration_steps: int = 100,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if integration_steps <= 0:
            raise ValueError("integration_steps must be positive")

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.integration_steps = integration_steps

        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.time_embedding = nn.Sequential(
            TimestepEmbedding(context_dim),
            nn.Linear(context_dim, context_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads, context_dim) for _ in range(num_layers)]
        )
        self.output_proj = nn.Linear(hidden_dim, in_channels)

    def forward(self, waveforms: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Predict the velocity field for noisy waveforms.

        :param waveforms: Tensor of shape ``[batch, time, channels]``.
        :param times: Tensor of shape ``[batch]`` with continuous timesteps.
        :return: Predicted velocity with the same shape as ``waveforms``.
        """
        if waveforms.ndim != 3:
            raise ValueError("waveforms must have shape [batch, time, channels]")
        if waveforms.size(-1) != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {waveforms.size(-1)}"
            )

        hidden = self.input_proj(waveforms)
        context = self.time_embedding(times)
        for block in self.blocks:
            hidden = block(hidden, context)
        return self.output_proj(hidden)

    def interpolate(
        self,
        data: torch.Tensor,
        noise: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        """Linear interpolation from noise to data at given times."""
        time_shape = (times.size(0),) + (1,) * (data.ndim - 1)
        times = times.view(time_shape)
        return (1.0 - times) * noise + times * data

    @torch.no_grad()
    def sample(
        self,
        num_frames: int,
        device: torch.device | str,
        integration_steps: int | None = None,
    ) -> torch.Tensor:
        """Generate waveforms via Euler integration of the learned velocity field."""
        steps = integration_steps or self.integration_steps
        samples = torch.randn(1, num_frames, self.in_channels, device=device)
        step_size = 1.0 / steps
        for step in range(steps):
            times = torch.full((1,), step / steps, device=device, dtype=samples.dtype)
            samples = samples + step_size * self.forward(samples, times)
        return samples.squeeze(0)


if __name__ == "__main__":
    _ = DiT1D()
