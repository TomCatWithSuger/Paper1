from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock1D(nn.Module):
    """A two-layer convolutional block for one-dimensional signals."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(1, out_channels),
            nn.SiLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(1, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet1D(nn.Module):
    """A one-dimensional U-Net for waveform reconstruction."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: Sequence[int] = (8, 16, 32),
        kernel_size: int = 5,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if out_channels <= 0:
            raise ValueError("out_channels must be positive")
        if not channels or any(channel <= 0 for channel in channels):
            raise ValueError("channels must contain positive dimensions")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = tuple(channels)

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        current_channels = in_channels
        for channel in self.channels:
            self.encoders.append(ConvBlock1D(current_channels, channel, kernel_size))
            self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
            current_channels = channel

        self.bottleneck = ConvBlock1D(
            self.channels[-1], self.channels[-1] * 2, kernel_size
        )

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        current_channels = self.channels[-1] * 2
        for channel in reversed(self.channels):
            self.upconvs.append(
                nn.ConvTranspose1d(current_channels, channel, kernel_size=2, stride=2)
            )
            self.decoders.append(ConvBlock1D(channel * 2, channel, kernel_size))
            current_channels = channel

        self.output = nn.Conv1d(self.channels[0], out_channels, kernel_size=1)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Reconstruct a batch of padded waveforms."""
        squeeze_channel = waveforms.ndim == 2
        if squeeze_channel:
            waveforms = waveforms.unsqueeze(1)
        if waveforms.ndim != 3:
            raise ValueError("waveforms must have shape [batch, time] or [batch, channels, time]")
        if waveforms.size(1) != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {waveforms.size(1)}"
            )
        if waveforms.size(-1) < 2 ** len(self.channels):
            raise ValueError("waveform length is too short for the configured U-Net depth")

        skips = []
        x = waveforms
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            if x.size(-1) != skip.size(-1):
                x = F.interpolate(x, size=skip.size(-1), mode="linear", align_corners=False)
            x = decoder(torch.cat((skip, x), dim=1))

        reconstruction = self.output(x)
        if squeeze_channel and self.out_channels == 1:
            reconstruction = reconstruction.squeeze(1)
        return reconstruction


if __name__ == "__main__":
    _ = UNet1D()
