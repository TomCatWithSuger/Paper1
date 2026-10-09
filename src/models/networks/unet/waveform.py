"""带跳跃连接的一维波形重构网络。"""

import torch
import torch.nn.functional as F
from torch import nn

from src.models.components.blocks.waveform import ConvBlock1D


class UNet1D(nn.Module):
    """使用多尺度跳跃连接重构一维波形。"""

    def __init__(self, in_channels=1, out_channels=1, channels=(8, 16, 32), kernel_size=5):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("input/output channels must be positive")
        if not channels or any(channel <= 0 for channel in channels):
            raise ValueError("channels must contain positive dimensions")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = tuple(channels)
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        current = in_channels
        for channel in self.channels:
            self.encoders.append(ConvBlock1D(current, channel, kernel_size))
            self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
            current = channel
        self.bottleneck = ConvBlock1D(self.channels[-1], self.channels[-1] * 2, kernel_size)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        current = self.channels[-1] * 2
        for channel in reversed(self.channels):
            self.upconvs.append(nn.ConvTranspose1d(current, channel, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock1D(channel * 2, channel, kernel_size))
            current = channel
        self.output = nn.Conv1d(self.channels[0], out_channels, kernel_size=1)

    def forward(self, waveforms):
        squeeze_channel = waveforms.ndim == 2
        if squeeze_channel:
            waveforms = waveforms.unsqueeze(1)
        if waveforms.ndim != 3:
            raise ValueError("waveforms must have shape [batch, time] or [batch, channels, time]")
        if waveforms.size(1) != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {waveforms.size(1)}")
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
        return reconstruction.squeeze(1) if squeeze_channel and self.out_channels == 1 else reconstruction
