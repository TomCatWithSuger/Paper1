"""使用权重归一化和 GLU 的一维门控卷积。"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


class GatedConv1d(nn.Module):
    """使用权重归一化卷积和 GLU 的一维门控卷积层。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | None = None,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or kernel_size <= 0 or stride <= 0:
            raise ValueError("卷积参数必须为正数")
        resolved_padding = kernel_size // 2 if padding is None else padding
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels * 2,
                kernel_size=kernel_size,
                stride=stride,
                padding=resolved_padding,
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """执行门控卷积。"""
        return F.glu(self.conv(inputs), dim=1)


class GatedConvTranspose1d(nn.Module):
    """使用权重归一化转置卷积和 GLU 的一维上采样层。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels 必须为正数")
        self.conv = weight_norm(
            nn.ConvTranspose1d(
                channels,
                channels * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """执行两倍上采样和 GLU 门控。"""
        return F.glu(self.conv(inputs), dim=1)
