"""Mel 域条件 Flow Matching 的 FastVoiceGrad 风格一维 U-Net。

- 提供带权重归一化和 GLU 的卷积组件。
- 提供统一的 ``model(x_t, t, h)`` 速度场接口。
- 提供线性概率路径插值和多步 Euler 生成能力。
"""

# ====================
# 1. 导入
# ====================

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from src.models.components.conditional_flow_matching import ContinuousTimeEmbedding

# ====================
# 2. 定义
# 门控卷积组件
# ====================


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
        """初始化普通或下采样门控卷积。

        :param in_channels: 输入通道数。
        :param out_channels: GLU 门控后的输出通道数。
        :param kernel_size: 卷积核大小。
        :param stride: 卷积步长，取 2 时完成下采样。
        :param padding: 可选填充大小，默认保持步长为 1 时的序列长度。
        """
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
        """通过 GLU 将一半通道作为内容、另一半通道作为门控。"""
        return F.glu(self.conv(inputs), dim=1)


class GatedConvTranspose1d(nn.Module):
    """使用权重归一化转置卷积和 GLU 的一维上采样层。"""

    def __init__(self, channels: int) -> None:
        """初始化将时间长度放大两倍的门控转置卷积。"""
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
        """执行两倍上采样并应用 GLU 门控。"""
        return F.glu(self.conv(inputs), dim=1)


# ====================
# 3. 核心模型
# 条件 Flow Matching U-Net
# ====================


class ConditionalFlowMatchingUNet(nn.Module):
    """预测条件 Flow Matching 速度场的 12 层一维 U-Net。

    编码器逐步压缩时间分辨率以扩大感受野，瓶颈层在最低分辨率上建模长程结构，解码器再 恢复原始帧率。跳跃连接保留局部语音细节，GLU 控制特征通过比例，权重归一化用于稳定 深层卷积网络的训练。
    """

    # ====================
    # 3.1 初始化
    # ====================

    def __init__(
        self,
        n_mels: int = 80,
        condition_dim: int = 128,
        hidden_channels: int = 512,
        time_embedding_dim: int = 128,
        kernel_size: int = 3,
        integration_steps: int = 50,
    ) -> None:
        """初始化 FastVoiceGrad 风格的条件 U-Net。

        :param n_mels: 输入状态和预测速度的 Mel 通道数。
        :param condition_dim: 统一条件 ``h`` 的通道数。
        :param hidden_channels: 所有 U-Net 隐藏层的通道数，默认使用 512。
        :param time_embedding_dim: 连续时间嵌入的维数。
        :param kernel_size: 非缩放卷积的卷积核大小，必须为正奇数。
        :param integration_steps: 普通 Flow Matching 推理默认使用的 Euler 步数。
        """
        super().__init__()
        if n_mels <= 0 or condition_dim <= 0 or hidden_channels <= 0:
            raise ValueError("模型通道数必须为正数")
        if time_embedding_dim < 2 or integration_steps <= 0:
            raise ValueError("时间嵌入维数和积分步数必须有效")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size 必须为正奇数")

        self.n_mels = n_mels
        self.condition_dim = condition_dim
        self.hidden_channels = hidden_channels
        self.integration_steps = integration_steps

        self.time_embedding = nn.Sequential(
            ContinuousTimeEmbedding(time_embedding_dim),
            nn.Linear(time_embedding_dim, hidden_channels),
            nn.SiLU(),
        )

        self.input_conv = GatedConv1d(
            n_mels + condition_dim,
            hidden_channels,
            kernel_size,
        )
        self.encoder_block1 = GatedConv1d(hidden_channels, hidden_channels, kernel_size)
        self.downsample1 = GatedConv1d(
            hidden_channels,
            hidden_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.encoder_block2 = GatedConv1d(hidden_channels, hidden_channels, kernel_size)
        self.downsample2 = GatedConv1d(
            hidden_channels,
            hidden_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.bottleneck_block1 = GatedConv1d(hidden_channels, hidden_channels, kernel_size)
        self.bottleneck_block2 = GatedConv1d(hidden_channels, hidden_channels, kernel_size)
        self.upsample1 = GatedConvTranspose1d(hidden_channels)
        self.decoder_block1 = GatedConv1d(hidden_channels, hidden_channels, kernel_size)
        self.upsample2 = GatedConvTranspose1d(hidden_channels)
        self.decoder_block2 = GatedConv1d(hidden_channels, hidden_channels, kernel_size)
        self.output_conv = weight_norm(
            nn.Conv1d(hidden_channels, n_mels, kernel_size=kernel_size, padding=kernel_size // 2)
        )

    # ====================
    # 3.2 工具函数
    # 张量尺寸与掩码处理
    # ====================

    @staticmethod
    def _resize_mask(mask: torch.Tensor | None, frames: int) -> torch.Tensor | None:
        """使用最近邻插值将有效帧掩码调整到当前 U-Net 分辨率。"""
        if mask is None or mask.size(1) == frames:
            return mask
        resized = F.interpolate(mask.unsqueeze(1).float(), size=frames, mode="nearest")
        return resized.squeeze(1).bool()

    @staticmethod
    def _apply_mask(hidden: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """将当前分辨率下的填充位置置零。"""
        if mask is None:
            return hidden
        return hidden * mask.unsqueeze(1).to(hidden.dtype)

    @staticmethod
    def _align(hidden: torch.Tensor, frames: int) -> torch.Tensor:
        """修正奇数长度在上下采样后产生的单帧尺寸差异。"""
        if hidden.size(2) == frames:
            return hidden
        return F.interpolate(hidden, size=frames, mode="linear", align_corners=False)

    @staticmethod
    def _add_time(hidden: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        """将同一个全局流时间特征注入当前分辨率的所有帧。"""
        return hidden + time_embedding.unsqueeze(2)

    # ====================
    # 3.3 内部函数
    # 输入校验与共享主干
    # ====================

    def _validate_forward_inputs(
        self,
        noisy_mels: torch.Tensor,
        times: torch.Tensor,
        condition: torch.Tensor,
    ) -> None:
        """校验速度场输入的维度、通道数和批大小。"""
        if noisy_mels.ndim != 3 or condition.ndim != 3:
            raise ValueError("noisy_mels 和 condition 必须是三维张量")
        if noisy_mels.size(1) != self.n_mels:
            raise ValueError(f"noisy_mels 必须包含 {self.n_mels} 个通道")
        if condition.size(1) != self.condition_dim:
            raise ValueError(f"condition 必须包含 {self.condition_dim} 个通道")
        if noisy_mels.size(0) != condition.size(0):
            raise ValueError("noisy_mels 和 condition 的 batch 大小必须一致")
        if times.ndim != 1 or times.size(0) != noisy_mels.size(0):
            raise ValueError("times 必须具有形状 [batch]")

    def _forward_with_time_embedding(
        self,
        noisy_mels: torch.Tensor,
        condition: torch.Tensor,
        time_embedding: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """使用预先构造的时间条件执行共享 U-Net 主干。"""
        frames = noisy_mels.size(2)
        condition = self._align(condition, frames)
        if target_mask is not None:
            valid = target_mask.unsqueeze(1).to(noisy_mels.dtype)
            noisy_mels = noisy_mels * valid
            condition = condition * valid

        hidden = self.input_conv(torch.cat((noisy_mels, condition), dim=1))
        hidden = self._apply_mask(self._add_time(hidden, time_embedding), target_mask)
        skip1 = self.encoder_block1(hidden)
        skip1 = self._apply_mask(self._add_time(skip1, time_embedding), target_mask)

        hidden = self.downsample1(skip1)
        mask1 = self._resize_mask(target_mask, hidden.size(2))
        hidden = self._apply_mask(self._add_time(hidden, time_embedding), mask1)
        skip2 = self.encoder_block2(hidden)
        skip2 = self._apply_mask(self._add_time(skip2, time_embedding), mask1)

        hidden = self.downsample2(skip2)
        mask2 = self._resize_mask(target_mask, hidden.size(2))
        hidden = self._apply_mask(self._add_time(hidden, time_embedding), mask2)
        hidden = self.bottleneck_block1(hidden)
        hidden = self._apply_mask(self._add_time(hidden, time_embedding), mask2)
        hidden = self.bottleneck_block2(hidden)
        hidden = self._apply_mask(self._add_time(hidden, time_embedding), mask2)

        hidden = self._align(self.upsample1(hidden), skip2.size(2))
        hidden = self._apply_mask(self._add_time(hidden, time_embedding), mask1)
        hidden = self.decoder_block1(hidden + skip2)
        hidden = self._apply_mask(self._add_time(hidden, time_embedding), mask1)

        hidden = self._align(self.upsample2(hidden), skip1.size(2))
        hidden = self._apply_mask(self._add_time(hidden, time_embedding), target_mask)
        hidden = self.decoder_block2(hidden + skip1)
        hidden = self._apply_mask(self._add_time(hidden, time_embedding), target_mask)

        velocity = self.output_conv(hidden)
        return self._apply_mask(velocity, target_mask)

    # ====================
    # 3.4 核心流程
    # 速度场预测
    # ====================

    def forward(
        self,
        noisy_mels: torch.Tensor,
        times: torch.Tensor,
        condition: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """根据 ``x_t``、``t`` 和统一条件 ``h`` 预测 Mel 速度。

        :param noisy_mels: 形状为 ``[batch, n_mels, frames]`` 的带噪状态 ``x_t``。
        :param times: 形状为 ``[batch]`` 的连续流时间。
        :param condition: 形状为 ``[batch, condition_dim, condition_frames]`` 的条件 ``h``。
        :param target_mask: 形状为 ``[batch, frames]`` 的有效帧掩码。
        :return: 与 ``noisy_mels`` 形状相同的速度张量。
        """
        self._validate_forward_inputs(noisy_mels, times, condition)
        time_embedding = self.time_embedding(times)
        return self._forward_with_time_embedding(
            noisy_mels=noisy_mels,
            condition=condition,
            time_embedding=time_embedding,
            target_mask=target_mask,
        )

    # ====================
    # 3.5 对外接口
    # 插值与生成
    # ====================

    @staticmethod
    def interpolate(
        data: torch.Tensor,
        noise: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        """构造从高斯噪声 ``x_0`` 到干净 Mel ``x_1`` 的线性概率路径。"""
        time_shape = (times.size(0),) + (1,) * (data.ndim - 1)
        return (1.0 - times.view(time_shape)) * noise + times.view(time_shape) * data

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        integration_steps: int | None = None,
    ) -> torch.Tensor:
        """使用多步 Euler 积分从高斯噪声生成目标 Mel 频谱。"""
        steps = integration_steps or self.integration_steps
        if steps <= 0:
            raise ValueError("integration_steps 必须为正数")

        samples = torch.randn(
            condition.size(0),
            self.n_mels,
            condition.size(2),
            device=condition.device,
            dtype=condition.dtype,
        )
        step_size = 1.0 / steps
        for step in range(steps):
            times = torch.full(
                (samples.size(0),),
                step / steps,
                device=samples.device,
                dtype=samples.dtype,
            )
            samples = samples + step_size * self.forward(
                noisy_mels=samples,
                times=times,
                condition=condition,
                target_mask=target_mask,
            )
        return self._apply_mask(samples, target_mask)


# ====================
# 4. 入口
# ====================


if __name__ == "__main__":
    _ = ConditionalFlowMatchingUNet()
