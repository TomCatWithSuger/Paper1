"""Mel 域条件 Flow Matching 的 FastVoiceGrad 风格一维 U-Net。

- 提供带权重归一化和 GLU 的卷积组件。
- 提供统一的 ``model(x_t, t, h)`` 速度场接口。
- 提供线性概率路径插值和多步 Euler 生成能力。
"""

# ====================
# 1. 导入
# ====================

import torch

from src.models.components.flow_matching_backbone import ConditionalUNetBackbone

# ====================
# 3. 核心模型
# 条件 Flow Matching U-Net
# ====================


class ConditionalFlowMatchingUNet(ConditionalUNetBackbone):
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
        super().__init__(
            n_mels=n_mels,
            condition_dim=condition_dim,
            hidden_channels=hidden_channels,
            time_embedding_dim=time_embedding_dim,
            kernel_size=kernel_size,
        )
        if integration_steps <= 0:
            raise ValueError("integration_steps 必须为正数")
        self.integration_steps = integration_steps

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
        self._validate_inputs(noisy_mels, condition)
        if times.ndim != 1 or times.size(0) != noisy_mels.size(0):
            raise ValueError("times 必须具有形状 [batch]")
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
