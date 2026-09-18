"""标准 Flow Matching 基线使用的条件速度场网络。

网络接收带噪 Mel 状态 ``x_t``、连续时间 ``t`` 和 ``ConditionEncoder`` 生成的统一条件
``h``，预测样本从 ``t=0`` 的高斯噪声移动到 ``t=1`` 的干净语音所需的速度。统一使用
``model(x_t, t, h)`` 接口，便于后续 Mean Flow、U-Net 或 DiT 主干复用相同条件管线。
"""

from math import log

import torch
import torch.nn.functional as F
from torch import nn

from src.models.components.flow_matching_backbone import ContinuousTimeEmbedding


class ConditionalTransformerBlock(nn.Module):
    """执行由流时间控制的自注意力和前馈更新。

    流时间为两个子层分别生成逐特征缩放和偏移参数。相比只在输入处加入一次时间向量，这种 自适应归一化使每个 Transformer 块都能根据概率路径上的当前位置调整行为。
    """

    def __init__(self, hidden_dim: int, num_heads: int, condition_dim: int) -> None:
        """初始化注意力、时间调制和前馈层。

        :param hidden_dim: Transformer 令牌的特征维数。
        :param num_heads: 自注意力头数。
        :param condition_dim: 时间嵌入的维数。
        """
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.modulation = nn.Linear(condition_dim, hidden_dim * 4)

    @staticmethod
    def _modulate(
        hidden: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        """对归一化后的序列特征应用自适应缩放和偏移。"""
        return hidden * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(
        self,
        hidden: torch.Tensor,
        condition: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """执行一个由时间控制的残差 Transformer 块。

        :param hidden: 形状为 ``[batch, frames, hidden_dim]`` 的序列特征。
        :param condition: 形状为 ``[batch, condition_dim]`` 的时间嵌入。
        :param padding_mask: 填充帧位置为 ``True``，这些位置会被自注意力忽略。
        :return: 与 ``hidden`` 形状相同的更新后序列。
        """
        # 分离调制参数，使注意力层和前馈层能够以不同方式响应时间条件。
        attention_scale, attention_shift, ffn_scale, ffn_shift = self.modulation(condition).chunk(
            4, dim=1
        )
        normalized = self._modulate(self.norm1(hidden), attention_scale, attention_shift)
        attended, _ = self.attention(
            query=normalized,
            key=normalized,
            value=normalized,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        hidden = hidden + attended
        normalized = self._modulate(self.norm2(hidden), ffn_scale, ffn_shift)
        return hidden + self.feed_forward(normalized)


class ConditionalFlowMatchingTransformer(nn.Module):
    """预测 Mel 域中的条件 Flow Matching 速度场。

    每个 Mel 帧都会投影为 Transformer 令牌。带噪状态 ``x_t``、结构化条件 ``h`` 和位置
    嵌入在同一隐藏空间中相加，流时间则通过每个 Transformer 块的自适应归一化独立注入。
    输出与 ``x_t`` 形状相同，表示 ``dx_t / dt``。
    """

    def __init__(
        self,
        n_mels: int = 80,
        hidden_dim: int = 256,
        condition_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        integration_steps: int = 50,
    ) -> None:
        """初始化条件 Flow Matching Transformer。

        :param n_mels: 输入和预测速度使用的 Mel 频率维数。
        :param hidden_dim: Transformer 令牌的特征维数。
        :param condition_dim: 条件 ``h`` 和时间嵌入的通道数。
        :param num_layers: 条件 Transformer 块的数量。
        :param num_heads: 每个块的注意力头数。
        :param integration_steps: 基线推理默认使用的 Euler 积分步数。
        """
        super().__init__()
        if n_mels <= 0 or hidden_dim <= 0 or condition_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if num_layers <= 0 or num_heads <= 0 or integration_steps <= 0:
            raise ValueError("layer, head, and integration counts must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.n_mels = n_mels
        self.hidden_dim = hidden_dim
        self.integration_steps = integration_steps

        self.noisy_projection = nn.Linear(n_mels, hidden_dim)
        self.condition_projection = nn.Linear(condition_dim, hidden_dim)
        self.time_embedding = nn.Sequential(
            ContinuousTimeEmbedding(condition_dim),
            nn.Linear(condition_dim, condition_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [
                ConditionalTransformerBlock(hidden_dim, num_heads, condition_dim)
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, n_mels)

    @staticmethod
    def _position_embedding(
        sequence_length: int,
        embedding_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """创建形状为 ``[1, frames, hidden_dim]`` 的帧位置特征。

        自注意力本身无法区分帧顺序，因此需要显式位置特征来保留语音帧的先后关系。
        """
        positions = torch.arange(sequence_length, device=device, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            -log(10_000)
            * torch.arange(0, embedding_dim, 2, device=device, dtype=torch.float32)
            / embedding_dim
        )
        embeddings = torch.zeros(sequence_length, embedding_dim, device=device)
        embeddings[:, 0::2] = torch.sin(positions * frequencies)
        embeddings[:, 1::2] = torch.cos(positions * frequencies[: embedding_dim // 2])
        return embeddings.to(dtype=dtype).unsqueeze(0)

    def forward(
        self,
        noisy_mels: torch.Tensor,
        times: torch.Tensor,
        condition: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """根据 ``x_t``、``t`` 和统一条件 ``h`` 预测速度。

        :param noisy_mels: 形状为 ``[batch, n_mels, frames]`` 的流状态 ``x_t``。
        :param times: 形状为 ``[batch]`` 的连续时间。
        :param condition: 结构化条件 ``h``，形状为
            ``[batch, condition_dim, condition_frames]``.
        :param target_mask: 形状为 ``[batch, frames]`` 的有效输出帧掩码。
        :return: 与 ``noisy_mels`` 形状相同的速度张量。
        """
        if noisy_mels.ndim != 3:
            raise ValueError("noisy_mels must have shape [batch, n_mels, frames]")
        if condition.ndim != 3:
            raise ValueError("condition must have shape [batch, condition_dim, frames]")
        if noisy_mels.size(1) != self.n_mels:
            raise ValueError(f"noisy_mels must contain {self.n_mels} channels")
        if condition.size(0) != noisy_mels.size(0):
            raise ValueError("condition batch size must match noisy_mels")
        if times.ndim != 1 or times.size(0) != noisy_mels.size(0):
            raise ValueError("times must have shape [batch]")

        if condition.size(2) != noisy_mels.size(2):
            condition = F.interpolate(
                condition,
                size=noisy_mels.size(2),
                mode="linear",
                align_corners=False,
            )
        noisy_hidden = self.noisy_projection(noisy_mels.transpose(1, 2))
        condition_hidden = self.condition_projection(condition.transpose(1, 2))
        position = self._position_embedding(
            noisy_mels.size(2),
            self.hidden_dim,
            noisy_mels.device,
            noisy_hidden.dtype,
        )
        # 内容、说话人身份、带噪状态和位置信息只在隐藏空间中融合。
        hidden = noisy_hidden + condition_hidden + position
        time_condition = self.time_embedding(times)

        # MultiheadAttention 要求被忽略的填充位置为 True。
        padding_mask = None if target_mask is None else ~target_mask
        for block in self.blocks:
            hidden = block(hidden, time_condition, padding_mask)
        velocity = self.output_projection(self.output_norm(hidden)).transpose(1, 2)
        if target_mask is not None:
            velocity = velocity * target_mask.unsqueeze(1).to(velocity.dtype)
        return velocity

    @staticmethod
    def interpolate(
        data: torch.Tensor,
        noise: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        """构造标准线性 Flow Matching 概率路径。

        路径为 ``x_t = (1 - t) * x_0 + t * x_1``，其中 ``x_0`` 是高斯噪声，``x_1`` 是
        干净的目标 Mel 频谱。

        :param data: 干净端点 ``x_1``。
        :param noise: 与 ``data`` 形状相同的高斯噪声端点 ``x_0``。
        :param times: 形状为 ``[batch]`` 的逐样本插值时间。
        :return: 与 ``data`` 形状相同的插值状态 ``x_t``。
        """
        time_shape = (times.size(0),) + (1,) * (data.ndim - 1)
        times = times.view(time_shape)
        return (1.0 - times) * noise + times * data

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        integration_steps: int | None = None,
    ) -> torch.Tensor:
        """通过积分学习到的速度场生成 Mel 频谱。

        采样从 ``t=0`` 的高斯噪声开始，重复执行
        ``x <- x + dt * v_theta(x, t, h)`` 直到 ``t=1``。这里故意保留多步求解器作为普通
        Flow Matching 基线，后续再单独实现一步 Mean Flow。

        :param condition: 决定内容和说话人身份的统一条件 ``h``。
        :param target_mask: 生成结果的有效帧掩码。
        :param integration_steps: 可选的 Euler 更新步数。
        :return: 生成的 log-Mel 张量，形状为 ``[batch, n_mels, frames]``。
        """
        steps = integration_steps or self.integration_steps
        if steps <= 0:
            raise ValueError("integration_steps must be positive")

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
            velocity = self.forward(
                noisy_mels=samples,
                times=times,
                condition=condition,
                target_mask=target_mask,
            )
            samples = samples + step_size * velocity
        if target_mask is not None:
            samples = samples * target_mask.unsqueeze(1).to(samples.dtype)
        return samples


if __name__ == "__main__":
    _ = ConditionalFlowMatchingTransformer()
