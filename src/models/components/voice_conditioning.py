"""语音转换 Flow Matching 基线使用的结构化条件编码模块。

该模块分别提取随时间变化的语音内容和整句稳定的说话人身份，再将两者融合为逐帧条件
``h``。因此，Flow Matching 网络只依赖 ``(x_t, t, h)``，无需了解条件特征的提取方式。
"""

import torch
import torch.nn.functional as F
from torch import nn


class ContentEncoder(nn.Module):
    """在保留 Mel 帧序列的同时编码随时间变化的语音内容。

    语音内容随时间变化，因此这里使用卷积而不是时间池化。输出保持 ``[batch, output_dim, frames]``，使每个生成帧都能使用对应的源语音内容作为条件。
    """

    def __init__(self, n_mels: int = 80, hidden_dim: int = 128, output_dim: int = 128) -> None:
        """初始化逐帧内容编码器。

        :param n_mels: 输入 log-Mel 频谱的频率维数。
        :param hidden_dim: 中间卷积特征的通道数。
        :param output_dim: 逐帧内容特征的通道数。
        """
        super().__init__()
        if n_mels <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("encoder dimensions must be positive")
        self.network = nn.Sequential(
            nn.Conv1d(n_mels, hidden_dim, kernel_size=5, padding=2),
            nn.GroupNorm(1, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=3, padding=1),
        )

    def forward(self, mels: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """返回逐帧内容特征，并将填充位置置零。

        :param mels: 形状为 ``[batch, n_mels, frames]`` 的 log-Mel 频谱。
        :param mask: 可选的有效帧掩码，形状为 ``[batch, frames]``。
        :return: 形状为 ``[batch, output_dim, frames]`` 的内容特征。
        """
        if mels.ndim != 3:
            raise ValueError("mels must have shape [batch, n_mels, frames]")
        features = self.network(mels)
        if mask is not None:
            features = features * mask.unsqueeze(1).to(features.dtype)
        return features


class SpeakerEncoder(nn.Module):
    """将参考语音压缩为句子级说话人表示。

    说话人身份在一段语音内应保持稳定，因此将时间特征池化为一个全局向量，而不是保留逐帧 表示。该向量可以与另一段语音的内容特征组合，用于零样本语音转换。
    """

    def __init__(self, n_mels: int = 80, hidden_dim: int = 128, output_dim: int = 128) -> None:
        """初始化说话人编码器和时间池化后的投影层。

        :param n_mels: 输入 log-Mel 频谱的频率维数。
        :param hidden_dim: 卷积说话人特征的通道数。
        :param output_dim: 句子级说话人向量的维数。
        """
        super().__init__()
        if n_mels <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("encoder dimensions must be positive")
        self.network = nn.Sequential(
            nn.Conv1d(n_mels, hidden_dim, kernel_size=5, padding=2),
            nn.GroupNorm(1, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, mels: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """为每段语音返回一个考虑有效帧掩码的说话人向量。

        :param mels: 形状为 ``[batch, n_mels, frames]`` 的参考语音 log-Mel 频谱。
        :param mask: 可选的有效帧掩码，形状为 ``[batch, frames]``。
        :return: 形状为 ``[batch, output_dim]`` 的说话人特征。
        """
        if mels.ndim != 3:
            raise ValueError("mels must have shape [batch, n_mels, frames]")
        hidden = self.network(mels).transpose(1, 2)
        if mask is None:
            pooled = hidden.mean(dim=1)
        else:
            weights = mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.output_projection(pooled)


class ConditionEncoder(nn.Module):
    """将内容分支和说话人分支融合为统一条件 ``h``。

    内容特征保留时间变化，说话人向量则复制到每个时间位置。逐点卷积负责学习每一帧应使用 两个分支中的多少信息，同时避免再次混合相邻帧。
    """

    def __init__(
        self,
        content_encoder: nn.Module,
        speaker_encoder: nn.Module,
        content_dim: int = 128,
        speaker_dim: int = 128,
        output_dim: int = 128,
    ) -> None:
        """初始化结构化条件融合模块。

        :param content_encoder: 输出 ``[batch, content_dim, frames]`` 的内容编码器。
        :param speaker_encoder: 输出 ``[batch, speaker_dim]`` 的说话人编码器。
        :param content_dim: 内容编码器输出的通道数。
        :param speaker_dim: 说话人编码器输出的特征维数。
        :param output_dim: 统一条件 ``h`` 的通道数。
        """
        super().__init__()
        if content_dim <= 0 or speaker_dim <= 0 or output_dim <= 0:
            raise ValueError("condition dimensions must be positive")
        self.content_encoder = content_encoder
        self.speaker_encoder = speaker_encoder
        self.fusion = nn.Sequential(
            nn.Conv1d(content_dim + speaker_dim, output_dim, kernel_size=1),
            nn.GroupNorm(1, output_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        content_mels: torch.Tensor,
        speaker_mels: torch.Tensor,
        content_mask: torch.Tensor | None = None,
        speaker_mask: torch.Tensor | None = None,
        target_frames: int | None = None,
    ) -> torch.Tensor:
        """分别编码内容和说话人输入，并返回与目标帧对齐的条件 ``h``。

        基线训练时，``content_mels`` 和 ``speaker_mels`` 来自同一段干净语音。语音转换时，
        内容来自源语音，说话人身份来自另一段目标说话人参考语音。

        :param content_mels: 源内容频谱，形状为 ``[batch, n_mels, frames]``。
        :param speaker_mels: 说话人参考频谱，形状为 ``[batch, n_mels, ref_frames]``。
        :param content_mask: ``content_mels`` 的有效帧掩码。
        :param speaker_mask: ``speaker_mels`` 的有效帧掩码。
        :param target_frames: Flow Matching 状态需要的可选输出帧数。
        :return: 形状为 ``[batch, output_dim, target_frames]`` 的统一条件 ``h``。
        """
        content_features = self.content_encoder(content_mels, content_mask)
        speaker_features = self.speaker_encoder(speaker_mels, speaker_mask)
        frames = target_frames or content_features.size(2)
        if content_features.size(2) != frames:
            content_features = F.interpolate(
                content_features,
                size=frames,
                mode="linear",
                align_corners=False,
            )
        # 说话人身份是全局特征，因此每个输出帧使用同一个说话人向量。
        expanded_speaker = speaker_features.unsqueeze(2).expand(-1, -1, frames)
        condition = self.fusion(torch.cat((content_features, expanded_speaker), dim=1))
        if content_mask is not None and content_mask.size(1) == frames:
            condition = condition * content_mask.unsqueeze(1).to(condition.dtype)
        return condition


if __name__ == "__main__":
    _ = ConditionEncoder(
        content_encoder=ContentEncoder(),
        speaker_encoder=SpeakerEncoder(),
    )
