"""条件语音方法的组件所有权和规范输入接口。"""

import torch
from torch import nn


class VoiceMethod(nn.Module):
    """统一条件语音方法的条件编码和生成入口。"""

    requires_source_mels = False

    def __init__(self, network: nn.Module, condition_encoder: nn.Module) -> None:
        super().__init__()
        self.network = network
        self.condition_encoder = condition_encoder

    def encode_condition(self, content_features, content_lengths, speaker_features, target_mask):
        """将内容与说话人特征编码为统一条件表示。"""
        return self.condition_encoder(
            content_features=content_features,
            content_lengths=content_lengths,
            speaker_features=speaker_features,
            target_mask=target_mask,
        )

    @staticmethod
    def apply_mask(samples, target_mask):
        if target_mask is None:
            return samples
        return samples * target_mask.unsqueeze(1).to(samples.dtype)

    @torch.no_grad()
    def generate(
        self,
        content_features: torch.Tensor,
        content_lengths: torch.Tensor,
        speaker_features: torch.Tensor,
        target_mask: torch.Tensor,
        *,
        source_mels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """编码推理条件并生成应用有效帧掩码的 Mel。"""
        condition = self.encode_condition(
            content_features, content_lengths, speaker_features, target_mask
        )
        if self.requires_source_mels:
            return self.sample(condition, target_mask, source_mels=source_mels)
        return self.sample(condition, target_mask)
