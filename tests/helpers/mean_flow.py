"""模块说明：提供 MeanFlow 测试的小型网络、训练模块和合成批次构造器。"""

# ====================
# 1. Imports
# ====================

import pytest
import torch
from lightning import LightningModule
from torch import nn
from torch.nn.utils import parametrize

from src.models.components.conditioning.meanvc import CachedConditionEncoder
from src.models.methods.base import VoiceMethod
from src.models.methods.generative.flow.cfm.conditional_flow_matching import (
    ConditionalFlowMatching,
)
from src.models.methods.generative.flow.mean_flow.conditional_mean_flow import (
    ConditionalMeanFlow,
)
from src.models.methods.generative.flow.mean_flow.meanvoiceflow import MeanVoiceFlow
from src.models.modules.voice_flow_module import VoiceFlowModule
from src.models.networks.unet.conditional import ConditionalUNetBackbone, UNet

# ====================
# 3. Utilities
# ====================


def _network() -> UNet:
    """创建用于测试的小型 MeanFlow U-Net。"""
    return UNet(
        n_mels=8,
        condition_dim=8,
        hidden_channels=16,
        time_embedding_dim=8,
        kernel_size=3,
        time_fields=("time", "interval"),
    )


def _module(
    equal_time_probability: float = 0.75,
    validation_sampling_count: int = 4,
    test_sampling_count: int = 8,
) -> VoiceFlowModule:
    """创建用于测试的完整 B2 模块。"""
    return VoiceFlowModule(
        method=ConditionalMeanFlow(
            _network(),
            CachedConditionEncoder(content_dim=8, speaker_dim=8, output_dim=8),
            equal_time_probability=equal_time_probability,
        ),
        optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
        scheduler=None,
        validation_sampling_count=validation_sampling_count,
        test_sampling_count=test_sampling_count,
    )


def _batch() -> dict[str, torch.Tensor]:
    """创建离线条件特征训练批次。"""
    target_mask = torch.ones(3, 12, dtype=torch.bool)
    target_mask[1, 10:] = False
    target_mask[2, 8:] = False
    return {
        "target_mel_spectrograms": torch.randn(3, 8, 12),
        "target_mel_attention_mask": target_mask,
        "target_content_features": torch.randn(3, 8, 10),
        "target_content_feature_lengths": torch.tensor([10, 9, 8]),
        "target_speaker_features": torch.randn(3, 8),
    }
