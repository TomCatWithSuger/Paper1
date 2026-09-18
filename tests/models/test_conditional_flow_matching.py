import torch
from torch import nn
from torch.nn.utils import parametrize

from src.models.components.conditional_flow_matching import (
    ConditionalFlowMatchingTransformer,
    ContinuousTimeEmbedding,
)
from src.models.components.conditional_flow_matching_unet import (
    ConditionalFlowMatchingUNet,
)
from src.models.components.meanvc_conditioning import CachedConditionEncoder
from src.models.conditional_flow_matching_module import ConditionalFlowMatchingLitModule


def _condition_encoder() -> CachedConditionEncoder:
    """创建离线特征条件融合器。"""
    return CachedConditionEncoder(
        content_dim=8,
        speaker_dim=8,
        output_dim=8,
    )


def _network() -> ConditionalFlowMatchingUNet:
    """创建用于测试的小型条件 Flow Matching U-Net。"""
    return ConditionalFlowMatchingUNet(
        n_mels=8,
        condition_dim=8,
        hidden_channels=16,
        time_embedding_dim=8,
        kernel_size=3,
        integration_steps=2,
    )


def _module() -> ConditionalFlowMatchingLitModule:
    """创建用于测试的完整条件 Flow Matching 基线。"""
    return ConditionalFlowMatchingLitModule(
        net=_network(),
        condition_encoder=_condition_encoder(),
        optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
        scheduler=None,
    )


def test_condition_encoder_shapes() -> None:
    """测试缓存内容和说话人条件的融合结果。"""
    encoder = _condition_encoder()
    content = torch.randn(2, 8, 10)
    speaker = torch.randn(2, 8)
    content_lengths = torch.tensor([10, 8])
    target_mask = torch.ones(2, 12, dtype=torch.bool)
    target_mask[1, 9:] = False

    condition = encoder(
        content_features=content,
        content_lengths=content_lengths,
        speaker_features=speaker,
        target_mask=target_mask,
    )

    assert condition.shape == (2, 8, 12)
    assert torch.count_nonzero(condition[1, :, 9:]) == 0


def test_continuous_time_embedding_derivative_is_bounded() -> None:
    """测试时间嵌入不会在 MeanFlow JVP 中放大时间导数。"""
    embedding = ContinuousTimeEmbedding(128)
    times = torch.tensor([0.37])

    derivative = torch.autograd.functional.jacobian(embedding, times)

    assert derivative.abs().max() <= 2 * torch.pi


def test_conditional_flow_matching_shapes() -> None:
    """测试 U-Net 的统一接口、奇数帧长度和 Euler 采样。"""
    net = _network()
    target = torch.randn(2, 8, 13)
    condition = torch.randn(2, 8, 13)
    target_mask = torch.ones(2, 13, dtype=torch.bool)
    target_mask[1, 10:] = False
    times = torch.tensor([0.25, 0.75])

    velocity = net(
        noisy_mels=target,
        times=times,
        condition=condition,
        target_mask=target_mask,
    )
    interpolated = net.interpolate(target, torch.zeros_like(target), times)
    generated = net.sample(
        condition=condition,
        target_mask=target_mask,
        integration_steps=2,
    )

    assert velocity.shape == target.shape
    assert torch.count_nonzero(velocity[1, :, 10:]) == 0
    assert interpolated.shape == target.shape
    assert generated.shape == target.shape
    assert torch.count_nonzero(generated[1, :, 10:]) == 0


def test_conditional_flow_matching_unet_structure() -> None:
    """测试 12 层卷积、两次上采样和所有卷积的权重归一化。"""
    net = _network()
    convolutions = [
        module for module in net.modules() if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d))
    ]
    transpose_convolutions = [
        module for module in convolutions if isinstance(module, nn.ConvTranspose1d)
    ]

    assert len(convolutions) == 12
    assert len(transpose_convolutions) == 2
    assert all(parametrize.is_parametrized(module, "weight") for module in convolutions)


def test_conditional_flow_matching_transformer_is_preserved() -> None:
    """测试原 Transformer 速度场仍可通过统一条件接口使用。"""
    net = ConditionalFlowMatchingTransformer(
        n_mels=8,
        hidden_dim=16,
        condition_dim=8,
        num_layers=1,
        num_heads=2,
        integration_steps=2,
    )
    noisy_mels = torch.randn(2, 8, 12)
    condition = torch.randn(2, 8, 12)

    velocity = net(noisy_mels, torch.rand(2), condition)

    assert velocity.shape == noisy_mels.shape


def test_conditional_flow_matching_model_step() -> None:
    """测试标准自重构 Flow Matching 目标。"""
    module = _module()
    target = torch.randn(3, 8, 12)
    target_mask = torch.ones(3, 12, dtype=torch.bool)
    target_mask[1, 10:] = False
    target_mask[2, 8:] = False
    batch = {
        "target_mel_spectrograms": target,
        "target_mel_attention_mask": target_mask,
        "target_content_features": torch.randn(3, 8, 10),
        "target_content_feature_lengths": torch.tensor([10, 9, 8]),
        "target_speaker_features": torch.randn(3, 8),
    }

    loss, predicted_velocity, target_velocity, path_samples = module.model_step(batch)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss >= 0
    assert predicted_velocity.shape == target.shape
    assert target_velocity.shape == target.shape
    assert path_samples.shape == target.shape


def test_conditional_flow_matching_vctk_batch_fallback() -> None:
    """测试阶段一 VCTK 字段下的训练和零样本预测。"""
    module = _module()
    mel_mask = torch.ones(2, 11, dtype=torch.bool)
    batch = {
        "mel_spectrograms": torch.randn(2, 8, 11),
        "mel_attention_mask": mel_mask,
        "content_features": torch.randn(2, 8, 9),
        "content_feature_lengths": torch.tensor([9, 8]),
        "speaker_features": torch.randn(2, 8),
        "reference_speaker_features": torch.randn(2, 8),
    }

    loss, _, _, _ = module.model_step(batch)
    generated = module.predict_step(batch, batch_idx=0)

    assert torch.isfinite(loss)
    assert generated.shape == batch["mel_spectrograms"].shape
