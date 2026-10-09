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


def test_conditional_mean_flow_shapes_and_structure() -> None:
    """测试双时间接口和保持不变的 12 层卷积主干。"""
    net = _network()
    method = ConditionalMeanFlow(net, CachedConditionEncoder(8, 8, 8))
    states = torch.randn(2, 8, 13)
    condition = torch.randn(2, 8, 13)
    target_mask = torch.ones(2, 13, dtype=torch.bool)
    target_mask[1, 10:] = False

    velocity = method(
        noisy_mels=states,
        start_times=torch.tensor([0.1, 0.4]),
        end_times=torch.tensor([0.5, 0.9]),
        condition=condition,
        target_mask=target_mask,
    )
    convolutions = [
        module for module in net.modules() if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d))
    ]

    assert velocity.shape == states.shape
    assert torch.count_nonzero(velocity[1, :, 10:]) == 0
    assert len(convolutions) == 12
    assert all(parametrize.is_parametrized(module, "weight") for module in convolutions)


def test_flow_matching_unets_share_only_the_backbone() -> None:
    """测试三个实验网络是共享中立主干的平级实现。"""
    assert issubclass(UNet, ConditionalUNetBackbone)
    for fields in (("time",), ("time", "interval"), ("time", "interval", "source_time")):
        network = UNet(8, 8, 16, 8, time_fields=fields)
        assert type(network) is UNet
        assert not hasattr(network, "sample")
        assert not hasattr(network, "interpolate")
        assert network.time_fields == fields


def test_voice_flow_lightning_modules_are_independent() -> None:
    """测试 B1、B2、B3 共享训练入口但使用平级方法。"""
    assert issubclass(VoiceFlowModule, LightningModule)
    assert all(
        issubclass(method, VoiceMethod)
        for method in (
            ConditionalFlowMatching,
            ConditionalMeanFlow,
            MeanVoiceFlow,
        )
    )
    assert not issubclass(ConditionalMeanFlow, ConditionalFlowMatching)
    assert not issubclass(MeanVoiceFlow, ConditionalMeanFlow)
    assert not issubclass(MeanVoiceFlow, ConditionalFlowMatching)


def test_conditional_mean_flow_time_intervals_are_ordered() -> None:
    """测试训练时间满足 ``0 <= r <= t <= 1``。"""
    start_times, end_times = _module().method._sample_time_intervals(
        batch_size=128,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert torch.all(start_times >= 0)
    assert torch.all(start_times <= end_times)
    assert torch.all(end_times <= 1)


def test_conditional_mean_flow_equal_time_ratio() -> None:
    """测试约 75% 样本使用 ``r=t``，其余样本使用 ``r<t``。"""
    torch.manual_seed(123)
    start_times, end_times = _module().method._sample_time_intervals(
        batch_size=20_000,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    equal_time_ratio = (start_times == end_times).float().mean()

    assert torch.isclose(equal_time_ratio, torch.tensor(0.75), atol=0.02)
    assert torch.all(start_times[start_times != end_times] < end_times[start_times != end_times])


def test_conditional_mean_flow_equal_endpoints_reduce_to_velocity(monkeypatch) -> None:
    """测试 ``r=t`` 时 MeanFlow 目标退化为瞬时速度。"""
    module = _module()
    batch = _batch()
    target_mels = batch["target_mel_spectrograms"]
    fixed_times = torch.full((target_mels.size(0),), 0.5)

    def fixed_intervals(
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert batch_size == fixed_times.size(0)
        assert generator is None
        times = fixed_times.to(device=device, dtype=dtype)
        return times, times

    monkeypatch.setattr(module.method, "_sample_time_intervals", fixed_intervals)

    _, _, target_velocity, path_samples = module.model_step(batch)
    expected_velocity = 2.0 * (path_samples - target_mels)

    assert torch.allclose(target_velocity, expected_velocity, rtol=1e-5, atol=1e-6)


def test_conditional_mean_flow_model_step_backward() -> None:
    """测试 JVP 目标可计算且预测分支可正常反向传播。"""
    module = _module()

    loss, predicted_velocity, target_velocity, path_samples = module.model_step(_batch())
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert predicted_velocity.shape == target_velocity.shape == path_samples.shape
    assert any(parameter.grad is not None for parameter in module.parameters())
    assert all(
        torch.all(torch.isfinite(parameter.grad))
        for parameter in module.parameters()
        if parameter.grad is not None
    )


def test_conditional_mean_flow_fixed_evaluation_sampling() -> None:
    """测试验证采样不受全局随机状态影响。"""
    module = _module()
    module.eval()
    batch = _batch()

    first_loss = module._evaluation_loss(
        batch=batch,
        batch_idx=2,
        sampling_count=module.validation_sampling_count,
        seed_offset=0,
    )
    torch.manual_seed(999)
    _ = torch.randn(100)
    second_loss = module._evaluation_loss(
        batch=batch,
        batch_idx=2,
        sampling_count=module.validation_sampling_count,
        seed_offset=0,
    )

    assert module.validation_sampling_count == 4
    assert module.test_sampling_count == 8
    assert torch.equal(first_loss, second_loss)


def test_conditional_mean_flow_one_step_sample(monkeypatch) -> None:
    """测试推理仅按 ``x_0=x_1-u(x_1,0,1,h)`` 更新一次。"""
    method = _module().method
    condition = torch.randn(2, 8, 9)
    target_mask = torch.ones(2, 9, dtype=torch.bool)
    calls = 0

    def constant_velocity(noisy_mels, *args, **kwargs) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return torch.ones_like(noisy_mels)

    monkeypatch.setattr(method, "forward", constant_velocity)
    torch.manual_seed(123)
    initial_noise = torch.randn(2, 8, 9)
    torch.manual_seed(123)
    generated = method.sample(condition=condition, target_mask=target_mask)

    assert calls == 1
    assert torch.allclose(generated, initial_noise - 1.0)
