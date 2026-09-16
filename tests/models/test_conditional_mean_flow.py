import torch
from torch import nn
from torch.nn.utils import parametrize

from src.models.components.conditional_mean_flow_unet import ConditionalMeanFlowUNet
from src.models.components.meanvc_conditioning import CachedConditionEncoder
from src.models.conditional_mean_flow_module import ConditionalMeanFlowLitModule


def _network() -> ConditionalMeanFlowUNet:
    """创建用于测试的小型 MeanFlow U-Net。"""
    return ConditionalMeanFlowUNet(
        n_mels=8,
        condition_dim=8,
        hidden_channels=16,
        time_embedding_dim=8,
        kernel_size=3,
    )


def _module(equal_time_probability: float = 0.75) -> ConditionalMeanFlowLitModule:
    """创建用于测试的完整 B2 模块。"""
    return ConditionalMeanFlowLitModule(
        net=_network(),
        condition_encoder=CachedConditionEncoder(
            content_dim=8,
            speaker_dim=8,
            output_dim=8,
        ),
        optimizer=lambda params: torch.optim.Adam(params, lr=1e-3),
        scheduler=None,
        equal_time_probability=equal_time_probability,
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
    states = torch.randn(2, 8, 13)
    condition = torch.randn(2, 8, 13)
    target_mask = torch.ones(2, 13, dtype=torch.bool)
    target_mask[1, 10:] = False

    velocity = net.mean_flow(
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


def test_conditional_mean_flow_time_intervals_are_ordered() -> None:
    """测试训练时间满足 ``0 <= r <= t <= 1``。"""
    start_times, end_times = _module()._sample_time_intervals(
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
    start_times, end_times = _module()._sample_time_intervals(
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert batch_size == fixed_times.size(0)
        times = fixed_times.to(device=device, dtype=dtype)
        return times, times

    monkeypatch.setattr(module, "_sample_time_intervals", fixed_intervals)

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


def test_conditional_mean_flow_one_step_sample(monkeypatch) -> None:
    """测试推理仅按 ``x_0=x_1-u(x_1,0,1,h)`` 更新一次。"""
    net = _network()
    condition = torch.randn(2, 8, 9)
    target_mask = torch.ones(2, 9, dtype=torch.bool)
    calls = 0

    def constant_velocity(**kwargs) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return torch.ones_like(kwargs["noisy_mels"])

    monkeypatch.setattr(net, "mean_flow", constant_velocity)
    torch.manual_seed(123)
    initial_noise = torch.randn(2, 8, 9)
    torch.manual_seed(123)
    generated = net.sample(condition=condition, target_mask=target_mask)

    assert calls == 1
    assert torch.allclose(generated, initial_noise - 1.0)
