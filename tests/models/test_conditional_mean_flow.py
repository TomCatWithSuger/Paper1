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


def test_conditional_mean_flow_jvp_target_includes_nonzero_interval() -> None:
    """验证非零时间区间的 JVP 修正项包含时间字段和状态导数。"""

    class ZeroConditionEncoder(nn.Module):
        def forward(self, content_features, content_lengths, speaker_features, target_mask):
            return torch.zeros(
                target_mask.size(0), 1, target_mask.size(1), dtype=target_mask.dtype
            )

    class AnalyticNetwork(nn.Module):
        n_mels = 1

        def forward(self, states, times, condition, target_mask):
            time = times["time"].view(-1, 1, 1)
            interval = times["interval"].view(-1, 1, 1)
            return 2.0 * states + 3.0 * time + 5.0 * interval

    method = ConditionalMeanFlow(AnalyticNetwork(), ZeroConditionEncoder(), 0.0)
    target_mels = torch.tensor([[[1.0, 2.0]]])
    target_mask = torch.ones(1, 2, dtype=torch.bool)

    def fixed_intervals(batch_size, device, dtype, generator=None):
        assert batch_size == 1
        start = torch.tensor([0.2], device=device, dtype=dtype)
        end = torch.tensor([0.8], device=device, dtype=dtype)
        return start, end

    method._sample_time_intervals = fixed_intervals
    _, _, target_velocity, path_samples = method.compute_loss(
        target_mels,
        target_mask,
        torch.zeros(1, 1, 1),
        torch.ones(1, dtype=torch.long),
        torch.zeros(1, 1),
    )
    interval = 0.6
    noise = (path_samples - (1.0 - 0.8) * target_mels) / 0.8
    instantaneous_velocity = noise - target_mels
    expected_velocity = instantaneous_velocity - interval * (2.0 * instantaneous_velocity + 8.0)

    assert torch.allclose(target_velocity, expected_velocity, atol=1e-5)


def test_conditional_mean_flow_adaptive_loss_is_optional_and_masked() -> None:
    """验证加权可关闭，且只使用有效帧计算样本误差。"""
    prediction = torch.tensor([[[0.0, 0.0, 99.0]], [[2.0, 2.0, 2.0]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    target_mask = torch.tensor([[True, True, False], [True, True, True]])
    raw_method = _module().method
    adaptive_method = ConditionalMeanFlow(
        _network(), CachedConditionEncoder(8, 8, 8), adaptive_loss_power=1.0
    )

    raw_loss = raw_method._mean_flow_loss(prediction, target, target_mask)
    expected_raw = torch.tensor(12.0 / 5.0)
    assert torch.allclose(raw_loss, expected_raw)

    adaptive_loss = adaptive_method._mean_flow_loss(prediction, target, target_mask)
    adaptive_loss.backward()
    assert adaptive_loss < raw_loss
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[0, 0, 2] == 0


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


def test_conditional_mean_flow_diagnostics_are_unweighted_masked_means() -> None:
    """验证诊断返回未加权掩码均值，且 raw_meanflow_mse 等于预测与目标的掩码 MSE。"""
    module = _module()
    module.eval()
    batch = _batch()
    target_mask = batch["target_mel_attention_mask"]

    result = module.model_step(batch, return_diagnostics=True)
    assert len(result) == 5
    _, predicted_velocity, target_velocity, _, diagnostics = result
    assert set(diagnostics) == {
        "raw_meanflow_mse",
        "velocity_mse",
        "nonzero_interval_mse",
        "jvp_correction_rms",
    }
    valid = target_mask.unsqueeze(1).to(dtype=predicted_velocity.dtype)
    elements = valid.sum() * predicted_velocity.size(1)
    expected_raw = ((predicted_velocity - target_velocity).square() * valid).sum() / elements
    assert torch.allclose(diagnostics["raw_meanflow_mse"], expected_raw)
    assert torch.isfinite(diagnostics["raw_meanflow_mse"])
    assert torch.isfinite(diagnostics["velocity_mse"])
    assert diagnostics["jvp_correction_rms"] >= 0


def test_conditional_mean_flow_diagnostics_empty_nonzero_group_is_nan(monkeypatch) -> None:
    """验证非零区间组为空时 nonzero_interval_mse 记为 NaN 而非 0。"""
    module = _module()
    module.eval()
    batch = _batch()
    target_mels = batch["target_mel_spectrograms"]
    fixed = torch.full((target_mels.size(0),), 0.5)

    def equal_intervals(batch_size, device, dtype, generator=None):
        times = fixed.to(device=device, dtype=dtype)
        return times, times

    monkeypatch.setattr(module.method, "_sample_time_intervals", equal_intervals)
    _, _, _, _, diagnostics = module.model_step(batch, return_diagnostics=True)

    assert torch.isnan(diagnostics["nonzero_interval_mse"])
    assert diagnostics["jvp_correction_rms"] == 0
    assert torch.isfinite(diagnostics["raw_meanflow_mse"])


def test_conditional_mean_flow_diagnostics_nonzero_interval_when_present(monkeypatch) -> None:
    """验证存在非零区间时 nonzero_interval_mse 有限且等于掩码 MSE。"""
    module = _module()
    module.eval()
    batch = _batch()
    target_mels = batch["target_mel_spectrograms"]
    target_mask = batch["target_mel_attention_mask"]

    def nonzero_intervals(batch_size, device, dtype, generator=None):
        start = torch.full((batch_size,), 0.1, device=device, dtype=dtype)
        end = torch.full((batch_size,), 0.9, device=device, dtype=dtype)
        return start, end

    monkeypatch.setattr(module.method, "_sample_time_intervals", nonzero_intervals)
    _, predicted_velocity, target_velocity, _, diagnostics = module.model_step(
        batch, return_diagnostics=True
    )
    valid = target_mask.unsqueeze(1).to(dtype=predicted_velocity.dtype)
    elements = valid.sum() * predicted_velocity.size(1)
    expected = ((predicted_velocity - target_velocity).square() * valid).sum() / elements

    assert torch.isfinite(diagnostics["nonzero_interval_mse"])
    assert torch.allclose(diagnostics["nonzero_interval_mse"], expected)
    assert diagnostics["jvp_correction_rms"] > 0


def test_conditional_mean_flow_diagnostics_fixed_sampling_is_deterministic() -> None:
    """验证固定采样下诊断均值不受全局随机状态影响。"""
    module = _module(equal_time_probability=0.0)
    module.eval()
    batch = _batch()

    loss_a, diag_a = module._objectives(batch, 2, 4, 0)
    torch.manual_seed(777)
    _ = torch.randn(50)
    loss_b, diag_b = module._objectives(batch, 2, 4, 0)

    assert torch.equal(loss_a, loss_b)
    assert set(diag_a) == set(diag_b) == {
        "raw_meanflow_mse",
        "velocity_mse",
        "nonzero_interval_mse",
        "jvp_correction_rms",
    }
    for key in diag_a:
        assert torch.equal(diag_a[key], diag_b[key])


def test_reject_inference_mode_jvp_guard() -> None:
    """验证守卫函数仅在推理模式下拒绝前向自动微分。"""
    from src.models.methods.base import reject_inference_mode_jvp

    reject_inference_mode_jvp()
    with torch.inference_mode():
        with pytest.raises(RuntimeError, match="inference_mode"):
            reject_inference_mode_jvp()


def test_conditional_mean_flow_compute_loss_rejects_inference_mode() -> None:
    """验证 B2 compute_loss 在推理模式下抛出而非静默退化。"""
    method = _module().method
    batch = _batch()
    with torch.inference_mode():
        with pytest.raises(RuntimeError, match="inference_mode"):
            method.compute_loss(
                batch["target_mel_spectrograms"],
                batch["target_mel_attention_mask"],
                batch["target_content_features"],
                batch["target_content_feature_lengths"],
                batch["target_speaker_features"],
            )


def test_meanvoiceflow_compute_loss_rejects_inference_mode() -> None:
    """验证 B3 compute_loss 同样受推理模式守卫保护。"""
    method = MeanVoiceFlow(
        _network(),
        CachedConditionEncoder(content_dim=8, speaker_dim=8, output_dim=8),
    )
    batch = _batch()
    with torch.inference_mode():
        with pytest.raises(RuntimeError, match="inference_mode"):
            method.compute_loss(
                batch["target_mel_spectrograms"],
                batch["target_mel_attention_mask"],
                batch["target_content_features"],
                batch["target_content_feature_lengths"],
                batch["target_speaker_features"],
            )


def test_conditional_mean_flow_no_grad_matches_normal() -> None:
    """验证 no_grad 不破坏 JVP（仅 inference_mode 有害）。"""
    module = _module()
    batch = _batch()

    torch.manual_seed(0)
    loss_normal, _, target_normal, _ = module.model_step(batch)
    torch.manual_seed(0)
    with torch.no_grad():
        loss_nograd, _, target_nograd, _ = module.model_step(batch)

    assert torch.allclose(loss_normal, loss_nograd)
    assert torch.allclose(target_normal, target_nograd)


def test_validation_logs_diagnostics_under_trainer() -> None:
    """验证 Trainer.validate 入口（inference_mode=False）记录全部 B2 诊断指标。"""
    from lightning import Trainer
    from torch.utils.data import DataLoader

    module = _module(equal_time_probability=0.0, validation_sampling_count=2)
    loader = DataLoader([_batch()], batch_size=None)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        inference_mode=False,
    )
    trainer.validate(module, dataloaders=loader)

    metrics = trainer.callback_metrics
    for key in (
        "val/loss",
        "val/raw_meanflow_mse",
        "val/velocity_mse",
        "val/nonzero_interval_mse",
        "val/jvp_correction_rms",
        "val/one_step_reconstruction_l1",
    ):
        assert key in metrics, f"缺少诊断指标 {key}"


def test_mean_flow_loss_zero_power_matches_masked_mse() -> None:
    """验证 power=0 时 _mean_flow_loss 与 masked_mse 值及梯度一致。"""
    from src.losses.masked_regression import masked_mse

    method = _module().method  # adaptive_loss_power=0
    target = torch.randn(3, 8, 12)
    target_mask = torch.ones(3, 12, dtype=torch.bool)
    target_mask[1, 10:] = False
    target_mask[2, 8:] = False
    p_custom = torch.randn(3, 8, 12, requires_grad=True)
    p_ref = p_custom.detach().clone().requires_grad_(True)

    loss_custom = method._mean_flow_loss(p_custom, target, target_mask)
    loss_ref = masked_mse(p_ref, target, target_mask)
    loss_custom.backward()
    loss_ref.backward()

    assert torch.allclose(loss_custom, loss_ref)
    assert torch.allclose(p_custom.grad, p_ref.grad)


def test_mean_flow_loss_weights_are_detached() -> None:
    """验证自适应权重对预测停止梯度（无额外权重梯度项）。"""
    method = ConditionalMeanFlow(
        _network(),
        CachedConditionEncoder(8, 8, 8),
        adaptive_loss_power=1.0,
        adaptive_loss_epsilon=1e-3,
    )
    prediction = torch.randn(2, 4, 5, requires_grad=True)
    target = torch.randn(2, 4, 5)
    target_mask = torch.ones(2, 5, dtype=torch.bool)
    target_mask[1, 3:] = False

    loss = method._mean_flow_loss(prediction, target, target_mask)

    valid = target_mask.unsqueeze(1).to(dtype=prediction.dtype)
    squared_error = (prediction - target).square() * valid
    sample_counts = valid.sum(dim=(1, 2)) * prediction.size(1)
    sample_mse = squared_error.sum(dim=(1, 2)) / sample_counts.clamp_min(1.0)
    with torch.no_grad():
        weights = (sample_mse + 1e-3).pow(-1.0)
    expected = (weights * sample_mse * sample_counts).sum() / sample_counts.sum().clamp_min(1.0)

    assert torch.allclose(loss, expected)
    grad = torch.autograd.grad(loss, prediction, retain_graph=True)[0]
    grad_expected = torch.autograd.grad(expected, prediction)[0]
    assert torch.allclose(grad, grad_expected)


def test_mean_flow_targets_are_stop_gradient() -> None:
    """验证 JVP 目标为停止梯度，仅预测分支接收梯度。"""
    module = _module()
    loss, predicted_velocity, target_velocity, _ = module.model_step(_batch())

    assert predicted_velocity.requires_grad is True
    assert target_velocity.requires_grad is False

    loss.backward()
    assert torch.isfinite(loss)


def test_mean_flow_loss_normalizes_by_valid_length() -> None:
    """验证损失按有效帧数归一化（长样本权重更大）。"""
    method = _module().method
    target = torch.zeros(2, 1, 4)
    prediction = torch.zeros(2, 1, 4)
    prediction[0, 0, :] = 1.0  # 4 个有效帧，误差 1
    prediction[1, 0, 0] = 3.0  # 1 个有效帧，误差 9
    target_mask = torch.tensor([[True, True, True, True], [True, False, False, False]])

    loss = method._mean_flow_loss(prediction, target, target_mask)

    # 总平方误差 4+9=13，总有效元素 (4+1)*1=5 → 2.6（而非按样本均值 5.0）
    assert torch.allclose(loss, torch.tensor(2.6))


def test_conditional_mean_flow_rejects_invalid_adaptive_params() -> None:
    """验证自适应损失与时间区间非法参数被拒绝。"""
    with pytest.raises(ValueError):
        ConditionalMeanFlow(_network(), CachedConditionEncoder(8, 8, 8), adaptive_loss_power=-1.0)
    with pytest.raises(ValueError):
        ConditionalMeanFlow(_network(), CachedConditionEncoder(8, 8, 8), adaptive_loss_epsilon=0.0)
    with pytest.raises(ValueError):
        ConditionalMeanFlow(_network(), CachedConditionEncoder(8, 8, 8), equal_time_probability=1.5)


def test_mean_flow_loss_empty_mask_contract() -> None:
    """验证全掩码批次损失为 0 且梯度有限为零。"""
    method = _module().method
    prediction = torch.randn(2, 4, 5, requires_grad=True)
    target = torch.randn(2, 4, 5)
    target_mask = torch.zeros(2, 5, dtype=torch.bool)

    loss = method._mean_flow_loss(prediction, target, target_mask)
    loss.backward()

    assert loss.item() == 0.0
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert torch.all(prediction.grad == 0)


def test_generate_invariant_under_fixed_seed_and_state_dict() -> None:
    """验证固定种子下生成可复现，且 state_dict 往返不改变生成。"""
    method = _module().method
    method.eval()
    content = torch.randn(2, 8, 9)
    lengths = torch.tensor([9, 9])
    speaker = torch.randn(2, 8)
    target_mask = torch.ones(2, 9, dtype=torch.bool)

    torch.manual_seed(42)
    first = method.generate(content, lengths, speaker, target_mask)
    torch.manual_seed(42)
    second = method.generate(content, lengths, speaker, target_mask)
    assert torch.allclose(first, second)

    state = {k: v.clone() for k, v in method.state_dict().items()}
    torch.manual_seed(42)
    before = method.generate(content, lengths, speaker, target_mask)
    method.load_state_dict(state)
    torch.manual_seed(42)
    after = method.generate(content, lengths, speaker, target_mask)

    assert torch.allclose(before, after)
    assert torch.allclose(before, first)
