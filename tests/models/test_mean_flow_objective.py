"""模块说明：验证 MeanFlow 自适应目标、有效帧归一化、梯度边界和生成复现。"""

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
from tests.helpers.mean_flow import _batch as _batch
from tests.helpers.mean_flow import _module as _module
from tests.helpers.mean_flow import _network as _network

# ====================
# 6. Public API
# ====================


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
    assert p_custom.grad is not None and p_ref.grad is not None
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
        ConditionalMeanFlow(
            _network(), CachedConditionEncoder(8, 8, 8), equal_time_probability=1.5
        )


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
