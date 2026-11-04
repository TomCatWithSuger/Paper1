"""模块说明：提供确定性的线性概率路径与可微 Euler 积分。"""

# ====================
# 1. Imports
# ====================

from collections.abc import Callable

import torch

# ====================
# 6. Public API
# 对外提供的纯计算函数
# ====================


def linear_path(data: torch.Tensor, noise: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    """按每个样本的时间在线性噪声到数据路径上插值。"""
    if data.shape != noise.shape or data.ndim < 1:
        raise ValueError("data 与 noise 必须具有相同的非标量形状")
    if times.ndim != 1 or times.size(0) != data.size(0):
        raise ValueError("times 必须是一维张量且与 batch 大小一致")
    expanded_times = times.view((times.size(0),) + (1,) * (data.ndim - 1))
    return (1.0 - expanded_times) * noise + expanded_times * data


def euler_integrate(
    field: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    initial_state: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """在 [0, 1] 上积分同形状向量场；不采样噪声，也不关闭梯度。"""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps 必须为正整数")
    if initial_state.ndim < 1:
        raise ValueError("initial_state 必须包含 batch 维度")
    samples = initial_state
    step_size = 1.0 / steps
    for step in range(steps):
        times = torch.full(
            (samples.size(0),), step / steps, device=samples.device, dtype=samples.dtype
        )
        velocity = field(samples, times)
        if velocity.shape != samples.shape:
            raise ValueError("向量场输出必须与状态形状一致")
        samples = samples + step_size * velocity
    return samples
