"""有效 Mel 帧上的回归损失。"""

import torch
import torch.nn.functional as F


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """计算有效帧上的通道平均平方误差。"""
    valid = mask.unsqueeze(1).to(dtype=prediction.dtype)
    squared_error = F.mse_loss(prediction, target, reduction="none") * valid
    denominator = valid.sum() * prediction.size(1)
    return squared_error.sum() / denominator.clamp_min(1.0)
