"""模块说明：提供纯 Tensor 函数的 JVP 类型边界，实际计算委托给 PyTorch。"""

# ====================
# 1. Imports
# ====================

from collections.abc import Callable
from typing import Literal, cast

import torch

# ====================
# 6. Public API
# ====================


def tensor_jvp(
    function: Callable[..., torch.Tensor],
    primals: tuple[torch.Tensor, ...],
    tangents: tuple[torch.Tensor, ...],
    *,
    has_aux: Literal[False] = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回函数值与方向导数；不接受 PyTree 或辅助返回值，不关闭梯度。"""
    if has_aux:
        raise ValueError("tensor_jvp 不支持辅助返回值")
    operation = cast(Callable[..., tuple[torch.Tensor, torch.Tensor]], getattr(torch.func, "jvp"))
    return operation(function, primals, tangents, has_aux=has_aux)
