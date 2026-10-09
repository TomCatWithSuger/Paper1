"""保留 B3 按优化器步更新的线性预热和余弦衰减。"""

from math import cos, pi

from torch.optim.lr_scheduler import LambdaLR


class WarmupCosineLR(LambdaLR):
    """按优化器步执行线性预热和余弦衰减。"""

    def __init__(self, optimizer, total_steps: int, warmup_steps: int = 10000):
        if warmup_steps < 0:
            raise ValueError("warmup_steps 不能为负数")
        self.warmup_steps = warmup_steps
        self.total_steps = max(int(total_steps), warmup_steps + 1)
        super().__init__(optimizer, self._multiplier)

    def _multiplier(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        return 0.5 * (1.0 + cos(pi * min(max(progress, 0.0), 1.0)))
