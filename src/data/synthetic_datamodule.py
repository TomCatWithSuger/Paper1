"""模块说明：提供无需下载的合成张量批次，用于实验接入和生命周期冒烟验证。"""

# ====================
# 1. Imports
# ====================

from collections.abc import Sequence

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset

# ====================
# 6. Public API
# ====================


class SyntheticDataModule(LightningDataModule):
    """生成独立的训练、验证和测试样本；标签仅用于通用二元批次契约。"""

    def __init__(
        self,
        input_shape: Sequence[int] = (1, 8, 8),
        num_samples: Sequence[int] = (32, 8, 8),
        batch_size: int = 8,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if not input_shape or any(size <= 0 for size in input_shape):
            raise ValueError("input_shape 必须包含正整数维度")
        if len(num_samples) != 3 or any(size <= 0 for size in num_samples):
            raise ValueError("num_samples 必须包含训练、验证、测试的三个正整数")
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正整数")
        self.save_hyperparameters(logger=False)
        self.input_shape = tuple(input_shape)
        self.num_samples = tuple(num_samples)
        self.batch_size = batch_size
        self.seed = seed
        self._datasets: list[TensorDataset] = []

    def setup(self, stage: str | None = None) -> None:
        """使用分片独立的随机状态构造数据，重复调用保留已有样本。"""
        if self._datasets:
            return
        for index, count in enumerate(self.num_samples):
            generator = torch.Generator().manual_seed(self.seed + index)
            samples = torch.randn(count, *self.input_shape, generator=generator)
            labels = torch.zeros(count, dtype=torch.long)
            self._datasets.append(TensorDataset(samples, labels))

    def _loader(self, split: int, shuffle: bool = False) -> DataLoader:
        self.setup()
        return DataLoader(
            self._datasets[split],
            batch_size=self.batch_size,
            shuffle=shuffle,
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=0,
        )

    def train_dataloader(self) -> DataLoader:
        """返回仅打乱训练分片的批次。"""
        return self._loader(0, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """返回顺序固定的验证批次。"""
        return self._loader(1)

    def test_dataloader(self) -> DataLoader:
        """返回顺序固定的测试批次。"""
        return self._loader(2)

    def predict_dataloader(self) -> DataLoader:
        """使用测试分片验证预测接口，不重新产生样本。"""
        return self._loader(2)
