"""B2 条件 MeanFlow 的 Lightning 训练模块。"""

from collections.abc import Callable, Mapping
from typing import Any, Protocol, cast

import torch
from lightning.pytorch.utilities.types import LRSchedulerTypeUnion
from torch.optim import Optimizer

from src.models.conditional_flow_matching_module import ConditionalFlowMatchingLitModule


class ConditionalMeanFlowNetwork(Protocol):
    """条件 MeanFlow 主干需要满足的结构接口。"""

    def mean_flow(
        self,
        noisy_mels: torch.Tensor,
        start_times: torch.Tensor,
        end_times: torch.Tensor,
        condition: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    def interpolate(
        self,
        data: torch.Tensor,
        noise: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor: ...

    def sample(
        self,
        condition: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


class ConditionalMeanFlowLitModule(ConditionalFlowMatchingLitModule):
    """仅将 B1 的普通 Flow Matching 目标替换为 MeanFlow 目标。"""

    _TEST_SEED_OFFSET = 1_000_000_000

    def __init__(
        self,
        net: torch.nn.Module,
        condition_encoder: torch.nn.Module,
        optimizer: Callable[..., Optimizer],
        scheduler: Callable[..., LRSchedulerTypeUnion] | None,
        equal_time_probability: float = 0.75,
        validation_sampling_count: int = 4,
        test_sampling_count: int = 8,
        evaluation_seed: int = 12345,
        compile: bool = False,
    ) -> None:
        """初始化条件编码器、优化器和 MeanFlow 评估参数。"""
        if not 0.0 <= equal_time_probability <= 1.0:
            raise ValueError("equal_time_probability 必须位于 [0, 1]")
        if validation_sampling_count <= 0 or test_sampling_count <= 0:
            raise ValueError("验证和测试采样次数必须为正数")
        if evaluation_seed < 0:
            raise ValueError("evaluation_seed 不能为负数")
        super().__init__(
            net=net,
            condition_encoder=condition_encoder,
            optimizer=optimizer,
            scheduler=scheduler,
            compile=compile,
        )
        self.equal_time_probability = equal_time_probability
        self.validation_sampling_count = validation_sampling_count
        self.test_sampling_count = test_sampling_count
        self.evaluation_seed = evaluation_seed

    @property
    def mean_flow_net(self) -> ConditionalMeanFlowNetwork:
        """返回具有 MeanFlow 双时间接口的网络视图。"""
        return cast(ConditionalMeanFlowNetwork, cast(object, self.net))

    def _sample_time_intervals(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """混合采样瞬时速度和区间平均速度对应的时间端点。"""
        first = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
        second = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
        start_times = torch.minimum(first, second)
        end_times = torch.maximum(first, second)
        equal_time_mask = (
            torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
            < self.equal_time_probability
        )
        start_times = torch.where(equal_time_mask, end_times, start_times)
        return start_times, end_times

    def model_step(
        self,
        batch: Mapping[str, Any],
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """构造 MeanFlow 目标并计算有效 Mel 帧上的带掩码 MSE。"""
        target_mels, target_mask, content_features, content_lengths, speaker_features = (
            self._training_data(batch)
        )
        condition = self.encode_condition(
            content_features=content_features,
            content_lengths=content_lengths,
            speaker_features=speaker_features,
            target_mask=target_mask,
        )
        noise = torch.randn(
            target_mels.shape,
            device=target_mels.device,
            dtype=target_mels.dtype,
            generator=generator,
        )
        start_times, end_times = self._sample_time_intervals(
            batch_size=target_mels.size(0),
            device=target_mels.device,
            dtype=target_mels.dtype,
            generator=generator,
        )
        path_samples = self.mean_flow_net.interpolate(target_mels, noise, end_times)
        instantaneous_velocity = noise - target_mels

        def average_velocity(
            states: torch.Tensor,
            times: torch.Tensor,
            interval_starts: torch.Tensor,
        ) -> torch.Tensor:
            return self.mean_flow_net.mean_flow(
                noisy_mels=states,
                start_times=interval_starts,
                end_times=times,
                condition=condition,
                target_mask=target_mask,
            )

        predicted_velocity = average_velocity(path_samples, end_times, start_times)
        with torch.no_grad(), torch.autocast(device_type=target_mels.device.type, enabled=False):
            jvp_fn = cast(
                Callable[..., tuple[torch.Tensor, torch.Tensor]],
                getattr(torch.func, "jvp"),
            )
            _, total_derivative = jvp_fn(
                average_velocity,
                (path_samples, end_times, start_times),
                (
                    instantaneous_velocity,
                    torch.ones_like(end_times),
                    torch.zeros_like(start_times),
                ),
                has_aux=False,
            )

        interval_shape = (end_times.size(0),) + (1,) * (target_mels.ndim - 1)
        interval = (end_times - start_times).view(interval_shape)
        target_velocity = (instantaneous_velocity - interval * total_derivative).detach()
        loss = self._masked_mse(predicted_velocity, target_velocity, target_mask)
        return loss, predicted_velocity, target_velocity, path_samples

    def _evaluation_loss(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
        sampling_count: int,
        seed_offset: int,
    ) -> torch.Tensor:
        """对固定的多组噪声和时间区间计算平均损失。"""
        target_mels = self._tensor(batch, "target_mel_spectrograms", "mel_spectrograms")
        losses = []
        for sample_idx in range(sampling_count):
            generator = torch.Generator(device=target_mels.device)
            generator.manual_seed(
                self.evaluation_seed + seed_offset + batch_idx * sampling_count + sample_idx
            )
            loss, _, _, _ = self.model_step(batch, generator=generator)
            losses.append(loss)
        return torch.stack(losses).mean()

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> None:
        """使用四组固定采样计算稳定的验证损失。"""
        loss = self._evaluation_loss(
            batch=batch,
            batch_idx=batch_idx,
            sampling_count=self.validation_sampling_count,
            seed_offset=0,
        )
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Mapping[str, Any], batch_idx: int) -> None:
        """使用八组固定采样计算稳定的测试损失。"""
        loss = self._evaluation_loss(
            batch=batch,
            batch_idx=batch_idx,
            sampling_count=self.test_sampling_count,
            seed_offset=self._TEST_SEED_OFFSET,
        )
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

if __name__ == "__main__":
    pass
