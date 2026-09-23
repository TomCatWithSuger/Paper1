"""B3 MeanVoiceFlow 的 Lightning 训练模块。"""

# ====================
# 1. 导入
# ====================

from collections.abc import Callable, Mapping
from math import cos, pi
from typing import Any, Protocol, cast

import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import (
    LRSchedulerTypeUnion,
    OptimizerLRScheduler,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from src.models.voice_flow_module_base import VoiceFlowLitModuleBase


# ====================
# 2. 定义
# ====================


class MeanVoiceFlowNetwork(Protocol):
    """MeanVoiceFlow 主干需要满足的结构接口。"""

    def mean_voice_flow(
        self,
        noisy_mels: torch.Tensor,
        start_times: torch.Tensor,
        end_times: torch.Tensor,
        source_diffusion_times: torch.Tensor,
        condition: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    def interpolate(
        self,
        data: torch.Tensor,
        prior: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor: ...

    def sample(
        self,
        condition: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        integration_steps: int | None = None,
        *,
        source_mels: torch.Tensor | None = None,
        source_diffusion_time: float = 0.95,
    ) -> torch.Tensor: ...


# ====================
# 3. 核心模块
# ====================


class MeanVoiceFlowLitModule(VoiceFlowLitModuleBase):
    """实现 MeanFlow、零输入约束和条件扩散输入训练。"""

    def __init__(
        self,
        net: torch.nn.Module,
        condition_encoder: torch.nn.Module,
        optimizer: Callable[..., Optimizer],
        scheduler: Callable[..., LRSchedulerTypeUnion] | None,
        equal_time_probability: float = 0.75,
        conditional_input_probability: float = 0.5,
        zero_reconstruction_weight: float = 1.0,
        zero_reconstruction_margin: float = 0.3,
        adaptive_loss_epsilon: float = 1e-3,
        inference_source_diffusion_time: float = 0.95,
        warmup_steps: int = 10_000,
        compile: bool = False,
    ) -> None:
        """初始化 MeanVoiceFlow 的训练策略参数。"""
        if not 0.0 <= equal_time_probability <= 1.0:
            raise ValueError("equal_time_probability 必须位于 [0, 1]")
        if not 0.0 <= conditional_input_probability <= 1.0:
            raise ValueError("conditional_input_probability 必须位于 [0, 1]")
        if zero_reconstruction_weight < 0.0:
            raise ValueError("zero_reconstruction_weight 不能为负数")
        if not 0.0 <= zero_reconstruction_margin <= 1.0:
            raise ValueError("zero_reconstruction_margin 必须位于 [0, 1]")
        if adaptive_loss_epsilon <= 0.0:
            raise ValueError("adaptive_loss_epsilon 必须为正数")
        if not 0.0 <= inference_source_diffusion_time <= 1.0:
            raise ValueError("inference_source_diffusion_time 必须位于 [0, 1]")
        if warmup_steps < 0:
            raise ValueError("warmup_steps 不能为负数")
        super().__init__(
            net=net,
            condition_encoder=condition_encoder,
            optimizer=optimizer,
            scheduler=scheduler,
            compile=compile,
        )
        self.equal_time_probability = equal_time_probability
        self.conditional_input_probability = conditional_input_probability
        self.zero_reconstruction_weight = zero_reconstruction_weight
        self.zero_reconstruction_margin = zero_reconstruction_margin
        self.adaptive_loss_epsilon = adaptive_loss_epsilon
        self.inference_source_diffusion_time = inference_source_diffusion_time
        self.warmup_steps = warmup_steps

    @property
    def mean_voice_flow_net(self) -> MeanVoiceFlowNetwork:
        """返回具有完整 MeanVoiceFlow 接口的网络视图。"""
        return cast(MeanVoiceFlowNetwork, cast(object, self.net))

    # ====================
    # 3.1 时间与先验采样
    # ====================

    @staticmethod
    def _sample_logit_normal(
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """通过标准正态变量的 sigmoid 采样时间。"""
        return torch.sigmoid(torch.randn(batch_size, device=device, dtype=dtype))

    def _sample_time_intervals(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """使用 logit-normal 分布采样 MeanFlow 时间区间。"""
        first = self._sample_logit_normal(batch_size, device, dtype)
        second = self._sample_logit_normal(batch_size, device, dtype)
        start_times = torch.minimum(first, second)
        end_times = torch.maximum(first, second)
        equal_time_mask = (
            torch.rand(batch_size, device=device, dtype=dtype) < self.equal_time_probability
        )
        return torch.where(equal_time_mask, end_times, start_times), end_times

    @staticmethod
    def _shuffle_speaker_features(speaker_features: torch.Tensor) -> torch.Tensor:
        """通过随机循环位移构造不同样本的伪源说话人条件。"""
        batch_size = speaker_features.size(0)
        if batch_size < 2:
            return speaker_features
        shift = int(torch.randint(1, batch_size, (), device=speaker_features.device).item())
        return speaker_features.roll(shifts=shift, dims=0)

    def _build_training_prior(
        self,
        noise: torch.Tensor,
        content_features: torch.Tensor,
        content_lengths: torch.Tensor,
        speaker_features: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """混合纯噪声和停止梯度的伪源扩散表示。"""
        batch_size = noise.size(0)
        pure_noise_times = torch.ones(batch_size, device=noise.device, dtype=noise.dtype)
        if batch_size < 2 or self.conditional_input_probability == 0.0:
            return noise, pure_noise_times, torch.zeros_like(pure_noise_times, dtype=torch.bool)

        source_diffusion_times = self._sample_logit_normal(
            batch_size=batch_size,
            device=noise.device,
            dtype=noise.dtype,
        )
        source_condition = self.encode_condition(
            content_features=content_features,
            content_lengths=content_lengths,
            speaker_features=self._shuffle_speaker_features(speaker_features),
            target_mask=target_mask,
        )
        with torch.no_grad():
            pseudo_source_velocity = self.mean_voice_flow_net.mean_voice_flow(
                noisy_mels=noise,
                start_times=source_diffusion_times,
                end_times=pure_noise_times,
                source_diffusion_times=pure_noise_times,
                condition=source_condition,
                target_mask=target_mask,
            )
            interval_shape = (batch_size,) + (1,) * (noise.ndim - 1)
            pseudo_source = noise - (
                (1.0 - source_diffusion_times).view(interval_shape) * pseudo_source_velocity
            )

        conditional_mask = (
            torch.rand(batch_size, device=noise.device, dtype=noise.dtype)
            < self.conditional_input_probability
        )
        mask_shape = (batch_size,) + (1,) * (noise.ndim - 1)
        training_prior = torch.where(conditional_mask.view(mask_shape), pseudo_source, noise)
        effective_diffusion_times = torch.where(
            conditional_mask,
            source_diffusion_times,
            pure_noise_times,
        )
        return training_prior.detach(), effective_diffusion_times, conditional_mask

    # ====================
    # 3.2 损失函数
    # ====================

    def _adaptive_mean_flow_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """计算停止梯度归一化的自适应 MeanFlow 距离。"""
        valid = mask.unsqueeze(1).to(prediction.dtype)
        error_energy = ((prediction - target).square() * valid).sum(dim=(1, 2))
        denominator = (error_energy + self.adaptive_loss_epsilon).detach()
        return (error_energy / denominator).mean()

    def _masked_ssim_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """计算 Mel 频谱上的局部结构相似度损失。"""
        valid = mask.unsqueeze(1).expand_as(target)
        positive_infinity = torch.full_like(target, torch.inf)
        negative_infinity = torch.full_like(target, -torch.inf)
        target_min = torch.where(valid, target, positive_infinity).amin(
            dim=(1, 2), keepdim=True
        )
        target_max = torch.where(valid, target, negative_infinity).amax(
            dim=(1, 2), keepdim=True
        )
        target_range = (target_max - target_min).clamp_min(1e-6)
        normalized_prediction = (prediction - target_min) / target_range
        normalized_target = (target - target_min) / target_range

        prediction_image = normalized_prediction.unsqueeze(1)
        target_image = normalized_target.unsqueeze(1)
        valid_image = valid.unsqueeze(1).to(prediction.dtype)
        local_weight = F.avg_pool2d(valid_image, kernel_size=3, stride=1, padding=1)
        safe_weight = local_weight.clamp_min(1e-6)

        prediction_mean = (
            F.avg_pool2d(
                prediction_image * valid_image,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            / safe_weight
        )
        target_mean = (
            F.avg_pool2d(
                target_image * valid_image,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            / safe_weight
        )
        prediction_second_moment = (
            F.avg_pool2d(
                prediction_image.square() * valid_image,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            / safe_weight
        )
        target_second_moment = (
            F.avg_pool2d(
                target_image.square() * valid_image,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            / safe_weight
        )
        cross_moment = (
            F.avg_pool2d(
                prediction_image * target_image * valid_image,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            / safe_weight
        )

        prediction_variance = (prediction_second_moment - prediction_mean.square()).clamp_min(0)
        target_variance = (target_second_moment - target_mean.square()).clamp_min(0)
        covariance = cross_moment - prediction_mean * target_mean
        luminance = 2 * prediction_mean * target_mean + 0.01**2
        contrast = 2 * covariance + 0.03**2
        normalization = (
            (prediction_mean.square() + target_mean.square() + 0.01**2)
            * (prediction_variance + target_variance + 0.03**2)
        ).clamp_min(1e-6)
        ssim_map = luminance * contrast / normalization
        valid_windows = (local_weight > 0).to(prediction.dtype)
        per_sample_ssim = (ssim_map * valid_windows).sum(dim=(1, 2, 3)) / valid_windows.sum(
            dim=(1, 2, 3)
        ).clamp_min(1.0)
        return (1.0 - per_sample_ssim).clamp_min(self.zero_reconstruction_margin).mean()

    # ====================
    # 3.3 核心流程
    # ====================

    def model_step(
        self, batch: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算 MeanFlow、零输入约束和条件扩散输入训练目标。"""
        target_mels, target_mask, content_features, content_lengths, speaker_features = (
            self._training_data(batch)
        )
        target_condition = self.encode_condition(
            content_features=content_features,
            content_lengths=content_lengths,
            speaker_features=speaker_features,
            target_mask=target_mask,
        )
        noise = torch.randn_like(target_mels)
        training_prior, source_diffusion_times, _ = self._build_training_prior(
            noise=noise,
            content_features=content_features,
            content_lengths=content_lengths,
            speaker_features=speaker_features,
            target_mask=target_mask,
        )
        start_times, end_times = self._sample_time_intervals(
            batch_size=target_mels.size(0),
            device=target_mels.device,
            dtype=target_mels.dtype,
        )
        path_samples = self.mean_voice_flow_net.interpolate(
            target_mels,
            training_prior,
            end_times,
        )
        instantaneous_velocity = training_prior - target_mels

        def average_velocity(
            states: torch.Tensor,
            times: torch.Tensor,
            interval_starts: torch.Tensor,
        ) -> torch.Tensor:
            return self.mean_voice_flow_net.mean_voice_flow(
                noisy_mels=states,
                start_times=interval_starts,
                end_times=times,
                source_diffusion_times=source_diffusion_times,
                condition=target_condition,
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
        target_velocity = (
            instantaneous_velocity
            - (end_times - start_times).view(interval_shape) * total_derivative
        ).detach()
        mean_flow_loss = self._adaptive_mean_flow_loss(
            predicted_velocity,
            target_velocity,
            target_mask,
        )

        batch_size = target_mels.size(0)
        zero_times = torch.zeros(batch_size, device=target_mels.device, dtype=target_mels.dtype)
        one_times = torch.ones_like(zero_times)
        zero_velocity = self.mean_voice_flow_net.mean_voice_flow(
            noisy_mels=torch.zeros_like(target_mels),
            start_times=zero_times,
            end_times=one_times,
            source_diffusion_times=one_times,
            condition=target_condition,
            target_mask=target_mask,
        )
        zero_reconstruction = -zero_velocity
        zero_reconstruction_loss = self._masked_ssim_loss(
            zero_reconstruction,
            target_mels,
            target_mask,
        )
        loss = mean_flow_loss + self.zero_reconstruction_weight * zero_reconstruction_loss
        return loss, predicted_velocity, target_velocity, path_samples

    def predict_step(
        self,
        batch: Mapping[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        """使用源 Mel、源内容和参考说话人条件执行一步转换。"""
        source_mels = self._tensor(batch, "source_mel_spectrograms", "mel_spectrograms")
        source_mask, content_features, content_lengths, speaker_features = self._inference_data(
            batch
        )
        condition = self.encode_condition(
            content_features=content_features,
            content_lengths=content_lengths,
            speaker_features=speaker_features,
            target_mask=source_mask,
        )
        return self.mean_voice_flow_net.sample(
            condition=condition,
            target_mask=source_mask,
            source_mels=source_mels,
            source_diffusion_time=self.inference_source_diffusion_time,
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """配置 Adam、线性预热和余弦衰减。"""
        optimizer = self.optimizer_factory(params=self.parameters())
        total_steps = max(int(self.trainer.estimated_stepping_batches), self.warmup_steps + 1)

        def learning_rate_multiplier(step: int) -> float:
            if self.warmup_steps > 0 and step < self.warmup_steps:
                return (step + 1) / self.warmup_steps
            progress = (step - self.warmup_steps) / max(total_steps - self.warmup_steps, 1)
            return 0.5 * (1.0 + cos(pi * min(max(progress, 0.0), 1.0)))

        scheduler = LambdaLR(optimizer, lr_lambda=learning_rate_multiplier)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


# ====================
# 4. 入口
# ====================

if __name__ == "__main__":
    pass
