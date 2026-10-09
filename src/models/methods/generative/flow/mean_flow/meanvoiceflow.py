"""MeanVoiceFlow 的伪源先验、JVP 与零输入重构。"""

from collections.abc import Callable
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from src.models.methods.base import VoiceMethod
from src.models.methods.generative.flow.mean_flow.conditional_mean_flow import (
    ConditionalMeanFlow,
)


class MeanVoiceFlow(VoiceMethod):
    """结合伪源先验与零输入重构训练一步语音转换。"""

    requires_source_mels = True
    interpolate = staticmethod(ConditionalMeanFlow.interpolate)

    def __init__(
        self,
        network: nn.Module,
        condition_encoder: nn.Module,
        equal_time_probability=0.75,
        conditional_input_probability=0.5,
        zero_reconstruction_weight=1.0,
        zero_reconstruction_margin=0.3,
        adaptive_loss_epsilon=1e-3,
        inference_source_diffusion_time=0.95,
    ):
        super().__init__(network, condition_encoder)
        for name, value in (
            ("equal_time_probability", equal_time_probability),
            ("conditional_input_probability", conditional_input_probability),
            ("zero_reconstruction_margin", zero_reconstruction_margin),
            ("inference_source_diffusion_time", inference_source_diffusion_time),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} 必须位于 [0, 1]")
        if zero_reconstruction_weight < 0 or adaptive_loss_epsilon <= 0:
            raise ValueError("重构权重必须非负，epsilon 必须为正数")
        self.equal_time_probability = equal_time_probability
        self.conditional_input_probability = conditional_input_probability
        self.zero_reconstruction_weight = zero_reconstruction_weight
        self.zero_reconstruction_margin = zero_reconstruction_margin
        self.adaptive_loss_epsilon = adaptive_loss_epsilon
        self.inference_source_diffusion_time = inference_source_diffusion_time

    def forward(
        self,
        noisy_mels,
        start_times,
        end_times,
        source_diffusion_times,
        condition,
        target_mask=None,
    ):
        return self.network(
            noisy_mels,
            {
                "time": end_times,
                "interval": end_times - start_times,
                "source_time": source_diffusion_times,
            },
            condition,
            target_mask,
        )

    @staticmethod
    def _sample_logit_normal(batch_size, device, dtype):
        return torch.sigmoid(torch.randn(batch_size, device=device, dtype=dtype))

    def _sample_time_intervals(self, batch_size, device, dtype):
        first = self._sample_logit_normal(batch_size, device, dtype)
        second = self._sample_logit_normal(batch_size, device, dtype)
        start_times = torch.minimum(first, second)
        end_times = torch.maximum(first, second)
        equal_time_mask = (
            torch.rand(batch_size, device=device, dtype=dtype) < self.equal_time_probability
        )
        return torch.where(equal_time_mask, end_times, start_times), end_times

    @staticmethod
    def _shuffle_speaker_features(speaker_features):
        batch_size = speaker_features.size(0)
        if batch_size < 2:
            return speaker_features
        shift = int(torch.randint(1, batch_size, (), device=speaker_features.device).item())
        return speaker_features.roll(shifts=shift, dims=0)

    def _build_training_prior(
        self, noise, content_features, content_lengths, speaker_features, target_mask
    ):
        batch_size = noise.size(0)
        pure_noise_times = torch.ones(batch_size, device=noise.device, dtype=noise.dtype)
        if batch_size < 2 or self.conditional_input_probability == 0.0:
            return noise, pure_noise_times, torch.zeros_like(pure_noise_times, dtype=torch.bool)
        source_diffusion_times = self._sample_logit_normal(batch_size, noise.device, noise.dtype)
        source_condition = self.encode_condition(
            content_features=content_features,
            content_lengths=content_lengths,
            speaker_features=self._shuffle_speaker_features(speaker_features),
            target_mask=target_mask,
        )
        with torch.no_grad():
            pseudo_source_velocity = self(
                noise,
                source_diffusion_times,
                pure_noise_times,
                pure_noise_times,
                source_condition,
                target_mask,
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
        effective_times = torch.where(conditional_mask, source_diffusion_times, pure_noise_times)
        return training_prior.detach(), effective_times, conditional_mask

    def _adaptive_mean_flow_loss(self, prediction, target, mask):
        valid = mask.unsqueeze(1).to(prediction.dtype)
        error_energy = ((prediction - target).square() * valid).sum(dim=(1, 2))
        denominator = (error_energy + self.adaptive_loss_epsilon).detach()
        return (error_energy / denominator).mean()

    def _masked_ssim_loss(self, prediction, target, mask):
        valid = mask.unsqueeze(1).expand_as(target)
        positive_infinity = torch.full_like(target, torch.inf)
        negative_infinity = torch.full_like(target, -torch.inf)
        target_min = torch.where(valid, target, positive_infinity).amin(dim=(1, 2), keepdim=True)
        target_max = torch.where(valid, target, negative_infinity).amax(dim=(1, 2), keepdim=True)
        target_range = (target_max - target_min).clamp_min(1e-6)
        normalized_prediction = (prediction - target_min) / target_range
        normalized_target = (target - target_min) / target_range
        prediction_image = normalized_prediction.unsqueeze(1)
        target_image = normalized_target.unsqueeze(1)
        valid_image = valid.unsqueeze(1).to(prediction.dtype)
        local_weight = F.avg_pool2d(valid_image, kernel_size=3, stride=1, padding=1)
        safe_weight = local_weight.clamp_min(1e-6)
        prediction_mean = (
            F.avg_pool2d(prediction_image * valid_image, kernel_size=3, stride=1, padding=1)
            / safe_weight
        )
        target_mean = (
            F.avg_pool2d(target_image * valid_image, kernel_size=3, stride=1, padding=1)
            / safe_weight
        )
        prediction_second_moment = (
            F.avg_pool2d(
                prediction_image.square() * valid_image, kernel_size=3, stride=1, padding=1
            )
            / safe_weight
        )
        target_second_moment = (
            F.avg_pool2d(target_image.square() * valid_image, kernel_size=3, stride=1, padding=1)
            / safe_weight
        )
        cross_moment = (
            F.avg_pool2d(
                prediction_image * target_image * valid_image, kernel_size=3, stride=1, padding=1
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

    def compute_loss(
        self,
        target_mels: torch.Tensor,
        target_mask: torch.Tensor,
        content_features: torch.Tensor,
        content_lengths: torch.Tensor,
        speaker_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回自适应 MeanFlow 损失、预测、目标和路径状态。"""
        target_condition = self.encode_condition(
            content_features, content_lengths, speaker_features, target_mask
        )
        noise = torch.randn_like(target_mels)
        training_prior, source_diffusion_times, _ = self._build_training_prior(
            noise, content_features, content_lengths, speaker_features, target_mask
        )
        start_times, end_times = self._sample_time_intervals(
            target_mels.size(0), target_mels.device, target_mels.dtype
        )
        path_samples = self.interpolate(target_mels, training_prior, end_times)
        instantaneous_velocity = training_prior - target_mels

        def average_velocity(states, times, interval_starts):
            return self(
                states,
                interval_starts,
                times,
                source_diffusion_times,
                target_condition,
                target_mask,
            )

        predicted_velocity = average_velocity(path_samples, end_times, start_times)
        with torch.no_grad(), torch.autocast(device_type=target_mels.device.type, enabled=False):
            jvp_fn = cast(
                Callable[..., tuple[torch.Tensor, torch.Tensor]], getattr(torch.func, "jvp")
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
            predicted_velocity, target_velocity, target_mask
        )
        zero_times = torch.zeros(
            target_mels.size(0), device=target_mels.device, dtype=target_mels.dtype
        )
        one_times = torch.ones_like(zero_times)
        zero_velocity = self(
            torch.zeros_like(target_mels),
            zero_times,
            one_times,
            one_times,
            target_condition,
            target_mask,
        )
        reconstruction_loss = self._masked_ssim_loss(-zero_velocity, target_mels, target_mask)
        loss = mean_flow_loss + self.zero_reconstruction_weight * reconstruction_loss
        return loss, predicted_velocity, target_velocity, path_samples

    @torch.no_grad()
    def sample(
        self,
        condition,
        target_mask=None,
        integration_steps=None,
        *,
        source_mels=None,
        source_diffusion_time=None,
    ):
        """从扩散源 Mel 和条件特征执行一步语音转换。"""
        if integration_steps not in (None, 1):
            raise ValueError("MeanVoiceFlow 仅支持一步生成")
        if source_mels is None:
            raise ValueError("MeanVoiceFlow 推理需要 source_mels")
        diffusion_time = (
            self.inference_source_diffusion_time
            if source_diffusion_time is None
            else source_diffusion_time
        )
        if not 0.0 <= diffusion_time <= 1.0:
            raise ValueError("source_diffusion_time 必须位于 [0, 1]")
        noise = torch.randn_like(source_mels)
        diffusion_times = torch.full(
            (source_mels.size(0),),
            diffusion_time,
            device=source_mels.device,
            dtype=source_mels.dtype,
        )
        diffused_source = self.interpolate(source_mels, noise, diffusion_times)
        end_times = torch.ones_like(diffusion_times)
        start_times = torch.zeros_like(diffusion_times)
        velocity = self(
            diffused_source, start_times, end_times, diffusion_times, condition, target_mask
        )
        return self.apply_mask(diffused_source - velocity, target_mask)
