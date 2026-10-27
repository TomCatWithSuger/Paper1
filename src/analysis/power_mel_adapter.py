"""模块说明：

- 将项目的 HTK 功率 Mel 转为冻结 Vocos 的归一化幅度 Mel。
- 依据权重内保存的真实滤波器构造统一、无可训练参数的近似适配。
"""

# ====================
# 1. Imports
# ====================

import torch

from src.analysis.vocoder.audio import _ours_mel_centers, _ours_mel_filterbank

# ====================
# 2. Definitions
# ====================


class PowerMelAdapter:
    """以带内平均功率近似频谱幅度；不能逆转 Mel 压缩丢失的细节。"""

    def __init__(self, target_filterbank: torch.Tensor):
        source = _ours_mel_filterbank()
        self.source_mass = source.sum(dim=1).clamp_min(1e-12)
        centers = _ours_mel_centers()
        frequencies = torch.linspace(0, 22050 / 2, 513)
        interpolation = torch.zeros(513, 80)
        for i, frequency in enumerate(frequencies):
            if frequency <= centers[0]:
                interpolation[i, 0] = 1
            elif frequency >= centers[-1]:
                interpolation[i, -1] = 1
            else:
                right = int(torch.searchsorted(centers, frequency))
                weight = (frequency - centers[right - 1]) / (centers[right] - centers[right - 1])
                interpolation[i, right - 1] = 1 - weight
                interpolation[i, right] = weight
        assert target_filterbank.shape == (513, 80)
        self.amplitude_projection = target_filterbank.T.float() @ interpolation

    def __call__(self, log_power_mel: torch.Tensor):
        """先除源频带质量、开方，再按真实目标滤波器积分并取对数。"""
        value = log_power_mel.float()
        amplitude = torch.exp(0.5 * (value - self.source_mass.to(value.device).log()[:, None]))
        projected = self.amplitude_projection.to(value.device) @ amplitude
        return projected.clamp_min(1e-7).log()


# ====================
# 6. Public API
# ====================


def from_vocos_weights(path):
    """读取本地声码器内的滤波器；不使用其他模型或在线配置。"""
    weights = torch.load(path, map_location="cpu", weights_only=True)
    return PowerMelAdapter(weights["feature_extractor.mel_spec.mel_scale.fb"])
