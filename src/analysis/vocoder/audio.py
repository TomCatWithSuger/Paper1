"""处理波形读写、项目 Mel 提取与声码器频带适配。"""

# ====================
# 1. Imports
# ====================


import math
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from src.analysis.vocoder.definitions import (
    F_MAX,
    F_MIN,
    HOP_LENGTH,
    LOG_MEL_FLOOR,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    VOCOS_LOG_FLOOR,
    WIN_LENGTH,
)

# ====================
# 4. Internal Functions
# 内部业务实现
# ====================


def _load_audio(path: Path, target_sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """读取音频为单声道 float32 波形并重采样。"""
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio).mean(dim=1)
    if int(sample_rate) != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, int(sample_rate), target_sample_rate)
    return waveform.contiguous()


def _save_waveform(path: Path, waveform: torch.Tensor) -> None:
    """以 16bit PCM 保存波形。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform.detach().cpu().numpy(), SAMPLE_RATE, subtype="PCM_16")


def _ours_mel_centers() -> torch.Tensor:
    """计算项目 HTK 式三角滤波器组的中心频率。"""

    def hz_to_mel(frequency: torch.Tensor) -> torch.Tensor:
        return 2595.0 * torch.log10(1.0 + frequency / 700.0)

    def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
        return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)

    mel_points = torch.linspace(
        hz_to_mel(torch.tensor(F_MIN)), hz_to_mel(torch.tensor(F_MAX)), N_MELS + 2
    )
    return mel_to_hz(mel_points)[1:-1]


def _slaney_mel_centers() -> torch.Tensor:
    """计算 Vocos(slaney) 滤波器组的中心频率。"""
    mel_scale = torchaudio.transforms.MelScale(
        n_mels=N_MELS,
        sample_rate=SAMPLE_RATE,
        f_min=F_MIN,
        f_max=F_MAX,
        n_stft=N_FFT // 2 + 1,
        norm="slaney",
        mel_scale="slaney",
    )
    # torchaudio 的 fb 形状为 [n_stft, n_mels]，沿频率轴取峰值作为中心频率
    bin_width = SAMPLE_RATE / N_FFT
    return mel_scale.fb.argmax(dim=0).float() * bin_width


def _build_mel_adapter() -> torch.Tensor:
    """构建 HTK→slaney 频带线性插值矩阵 [n_mels, n_mels]。"""
    source = _ours_mel_centers()
    target = _slaney_mel_centers()
    matrix = torch.zeros(N_MELS, N_MELS)
    for row, center in enumerate(target.tolist()):
        if center <= source[0]:
            matrix[row, 0] = 1.0
            continue
        if center >= source[-1]:
            matrix[row, -1] = 1.0
            continue
        index = int(torch.searchsorted(source, torch.tensor(center)).item())
        left, right = source[index - 1], source[index]
        weight = (center - float(left)) / max(float(right - left), 1e-12)
        matrix[row, index - 1] = 1.0 - weight
        matrix[row, index] = weight
    return matrix


def _adapt_log_mel(log_mel: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """将项目 log 功率 Mel 适配为 Vocos 输入（log 幅度 + slaney 频带）。

    log_mel: [n_mels, frames]
    """
    amplitude = log_mel.float() * 0.5
    adapted = matrix @ amplitude
    return adapted.clamp_min(math.log(VOCOS_LOG_FLOOR))


def _ours_mel_filterbank() -> torch.Tensor:
    """构建项目自定义 HTK 式三角滤波器组（与 VCTKCollator 完全一致）。"""

    def hz_to_mel(frequency: torch.Tensor) -> torch.Tensor:
        return 2595.0 * torch.log10(1.0 + frequency / 700.0)

    def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
        return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)

    mel_points = torch.linspace(
        hz_to_mel(torch.tensor(F_MIN)), hz_to_mel(torch.tensor(F_MAX)), N_MELS + 2
    )
    frequency_points = mel_to_hz(mel_points)
    fft_frequencies = torch.linspace(0.0, SAMPLE_RATE / 2, N_FFT // 2 + 1)
    lower = frequency_points[:-2].unsqueeze(1)
    center = frequency_points[1:-1].unsqueeze(1)
    upper = frequency_points[2:].unsqueeze(1)
    ascending = (fft_frequencies.unsqueeze(0) - lower) / (center - lower)
    descending = (upper - fft_frequencies.unsqueeze(0)) / (upper - center)
    return torch.minimum(ascending, descending).clamp_min(0.0)


def _project_log_mel(waveform: torch.Tensor, filterbank: torch.Tensor) -> torch.Tensor:
    """计算项目口径的 log 功率 Mel。"""
    spectrum = torch.stft(
        waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=torch.hann_window(WIN_LENGTH),
        center=True,
        pad_mode="constant",
        return_complex=True,
    )
    mel = filterbank @ spectrum.abs().pow(2)
    return torch.log(mel.clamp_min(LOG_MEL_FLOOR))
