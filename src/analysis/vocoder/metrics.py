"""计算文本错误率、基频和能量轮廓指标。"""

# ====================
# 1. Imports
# ====================


import math
import re
from typing import Any, Callable, Sequence

import torch
import torchaudio

from src.analysis.vocoder.definitions import (
    CONTOUR_POINTS,
    ENERGY_HOP,
    ENERGY_WIN,
    F0_FRAME_TIME,
    F0_FREQ_HIGH,
    F0_FREQ_LOW,
    SAMPLE_RATE,
)

# ====================
# 4. Internal Functions
# 内部业务实现
# ====================


def _normalize_text(text: str, normalizer: Callable[[str], str] | None = None) -> str:
    """归一化文本：小写、去标点、压缩空白。"""
    if normalizer is not None:
        return normalizer(text).strip()
    lowered = text.lower()
    cleaned = re.sub(r"[^\w\s]", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def _edit_distance(ref: Sequence[Any], hyp: Sequence[Any]) -> int:
    """计算两个序列的编辑距离。"""
    if len(ref) < len(hyp):
        ref, hyp = hyp, ref
    previous = list(range(len(hyp) + 1))
    for i, ref_item in enumerate(ref, start=1):
        current = [i] + [0] * len(hyp)
        for j, hyp_item in enumerate(hyp, start=1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ref_item != hyp_item),
            )
        previous = current
    return previous[-1]


def _word_error_rate(reference: str, hypothesis: str) -> float:
    """计算词错误率（WER）。"""
    ref_words = reference.split()
    if not ref_words:
        return math.nan
    return _edit_distance(ref_words, hypothesis.split()) / len(ref_words)


def _char_error_rate(reference: str, hypothesis: str) -> float:
    """计算字错误率（CER，忽略空格）。"""
    ref_chars = list(reference.replace(" ", ""))
    if not ref_chars:
        return math.nan
    hyp_chars = list(hypothesis.replace(" ", ""))
    return _edit_distance(ref_chars, hyp_chars) / len(ref_chars)


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    """计算两个等长序列的皮尔逊相关系数。"""
    if left.numel() < 2:
        return math.nan
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.norm() * right.norm()
    if float(denominator) <= 0.0:
        return math.nan
    return float((left * right).sum() / denominator)


def _resample_contour(values: torch.Tensor, points: int) -> torch.Tensor:
    """将轮廓序列线性重采样到固定点数（归一化时间轴）。"""
    if values.numel() == 0:
        return torch.full((points,), math.nan)
    if values.numel() == 1:
        return values.repeat(points)
    return torch.nn.functional.interpolate(
        values.view(1, 1, -1), size=points, mode="linear", align_corners=True
    ).view(-1)


def _frame_log_energy(waveform: torch.Tensor) -> torch.Tensor:
    """按固定窗移计算分帧 log-RMS 能量。"""
    if waveform.numel() < ENERGY_WIN:
        waveform = torch.nn.functional.pad(waveform, (0, ENERGY_WIN - waveform.numel()))
    frames = waveform.unfold(0, ENERGY_WIN, ENERGY_HOP)
    rms = frames.pow(2).mean(dim=1).clamp_min(1e-10).sqrt()
    return rms.log()


def _f0_contour(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """提取逐帧基频轮廓（0 表示清音）。"""
    return torchaudio.functional.detect_pitch_frequency(
        waveform,
        sample_rate,
        frame_time=F0_FRAME_TIME,
        freq_low=F0_FREQ_LOW,
        freq_high=F0_FREQ_HIGH,
    )


def _contour_metrics(generated: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    """计算 F0 与能量轮廓指标（生成 vs 参考）。"""
    gen_f0 = _resample_contour(_f0_contour(generated, SAMPLE_RATE), CONTOUR_POINTS)
    ref_f0 = _resample_contour(_f0_contour(reference, SAMPLE_RATE), CONTOUR_POINTS)
    voiced = (gen_f0 > 0) & (ref_f0 > 0)
    if int(voiced.sum()) >= 10:
        f0_corr = _pearson(gen_f0[voiced].log(), ref_f0[voiced].log())
        f0_log_mae = float((gen_f0[voiced].log() - ref_f0[voiced].log()).abs().mean())
    else:
        f0_corr = math.nan
        f0_log_mae = math.nan

    gen_energy = _resample_contour(_frame_log_energy(generated), CONTOUR_POINTS)
    ref_energy = _resample_contour(_frame_log_energy(reference), CONTOUR_POINTS)
    energy_corr = _pearson(gen_energy, ref_energy)
    energy_log_mae = float((gen_energy - ref_energy).abs().mean())

    return {
        "f0_corr": f0_corr,
        "f0_log_mae": f0_log_mae,
        "energy_corr": energy_corr,
        "energy_log_mae": energy_log_mae,
    }
