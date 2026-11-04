"""模块说明：

- 用显式清浊音判定计算源韵律保持与参考韵律接近度。
- 提供固定 NCCF 基频估计、能量轮廓及带覆盖率的比较指标。
"""

# ====================
# 1. Imports
# ====================

import numpy as np
import torch
import torchaudio

# ====================
# 2. Definitions
# ====================

PITCH_SR = 16000
FRAME = 640
HOP = 160
MIN_F0 = 60
MAX_F0 = 500
VOICING_THRESHOLD = 0.6
RMS_THRESHOLD = 1e-3


# ====================
# 3. Utilities
# ====================


def _corr(left, right):
    if len(left) < 10 or np.std(left) < 1e-8 or np.std(right) < 1e-8:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


# ====================
# 4. Internal Functions
# ====================


def pitch_track(waveform: torch.Tensor, sample_rate=22050):
    """按 10ms 帧移估计 NCCF 基频；静音和非周期帧明确记作清音。"""
    x = torchaudio.functional.resample(waveform.float().cpu(), sample_rate, PITCH_SR).numpy()
    x = np.pad(x, (FRAME // 2, FRAME // 2))
    frames = np.lib.stride_tricks.sliding_window_view(x, FRAME)[::HOP].astype(np.float64)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    frames = frames - frames.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(frames, n=2048, axis=1)
    autocorr = np.fft.irfft(spectrum * spectrum.conj(), n=2048, axis=1)[:, :FRAME]
    energy = np.pad(np.cumsum(frames**2, axis=1), ((0, 0), (1, 0)))
    lags = np.arange(PITCH_SR // MAX_F0, PITCH_SR // MIN_F0 + 1)
    denom = np.sqrt(energy[:, FRAME - lags] * (energy[:, -1:] - energy[:, lags]))
    nccf = autocorr[:, lags] / np.maximum(denom, 1e-12)
    # 优先取接近最强峰的第一个局部峰，降低整倍周期造成的倍频程错误。
    peaks = np.zeros_like(nccf, dtype=bool)
    peaks[:, 1:-1] = (nccf[:, 1:-1] >= nccf[:, :-2]) & (nccf[:, 1:-1] > nccf[:, 2:])
    peaks[:, 0] = nccf[:, 0] > nccf[:, 1]
    peaks[:, -1] = nccf[:, -1] >= nccf[:, -2]
    candidates = peaks & (nccf >= np.maximum(VOICING_THRESHOLD, nccf.max(axis=1)[:, None] * 0.9))
    chosen = np.where(candidates.any(axis=1), candidates.argmax(axis=1), nccf.argmax(axis=1))
    confidence = nccf[np.arange(len(frames)), chosen]
    voiced = (confidence >= VOICING_THRESHOLD) & (rms >= RMS_THRESHOLD)
    f0 = np.where(voiced, PITCH_SR / lags[chosen], 0.0)
    return {
        "f0": f0,
        "voiced": voiced,
        "confidence": confidence,
        "log_energy": np.log(np.maximum(rms, 1e-8)),
    }


# ====================
# 5. Core Logic
# ====================


def compare_tracks(generated, baseline, *, normalized_time=False):
    """源比较按真实时间对齐；参考比较单独按归一化时轴对齐。"""
    gen_n, base_n = len(generated["f0"]), len(baseline["f0"])
    size = min(gen_n, base_n)
    indices = (
        np.rint(np.linspace(0, base_n - 1, gen_n)).astype(int)
        if normalized_time
        else np.arange(size)
    )
    gen_indices = np.arange(gen_n) if normalized_time else np.arange(size)
    gv = generated["voiced"][gen_indices]
    bv = baseline["voiced"][indices]
    joint = gv & bv
    a = np.log(np.maximum(generated["f0"][gen_indices][joint], 1e-8))
    b = np.log(np.maximum(baseline["f0"][indices][joint], 1e-8))
    enough = len(a) >= 10
    centered = (a - a.mean()) - (b - b.mean()) if enough else np.array([])
    ge = generated["log_energy"][gen_indices]
    be = baseline["log_energy"][indices]
    return {
        "f0_corr": _corr(a, b),
        "f0_log_mae": float(np.mean(np.abs(a - b))) if enough else float("nan"),
        "f0_centered_rmse_semitones": (
            float(np.sqrt(np.mean(centered**2)) * 12 / np.log(2)) if enough else float("nan")
        ),
        "vuv_error": float(np.mean(gv != bv)),
        "joint_voiced_fraction": float(joint.mean()),
        "generated_voiced_fraction": float(gv.mean()),
        "baseline_voiced_fraction": float(bv.mean()),
        "aligned_frames": len(joint),
        "energy_corr": _corr(ge, be),
        "energy_log_mae": float(np.mean(np.abs(ge - be))),
    }


# ====================
# 6. Public API
# ====================


def compare_prosody(generated: torch.Tensor, source: torch.Tensor, reference: torch.Tensor):
    """输出明确区分 source/reference 的韵律指标。"""
    track = pitch_track(generated)
    result = {}
    for name, waveform in (("source", source), ("reference", reference)):
        metrics = compare_tracks(track, pitch_track(waveform), normalized_time=name == "reference")
        result.update({f"{name}_{key}": value for key, value in metrics.items()})
    return result
