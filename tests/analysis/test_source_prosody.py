"""检查基频估计和源/参考韵律口径，避免把目标韵律误认作保源能力。"""

import numpy as np
import pytest
import torch

from src.analysis.source_prosody import compare_prosody, compare_tracks, pitch_track


@pytest.mark.parametrize("frequency", [120, 220, 350])
def test_sine_pitch_and_silence(frequency):
    waveform = 0.1 * torch.sin(torch.arange(22050) * (2 * np.pi * frequency / 22050))
    track = pitch_track(waveform)
    assert np.median(track["f0"][track["voiced"]]) == pytest.approx(frequency, abs=5)
    assert track["voiced"].mean() > 0.9
    assert not pitch_track(torch.zeros(22050))["voiced"].any()


def test_source_and_reference_are_distinct():
    times = torch.arange(22050) / 22050
    source = 0.1 * torch.sin(2 * np.pi * 120 * times)
    reference = 0.1 * torch.sin(2 * np.pi * 220 * times)
    metrics = compare_prosody(source, source, reference)
    assert metrics["source_f0_log_mae"] == 0
    assert metrics["reference_f0_log_mae"] > 0.5
    assert metrics["source_vuv_error"] == 0


def test_voicing_failure_keeps_coverage_and_nan():
    silent = pitch_track(torch.zeros(22050))
    voiced = pitch_track(0.1 * torch.sin(torch.arange(22050) * (2 * np.pi * 120 / 22050)))
    metrics = compare_tracks(silent, voiced)
    assert metrics["joint_voiced_fraction"] == 0
    assert metrics["vuv_error"] > 0.9
    assert np.isnan(metrics["f0_corr"])
