"""检查真实功率到幅度调用口径是否保留常量频谱和幅度缩放。"""

import torch
import torchaudio

from src.analysis.power_mel_adapter import PowerMelAdapter
from src.analysis.vocoder.audio import _ours_mel_filterbank


def test_flat_spectrum_preserves_target_filterbank_scale():
    target = torchaudio.functional.melscale_fbanks(
        513, 0, 8000, 80, 22050, norm="slaney", mel_scale="slaney"
    )
    adapter = PowerMelAdapter(target)
    spectrum = torch.full((513, 4), 0.1)
    log_power = (_ours_mel_filterbank() @ spectrum.square()).log()
    expected = (target.T @ spectrum).log()
    torch.testing.assert_close(adapter(log_power), expected)
    torch.testing.assert_close(
        adapter(log_power + torch.tensor(4.0).log()), expected + torch.tensor(2.0).log()
    )
