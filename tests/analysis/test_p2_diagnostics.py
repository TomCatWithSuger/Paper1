"""模块说明：检查 P2 有效帧口径与真实方向导数有限差分，不加载预训练模型。"""

import pytest
import torch

from src.analysis.p2.intervals import decompose_native, fd_statistics, masked_values
from src.analysis.p2.reporting import read, source_means
from src.models.algorithms.autodiff import tensor_jvp
from src.models.components.conditioning.meanvc import CachedConditionEncoder
from src.models.methods.generative.flow.mean_flow.meanvoiceflow import MeanVoiceFlow
from src.models.networks.unet.conditional import UNet


def test_csv_numeric_metrics_preserve_text_and_source_means(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text(
        "source_id,metric,label\na,1.5,control\na,2.5,intervention\n", encoding="utf-8"
    )
    rows = read(path)
    assert rows[0] == {"source_id": "a", "metric": 1.5, "label": "control"}
    assert source_means(rows, "metric") == {"a": 2.0}


def test_tensor_jvp_preserves_values_and_backward_gradient():
    values = torch.tensor([2.0, 3.0], requires_grad=True)
    result, derivative = tensor_jvp(
        lambda states: states.square(),
        (values,),
        (torch.ones_like(values),),
        has_aux=False,
    )
    torch.testing.assert_close(result, values.square())
    torch.testing.assert_close(derivative, 2 * values)
    derivative.sum().backward()
    torch.testing.assert_close(values.grad, torch.full_like(values, 2.0))


def test_common_window_ignores_padding_and_equalizes_counts():
    values = torch.ones(2, 80, 320)
    values[0, :, 160:] = 1e6
    mask = torch.arange(320)[None] < torch.tensor([160, 320])[:, None]
    torch.testing.assert_close(masked_values(values, mask), torch.ones(2))
    torch.testing.assert_close(masked_values(values, mask, True), torch.ones(2))


def test_fd_separates_state_time_and_fixed_start():
    z = torch.ones(2, 3, 160, dtype=torch.float64)
    t = torch.tensor([0.3, 0.7], dtype=torch.float64)
    r = torch.tensor([0.1, 0.2], dtype=torch.float64)
    velocity = torch.full_like(z, 2)

    def function(states, ends, starts):
        return states.square() + (ends - starts)[:, None, None] * states

    _, automatic = tensor_jvp(
        function, (z, t, r), (velocity, torch.ones_like(t), torch.zeros_like(r))
    )
    h = 1e-4
    finite = (function(z + h * velocity, t + h, r) - function(z - h * velocity, t - h, r)) / (
        2 * h
    )
    rows = fd_statistics(automatic, finite, torch.ones(2, 160, dtype=torch.bool))
    assert all(row["relative_l2"] < 1e-9 and row["cosine"] > 0.999 for row in rows)
    wrong = 2 * z * velocity + (t - r)[:, None, None] * velocity
    assert all(
        row["relative_l2"] > 0.1
        for row in fd_statistics(automatic, wrong, torch.ones(2, 160, dtype=torch.bool))
    )


def test_native_b3_capture_preserves_loss_weights_and_restores_methods():
    torch.manual_seed(42)
    method = (
        MeanVoiceFlow(
            UNet(
                n_mels=2,
                condition_dim=4,
                hidden_channels=16,
                time_embedding_dim=8,
                time_fields=["time", "interval", "source_time"],
            ),
            CachedConditionEncoder(content_dim=3, speaker_dim=2, output_dim=4),
        )
        .eval()
        .requires_grad_(False)
    )
    batch = {
        "mel_spectrograms": torch.randn(2, 2, 160),
        "mel_attention_mask": torch.arange(160)[None] < torch.tensor([160, 150])[:, None],
        "content_features": torch.randn(2, 3, 80),
        "content_feature_lengths": torch.tensor([80, 75]),
        "speaker_features": torch.randn(2, 2),
        "speaker_ids": ["a", "b"],
    }
    state = {k: v.clone() for k, v in method.state_dict().items()}
    original = method._build_training_prior.__func__
    with torch.no_grad():
        expected = method.compute_loss(
            batch["mel_spectrograms"],
            batch["mel_attention_mask"],
            batch["content_features"],
            batch["content_feature_lengths"],
            batch["speaker_features"],
            torch.Generator().manual_seed(5),
            speaker_ids=batch["speaker_ids"],
        )[0]
        rows = decompose_native(method, batch, torch.Generator().manual_seed(5))
    assert sum(r["total_loss"] for r in rows) / 2 == pytest.approx(float(expected), abs=1e-6)
    assert sum(r["pseudo"] for r in rows) == 1
    assert method._build_training_prior.__func__ is original
    for key, value in method.state_dict().items():
        torch.testing.assert_close(value, state[key], atol=0, rtol=0)
