"""核验 P4 配对完整性和噪声聚合，使用人工分数，不下载模型。"""

# ====================
# 1. Imports
# ====================

from copy import deepcopy

import numpy as np
import pytest

from src.analysis.p4.identity import CONFIGS, paired_ci, source_means, validate_rows
from src.analysis.p4.scoring import unit_embedding
from src.analysis.p4.sensitivity import crossed_ci

# ====================
# 3. Utilities
# 通用工具函数
# ====================


def _rows(tmp_path):
    wav = tmp_path / "fake.wav"
    wav.touch()
    return [
        {
            "experiment_id": config,
            "source_id": "p1_001",
            "repeat": repeat,
            "reference_id": "p2_001",
            "source_speaker_id": "p1",
            "reference_speaker_id": "p2",
            "source_text": "A sentence.",
            "normalized_reference": "a sentence",
            "donor_id": "p1_002",
            "length_bin": "<3s",
            "partition": "diagnostic_A",
            "seed": str(123 + repeat),
            "source_time": dict(zip(CONFIGS, (0.8, 0.9, 0.95)))[config],
            "wav_path": str(wav),
            "cer": 0.1,
            "target_speaker_similarity": 0.2 + 0.2 * repeat,
            "source_speaker_similarity": 0.3,
        }
        for config in CONFIGS
        for repeat in (0, 1)
    ]


# ====================
# 6. Public API
# 对外提供的函数
# ====================


def test_repeated_noise_is_averaged_with_preference_before_aggregation(tmp_path):
    rows = validate_rows(_rows(tmp_path), {"p1": "F", "p2": "M"}, expected_sources=1)
    means = source_means(rows)
    assert len(means) == 3
    assert means[0]["target_preferred"] == 0.5
    assert abs(means[0]["target_preference_gap"]) < 1e-12
    assert means[0]["gender_pair"] == "F→M"


@pytest.mark.parametrize(
    "mutation", ["duplicate", "missing", "seed", "reference", "nan", "time", "fractional_repeat"]
)
def test_rejects_broken_pairing(tmp_path, mutation):
    rows = deepcopy(_rows(tmp_path))
    if mutation == "duplicate":
        rows.append(dict(rows[0]))
    elif mutation == "missing":
        rows.pop()
    elif mutation == "seed":
        rows[-1]["seed"] = 999
    elif mutation == "reference":
        rows[-1]["reference_id"] = "p2_999"
    elif mutation == "time":
        rows[-1]["source_time"] = 0.5
    elif mutation == "fractional_repeat":
        rows[-1]["repeat"] = 0.5
    else:
        rows[-1]["cer"] = float("nan")
    with pytest.raises(ValueError):
        validate_rows(rows, {"p1": "F", "p2": "M"}, expected_sources=1)


def test_cluster_ci_uses_source_units_and_preserves_constant_difference():
    metadata = [{"source_speaker_id": f"p{i // 2}", "source_text": str(i % 3)} for i in range(12)]
    assert paired_ci([0.1] * 12, metadata) == pytest.approx((0.1, 0.1))


@pytest.mark.parametrize("value", [[0, 0], [np.nan, 1], [np.inf, 1]])
def test_invalid_independent_embedding_is_rejected(value):
    with pytest.raises(ValueError):
        unit_embedding(value)


def test_crossed_ci_preserves_constant_delta_with_reused_target_speakers():
    metadata = [
        {
            "source_speaker_id": f"p{i // 2}",
            "reference_speaker_id": f"t{i % 2}",
            "source_text": str(i % 3),
        }
        for i in range(12)
    ]
    assert crossed_ci([-0.1] * 12, metadata) == pytest.approx((-0.1, -0.1))
