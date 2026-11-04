"""模块说明：用合成配对记录验证 P2 报告的统计单位、分层和检查点审计。"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.analysis.p1.evaluation import _csv
from src.analysis.p2 import reporting


def _metadata() -> list[dict]:
    """交叉设置说话人与文本，保留两种验证分区。"""
    return [
        {
            "source_id": f"p{index // 3}_{index % 3}",
            "source_speaker_id": f"p{index // 3}",
            "source_text": f"text {index % 3}",
            "audio_seconds": 2 + index,
            "partition": "diagnostic_A" if index < 3 else "confirmation_B",
            "length_bin": "short" if index < 3 else "long",
        }
        for index in range(6)
    ]


def _write_inputs(output: Path, metadata: list[dict]) -> None:
    """构造各阶段的真实文件格式，不替换被测报告和聚合函数。"""
    (output / "validation_manifest.json").write_text(json.dumps(metadata))
    intervals = []
    for index, item in enumerate(metadata):
        for group_index, group in enumerate(("zero", "short", "medium", "long", "full")):
            for repeat in range(2):
                value = float(index + group_index + repeat + 1)
                intervals.append(
                    {
                        **item,
                        "repeat": repeat,
                        "interval_group": group,
                        "r": 0.0,
                        "t": group_index / 4,
                        "delta": group_index / 4,
                        "valid_frames": index + 1,
                        **dict.fromkeys(reporting.INTERVAL_METRICS, value),
                        "terminal_reconstruction_l1": value,
                    }
                )
    for name in ("d1", "d1_center_matched", "d1_boundary"):
        (output / name).mkdir()
        _csv(output / name / "per_sample_intervals.csv", intervals)
    experiments = (
        "b1_anchor",
        "b2_base",
        "b2_content_donor",
        "b2_speaker_source",
        "b3_best_s080",
        "b3_best_s090",
        "b3_best_s095",
        "b3_best_s100",
    )
    quality = []
    for experiment_index, experiment in enumerate(experiments):
        for index, item in enumerate(metadata):
            for repeat in range(2):
                quality.append(
                    {
                        **item,
                        "experiment_id": experiment,
                        "repeat": repeat,
                        "donor_id": metadata[index ^ 1]["source_id"],
                        "normalized_reference": "hello world",
                        "normalized_hypothesis": "hello world",
                        **dict.fromkeys(
                            reporting.QUALITY_METRICS, index + repeat + experiment_index / 10
                        ),
                        "target_speaker_similarity": 0.8,
                        "source_speaker_similarity": 0.3,
                    }
                )
    (output / "conversion").mkdir()
    _csv(output / "conversion/per_sample_metrics.csv", quality)
    (output / "d2").mkdir()
    losses = []
    metrics = (
        "raw_mse",
        "raw_mse_common",
        "adaptive_loss",
        "ssim_loss",
        "total_loss",
        "jvp_correction_rms",
        "velocity_mse",
        "zero_reconstruction_l1",
        "error_energy",
        "adaptive_gradient_weight",
        "margin_hit",
    )
    for checkpoint in ("best", "last"):
        for index, item in enumerate(metadata):
            losses.append(
                {
                    **item,
                    "checkpoint": checkpoint,
                    "batch_index": index // 2,
                    "repeat": 0,
                    "pseudo": index % 2 == 0,
                    **dict.fromkeys(metrics, 2.0),
                }
            )
        (output / f"d2/{checkpoint}_model.json").write_text(
            json.dumps({"sha256": checkpoint, "epoch": 499 if checkpoint == "last" else 40})
        )
        _csv(
            output / f"d2/{checkpoint}_original_val_pairing_audit.csv",
            [
                {
                    "batch_size": 6,
                    "requested_pseudo": 3,
                    "effective_pseudo": 2,
                    "fallback_pure_noise": 1,
                    "intentional_pure_noise": 3,
                }
            ],
        )
    _csv(output / "d2/loss_components.csv", losses)


def test_full_report_preserves_pairing_and_bootstrap_units(tmp_path: Path) -> None:
    """从原始 CSV 生成全套报告，并用已知差值核验统计结果。"""
    metadata = _metadata()
    _write_inputs(tmp_path, metadata)
    reporting.run(tmp_path)

    paired = reporting.read(tmp_path / "d1/paired_vs_zero.csv")
    assert len(paired) == 12
    full = next(row for row in paired if row["group_a"] == "full")
    assert full["mean"] == pytest.approx(4)
    assert full["ci95_low"] == pytest.approx(4)
    assert full["ci95_high"] == pytest.approx(4)
    assert full["n_sources"] == 6

    enriched = reporting.read(tmp_path / "conversion/per_sample_enriched.csv")
    assert len(enriched) == 96
    assert all(row["target_preference_gap"] == pytest.approx(0.5) for row in enriched)
    assert all(row["donor_cer"] == 0 and row["donor_wer"] == 0 for row in enriched)
    quality = reporting.read(tmp_path / "conversion/paired_comparisons.csv")
    difference = next(
        row
        for row in quality
        if row["experiment_a"] == "b2_base" and row["subset"] == "all" and row["metric"] == "cer"
    )
    assert difference["mean"] == pytest.approx(0.1)
    assert difference["n_sources"] == 6

    components = reporting.read(tmp_path / "d2/component_intervals.csv")
    assert len(components) == 11
    assert all(row["n_microbatch_blocks"] == 3 for row in components)
    assert all(row["mean"] == row["ci95_low"] == row["ci95_high"] == 2 for row in components)
    validity = json.loads((tmp_path / "d2/checkpoint_comparison_validity.json").read_text())
    assert validity["can_compare_early_vs_final"] is True
    assert validity["same_file_hash"] is False
    associations = reporting.read(tmp_path / "d1/endpoint_quality_associations.csv")
    assert len(associations) == 4
    assert all(row["spearman"] == pytest.approx(1) for row in associations)
    assert all(row["causal_evidence"] == "False" for row in associations)
    priors = reporting.read(tmp_path / "d2/pseudo_source_summary.csv")
    assert priors[0]["effective_fraction"] == pytest.approx(1 / 3)
    assert priors[1]["effective_fraction"] == pytest.approx(0.5)


def test_nonfinite_values_and_frame_weighting() -> None:
    """缺失指标不产生伪均值，帧加权结果与源平均分别计算。"""
    metadata = _metadata()
    empty = reporting.mean_ci([np.nan] * len(metadata), metadata)
    assert empty["n_sources"] == 0 and np.isnan(empty["mean"])
    result = reporting.summary(
        [
            {"source_id": "a", "repeat": 0, "raw_mse": 1, "valid_frames": 1, "absent": np.nan},
            {"source_id": "b", "repeat": 0, "raw_mse": 3, "valid_frames": 3, "absent": np.nan},
        ],
        [],
        ["raw_mse", "absent"],
    )[0]
    assert result["raw_mse_mean"] == 2
    assert result["raw_mse_frame_weighted"] == 2.5
    assert result["valid_frames_total"] == 4
    assert "absent_mean" not in result
