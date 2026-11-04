"""模块说明：验证端点试验双队列生成、评分、续跑及源语句级配对报告。"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from src.analysis.endpoint_pilot import evaluation, reporting
from src.analysis.p4.identity import read_csv
from tests.helpers.diagnostic_models import (
    AnalyticModel,
    LocalAsr,
    LocalSpeaker,
    LocalVocos,
    diagnostic_batch,
)

pytestmark = pytest.mark.usefixtures("legacy_text_encoding")


@pytest.fixture()
def endpoint_runtime(tmp_path, monkeypatch):
    """满足生产协议的 768 个生成记录，仅替换外部模型和数据加载。"""
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    script = root / "scripts/analysis/run_endpoint_pilot.py"
    script.parent.mkdir(parents=True)
    script.write_text("# 测试源码快照\n", encoding="utf-8")
    (root / "src/local.py").write_text("# 测试源码快照\n", encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    manifests = {}
    for cohort, filename in (
        ("p2_fixed", "p2_validation_manifest.json"),
        ("reserved", "reserved_validation_manifest.json"),
    ):
        rows = [
            {
                "index": index,
                "source_id": f"p{index // 32 + 1}_{index:03d}",
                "source_speaker_id": f"p{index // 32 + 1}",
                "reference_id": "p3_000",
                "donor_id": "p3_000",
                "source_text": "hello world",
                "length_bin": "short",
                "audio_seconds": 0.1,
                "partition": cohort,
            }
            for index in range(64)
        ]
        manifests[cohort] = rows
        (output / filename).write_text(json.dumps(rows), encoding="utf-8")
    ids = {row["source_id"] for row in manifests["p2_fixed"]} | {"p3_000"}
    items = {
        uid: {
            "waveform": torch.ones(2205) * 0.1,
            "content_features": torch.ones(256, 4),
            "speaker_features": torch.ones(256),
        }
        for uid in ids
    }
    model = AnalyticModel()
    initial = root / "initial.ckpt"
    torch.save({"state_dict": model.state_dict(), "global_step": 422066, "epoch": 382}, initial)
    for arm in ("control", "intervention"):
        directory = output / arm
        (directory / "checkpoints").mkdir(parents=True)
        (directory / "TRAINING_COMPLETE.json").write_text(
            json.dumps({"updates": 5000}), encoding="utf-8"
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "global_step": 427066,
                "epoch": 387,
                "pilot_updates": 5000,
            },
            directory / "checkpoints/last.ckpt",
        )
    resource = root / "resource.bin"
    resource.write_bytes(b"fixture")
    (output / "identity.json").write_text(
        json.dumps({"resources": {str(resource): evaluation._sha256_file(resource)}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(evaluation, "ROOT", root)
    monkeypatch.setattr(evaluation, "BASE_CHECKPOINT", initial)
    monkeypatch.setattr(evaluation, "_model_config", lambda *args: {"model": {}})
    monkeypatch.setattr(evaluation.hydra.utils, "instantiate", lambda *args: model)
    monkeypatch.setattr(evaluation, "load_samples", lambda: ({}, None, {}))
    monkeypatch.setattr(evaluation, "prepared_items", lambda *args: items)
    monkeypatch.setattr(evaluation, "_collator", lambda *args: None)
    monkeypatch.setattr(
        evaluation, "batch_for", lambda rows, *args: diagnostic_batch(len(rows), 130)
    )
    monkeypatch.setattr(evaluation, "_AsrScorer", LocalAsr)
    monkeypatch.setattr(evaluation, "_SpeakerScorer", LocalSpeaker)
    vocos = LocalVocos()
    monkeypatch.setattr(evaluation, "_load_vocos", lambda *args: vocos)
    monkeypatch.setattr(evaluation, "from_vocos_weights", lambda *args: lambda mel: mel.float())
    monkeypatch.setattr(evaluation, "pitch_track", lambda wave: {"f0": np.ones(4)})
    monkeypatch.setattr(
        evaluation,
        "compare_tracks",
        lambda *args, **kwargs: {
            "f0_corr": 1.0,
            "f0_centered_rmse_semitones": 0.0,
            "vuv_error": 0.0,
            "joint_voiced_fraction": 1.0,
            "energy_corr": 1.0,
            "energy_log_mae": 0.0,
        },
    )
    return output, resource, model, vocos


def test_generate_score_resume_and_reports(endpoint_runtime, monkeypatch):
    """真实执行文件账本与 JVP，重复评估不重新生成或解码已完成样本。"""
    output, _, model, vocos = endpoint_runtime
    evaluation.run(output, "evaluate", torch.device("cpu"))
    directory = output / "evaluation"
    generated = evaluation._rows(directory / "generation.jsonl")
    diagnostic = read_csv(directory / "endpoint_diagnostics.csv")
    metrics = read_csv(directory / "per_sample_metrics.csv")
    assert len(generated) == len(metrics) == 768
    assert len(diagnostic) == 1536
    assert (
        len({(r["cohort"], r["experiment_id"], r["source_id"], r["repeat"]) for r in generated})
        == 768
    )
    full = next(row for row in diagnostic if row["interval"] == "full")
    assert float(full["jvp_rms"]) > 0
    assert float(full["reconstruction_l1"]) > 0
    assert all(float(row["jvp_rms"]) == 0 for row in diagnostic if row["interval"] == "zero")
    assert all(float(row["cer"]) == 0 for row in metrics)
    assert all(float(row["target_preference_gap"]) == 0 for row in metrics)
    assert all(Path(row["generated_path"]).is_file() for row in generated)
    complete = json.loads((directory / "EVALUATION_COMPLETE.json").read_text(encoding="utf-8"))
    assert complete["records"] == 768 and complete["noise_repeats"] == 2

    sample = Mock(wraps=model.method.sample)
    decode = Mock(wraps=vocos.decode)
    monkeypatch.setattr(model.method, "sample", sample)
    monkeypatch.setattr(vocos, "decode", decode)
    evaluation.run(output, "evaluate", torch.device("cpu"))
    sample.assert_not_called()
    decode.assert_not_called()
    assert len(evaluation._rows(directory / "generation.jsonl")) == 768

    reporting.evaluation_reports(output)
    paired = read_csv(directory / "paired_comparisons.csv")
    assert len(paired) == 2 * 3 * len(reporting.QUALITY_METRICS)
    assert all(int(row["n_sources"]) == 64 for row in paired)
    assert all(float(row["mean"]) == pytest.approx(0) for row in paired)
    samples = read_csv(directory / "control_intervention_per_sample.csv")
    assert len(samples) == 256
    assert all(float(row["cer_delta"]) == 0 for row in samples)
    endpoint_pairs = read_csv(directory / "endpoint_paired_comparisons.csv")
    assert not any(
        row["interval"] == "zero" and row["metric"] == "reconstruction_l1"
        for row in endpoint_pairs
    )
    zero = next(
        row for row in read_csv(directory / "endpoint_summary.csv") if row["interval"] == "zero"
    )
    assert "reconstruction_l1_mean" not in zero or zero["reconstruction_l1_mean"] == ""


def test_generate_rejects_resource_drift(endpoint_runtime):
    """资源指纹不匹配时，在模型推理和生成账本写入前拒绝运行。"""
    output, resource, _, _ = endpoint_runtime
    resource.write_bytes(b"changed")
    with pytest.raises(AssertionError):
        evaluation.generate(output, torch.device("cpu"))
    assert not (output / "evaluation/generation.jsonl").exists()


def test_training_statistics_ignore_missing_endpoint_measurements():
    """控制臂没有完整端点观测时，不能把缺失值计为零。"""
    assert reporting._statistics([None, None]) == {
        "n": 0,
        "mean": None,
        "p50": None,
        "p95": None,
        "max": None,
    }
    result = reporting._statistics([None, 1.0, 3.0])
    assert result["n"] == 2 and result["mean"] == result["p50"] == 2
