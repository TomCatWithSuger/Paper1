"""模块说明：验证统一评估的音频指标、账本恢复、缓存边界与运行身份。"""

# ====================
# 1. Imports
# ====================

import csv
import hashlib
import json
import math
import socket
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml
from torch.utils.tensorboard.writer import SummaryWriter

from src.analysis import baseline_validation as baseline
from src.analysis import unified_vc_evaluation as unified
from src.analysis import vocoder_evaluation as vocoder
from src.analysis.vocoder import audio as vocoder_audio
from src.analysis.vocoder import models as vocoder_models
from tests.helpers.evaluation_reports import _evaluate as _evaluate
from tests.helpers.evaluation_reports import _read_csv as _read_csv
from tests.helpers.evaluation_reports import _run_dir_for as _run_dir_for
from tests.helpers.evaluation_reports import _SyntheticMethod as _SyntheticMethod
from tests.helpers.evaluation_reports import _write_csv as _write_csv
from tests.helpers.evaluation_reports import _write_identity as _write_identity
from tests.helpers.evaluation_reports import _write_run as _write_run
from tests.helpers.evaluation_reports import audit as _audit_fixture
from tests.helpers.evaluation_reports import evaluation_data as _evaluation_data_fixture
from tests.helpers.evaluation_reports import model_boundary as _model_boundary_fixture
from tests.helpers.evaluation_reports import (
    offline_boundaries as _offline_boundaries_fixture,
)

# ====================
# 6. Public API
# ====================


def test_unified_real_audio_metrics_outputs_and_resume(audit, model_boundary, tmp_path):
    """真实读取音频和缓存，生成 Mel、逐样本指标、聚合报告并断点续跑。"""
    output = tmp_path / "evaluation"
    paths = _evaluate(
        audit,
        output,
        base_seed=100,
        limit=2,
        experiment_ids=["nfe_b3_01", "nfe_b1_05"],
    )
    rows = _read_csv(paths["per_sample"])
    assert len(rows) == 4
    assert [(row["experiment_id"], int(row["sample_index"])) for row in rows] == [
        ("nfe_b1_05", 0),
        ("nfe_b1_05", 1),
        ("nfe_b3_01", 0),
        ("nfe_b3_01", 1),
    ]
    assert [method.calls for _, _, method in model_boundary] == [
        [(99, 5), (100, 5), (101, 5)],
        [(99, 1), (100, 1), (101, 1)],
    ]
    for row in rows:
        offset = 1.0 if row["model"] == "B1" else 3.0
        assert float(row["mel_l1"]) == pytest.approx(offset)
        assert float(row["mel_l2_rmse"]) == pytest.approx(offset)
        assert float(row["mel_mse"]) == pytest.approx(offset**2)
        assert int(row["seed"]) == 100 + int(row["sample_index"])
        assert int(row["source_frames"]) == (64 + 16 * int(row["sentence_id"])) // 16 + 1
        assert int(row["target_frames"]) == int(row["source_frames"])
        duration = (64 + 16 * int(row["sentence_id"])) / 8000
        assert float(row["audio_seconds"]) == duration
        assert float(row["latency_ms"]) >= 0
        assert float(row["rtf"]) == pytest.approx(float(row["latency_ms"]) / 1000 / duration)
        saved = torch.load(row["generated_path"], weights_only=True)
        assert saved["mel"].dtype == torch.float16
        assert saved["mel"].shape == (2, int(row["source_frames"]))
        assert torch.allclose(
            saved["mel"].float(),
            torch.full_like(saved["mel"].float(), math.log(1e-5) + offset),
            atol=0.005,
        )
        assert saved["source_id"] == row["source_id"]
        assert saved["reference_id"] == row["reference_id"]
        assert saved["seed"] == int(row["seed"])
        assert saved["nfe"] == int(row["nfe"])
    aggregates = _read_csv(paths["aggregate"])
    assert [int(row["sample_count"]) for row in aggregates] == [2, 2]
    assert [float(row["mel_l1_mean"]) for row in aggregates] == pytest.approx([1, 3])
    assert _read_csv(paths["differences"]) == []
    report = paths["report"].read_text(encoding="utf-8")
    assert "完整 100 组结果尚未产生" in report
    assert "| nfe_b1_05 | B1 | 5 | 2 | 1.000000 |" in report
    before = paths["ledger"].read_bytes()
    _evaluate(
        audit,
        output,
        base_seed=100,
        limit=2,
        experiment_ids=["nfe_b3_01", "nfe_b1_05"],
    )
    assert paths["ledger"].read_bytes() == before
    assert all(not method.calls for _, _, method in model_boundary[2:])
    _evaluate(audit, output, base_seed=100, experiment_ids=["nfe_b1_05"])
    assert model_boundary[-1][2].calls == [(99, 5), (102, 5), (103, 5)]
    assert len(unified._load_ledger(paths["ledger"])) == 6
    assert len(_read_csv(paths["per_sample"])) == 4
    _evaluate(audit, output, base_seed=100, limit=1, experiment_ids=["nfe_b3_01"])
    selected = _read_csv(paths["per_sample"])
    assert len(selected) == 1 and selected[0]["model"] == "B3"
    assert len(unified._load_ledger(paths["ledger"])) == 6


def test_unified_complete_ledger_selects_best_nfe_and_difference_categories(
    audit, model_boundary, tmp_path
):
    """完整账本驱动真实报告，比较最佳 B1 的一百条配对差异。"""
    manifest = []
    records = []
    for index in range(100):
        source_id = f"p200_{index:03d}"
        manifest.append(
            {
                "index": index,
                "source_id": source_id,
                "reference_id": "p300_005",
                "source_speaker_id": "p200",
                "reference_speaker_id": "p300",
                "sentence_id": f"{index:03d}",
                "text": "合成样本",
            }
        )
        text_dir = audit.root / "txt" / "p200"
        (text_dir / f"{source_id}.txt").write_text("合成样本", encoding="utf-8")
        (audit.root / "wav48_silence_trimmed" / "p200" / f"{source_id}_mic1.wav").touch()
        for kind in ("content", "speaker"):
            (audit.cache / kind / f"{source_id}.pt").touch()
        for experiment, model, nfe, error in (
            ("nfe_b1_01", "B1", 1, 9.0),
            ("nfe_b1_50", "B1", 50, 2.0),
            ("nfe_b2_01", "B2", 1, (1.0, 3.0, 2.0)[index % 3]),
        ):
            records.append(
                {
                    **dict.fromkeys(unified.PER_SAMPLE_FIELDS, 0),
                    **{
                        key: value
                        for key, value in manifest[-1].items()
                        if key not in {"index", "text"}
                    },
                    "sample_index": index,
                    "experiment_id": experiment,
                    "model": model,
                    "nfe": nfe,
                    "text_length": 4,
                    "mel_l1": error,
                    "mel_l2_rmse": error * 2,
                    "mel_mse": error**2,
                    "latency_ms": index + 1,
                    "rtf": (index + 1) / 1000,
                    "generated_path": "",
                }
            )
    _write_csv(audit.paths["manifest"], manifest)
    output = tmp_path / "complete"
    output.mkdir()
    ledger = output / "evaluation_records.jsonl"
    ledger.write_text(
        "\n[]\n"
        + "\n".join(json.dumps(row, ensure_ascii=False) for row in reversed(records))
        + "\n",
        encoding="utf-8",
    )
    before = ledger.read_bytes()
    _write_identity(audit, output, ["nfe_b1_01", "nfe_b1_50", "nfe_b2_01"])
    paths = _evaluate(audit, output, experiment_ids=["nfe_b1_01", "nfe_b1_50", "nfe_b2_01"])
    assert ledger.read_bytes() == before
    assert all(not method.calls for _, _, method in model_boundary)
    rows = _read_csv(paths["per_sample"])
    assert len(rows) == 300
    assert [int(row["sample_index"]) for row in rows[:100]] == list(range(100))
    aggregates = {row["experiment_id"]: row for row in _read_csv(paths["aggregate"])}
    assert all(int(row["sample_count"]) == 100 for row in aggregates.values())
    b2 = aggregates["nfe_b2_01"]
    assert float(b2["mel_l1_mean"]) == pytest.approx(1.99)
    assert float(b2["mel_l1_median"]) == 2.0
    assert float(b2["mel_l2_rmse_mean"]) == pytest.approx(3.98)
    assert float(b2["latency_ms_mean"]) == 50.5
    assert float(b2["rtf_mean"]) == pytest.approx(0.0505)
    differences = _read_csv(paths["differences"])
    assert len(differences) == 100
    assert {row["b1_experiment_id"] for row in differences} == {"nfe_b1_50"}
    assert [row["category"] for row in differences[:3]] == ["提升", "下降", "不变"]
    assert [float(row["delta_mel_l1"]) for row in differences[:3]] == [-1, 1, 0]
    assert [float(row["delta_mel_l2_rmse"]) for row in differences[:3]] == [-2, 2, 0]
    report = paths["report"].read_text(encoding="utf-8")
    assert "B1 基线为 `nfe_b1_50`" in report
    assert "提升 34，下降 33，不变 33" in report
    assert "不能替代 CER/WER" in report


@pytest.mark.parametrize("limit", [0, -1])
def test_unified_rejects_nonpositive_limit_before_model_loading(audit, tmp_path, limit):
    """无效样本数在加载模型前被拒绝。"""
    with pytest.raises(ValueError, match="limit 必须为正数"):
        _evaluate(audit, tmp_path / "out", limit=limit)


def test_unified_rejects_unknown_or_nonrunnable_experiments(audit, tmp_path):
    """未知实验与待训练消融均不能进入评估。"""
    with pytest.raises(ValueError, match="未知或不可运行实验: ablation_b2_equal_time_00, missing"):
        _evaluate(
            audit,
            tmp_path / "out",
            experiment_ids=["missing", "ablation_b2_equal_time_00"],
        )


def test_unified_rejects_plan_without_pending_experiments(audit, tmp_path):
    """计划只包含待训练项目时拒绝继续。"""
    rows = [row for row in _read_csv(audit.paths["plan"]) if row["status"] == "待训练"]
    _write_csv(audit.paths["plan"], rows)
    with pytest.raises(ValueError, match="实验计划中没有待评估项目"):
        _evaluate(audit, tmp_path / "out")


def test_unified_reports_missing_manifest_ids_once(audit, tmp_path):
    """对源与参考缺失项去重并排序后报错。"""
    _write_csv(
        audit.paths["manifest"],
        [{"source_id": "z_missing", "reference_id": "a_missing"}] * 2,
    )
    with pytest.raises(FileNotFoundError, match="测试清单样本不存在: a_missing, z_missing$"):
        _evaluate(audit, tmp_path / "out")


def test_unified_validates_cache_only_for_selected_samples(audit, model_boundary, tmp_path):
    """限制样本数只检查所选配对缓存，完整评估则报告缺失。"""
    manifest = _read_csv(audit.paths["manifest"])
    assert manifest[0]["reference_id"] == "p300_005"
    cache_file = audit.cache / "speaker" / "p300_006.pt"
    cache_file.unlink()
    paths = _evaluate(audit, tmp_path / "one", limit=1, experiment_ids=["nfe_b2_01"])
    assert len(_read_csv(paths["per_sample"])) == 1
    with pytest.raises(FileNotFoundError, match="缺少 1 个缓存文件") as error:
        _evaluate(audit, tmp_path / "all", experiment_ids=["nfe_b2_01"])
    assert "speaker/p300_006.pt" in str(error.value)
    assert len(model_boundary) == 1


def test_missing_cache_preview_is_bounded(tmp_path):
    """大量缓存缺失时只列出前二十项与剩余计数。"""
    rows = [{"source_id": f"s{index:02d}", "reference_id": "reference"} for index in range(11)]
    with pytest.raises(FileNotFoundError, match="缺少 24 个缓存文件") as error:
        unified._validate_feature_cache(rows, tmp_path)
    message = str(error.value)
    assert len([line for line in message.splitlines() if line.startswith("- ")]) == 21
    assert "其余 4 个" in message
    assert "content/reference.pt" in message
    assert "speaker/s10.pt" not in message


def test_unified_rejects_unavailable_cuda_before_reading_inputs(audit, monkeypatch, tmp_path):
    """显式请求不可用 CUDA 时立即失败。"""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="当前 PyTorch 环境不可用 CUDA"):
        _evaluate(audit, tmp_path / "out", device="cuda")


def test_prepare_pair_rejects_same_speaker_fixed_reference(audit):
    """同说话人清单不能通过跨说话人配对流程。"""
    rows, by_id = unified._load_manifest_samples(audit.paths["manifest"], audit.root, "mic1")
    row = {**rows[0], "reference_id": "p200_006"}
    with pytest.raises(ValueError, match="reference pairing is unavailable"):
        unified._prepare_pair(
            row, by_id, audit.config, audit.cache, unified._collator(audit.config)
        )


@pytest.mark.parametrize("change", ["checkpoint", "reference", "nfe", "seed", "mel", "cache"])
def test_unified_identity_rejects_changed_parameters(audit, model_boundary, tmp_path, change):
    """修改 checkpoint、reference、NFE、seed、Mel 或缓存身份后续跑被拒绝。"""
    output = tmp_path / "evaluation"
    _evaluate(audit, output, base_seed=100, limit=1, experiment_ids=["nfe_b2_01"])
    if change == "seed":
        with pytest.raises(ValueError, match="评估身份不一致"):
            _evaluate(audit, output, base_seed=999, limit=1, experiment_ids=["nfe_b2_01"])
        return
    run_dir = _run_dir_for(audit, "nfe_b2_01")
    if change == "checkpoint":
        (run_dir / "checkpoints" / "epoch_002.ckpt").write_bytes(b"changed")
    elif change == "reference":
        rows = _read_csv(audit.paths["manifest"])
        rows[0]["reference_id"] = "p300_006"
        _write_csv(audit.paths["manifest"], rows)
    elif change == "nfe":
        rows = _read_csv(audit.paths["plan"])
        for row in rows:
            if row["experiment_id"] == "nfe_b2_01":
                row["nfe"] = "7"
        _write_csv(audit.paths["plan"], rows)
    elif change == "mel":
        config_path = run_dir / ".hydra" / "config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["data"]["n_mels"] = 99
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    elif change == "cache":
        (audit.cache / "content" / "p300_005.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="评估身份不一致"):
        _evaluate(audit, output, base_seed=100, limit=1, experiment_ids=["nfe_b2_01"])


def test_unified_resume_expands_limit_without_recomputing(audit, model_boundary, tmp_path):
    """Limit 扩展只补齐缺失样本，原账本保持不变且不重复生成。"""
    output = tmp_path / "evaluation"
    _evaluate(audit, output, base_seed=100, limit=1, experiment_ids=["nfe_b2_01"])
    ledger = output / "evaluation_records.jsonl"
    first_bytes = ledger.read_bytes()
    assert len(unified._load_ledger(ledger)) == 1
    total = len(_read_csv(audit.paths["manifest"]))
    _evaluate(audit, output, base_seed=100, experiment_ids=["nfe_b2_01"])
    assert ledger.read_bytes().startswith(first_bytes)
    assert len(unified._load_ledger(ledger)) == total
    # 样本 0（seed=100）未被重复生成，仅补齐后续样本。
    assert 100 not in [seed for seed, _ in model_boundary[-1][2].calls]


def test_unified_aggregation_counts_each_experiment_sample_once(audit, model_boundary, tmp_path):
    """重复账本记录在聚合中被去重，每个实验样本最多出现一次。"""
    output = tmp_path / "evaluation"
    _evaluate(audit, output, base_seed=100, limit=2, experiment_ids=["nfe_b2_01"])
    ledger = output / "evaluation_records.jsonl"
    records = unified._load_ledger(ledger)
    assert len(records) == 2
    unified._append_ledger(ledger, records[0])
    paths = _evaluate(audit, output, base_seed=100, limit=2, experiment_ids=["nfe_b2_01"])
    aggregates = _read_csv(paths["aggregate"])
    assert [int(row["sample_count"]) for row in aggregates] == [2]


def test_unified_resume_registers_appended_experiments(audit, model_boundary, tmp_path):
    output = tmp_path / "evaluation"
    _evaluate(audit, output, limit=1, experiment_ids=["nfe_b1_01"])
    _evaluate(audit, output, limit=1, experiment_ids=["nfe_b2_01"])
    identity = unified._load_identity(output / "evaluation_identity.json")
    assert identity is not None
    assert set(identity["experiments"]) == {"nfe_b1_01", "nfe_b2_01"}
    run_dir = _run_dir_for(audit, "nfe_b2_01")
    (run_dir / "checkpoints" / "epoch_002.ckpt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="评估身份不一致"):
        _evaluate(audit, output, limit=1, experiment_ids=["nfe_b2_01"])


def test_unified_resume_rejects_changed_generation_config(audit, model_boundary, tmp_path):
    output = tmp_path / "evaluation"
    run_dir = _run_dir_for(audit, "nfe_b3_01")
    config_path = run_dir / ".hydra" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["model"]["method"]["inference_source_diffusion_time"] = 0.95
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    _evaluate(audit, output, limit=1, experiment_ids=["nfe_b3_01"])
    config["model"]["method"]["inference_source_diffusion_time"] = 0.1
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="评估身份不一致"):
        _evaluate(audit, output, limit=1, experiment_ids=["nfe_b3_01"])


def test_unified_resume_rejects_records_without_experiment_identity(
    audit, model_boundary, tmp_path
):
    output = tmp_path / "evaluation"
    _evaluate(audit, output, limit=1, experiment_ids=["nfe_b1_01"])
    ledger = output / "evaluation_records.jsonl"
    record = unified._load_ledger(ledger)[0]
    unified._append_ledger(ledger, {**record, "experiment_id": "nfe_b2_01"})
    with pytest.raises(ValueError, match="账本缺少实验身份"):
        _evaluate(audit, output, limit=1, experiment_ids=["nfe_b2_01"])


def test_evaluate_sample_aligns_target_and_excludes_generated_padding(tmp_path, monkeypatch):
    """数值核对插值、有效帧裁剪、耗时和生成文件。"""
    batch = {
        "mel_attention_mask": torch.ones(1, 3, dtype=torch.bool),
        "mel_lengths": torch.tensor([3]),
        "content_features": torch.ones(1, 3, 5),
        "content_feature_lengths": torch.tensor([5]),
        "reference_speaker_features": torch.ones(1, 2),
    }
    target = torch.tensor([[0.0, 2.0], [10.0, 12.0]])
    row = {
        "index": "7",
        "source_id": "source",
        "source_speaker_id": "p200",
        "reference_id": "reference",
        "reference_speaker_id": "p300",
        "sentence_id": "005",
        "text": "合成",
    }
    spec = unified.EvaluationSpec("test", "B1", 5, tmp_path)
    # 超出有效帧的极端值不得进入距离计算或保存结果。
    generated = torch.tensor([[[0.0, 1.0, 2.0, 999.0], [10.0, 11.0, 12.0, -999.0]]])
    model = torch.nn.Module()
    monkeypatch.setattr(
        model,
        "method",
        SimpleNamespace(
            requires_source_mels=False,
            encode_condition=lambda *args: None,
            sample=lambda *args, **kwargs: generated,
        ),
        raising=False,
    )
    ticks = iter([10.0, 10.025])
    monkeypatch.setattr(unified, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    record = unified._evaluate_sample(
        model, batch, target, 800, 8000, spec, row, 123, torch.device("cpu"), tmp_path
    )
    assert record["mel_l1"] == 0 and record["mel_mse"] == 0 and record["mel_l2_rmse"] == 0
    assert (
        record["source_frames"],
        record["target_frames"],
        record["audio_seconds"],
    ) == (
        3,
        2,
        0.1,
    )
    assert record["latency_ms"] == pytest.approx(25.0)
    assert record["rtf"] == pytest.approx(0.25)
    saved = torch.load(record["generated_path"], weights_only=True)
    assert saved["mel"].shape == (2, 3)
    assert saved["mel"].tolist() == [[0, 1, 2], [10, 11, 12]]
