"""模块说明：验证基线实验报告、配置一致性、清单和权重选择。"""

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
from src.analysis.unified import identity
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


def test_baseline_report_writes_real_metrics_manifest_and_plan(audit):
    """从真实事件日志生成六种输出，并核对清单指纹与数据划分。"""
    assert set(audit.paths) == {
        "report",
        "metrics",
        "conditions",
        "manifest",
        "dataset",
        "plan",
    }
    assert all(path.is_file() and path.parent == audit.output for path in audit.paths.values())
    metrics = _read_csv(audit.paths["metrics"])
    assert len(metrics) == 9
    assert {row["experiment"] for row in metrics} == {"B1", "B2", "B3"}
    for row in metrics:
        label = row["experiment"]
        assert Path(row["run_dir"]).name == f"2026-02-{label[-1]}_{label}"
    val = next(row for row in metrics if row["metric"] == "val/loss")
    assert [float(val[key]) for key in ("count", "best", "best_step", "final", "final_step")] == [
        3,
        2,
        1,
        3,
        2,
    ]
    train = next(row for row in metrics if row["metric"] == "train/loss")
    assert float(train["best"]) == 4.0
    conditions = _read_csv(audit.paths["conditions"])
    assert len(conditions) == len(baseline.CONTROLLED_CONFIG_PATHS)
    assert all(row["consistent"] == "True" for row in conditions)
    speakers = next(
        row for row in conditions if row["config_path"].endswith("evaluation_speaker_ids")
    )
    assert json.loads(speakers["B1"]) == ["p200", "p300"]
    manifest = _read_csv(audit.paths["manifest"])
    assert [row["source_id"] for row in manifest] == [
        "p200_005",
        "p200_006",
        "p300_005",
        "p300_006",
    ]
    digest = hashlib.sha256()
    for index, row in enumerate(manifest):
        assert row["source_speaker_id"] != row["reference_speaker_id"]
        assert row["reference_id"] == manifest[(index + 2) % 4]["source_id"]
        assert row["text"] == f"测试，句子 {int(row['sentence_id'])}"
        assert (audit.root / row["audio_path"]).is_file()
        digest.update(
            json.dumps(
                {**row, "index": int(row["index"])}, ensure_ascii=False, sort_keys=True
            ).encode("utf-8")
        )
        digest.update(b"\n")
    summary = json.loads(audit.paths["dataset"].read_text(encoding="utf-8"))
    assert [
        summary[key]
        for key in (
            "discovered_samples",
            "train_samples",
            "validation_samples",
            "test_samples",
            "test_speakers",
            "test_sentences",
        )
    ] == [8, 2, 2, 4, 2, 2]
    assert summary["test_manifest_sha256"] == digest.hexdigest()
    assert summary["metadata_modified"] is False
    plan = _read_csv(audit.paths["plan"])
    assert len(plan) == 7
    assert [spec.nfe for spec in unified._load_specs(audit.paths["plan"])] == [
        1,
        5,
        10,
        50,
        1,
        1,
    ]
    assert next(row for row in plan if row["status"] == "待训练")["run_dir"] == "需单独训练"
    report = audit.paths["report"].read_text(encoding="utf-8")
    assert "关键参数一致" in report
    assert "训练 2 / 验证 2 / 测试 4" in report
    assert "2.000000" in report and "1.500000" in report
    assert digest.hexdigest() in report
    assert "尚不能证明 MeanFlow 更适合 VC" in report
    before = audit.paths["manifest"].read_bytes()
    baseline.analyze_baselines(audit.runs, audit.root, audit.output)
    assert audit.paths["manifest"].read_bytes() == before


def test_baseline_report_exposes_non_split_config_mismatch(evaluation_data):
    """非划分条件不一致时仍生成报告，但明确指出差异。"""
    path = evaluation_data.runs / "2026-02-2_B2" / ".hydra" / "config.yaml"
    config = deepcopy(evaluation_data.config)
    config["seed"] = 99
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    paths = baseline.analyze_baselines(
        evaluation_data.runs, evaluation_data.root, evaluation_data.output
    )
    row = next(row for row in _read_csv(paths["conditions"]) if row["config_path"] == "seed")
    assert (row["B1"], row["B2"], row["B3"], row["consistent"]) == (
        "7",
        "99",
        "7",
        "False",
    )
    assert "以下条件不一致：`seed`" in paths["report"].read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "key,value",
    [("seed", 99), ("evaluation_sentence_ids", ["005"]), ("val_ratio", 0.25)],
)
def test_baseline_rejects_inconsistent_split(evaluation_data, key, value):
    """划分参数不一致时拒绝生成误导性的测试清单。"""
    path = evaluation_data.runs / "2026-02-2_B2" / ".hydra" / "config.yaml"
    config = deepcopy(evaluation_data.config)
    config["data"]["split_strategy"][key] = value
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match=f"数据划分配置不一致: {key}"):
        baseline.analyze_baselines(
            evaluation_data.runs, evaluation_data.root, evaluation_data.output
        )
    assert not (evaluation_data.output / "test_manifest.csv").exists()


def test_baseline_missing_runs_and_dataset(tmp_path):
    """缺失实验和数据目录时给出可定位错误。"""
    runs = tmp_path / "runs"
    runs.mkdir()
    with pytest.raises(FileNotFoundError, match="B1, B2, B3"):
        baseline.analyze_baselines(runs, tmp_path, tmp_path / "out")
    with pytest.raises(FileNotFoundError, match="未找到 VCTK 目录"):
        baseline._resolve_vctk_root(tmp_path)


@pytest.mark.parametrize("tags", ["{}", "['valid', 1]", "'not a list'"])
def test_baseline_rejects_invalid_tags(tmp_path, tags):
    """标签必须是纯字符串列表。"""
    (tmp_path / "tags.log").write_text(tags, encoding="utf-8")
    with pytest.raises(ValueError, match="无效 tags.log"):
        baseline._read_tags(tmp_path)


@pytest.mark.parametrize("reader", [baseline._read_config, unified._model_config])
def test_reports_reject_non_mapping_config(tmp_path, reader):
    """拒绝不能作为 Hydra 配置映射的 YAML。"""
    (tmp_path / ".hydra").mkdir()
    (tmp_path / ".hydra" / "config.yaml").write_text("- invalid\n", encoding="utf-8")
    with pytest.raises(ValueError, match="无效 Hydra 配置"):
        reader(tmp_path)


@pytest.mark.parametrize("count", [0, 1, 2])
def test_checkpoint_selection_ignores_last_and_rejects_ambiguity(tmp_path, count):
    """审计展示排序后的名称，统一评估只接受唯一最佳检查点。"""
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "last.ckpt").touch()
    for index in range(count):
        (checkpoints / f"epoch_{index}.ckpt").touch()
    assert baseline._checkpoint_name(tmp_path) == (
        f"epoch_{count - 1}.ckpt" if count else "未找到"
    )
    if count == 1:
        assert unified._best_checkpoint(tmp_path) == checkpoints / "epoch_0.ckpt"
    else:
        with pytest.raises(ValueError, match=f"实际找到 {count} 个"):
            unified._best_checkpoint(tmp_path)
