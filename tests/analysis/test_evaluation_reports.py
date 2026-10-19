"""使用临时日志、音频和特征验证审计与统一评估报告，不加载真实模型。"""

# ====================
# 1. 导入
# ====================

import csv
import hashlib
import json
import math
import socket
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

# ====================
# 2. 测试辅助
# ====================


def _read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_run(path: Path, label, config):
    (path / ".hydra").mkdir(parents=True)
    (path / ".hydra" / "config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True), encoding="utf-8"
    )
    (path / "tags.log").write_text(repr([baseline.RUN_LABEL_TAGS[label]]), encoding="utf-8")
    (path / "checkpoints").mkdir()
    (path / "checkpoints" / "last.ckpt").touch()
    (path / "checkpoints" / "epoch_002.ckpt").touch()
    writer = SummaryWriter(str(path / "tensorboard" / "version_0"))
    try:
        for tag, values in {
            "train/loss": [8.0, 4.0, 5.0],
            "val/loss": [6.0, 2.0, 3.0],
            "test/loss": [1.5],
        }.items():
            for step, value in enumerate(values):
                writer.add_scalar(tag, value, global_step=step)
    finally:
        writer.close()


class _SyntheticMethod:
    """提供可核验的模型边界，保留真实数据处理与评估流程。"""

    def __init__(self, requires_source_mels, offset):
        self.requires_source_mels = requires_source_mels
        self.offset = offset
        self.calls = []

    def encode_condition(self, content, lengths, speaker, mask):
        assert content.shape[:2] == (1, 3)
        assert lengths.tolist() == [5]
        assert speaker.shape == (1, 2)
        return torch.full((1, 2, mask.size(1)), math.log(1e-5) + self.offset)

    def sample(self, condition, mask, **kwargs):
        assert torch.is_inference_mode_enabled()
        assert mask.all()
        assert ("source_mels" in kwargs) == self.requires_source_mels
        if self.requires_source_mels:
            assert kwargs["source_mels"].shape == condition.shape
        self.calls.append((torch.initial_seed(), kwargs["integration_steps"]))
        return condition


# ====================
# 3. 临时资源
# ====================


@pytest.fixture(autouse=True)
def offline_boundaries(monkeypatch):
    """禁止测试联网或意外进入真实模型加载。"""

    def forbidden(*args, **kwargs):
        pytest.fail("测试不得联网或加载真实模型")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(unified, "_load_model", forbidden)
    with torch.random.fork_rng(devices=[]):
        yield


@pytest.fixture
def evaluation_data(tmp_path):
    """构造三个实验、八条语音及四条固定测试配对。"""
    config = {
        "seed": 7,
        "data": {
            "microphone": "mic1",
            "sample_rate": 8000,
            "max_duration_seconds": 1.0,
            "n_fft": 32,
            "win_length": 32,
            "hop_length": 16,
            "n_mels": 2,
            "f_min": 0.0,
            "f_max": 4000.0,
            "log_mel_floor": 1e-5,
            "test_reference_mode": "different_speaker",
            "split_strategy": {
                "evaluation_speaker_ids": ["p200", "p300"],
                "evaluation_sentence_ids": ["005", "006"],
                "val_ratio": 0.5,
                "seed": 7,
            },
        },
        "model": {"method": {"condition_encoder": {"content_dim": 3}}},
    }
    runs = tmp_path / "runs"
    for label in baseline.RUN_LABEL_TAGS:
        _write_run(runs / f"2026-02-{label[-1]}_{label}", label, config)
    _write_run(runs / "2026-01-01_old_B1", "B1", {**config, "seed": 999})
    (runs / "unrelated").mkdir()
    (runs / "notes.txt").write_text("忽略非实验文件", encoding="utf-8")
    root = tmp_path / "data" / "VCTK"
    cache = tmp_path / "cache"
    for kind in ("content", "speaker"):
        (cache / kind).mkdir(parents=True)
    for speaker, sentences in (("p100", range(1, 5)), ("p200", (5, 6)), ("p300", (5, 6))):
        for sentence in sentences:
            utterance = f"{speaker}_{sentence:03d}"
            text_dir = root / "txt" / speaker
            audio_dir = root / "wav48_silence_trimmed" / speaker
            text_dir.mkdir(parents=True, exist_ok=True)
            audio_dir.mkdir(parents=True, exist_ok=True)
            (text_dir / f"{utterance}.txt").write_text(
                f"测试，句子 {sentence}\n", encoding="utf-8"
            )
            sf.write(audio_dir / f"{utterance}_mic1.wav", np.zeros(64 + 16 * sentence), 8000)
            torch.save(torch.ones(3, 5), cache / "content" / f"{utterance}.pt")
            torch.save(torch.tensor([1.0, 2.0]), cache / "speaker" / f"{utterance}.pt")
    return SimpleNamespace(
        root=root, runs=runs, cache=cache, config=config, output=tmp_path / "audit"
    )


@pytest.fixture
def audit(evaluation_data):
    paths = baseline.analyze_baselines(
        evaluation_data.runs, evaluation_data.root.parent, evaluation_data.output
    )
    return SimpleNamespace(**vars(evaluation_data), paths=paths)


@pytest.fixture
def model_boundary(monkeypatch):
    loaded = []

    def load(run_dir, device):
        config = unified._model_config(run_dir)
        label = run_dir.name[-2:]
        method = _SyntheticMethod(label == "B3", {"B1": 1.0, "B2": 2.0, "B3": 3.0}[label])
        loaded.append((run_dir, device, method))
        return SimpleNamespace(method=method), config

    monkeypatch.setattr(unified, "_load_model", load)
    return loaded


def _evaluate(audit, output, **kwargs):
    return unified.evaluate_unified(
        audit.paths["plan"], audit.paths["manifest"], audit.root, audit.cache, output, **kwargs
    )


def _write_identity(audit, output, experiment_ids, base_seed=12345, device="cpu"):
    """为预置账本构造与 _execute 一致的评估身份文件。"""
    selected = set(experiment_ids)
    specs = [spec for spec in unified._load_specs(audit.paths["plan"]) if spec.experiment_id in selected]
    rows = unified._read_csv(audit.paths["manifest"])
    identity = unified._build_run_identity(
        specs,
        audit.paths["manifest"],
        rows,
        audit.cache,
        unified._model_config(specs[0].run_dir),
        device,
        base_seed,
    )
    (output / "evaluation_identity.json").write_text(
        json.dumps(identity, ensure_ascii=False), encoding="utf-8"
    )


# ====================
# 4. 基线审计报告
# ====================


def test_baseline_report_writes_real_metrics_manifest_and_plan(audit):
    """从真实事件日志生成六种输出，并核对清单指纹与数据划分。"""
    assert set(audit.paths) == {"report", "metrics", "conditions", "manifest", "dataset", "plan"}
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
    assert [spec.nfe for spec in unified._load_specs(audit.paths["plan"])] == [1, 5, 10, 50, 1, 1]
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
    assert (row["B1"], row["B2"], row["B3"], row["consistent"]) == ("7", "99", "7", "False")
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


# ====================
# 5. 统一评估报告
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
        _evaluate(audit, tmp_path / "out", experiment_ids=["missing", "ablation_b2_equal_time_00"])


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


# ====================
# 断点身份回归
# ====================


def _run_dir_for(audit, experiment_id):
    for spec in unified._load_specs(audit.paths["plan"]):
        if spec.experiment_id == experiment_id:
            return spec.run_dir
    raise AssertionError(experiment_id)


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
    """limit 扩展只补齐缺失样本，原账本保持不变且不重复生成。"""
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
    assert (record["source_frames"], record["target_frames"], record["audio_seconds"]) == (
        3,
        2,
        0.1,
    )
    assert record["latency_ms"] == pytest.approx(25.0)
    assert record["rtf"] == pytest.approx(0.25)
    saved = torch.load(record["generated_path"], weights_only=True)
    assert saved["mel"].shape == (2, 3)
    assert saved["mel"].tolist() == [[0, 1, 2], [10, 11, 12]]
