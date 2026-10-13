"""审计 B1、B2、B3 基线实验并生成可复现实验记录。"""

# ====================
# 1. Imports
# ====================

import ast
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.data.components.vctk_splits import (
    UnseenSpeakerSentenceSplit,
    discover_vctk_samples,
)
from src.data.vctk_datamodule import VCTKDataset

# ====================
# 2. Definitions
# ====================

RUN_LABEL_TAGS = {
    "B1": "voice-conversion-baseline",
    "B2": "voice-conversion-b2",
    "B3": "voice-conversion-b3",
}

CONTROLLED_CONFIG_PATHS = (
    "seed",
    "data.sample_rate",
    "data.max_duration_seconds",
    "data.n_fft",
    "data.win_length",
    "data.hop_length",
    "data.n_mels",
    "data.f_min",
    "data.f_max",
    "data.log_mel_floor",
    "data.test_reference_mode",
    "data.split_strategy.evaluation_speaker_ids",
    "data.split_strategy.evaluation_sentence_ids",
    "data.split_strategy.val_ratio",
    "data.split_strategy.seed",
    "model.method.condition_encoder.content_dim",
    "model.method.condition_encoder.speaker_dim",
    "model.method.condition_encoder.output_dim",
    "model.method.network.n_mels",
    "model.method.network.condition_dim",
    "model.method.network.hidden_channels",
    "model.method.network.time_embedding_dim",
    "model.method.network.kernel_size",
)


@dataclass(frozen=True)
class RunSummary:
    """保存一次实验的配置与 TensorBoard 指标摘要。"""

    label: str
    run_dir: Path
    tags: tuple[str, ...]
    config: dict[str, Any]
    checkpoint: str
    metrics: tuple[dict[str, str | int | float], ...]


# ====================
# 3. Utilities
# ====================


def _nested_value(mapping: dict[str, Any], dotted_path: str) -> Any:
    value: Any = mapping
    for key in dotted_path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _format_metric(value: Any) -> str:
    return "—" if value is None else f"{float(value):.6f}"


def _read_tags(run_dir: Path) -> tuple[str, ...]:
    value = ast.literal_eval((run_dir / "tags.log").read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(tag, str) for tag in value):
        raise ValueError(f"无效 tags.log: {run_dir}")
    return tuple(value)


def _read_config(run_dir: Path) -> dict[str, Any]:
    value = yaml.safe_load((run_dir / ".hydra" / "config.yaml").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"无效 Hydra 配置: {run_dir}")
    return value


def _checkpoint_name(run_dir: Path) -> str:
    checkpoints = sorted(
        path.name for path in (run_dir / "checkpoints").glob("*.ckpt") if path.name != "last.ckpt"
    )
    return checkpoints[-1] if checkpoints else "未找到"


def _scalar_metrics(run_dir: Path) -> tuple[dict[str, str | int | float], ...]:
    event_dir = run_dir / "tensorboard" / "version_0"
    accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    rows: list[dict[str, str | int | float]] = []
    for metric in ("train/loss", "val/loss", "test/loss"):
        points = accumulator.Scalars(metric)
        if not points:
            continue
        best = min(points, key=lambda point: point.value)
        final = points[-1]
        rows.append(
            {
                "metric": metric,
                "count": len(points),
                "best": float(best.value),
                "best_step": int(best.step),
                "final": float(final.value),
                "final_step": int(final.step),
            }
        )
    return tuple(rows)


def _metric(summary: RunSummary, name: str, field: str) -> float | int | None:
    for row in summary.metrics:
        if row["metric"] == name:
            value = row[field]
            return value if isinstance(value, (float, int)) else None
    return None


# ====================
# 4. Internal Functions
# ====================


def _discover_runs(runs_root: Path) -> list[RunSummary]:
    selected: dict[str, Path] = {}
    for run_dir in sorted(runs_root.iterdir(), reverse=True):
        if not run_dir.is_dir() or not (run_dir / "tags.log").is_file():
            continue
        tags = _read_tags(run_dir)
        for label, identifying_tag in RUN_LABEL_TAGS.items():
            if label not in selected and identifying_tag in tags:
                selected[label] = run_dir
    missing = [label for label in RUN_LABEL_TAGS if label not in selected]
    if missing:
        raise FileNotFoundError(f"未找到实验记录: {', '.join(missing)}")

    summaries = []
    for label in RUN_LABEL_TAGS:
        run_dir = selected[label]
        summaries.append(
            RunSummary(
                label=label,
                run_dir=run_dir,
                tags=_read_tags(run_dir),
                config=_read_config(run_dir),
                checkpoint=_checkpoint_name(run_dir),
                metrics=_scalar_metrics(run_dir),
            )
        )
    return summaries


def _resolve_vctk_root(data_root: Path) -> Path:
    candidates = (data_root, data_root / "VCTK")
    for candidate in candidates:
        if (candidate / "txt").is_dir() and (candidate / "wav48_silence_trimmed").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"未找到 VCTK 目录；已检查: {', '.join(str(path) for path in candidates)}"
    )


def _shared_split_config(runs: Sequence[RunSummary]) -> dict[str, Any]:
    keys = (
        "evaluation_speaker_ids",
        "evaluation_sentence_ids",
        "val_ratio",
        "seed",
    )
    values = {
        key: [_nested_value(run.config, f"data.split_strategy.{key}") for run in runs]
        for key in keys
    }
    mismatched = [
        key for key, items in values.items() if any(item != items[0] for item in items[1:])
    ]
    if mismatched:
        raise ValueError(f"B1/B2/B3 数据划分配置不一致: {', '.join(mismatched)}")
    return {key: items[0] for key, items in values.items()}


def _build_test_manifest(
    vctk_root: Path,
    runs: Sequence[RunSummary],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split_config = _shared_split_config(runs)
    microphone = _nested_value(runs[0].config, "data.microphone")
    samples = discover_vctk_samples(vctk_root, str(microphone))
    strategy = UnseenSpeakerSentenceSplit(**split_config)
    split = strategy.split(samples)
    dataset = VCTKDataset(
        samples=split.test,
        sample_rate=int(_nested_value(runs[0].config, "data.sample_rate")),
        max_duration_seconds=float(_nested_value(runs[0].config, "data.max_duration_seconds")),
        randomize=False,
        reference_mode="different_speaker",
    )

    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for index, source in enumerate(dataset.samples):
        reference = dataset.samples[dataset._reference_index(index, source.speaker_id)]
        relative_audio = source.audio_path.relative_to(vctk_root)
        row = {
            "index": index,
            "source_id": source.utterance_id,
            "source_speaker_id": source.speaker_id,
            "sentence_id": source.sentence_id,
            "reference_id": reference.utterance_id,
            "reference_speaker_id": reference.speaker_id,
            "text": source.text,
            "audio_path": str(relative_audio),
        }
        rows.append(row)
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")

    dataset_summary = {
        "data_root_argument": str(vctk_root.parent),
        "resolved_vctk_root": str(vctk_root),
        "discovered_samples": len(samples),
        "train_samples": len(split.train),
        "validation_samples": len(split.val),
        "test_samples": len(split.test),
        "test_speakers": len({sample.speaker_id for sample in split.test}),
        "test_sentences": len({sample.sentence_id for sample in split.test}),
        "test_manifest_sha256": digest.hexdigest(),
        "metadata_modified": False,
    }
    return rows, dataset_summary


def _controlled_rows(runs: Sequence[RunSummary]) -> list[dict[str, Any]]:
    rows = []
    for config_path in CONTROLLED_CONFIG_PATHS:
        values = [_nested_value(run.config, config_path) for run in runs]
        rows.append(
            {
                "config_path": config_path,
                "B1": _format_value(values[0]),
                "B2": _format_value(values[1]),
                "B3": _format_value(values[2]),
                "consistent": values[0] == values[1] == values[2],
            }
        )
    return rows


def _experiment_plan_rows(runs: Sequence[RunSummary]) -> list[dict[str, Any]]:
    run_paths = {run.label: str(run.run_dir) for run in runs}
    return [
        {
            "experiment_id": "nfe_b1_01",
            "model": "B1",
            "nfe": 1,
            "ablation": "无",
            "run_dir": run_paths["B1"],
            "status": "待运行统一转换评估",
        },
        {
            "experiment_id": "nfe_b1_05",
            "model": "B1",
            "nfe": 5,
            "ablation": "无",
            "run_dir": run_paths["B1"],
            "status": "待运行统一转换评估",
        },
        {
            "experiment_id": "nfe_b1_10",
            "model": "B1",
            "nfe": 10,
            "ablation": "无",
            "run_dir": run_paths["B1"],
            "status": "待运行统一转换评估",
        },
        {
            "experiment_id": "nfe_b1_50",
            "model": "B1",
            "nfe": 50,
            "ablation": "无",
            "run_dir": run_paths["B1"],
            "status": "待运行统一转换评估",
        },
        {
            "experiment_id": "nfe_b2_01",
            "model": "B2",
            "nfe": 1,
            "ablation": "equal_time_probability=0.75",
            "run_dir": run_paths["B2"],
            "status": "待运行统一转换评估",
        },
        {
            "experiment_id": "ablation_b2_equal_time_00",
            "model": "B2",
            "nfe": 1,
            "ablation": "equal_time_probability=0.0",
            "run_dir": "需单独训练",
            "status": "待训练",
        },
        {
            "experiment_id": "nfe_b3_01",
            "model": "B3",
            "nfe": 1,
            "ablation": "当前完整方法",
            "run_dir": run_paths["B3"],
            "status": "待运行统一转换评估",
        },
    ]


def _report_text(
    runs: Sequence[RunSummary],
    dataset_summary: dict[str, Any],
    controlled_rows: Sequence[dict[str, Any]],
) -> str:
    inconsistent = [row["config_path"] for row in controlled_rows if not row["consistent"]]
    summaries = {run.label: run for run in runs}
    lines = [
        "# B1/B2/B3 基线验证审计",
        "",
        f"> 生成日期：{date.today().isoformat()}",
        "",
        "## 1. 当前结论",
        "",
        "现有记录只能证明三种训练目标都完成了训练，**尚不能证明 MeanFlow 更适合 VC**。",
        "`test/loss` 是各自训练目标的自重构损失，不是跨说话人转换质量；B3 的损失定义与 B1/B2 不同，三者数值不能直接排序。",
        "在完成固定测试清单上的统一生成、质量指标和效率计时前，不建议设计第四个模型。",
        "",
        "## 2. 已定位实验",
        "",
        "| 实验 | 运行目录 | 最优检查点 | 最优 val/loss | test/loss |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for run in runs:
        lines.append(
            "| "
            + " | ".join(
                (
                    run.label,
                    str(run.run_dir),
                    run.checkpoint,
                    _format_metric(_metric(run, "val/loss", "best")),
                    _format_metric(_metric(run, "test/loss", "final")),
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "### 训练记录解读",
            "",
            f"- B1 的最优 `val/loss` 为 {_format_metric(_metric(summaries['B1'], 'val/loss', 'best'))}，最佳检查点为 `{summaries['B1'].checkpoint}`。",
            f"- B2 的最优 `val/loss` 为 {_format_metric(_metric(summaries['B2'], 'val/loss', 'best'))}，最佳检查点测试损失为 {_format_metric(_metric(summaries['B2'], 'test/loss', 'final'))}；验证/测试尺度差异较大，统一转换评估前应先复核固定采样和数据分布。",
            f"- B3 的最优 `val/loss` 为 {_format_metric(_metric(summaries['B3'], 'val/loss', 'best'))}；其自适应损失和零输入下界使该数值不能与 B1/B2 的 masked MSE 比较。",
            "",
            "## 3. 数据集适配与固定测试清单",
            "",
            f"- 输入数据根目录：`{dataset_summary['data_root_argument']}`",
            f"- 自动适配后的 VCTK 根目录：`{dataset_summary['resolved_vctk_root']}`",
            f"- 发现样本：{dataset_summary['discovered_samples']}",
            f"- 划分：训练 {dataset_summary['train_samples']} / 验证 {dataset_summary['validation_samples']} / 测试 {dataset_summary['test_samples']}",
            f"- 测试集合：{dataset_summary['test_speakers']} 个说话人 × {dataset_summary['test_sentences']} 个句子",
            f"- 固定测试清单 SHA256：`{dataset_summary['test_manifest_sha256']}`",
            "- 数据适配只解析现有目录结构并生成外部清单，未修改数据集元数据。",
            "",
            "## 4. 受控条件审计",
            "",
        ]
    )
    if inconsistent:
        lines.append("以下条件不一致：" + "、".join(f"`{item}`" for item in inconsistent) + "。")
    else:
        lines.append("数据划分、Mel 参数、条件维度和 U-Net 主干关键参数一致。")

    lines.extend(
        [
            "",
            "已确认的非受控差异包括：B3 使用不同 Adam betas、预热/余弦调度和不同损失；B1 未配置固定多次验证采样；B3 验证仍为随机单次估计。",
            "因此现有最佳 checkpoint 选择过程并不完全等价，后续统一评估必须直接加载固定 checkpoint，而不能比较训练曲线高低。",
            "",
            "## 5. 指标完成状态",
            "",
            "| 类别 | 指标 | 当前状态 | 下一步 |",
            "| --- | --- | --- | --- |",
            "| 内容保持 | CER / WER | 未完成 | 接入与统一声码器输出匹配的 ASR |",
            "| 内容保持 | 内容 embedding 距离 | 未完成 | 对生成音频重新提取内容特征 |",
            "| 说话人保持 | ECAPA/WavLM cosine | 未完成 | 对生成音频重新提取说话人特征 |",
            "| 音质 | Mel 距离 | 可先完成 | 固定随机种子后比较生成 Mel 与目标 Mel |",
            "| 音质 | MOS | 未完成 | 客观链稳定后再设计主观实验 |",
            "| 效率 | NFE | 已定义 | B1: 1/5/10/50；B2/B3: 1 |",
            "| 效率 | RTF / latency | 未完成 | 同设备、同 batch、预热后计时 |",
            "",
            "## 6. 下一轮实验顺序",
            "",
            "1. 使用 `test_manifest.csv` 固定全部 source/reference 组合。",
            "2. 对 B1 执行 1、5、10、50 NFE，对 B2/B3 执行 1 NFE，并保存逐样本生成结果和耗时。",
            "3. 首先完成 Mel 距离、NFE、RTF、latency；接入统一声码器后再完成 CER/WER 和生成音频 embedding 指标。",
            "4. 以逐样本结果计算 `ΔCER = CER_B1(50) - CER_B2(1)`，按说话人、句长和内容复杂度分组。",
            "5. 只有当统一结果显示 B2(1 NFE) 接近 B1(50 NFE)，才能主张 MeanFlow 成功压缩生成步数。",
            "6. 时间采样消融需要重新训练 `equal_time_probability=0.0` 的 B2，不能通过修改现有 checkpoint 推断。",
            "",
            "## 7. 输出记录",
            "",
            "- `run_metrics.csv`：训练日志标量摘要。",
            "- `controlled_conditions.csv`：三组配置的一致性审计。",
            "- `test_manifest.csv`：固定 source/reference 测试清单。",
            "- `dataset_summary.json`：数据适配结果和清单指纹。",
            "- `experiment_plan.csv`：下一轮 NFE 与消融实验记录表。",
            "",
        ]
    )
    return "\n".join(lines)


# ====================
# 5. Core Logic
# ====================


def _execute(runs_root: Path, data_root: Path, output_dir: Path) -> dict[str, Path]:
    runs_root = runs_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = _discover_runs(runs_root)
    vctk_root = _resolve_vctk_root(data_root.expanduser().resolve())
    manifest_rows, dataset_summary = _build_test_manifest(vctk_root, runs)
    controlled_rows = _controlled_rows(runs)
    experiment_rows = _experiment_plan_rows(runs)

    metric_rows = [
        {"experiment": run.label, "run_dir": str(run.run_dir), **metric}
        for run in runs
        for metric in run.metrics
    ]
    paths = {
        "report": output_dir / "baseline_validation_report.md",
        "metrics": output_dir / "run_metrics.csv",
        "conditions": output_dir / "controlled_conditions.csv",
        "manifest": output_dir / "test_manifest.csv",
        "dataset": output_dir / "dataset_summary.json",
        "plan": output_dir / "experiment_plan.csv",
    }
    _write_csv(
        paths["metrics"],
        metric_rows,
        ("experiment", "run_dir", "metric", "count", "best", "best_step", "final", "final_step"),
    )
    _write_csv(
        paths["conditions"],
        controlled_rows,
        ("config_path", "B1", "B2", "B3", "consistent"),
    )
    _write_csv(
        paths["manifest"],
        manifest_rows,
        (
            "index",
            "source_id",
            "source_speaker_id",
            "sentence_id",
            "reference_id",
            "reference_speaker_id",
            "text",
            "audio_path",
        ),
    )
    _write_csv(
        paths["plan"],
        experiment_rows,
        ("experiment_id", "model", "nfe", "ablation", "run_dir", "status"),
    )
    paths["dataset"].write_text(
        json.dumps(dataset_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["report"].write_text(
        _report_text(runs, dataset_summary, controlled_rows),
        encoding="utf-8",
    )
    return paths


# ====================
# 6. Public API
# ====================


def analyze_baselines(runs_root: Path, data_root: Path, output_dir: Path) -> dict[str, Path]:
    """生成 B1/B2/B3 审计报告、固定测试清单和后续实验记录。"""
    return _execute(runs_root, data_root, output_dir)
