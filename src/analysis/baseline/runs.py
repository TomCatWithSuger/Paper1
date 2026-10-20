"""读取实验配置、检查点与训练指标。"""


# ====================
# 1. Imports
# ====================


import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ====================
# 2. Definitions
# 常量 / 配置 / 类型
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
# 4. Internal Functions
# 内部业务实现
# ====================


def _nested_value(mapping: dict[str, Any], dotted_path: str) -> Any:
    value: Any = mapping
    for key in dotted_path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


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
