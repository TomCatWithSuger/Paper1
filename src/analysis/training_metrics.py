"""分析单次 Lightning 训练的指标并输出曲线和统计摘要。"""

import json
from pathlib import Path
from typing import Sequence

from .metrics_io import (
    available_metrics,
    metric_points,
    read_metrics,
    resolve_metrics_csv,
    summarize_metric,
)
from .svg_utils import write_line_chart

DEFAULT_METRICS = ("train/loss", "val/loss", "test/loss")


def analyze(
    input_path: Path,
    metrics: Sequence[str],
    output_dir: Path | None,
    title: str,
) -> tuple[Path, Path]:
    """分析训练指标并输出 SVG 和 JSON。"""
    metrics_path = resolve_metrics_csv(input_path)
    rows = read_metrics(metrics_path)
    selected = {metric: points for metric in metrics if (points := metric_points(rows, metric))}
    if not selected:
        names = ", ".join(available_metrics(rows)) or "无"
        raise ValueError(f"未找到指定指标，可用指标：{names}")

    resolved_output = (output_dir or metrics_path.parent / "analysis").resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    chart_path = resolved_output / "training_curves.svg"
    summary_path = resolved_output / "metrics_summary.json"

    write_line_chart(
        series=selected,
        output=chart_path,
        title=title,
        y_label="指标值",
    )
    summary = {
        "source": str(metrics_path),
        "metrics": {
            metric: summarize_metric(points, minimize="loss" in metric.lower())
            for metric, points in selected.items()
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return chart_path, summary_path
