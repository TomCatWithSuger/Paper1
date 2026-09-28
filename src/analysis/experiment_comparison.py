"""对比多个 Lightning 实验的最优和最终指标。"""

import csv
from pathlib import Path

from .metrics_io import load_metric, summarize_metric
from .svg_utils import write_bar_chart


def compare(
    runs: list[tuple[str, Path]],
    metric: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    """汇总多个实验并绘制最优指标柱状图。"""
    if len(runs) < 2:
        raise ValueError("至少需要两个实验进行对比")

    records: list[dict[str, str | int | float]] = []
    for name, path in runs:
        metrics_path, points = load_metric(path, metric)
        summary = summarize_metric(points, minimize="loss" in metric.lower())
        records.append(
            {
                "experiment": name,
                "source": str(metrics_path),
                "metric": metric,
                "count": summary["count"],
                "minimum": summary["minimum"],
                "maximum": summary["maximum"],
                "final": summary["final"],
                "best": summary["best"],
                "best_position": summary["best_position"],
            }
        )

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "experiment_comparison.csv"
    chart_path = output_dir / "experiment_comparison.svg"
    with table_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    write_bar_chart(
        labels=[str(record["experiment"]) for record in records],
        values=[float(record["best"]) for record in records],
        output=chart_path,
        title=f"实验最优指标对比：{metric}",
        y_label=metric,
    )
    return table_path, chart_path
