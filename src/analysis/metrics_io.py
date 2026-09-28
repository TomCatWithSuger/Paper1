"""训练指标读取工具。

- 定位 Lightning CSV logger 生成的 ``metrics.csv``。
- 提取按 epoch 或 step 排序的指标序列。
- 计算最优值、最终值和对应训练位置。
"""

# ====================
# 1. 导入
# ====================

import csv
import math
from pathlib import Path
from typing import TypedDict

# ====================
# 2. 定义
# ====================


class MetricSummary(TypedDict):
    """单项训练指标的统计结果。"""

    count: int
    minimum: float
    maximum: float
    final: float
    best: float
    best_position: float


# ====================
# 3. 工具函数
# ====================


def _parse_float(value: str | None) -> float | None:
    """将有效有限数字转换为浮点数。"""
    if value is None or not value.strip():
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _position(row: dict[str, str], fallback: int) -> float:
    """优先使用 epoch，其次使用 step 作为横坐标。"""
    for key in ("epoch", "step"):
        value = _parse_float(row.get(key))
        if value is not None:
            return value
    return float(fallback)


# ====================
# 4. 核心流程
# ====================


def resolve_metrics_csv(path: Path) -> Path:
    """从文件或运行目录中定位唯一的 ``metrics.csv``。"""
    path = path.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(f"指标文件必须是 CSV: {path}")
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"路径不存在: {path}")

    matches = sorted(path.rglob("metrics.csv"))
    if not matches:
        raise FileNotFoundError(f"未在 {path} 下找到 metrics.csv，请在训练时添加 logger=csv")
    if len(matches) > 1:
        listed = "\n".join(f"- {item}" for item in matches)
        raise ValueError(f"找到多个 metrics.csv，请指定具体文件：\n{listed}")
    return matches[0]


def read_metrics(path: Path) -> list[dict[str, str]]:
    """读取 Lightning CSV 指标记录。"""
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def available_metrics(rows: list[dict[str, str]]) -> list[str]:
    """返回至少包含一个数值的指标列。"""
    if not rows:
        return []
    ignored = {"epoch", "step"}
    return [
        key
        for key in rows[0]
        if key not in ignored and any(_parse_float(row.get(key)) is not None for row in rows)
    ]


def metric_points(rows: list[dict[str, str]], metric: str) -> list[tuple[float, float]]:
    """提取指标序列，并保留同一训练位置最后记录的值。"""
    points: dict[float, float] = {}
    for index, row in enumerate(rows):
        value = _parse_float(row.get(metric))
        if value is not None:
            points[_position(row, index)] = value
    return sorted(points.items())


def summarize_metric(
    points: list[tuple[float, float]],
    minimize: bool = True,
) -> MetricSummary:
    """统计指标的范围、最终值和最优位置。"""
    if not points:
        raise ValueError("指标序列不能为空")
    best_position, best_value = min(
        points,
        key=lambda item: item[1] if minimize else -item[1],
    )
    values = [value for _, value in points]
    return {
        "count": len(points),
        "minimum": min(values),
        "maximum": max(values),
        "final": values[-1],
        "best": best_value,
        "best_position": best_position,
    }


# ====================
# 5. 对外接口
# ====================


def load_metric(path: Path, metric: str) -> tuple[Path, list[tuple[float, float]]]:
    """定位 CSV 并读取指定指标。"""
    metrics_path = resolve_metrics_csv(path)
    points = metric_points(read_metrics(metrics_path), metric)
    if not points:
        raise ValueError(f"{metrics_path} 中不存在有效指标 {metric!r}")
    return metrics_path, points
