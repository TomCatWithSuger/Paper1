"""无第三方绘图库依赖的 SVG 输出工具。

- 绘制多序列折线图。
- 绘制实验指标柱状图。
- 绘制二维 Mel 频谱热力图。
"""

# ====================
# 1. 导入
# ====================

from html import escape
from pathlib import Path
from typing import Sequence

# ====================
# 2. 定义
# ====================

COLORS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2")


# ====================
# 3. 工具函数
# ====================


def _scale(
    value: float, lower: float, upper: float, target_lower: float, target_upper: float
) -> float:
    """将数值线性映射到目标区间。"""
    if upper == lower:
        return (target_lower + target_upper) / 2.0
    ratio = (value - lower) / (upper - lower)
    return target_lower + ratio * (target_upper - target_lower)


def _document(width: int, height: int, body: str) -> str:
    """构造完整 SVG 文档。"""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        f"{body}\n</svg>\n"
    )


def _write(path: Path, content: str) -> None:
    """写入 UTF-8 SVG 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ====================
# 4. 核心流程
# ====================


def write_line_chart(
    series: dict[str, list[tuple[float, float]]],
    output: Path,
    title: str,
    x_label: str = "epoch / step",
    y_label: str = "value",
) -> None:
    """将多项指标绘制为 SVG 折线图。"""
    nonempty = {name: points for name, points in series.items() if points}
    if not nonempty:
        raise ValueError("至少需要一个非空指标序列")

    width, height = 1000, 620
    left, right, top, bottom = 90, 250, 70, 80
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_points = [point for points in nonempty.values() for point in points]
    x_values = [point[0] for point in all_points]
    y_values = [point[1] for point in all_points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    y_padding = (y_max - y_min) * 0.05 or 0.5
    y_min -= y_padding
    y_max += y_padding

    elements = [
        f'<text x="{width / 2}" y="35" text-anchor="middle" font-size="22" '
        f'font-family="sans-serif">{escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" '
        f'y2="{top + plot_height}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" ' 'stroke="#111827"/>',
    ]

    for index in range(6):
        ratio = index / 5
        x = left + ratio * plot_width
        x_value = x_min + ratio * (x_max - x_min)
        y = top + plot_height - ratio * plot_height
        y_value = y_min + ratio * (y_max - y_min)
        elements.extend(
            [
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" '
                'stroke="#e5e7eb"/>',
                f'<text x="{x:.2f}" y="{top + plot_height + 24}" text-anchor="middle" '
                f'font-size="12" font-family="sans-serif">{x_value:.3g}</text>',
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" '
                'stroke="#e5e7eb"/>',
                f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="12" '
                f'font-family="sans-serif">{y_value:.4g}</text>',
            ]
        )

    for index, (name, points) in enumerate(nonempty.items()):
        color = COLORS[index % len(COLORS)]
        coordinates = " ".join(
            f"{_scale(x, x_min, x_max, left, left + plot_width):.2f},"
            f"{_scale(y, y_min, y_max, top + plot_height, top):.2f}"
            for x, y in points
        )
        elements.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
            'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        legend_y = top + index * 28
        elements.extend(
            [
                f'<line x1="{left + plot_width + 25}" y1="{legend_y}" '
                f'x2="{left + plot_width + 55}" y2="{legend_y}" stroke="{color}" '
                'stroke-width="3"/>',
                f'<text x="{left + plot_width + 65}" y="{legend_y + 4}" font-size="13" '
                f'font-family="sans-serif">{escape(name)}</text>',
            ]
        )

    elements.extend(
        [
            f'<text x="{left + plot_width / 2}" y="{height - 25}" text-anchor="middle" '
            f'font-size="14" font-family="sans-serif">{escape(x_label)}</text>',
            f'<text x="22" y="{top + plot_height / 2}" text-anchor="middle" font-size="14" '
            f'font-family="sans-serif" transform="rotate(-90 22 {top + plot_height / 2})">'
            f"{escape(y_label)}</text>",
        ]
    )
    _write(output, _document(width, height, "\n".join(elements)))


def write_bar_chart(
    labels: Sequence[str],
    values: Sequence[float],
    output: Path,
    title: str,
    y_label: str,
) -> None:
    """将多个实验的单项指标绘制为 SVG 柱状图。"""
    if not labels or len(labels) != len(values):
        raise ValueError("标签和值必须具有相同的非零长度")

    width, height = 900, 560
    left, right, top, bottom = 90, 40, 70, 120
    plot_width = width - left - right
    plot_height = height - top - bottom
    lower = min(0.0, min(values))
    upper = max(values)
    padding = (upper - lower) * 0.1 or 1.0
    upper += padding
    baseline = _scale(0.0, lower, upper, top + plot_height, top)
    slot = plot_width / len(values)
    bar_width = slot * 0.62

    elements = [
        f'<text x="{width / 2}" y="35" text-anchor="middle" font-size="22" '
        f'font-family="sans-serif">{escape(title)}</text>',
        f'<line x1="{left}" y1="{baseline:.2f}" x2="{left + plot_width}" '
        f'y2="{baseline:.2f}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" ' 'stroke="#111827"/>',
    ]

    for index in range(6):
        ratio = index / 5
        y = top + plot_height - ratio * plot_height
        value = lower + ratio * (upper - lower)
        elements.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" '
                'stroke="#e5e7eb"/>',
                f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="12" '
                f'font-family="sans-serif">{value:.4g}</text>',
            ]
        )

    for index, (label, value) in enumerate(zip(labels, values)):
        x = left + index * slot + (slot - bar_width) / 2
        value_y = _scale(value, lower, upper, top + plot_height, top)
        y = min(value_y, baseline)
        bar_height = abs(baseline - value_y)
        color = COLORS[index % len(COLORS)]
        elements.extend(
            [
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
                f'height="{bar_height:.2f}" fill="{color}" rx="3"/>',
                f'<text x="{x + bar_width / 2:.2f}" y="{y - 8:.2f}" text-anchor="middle" '
                f'font-size="12" font-family="sans-serif">{value:.5g}</text>',
                f'<text x="{x + bar_width / 2:.2f}" y="{top + plot_height + 25}" '
                f'text-anchor="end" font-size="13" font-family="sans-serif" '
                f'transform="rotate(-35 {x + bar_width / 2:.2f} {top + plot_height + 25})">'
                f"{escape(label)}</text>",
            ]
        )

    elements.append(
        f'<text x="22" y="{top + plot_height / 2}" text-anchor="middle" font-size="14" '
        f'font-family="sans-serif" transform="rotate(-90 22 {top + plot_height / 2})">'
        f"{escape(y_label)}</text>"
    )
    _write(output, _document(width, height, "\n".join(elements)))


def write_heatmap(
    matrix: Sequence[Sequence[float]],
    output: Path,
    title: str,
) -> None:
    """将二维矩阵绘制为 SVG 热力图。"""
    if not matrix or not matrix[0]:
        raise ValueError("热力图矩阵不能为空")

    rows = len(matrix)
    columns = len(matrix[0])
    values = [value for row in matrix for value in row]
    lower, upper = min(values), max(values)
    width, height = 1100, 560
    left, right, top, bottom = 80, 100, 70, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    cell_width = plot_width / columns
    cell_height = plot_height / rows

    def color(value: float) -> str:
        ratio = 0.5 if upper == lower else (value - lower) / (upper - lower)
        red = int(255 * ratio)
        green = int(190 * min(ratio * 2, (1 - ratio) * 2))
        blue = int(255 * (1 - ratio))
        return f"#{red:02x}{green:02x}{blue:02x}"

    elements = [
        f'<text x="{width / 2}" y="35" text-anchor="middle" font-size="22" '
        f'font-family="sans-serif">{escape(title)}</text>'
    ]
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            elements.append(
                f'<rect x="{left + column_index * cell_width:.2f}" '
                f'y="{top + (rows - row_index - 1) * cell_height:.2f}" '
                f'width="{cell_width + 0.2:.2f}" height="{cell_height + 0.2:.2f}" '
                f'fill="{color(value)}"/>'
            )

    elements.extend(
        [
            f'<text x="{left + plot_width / 2}" y="{height - 22}" text-anchor="middle" '
            'font-size="14" font-family="sans-serif">帧</text>',
            f'<text x="22" y="{top + plot_height / 2}" text-anchor="middle" font-size="14" '
            f'font-family="sans-serif" transform="rotate(-90 22 {top + plot_height / 2})">'
            "Mel 通道</text>",
            f'<text x="{left + plot_width + 18}" y="{top + 12}" font-size="12" '
            f'font-family="sans-serif">最大值 {upper:.4g}</text>',
            f'<text x="{left + plot_width + 18}" y="{top + plot_height}" font-size="12" '
            f'font-family="sans-serif">最小值 {lower:.4g}</text>',
        ]
    )
    _write(output, _document(width, height, "\n".join(elements)))
