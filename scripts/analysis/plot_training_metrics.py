"""绘制单次 Lightning 训练的损失曲线并生成统计摘要。"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.training_metrics import (  # noqa: E402
    DEFAULT_METRICS as DEFAULT_METRICS,
    analyze as analyze,
)


def _parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="绘制 Lightning CSV 训练指标")
    parser.add_argument("input", type=Path, help="metrics.csv 或包含该文件的运行目录")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="需要绘制的指标列",
    )
    parser.add_argument("--output-dir", type=Path, help="输出目录，默认位于指标文件旁")
    parser.add_argument("--title", default="训练指标曲线", help="图表标题")
    return parser


def main() -> None:
    """执行训练指标分析。"""
    args = _parser().parse_args()
    chart_path, summary_path = analyze(
        input_path=args.input,
        metrics=args.metrics,
        output_dir=args.output_dir,
        title=args.title,
    )
    print(f"曲线图: {chart_path}")
    print(f"统计摘要: {summary_path}")


if __name__ == "__main__":
    main()
