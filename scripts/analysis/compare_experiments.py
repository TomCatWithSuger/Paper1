"""对比多个 Lightning 实验的最优和最终指标。"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.experiment_comparison import compare as compare  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="对比 B1、B2 或其他 Lightning 实验")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="名称=路径",
        help="实验名称和 metrics.csv 或运行目录，可重复传入",
    )
    parser.add_argument("--metric", default="val/loss", help="用于比较的指标")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_results"))
    return parser


def _parse_run(value: str) -> tuple[str, Path]:
    """解析 ``名称=路径`` 格式的实验参数。"""
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise ValueError(f"实验参数必须采用 名称=路径 格式: {value}")
    return name.strip(), Path(path.strip())


def main() -> None:
    """执行实验指标对比。"""
    args = _parser().parse_args()
    table_path, chart_path = compare(
        runs=[_parse_run(value) for value in args.run],
        metric=args.metric,
        output_dir=args.output_dir,
    )
    print(f"对比表: {table_path}")
    print(f"柱状图: {chart_path}")


if __name__ == "__main__":
    main()
