"""生成 B1、B2、B3 基线验证审计与实验记录。"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.baseline_validation import analyze_baselines  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="审计最近的 B1/B2/B3 训练记录")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("logs/train/runs"),
        help="Lightning 训练运行目录",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="VCTK 根目录或包含 VCTK 子目录的数据集根目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_results/baseline_validation"),
        help="报告与实验记录输出目录",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    outputs = analyze_baselines(args.runs_root, args.data_root, args.output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
