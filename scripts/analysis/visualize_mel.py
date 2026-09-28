"""读取 Mel 张量并输出热力图和数值统计。"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.mel import _load_array as _load_array  # noqa: E402
from src.analysis.mel import _prepare_mel as _prepare_mel
from src.analysis.mel import _select_mapping_value as _select_mapping_value
from src.analysis.mel import visualize as visualize


def _parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="可视化 .pt、.npy 或 .npz Mel 张量")
    parser.add_argument("input", type=Path, help="Mel 张量文件")
    parser.add_argument("--key", help="字典或 npz 中的张量键")
    parser.add_argument("--batch-index", type=int, default=0, help="三维张量的样本索引")
    parser.add_argument("--transpose", action="store_true", help="交换 Mel 和时间维")
    parser.add_argument("--max-frames", type=int, default=300, help="SVG 最大显示帧数")
    parser.add_argument("--output-dir", type=Path, help="输出目录")
    parser.add_argument("--title", help="热力图标题")
    return parser


def main() -> None:
    """执行 Mel 可视化。"""
    args = _parser().parse_args()
    chart_path, summary_path = visualize(
        input_path=args.input,
        key=args.key,
        batch_index=args.batch_index,
        transpose=args.transpose,
        max_frames=args.max_frames,
        output_dir=args.output_dir,
        title=args.title,
    )
    print(f"热力图: {chart_path}")
    print(f"统计摘要: {summary_path}")


if __name__ == "__main__":
    main()
