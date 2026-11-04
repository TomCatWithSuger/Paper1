"""模块说明：只读已有转换音频，执行离线 P4 身份诊断与独立评分。"""

# ====================
# 1. Imports
# ====================

import argparse
import os
import socket
import sys
from pathlib import Path

# ====================
# 2. Definitions
# 常量 / 配置 / 类型
# ====================

ROOT = Path(__file__).resolve().parents[2]


# ====================
# 3. Utilities
# 通用工具函数
# ====================


def _deny_network(*args, **kwargs):
    """阻止模型库隐式下载。"""
    raise RuntimeError("P4 禁止网络连接和自动下载")


# ====================
# 6. Public API
# 对外提供的函数
# ====================


def main():
    """解析本地资源路径，调度身份分层、独立评分或敏感性分析。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["analyze", "score", "sensitivity"], required=True)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=ROOT
        / "analysis_results/p2_mechanisms_20261024/conversion/per_sample_enriched.csv",
    )
    parser.add_argument("--dataset", type=Path, default=ROOT.parent / "data/VCTK")
    parser.add_argument("--speaker-info", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--campplus-model", type=Path)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    socket.socket.connect = _deny_network
    socket.socket.connect_ex = _deny_network
    sys.path.insert(0, str(ROOT))
    if args.stage == "analyze":
        from src.analysis.p4.identity import run

        run(
            args.input_csv,
            args.speaker_info or args.dataset / "speaker-info.txt",
            args.dataset,
            args.output_dir,
        )
    elif args.stage == "score":
        if args.campplus_model is None:
            parser.error("score 阶段需要 --campplus-model 指向本地权重")
        from src.analysis.p4.scoring import run

        run(args.output_dir, args.campplus_model)
    else:
        from src.analysis.p4.sensitivity import run

        run(args.output_dir)
    print(args.output_dir.resolve())


# ====================
# 7. Entry Point
# 可选：直接运行该文件
# ====================

if __name__ == "__main__":
    main()
