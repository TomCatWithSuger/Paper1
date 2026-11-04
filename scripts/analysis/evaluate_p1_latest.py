"""模块说明：执行最新三组权重的离线 P1 预检、冒烟与完整评估。"""

# ====================
# 1. Imports
# ====================

import argparse
import os
import socket
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _deny_network(*args, **kwargs):
    """评估进程禁止网络连接，防止任何依赖隐式下载。"""
    raise RuntimeError("P1离线评估禁止网络连接和自动下载")


socket.socket.connect = _deny_network
socket.socket.connect_ex = _deny_network
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.p1.evaluation import run

# ====================
# 6. Public API
# ====================


def main():
    """解析阶段与独立输出目录。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=["preflight", "smoke", "full"], required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.output_dir, args.stage, args.device)


# ====================
# 7. Entry Point
# ====================

if __name__ == "__main__":
    main()
