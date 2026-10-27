"""模块说明：运行两个B2完整端点覆盖训练臂及冻结验证评估，禁止联网。"""

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
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


_LOCAL_CONNECT = socket.socket.connect
_LOCAL_CONNECT_EX = socket.socket.connect_ex


def _deny_network(self, address):
    """保留DataLoader本机UNIX进程通信，阻止IP连接与隐式下载。"""
    if self.family == socket.AF_UNIX:
        return _LOCAL_CONNECT(self, address)
    raise RuntimeError("端点pilot禁止联网和自动下载")


def _deny_network_ex(self, address):
    if self.family == socket.AF_UNIX:
        return _LOCAL_CONNECT_EX(self, address)
    raise RuntimeError("端点pilot禁止联网和自动下载")


socket.socket.connect = _deny_network
socket.socket.connect_ex = _deny_network_ex
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ====================
# 6. Public API
# ====================


def main():
    """解析实验阶段与训练臂，调度受控 pilot 或冻结评估。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=["prepare", "preflight", "train", "evaluate", "report"], required=True
    )
    parser.add_argument("--arm", choices=["control", "intervention", "both"], default="both")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(2)
    output = args.output_dir.resolve()
    if args.stage in ["prepare", "preflight", "train"]:
        from src.analysis.endpoint_pilot.training import run

        run(output, args.stage, torch.device(args.device), args.arm, args.workers)
    else:
        from src.analysis.endpoint_pilot.evaluation import run

        run(output, args.stage, torch.device(args.device))


# ====================
# 7. Entry Point
# ====================

if __name__ == "__main__":
    main()
