"""模块说明：执行冻结模型的独立验证集 P2 机制诊断，禁止联网。"""

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
    """防止依赖隐式下载。"""
    raise RuntimeError("P2诊断禁止网络连接和自动下载")


socket.socket.connect = _deny_network
socket.socket.connect_ex = _deny_network
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ====================
# 6. Public API
# ====================


def main():
    """解析诊断阶段，固定数值精度并调度只读 P2 诊断。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=[
            "prepare",
            "d1",
            "d1_center",
            "d1_boundary",
            "precision",
            "endpoint_fd",
            "d2",
            "conversion",
            "score",
            "controls",
            "report",
        ],
        required=True,
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = args.output_dir.resolve()
    if args.stage != "prepare":
        from src.analysis.p1.evaluation import _json
        from src.analysis.p2.common import snapshot

        directory = output / "runtime" / f"{args.stage}_{'smoke' if args.smoke else 'full'}"
        directory.mkdir(parents=True, exist_ok=True)
        sources = snapshot(directory)
        _json(
            directory / "runtime.json",
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": args.device,
                "tf32": False,
                "stage": args.stage,
                "smoke": args.smoke,
                "source_sha256": sources,
            },
        )
    if args.stage == "prepare":
        from src.analysis.p2.common import run

        run(output)
    elif args.stage in ("d1", "d1_center", "d1_boundary", "d2"):
        from src.analysis.p2.intervals import run

        run(output, args.stage, torch.device(args.device), 4 if args.smoke else None)
    elif args.stage == "precision":
        from src.analysis.p2.precision import run

        run(output, torch.device(args.device))
    elif args.stage == "endpoint_fd":
        from src.analysis.p2.precision import run_endpoint

        run_endpoint(output, torch.device(args.device))
    else:
        from src.analysis.p2.conversion import run

        run(output, args.stage, torch.device(args.device), args.smoke)


# ====================
# 7. Entry Point
# ====================

if __name__ == "__main__":
    main()
