"""Vocoder 波形域评估入口脚本。

将统一评估生成的 Mel 经 Vocos 声码器合成为波形， 计算 WER/CER、说话人相似度、F0、能量与 RTF 等指标。
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.vocoder_evaluation import evaluate_vocoder_metrics  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unified-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis_results" / "unified_evaluation",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "analysis_results" / "baseline_validation" / "test_manifest.csv",
    )
    parser.add_argument(
        "--vctk-root", type=Path, default=Path("/mnt/data2/home/tsx/project/data/VCTK")
    )
    parser.add_argument(
        "--vocos-config", type=Path, default=PROJECT_ROOT / "pretrained" / "vocos" / "config.yaml"
    )
    parser.add_argument(
        "--vocos-weights",
        type=Path,
        default=PROJECT_ROOT / "pretrained" / "vocos" / "pytorch_model.bin",
    )
    parser.add_argument(
        "--asr-model-dir", type=Path, default=PROJECT_ROOT / "pretrained" / "whisper-small"
    )
    parser.add_argument(
        "--speaker-source",
        type=Path,
        default=PROJECT_ROOT / "pretrained" / "meanvc" / "ecapa_tdnn.py",
    )
    parser.add_argument(
        "--speaker-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "pretrained" / "meanvc" / "wavlm_large_finetune.pth",
    )
    parser.add_argument(
        "--wavlm-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "pretrained" / "meanvc" / "wavlm_large.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis_results" / "vocoder_evaluation",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-control", action="store_true")
    return parser.parse_args()


def main() -> None:
    """执行波形域评估。"""
    args = _parse_args()
    summary = evaluate_vocoder_metrics(
        unified_dir=args.unified_dir.resolve(),
        manifest_path=args.manifest.resolve(),
        vctk_root=args.vctk_root.resolve(),
        vocos_config=args.vocos_config.resolve(),
        vocos_weights=args.vocos_weights.resolve(),
        asr_model_dir=args.asr_model_dir.resolve(),
        speaker_source=args.speaker_source.resolve(),
        speaker_checkpoint=args.speaker_checkpoint.resolve(),
        wavlm_checkpoint=args.wavlm_checkpoint.resolve(),
        output_dir=args.output_dir.resolve(),
        device=args.device,
        limit=args.limit,
        run_control=not args.skip_control,
    )
    print(f"评估完成，报告: {summary['report']}")


if __name__ == "__main__":
    main()
